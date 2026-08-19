"""Inspect first-round top-ten selections without reading candidate oracle labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from value_sced.acquisition import (
    PoolState, decision_sensitive_scores, ensemble_disagreement,
    initial_target_indices, operational_criticality, select_diversity,
    select_highest_scores, select_random, source_standardisation,
)
from value_sced.grid import build_target_grid
from value_sced.surrogate import source_train_validation_indices, train_ensemble


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def features(records: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([r["gross_demand_mw"] + r["renewable_mw"] for r in records], dtype=float)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    seed = int(config["evaluation"]["debugging_seeds"][0])
    source, pool = load(ROOT / "results/raw/source.jsonl"), load(ROOT / "results/raw/target_pool_features.jsonl")
    source_x, pool_x = features(source), features(pool)
    source_y = np.asarray([r["oracle"]["generation_mw"] for r in source], dtype=float)
    train, validation = source_train_validation_indices(
        len(source), float(config["surrogate"]["source_validation_fraction"]), seed
    )
    model = config["surrogate"]
    torch.use_deterministic_algorithms(True)
    ensemble = train_ensemble(
        source_x[train], source_y[train], source_x[validation], source_y[validation],
        ensemble_size=int(model["ensemble_size"]), hidden_sizes=tuple(model["hidden_sizes"]),
        learning_rate=float(model["learning_rate"]), max_epochs=int(model["max_epochs"]),
        patience=int(model["early_stopping_patience"]), seed=seed,
    )
    initial = initial_target_indices(len(pool), int(config["data"]["initial_target_labels"]), seed)
    state = PoolState.initialise(len(pool), initial)
    remaining = np.asarray(state.remaining)
    member_predictions = ensemble.predict_members(pool_x[remaining])
    uncertainty = ensemble_disagreement(member_predictions)
    demands = np.asarray([pool[i]["net_demand_mw"] for i in remaining], dtype=float)
    criticality = operational_criticality(
        build_target_grid(), demands, member_predictions.mean(axis=0),
        config["grid"]["contingency_lines"],
    )
    decision = decision_sensitive_scores(
        uncertainty, criticality, float(config["acquisition"]["decision_lambda"])
    )
    mean, scale = source_standardisation(source_x[train])
    rngs = {name: np.random.default_rng(seed) for name in config["acquisition"]["methods"]}
    selections = {
        "random": select_random(state.remaining, 10, rngs["random"]),
        "diversity": select_diversity(
            pool_x, state.remaining, state.labelled, source_x[train], 10,
            mean, scale, rngs["diversity"],
        ),
        "uncertainty": select_highest_scores(state.remaining, uncertainty, 10, rngs["uncertainty"]),
        "decision_sensitive": select_highest_scores(
            state.remaining, decision, 10, rngs["decision_sensitive"]
        ),
    }
    position = {int(index): offset for offset, index in enumerate(remaining)}
    report = {
        "seed": seed,
        "candidate_oracle_labels_accessed": False,
        "score_ranges": {
            "uncertainty": [float(uncertainty.min()), float(uncertainty.max())],
            "criticality": [float(criticality.min()), float(criticality.max())],
            "decision_sensitive": [float(decision.min()), float(decision.max())],
        },
        "top_ten": {
            method: [{
                "scenario_id": pool[index]["scenario_id"],
                "net_demand_total_mw": float(sum(pool[index]["net_demand_mw"])),
                **({
                    "uncertainty": float(uncertainty[position[index]]),
                    "criticality": float(criticality[position[index]]),
                    "decision_sensitive_score": float(decision[position[index]]),
                } if method in {"uncertainty", "decision_sensitive"} else {}),
            } for index in selected]
            for method, selected in selections.items()
        },
    }
    output = ROOT / "results/processed/acquisition_top_ten.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["score_ranges"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
