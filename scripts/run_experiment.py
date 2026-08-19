"""Run one leakage-safe, budgeted target-adaptation experiment seed."""
from __future__ import annotations
import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from value_sced.acquisition import (PoolState, select_decision_sensitive, select_diversity,
    select_random, select_uncertainty, source_standardisation)
from value_sced.experiment import OracleLabelStore
from value_sced.grid import build_target_grid
from value_sced.metrics import evaluate_predictions
from value_sced.surrogate import source_train_validation_indices, train_ensemble

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def arrays(records, labelled=True):
    x = np.asarray([r["gross_demand_mw"] + r["renewable_mw"] for r in records], float)
    d = np.asarray([r["net_demand_mw"] for r in records], float)
    if not labelled: return x, d
    y = np.asarray([r["oracle"]["generation_mw"] for r in records], float)
    c = np.asarray([r["oracle"]["objective_cost"] for r in records], float)
    return x, d, y, c

def fit(cfg, seed, sx, sy, train_idx, val_idx, tx, ty):
    spec = cfg["surrogate"]
    return train_ensemble(np.concatenate((sx[train_idx], tx)), np.concatenate((sy[train_idx], ty)),
        sx[val_idx], sy[val_idx], ensemble_size=int(spec["ensemble_size"]),
        hidden_sizes=tuple(spec["hidden_sizes"]), learning_rate=float(spec["learning_rate"]),
        max_epochs=int(spec["max_epochs"]), patience=int(spec["early_stopping_patience"]),
        seed=seed, scaling_features=sx[train_idx], scaling_targets=sy[train_idx])

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int); args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(cfg["evaluation"]["debugging_seeds"][0])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    source = load(ROOT / "results/raw/source.jsonl")
    feature_records = load(ROOT / "results/raw/target_pool_features.jsonl")
    label_records = load(ROOT / "results/raw/target_pool_hidden_labels.jsonl")
    test = load(ROOT / "results/raw/target_test.jsonl")
    sx, _, sy, _ = arrays(source); px, pd = arrays(feature_records, False)
    py = np.asarray([r["oracle"]["generation_mw"] for r in label_records], float)
    tx, td, ty, tc = arrays(test)
    pool_ids = [r["scenario_id"] for r in feature_records]
    if pool_ids != [r["scenario_id"] for r in label_records]: raise RuntimeError("pool feature/label identity mismatch")
    if set(pool_ids) & {r["scenario_id"] for r in test}: raise RuntimeError("pool/test overlap")
    frozen = json.loads((ROOT / "results/raw/initial_target_selections.json").read_text())
    initial = tuple(pool_ids.index(i) for i in frozen["selections"][str(seed)])
    train_idx, val_idx = source_train_validation_indices(len(source), float(cfg["surrogate"]["source_validation_fraction"]), seed)
    grid = build_target_grid(); contingencies = tuple(cfg["grid"]["contingency_lines"]); oracle = cfg["oracle"]
    out = ROOT / "results/processed" / f"seed_{seed}"; out.mkdir(parents=True, exist_ok=True)
    rows, sample_rows, histories = [], [], {}; failures = []
    source_mean, source_scale = source_standardisation(sx[train_idx])
    for method in cfg["acquisition"]["methods"]:
        state = PoolState.initialise(len(px), initial); store = OracleLabelStore({i: py[i] for i in range(len(py))})
        revealed = store.reveal(initial); rng = np.random.default_rng(seed); histories[method] = [list(initial)]
        for budget in cfg["evaluation"]["budgets"][1:]:
            try:
                labelled = tuple(state.labelled); labelled_y = np.asarray([revealed[i] for i in labelled])
                ensemble = fit(cfg, seed, sx, sy, train_idx, val_idx, px[list(labelled)], labelled_y)
                metrics, samples = evaluate_predictions(grid, td, ensemble.predict(tx), ty, tc,
                    contingency_lines=contingencies, balance_tolerance_mw=float(oracle["balance_tolerance_mw"]),
                    limit_tolerance_mw=float(oracle["limit_tolerance_mw"]))
                rows.append({"seed": seed, "method": method, "budget": budget, **metrics})
                for sample, record in zip(samples, test, strict=True):
                    sample_rows.append({"seed": seed, "method": method, "budget": budget,
                        "scenario_id": record["scenario_id"], **sample})
                if budget == cfg["evaluation"]["budgets"][-1]: continue
                remaining = tuple(state.remaining); size = int(cfg["data"]["acquisition_batch_size"])
                if method == "random": batch = select_random(remaining, size, rng)
                elif method == "diversity": batch = select_diversity(px, remaining, state.labelled,
                    sx[train_idx], size, source_mean, source_scale, rng)
                else:
                    members = ensemble.predict_members(px[list(remaining)])
                    if method == "uncertainty": batch = select_uncertainty(remaining, members, size, rng)
                    else: batch = select_decision_sensitive(remaining, members, pd[list(remaining)], grid,
                        contingencies, size, rng, float(cfg["acquisition"]["decision_lambda"]))
                state.acquire(batch); revealed.update(store.reveal(batch)); histories[method].append(list(batch))
            except Exception as error:
                failures.append(f"seed={seed} method={method} budget={budget}: {error}"); raise
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in sample_rows: stream.write(json.dumps(row) + "\n")
    audit = {"seed": seed, "methods": list(cfg["acquisition"]["methods"]),
        "budgets": list(cfg["evaluation"]["budgets"])[1:], "source_train_count": len(train_idx),
        "source_validation_count": len(val_idx), "pool_count": len(pool_ids), "test_count": len(test),
        "pool_test_disjoint": True, "identical_initial_set_across_methods": True,
        "retraining_rule": "from scratch at each checkpoint; source-only validation and scaling",
        "selection_indices_by_round": histories, "completed_rows": len(rows)}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    text = "# Failure log\n\n" + ("\n".join(f"- {x}" for x in failures) if failures else "No failures.\n")
    (out / "failure_log.md").write_text(text, encoding="utf-8")
    print(f"Completed seed {seed}: {len(rows)} method-budget results in {out}")

if __name__ == "__main__":
    try: main()
    except Exception as error: print(f"Experiment failed: {error}", file=sys.stderr); raise
