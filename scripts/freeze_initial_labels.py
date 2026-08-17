"""Freeze the common initial target cases for every configured experiment seed."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from value_sced.acquisition import initial_target_indices


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    data, evaluation = config["data"], config["evaluation"]
    seeds = evaluation["final_seeds"] + evaluation["debugging_seeds"]
    pool_records = [json.loads(line) for line in (
        ROOT / "results/raw/target_pool_features.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    if len(pool_records) != int(data["target_candidate_size"]):
        raise RuntimeError("candidate file does not match the frozen configuration")
    selections = {}
    for seed in seeds:
        indices = initial_target_indices(len(pool_records), int(data["initial_target_labels"]), int(seed))
        selections[str(seed)] = [pool_records[i]["scenario_id"] for i in indices]
    output = ROOT / "results/raw/initial_target_selections.json"
    output.write_text(json.dumps({"method": "numpy_default_rng_without_replacement", "selections": selections}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
