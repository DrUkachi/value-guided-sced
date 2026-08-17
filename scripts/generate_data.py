"""Generate and freeze the source, target-pool, and target-test datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml

from value_sced.data import sample_operating_condition
from value_sced.grid import Grid, build_source_grid, build_target_grid
from value_sced.oracle import solve_economic_dispatch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "base.yaml"
RAW = ROOT / "results" / "raw"
FIGURES = ROOT / "results" / "figures"


def write_jsonl(path: Path, records: list[dict[str, object]]) -> str:
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def maximum_loading(oracle: dict[str, object], grid: Grid) -> float:
    ratios = [
        abs(flow) / line.limit_mw
        for flow, line in zip(oracle["line_flows_mw"], grid.lines, strict=True)
    ]
    for numbers, flows in zip(
        oracle["contingency_line_numbers"],
        oracle["contingency_line_flows_mw"],
        strict=True,
    ):
        limits = {line.number: line.limit_mw for line in grid.lines}
        ratios.extend(abs(flow) / limits[number] for number, flow in zip(numbers, flows, strict=True))
    return max(ratios)


def generate_split(
    *,
    count: int,
    split: str,
    topology: str,
    grid: Grid,
    rng: np.random.Generator,
    data_config: dict[str, object],
    contingencies: tuple[int, ...],
) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    attempts = 0
    maximum_attempts = int(data_config["maximum_sampling_attempts"])
    renewable_indices = tuple(int(bus) - 1 for bus in data_config["renewable_buses"])
    while len(records) < count and attempts < maximum_attempts:
        attempts += 1
        condition = sample_operating_condition(
            rng,
            tuple(float(x) for x in data_config["total_gross_demand_mw"]),
            np.asarray(data_config["demand_profile"], dtype=float),
            float(data_config["demand_concentration"]),
            renewable_indices,
            tuple(float(x) for x in data_config["renewable_fraction_of_local_demand"]),
        )
        result = solve_economic_dispatch(grid, condition.net_demand_mw, contingencies)
        if not result.success:
            continue
        index = len(records)
        records.append(
            {
                "scenario_id": f"{split}-{index:04d}",
                "split": split,
                "seed": int(data_config["generation_seed"]),
                "topology": topology,
                **condition.to_dict(),
                "oracle": {
                    "solver": "scipy.optimize.linprog-highs",
                    "scipy_version": scipy.__version__,
                    "contingency_outages": list(contingencies),
                    **result.to_dict(),
                },
            }
        )
    if len(records) != count:
        raise RuntimeError(f"generated {len(records)} of {count} feasible {split} cases")
    return records, attempts - count


def plot_diagnostics(
    source: list[dict[str, object]],
    target: list[dict[str, object]],
    source_grid: Grid,
    target_grid: Grid,
    binding_tolerance: float,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for records, label, grid, color in (
        (source, "source", source_grid, "tab:blue"),
        (target, "target", target_grid, "tab:orange"),
    ):
        net_totals = [sum(record["net_demand_mw"]) for record in records]
        loads = [maximum_loading(record["oracle"], grid) for record in records]
        axes[0].hist(net_totals, bins=20, alpha=0.55, density=True, label=label, color=color)
        axes[1].hist(loads, bins=20, alpha=0.55, density=True, label=label, color=color)
        binding = np.mean(np.asarray(loads) >= 1.0 - binding_tolerance)
        axes[2].bar(label, binding, color=color)
    axes[0].set(title="Scenario distribution", xlabel="Total net demand (MW)", ylabel="Density")
    axes[1].set(title="Security loading", xlabel="Maximum |flow| / limit", ylabel="Density")
    axes[2].set(title="Binding-constraint frequency", ylabel="Fraction of scenarios", ylim=(0, 1))
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "source_target_diagnostics.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing frozen dataset (requires a documented protocol change)",
    )
    arguments = parser.parse_args()
    manifest_path = RAW / "dataset_manifest.json"
    if manifest_path.exists() and not arguments.force:
        raise SystemExit(
            f"Refusing to overwrite frozen data at {manifest_path}; use --force only after documenting the change."
        )
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_config = config["data"]
    contingencies = tuple(int(x) for x in config["grid"]["contingency_lines"])
    rng = np.random.default_rng(int(data_config["generation_seed"]))
    source_grid, target_grid = build_source_grid(), build_target_grid()
    RAW.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    source, source_rejected = generate_split(
        count=int(data_config["source_size"]), split="source", topology="source",
        grid=source_grid, rng=rng, data_config=data_config, contingencies=contingencies,
    )
    pool, pool_rejected = generate_split(
        count=int(data_config["target_candidate_size"]), split="target_pool", topology="target",
        grid=target_grid, rng=rng, data_config=data_config, contingencies=contingencies,
    )
    test, test_rejected = generate_split(
        count=int(data_config["target_test_size"]), split="target_test", topology="target",
        grid=target_grid, rng=rng, data_config=data_config, contingencies=contingencies,
    )

    hashes = {"source.jsonl": write_jsonl(RAW / "source.jsonl", source)}
    feature_keys = ("scenario_id", "split", "seed", "topology", "gross_demand_mw", "renewable_mw", "net_demand_mw")
    pool_features = [{key: record[key] for key in feature_keys} for record in pool]
    pool_labels = [{"scenario_id": record["scenario_id"], "oracle": record["oracle"]} for record in pool]
    hashes["target_pool_features.jsonl"] = write_jsonl(RAW / "target_pool_features.jsonl", pool_features)
    hashes["target_pool_hidden_labels.jsonl"] = write_jsonl(RAW / "target_pool_hidden_labels.jsonl", pool_labels)
    hashes["target_test.jsonl"] = write_jsonl(RAW / "target_test.jsonl", test)
    plot_diagnostics(source, pool + test, source_grid, target_grid, float(config["oracle"]["binding_tolerance_mw"]))
    manifest = {
        "frozen": True,
        "configuration": str(CONFIG_PATH.relative_to(ROOT)),
        "seed": int(data_config["generation_seed"]),
        "counts": {"source": len(source), "target_pool": len(pool), "target_test": len(test)},
        "rejected_infeasible_draws": {"source": source_rejected, "target_pool": pool_rejected, "target_test": test_rejected},
        "sha256": hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
