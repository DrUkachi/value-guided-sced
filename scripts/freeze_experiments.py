"""Create Tuesday ablations, repair comparison, and freeze-ready diagnostics."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/processed"

def read_csv(path):
    with path.open(encoding="utf-8") as stream: return list(csv.DictReader(stream))

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def artifact_manifest(commit: str) -> dict:
    files = [ROOT / "configs/base.yaml"]
    for directory in (ROOT / "results/raw", ROOT / "results/processed", ROOT / "results/figures"):
        files.extend(path for path in directory.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    return {"frozen_artifacts_commit": commit, "hash_algorithm": "sha256", "files": {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default="PENDING_FIRST_FREEZE_COMMIT")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    seeds = config["evaluation"]["final_seeds"]
    main = read_csv(OUT / "main_results.csv")
    ablation = []
    for budget in config["evaluation"]["budgets"][1:]:
        for seed in seeds:
            cell = {(r["method"]): r for r in main if int(r["seed"]) == seed and int(r["budget"]) == budget}
            uncertainty, combined = cell["uncertainty"], cell["decision_sensitive"]
            ablation.append({"seed": seed, "budget": budget,
                "uncertainty_mae_mw": uncertainty["dispatch_mae_mw"],
                "combined_mae_mw": combined["dispatch_mae_mw"],
                "combined_minus_uncertainty_mae_mw": float(combined["dispatch_mae_mw"]) - float(uncertainty["dispatch_mae_mw"]),
                "uncertainty_mean_regret": uncertainty["mean_repaired_cost_regret"],
                "combined_mean_regret": combined["mean_repaired_cost_regret"],
                "combined_minus_uncertainty_mean_regret": float(combined["mean_repaired_cost_regret"]) - float(uncertainty["mean_repaired_cost_regret"])})
    write_csv(OUT / "operational_risk_ablation.csv", ablation)

    decision_rows = []
    prediction_cache = {}
    for seed in seeds:
        path = OUT / f"seed_{seed}" / "predictions.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        prediction_cache[seed] = records
        for method in config["acquisition"]["methods"]:
            for budget in config["evaluation"]["budgets"][1:]:
                group = [r for r in records if r["method"] == method and r["budget"] == budget]
                repaired_success = [np.all(np.isfinite(r["repaired_generation_mw"])) and np.isfinite(r["repaired_cost_regret"]) for r in group]
                decision_rows.append({"seed": seed, "method": method, "budget": budget,
                    "sample_count": len(group),
                    "raw_feasible_percentage": 100 * np.mean([r["raw_feasible"] for r in group]),
                    "repaired_success_percentage": 100 * np.mean(repaired_success),
                    "mean_raw_balance_error_mw": np.mean([r["balance_error_mw"] for r in group]),
                    "mean_raw_generator_violation_mw": np.mean([r["generator_limit_violation_mw"] for r in group]),
                    "mean_raw_contingency_overload_mw": np.mean([r["contingency_line_overload_mw"] for r in group]),
                    "mean_repair_magnitude_mw": np.mean([r["repair_magnitude_mw"] for r in group]),
                    "mean_repaired_cost_regret": np.mean([r["repaired_cost_regret"] for r in group])})
    write_csv(OUT / "raw_vs_repaired.csv", decision_rows)

    candidates = []
    for seed in seeds:
        records = prediction_cache[seed]
        uncertainty = {(r["budget"], r["scenario_id"]): r for r in records if r["method"] == "uncertainty"}
        for combined in (r for r in records if r["method"] == "decision_sensitive"):
            other = uncertainty[(combined["budget"], combined["scenario_id"])]
            candidates.append((combined["repaired_cost_regret"] - other["repaired_cost_regret"], seed, combined, other))
    delta, seed, combined, uncertainty = max(candidates, key=lambda item: item[0])
    report = f"""# Counterexample: operational weighting can hurt

The largest per-case disadvantage of the combined decision-sensitive score relative to uncertainty occurred for seed {seed}, budget {combined['budget']}, scenario `{combined['scenario_id']}`.

| Quantity | Uncertainty | Decision-sensitive |
|---|---:|---:|
| Repaired-cost regret | {uncertainty['repaired_cost_regret']:.3f} | {combined['repaired_cost_regret']:.3f} |
| Repair magnitude (MW) | {uncertainty['repair_magnitude_mw']:.3f} | {combined['repair_magnitude_mw']:.3f} |
| Raw balance error (MW) | {uncertainty['balance_error_mw']:.3f} | {combined['balance_error_mw']:.3f} |
| Contingency overload (MW) | {uncertainty['contingency_line_overload_mw']:.3f} | {combined['contingency_line_overload_mw']:.3f} |

The operationally weighted model incurred {delta:.3f} more regret on this case. This does not show that the criticality term always hurts; it shows that a criticality estimate derived from imperfect model predictions can redirect labels in a way that worsens particular unseen cases. The aggregate ablation must therefore be reported alongside this counterexample.
"""
    (ROOT / "docs/counterexample.md").write_text(report, encoding="utf-8")
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest(args.commit), indent=2) + "\n", encoding="utf-8"
    )
    print("Saved operational-risk ablation, raw/repair comparison, and counterexample")

if __name__ == "__main__": main()
