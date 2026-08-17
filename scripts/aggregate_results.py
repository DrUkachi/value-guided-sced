"""Create a provisional table and learning curves from completed seed runs."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "results/processed").glob("seed_*/results.csv"))
if not files: raise SystemExit("No per-seed results found")
rows = []
for path in files:
    with path.open(encoding="utf-8") as stream: rows.extend(csv.DictReader(stream))
table = ROOT / "results/processed/provisional_results.csv"
with table.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
config = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
final_seeds = {str(seed) for seed in config["evaluation"]["final_seeds"]}
main_rows = [row for row in rows if row["seed"] in final_seeds]
expected = len(final_seeds) * len(config["acquisition"]["methods"]) * (len(config["evaluation"]["budgets"]) - 1)
if len(main_rows) != expected:
    raise RuntimeError(f"expected {expected} final-seed rows, found {len(main_rows)}")
main_table = ROOT / "results/processed/main_results.csv"
with main_table.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(main_rows[0])); writer.writeheader(); writer.writerows(main_rows)
metrics = ("dispatch_mae_mw", "raw_feasible_percentage", "mean_repaired_cost_regret")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
debugging_seeds = {str(seed) for seed in config["evaluation"]["debugging_seeds"]}
plot_rows = [row for row in rows if row["seed"] in debugging_seeds]
for method in sorted({row["method"] for row in plot_rows}):
    selected = sorted((row for row in plot_rows if row["method"] == method), key=lambda row: int(row["budget"]))
    for axis, metric in zip(axes, metrics, strict=True):
        axis.plot([int(row["budget"]) for row in selected], [float(row[metric]) for row in selected], marker="o", label=method)
for axis, metric in zip(axes, metrics, strict=True):
    axis.set(xlabel="Target labels", title=metric.replace("_", " ")); axis.grid(alpha=.25)
axes[0].legend(fontsize=8); fig.tight_layout()
figure = ROOT / "results/figures/provisional_learning_curves.png"
fig.savefig(figure, dpi=180); plt.close(fig)
print(f"Saved {table}, {main_table}, and {figure}")
