from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

metric_files = [
    METRICS_DIR / "logreg_metrics.csv",
    METRICS_DIR / "knn_metrics.csv",
]

summary_path = METRICS_DIR / "metrics_summary.csv"
figure_path = FIG_DIR / "baseline_model_comparison.png"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Summarizing baseline metrics")
print("=" * 80)

dfs = []

for path in metric_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing metric file: {path}")
    df = pd.read_csv(path)
    dfs.append(df)

summary = pd.concat(dfs, ignore_index=True)

# Sort by macro_f1, because macro F1 is important for imbalanced cell types
summary = summary.sort_values("macro_f1", ascending=False)

summary.to_csv(summary_path, index=False)

print("\nMetrics summary:")
print(summary)

# Plot main metrics
plot_df = summary.set_index("model")[["accuracy", "macro_f1", "weighted_f1"]]

ax = plot_df.plot(kind="bar", figsize=(8, 5))
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Baseline model comparison")
ax.legend(loc="lower right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(figure_path, dpi=300)
plt.close()

print("\nSaved summary to:", summary_path)
print("Saved figure to:", figure_path)

print("\nDone.")