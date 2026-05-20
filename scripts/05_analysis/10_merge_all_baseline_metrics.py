from pathlib import Path
import pandas as pd

METRICS_DIR = Path("results/metrics")

random_path = METRICS_DIR / "metrics_summary.csv"
cross_path = METRICS_DIR / "cross_tissue_baseline_metrics.csv"
out_path = METRICS_DIR / "all_baseline_metrics_summary.csv"

print("=" * 80)
print("Merging all baseline metrics")
print("=" * 80)

random_df = pd.read_csv(random_path)
random_df["setting"] = "random_stratified_split"
random_df["train_tissue"] = "mixed"
random_df["test_tissue"] = "mixed"

# Reorder random columns to match cross-tissue as much as possible
cross_df = pd.read_csv(cross_path)

# Make sure both have same columns
all_columns = sorted(set(random_df.columns).union(set(cross_df.columns)))

for col in all_columns:
    if col not in random_df.columns:
        random_df[col] = None
    if col not in cross_df.columns:
        cross_df[col] = None

random_df = random_df[all_columns]
cross_df = cross_df[all_columns]

merged = pd.concat([random_df, cross_df], ignore_index=True)

# Put useful columns first
preferred_cols = [
    "model",
    "setting",
    "train_tissue",
    "test_tissue",
    "n_train",
    "n_test",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]

other_cols = [c for c in merged.columns if c not in preferred_cols]
merged = merged[preferred_cols + other_cols]

merged.to_csv(out_path, index=False)

print("\nMerged metrics:")
print(merged[preferred_cols])

print("\nSaved to:", out_path)
print("\nDone.")