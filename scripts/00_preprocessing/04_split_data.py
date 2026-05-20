from pathlib import Path
import scanpy as sc
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/processed/adata_preprocessed_subset.h5ad")
TRAIN_PATH = Path("data/processed/train_subset.h5ad")
TEST_PATH = Path("data/processed/test_subset.h5ad")
SPLIT_SUMMARY_PATH = Path("results/tables/split_summary.csv")

LABEL_COL = "celltype_l4"
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("=" * 80)
print("Loading preprocessed dataset")
print("=" * 80)

adata = sc.read_h5ad(INPUT_PATH)
print(adata)

if LABEL_COL not in adata.obs.columns:
    raise ValueError(f"Label column not found: {LABEL_COL}")

print("\nFull label distribution:")
print(adata.obs[LABEL_COL].value_counts())

train_idx, test_idx = train_test_split(
    adata.obs_names,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=adata.obs[LABEL_COL],
)

adata_train = adata[train_idx].copy()
adata_test = adata[test_idx].copy()

print("\nTrain AnnData:")
print(adata_train)

print("\nTest AnnData:")
print(adata_test)

print("\nTrain label distribution:")
train_counts = adata_train.obs[LABEL_COL].value_counts()
print(train_counts)

print("\nTest label distribution:")
test_counts = adata_test.obs[LABEL_COL].value_counts()
print(test_counts)

# Check leakage
overlap = set(adata_train.obs_names).intersection(set(adata_test.obs_names))
if overlap:
    raise ValueError(f"Data leakage detected: {len(overlap)} overlapping cells")

print("\nNo overlap between train and test cells.")

summary = pd.DataFrame({
    "celltype_l4": sorted(adata.obs[LABEL_COL].unique()),
})

summary["full_n"] = summary["celltype_l4"].map(adata.obs[LABEL_COL].value_counts())
summary["train_n"] = summary["celltype_l4"].map(train_counts)
summary["test_n"] = summary["celltype_l4"].map(test_counts)
summary["test_fraction"] = summary["test_n"] / summary["full_n"]

SPLIT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(SPLIT_SUMMARY_PATH, index=False)

adata_train.write_h5ad(TRAIN_PATH)
adata_test.write_h5ad(TEST_PATH)

print("\nSaved train data to:", TRAIN_PATH)
print("Saved test data to:", TEST_PATH)
print("Saved split summary to:", SPLIT_SUMMARY_PATH)

print("\nDone.")