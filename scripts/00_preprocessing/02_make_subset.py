from pathlib import Path
import scanpy as sc
import pandas as pd

RAW_PATH = Path("data/raw/innate_lymphocytes_jia.h5ad")
OUT_PATH = Path("data/processed/adata_balanced_subset.h5ad")
SUMMARY_PATH = Path("results/tables/subset_celltype_summary.csv")

LABEL_COL = "celltype_l4"
MAX_CELLS_PER_CLASS = 3000
RANDOM_STATE = 42

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Loading raw dataset")
print("=" * 80)

adata = sc.read_h5ad(RAW_PATH)

print(adata)

if LABEL_COL not in adata.obs.columns:
    raise ValueError(f"Label column not found: {LABEL_COL}")

print("\nOriginal label distribution:")
print(adata.obs[LABEL_COL].value_counts())

# Remove missing labels if any
adata = adata[adata.obs[LABEL_COL].notna()].copy()

# Balanced downsampling by label
sampled_indices = []

for label, df in adata.obs.groupby(LABEL_COL, observed=True):
    n = min(len(df), MAX_CELLS_PER_CLASS)
    sampled = df.sample(n=n, random_state=RANDOM_STATE).index
    sampled_indices.extend(sampled)

adata_subset = adata[sampled_indices].copy()

print("\nSubset AnnData:")
print(adata_subset)

print("\nSubset label distribution:")
summary = adata_subset.obs[LABEL_COL].value_counts().reset_index()
summary.columns = [LABEL_COL, "n_cells"]
print(summary)

adata_subset.write_h5ad(OUT_PATH)
summary.to_csv(SUMMARY_PATH, index=False)

print("\nSaved subset to:", OUT_PATH)
print("Saved summary to:", SUMMARY_PATH)
print("\nDone.")