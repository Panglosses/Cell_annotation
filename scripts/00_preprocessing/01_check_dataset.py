from pathlib import Path
import scanpy as sc

DATA_PATH = Path("data/raw/innate_lymphocytes_jia.h5ad")

print("=" * 80)
print("Checking dataset")
print("=" * 80)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

adata = sc.read_h5ad(DATA_PATH)

print("\nAnnData object:")
print(adata)

print("\nNumber of cells:", adata.n_obs)
print("Number of genes:", adata.n_vars)

print("\nobs columns:")
for col in adata.obs.columns:
    print("-", col)

print("\nPossible cell type / annotation columns:")
for col in adata.obs.columns:
    name = col.lower()
    if (
        "cell" in name
        or "type" in name
        or "annot" in name
        or "label" in name
        or "cluster" in name
    ):
        print("\n" + "=" * 80)
        print(f"Column: {col}")
        print("Unique values:", adata.obs[col].nunique())
        print(adata.obs[col].value_counts(dropna=False).head(30))

print("\nPossible tissue / batch / donor / sample columns:")
keywords = [
    "tissue",
    "disease",
    "assay",
    "donor",
    "sample",
    "batch",
    "patient",
    "subject",
    "dataset",
]

for col in adata.obs.columns:
    name = col.lower()
    if any(k in name for k in keywords):
        print("\n" + "=" * 80)
        print(f"Column: {col}")
        print("Unique values:", adata.obs[col].nunique())
        print(adata.obs[col].value_counts(dropna=False).head(20))

print("\nBasic checks:")
print("cells > 2000:", adata.n_obs > 2000)
print("genes > 1000:", adata.n_vars > 1000)

print("\nDone.")