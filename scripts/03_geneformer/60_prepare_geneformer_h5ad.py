from pathlib import Path
import numpy as np
import scanpy as sc
from scipy import sparse

OUT_BASE = Path("data/geneformer")
OUT_BASE.mkdir(parents=True, exist_ok=True)

for split, inpath in [
    ("train", "data/processed/train_subset.h5ad"),
    ("test", "data/processed/test_subset.h5ad"),
]:
    adata = sc.read_h5ad(inpath).copy()

    if "counts_or_original" in adata.layers:
        adata.X = adata.layers["counts_or_original"].copy()
        print(split, "using counts_or_original layer")
    else:
        print(split, "WARNING: using adata.X")

    # Geneformer needs Ensembl IDs.
    first_ids = [str(x) for x in adata.var_names[:20]]
    if sum(x.startswith("ENSG") for x in first_ids) < 5:
        raise ValueError(f"var_names do not look like Ensembl IDs: {first_ids[:10]}")

    adata.var["ensembl_id"] = adata.var_names.astype(str).str.replace(r"\.\d+$", "", regex=True)

    X = adata.X
    if sparse.issparse(X):
        n_counts = np.asarray(X.sum(axis=1)).ravel()
    else:
        n_counts = np.asarray(X.sum(axis=1)).ravel()

    adata.obs["n_counts"] = n_counts
    adata.obs["joinid"] = adata.obs_names.astype(str)
    adata.obs["celltype_l4_geneformer"] = adata.obs["celltype_l4"].astype(str)
    adata.obs["celltype_l3_geneformer"] = adata.obs["celltype_l3"].astype(str)

    outdir = OUT_BASE / f"{split}_input"
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"{split}.h5ad"
    adata.write_h5ad(outpath)

    print("saved:", outpath)
    print(adata)
    print("n_counts min/median/max:", np.min(n_counts), np.median(n_counts), np.max(n_counts))
