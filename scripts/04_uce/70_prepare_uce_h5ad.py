from pathlib import Path
import numpy as np
import scanpy as sc
from scipy import sparse

OUTDIR = Path("data/uce")
OUTDIR.mkdir(parents=True, exist_ok=True)

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

    if "gene_symbol" not in adata.var.columns:
        raise KeyError("gene_symbol column not found in adata.var")

    gene_symbols = adata.var["gene_symbol"].astype(str).values
    valid = (gene_symbols != "") & (gene_symbols != "nan") & (gene_symbols != "None")
    adata = adata[:, valid].copy()

    adata.var_names = adata.var["gene_symbol"].astype(str).values
    adata.var_names_make_unique()

    adata.obs["joinid"] = adata.obs_names.astype(str)
    adata.obs["celltype_l4_uce"] = adata.obs["celltype_l4"].astype(str)
    adata.obs["celltype_l3_uce"] = adata.obs["celltype_l3"].astype(str)

    X = adata.X
    if sparse.issparse(X):
        n_counts = np.asarray(X.sum(axis=1)).ravel()
    else:
        n_counts = np.asarray(X.sum(axis=1)).ravel()
    adata.obs["n_counts_uce"] = n_counts

    outpath = OUTDIR / f"{split}_uce_input.h5ad"
    adata.write_h5ad(outpath)

    print("saved", outpath, adata.shape)
    print("var_names head:", list(adata.var_names[:10]))
    print("n_counts min/median/max:", float(n_counts.min()), float(np.median(n_counts)), float(n_counts.max()))
