from pathlib import Path
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = Path("data/processed/adata_balanced_subset.h5ad")
OUTPUT_PATH = Path("data/processed/adata_preprocessed_subset.h5ad")

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

LABEL_COL = "celltype_l4"
TISSUE_COL = "tissue"
DONOR_COL = "donor_id"
ASSAY_COL = "assay"
MITO_COL = "percent_mitochondrial"

MIN_GENES = 200
MAX_MITO = 20
N_TOP_GENES = 2000
N_PCS = 50

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Loading balanced subset")
print("=" * 80)

adata = sc.read_h5ad(INPUT_PATH)
print(adata)

required_cols = [LABEL_COL, TISSUE_COL, DONOR_COL, ASSAY_COL, MITO_COL]
for col in required_cols:
    if col not in adata.obs.columns:
        raise ValueError(f"Required column not found: {col}")

print("\nOriginal shape:")
print(adata.shape)

print("\nOriginal label distribution:")
print(adata.obs[LABEL_COL].value_counts())

print("\nOriginal tissue distribution:")
print(adata.obs[TISSUE_COL].value_counts())

print("\nOriginal assay distribution:")
print(adata.obs[ASSAY_COL].value_counts())

# Save QC summary before filtering
qc_before = pd.DataFrame({
    "metric": [
        "n_cells",
        "n_genes",
        "median_nCount_RNA",
        "median_nFeature_RNA",
        "median_percent_mitochondrial",
    ],
    "value": [
        adata.n_obs,
        adata.n_vars,
        adata.obs["nCount_RNA"].median() if "nCount_RNA" in adata.obs else None,
        adata.obs["nFeature_RNA"].median() if "nFeature_RNA" in adata.obs else None,
        adata.obs[MITO_COL].median(),
    ],
})
qc_before.to_csv(TABLE_DIR / "qc_before_filtering.csv", index=False)

print("\nFiltering low-quality cells...")
adata = adata[adata.obs["nFeature_RNA"] >= MIN_GENES].copy()
adata = adata[adata.obs[MITO_COL] <= MAX_MITO].copy()

print("\nShape after QC:")
print(adata.shape)

print("\nLabel distribution after QC:")
print(adata.obs[LABEL_COL].value_counts())

qc_after = pd.DataFrame({
    "metric": [
        "n_cells",
        "n_genes",
        "median_nCount_RNA",
        "median_nFeature_RNA",
        "median_percent_mitochondrial",
    ],
    "value": [
        adata.n_obs,
        adata.n_vars,
        adata.obs["nCount_RNA"].median() if "nCount_RNA" in adata.obs else None,
        adata.obs["nFeature_RNA"].median() if "nFeature_RNA" in adata.obs else None,
        adata.obs[MITO_COL].median(),
    ],
})
qc_after.to_csv(TABLE_DIR / "qc_after_filtering.csv", index=False)

# Keep raw counts-like matrix in a layer if possible
print("\nSaving current X to layer: counts_or_original")
adata.layers["counts_or_original"] = adata.X.copy()

print("\nNormalization and log transform...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print("\nSelecting highly variable genes...")
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=N_TOP_GENES,
    flavor="seurat",
)

print("Number of highly variable genes:", int(adata.var["highly_variable"].sum()))

print("\nScaling, PCA, neighbors, UMAP...")
adata_hvg = adata[:, adata.var["highly_variable"]].copy()

sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=N_PCS, svd_solver="arpack")
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=N_PCS)
sc.tl.umap(adata_hvg)

# Put embeddings back into main AnnData
adata.obsm["X_pca_custom"] = adata_hvg.obsm["X_pca"]
adata.obsm["X_umap_custom"] = adata_hvg.obsm["X_umap"]

print("\nPlotting UMAPs...")

# Temporarily tell scanpy to use our custom UMAP
adata.obsm["X_umap"] = adata.obsm["X_umap_custom"]

sc.pl.umap(
    adata,
    color=LABEL_COL,
    legend_loc="right margin",
    show=False,
)
plt.savefig(FIG_DIR / "umap_celltype_l4.png", dpi=300, bbox_inches="tight")
plt.close()

sc.pl.umap(
    adata,
    color=TISSUE_COL,
    legend_loc="right margin",
    show=False,
)
plt.savefig(FIG_DIR / "umap_tissue.png", dpi=300, bbox_inches="tight")
plt.close()

sc.pl.umap(
    adata,
    color=ASSAY_COL,
    legend_loc="right margin",
    show=False,
)
plt.savefig(FIG_DIR / "umap_assay.png", dpi=300, bbox_inches="tight")
plt.close()

sc.pl.umap(
    adata,
    color=DONOR_COL,
    legend_loc="right margin",
    show=False,
)
plt.savefig(FIG_DIR / "umap_donor.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaving processed AnnData...")
adata.write_h5ad(OUTPUT_PATH)

print("\nSaved processed data to:", OUTPUT_PATH)
print("Saved figures to:", FIG_DIR)

print("\nFinal AnnData:")
print(adata)

print("\nDone.")