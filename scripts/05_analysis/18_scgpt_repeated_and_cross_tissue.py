#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torchtext
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

torchtext.disable_torchtext_deprecation_warning()
import scgpt as scg


PROJECT = Path("/root/dty/project/celltype_fm_benchmark")
MODEL_DIR = Path("/root/dty/models/scGPT_human")
OUT = PROJECT / "rerun_logs" / "depth_extension"
OUT.mkdir(parents=True, exist_ok=True)
for sub in ["metrics", "tables", "predictions", "embeddings"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

LABEL_COL = "celltype_l4"
TISSUE_COL = "tissue"
GENE_COL = "gene_symbol"


def make_clean_h5ad() -> Path:
    src = PROJECT / "adata_preprocessed_subset.h5ad"
    dst = OUT / "adata_preprocessed_subset_clean_for_scanpy.h5ad"
    if not dst.exists():
        shutil.copy2(src, dst)
        with h5py.File(dst, "a") as f:
            if "uns/log1p/base" in f:
                del f["uns/log1p/base"]
    return dst


def prepare_for_scgpt(adata):
    adata = adata.copy()
    if "counts_or_original" in adata.layers:
        adata.X = adata.layers["counts_or_original"].copy()
    return adata


def get_embedding(adata):
    candidates = ["X_scGPT", "X_scgpt", "X_cell_emb", "X_emb"]
    for key in candidates:
        if key in adata.obsm:
            return np.asarray(adata.obsm[key]), f"obsm[{key}]"
    x = np.asarray(adata.X)
    if x.ndim == 2 and x.shape[1] < 5000:
        return x, "X"
    raise RuntimeError(f"No scGPT embedding found; obsm={list(adata.obsm.keys())}, X={getattr(adata.X, 'shape', None)}")


def score(y_true, y_pred):
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf,
        "weighted_precision": wp,
        "weighted_recall": wr,
        "weighted_f1": wf,
    }


def save_outputs(model_name, y_true, y_pred, cell_ids):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    pd.DataFrame(
        {"cell_id": cell_ids.astype(str), "true_label": y_true, "pred_label": y_pred, "model": model_name}
    ).to_csv(OUT / "predictions" / f"{model_name}_predictions.csv", index=False)
    metrics = pd.DataFrame([{"model": model_name, **score(y_true, y_pred)}])
    metrics.to_csv(OUT / "metrics" / f"{model_name}_metrics.csv", index=False)
    pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).transpose().to_csv(
        OUT / "tables" / f"{model_name}_classification_report.csv"
    )
    labels = sorted(pd.Series(y_true).unique())
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels).to_csv(
        OUT / "metrics" / f"{model_name}_confusion_matrix.csv"
    )
    return metrics


def train_eval_lr_mlp(x_train, y_train, x_test, y_test, cell_ids, prefix):
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(x_train, y_train)
    lr_metrics = save_outputs(f"{prefix}_lr", y_test, lr.predict(x_test), cell_ids)

    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train)
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        ),
    )
    mlp.fit(x_train, y_train_int)
    pred = le.inverse_transform(mlp.predict(x_test).astype(int))
    mlp_metrics = save_outputs(f"{prefix}_mlp", y_test, pred, cell_ids)
    return pd.concat([lr_metrics, mlp_metrics], ignore_index=True)


def repeated_scgpt_splits(x, y, seeds=range(10)):
    rows = []
    for seed in seeds:
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        lr.fit(x_train, y_train)
        rows.append({"split_seed": seed, "model": "scGPT_LR", **score(y_test, lr.predict(x_test))})

        le = LabelEncoder()
        y_train_int = le.fit_transform(y_train)
        mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=seed,
            ),
        )
        mlp.fit(x_train, y_train_int)
        pred = le.inverse_transform(mlp.predict(x_test).astype(int))
        rows.append({"split_seed": seed, "model": "scGPT_MLP", **score(y_test, pred)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics" / "repeated_split_scgpt_metrics_by_seed.csv", index=False)
    summary = (
        df.groupby("model")
        .agg(
            n_splits=("split_seed", "nunique"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_sd=("weighted_f1", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "metrics" / "repeated_split_scgpt_summary.csv", index=False)
    return summary


def main():
    clean = make_clean_h5ad()
    adata = sc.read_h5ad(clean)
    adata.obs[TISSUE_COL] = adata.obs[TISSUE_COL].astype(str)
    adata.obs[LABEL_COL] = adata.obs[LABEL_COL].astype(str)

    blood = prepare_for_scgpt(adata[adata.obs[TISSUE_COL] == "blood"].copy())
    syn = prepare_for_scgpt(adata[adata.obs[TISSUE_COL] == "synovial fluid"].copy())
    if blood.n_obs == 0 or syn.n_obs == 0:
        raise RuntimeError(adata.obs[TISSUE_COL].value_counts().to_string())

    print("blood", blood.shape, "synovial", syn.shape)
    print("embedding blood with scGPT")
    blood_emb = scg.tasks.embed_data(
        blood,
        MODEL_DIR,
        gene_col=GENE_COL,
        max_length=1200,
        batch_size=8,
        obs_to_save=[LABEL_COL, TISSUE_COL],
        device="cuda",
        use_fast_transformer=False,
        return_new_adata=True,
    )
    print("embedding synovial fluid with scGPT")
    syn_emb = scg.tasks.embed_data(
        syn,
        MODEL_DIR,
        gene_col=GENE_COL,
        max_length=1200,
        batch_size=8,
        obs_to_save=[LABEL_COL, TISSUE_COL],
        device="cuda",
        use_fast_transformer=False,
        return_new_adata=True,
    )

    x_blood, blood_key = get_embedding(blood_emb)
    x_syn, syn_key = get_embedding(syn_emb)
    np.save(OUT / "embeddings" / "blood_X_scGPT.npy", x_blood)
    np.save(OUT / "embeddings" / "synovial_X_scGPT.npy", x_syn)
    pd.DataFrame({"cell_id": blood.obs_names.astype(str), "celltype_l4": blood.obs[LABEL_COL].values, "tissue": "blood"}).to_csv(
        OUT / "embeddings" / "blood_scgpt_obs.csv", index=False
    )
    pd.DataFrame({"cell_id": syn.obs_names.astype(str), "celltype_l4": syn.obs[LABEL_COL].values, "tissue": "synovial fluid"}).to_csv(
        OUT / "embeddings" / "synovial_scgpt_obs.csv", index=False
    )

    y_blood = blood.obs[LABEL_COL].values.astype(str)
    y_syn = syn.obs[LABEL_COL].values.astype(str)
    cross = train_eval_lr_mlp(
        x_blood,
        y_blood,
        x_syn,
        y_syn,
        syn.obs_names.astype(str).values,
        "scgpt_blood_to_synovial",
    )
    cross.to_csv(OUT / "metrics" / "scgpt_blood_to_synovial_summary.csv", index=False)

    x_all = np.vstack([x_blood, x_syn])
    y_all = np.concatenate([y_blood, y_syn])
    repeated = repeated_scgpt_splits(x_all, y_all)

    audit = {
        "model": "scGPT",
        "model_dir": str(MODEL_DIR),
        "blood_cells": int(blood.n_obs),
        "synovial_cells": int(syn.n_obs),
        "blood_embedding_shape": list(x_blood.shape),
        "synovial_embedding_shape": list(x_syn.shape),
        "embedding_keys": {"blood": blood_key, "synovial": syn_key},
        "outputs": str(OUT),
    }
    with open(OUT / "metrics" / "scgpt_depth_extension_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print("===== scGPT cross tissue =====")
    print(cross.to_string(index=False))
    print("===== scGPT repeated splits =====")
    print(repeated.to_string(index=False))


if __name__ == "__main__":
    main()
