#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT = Path("/root/dty/project/celltype_fm_benchmark")
OUT = PROJECT / "rerun_logs" / "depth_extension"
OUT.mkdir(parents=True, exist_ok=True)


def decode_array(x):
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in x], dtype=object)


def read_obs_col(obs, col):
    obj = obs[col]
    if isinstance(obj, h5py.Group):
        cats = decode_array(obj["categories"][()])
        codes = obj["codes"][()]
        vals = np.empty(len(codes), dtype=object)
        vals[:] = None
        mask = codes >= 0
        vals[mask] = cats[codes[mask]]
        return vals.astype(str)
    return decode_array(obj[()]).astype(str)


def read_data():
    path = PROJECT / "adata_preprocessed_subset.h5ad"
    with h5py.File(path, "r") as f:
        x = f["obsm"]["X_pca_custom"][()]
        y = read_obs_col(f["obs"], "celltype_l4")
    return x, y


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


def main():
    x, y = read_data()
    rows = []
    seeds = list(range(10))
    for seed in seeds:
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        models = {
            "PCA_LR": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=-1),
            "PCA_kNN": KNeighborsClassifier(n_neighbors=15, weights="distance", metric="euclidean"),
        }
        for name, clf in models.items():
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            rows.append({"split_seed": seed, "model": name, **score(y_test, pred)})

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
        rows.append({"split_seed": seed, "model": "PCA_MLP", **score(y_test, pred)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "repeated_split_pca_metrics_by_seed.csv", index=False)
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
    summary.to_csv(OUT / "repeated_split_pca_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
