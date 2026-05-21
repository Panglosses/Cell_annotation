#!/usr/bin/env python
"""Post hoc checks used after the main benchmark.

This script is intentionally separate from the main benchmark pipeline. It does
not rerun foundation-model embedding extraction. It uses the saved train/test
h5ad files and saved prediction CSVs to:

1. train PCA + Logistic Regression and PCA + MLP on the existing split;
2. compute paired bootstrap uncertainty for weighted F1;
3. perform an internal C1/C2/C3 train/test marker-direction check.

Run from the project root after the main outputs have been generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


LABEL_COL = "celltype_l4"
PCA_KEY = "X_pca_custom"


def decode_array(x):
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in x], dtype=object)


def read_obs_col(obs_group, col):
    obj = obs_group[col]
    if isinstance(obj, h5py.Group):
        cats = decode_array(obj["categories"][()])
        codes = obj["codes"][()]
        vals = np.empty(len(codes), dtype=object)
        vals[:] = None
        mask = codes >= 0
        vals[mask] = cats[codes[mask]]
        return vals.astype(str)
    return decode_array(obj[()]).astype(str)


def read_csr(group):
    shape = tuple(int(x) for x in group.attrs["shape"])
    return sparse.csr_matrix((group["data"][()], group["indices"][()], group["indptr"][()]), shape=shape)


def read_h5ad_minimal(path: Path, need_x: bool = False):
    with h5py.File(path, "r") as f:
        return {
            "cell_ids": read_obs_col(f["obs"], "_index"),
            "labels": read_obs_col(f["obs"], LABEL_COL),
            "pca": f["obsm"][PCA_KEY][()],
            "genes": decode_array(f["var"]["gene_symbol"][()]),
            "x": read_csr(f["X"]) if need_x else None,
        }


def save_outputs(out, model_name, method, y_true, y_pred, cell_ids):
    pred = pd.DataFrame(
        {"cell_id": cell_ids.astype(str), "true_label": y_true.astype(str), "pred_label": y_pred.astype(str), "model": model_name}
    )
    pred.to_csv(out / "predictions" / f"{model_name}_predictions.csv", index=False)

    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = pd.DataFrame([{
        "model": model_name,
        "method": method,
        "accuracy": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf,
        "weighted_precision": wp,
        "weighted_recall": wr,
        "weighted_f1": wf,
    }])
    metrics.to_csv(out / "metrics" / f"{model_name}_metrics.csv", index=False)

    pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).transpose().to_csv(
        out / "tables" / f"{model_name}_classification_report.csv"
    )
    labels = sorted(pd.Series(y_true).unique())
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels).to_csv(
        out / "metrics" / f"{model_name}_confusion_matrix.csv"
    )
    return metrics


def read_prediction_file(path, alias):
    df = pd.read_csv(path)[["cell_id", "true_label", "pred_label"]].copy()
    df["cell_id"] = df["cell_id"].astype(str)
    df["true_label"] = df["true_label"].astype(str)
    df["pred_label"] = df["pred_label"].astype(str)
    return df.rename(columns={"pred_label": alias})


def bootstrap_weighted_f1(out, pred_files, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    merged = None
    for alias, path in pred_files.items():
        if not path.exists():
            continue
        df = read_prediction_file(path, alias)
        merged = df if merged is None else merged.merge(df, on=["cell_id", "true_label"], how="inner")

    y_true = merged["true_label"].to_numpy()
    model_cols = list(pred_files.keys())
    model_cols = [m for m in model_cols if m in merged.columns]
    n = len(merged)
    boot = {m: np.empty(n_boot, dtype=float) for m in model_cols}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        for m in model_cols:
            boot[m][b] = f1_score(yt, merged[m].to_numpy()[idx], average="weighted")

    ci_rows = []
    for m in model_cols:
        vals = boot[m]
        ci_rows.append({
            "model": m,
            "n_cells": n,
            "observed_weighted_f1": f1_score(y_true, merged[m].to_numpy(), average="weighted"),
            "bootstrap_mean": vals.mean(),
            "ci95_low": np.quantile(vals, 0.025),
            "ci95_high": np.quantile(vals, 0.975),
        })
    pd.DataFrame(ci_rows).to_csv(out / "tables" / "paired_bootstrap_weighted_f1_ci.csv", index=False)

    comparisons = [
        ("PCA_LR", "PCA_MLP"),
        ("PCA_LR", "scGPT_MLP"),
        ("PCA_LR", "UCE_MLP"),
        ("PCA_LR", "Geneformer_MLP"),
        ("scGPT_LR", "scGPT_MLP"),
        ("UCE_LR", "UCE_MLP"),
        ("Geneformer_LR", "Geneformer_MLP"),
        ("scGPT_MLP", "UCE_MLP"),
        ("PCA_MLP", "scGPT_MLP"),
        ("PCA_MLP", "UCE_MLP"),
    ]
    diff_rows = []
    for a, b in comparisons:
        if a not in boot or b not in boot:
            continue
        diff = boot[b] - boot[a]
        obs = f1_score(y_true, merged[b].to_numpy(), average="weighted") - f1_score(y_true, merged[a].to_numpy(), average="weighted")
        p_two = min(1.0, float(2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))
        diff_rows.append({
            "comparison": f"{b} minus {a}",
            "observed_diff": obs,
            "ci95_low": np.quantile(diff, 0.025),
            "ci95_high": np.quantile(diff, 0.975),
            "bootstrap_two_sided_p": p_two,
        })
    pd.DataFrame(diff_rows).to_csv(out / "tables" / "paired_bootstrap_weighted_f1_differences.csv", index=False)


def marker_stability(out, train, test, states=("C1", "C2", "C3")):
    rows = []
    genes = train["genes"].astype(str)
    for state in states:
        tr_mask = train["labels"] == state
        te_mask = test["labels"] == state
        tr_in = np.asarray(train["x"][tr_mask].mean(axis=0)).ravel()
        tr_out = np.asarray(train["x"][~tr_mask].mean(axis=0)).ravel()
        te_in = np.asarray(test["x"][te_mask].mean(axis=0)).ravel()
        te_out = np.asarray(test["x"][~te_mask].mean(axis=0)).ravel()
        tr_diff = tr_in - tr_out
        te_diff = te_in - te_out
        top20 = np.argsort(-tr_diff)[:20]
        top50 = np.argsort(-tr_diff)[:50]
        rho, p = spearmanr(tr_diff[top50], te_diff[top50])
        rows.append({
            "state": state,
            "train_cells": int(tr_mask.sum()),
            "test_cells": int(te_mask.sum()),
            "top20_test_direction_consistency": float(np.mean(te_diff[top20] > 0)),
            "top50_train_test_spearman": float(rho),
            "top50_spearman_p": float(p),
            "note": "Internal train/test marker-direction check only; not external biological validation.",
        })
    pd.DataFrame(rows).to_csv(out / "tables" / "C1_C2_C3_internal_marker_stability_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-root", type=Path, default=Path("results/posthoc"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    out = args.output_root
    for sub in ["metrics", "predictions", "tables"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    train = read_h5ad_minimal(args.project_root / "data/processed/train_subset.h5ad")
    test = read_h5ad_minimal(args.project_root / "data/processed/test_subset.h5ad")

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, n_jobs=-1)
    lr.fit(train["pca"], train["labels"])
    lr_metrics = save_outputs(out, "pca_logreg_same_split", "PCA_50_plus_logistic_regression_recomputed_same_split", test["labels"], lr.predict(test["pca"]), test["cell_ids"])

    le = LabelEncoder()
    y_train = le.fit_transform(train["labels"])
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=300, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=42),
    )
    mlp.fit(train["pca"], y_train)
    mlp_pred = le.inverse_transform(mlp.predict(test["pca"]).astype(int))
    mlp_metrics = save_outputs(out, "pca_mlp_same_split", "PCA_50_plus_MLPClassifier_same_architecture_as_foundation_MLP_probes", test["labels"], mlp_pred, test["cell_ids"])

    bootstrap_weighted_f1(
        out,
        {
            "PCA_LR": out / "predictions" / "pca_logreg_same_split_predictions.csv",
            "PCA_MLP": out / "predictions" / "pca_mlp_same_split_predictions.csv",
            "scGPT_LR": args.results_root / "predictions" / "scgpt_predictions.csv",
            "scGPT_MLP": args.results_root / "predictions" / "scgpt_mlp_probe_predictions.csv",
            "UCE_LR": args.results_root / "predictions" / "uce_logreg_probe_predictions.csv",
            "UCE_MLP": args.results_root / "predictions" / "uce_mlp_probe_predictions.csv",
            "Geneformer_LR": args.results_root / "predictions" / "geneformer_logreg_probe_predictions.csv",
            "Geneformer_MLP": args.results_root / "predictions" / "geneformer_mlp_probe_predictions.csv",
        },
        n_boot=args.n_bootstrap,
    )

    train_x = read_h5ad_minimal(args.project_root / "data/processed/train_subset.h5ad", need_x=True)
    test_x = read_h5ad_minimal(args.project_root / "data/processed/test_subset.h5ad", need_x=True)
    marker_stability(out, train_x, test_x)

    with open(out / "analysis_summary.json", "w") as f:
        json.dump({
            "pca_logreg_weighted_f1": float(lr_metrics.loc[0, "weighted_f1"]),
            "pca_mlp_weighted_f1": float(mlp_metrics.loc[0, "weighted_f1"]),
            "n_bootstrap": args.n_bootstrap,
        }, f, indent=2)


if __name__ == "__main__":
    main()
