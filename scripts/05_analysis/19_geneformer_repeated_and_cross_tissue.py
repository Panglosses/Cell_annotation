from pathlib import Path
import json

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE = Path("/root/dty/project/celltype_fm_benchmark")
OUT = BASE / "rerun_logs" / "depth_extension"
METRICS = OUT / "metrics"
TABLES = OUT / "tables"
PREDS = OUT / "predictions"
for d in [METRICS, TABLES, PREDS]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = BASE / "data/geneformer/embeddings/jia_train_geneformer_v1.csv"
TEST_CSV = BASE / "data/geneformer/embeddings/jia_test_geneformer_v1.csv"
TRAIN_H5AD = BASE / "data/processed/train_subset.h5ad"
TEST_H5AD = BASE / "data/processed/test_subset.h5ad"

LABEL = "celltype_l4"
EMBED_COLS = [str(i) for i in range(256)]
SEEDS = list(range(10))


def load_embeddings():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    all_df = pd.concat([train, test], ignore_index=True)
    missing = [c for c in EMBED_COLS if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing embedding columns: {missing[:10]}")

    obs_parts = []
    for p in [TRAIN_H5AD, TEST_H5AD]:
        ad = sc.read_h5ad(p)
        obs = ad.obs[["celltype_l4", "celltype_l3", "tissue"]].copy()
        obs["joinid"] = ad.obs["joinid"].astype(str).values if "joinid" in ad.obs.columns else ad.obs_names.astype(str)
        obs_parts.append(obs)
    obs_df = pd.concat(obs_parts, axis=0).drop_duplicates("joinid")

    all_df["joinid"] = all_df["joinid"].astype(str)
    merged = all_df.merge(obs_df[["joinid", "tissue"]], on="joinid", how="left")
    if merged["tissue"].isna().any():
        raise ValueError(f"Missing tissue for {int(merged['tissue'].isna().sum())} cells")
    return merged


def metrics_row(model, setting, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "model": model,
        "setting": setting,
        "accuracy": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf,
        "weighted_precision": wp,
        "weighted_recall": wr,
        "weighted_f1": wf,
    }


def make_lr(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )


def make_mlp(seed):
    return make_pipeline(
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
            verbose=False,
        ),
    )


def fit_predict(model_name, clf, X_train, y_train, X_test):
    if model_name.endswith("MLP"):
        le = LabelEncoder()
        y_int = le.fit_transform(y_train)
        clf.fit(X_train, y_int)
        pred_int = clf.predict(X_test)
        return le.inverse_transform(pred_int.astype(int))
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def repeated_splits(df):
    X = df[EMBED_COLS].to_numpy(np.float32)
    y = df[LABEL].astype(str).to_numpy()
    rows = []
    for seed in SEEDS:
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
        for model_name, factory in [
            ("Geneformer_LR", make_lr),
            ("Geneformer_MLP", make_mlp),
        ]:
            clf = factory(seed)
            pred = fit_predict(model_name, clf, X[train_idx], y[train_idx], X[test_idx])
            row = metrics_row(model_name, "repeated_random_split", y[test_idx], pred)
            row["seed"] = seed
            rows.append(row)
            print(model_name, "seed", seed, "weighted_f1", row["weighted_f1"])
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(METRICS / "repeated_split_geneformer_metrics_by_seed.csv", index=False)
    summary = (
        by_seed.groupby("model")
        .agg(
            n_splits=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_sd=("weighted_f1", "std"),
        )
        .reset_index()
    )
    summary.to_csv(METRICS / "repeated_split_geneformer_summary.csv", index=False)
    print(summary.to_string(index=False))


def cross_tissue(df):
    blood = df[df["tissue"].astype(str).str.lower().eq("blood")].copy()
    syn = df[df["tissue"].astype(str).str.lower().eq("synovial fluid")].copy()
    X_train = blood[EMBED_COLS].to_numpy(np.float32)
    y_train = blood[LABEL].astype(str).to_numpy()
    X_test = syn[EMBED_COLS].to_numpy(np.float32)
    y_true = syn[LABEL].astype(str).to_numpy()
    rows = []
    for model_name, clf in [
        ("geneformer_blood_to_synovial_lr", make_lr(42)),
        ("geneformer_blood_to_synovial_mlp", make_mlp(42)),
    ]:
        pred = fit_predict("Geneformer_MLP" if model_name.endswith("mlp") else "Geneformer_LR", clf, X_train, y_train, X_test)
        rows.append(metrics_row(model_name, "blood_to_synovial", y_true, pred))
        pred_df = pd.DataFrame({
            "cell_id": syn["joinid"].astype(str).values,
            "true_label": y_true,
            "pred_label": pred.astype(str),
            "model": model_name,
        })
        pred_df.to_csv(PREDS / f"{model_name}_predictions.csv", index=False)
        pd.DataFrame(classification_report(y_true, pred, output_dict=True, zero_division=0)).transpose().to_csv(
            TABLES / f"{model_name}_classification_report.csv"
        )
        labels = sorted(pd.Series(y_true).unique())
        pd.DataFrame(confusion_matrix(y_true, pred, labels=labels), index=labels, columns=labels).to_csv(
            METRICS / f"{model_name}_confusion_matrix.csv"
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(METRICS / "geneformer_blood_to_synovial_summary.csv", index=False)
    audit = {
        "foundation_model": "Geneformer",
        "checkpoint": "Geneformer-V1-10M",
        "embedding_source": "rerun EmbExtractor cell embeddings",
        "train_tissue": "blood",
        "test_tissue": "synovial fluid",
        "n_blood": int(len(blood)),
        "n_synovial_fluid": int(len(syn)),
        "label_col": LABEL,
        "embedding_dim": len(EMBED_COLS),
    }
    (METRICS / "geneformer_blood_to_synovial_audit.json").write_text(json.dumps(audit, indent=2))
    print(summary.to_string(index=False))


def main():
    df = load_embeddings()
    print("merged", df.shape)
    print(df["tissue"].value_counts().to_string())
    print(df[LABEL].value_counts().to_string())
    repeated_splits(df)
    cross_tissue(df)


if __name__ == "__main__":
    main()
