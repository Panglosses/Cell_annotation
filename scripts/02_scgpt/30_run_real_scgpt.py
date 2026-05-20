from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

LABEL_COL = "celltype_l4"


def require_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Required {name} not found: {path}")


def get_scgpt_embedding(adata):
    """
    scgpt.tasks.embed_data(return_new_adata=True) may return embeddings either:
    1. in adata.obsm["X_scGPT"] / similar keys, or
    2. directly in adata.X.
    We accept both, but never fall back to PCA.
    """
    candidates = ["X_scGPT", "X_scgpt", "X_cell_emb", "X_emb"]
    for key in candidates:
        if key in adata.obsm:
            X = np.asarray(adata.obsm[key])
            return X, key

    if adata.X is not None:
        X = np.asarray(adata.X)
        # scGPT embeddings should be a cell x embedding matrix, usually hundreds of dims,
        # not the original 22144-gene expression matrix.
        if X.ndim == 2 and X.shape[1] < 5000:
            return X, "X"

    raise KeyError(
        f"No valid scGPT embedding found. obsm keys: {list(adata.obsm.keys())}; X shape: {getattr(adata.X, 'shape', None)}"
    )


def prepare_adata_for_scgpt(adata):
    """
    Use counts_or_original layer if available.
    This avoids accidentally embedding PCA/scaled values.
    """
    adata = adata.copy()
    if "counts_or_original" in adata.layers:
        adata.X = adata.layers["counts_or_original"].copy()
        print("Using layer counts_or_original as X")
    else:
        print("WARNING: counts_or_original layer not found; using adata.X")
    return adata


def evaluate_and_save(test_adata, pred_labels, model_name):
    y_true = test_adata.obs[LABEL_COL].astype(str).values
    y_pred = np.asarray(pred_labels).astype(str)

    pred_df = pd.DataFrame({
        "cell_id": test_adata.obs_names.astype(str),
        "true_label": y_true,
        "pred_label": y_pred,
        "model": model_name,
    })

    assert len(pred_df) == test_adata.n_obs
    assert pred_df["pred_label"].notna().all()
    assert pred_df["true_label"].tolist() == test_adata.obs[LABEL_COL].astype(str).tolist()

    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    pred_path = Path(f"results/predictions/{model_name}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = pd.DataFrame([{
        "model": model_name,
        "method": "real_scgpt_embedding_plus_logistic_regression",
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }])
    metrics.to_csv(f"results/metrics/{model_name}_metrics.csv", index=False)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        f"results/tables/{model_name}_classification_report.csv"
    )

    labels = sorted(pd.Series(y_true).unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        f"results/metrics/{model_name}_confusion_matrix.csv"
    )

    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved predictions: {pred_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train_subset.h5ad")
    parser.add_argument("--test", default="data/processed/test_subset.h5ad")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gene-col", default="gene_symbol")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    require_file(model_dir / "vocab.json", "scGPT vocab.json")
    require_file(model_dir / "args.json", "scGPT args.json")
    require_file(model_dir / "best_model.pt", "scGPT best_model.pt")

    print("=" * 80)
    print("REAL scGPT embedding workflow")
    print("=" * 80)
    print("Model directory:", model_dir.resolve())

    import torchtext
    torchtext.disable_torchtext_deprecation_warning()

    import scgpt as scg

    train_raw = sc.read_h5ad(args.train)
    test_raw = sc.read_h5ad(args.test)

    if LABEL_COL not in train_raw.obs or LABEL_COL not in test_raw.obs:
        raise KeyError(f"Missing label column: {LABEL_COL}")

    if args.gene_col not in train_raw.var.columns:
        raise KeyError(f"{args.gene_col} not found in train.var. Available: {list(train_raw.var.columns)}")

    if args.gene_col not in test_raw.var.columns:
        raise KeyError(f"{args.gene_col} not found in test.var. Available: {list(test_raw.var.columns)}")

    print("Train raw:", train_raw)
    print("Test raw:", test_raw)
    print("Gene column:", args.gene_col)

    train = prepare_adata_for_scgpt(train_raw)
    test = prepare_adata_for_scgpt(test_raw)

    print("\nEmbedding train data with real scGPT model...")
    train_emb_adata = scg.tasks.embed_data(
        train,
        model_dir,
        gene_col=args.gene_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        obs_to_save=[LABEL_COL],
        device=args.device,
        use_fast_transformer=False,
        return_new_adata=True,
    )

    print("\nEmbedding test data with real scGPT model...")
    test_emb_adata = scg.tasks.embed_data(
        test,
        model_dir,
        gene_col=args.gene_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        obs_to_save=[LABEL_COL],
        device=args.device,
        use_fast_transformer=False,
        return_new_adata=True,
    )

    X_train, train_key = get_scgpt_embedding(train_emb_adata)
    X_test, test_key = get_scgpt_embedding(test_emb_adata)

    print("\nEmbedding keys:", train_key, test_key)
    print("Train embedding shape:", X_train.shape)
    print("Test embedding shape:", X_test.shape)

    if X_train.shape[0] != train_raw.n_obs or X_test.shape[0] != test_raw.n_obs:
        raise ValueError("Embedding row count does not match AnnData cells.")

    np.save("data/processed/train_X_scGPT.npy", X_train)
    np.save("data/processed/test_X_scGPT.npy", X_test)

    print("\nTraining LogisticRegression on real scGPT embeddings...")
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, train_raw.obs[LABEL_COL].astype(str).values)

    print("Predicting test labels...")
    pred = clf.predict(X_test)

    evaluate_and_save(test_raw, pred, "scgpt")

    audit = {
        "model": "scgpt",
        "model_dir": str(model_dir.resolve()),
        "required_files": ["vocab.json", "args.json", "best_model.pt"],
        "gene_col": args.gene_col,
        "embedding_key_train": train_key,
        "embedding_key_test": test_key,
        "train_embedding_shape": list(X_train.shape),
        "test_embedding_shape": list(X_test.shape),
        "method": "scgpt.tasks.embed_data + LogisticRegression",
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    with open("results/metrics/scgpt_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print("\nAudit saved to results/metrics/scgpt_audit.json")
    print("DONE: real scGPT workflow")


if __name__ == "__main__":
    main()
