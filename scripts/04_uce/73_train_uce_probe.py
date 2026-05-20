from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

TRAIN_UCE = Path("data/uce/train_outputtrain_uce_input_uce_adata.h5ad")
TEST_UCE = Path("data/uce/test_outputtest_uce_input_uce_adata.h5ad")

LABEL_COL = "celltype_l4"
Path("results/predictions").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)


def get_embedding(adata):
    print("Available obsm keys:", list(adata.obsm.keys()))
    candidates = [
        "X_uce",
        "X_UCE",
        "uce",
        "UCE",
        "X_emb",
        "X_embedding",
        "X_cell_emb",
    ]
    for key in candidates:
        if key in adata.obsm:
            X = adata.obsm[key]
            if sparse.issparse(X):
                X = X.toarray()
            X = np.asarray(X)
            if X.ndim == 2 and X.shape[0] == adata.n_obs:
                return X, f"obsm[{key}]"

    # If UCE wrote embeddings into X, use X only if it looks like embedding dimensions.
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    if X.ndim == 2 and X.shape[0] == adata.n_obs and X.shape[1] <= 5000:
        return X, "X"

    raise RuntimeError(
        f"No valid UCE embedding found. X shape={X.shape}, obsm={list(adata.obsm.keys())}"
    )


def save_outputs(model_name, method, y_true, y_pred, cell_ids, X_train, X_test, embedding_source):
    y_pred = np.asarray(y_pred).astype(str)

    pred_df = pd.DataFrame({
        "cell_id": cell_ids.astype(str),
        "true_label": y_true.astype(str),
        "pred_label": y_pred,
        "model": model_name,
    })
    pred_df.to_csv(f"results/predictions/{model_name}_predictions.csv", index=False)

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

    audit = {
        "foundation_model": "UCE",
        "checkpoint": "/root/dty/models/UCE/model_files/4layer_model.torch",
        "embedding_source": embedding_source,
        "train_uce_h5ad": str(TRAIN_UCE),
        "test_uce_h5ad": str(TEST_UCE),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "label_col": LABEL_COL,
        "method": method,
    }
    with open(f"results/metrics/{model_name}_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print()
    print(metrics.to_string(index=False))


print("=" * 80)
print("UCE embedding probe")
print("=" * 80)

train = sc.read_h5ad(TRAIN_UCE)
test = sc.read_h5ad(TEST_UCE)

print("Train:", train)
print("Test:", test)

X_train, train_source = get_embedding(train)
X_test, test_source = get_embedding(test)

print("Train embedding source:", train_source)
print("Test embedding source:", test_source)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

if X_train.shape[0] != train.n_obs:
    raise ValueError("X_train row count does not match train cells")
if X_test.shape[0] != test.n_obs:
    raise ValueError("X_test row count does not match test cells")

# UCE output should preserve obs labels from input.
if LABEL_COL in train.obs.columns:
    y_train = train.obs[LABEL_COL].astype(str).values
elif "celltype_l4_uce" in train.obs.columns:
    y_train = train.obs["celltype_l4_uce"].astype(str).values
else:
    raise KeyError("No celltype_l4 label found in train.obs")

if LABEL_COL in test.obs.columns:
    y_true = test.obs[LABEL_COL].astype(str).values
elif "celltype_l4_uce" in test.obs.columns:
    y_true = test.obs["celltype_l4_uce"].astype(str).values
else:
    raise KeyError("No celltype_l4 label found in test.obs")

if "joinid" in test.obs.columns:
    cell_ids = test.obs["joinid"].astype(str).values
else:
    cell_ids = test.obs_names.astype(str).values

print("Train label distribution:")
print(pd.Series(y_train).value_counts())
print("Test label distribution:")
print(pd.Series(y_true).value_counts())


# Logistic Regression probe
logreg_name = "uce_logreg_probe"
logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
)

print("\nTraining UCE + Logistic Regression...")
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)

save_outputs(
    model_name=logreg_name,
    method="real_uce_embedding_plus_logistic_regression",
    y_true=y_true,
    y_pred=pred_logreg,
    cell_ids=cell_ids,
    X_train=X_train,
    X_test=X_test,
    embedding_source=train_source,
)


# MLP probe
mlp_name = "uce_mlp_probe"
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
        verbose=True,
    ),
)

print("\nTraining UCE + MLP...")
mlp.fit(X_train, y_train_int)
pred_mlp_int = mlp.predict(X_test)
pred_mlp = le.inverse_transform(pred_mlp_int.astype(int))

save_outputs(
    model_name=mlp_name,
    method="real_uce_embedding_plus_mlp_classifier",
    y_true=y_true,
    y_pred=pred_mlp,
    cell_ids=cell_ids,
    X_train=X_train,
    X_test=X_test,
    embedding_source=train_source,
)

print("\nDONE")
