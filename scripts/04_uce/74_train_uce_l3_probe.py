from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

TRAIN_UCE = Path("data/uce/train_outputtrain_uce_input_uce_adata.h5ad")
TEST_UCE = Path("data/uce/test_outputtest_uce_input_uce_adata.h5ad")

LABEL_COL = "celltype_l3"
MODEL_NAME = "uce_l3_logreg_probe"

Path("results/predictions").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

def get_x(adata):
    if "X_uce" in adata.obsm:
        X = adata.obsm["X_uce"]
    else:
        raise KeyError(f"X_uce not found. obsm keys: {list(adata.obsm.keys())}")
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)

train = sc.read_h5ad(TRAIN_UCE)
test = sc.read_h5ad(TEST_UCE)

X_train = get_x(train)
X_test = get_x(test)

if LABEL_COL in train.obs.columns:
    y_train = train.obs[LABEL_COL].astype(str).values
elif "celltype_l3_uce" in train.obs.columns:
    y_train = train.obs["celltype_l3_uce"].astype(str).values
else:
    raise KeyError("No celltype_l3 label found in train.obs")

if LABEL_COL in test.obs.columns:
    y_true = test.obs[LABEL_COL].astype(str).values
elif "celltype_l3_uce" in test.obs.columns:
    y_true = test.obs["celltype_l3_uce"].astype(str).values
else:
    raise KeyError("No celltype_l3 label found in test.obs")

cell_ids = test.obs["joinid"].astype(str).values if "joinid" in test.obs.columns else test.obs_names.astype(str).values

print("=" * 80)
print("UCE celltype_l3 Logistic Regression probe")
print("=" * 80)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Train labels:")
print(pd.Series(y_train).value_counts())
print("Test labels:")
print(pd.Series(y_true).value_counts())

clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pd.DataFrame({
    "cell_id": cell_ids,
    "true_label": y_true,
    "pred_label": y_pred.astype(str),
    "model": MODEL_NAME,
}).to_csv(f"results/predictions/{MODEL_NAME}_predictions.csv", index=False)

acc = accuracy_score(y_true, y_pred)
mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

metrics = pd.DataFrame([{
    "model": MODEL_NAME,
    "label_col": LABEL_COL,
    "method": "real_uce_embedding_plus_logistic_regression",
    "accuracy": acc,
    "macro_precision": mp,
    "macro_recall": mr,
    "macro_f1": mf,
    "weighted_precision": wp,
    "weighted_recall": wr,
    "weighted_f1": wf,
}])
metrics.to_csv(f"results/metrics/{MODEL_NAME}_metrics.csv", index=False)

pd.DataFrame(
    classification_report(y_true, y_pred, output_dict=True, zero_division=0)
).transpose().to_csv(f"results/tables/{MODEL_NAME}_classification_report.csv")

labels = sorted(pd.Series(y_true).unique())
pd.DataFrame(
    confusion_matrix(y_true, y_pred, labels=labels),
    index=labels,
    columns=labels,
).to_csv(f"results/metrics/{MODEL_NAME}_confusion_matrix.csv")

print()
print(metrics.to_string(index=False))
print("DONE")
