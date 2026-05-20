from pathlib import Path
import scanpy as sc
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

TRAIN_PATH = Path("data/processed/train_subset.h5ad")
TEST_PATH = Path("data/processed/test_subset.h5ad")

PRED_PATH = Path("results/predictions/knn_predictions.csv")
METRIC_PATH = Path("results/metrics/knn_metrics.csv")
REPORT_PATH = Path("results/metrics/knn_classification_report.csv")
CONFUSION_PATH = Path("results/metrics/knn_confusion_matrix.csv")

LABEL_COL = "celltype_l4"
PCA_KEY = "X_pca_custom"

N_NEIGHBORS = 15

print("=" * 80)
print("Loading train/test data")
print("=" * 80)

adata_train = sc.read_h5ad(TRAIN_PATH)
adata_test = sc.read_h5ad(TEST_PATH)

print("Train:", adata_train)
print("Test:", adata_test)

if PCA_KEY not in adata_train.obsm:
    raise ValueError(f"{PCA_KEY} not found in train obsm")

if PCA_KEY not in adata_test.obsm:
    raise ValueError(f"{PCA_KEY} not found in test obsm")

X_train = adata_train.obsm[PCA_KEY]
X_test = adata_test.obsm[PCA_KEY]

y_train = adata_train.obs[LABEL_COL].astype(str)
y_test = adata_test.obs[LABEL_COL].astype(str)

print("\nTraining kNN...")
clf = KNeighborsClassifier(
    n_neighbors=N_NEIGHBORS,
    weights="distance",
    metric="euclidean",
)

clf.fit(X_train, y_train)

print("\nPredicting...")
y_pred = clf.predict(X_test)

PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)

pred_df = pd.DataFrame({
    "cell_id": adata_test.obs_names,
    "true_label": y_test.values,
    "pred_label": y_pred,
    "model": "knn",
})

pred_df.to_csv(PRED_PATH, index=False)

accuracy = accuracy_score(y_test, y_pred)

macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)

weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)

metrics_df = pd.DataFrame([{
    "model": "knn",
    "accuracy": accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "weighted_precision": weighted_precision,
    "weighted_recall": weighted_recall,
    "weighted_f1": weighted_f1,
}])

metrics_df.to_csv(METRIC_PATH, index=False)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORT_PATH)

labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(CONFUSION_PATH)

print("\nMetrics:")
print(metrics_df)

print("\nSaved predictions to:", PRED_PATH)
print("Saved metrics to:", METRIC_PATH)
print("Saved classification report to:", REPORT_PATH)
print("Saved confusion matrix to:", CONFUSION_PATH)

print("\nDone.")