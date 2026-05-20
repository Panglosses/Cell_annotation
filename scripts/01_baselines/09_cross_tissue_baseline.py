from pathlib import Path
import scanpy as sc
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

INPUT_PATH = Path("data/processed/adata_preprocessed_subset.h5ad")

PRED_DIR = Path("results/predictions")
METRIC_DIR = Path("results/metrics")
TABLE_DIR = Path("results/tables")

LABEL_COL = "celltype_l4"
TISSUE_COL = "tissue"
PCA_KEY = "X_pca_custom"

TRAIN_TISSUE = "blood"
TEST_TISSUE = "synovial fluid"

PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Cross-tissue baseline")
print("=" * 80)

adata = sc.read_h5ad(INPUT_PATH)
print(adata)

print("\nTissue distribution:")
print(adata.obs[TISSUE_COL].value_counts())

adata_train = adata[adata.obs[TISSUE_COL] == TRAIN_TISSUE].copy()
adata_test = adata[adata.obs[TISSUE_COL] == TEST_TISSUE].copy()

print("\nTrain tissue:", TRAIN_TISSUE)
print(adata_train)
print(adata_train.obs[LABEL_COL].value_counts())

print("\nTest tissue:", TEST_TISSUE)
print(adata_test)
print(adata_test.obs[LABEL_COL].value_counts())

# Keep only labels present in both train and test
train_labels = set(adata_train.obs[LABEL_COL].astype(str))
test_labels = set(adata_test.obs[LABEL_COL].astype(str))
common_labels = sorted(train_labels.intersection(test_labels))

print("\nCommon labels:")
print(common_labels)

adata_train = adata_train[adata_train.obs[LABEL_COL].isin(common_labels)].copy()
adata_test = adata_test[adata_test.obs[LABEL_COL].isin(common_labels)].copy()

print("\nAfter keeping common labels:")
print("Train:", adata_train)
print("Test:", adata_test)

X_train = adata_train.obsm[PCA_KEY]
X_test = adata_test.obsm[PCA_KEY]

y_train = adata_train.obs[LABEL_COL].astype(str)
y_test = adata_test.obs[LABEL_COL].astype(str)

models = {
    "logreg_cross_tissue_blood_to_synovial": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "knn_cross_tissue_blood_to_synovial": KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        metric="euclidean",
    ),
}

all_metrics = []

for model_name, clf in models.items():
    print("\n" + "=" * 80)
    print("Training:", model_name)
    print("=" * 80)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    pred_df = pd.DataFrame({
        "cell_id": adata_test.obs_names,
        "true_label": y_test.values,
        "pred_label": y_pred,
        "model": model_name,
    })

    pred_path = PRED_DIR / f"{model_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    accuracy = accuracy_score(y_test, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "model": model_name,
        "setting": "cross_tissue_blood_to_synovial",
        "train_tissue": TRAIN_TISSUE,
        "test_tissue": TEST_TISSUE,
        "n_train": adata_train.n_obs,
        "n_test": adata_test.n_obs,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    all_metrics.append(metrics)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(METRIC_DIR / f"{model_name}_classification_report.csv")

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(METRIC_DIR / f"{model_name}_confusion_matrix.csv")

    print("Metrics:")
    print(pd.DataFrame([metrics]))

metrics_df = pd.DataFrame(all_metrics)
metrics_path = METRIC_DIR / "cross_tissue_baseline_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\nSaved cross-tissue metrics to:", metrics_path)
print("\nDone.")