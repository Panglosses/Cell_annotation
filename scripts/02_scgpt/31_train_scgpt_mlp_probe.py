from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

LABEL_COL = "celltype_l4"
MODEL_NAME = "scgpt_mlp_probe"

Path("results/predictions").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("scGPT embedding + MLP probe")
print("=" * 80)

train = sc.read_h5ad("data/processed/train_subset.h5ad")
test = sc.read_h5ad("data/processed/test_subset.h5ad")

X_train = np.load("data/processed/train_X_scGPT.npy")
X_test = np.load("data/processed/test_X_scGPT.npy")

y_train_str = train.obs[LABEL_COL].astype(str).values
y_true = test.obs[LABEL_COL].astype(str).values

le = LabelEncoder()
y_train = le.fit_transform(y_train_str)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Labels:", list(le.classes_))
print("y_train labels:")
print(pd.Series(y_train_str).value_counts())

assert X_train.shape[0] == train.n_obs
assert X_test.shape[0] == test.n_obs

clf = make_pipeline(
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

print("Training MLP on scGPT embeddings...")
clf.fit(X_train, y_train)

print("Predicting...")
y_pred_int = clf.predict(X_test)
y_pred = le.inverse_transform(y_pred_int.astype(int))

pred_df = pd.DataFrame({
    "cell_id": test.obs_names.astype(str),
    "true_label": y_true,
    "pred_label": y_pred.astype(str),
    "model": MODEL_NAME,
})
pred_df.to_csv(f"results/predictions/{MODEL_NAME}_predictions.csv", index=False)

accuracy = accuracy_score(y_true, y_pred)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

metrics = pd.DataFrame([{
    "model": MODEL_NAME,
    "method": "real_scgpt_embedding_plus_mlp_classifier",
    "accuracy": accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "weighted_precision": weighted_precision,
    "weighted_recall": weighted_recall,
    "weighted_f1": weighted_f1,
}])
metrics.to_csv(f"results/metrics/{MODEL_NAME}_metrics.csv", index=False)

report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv(
    f"results/tables/{MODEL_NAME}_classification_report.csv"
)

labels = sorted(pd.Series(y_true).unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)
pd.DataFrame(cm, index=labels, columns=labels).to_csv(
    f"results/metrics/{MODEL_NAME}_confusion_matrix.csv"
)

audit = {
    "model": MODEL_NAME,
    "input_embedding": "data/processed/train_X_scGPT.npy and test_X_scGPT.npy",
    "embedding_source": "real scGPT checkpoint embeddings",
    "classifier": "sklearn MLPClassifier",
    "hidden_layer_sizes": [256, 128],
    "label_col": LABEL_COL,
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "label_classes": list(le.classes_),
}
with open(f"results/metrics/{MODEL_NAME}_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

print("\nMetrics:")
print(metrics.to_string(index=False))
print("DONE")
