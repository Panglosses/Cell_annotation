from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

LABEL_COL = "celltype_l3"
MODEL_NAME = "scgpt_l3_logreg_probe"

Path("results/predictions").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

train = sc.read_h5ad("data/processed/train_subset.h5ad")
test = sc.read_h5ad("data/processed/test_subset.h5ad")

X_train = np.load("data/processed/train_X_scGPT.npy")
X_test = np.load("data/processed/test_X_scGPT.npy")

y_train = train.obs[LABEL_COL].astype(str).values
y_true = test.obs[LABEL_COL].astype(str).values

print("Label distribution train:")
print(pd.Series(y_train).value_counts())
print("Label distribution test:")
print(pd.Series(y_true).value_counts())

clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pd.DataFrame({
    "cell_id": test.obs_names.astype(str),
    "true_label": y_true,
    "pred_label": y_pred.astype(str),
    "model": MODEL_NAME,
}).to_csv(f"results/predictions/{MODEL_NAME}_predictions.csv", index=False)

accuracy = accuracy_score(y_true, y_pred)
mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

metrics = pd.DataFrame([{
    "model": MODEL_NAME,
    "label_col": LABEL_COL,
    "method": "real_scgpt_embedding_plus_logistic_regression",
    "accuracy": accuracy,
    "macro_precision": mp,
    "macro_recall": mr,
    "macro_f1": mf,
    "weighted_precision": wp,
    "weighted_recall": wr,
    "weighted_f1": wf,
}])
metrics.to_csv(f"results/metrics/{MODEL_NAME}_metrics.csv", index=False)

pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).transpose().to_csv(
    f"results/tables/{MODEL_NAME}_classification_report.csv"
)

labels = sorted(pd.Series(y_true).unique())
pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels).to_csv(
    f"results/metrics/{MODEL_NAME}_confusion_matrix.csv"
)

print()
print(metrics.to_string(index=False))
