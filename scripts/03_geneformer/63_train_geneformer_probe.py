from pathlib import Path
import json
import numpy as np
import pandas as pd

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

LABEL_COL = "celltype_l4"

Path("results/predictions").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path("data/geneformer/embeddings/jia_train_geneformer_v1.csv")
TEST_CSV = Path("data/geneformer/embeddings/jia_test_geneformer_v1.csv")

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

embed_cols = [str(i) for i in range(256)]
missing = [c for c in embed_cols if c not in train.columns or c not in test.columns]
if missing:
    raise ValueError(f"Missing embedding columns: {missing[:10]}")

X_train = train[embed_cols].values.astype(np.float32)
X_test = test[embed_cols].values.astype(np.float32)

y_train = train[LABEL_COL].astype(str).values
y_true = test[LABEL_COL].astype(str).values

print("=" * 80)
print("Geneformer V1-10M embedding probe")
print("=" * 80)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Train label distribution:")
print(pd.Series(y_train).value_counts())
print("Test label distribution:")
print(pd.Series(y_true).value_counts())


def save_outputs(model_name, method, y_pred):
    y_pred = np.asarray(y_pred).astype(str)

    pred_df = pd.DataFrame({
        "cell_id": test["joinid"].astype(str).values,
        "true_label": y_true,
        "pred_label": y_pred,
        "model": model_name,
    })
    pred_df.to_csv(f"results/predictions/{model_name}_predictions.csv", index=False)

    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

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
        "model": model_name,
        "foundation_model": "Geneformer",
        "checkpoint": "Geneformer-V1-10M",
        "embedding_source": "EmbExtractor cell embeddings",
        "embedding_dim": 256,
        "train_csv": str(TRAIN_CSV),
        "test_csv": str(TEST_CSV),
        "label_col": LABEL_COL,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "method": method,
    }
    with open(f"results/metrics/{model_name}_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print()
    print(metrics.to_string(index=False))


# 1. Logistic Regression probe
logreg_name = "geneformer_logreg_probe"
logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
)
print("\nTraining Geneformer + Logistic Regression...")
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
save_outputs(
    logreg_name,
    "real_geneformer_v1_embedding_plus_logistic_regression",
    pred_logreg,
)


# 2. MLP probe
mlp_name = "geneformer_mlp_probe"

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

print("\nTraining Geneformer + MLP...")
mlp.fit(X_train, y_train_int)
pred_mlp_int = mlp.predict(X_test)
pred_mlp = le.inverse_transform(pred_mlp_int.astype(int))
save_outputs(
    mlp_name,
    "real_geneformer_v1_embedding_plus_mlp_classifier",
    pred_mlp,
)

print("\nDONE")
