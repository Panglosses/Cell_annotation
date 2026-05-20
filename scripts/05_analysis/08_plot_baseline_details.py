from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

models = {
    "logistic_regression": {
        "report": METRICS_DIR / "logreg_classification_report.csv",
        "confusion": METRICS_DIR / "logreg_confusion_matrix.csv",
    },
    "knn": {
        "report": METRICS_DIR / "knn_classification_report.csv",
        "confusion": METRICS_DIR / "knn_confusion_matrix.csv",
    },
}

all_per_type = []

print("=" * 80)
print("Plotting baseline details")
print("=" * 80)

for model_name, files in models.items():
    report_path = files["report"]
    confusion_path = files["confusion"]

    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")
    if not confusion_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {confusion_path}")

    print(f"\nProcessing model: {model_name}")

    report = pd.read_csv(report_path, index_col=0)

    # Keep only real cell types, remove aggregate rows
    exclude_rows = {"accuracy", "macro avg", "weighted avg"}
    per_type = report.loc[[idx for idx in report.index if idx not in exclude_rows]].copy()
    per_type["model"] = model_name
    per_type["celltype_l4"] = per_type.index

    all_per_type.append(per_type.reset_index(drop=True))

    per_type_sorted = per_type.sort_values("f1-score", ascending=True)

    # Per-cell-type F1 plot
    plt.figure(figsize=(8, 5))
    plt.barh(per_type_sorted["celltype_l4"], per_type_sorted["f1-score"])
    plt.xlabel("F1 score")
    plt.ylabel("Cell type")
    plt.title(f"Per-cell-type F1: {model_name}")
    plt.xlim(0, 1)
    plt.tight_layout()
    out_path = FIG_DIR / f"per_celltype_f1_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

    # Confusion matrix plot
    cm = pd.read_csv(confusion_path, index_col=0)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm.values, aspect="auto")
    plt.colorbar(label="Number of cells")
    plt.xticks(range(len(cm.columns)), cm.columns, rotation=90)
    plt.yticks(range(len(cm.index)), cm.index)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion matrix: {model_name}")

    # Add numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm.values[i, j]
            if value > 0:
                plt.text(j, i, str(value), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    out_path = FIG_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

all_per_type_df = pd.concat(all_per_type, ignore_index=True)
out_table = TABLE_DIR / "per_celltype_baseline_metrics.csv"
all_per_type_df.to_csv(out_table, index=False)

print("\nSaved combined per-cell-type metrics to:", out_table)

print("\nWorst cell types by model:")
for model_name in models:
    tmp = all_per_type_df[all_per_type_df["model"] == model_name]
    tmp = tmp.sort_values("f1-score", ascending=True)
    print("\n", model_name)
    print(tmp[["celltype_l4", "precision", "recall", "f1-score", "support"]].head(5))

print("\nDone.")