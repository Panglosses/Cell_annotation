#!/usr/bin/env python
"""Bootstrap uncertainty estimates from saved prediction CSV files.

This script does not retrain any model. It resamples the existing test-cell
predictions and estimates uncertainty for accuracy, macro F1 and weighted F1.
For models evaluated on the same labelled cells, it also computes paired
bootstrap differences in weighted F1.

Outputs:
  results/tables/bootstrap_prediction_uncertainty.csv
  results/tables/bootstrap_paired_weighted_f1_differences.csv
  results/summaries/bootstrap_uncertainty_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
SUMMARIES = RESULTS / "summaries"

N_BOOT = 5000
SEED = 20260520


PREDICTION_FILES = {
    "scGPT + Logistic Regression": RESULTS / "scgpt_predictions.csv",
    "scGPT + MLP": RESULTS / "scgpt_mlp_probe_predictions.csv",
    "Geneformer V1-10M + Logistic Regression": RESULTS / "geneformer_logreg_probe_predictions.csv",
    "Geneformer V1-10M + MLP": RESULTS / "geneformer_mlp_probe_predictions.csv",
    "UCE + Logistic Regression": RESULTS / "uce_logreg_probe_predictions.csv",
    "UCE + MLP": RESULTS / "uce_mlp_probe_predictions.csv",
    "scGPT + Logistic Regression (celltype_l3)": RESULTS / "scgpt_l3_logreg_probe_predictions.csv",
    "UCE + Logistic Regression (celltype_l3)": RESULTS / "uce_l3_logreg_probe_predictions.csv",
}


PAIRED_COMPARISONS = [
    ("scGPT + MLP", "UCE + MLP"),
    ("scGPT + Logistic Regression", "UCE + Logistic Regression"),
    ("scGPT + MLP", "scGPT + Logistic Regression"),
    ("UCE + MLP", "UCE + Logistic Regression"),
    ("Geneformer V1-10M + MLP", "Geneformer V1-10M + Logistic Regression"),
    ("UCE + MLP", "Geneformer V1-10M + MLP"),
    ("scGPT + MLP", "Geneformer V1-10M + MLP"),
    ("UCE + Logistic Regression (celltype_l3)", "scGPT + Logistic Regression (celltype_l3)"),
]


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = sorted(set(y_true) | set(y_pred))
    supports = []
    f1_values = []
    for label in labels:
        true_is_label = y_true == label
        pred_is_label = y_pred == label
        tp = int(np.sum(true_is_label & pred_is_label))
        fp = int(np.sum(~true_is_label & pred_is_label))
        fn = int(np.sum(true_is_label & ~pred_is_label))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        supports.append(int(np.sum(true_is_label)))
        f1_values.append(f1)

    supports_arr = np.array(supports, dtype=float)
    f1_arr = np.array(f1_values, dtype=float)
    total_support = supports_arr.sum()
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1_arr)),
        "weighted_f1": float(np.sum(f1_arr * supports_arr) / total_support)
        if total_support
        else 0.0,
    }


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_single_model(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    point = metric_values(y_true, y_pred)

    n = len(df)
    boot = {"accuracy": [], "macro_f1": [], "weighted_f1": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        vals = metric_values(y_true[idx], y_pred[idx])
        for key, value in vals.items():
            boot[key].append(value)

    ci = {key: percentile_ci(np.array(values)) for key, values in boot.items()}
    return point, ci


def bootstrap_paired_difference(
    left: pd.DataFrame, right: pd.DataFrame, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    merged = left[["cell_id", "true_label", "pred_label"]].merge(
        right[["cell_id", "true_label", "pred_label"]],
        on="cell_id",
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        raise ValueError("No shared cell_id values for paired comparison.")
    if not (merged["true_label_left"] == merged["true_label_right"]).all():
        raise ValueError("Shared cell_id values have mismatched true labels.")

    y_true = merged["true_label_left"].to_numpy()
    pred_left = merged["pred_label_left"].to_numpy()
    pred_right = merged["pred_label_right"].to_numpy()
    point = (
        metric_values(y_true, pred_left)["weighted_f1"]
        - metric_values(y_true, pred_right)["weighted_f1"]
    )

    n = len(merged)
    diffs = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (
            metric_values(y_true[idx], pred_left[idx])["weighted_f1"]
            - metric_values(y_true[idx], pred_right[idx])["weighted_f1"]
        )

    lo, hi = percentile_ci(diffs)
    p_two_sided = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_two_sided = min(float(p_two_sided), 1.0)
    return float(point), lo, hi, p_two_sided


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    SUMMARIES.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    predictions: dict[str, pd.DataFrame] = {}
    rows = []

    for model, path in PREDICTION_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"cell_id", "true_label", "pred_label"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        predictions[model] = df

        point, ci = bootstrap_single_model(df, rng)
        label_level = "celltype_l3" if "celltype_l3" in model else "celltype_l4"
        for metric, value in point.items():
            lo, hi = ci[metric]
            rows.append(
                {
                    "model": model,
                    "label_level": label_level,
                    "n_cells": len(df),
                    "metric": metric,
                    "point_estimate": value,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_bootstrap": N_BOOT,
                }
            )

    uncertainty = pd.DataFrame(rows)
    uncertainty.to_csv(TABLES / "bootstrap_prediction_uncertainty.csv", index=False)

    diff_rows = []
    for left_name, right_name in PAIRED_COMPARISONS:
        if left_name not in predictions or right_name not in predictions:
            continue
        point, lo, hi, p = bootstrap_paired_difference(
            predictions[left_name], predictions[right_name], rng
        )
        label_level = "celltype_l3" if "celltype_l3" in left_name else "celltype_l4"
        diff_rows.append(
            {
                "comparison": f"{left_name} minus {right_name}",
                "label_level": label_level,
                "metric": "weighted_f1_difference",
                "point_difference": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "bootstrap_p_two_sided_directional": p,
                "n_bootstrap": N_BOOT,
            }
        )

    paired = pd.DataFrame(diff_rows)
    paired.to_csv(TABLES / "bootstrap_paired_weighted_f1_differences.csv", index=False)

    lines = [
        "Bootstrap uncertainty analysis",
        f"Bootstrap resamples: {N_BOOT}",
        f"Random seed: {SEED}",
        "",
        "Single-model weighted F1 intervals:",
    ]
    wf1 = uncertainty[uncertainty["metric"] == "weighted_f1"].copy()
    for _, row in wf1.iterrows():
        lines.append(
            f"- {row['model']}: {row['point_estimate']:.4f} "
            f"(95% bootstrap CI {row['ci95_low']:.4f}-{row['ci95_high']:.4f})"
        )

    lines.extend(["", "Paired weighted F1 differences:"])
    for _, row in paired.iterrows():
        lines.append(
            f"- {row['comparison']}: {row['point_difference']:.4f} "
            f"(95% bootstrap CI {row['ci95_low']:.4f}-{row['ci95_high']:.4f}; "
            f"directional bootstrap p={row['bootstrap_p_two_sided_directional']:.4g})"
        )

    lines.extend(
        [
            "",
            "Note: PCA baseline prediction files were not present in the repository, "
            "so paired bootstrap comparisons against PCA models were not computed here.",
        ]
    )
    (SUMMARIES / "bootstrap_uncertainty_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
