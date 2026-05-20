from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_DIRS = [RESULTS / "tables"]
FIG_DIRS = [RESULTS / "figures"]


MODELS = {
    "PCA + LR": {
        "confusion": RESULTS / "metrics" / "logreg_confusion_matrix.csv",
        "report": RESULTS / "metrics" / "logreg_classification_report.csv",
    },
    "scGPT MLP": {
        "confusion": RESULTS / "metrics" / "scgpt_mlp_probe_confusion_matrix.csv",
        "report": RESULTS / "tables" / "scgpt_mlp_probe_classification_report.csv",
    },
    "Geneformer MLP": {
        "confusion": RESULTS / "metrics" / "geneformer_mlp_probe_confusion_matrix.csv",
        "report": RESULTS / "tables" / "geneformer_mlp_probe_classification_report.csv",
    },
    "UCE MLP": {
        "confusion": RESULTS / "metrics" / "uce_mlp_probe_confusion_matrix.csv",
        "report": RESULTS / "tables" / "uce_mlp_probe_classification_report.csv",
    },
}


def clean_label(label: str) -> str:
    return label.replace("Na茂ve-like ILC", "Naive-like ILC").replace("Naïve-like ILC", "Naive-like ILC")


def read_confusion(path: Path) -> tuple[list[str], dict[str, dict[str, int]], dict[str, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    labels = [clean_label(x) for x in rows[0][1:]]
    matrix: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for row in rows[1:]:
        true = clean_label(row[0])
        counts = [int(x) for x in row[1:]]
        matrix[true] = dict(zip(labels, counts))
        totals[true] = sum(counts)
    return labels, matrix, totals


def read_f1(path: Path, label: str) -> float:
    df = pd.read_csv(path, index_col=0)
    df.index = [clean_label(str(x)) for x in df.index]
    return float(df.loc[label, "f1-score"])


def fmt_count_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total * 100:.1f}%)"


def build_table() -> pd.DataFrame:
    rows = []
    for model, paths in MODELS.items():
        _, matrix, totals = read_confusion(paths["confusion"])
        c1_to_bright = fmt_count_rate(matrix["C1"]["CD56bright NK"], totals["C1"])
        c1_c2_count = matrix["C1"]["C2"] + matrix["C2"]["C1"]
        c1_c2_total = totals["C1"] + totals["C2"]
        c2_c3_count = matrix["C2"]["C3"] + matrix["C3"]["C2"]
        c2_c3_total = totals["C2"] + totals["C3"]
        naive_to_c1 = matrix["Naive-like ILC"]["C1"]
        naive_total = totals["Naive-like ILC"]
        rows.append(
            {
                "model": model,
                "C1_to_CD56bright_NK": c1_to_bright,
                "C1_C2_bidirectional": fmt_count_rate(c1_c2_count, c1_c2_total),
                "C2_C3_bidirectional": fmt_count_rate(c2_c3_count, c2_c3_total),
                "Naive_like_ILC_to_C1": fmt_count_rate(naive_to_c1, naive_total),
                "CD56dim_NK_F1": f"{read_f1(paths['report'], 'CD56dim NK'):.3f}",
                "ILC2_F1": f"{read_f1(paths['report'], 'ILC2'):.3f}",
            }
        )
    return pd.DataFrame(rows)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_table(df: pd.DataFrame) -> Image.Image:
    headers = [
        "Model",
        "C1 ->\nCD56bright NK",
        "C1 <-> C2",
        "C2 <-> C3",
        "Naive-like\nILC -> C1",
        "CD56dim\nNK F1",
        "ILC2 F1",
    ]
    cols = list(df.columns)
    widths = [260, 300, 280, 280, 290, 180, 160]
    row_h = 120
    header_h = 135
    margin = 70
    title_h = 95
    w = margin * 2 + sum(widths)
    h = margin * 2 + title_h + header_h + row_h * len(df) + 85
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_head, f_body, f_note = font(34, True), font(22, True), font(22), font(19)
    draw.text((margin, 35), "Supplementary Table 2. Verified confusion and per-class F1 summary", font=f_title, fill="#111111")
    y = margin + title_h
    x = margin
    for j, head in enumerate(headers):
        draw.rectangle((x, y, x + widths[j], y + header_h), fill="#F2F4F7", outline="#D6DCE5", width=2)
        lines = head.split("\n")
        for k, line in enumerate(lines):
            draw.text((x + 14, y + 25 + k * 28), line, font=f_head, fill="#111111")
        x += widths[j]
    y += header_h
    for i, row in df.iterrows():
        x = margin
        fill = "#FFFFFF" if i % 2 == 0 else "#FAFBFC"
        for j, col in enumerate(cols):
            draw.rectangle((x, y, x + widths[j], y + row_h), fill=fill, outline="#E0E5EC", width=2)
            draw.text((x + 14, y + 42), str(row[col]), font=f_body, fill="#111111")
            x += widths[j]
        y += row_h
    note = "Confusion entries are count/true-class support (%). F1 values are from existing classification reports."
    draw.text((margin, h - 62), note, font=f_note, fill="#555555")
    return img


def save_outputs(df: pd.DataFrame) -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "supplementary_table2_confusion_f1_summary.csv", index=False)
    img = render_table(df)
    for fig_dir in FIG_DIRS:
        fig_dir.mkdir(parents=True, exist_ok=True)
        png = fig_dir / "supplementary_table2_confusion_f1_summary.png"
        pdf = fig_dir / "supplementary_table2_confusion_f1_summary.pdf"
        img.save(png, dpi=(300, 300))
        img.convert("RGB").save(pdf, "PDF", resolution=300)
        print(png)
        print(pdf)


def main() -> None:
    df = build_table()
    save_outputs(df)
    print("CSV outputs:")
    for out_dir in OUT_DIRS:
        print(out_dir / "supplementary_table2_confusion_f1_summary.csv")


if __name__ == "__main__":
    main()
