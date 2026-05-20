from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT_DIRS = [RESULTS / "figures"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def save_png_pdf(img: Image.Image, stem: str) -> list[Path]:
    paths: list[Path] = []
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{stem}.png"
        pdf = out_dir / f"{stem}.pdf"
        img.save(png, dpi=(300, 300))
        img.convert("RGB").save(pdf, "PDF", resolution=300)
        paths.extend([png, pdf])
    return paths


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def read_metric(path: Path) -> float:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    return float(row["weighted_f1"])


def figure1() -> Image.Image:
    metrics = RESULTS / "metrics"
    data = [
        ("PCA + LR", read_metric(metrics / "logreg_metrics.csv"), "#4C78A8"),
        ("PCA + kNN", read_metric(metrics / "knn_metrics.csv"), "#4C78A8"),
        ("scGPT LR", read_metric(metrics / "scgpt_metrics.csv"), "#59A14F"),
        ("scGPT MLP", read_metric(metrics / "scgpt_mlp_probe_metrics.csv"), "#59A14F"),
        ("Geneformer LR", read_metric(metrics / "geneformer_logreg_probe_metrics.csv"), "#F28E2B"),
        ("Geneformer MLP", read_metric(metrics / "geneformer_mlp_probe_metrics.csv"), "#F28E2B"),
        ("UCE LR", read_metric(metrics / "uce_logreg_probe_metrics.csv"), "#B07AA1"),
        ("UCE MLP", read_metric(metrics / "uce_mlp_probe_metrics.csv"), "#B07AA1"),
    ]

    width, height = 2400, 1500
    margin_l, margin_r, margin_t, margin_b = 210, 80, 150, 300
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_axis, f_tick, f_label = font(46, True), font(34, True), font(28), font(26)

    draw.text((margin_l, 45), "Celltype_l4 weighted F1 across models", font=f_title, fill="#111111")
    draw.line((margin_l, margin_t + plot_h, width - margin_r, margin_t + plot_h), fill="#222222", width=3)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="#222222", width=3)

    y_max = 1.0
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = margin_t + plot_h - tick / y_max * plot_h
        draw.line((margin_l - 10, y, width - margin_r, y), fill="#E6E6E6", width=2)
        draw.text((85, y - 18), f"{tick:.1f}", font=f_tick, fill="#333333")

    bar_gap = 36
    bar_w = (plot_w - bar_gap * (len(data) + 1)) / len(data)
    for i, (label, value, color) in enumerate(data):
        x0 = margin_l + bar_gap + i * (bar_w + bar_gap)
        x1 = x0 + bar_w
        y1 = margin_t + plot_h
        y0 = y1 - value / y_max * plot_h
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=color)
        text_center(draw, ((x0 + x1) / 2, y0 - 34), f"{value:.3f}", f_label, "#222222")
        parts = label.split(" ")
        if len(parts) > 2:
            tick_label = " ".join(parts[:-1]) + "\n" + parts[-1]
        else:
            tick_label = label
        for j, line in enumerate(tick_label.split("\n")):
            text_center(draw, ((x0 + x1) / 2, y1 + 38 + j * 32), line, f_tick, "#222222")

    draw.text((margin_l + plot_w / 2 - 155, height - 78), "Model and probe", font=f_axis, fill="#111111")
    # Rotated y-axis label.
    ylab = Image.new("RGBA", (360, 60), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylab)
    yd.text((0, 0), "Weighted F1", font=f_axis, fill="#111111")
    ylab = ylab.rotate(90, expand=True)
    img.paste(ylab, (28, margin_t + plot_h // 2 - ylab.height // 2), ylab)

    draw.text((margin_l, height - 130), "All bars use random-split celltype_l4 evaluation; celltype_l3 and cross-tissue rows are excluded.", font=f_label, fill="#555555")
    return img


def clean_label(label: str) -> str:
    return label.replace("Na茂ve-like ILC", "Naive-like ILC").replace("Naïve-like ILC", "Naive-like ILC")


def read_f1_report(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, index_col=0)
    return {clean_label(str(idx)): float(row["f1-score"]) for idx, row in df.iterrows() if str(idx) not in {"accuracy", "macro avg", "weighted avg"}}


def figure2() -> Image.Image:
    tables = RESULTS / "tables"
    reports = {
        "PCA + LR": RESULTS / "metrics" / "logreg_classification_report.csv",
        "scGPT MLP": tables / "scgpt_mlp_probe_classification_report.csv",
        "Geneformer MLP": tables / "geneformer_mlp_probe_classification_report.csv",
        "UCE MLP": tables / "uce_mlp_probe_classification_report.csv",
    }
    values = {model: read_f1_report(path) for model, path in reports.items()}
    labels = ["C1", "C2", "C3", "CD56bright NK", "CD56dim NK", "Cycling NK", "ILC2", "ILC3", "Naive-like ILC"]
    models = list(reports)

    width, height = 2600, 1250
    margin_l, margin_t = 330, 190
    cell_w, cell_h = 225, 150
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_axis, f_tick, f_cell = font(46, True), font(30, True), font(25), font(29, True)
    draw.text((margin_l, 45), "Per-class F1 for selected celltype_l4 models", font=f_title, fill="#111111")

    def color(v: float) -> tuple[int, int, int]:
        # Blue scale from light to dark.
        lo = (239, 246, 255)
        hi = (37, 99, 160)
        t = max(0.0, min(1.0, (v - 0.35) / 0.65))
        return tuple(int(lo[i] + t * (hi[i] - lo[i])) for i in range(3))

    for j, label in enumerate(labels):
        x = margin_l + j * cell_w
        for k, line in enumerate(label.replace("CD56bright", "CD56bright\n").replace("CD56dim", "CD56dim\n").replace("Naive-like", "Naive-like\n").split("\n")):
            text_center(draw, (x + cell_w / 2, margin_t - 65 + k * 29), line, f_tick, "#222222")

    for i, model in enumerate(models):
        y = margin_t + i * cell_h
        draw.text((55, y + cell_h / 2 - 18), model, font=f_axis, fill="#222222")
        for j, label in enumerate(labels):
            x = margin_l + j * cell_w
            v = values[model][label]
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=color(v), outline="#FFFFFF", width=5)
            fill = "white" if v >= 0.72 else "#111111"
            text_center(draw, (x + cell_w / 2, y + cell_h / 2), f"{v:.2f}", f_cell, fill)

    draw.rectangle((margin_l, margin_t, margin_l + len(labels) * cell_w, margin_t + len(models) * cell_h), outline="#333333", width=3)

    # Legend.
    legend_x, legend_y = margin_l, height - 210
    draw.text((legend_x, legend_y - 55), "F1 score", font=f_axis, fill="#111111")
    for n in range(120):
        v = 0.35 + n / 119 * 0.65
        draw.rectangle((legend_x + n * 5, legend_y, legend_x + n * 5 + 5, legend_y + 36), fill=color(v))
    draw.text((legend_x, legend_y + 48), "0.35", font=f_tick, fill="#333333")
    draw.text((legend_x + 560, legend_y + 48), "1.00", font=f_tick, fill="#333333")
    draw.text((margin_l, height - 65), "Values are f1-score entries from existing celltype_l4 classification reports.", font=f_tick, fill="#555555")
    return img


def main() -> None:
    outputs = []
    outputs += save_png_pdf(figure1(), "figure1_celltype_l4_weighted_f1")
    outputs += save_png_pdf(figure2(), "figure2_per_class_f1_heatmap")
    print("Generated files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
