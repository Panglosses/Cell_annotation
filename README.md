# JIA NK/ILC cell annotation benchmark

This repository contains the code and output files for my mini-project on automatic NK/ILC cell-state annotation in juvenile idiopathic arthritis (JIA). I compared conventional PCA-based classifiers with frozen embeddings from three single-cell foundation models: scGPT, Geneformer V1-10M and UCE.

The main question was whether pretrained single-cell embeddings improve fine-grained `celltype_l4` annotation, or whether a dataset-specific baseline is still stronger for this JIA NK/ILC task.

## What is included

```text
requirements/      Environment files used for the run
scripts/           Numbered scripts for preprocessing, models and analysis
results/           Metrics, tables, summaries and figures
README.md          This file
```

The repository does **not** include the raw `.h5ad` file or pretrained model checkpoints, because those files are too large and/or externally distributed.

## Project structure

```text
scripts/
  00_preprocessing/   data check, downsampling, QC, normalization, PCA, split
  01_baselines/       PCA + Logistic Regression, PCA + kNN, cross-tissue baseline
  02_scgpt/           scGPT embedding extraction and downstream probes
  03_geneformer/      Geneformer input preparation, tokenization, embeddings, probes
  04_uce/             UCE input preparation and downstream probes
  05_analysis/        metric summaries, report figures, supplementary and post hoc checks

results/
  metrics/            model-level metrics, confusion matrices, audit files
  tables/             classification reports, QC/split tables, supplementary table
  figures/            report figures and supplementary figures
  summaries/          short text summaries for scGPT, Geneformer and UCE

```

## Data and labels

The input dataset is an h5ad file containing NK and innate lymphoid cell (ILC) single-cell RNA-seq profiles from JIA samples, including blood and synovial-fluid cells.

Labels used in this project:

- `celltype_l3`: broad NK vs ILC label
- `celltype_l4`: fine labels including C1, C2, C3, CD56bright NK, CD56dim NK, Cycling NK, ILC2, ILC3 and Naive-like ILC

Dataset used for the final benchmark:

- 16,836 cells x 22,144 genes
- 13,468 training cells and 3,368 test cells
- stratified 80/20 split using `celltype_l4`
- uneven `celltype_l4` class sizes, so weighted F1 was used as the main ranking metric and macro F1 was also reported

I used `celltype_l4` as the main task because it tests fine NK/ILC subtype or state annotation, not just broad NK versus ILC separation. This matters for JIA because synovial-fluid cells come from the local inflammatory compartment and may carry tissue- or activation-related transcriptional differences.

Expected raw data location for reruns:

```text
data/raw/innate_lymphocytes_jia.h5ad
```

## Models compared

| Model | Feature representation | Classifier | Label setting |
|---|---|---|---|
| Logistic Regression baseline | PCA of normalized expression | LR | `celltype_l4` |
| kNN baseline | PCA of normalized expression | kNN | `celltype_l4` |
| scGPT | 512-dim frozen embeddings | LR, MLP | `celltype_l4`; LR also `celltype_l3` |
| Geneformer V1-10M | 256-dim frozen embeddings | LR, MLP | `celltype_l4` |
| UCE | 1280-dim frozen embeddings | LR, MLP | `celltype_l4`; LR also `celltype_l3` |
| Cross-tissue baseline | PCA of normalized expression | LR, kNN | blood to synovial fluid |

All foundation-model embeddings were used frozen. No foundation model was fine-tuned.

## Method choices

The PCA + Logistic Regression baseline was included as a strong supervised baseline, not as a weak comparator. It directly learns the study-specific `celltype_l4` label space from normalized expression.

The foundation models were tested as frozen embedding generators. Logistic Regression was used as a linear probe, and MLP was added to test whether nonlinear boundaries helped recover more information from the embeddings.

A PCA + MLP check was added after the main report figures were fixed. I kept it separate from the main benchmark table because it was a post hoc fairness check, not part of the original model set.

Geneformer V1-10M was the feasible Geneformer checkpoint used here. Its result should not be read as a statement about every Geneformer model size or a fine-tuned Geneformer workflow.

## Main results

Full metrics are in:

```text
results/metrics/consolidated_main_metrics.csv
```

| Model | Setting | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Logistic Regression baseline | `celltype_l4`, random split | 0.9068 | 0.8952 | 0.9065 |
| kNN baseline | `celltype_l4`, random split | 0.8890 | 0.8751 | 0.8875 |
| scGPT LR | `celltype_l4`, random split | 0.7782 | 0.7566 | 0.7759 |
| scGPT MLP | `celltype_l4`, random split | 0.8584 | 0.8381 | 0.8583 |
| Geneformer LR | `celltype_l4`, random split | 0.7007 | 0.6511 | 0.7018 |
| Geneformer MLP | `celltype_l4`, random split | 0.7046 | 0.6594 | 0.7010 |
| UCE LR | `celltype_l4`, random split | 0.8382 | 0.8031 | 0.8376 |
| UCE MLP | `celltype_l4`, random split | 0.8566 | 0.8445 | 0.8558 |
| scGPT LR | `celltype_l3`, random split | 0.9561 | 0.8998 | 0.9582 |
| UCE LR | `celltype_l3`, random split | 0.9733 | 0.9364 | 0.9741 |
| Logistic Regression baseline | blood to synovial fluid | 0.7504 | 0.6706 | 0.7535 |
| kNN baseline | blood to synovial fluid | 0.3965 | 0.4446 | 0.4376 |

The random-split `celltype_l4`, random-split `celltype_l3` and blood-to-synovial-fluid rows are different evaluation settings, so they should not be treated as one single ranking.

## Key takeaways

- In the main table, the best fine-grained `celltype_l4` result was PCA + Logistic Regression (weighted F1 = 0.9065). A post hoc PCA + MLP check reached weighted F1 = 0.9141.
- Among frozen foundation-model embeddings, scGPT + MLP and UCE + MLP were close to each other (weighted F1 around 0.856), but neither exceeded the PCA baseline.
- Geneformer V1-10M was lower on this task (weighted F1 around 0.70).
- scGPT and UCE performed much better on broad `celltype_l3` labels, which suggests that frozen embeddings captured NK/ILC lineage structure better than fine subtype or state boundaries.
- For this dataset and frozen-embedding setup, dataset-specific PCA classifiers remained the strongest practical methods. scGPT and UCE were still useful as representation models, especially for broader lineage-level structure.

## Interpretation notes

The gap between PCA + Logistic Regression and the best foundation-model probes is meaningful because the models used the same random train/test split for `celltype_l4`. It does not mean the foundation models failed biologically. Their stronger `celltype_l3` results suggest that they captured broad NK/ILC identity, while the fine labels were harder.

The post hoc PCA + MLP result is in `results/metrics/pca_mlp_same_split_metrics.csv`. Paired bootstrap tables with the added PCA checks are in `results/tables/paired_bootstrap_weighted_f1_ci_posthoc.csv` and `results/tables/paired_bootstrap_weighted_f1_differences_posthoc.csv`.

The main fine-label confusions involved C1, C2, C3, CD56bright NK and Naive-like ILC. These numbers are summarized in `results/tables/supplementary_table2_confusion_f1_summary.csv`.

An internal C1/C2/C3 marker-direction check is provided in `results/tables/C1_C2_C3_internal_marker_stability_summary.csv`. It checks train/test consistency inside this dataset only; it is not external biological validation.

Post hoc repeated-split checks were added for PCA, scGPT and Geneformer. The summary files are `results/metrics/repeated_split_pca_summary.csv`, `results/metrics/repeated_split_scgpt_summary.csv` and `results/metrics/repeated_split_geneformer_summary.csv`; by-seed files are in the same folder.

Blood-to-synovial-fluid transfer was included as a secondary domain-shift test. The original PCA baseline transfer results are in the main metrics table, and additional scGPT/Geneformer transfer summaries are in `results/metrics/scgpt_blood_to_synovial_summary.csv` and `results/metrics/geneformer_blood_to_synovial_summary.csv`. These rows should not be compared directly with the random-split rows as one single ranking.

## Report figures

Main figures generated for the report:

- `results/figures/figure1_celltype_l4_weighted_f1.png`
- `results/figures/figure2_per_class_f1_heatmap.png`
- `results/figures/supplementary_table2_confusion_f1_summary.png`

The report documents are submitted separately rather than stored in this GitHub-ready code folder.

## How to rerun

Create the environment:

```bash
bash requirements/setup_singlecell_env.sh
```

or:

```bash
conda env create -f requirements/conda_env.yml
```

Then run the numbered scripts from the repository root. The order is:

```bash
python scripts/00_preprocessing/01_check_dataset.py
python scripts/00_preprocessing/02_make_subset.py
python scripts/00_preprocessing/03_preprocess_subset.py
python scripts/00_preprocessing/04_split_data.py

python scripts/01_baselines/05_train_logreg_baseline.py
python scripts/01_baselines/06_train_knn_baseline.py
python scripts/01_baselines/09_cross_tissue_baseline.py

python scripts/02_scgpt/30_run_real_scgpt.py --model-dir /path/to/scGPT_human
python scripts/02_scgpt/31_train_scgpt_mlp_probe.py
python scripts/02_scgpt/32_train_scgpt_l3_probe.py

python scripts/03_geneformer/60_prepare_geneformer_h5ad.py
python scripts/03_geneformer/61_tokenize_geneformer_v1.py
python scripts/03_geneformer/62_extract_geneformer_embeddings_v1.py
python scripts/03_geneformer/63_train_geneformer_probe.py

python scripts/04_uce/70_prepare_uce_h5ad.py
# Run UCE embedding generation with the UCE command-line workflow here.
python scripts/04_uce/73_train_uce_probe.py
python scripts/04_uce/74_train_uce_l3_probe.py

python scripts/05_analysis/07_summarize_baselines.py
python scripts/05_analysis/08_plot_baseline_details.py
python scripts/05_analysis/10_merge_all_baseline_metrics.py
python scripts/05_analysis/11_make_report_figures.py
python scripts/05_analysis/12_make_supplementary_confusion_summary.py
python scripts/05_analysis/16_posthoc_pca_mlp_bootstrap_marker_checks.py
python scripts/05_analysis/17_repeated_split_pca.py
python scripts/05_analysis/18_scgpt_repeated_and_cross_tissue.py
python scripts/05_analysis/19_geneformer_repeated_and_cross_tissue.py
```

Tokenization note: `61_tokenize_geneformer_v1.py` is for Geneformer. It uses Geneformer's `TranscriptomeTokenizer`. UCE has its own reference/model files and is handled separately in `scripts/04_uce/`.

## External files needed for a full rerun

The following files must be supplied locally:

- raw data: `data/raw/innate_lymphocytes_jia.h5ad`
- scGPT human checkpoint directory: `vocab.json`, `args.json`, `best_model.pt`
- Geneformer V1-10M model and V1 dictionary files
- UCE 4-layer checkpoint and reference files: `4layer_model.torch`, `all_tokens.torch`, `species_chrom.csv`, `species_offsets.pkl`, `protein_embeddings/`

Some scripts still contain the model paths used during my run. These should be edited before rerunning on a different machine.

## Notes

- The main train/test comparison uses one stratified random split from the same study.
- Post hoc repeated-split checks were completed for PCA, scGPT and Geneformer, but not UCE.
- Blood-to-synovial transfer was tested for PCA baselines, scGPT and Geneformer. UCE was included in the main random-split benchmark but not in the post hoc transfer check because reusable split-level UCE embedding objects were not retained.
- Geneformer `celltype_l3` was not run, so the broad-label comparison is limited to scGPT and UCE.
- The foundation models were used as frozen embedding generators.

