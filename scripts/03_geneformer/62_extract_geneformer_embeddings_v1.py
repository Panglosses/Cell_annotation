from pathlib import Path
import pandas as pd
from geneformer import EmbExtractor

MODEL_DIR = Path("/root/dty/models/geneformer/Geneformer/Geneformer-V1-10M")
TOKEN_DICT = Path("/root/dty/models/geneformer/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")

OUTDIR = Path("data/geneformer/embeddings")
OUTDIR.mkdir(parents=True, exist_ok=True)

if not MODEL_DIR.exists():
    raise FileNotFoundError(MODEL_DIR)
if not TOKEN_DICT.exists():
    raise FileNotFoundError(TOKEN_DICT)

print("=" * 80)
print("Geneformer V1-10M embedding extraction")
print("=" * 80)
print("MODEL_DIR:", MODEL_DIR)
print("TOKEN_DICT:", TOKEN_DICT)

embex = EmbExtractor(
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    filter_data=None,
    max_ncells=20000,          # important: avoid default downsampling to 1000
    emb_layer=-1,
    emb_label=["joinid", "celltype_l4", "celltype_l3"],
    labels_to_plot=None,
    forward_batch_size=16,     # conservative for P100 16GB
    nproc=4,
    summary_stat=None,
    save_tdigest=False,
    model_version="V1",
    token_dictionary_file=str(TOKEN_DICT),
)

for split in ["train", "test"]:
    input_dataset = Path(f"data/geneformer/{split}_tokenized/jia_{split}_geneformer_v1.dataset")
    output_prefix = f"jia_{split}_geneformer_v1"

    if not input_dataset.exists():
        raise FileNotFoundError(input_dataset)

    print("\n" + "=" * 80)
    print("Extracting", split)
    print("input_dataset:", input_dataset)
    print("=" * 80)

    embex.extract_embs(
        model_directory=str(MODEL_DIR),
        input_data_file=str(input_dataset),
        output_directory=str(OUTDIR),
        output_prefix=output_prefix,
        output_torch_embs=True,
    )

print("\nGenerated files:")
for p in sorted(OUTDIR.iterdir()):
    print(p, p.stat().st_size)

print("DONE")
