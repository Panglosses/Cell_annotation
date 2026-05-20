from pathlib import Path
from geneformer import TranscriptomeTokenizer

BASE = Path("/root/dty/models/geneformer/Geneformer/geneformer/gene_dictionaries_30m")

gene_median_file = BASE / "gene_median_dictionary_gc30M.pkl"
token_dictionary_file = BASE / "token_dictionary_gc30M.pkl"
gene_mapping_file = BASE / "ensembl_mapping_dict_gc30M.pkl"

for p in [gene_median_file, token_dictionary_file, gene_mapping_file]:
    if not p.exists():
        raise FileNotFoundError(p)

for split in ["train", "test"]:
    input_dir = Path(f"data/geneformer/{split}_input")
    output_dir = Path(f"data/geneformer/{split}_tokenized")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Tokenizing", split)
    print("=" * 80)

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "joinid": "joinid",
            "celltype_l4_geneformer": "celltype_l4",
            "celltype_l3_geneformer": "celltype_l3",
        },
        nproc=4,
        model_input_size=2048,
        special_token=False,
        model_version="V1",
        gene_median_file=gene_median_file,
        token_dictionary_file=token_dictionary_file,
        gene_mapping_file=gene_mapping_file,
    )

    tk.tokenize_data(
        data_directory=str(input_dir),
        output_directory=str(output_dir),
        output_prefix=f"jia_{split}_geneformer_v1",
        file_format="h5ad",
    )

    print("Output files:")
    for p in output_dir.iterdir():
        print(p)
