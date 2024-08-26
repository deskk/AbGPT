# AbGPT
Official repository for AbGPT: [De Novo B-Cell Receptor Design via Generative Language Modeling](https://www.biorxiv.org/xxxxxxxxxx).

## Setup
To use AbGPT, install via pip:
```bash
pip install abgpt
```

<!-- Alternatively, you can clone this repository and install the package locally:
```bash
$ git clone git@github.com:BaratiLab/AbGPT.git 
$ pip install AbGPT
``` -->

## Command line usage

### Full sequence generation
To generate 1000 light chain sequences starting with "QLQL":
```bash
abgpt_generate --chain_type light --starting_residue QLQL --num_seqs 1000
```

To generate a BCR library with 1000 sequences for a number of starting residue (e.g., "QVQL", "EVQL", "VQLV") in the heavy chain:
```bash
abgpt_generate --chain_type heavy --starting_residue QVQL,EVQL,VQLV --num_seqs_each_starting_residue 1000
```

To generate a BCR library with 1000 sequences for a number of starting residue (e.g., "QVQL", "EVQL", "VQLV") in the light chain:
```bash
abgpt_generate --chain_type light --starting_residue EIVL,EIVM,DIQM --num_seqs_each_starting_residue 1000
```
