# Data

The STR data used in this project is based on HipSTR calls on the 1000 Genomes Project samples. The HipSTR calls are from the paper "A deep population reference panel of tandem repeat variation" ([DOI 10.1038/s41467-023-42278-3](https://www.nature.com/articles/s41467-023-42278-3)). StatSTR was used to compute statistics based on the HipSTR calls (mode, heterozygosity, and number called). Children of trios were removed before computing the statistics. The scripts used to go from the HipSTR calls to a table of STR statistics are in `data/scripts/get_info_from_hipSTR/`. The final table of STR statistics is in `data/STR_data/HipSTR_data/1000G_HipSTR_stats.tsv` and can be downloaded from Zenodo ([DOI 10.5281/zenodo.20451906](https://doi.org/10.5281/zenodo.20451906)). The following preprocessing steps include a script to download this STR statistics table.


## Downloading and processing STR data

### 1. Download STR statistics table

The STR statistics table can be downloaded from Zenodo by running `data/STR_data/HipSTR_data/download.sh`.

### 2. Download reference genome, regions to exclude, and pretraining splits

The reference genome is used to make sure STRs don't contain imperfections and downstream is used to load the sequences for the model training, interpretation, and other analyses. The regions to exclude are used to filter out STRs that overlap problematic regions of the genome. These include blacklisted regions from ENCODE, repetitive and mobile elements from RepeatMasker, and segmental duplications from the UCSC Genome Browser genomicSuperDups. The pretraining splits are used to make sure the STR datasets are split in a way that is consistent with the Caduceus/HyenaDNA/Basenji pretraining.

```
# Download reference genome
./data/reference_genome/download.sh

# Download regions to exclude
./data/invalid_regions/download.sh

# Download pretraining splits
./data/basenji_splits/download.sh
```

### 3. Filter and split STRs to create the final dataset

`data/scripts/create_STR_data_files/create_STR_data_files.py` reads the STR statistics table and produces the final per-motif-length datasets. For each locus it: filters to the requested motif length and a minimum number of called samples (`--min-num-called`, default 2000); verifies the tract is a perfect repeat in the reference genome and that the annotated boundaries don't truncate a longer tract; requires `--n-flanking` bases of flanking sequence on each side; drops loci overlapping the ENCODE blacklist, segmental duplications, or mobile elements; assigns a train/val/test split; and adds a reverse-complement entry for every locus in the same split.

Splits are assigned with `--split-mode pretraining`: each locus is placed by which Caduceus/HyenaDNA/Basenji pretraining-split interval its flanking window (STR ± `--n-flanking`) overlaps, with priority train > val > test. This guarantees a `test` locus's flanks were never seen during pretraining. Loci whose window overlaps no pretraining interval default to `test`. (A `--split-mode chromosome` option also exists: chr13 = val, chr14 = test, rest = train.)

Run the wrapper, which processes motif lengths 1 and 2 with 10 kb flanks:

```
cd data/scripts/create_STR_data_files
./run_create_STR_data_files.sh
```

Output is written to `data/STR_data/HipSTR_labeled_STRs/` as `str_len_{STR_LEN}_n_flanking_{N_FLANKING}.tsv` (e.g. `str_len_2_n_flanking_10000.tsv`), with a matching `.log` file. Each output row is one STR orientation with columns: `ID`, `chrom`, `str_start`, `str_end` (0-based half-open), `motif`, `ref_copy_number`, the carried label columns (`heterozygosity`, `mode_copy_number`, `num_called_total`), `split`, and `rev_comp`.


## Preliminary analysis and visualizations

The script `data/scripts/plot_label_dist.py` does the following preliminary analyses and visualizations of the STR datasets:

1. Plots the distribution of each label (heterozygosity and mode copy number) in the two datasets (homopolymer and dinucleotide STRs). Also plots the log(x+1) distribution of the mode copy number to visualize the distribution that will be used for model training.

2. Plots the relationship between heterozygosity and mode copy number in the two datasets. Computes the Pearson and Spearman correlations between these two labels in each dataset as well as the correlations between each label and the reference copy number.
