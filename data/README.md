# Data

The STR data used in this project is based on HipSTR calls on the 1000 Genomes Project samples. The HipSTR calls are from the paper "A deep population reference panel of tandem repeat variation" ([DOI 10.1038/s41467-023-42278-3](https://www.nature.com/articles/s41467-023-42278-3)). StatSTR was used to compute statistics based on the HipSTR calls (mode, heterozygosity, and number called). Children of trios were removed before computing the statistics. The scripts used to go from the HipSTR calls to a table of STR statistics are in `data/scripts/get_info_from_hipSTR/`. The final table of STR statistics is in `data/STR_data/HipSTR_data/1000G_HipSTR_stats.tsv` and can be downloaded from Zenodo ([DOI 10.5281/zenodo.20451906](https://doi.org/10.5281/zenodo.20451906)). The following preprocessing steps include a script to download this STR statistics table.


## Downloading and processing STR data

### 1. Download STR statistics table

The STR statistics table can be downloaded from Zenodo by running `data/STR_data/HipSTR_data/download.sh`. The count of STRs in the table for each motif length can be found by running `data/scripts/create_STR_data_files/count_motif_lengths.py`. The output is:

```
motif_len       count
		1      611020
		2      237475
		3       67898
		4      195946
		5       83214
		6       60607
```

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

`data/scripts/create_STR_data_files/create_STR_data_files.py` reads the STR statistics table and produces the final per-motif-length datasets for lengths 1-6. For each locus it: filters to the requested motif length and a minimum number of called samples (`--min-num-called`, default 2000); verifies the tract is a perfect repeat in the reference genome and that the annotated boundaries don't truncate a longer tract; requires `--n-flanking` bases of flanking sequence on each side; drops loci overlapping the ENCODE blacklist, segmental duplications, or mobile elements; assigns a train/val/test split; and adds a reverse-complement entry for every locus in the same split.

Splits are assigned with `--split-mode pretraining`: each locus is placed by which Caduceus/HyenaDNA/Basenji pretraining-split interval its flanking window (STR ± `--n-flanking`) overlaps, with priority train > val > test. This guarantees a `test` locus's flanks were never seen during pretraining. Loci whose window overlaps no pretraining interval default to `test`. (A `--split-mode chromosome` option also exists: chr13 = val, chr14 = test, rest = train.)

Run the wrapper, which processes motif lengths 1, 2, 3, 4, 5, and 6 with 10 kb flanks:

```
cd data/scripts/create_STR_data_files
./run_create_STR_data_files.sh
```

Output is written to `data/STR_data/HipSTR_labeled_STRs/` as `str_len_{STR_LEN}_n_flanking_{N_FLANKING}.tsv` (e.g. `str_len_2_n_flanking_10000.tsv`), with a matching `.log` file. Each output row is one STR orientation with columns: `ID`, `chrom`, `str_start`, `str_end` (0-based half-open), `motif`, `ref_copy_number`, the carried label columns (`heterozygosity`, `mode_copy_number`, `num_called_total`), `split`, and `rev_comp`.


## Preliminary analysis and visualizations of STR data

The script `data/scripts/plot_label_dist.py` does the following preliminary analyses and visualizations of the STR datasets:

1. Plots the distribution of each label (heterozygosity and mode copy number) in the datasets (by STR length). Also plots the log(x+1) distribution of the mode copy number to visualize the distribution that will be used for model training.

2. Plots the relationship between heterozygosity and mode copy number in the two datasets. Computes the Pearson and Spearman correlations between these two labels in each dataset as well as the correlations between each label and the reference copy number.


## Annotations

Annotations are used to associate STRs, and the model's per-position importance scores, with genomic attributes. Each one lives in its own directory under `data/annotations/` and is registered in `scripts/annotation_analysis/annotation_config.yaml`. That config is the registry the analysis scripts read, through `load_annotation()` in `scripts/annotation_analysis/annotation_utils.py`; it records each track's file type (`bigwig` or `bed`), whether its values are `continuous` or `discrete`, its path, and the source DOI (see the comment block at the top of the config for the full field schema). Adding an annotation therefore means two things: getting the file into `data/annotations/`, and adding an entry to that config. `scripts/annotation_analysis/inspect_annotation.py` is a quick way to sanity-check a new entry.

No annotation data file is checked into git (`.gitignore` keeps only the directories and the `download.sh` scripts) -- each is between roughly 1 GB and several GB. The steps below reproduce all of them.

Two of the annotations are downloaded from their publishers and are genome-wide. The other two are computed here from the reference genome, and are **not** genome-wide: they cover only the STR +/- `n_flanking` (10 kb) regions of the motif length 3 and 4 datasets, which are the lengths the downstream analyses use. Merged across both datasets that is 857 Mbp over 34,283 intervals (53,177 loci). Positions outside those regions simply have no data, and read back as NaN through pyBigWig exactly as genuinely missing bigWig data does -- so a NaN means "not covered", not "no signal". Both computed tracks need the reference genome and the STR tables to exist already, so run them after "Downloading and processing STR data" above.

### 1. Download phyloP conservation scores

```
./data/annotations/phyloP/download.sh
```

Downloads `hg38.phyloP447way.bw`, the UCSC phyloP track from the 447-way mammalian alignment ([DOI 10.1038/s41586-023-06798-8](https://doi.org/10.1038/s41586-023-06798-8)). Registered as `phyloP`. Positive values are conserved (evolving slower than neutral) and negative values accelerated (faster than neutral); the absolute value is a -log p-value against a neutral-evolution null.

### 2. Download full-stack ChromHMM chromatin states

```
./data/annotations/full_stack_ChromHMM/download.sh
```

Downloads the hg38 full-stack ChromHMM 100-state segmentation ([DOI 10.1186/s13059-021-02572-z](https://doi.org/10.1186/s13059-021-02572-z)) as two files: `hg38_genome_100_segments.bed.gz` (the segments) and `state_annotations_processed.csv` (the state metadata). This is the one discrete annotation, and the only one needing a join: the bed's segment names are `{state number}_{mnemonic}`, but that leading number is the model's internal state ordering and does **not** match the csv's `state` column, so only the mnemonic is a usable join key (`load_chromhmm()` in `annotation_utils.py` handles this). It is registered twice against the same pair of files, differing only in which joined column is the value: `ChromHMM_state` for the 100-state mnemonic, and `ChromHMM_group` for the 16 coarser state groups.

### 3. Compute local GC content

```
cd data/scripts
./run_create_GC_content_anno_file.sh
```

Builds one per-bp bigWig per window size in the wrapper's `WINDOW_SIZES`, each holding, at every covered position, the GC fraction of a centered window of that many bases (window sizes must be odd, so the window is symmetric). N bases are excluded from both the numerator and the denominator. Windows are pulled from the FASTA rather than from the covered regions, so a position at the edge of a covered region still gets a full window; only true chromosome ends truncate it. GC fraction is reverse-complement invariant, so one forward-strand track serves both the forward and `rev_comp` rows of the STR tables.

Each window size `n` is written to `data/annotations/GC_content/GC_content_<n>bp.bw` and must be registered in the config as `GC_content_<n>bp` to be visible to the analysis scripts. The wrapper builds 5, 9, and 25 bp, which are the three registered there.

### 4. Compute G4Hunter G-quadruplex scores

```
cd data/scripts
./run_create_G4Hunter_anno_file.sh
```

Builds one per-bp bigWig per window size (25, 9, and 5 bp by default) of G4Hunter scores ([DOI 10.1093/nar/gkw006](https://doi.org/10.1093/nar/gkw006)), which measure G-quadruplex-forming propensity from sequence alone. Each base scores `+min(k, 4)` inside a run of `k` consecutive Gs, `-min(k, 4)` inside a run of `k` consecutive Cs, and 0 for A/T/N; the value at a position is the mean of those per-base scores over a centered window. Run lengths are measured on the reference sequence rather than within the window, so a run crossing a window edge keeps its true (capped) length. As with GC content, N bases are left out of the window's denominator, which departs from the reference G4Hunter implementation (it divides by the full window width) in order to keep a window containing an assembly gap comparable to one without instead of diluting its score toward zero.

25 bp is G4Hunter's canonical window and the only one of the three that is a G4 caller -- the paper treats `score >= 1.2` as G4-forming, `1.5` for higher specificity, and those thresholds are calibrated for that width. 9 bp and 5 bp match the GC content family's window sizes so the two annotations can be compared at the *same* smoothing, instead of confounding annotation type with window width. At those widths the value is local G/C run content rather than G4 propensity, and the thresholds do not apply.

The raw G4Hunter score is **not** reverse-complement invariant: it negates, since one strand's G runs are the other's C runs (`score_revcomp == -score_fwd` at the mirrored position). Rather than leave that sign for every consumer to handle, **these tracks store the max of the two strands**, which is exactly `abs()` of the raw forward-strand score -- at a fixed coordinate the minus-strand window score is the negation of the plus-strand one, so `max(s, -s) == abs(s)`. Stored values are therefore non-negative, strand-agnostic G4 propensity: the same on a locus's forward and `rev_comp` rows, directly comparable to orientation-invariant labels, and needing no sign handling anywhere downstream.

The abs is applied to the window mean, not per base, which is the point of the score: a window whose G runs and C runs offset each other stays near zero, correctly, because it is not G4-forming on either strand. (Abs-per-base would instead measure G+C run-richness, a much more GC-content-like quantity.) The signed score remains available in code as `g4hunter_signed_score_track()` for any future analysis that orients G4s against something meaningful -- replication fork direction or transcription direction; the reference strand is not that.

The score converges on plain GC content as the window narrows. Spearman against a matched-window GC track is 0.45 at 25 bp, 0.57 at 9 bp, and 0.61 at 5 bp (0.96 at 1 bp, where it reduces to the G/C base indicator), so the narrow tracks are partly re-measuring GC and exist mainly as same-window controls.

At 25 bp the score is most its own quantity, and there it is clearly not a repackaging of GC: at fixed GC it spans nearly its full range (a 25 bp window of `GCGC...` scores 0.04 while `GGGG...` scores 4.0, both at GC = 1.0; telomeric `GGGTTAGGG...` scores 1.48 at GC = 0.52), only ~9% of top-decile-GC positions clear the 1.2 threshold, and no position below GC 0.4 does. High GC is close to necessary for a G4 and nowhere near sufficient. What the score actually tracks is run structure -- its Spearman with G/C-triplet-run density (+0.57) exceeds its correlation with GC content. Any positive G4Hunter result is still worth checking against GC at the same window -- partial correlation, or stratifying by GC bin -- to establish that it reflects run structure rather than base composition.

Each window size `n` is written to `data/annotations/G4Hunter/G4Hunter_<n>bp.bw` and registered in the config as `G4Hunter_<n>bp`.