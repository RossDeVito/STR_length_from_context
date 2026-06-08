#!/bin/bash

# Create data files using splits ALIGNED TO CADUCEUS PRETRAINING.
#
# Each locus is assigned train/val/test by which pretraining-split interval
# its flanking window (STR +/- N_FLANKING) overlaps, priority train>val>test.
# Loci whose window overlaps no pretraining interval default to 'test'
# (their flanks were never seen during pretraining).

N_FLANKING=10000
HIPSTR_STATS="../../STR_data/HipSTR_data/1000G_HipSTR_stats.tsv"
PRETRAIN_BED="../../basenji_splits/sequences_human.bed"

for STR_LEN in 1 2
do

	OUT_DIR="../../STR_data/HipSTR_labeled_STRs"
	LOG_FILE="${OUT_DIR}/str_len_${STR_LEN}_n_flanking_${N_FLANKING}.log"

	# Create output/log directory if it doesn't exist
	mkdir -p ${OUT_DIR}

	echo "Creating pretraining-split-aligned labeled sequences for:"
	echo "  STR length: ${STR_LEN}"
	echo "  Min number of flanking bases: ${N_FLANKING}"
	echo "Logging to ${LOG_FILE}"

	python create_STR_data_files.py \
		--str-len ${STR_LEN} \
		--n-flanking ${N_FLANKING} \
		--min-num-called 2000 \
		--hipstr-stats ${HIPSTR_STATS} \
		--ref-genome ../../reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa \
		--mobile-elements-bed ../../invalid_regions/mobile_elements.bed.gz \
		--seg-dups-bed ../../invalid_regions/seg_dups.bed.gz \
		--blacklist-bed ../../invalid_regions/blacklist.bed.gz \
		--split-mode pretraining \
		--pretraining-split-bed ${PRETRAIN_BED} \
		--output-dir ${OUT_DIR} | tee ${LOG_FILE}

done