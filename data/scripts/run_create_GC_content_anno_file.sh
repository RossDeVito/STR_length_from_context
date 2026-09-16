#!/bin/bash

# Create local-GC-content bigWig annotations at several window sizes,
# covering the STR +/- N_FLANKING regions used by the labeled HipSTR STR
# datasets (str_len_1..6). One bigWig per window size; window sizes must be
# odd (centered window, equal bases on each side).

WINDOW_SIZES=(101 501 2001 10001)
N_FLANKING=10000
STR_DATA_DIR="../STR_data/HipSTR_labeled_STRs"
REF_GENOME="../reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa"
OUT_DIR="../annotations/GC_content"

mkdir -p ${OUT_DIR}

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"
do

	LOG_FILE="${OUT_DIR}/GC_content_${WINDOW_SIZE}bp.log"

	echo "Creating GC content annotation for:"
	echo "  Window size: ${WINDOW_SIZE}"
	echo "  n_flanking:  ${N_FLANKING}"
	echo "Logging to ${LOG_FILE}"

	python create_GC_content_anno_file.py \
		--window-size ${WINDOW_SIZE} \
		--str-data-dir ${STR_DATA_DIR} \
		--n-flanking ${N_FLANKING} \
		--ref-genome ${REF_GENOME} \
		--output-dir ${OUT_DIR} | tee ${LOG_FILE}

done
