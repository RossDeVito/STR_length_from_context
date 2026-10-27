#!/bin/bash

# Create per-bp G4Hunter-score bigWig annotations at several window sizes,
# covering the STR +/- N_FLANKING regions used by the labeled HipSTR STR
# datasets for motif lengths STR_LENS. One bigWig per window size; window
# sizes must be odd (centered window, equal bases on each side).
#
# 25 is G4Hunter's canonical window (Bedrat et al. 2016) and the only one of
# these that is a G4 caller -- its 1.2 / 1.5 thresholds are calibrated for
# that width. 9 and 5 exist to match the GC_content family's window sizes,
# so the two annotations can be compared at the SAME smoothing rather than
# confounding annotation type with window width; at those widths the score
# is local G/C run content, not G4 propensity, and the thresholds do not
# apply. Note the score converges on GC content as the window narrows
# (Spearman vs matched-window GC: 0.45 at 25bp, 0.57 at 9bp, 0.61 at 5bp),
# so the narrow tracks are partly re-measuring GC.

WINDOW_SIZES=(25 9 5)
N_FLANKING=10000
STR_LENS="3 4"
STR_DATA_DIR="../STR_data/HipSTR_labeled_STRs"
REF_GENOME="../reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa"
OUT_DIR="../annotations/G4Hunter"

mkdir -p ${OUT_DIR}

for WINDOW_SIZE in "${WINDOW_SIZES[@]}"
do

	LOG_FILE="${OUT_DIR}/G4Hunter_${WINDOW_SIZE}bp.log"

	echo "Creating G4Hunter annotation for:"
	echo "  Window size: ${WINDOW_SIZE}"
	echo "  n_flanking:  ${N_FLANKING}"
	echo "  STR lens:    ${STR_LENS}"
	echo "Logging to ${LOG_FILE}"

	python create_G4Hunter_anno_file.py \
		--window-size ${WINDOW_SIZE} \
		--str-data-dir ${STR_DATA_DIR} \
		--str-lens ${STR_LENS} \
		--n-flanking ${N_FLANKING} \
		--ref-genome ${REF_GENOME} \
		--output-dir ${OUT_DIR} | tee ${LOG_FILE}

done
