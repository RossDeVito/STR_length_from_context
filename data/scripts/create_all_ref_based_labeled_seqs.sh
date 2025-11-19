#!/bin/bash

# Script creates all reference-based labeled sequences for STRs
# with motif lengths 1 and 2 for sequence based models with maximum
# sequence lengths of 1024, 32,768, 160,000, 450,000, and 1,000,000 bp.
# Maximum input sequence length will be at most half of the maximum
# for a model (as paper says this gives best results).

MAX_PERCENTILE=0.975
DIST_FROM_EDGE=250000

for STR_LEN in 1 2
do
	for MAX_SEQ_LEN in 1024 32768 160000 450000 1000000
	do
		N_FLANKING=$((MAX_SEQ_LEN / 4))

		LOG_DIR="../STR_data/reference_based_labeled_seqs"
		LOG_FILE="${LOG_DIR}/log_create_ref_seqs_strlen${STR_LEN}_maxseqlen${MAX_SEQ_LEN}_maxperc${MAX_PERCENTILE}_edge${DIST_FROM_EDGE}.txt"

		echo "Creating reference-based labeled sequences for:"
		echo "  STR length: ${STR_LEN}"
		echo "  Max sequence length: ${MAX_SEQ_LEN}"
		echo "  Number of flanking bases: ${N_FLANKING}"
		echo "  Distance from edge: ${DIST_FROM_EDGE}"
		echo "Logging to ${LOG_FILE}"
		
		python create_ref_based_labeled_seqs.py \
			--str-len ${STR_LEN} \
			--n-flanking ${N_FLANKING} \
			--copy-num-percentile-max ${MAX_PERCENTILE} \
			--distance-from-edge ${DIST_FROM_EDGE} | tee ${LOG_FILE}
	done
done

echo "All reference-based labeled sequences created."