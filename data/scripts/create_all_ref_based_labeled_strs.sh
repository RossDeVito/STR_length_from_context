#!/bin/bash

# Script identifies all perfect 1 and 2 bp motifs STRs in the HipSTR reference,
# and saves start and end indices along with split and other info in TSV files.

N_FLANKING=250000
MAX_PERCENTILE=0.975

for STR_LEN in 1 2
do

	LOG_DIR="../STR_data/reference_based_labeled_strs"
	LOG_FILE="${LOG_DIR}/str_len_${STR_LEN}_max_cn_perc_${MAX_PERCENTILE}_n_flanking_${N_FLANKING}.log"

	# Create log directory if it doesn't exist
	mkdir -p ${LOG_DIR}

	echo "Creating reference-based labeled sequences for:"
	echo "  STR length: ${STR_LEN}"
	echo "  Max sequence length: ${MAX_SEQ_LEN}"
	echo "  Number of flanking bases: ${N_FLANKING}"
	echo "Logging to ${LOG_FILE}"
	
	python create_ref_based_labeled_strs.py \
		--str-len ${STR_LEN} \
		--n-flanking ${N_FLANKING} \
		--copy-num-percentile-max ${MAX_PERCENTILE} | tee ${LOG_FILE}
done

echo "All reference-based labeled sequences created."