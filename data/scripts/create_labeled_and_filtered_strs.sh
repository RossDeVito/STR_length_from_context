#!/bin/bash

# Script identifies all perfect 1 and 2 bp motifs STRs in the HipSTR reference,
# and saves start and end indices along with split and other info in TSV files.

N_FLANKING=10000
MAX_PERCENTILE=0.975

for STR_LEN in 1 2
do

	LOG_DIR="../STR_data/reference_based_labeled_strs"
	LOG_FILE="${LOG_DIR}/str_len_${STR_LEN}_max_cn_perc_${MAX_PERCENTILE}_n_flanking_${N_FLANKING}.log"

	# Create log directory if it doesn't exist
	mkdir -p ${LOG_DIR}

	# Create file of perfect STRs
	echo "Creating reference-based labeled sequences for:"
	echo "  STR length: ${STR_LEN}"
	echo "  Max sequence length: ${MAX_SEQ_LEN}"
	echo "  Number of flanking bases: ${N_FLANKING}"
	echo "Logging to ${LOG_FILE}"
	
	python create_ref_based_labeled_strs.py \
		--str-len ${STR_LEN} \
		--n-flanking ${N_FLANKING} \
		--copy-num-percentile-max ${MAX_PERCENTILE} | tee ${LOG_FILE}

	# Filter STRs
	echo "Filtering STRs to remove those overlapping invalid regions..."
	mkdir -p ../STR_data/filtered_labeled_strs
	python remove_invalid_region_strs.py \
		--str_file "${LOG_DIR}/str_len_${STR_LEN}_max_cn_perc_${MAX_PERCENTILE}_n_flanking_${N_FLANKING}.tsv" \
		| tee "../STR_data/filtered_labeled_strs/str_len_${STR_LEN}_max_cn_perc_${MAX_PERCENTILE}_n_flanking_${N_FLANKING}_filtered.log"
done

echo "Labeled and filtered STRs created."