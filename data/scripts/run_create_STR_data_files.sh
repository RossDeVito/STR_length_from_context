#!/bin/bash

# Script to create data files used for models and analyses from EnsembleTR
# data.

N_FLANKING=10000

for STR_LEN in 1 2
do

	LOG_DIR="../STR_data/EnsembleTR_labeled_STRs"
	LOG_FILE="${LOG_DIR}/str_len_${STR_LEN}_n_flanking_${N_FLANKING}.log"

	# Create log directory if it doesn't exist
	mkdir -p ${LOG_DIR}

	echo "Creating EnsembleTR-based labeled sequences for:"
	echo "  STR length: ${STR_LEN}"
	echo "  Min number of flanking bases: ${N_FLANKING}"
	echo "Logging to ${LOG_FILE}"
	
	python create_STR_data_files.py \
		--str-len ${STR_LEN} \
		--n-flanking ${N_FLANKING} \
		--copy-number-stats mean,median,mode \
		--min-num-called 2000 \
		--repeat-info ../STR_data/EnsembleTR/ensembletr_raw_data/repeat_info.tsv \
		--afreq-het ../STR_data/EnsembleTR/ensembletr_raw_data/afreq_het.tsv \
		--ref-genome ../reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa \
		--output-dir ${LOG_DIR} | tee ${LOG_FILE}

done