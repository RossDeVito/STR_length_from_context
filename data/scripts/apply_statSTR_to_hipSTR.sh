#!/bin/bash

# Apply StatSTR to existing HipSTR calls for STRs in the 1000 Genomes Project

OUT_DIR="../STR_data/HipSTR_call_stats"
HIPSTR_CALL_DIR="/tscc/projects/ps-gymreklab/helia/ensembl/1000G_calls/hipstr/chrs"

# Create output directory if it doesn't exist
mkdir -p ${OUT_DIR}

# Install StatSTR if not already installed
pip install --upgrade pip
pip install trtools

# Loop through chromosomes 1-22 and apply StatSTR to each HipSTR VCF
for CHR_NUM in {1..22}
do
	INPUT_VCF="${HIPSTR_CALL_DIR}/hipstr_corrected_chr${CHR_NUM}.vcf.gz"
	OUTPUT_FILE="${OUT_DIR}/chr${CHR_NUM}_hipstr_statstr"

	echo "Processing chromosome ${CHR_NUM}..."
	echo "  Input VCF: ${INPUT_VCF}"
	echo "  Output file prefix: ${OUTPUT_FILE}"

	statSTR \
		--vcf ${INPUT_VCF} \
		--output-prefix ${OUTPUT_FILE} \
		--vcftype hipstr \
		--precision 5 \
		--het \
		--mode
done