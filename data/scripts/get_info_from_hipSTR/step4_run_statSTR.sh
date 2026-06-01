#!/bin/bash
# Step 4: Run statSTR per chromosome with the trio-filtered sample list,
# then concatenate all 22 outputs into one file.
#
# Computes per-locus mode (most frequent allele), heterozygosity (--het),
# and number of called alleles (--numcalled).
# statSTR uses cyvcf2.set_samples(), which silently drops sample IDs that
# aren't present in a given VCF -- so passing the full filtered list per
# chromosome is safe even if chromosomes have slightly different sample sets.
#
# statSTR writes output to <prefix>.tab.

set -euo pipefail

VCF_DIR=/tscc/projects/ps-gymreklab/helia/ensembl/ensemble_in/hipstr
OUT_BASE=/tscc/projects/ps-gymreklab/rdevito/str_len_pred/STR_length_from_context/data/STR_data/HipSTR_data
SAMPLES_FILE=${OUT_BASE}/filtered_samples.txt
PER_CHROM_DIR=${OUT_BASE}/per_chrom_statstr

mkdir -p ${PER_CHROM_DIR}

# Per-chromosome statSTR
for CHR_NUM in $(seq 1 22); do
	VCF=${VCF_DIR}/ensemble_input_chr${CHR_NUM}_hipstr_corrected.vcf.gz
	PREFIX=${PER_CHROM_DIR}/chr${CHR_NUM}
	echo "[$(date '+%H:%M:%S')] statSTR on chr${CHR_NUM}..."
	statSTR \
		--vcf ${VCF} \
		--vcftype hipstr \
		--samples ${SAMPLES_FILE} \
		--mode \
		--het \
		--numcalled \
		--out ${PREFIX}
done

# Concatenate. Take header from chr1, then append data rows from chr1..22.
COMBINED=${OUT_BASE}/all_chroms_statstr.tab
FIRST=${PER_CHROM_DIR}/chr1.tab

head -n 1 ${FIRST} > ${COMBINED}
for CHR_NUM in $(seq 1 22); do
	tail -n +2 ${PER_CHROM_DIR}/chr${CHR_NUM}.tab >> ${COMBINED}
done

echo
echo "[$(date '+%H:%M:%S')] Done."
echo "Combined output: ${COMBINED}"
echo "Total data rows: $(($(wc -l < ${COMBINED}) - 1))"