#!/bin/bash
# Step 1: Extract the union of samples across all HipSTR per-chromosome VCFs.
#
# Saves per-chromosome sample lists (for debugging / per-chrom intersection
# later if needed) plus the overall union. Also reports whether all chroms
# share the same sample set.

set -euo pipefail

BCFTOOLS=/tscc/projects/ps-gymreklab/rdevito/str_len_pred/tools/bcftools-1.23.1/install/bin/bcftools
VCF_DIR=/tscc/projects/ps-gymreklab/helia/ensembl/ensemble_in/hipstr
OUT_DIR=/tscc/projects/ps-gymreklab/rdevito/str_len_pred/STR_length_from_context/data/STR_data/HipSTR_data

PER_CHROM_DIR=${OUT_DIR}/per_chrom_samples
mkdir -p ${PER_CHROM_DIR}

UNION_FILE=${OUT_DIR}/all_samples_union.txt
INTER_FILE=${OUT_DIR}/all_samples_intersection.txt

# Extract per-chromosome sample lists
for CHR_NUM in $(seq 1 22); do
	VCF=${VCF_DIR}/ensemble_input_chr${CHR_NUM}_hipstr_corrected.vcf.gz
	OUT=${PER_CHROM_DIR}/chr${CHR_NUM}_samples.txt
	echo "[$(date '+%H:%M:%S')] Extracting samples from chr${CHR_NUM}..."
	${BCFTOOLS} query -l ${VCF} | sort > ${OUT}
done

# Union
cat ${PER_CHROM_DIR}/chr*_samples.txt | sort -u > ${UNION_FILE}

# Intersection (sample present in ALL 22 chromosomes)
# Start from chr1, repeatedly comm -12 with each subsequent chromosome.
TMP=$(mktemp)
cp ${PER_CHROM_DIR}/chr1_samples.txt ${TMP}
for CHR_NUM in $(seq 2 22); do
	comm -12 ${TMP} ${PER_CHROM_DIR}/chr${CHR_NUM}_samples.txt > ${TMP}.next
	mv ${TMP}.next ${TMP}
done
mv ${TMP} ${INTER_FILE}

N_UNION=$(wc -l < ${UNION_FILE})
N_INTER=$(wc -l < ${INTER_FILE})

echo
echo "Union size:        ${N_UNION}"
echo "Intersection size: ${N_INTER}"

if [ "${N_UNION}" -ne "${N_INTER}" ]; then
	echo "WARNING: chromosomes do not share an identical sample set."
	echo "  Difference: $((N_UNION - N_INTER)) sample(s)."
	echo "  Per-chrom sample counts:"
	wc -l ${PER_CHROM_DIR}/chr*_samples.txt
else
	echo "All 22 chromosomes share the same sample set."
fi