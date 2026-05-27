#!/bin/bash
# Step 2: Download 1000 Genomes metadata for trio removal and pop counts.
#
# The 3202-sample pedigree+population file from the NYGC high-coverage
# release contains FamilyID, SampleID, FatherID, MotherID, Sex,
# Population, Superpopulation in one whitespace-delimited table. That's
# everything we need for steps 3 & 4.

set -euo pipefail

OUT_DIR=/tscc/projects/ps-gymreklab/rdevito/str_len_pred/STR_length_from_context/data/1000_genomes
mkdir -p ${OUT_DIR}

PED_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt"

cd ${OUT_DIR}

echo "[$(date '+%H:%M:%S')] Downloading 3202-sample pedigree+population file..."
wget -nv -O 20130606_g1k_3202_samples_ped_population.txt "${PED_URL}"

echo
echo "First 3 lines:"
head -3 ${OUT_DIR}/20130606_g1k_3202_samples_ped_population.txt
echo
echo "Row count (incl. header): $(wc -l < ${OUT_DIR}/20130606_g1k_3202_samples_ped_population.txt)"