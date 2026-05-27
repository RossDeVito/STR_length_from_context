#!/bin/bash
# Step 3: Remove trio children from the HipSTR sample union and compute
# population / superpopulation counts on the kept samples.
#
# Wraps filter_trios_and_count.py. Outputs go to data/STR_data/HipSTR_data.

set -euo pipefail

BASE=/tscc/projects/ps-gymreklab/rdevito/str_len_pred/STR_length_from_context

SAMPLES_FILE=${BASE}/data/STR_data/HipSTR_data/all_samples_union.txt
PED_FILE=${BASE}/data/1000_genomes/20130606_g1k_3202_samples_ped_population.txt
OUT_DIR=${BASE}/data/STR_data/HipSTR_data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python ${SCRIPT_DIR}/filter_trios_and_count.py \
	--samples-file ${SAMPLES_FILE} \
	--ped-file ${PED_FILE} \
	--out-dir ${OUT_DIR}