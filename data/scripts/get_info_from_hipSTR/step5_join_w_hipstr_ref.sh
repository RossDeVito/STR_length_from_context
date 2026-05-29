#!/bin/bash
# Step 5: Join the combined statSTR output with the HipSTR reference BED.
#
# Reference BED columns (no header):
#   chrom  start  end  motif_len  ref_copy_number  hipstr_name  motif
# Join key: (chrom, start, end). Expected 1-to-1.

set -euo pipefail

BASE=/tscc/projects/ps-gymreklab/rdevito/str_len_pred/STR_length_from_context

STATSTR_FILE=${BASE}/data/STR_data/HipSTR_data/all_chroms_statstr.tab
REF_BED=${BASE}/data/STR_data/HipSTR-reference/hg38.hipstr_reference.bed
OUT_FILE=${BASE}/data/STR_data/HipSTR_data/all_chroms_statstr_with_ref.tab

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python ${SCRIPT_DIR}/merge_statstr_with_reference.py \
	--statstr-file ${STATSTR_FILE} \
	--ref-bed ${REF_BED} \
	--out-file ${OUT_FILE}