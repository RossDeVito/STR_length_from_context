#!/bin/bash

# Script creates all reference-based labeled sequences for STRs
# with motif lengths 1 and 2 for sequence based models with maximum
# sequence lengths of 1024, 32,768, 160,000, 450,000, and 1,000,000 bp.

MAX_PERCENTILE=0.975

for STR_LEN in 1 2
do
	for MAX_SEQ_LEN in 1024 32768 160000 450000 1000000
	do
		N_FLANKING=$((MAX_SEQ_LEN / 2))
		
		python create_ref_based_labeled_seqs.py \
			--str-len ${STR_LEN} \
			--n-flanking ${N_FLANKING} \
			--copy-num-percentile-max ${MAX_PERCENTILE}
	done
done

echo "All reference-based labeled sequences created."