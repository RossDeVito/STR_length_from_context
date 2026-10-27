#!/bin/bash

# Stranded/family-resolved RepeatMasker rows, for te_downstream_analysis.py.
#
# download.sh builds mobile_elements.bed.gz from the same rmsk.txt.gz, but
# keeps only (chrom, start, end, repClass) -- enough for the exclusion filter
# in create_STR_data_files.py and for nearby_repeat_elements.py, neither of
# which cares which way an element points or which family it belongs to.
#
# The poly(A)-tail analysis does care, on both counts:
#   strand    -- "downstream" is defined in the ELEMENT's orientation (past its
#                3' end), not the genome's, so the position of an STR relative
#                to an element is unreadable without it.
#   repFamily -- the poly(A)-retrotransposition classes are families, not
#                classes: L1 (not all of LINE), Alu (not all of SINE), SVA
#                (not all of Retroposon).
#
# The repClass filter below is character-for-character the one in download.sh,
# so the ROWS here are exactly the rows of mobile_elements.bed.gz -- same
# source, same release, same filter, three more columns. Keeping them
# identical is what lets the new analysis be read against the old one.
# te_downstream_analysis.py applies its own, tighter class restriction
# (SINE/LINE/LTR/DNA/Retroposon/RC -- no Satellite, no RNA) in Python.

# Download into the directory this script lives in, not the current working dir
cd "$(dirname "$0")"

RMSK_URL="https://hgdownload.soe.ucsc.edu/goldenpath/hg38/database/rmsk.txt.gz"

echo "--- RepeatMasker rows with strand + family ---"
# rmsk.txt columns (1-based): 6 genoName, 7 genoStart, 8 genoEnd, 10 strand,
# 11 repName, 12 repClass, 13 repFamily. genoStart/genoEnd are 0-based
# half-open, as in the bed written here.
wget -qO- $RMSK_URL | gunzip -c | awk 'BEGIN{OFS="\t"} {
    if ($12 ~ /SINE|LINE|LTR|DNA|Satellite|RC|RNA|Retroposon/) {
        print $6, $7, $8, $10, $11, $12, $13
    }
}' | gzip > rmsk_stranded.bed.gz
echo "Saved rmsk_stranded.bed.gz (7 columns: chrom, start, end, strand, repName, repClass, repFamily)"
