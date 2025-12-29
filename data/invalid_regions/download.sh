#!/bin/bash

# Define URLs (hg38)
RMSK_URL="https://hgdownload.soe.ucsc.edu/goldenpath/hg38/database/rmsk.txt.gz"
SEGDUPS_URL="https://hgdownload.soe.ucsc.edu/goldenpath/hg38/database/genomicSuperDups.txt.gz"
BLACKLIST_URL="https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz"

echo "--- 1. Processing Mobile Elements (RepeatMasker) ---"
# Filter: Keep SINE, LINE, LTR, DNA
# Extract: Columns 6(chrom), 7(start), 8(end), 12(repClass)
wget -qO- $RMSK_URL | gunzip -c | awk 'BEGIN{OFS="\t"} {
    if ($12 ~ /SINE|LINE|LTR|DNA/) {
        print $6, $7, $8, $12
    }
}' | gzip > mobile_elements.bed.gz
echo "Saved mobile_elements.bed.gz (4 columns: chrom, start, end, class)"

echo "--- 2. Processing Segmental Dups ---"
# Keep cols 2, 3, 4 (chrom, start, end)
wget -qO- $SEGDUPS_URL | gunzip -c | awk 'BEGIN{OFS="\t"} {print $2, $3, $4}' | gzip > seg_dups.bed.gz
echo "Saved seg_dups.bed.gz"

echo "--- 3. Processing Blacklist ---"
# Will extract only the first 3 columns (chrom, start, end)
wget -qO- $BLACKLIST_URL | gunzip -c | awk 'BEGIN{OFS="\t"} {print $1, $2, $3}' | gzip > blacklist.bed.gz
echo "Saved blacklist.bed.gz"
