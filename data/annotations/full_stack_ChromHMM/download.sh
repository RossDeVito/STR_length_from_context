#!/bin/bash

# Download hg38 full stack ChromHMM annotations from
# https://doi.org/10.1186/s13059-021-02572-z

# Download into the directory this script lives in, not the current working dir
cd "$(dirname "$0")"

wget https://public.hoffman2.idre.ucla.edu/ernst/2K9RS/full_stack/full_stack_annotation_public_release/hg38/hg38_genome_100_segments.bed.gz
wget https://public.hoffman2.idre.ucla.edu/ernst/2K9RS/full_stack/full_stack_annotation_public_release/hg38/state_annotations_processed.csv