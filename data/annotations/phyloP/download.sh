#!/bin/bash

# Download hg38 phyloP annotations using multiple alignment of 447 mammalian
# genomes.
#
# UCSC info: https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=cons447way
# Source: https://doi.org/10.1038/s41586-023-06798-8

# Download into the directory this script lives in, not the current working dir
cd "$(dirname "$0")"

wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/phyloP447way/hg38.phyloP447way.bw