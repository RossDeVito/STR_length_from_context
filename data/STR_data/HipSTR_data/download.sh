#!/bin/bash

# Download into the directory this script lives in, not the current working dir
cd "$(dirname "$0")"

wget https://zenodo.org/records/20499952/files/1000G_HipSTR_stats.tsv?download=1 -O 1000G_HipSTR_stats.tsv
