#!/bin/bash

# Download into the directory this script lives in, not the current working dir
cd "$(dirname "$0")"

wget https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed
