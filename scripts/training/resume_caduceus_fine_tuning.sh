#!/bin/bash

if [ -z "$1" ]; then
	echo "Error: No resume directory specified."
	echo "Usage: $0 <resume_directory> <output_directory>"
	exit 1
fi

if [ -z "$2" ]; then
	echo "Error: No output directory specified."
	echo "Usage: $0 <resume_directory> <output_directory>"
	exit 1
fi

RESUME_DIR="$1"
OUTPUT_DIR="$2"

echo "Resuming Caduceus fine-tuning from directory: $RESUME_DIR"

python -m seq_models.caduceus.resume_fine_tune \
	--resume_dir "${RESUME_DIR}" \
	--output_dir "${OUTPUT_DIR}"
