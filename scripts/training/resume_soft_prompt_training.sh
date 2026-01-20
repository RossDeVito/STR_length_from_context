#!/bin/bash

# Check if resume and output directory arguments were provided
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

echo "Resuming soft prompt training from directory: $RESUME_DIR"

# Run the resume command
python -m seq_models.hyenaDNA.resume_soft_prompt_training \
	--resume_dir "${RESUME_DIR}" \
	--output_dir "${OUTPUT_DIR}"