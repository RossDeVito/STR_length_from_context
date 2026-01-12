#!/bin/bash

OUTPUT_DIR="scripts/training/output/soft_prompt"

# Check if resume dir argument was provided
if [ -z "$1" ]; then
	echo "Error: No resume directory specified."
	echo "Usage: $0 <resume_directory>"
	exit 1
fi

RESUME_DIR="$1"

echo "Resuming soft prompt training from directory: $RESUME_DIR"

# Run the resume command
python -m seq_models.hyenaDNA.resume_soft_prompt_training \
	--resume_dir "${RESUME_DIR}" \
	--output_dir "${OUTPUT_DIR}"