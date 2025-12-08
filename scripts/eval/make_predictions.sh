#!/bin/bash

MODEL_DIR="scripts/training/output/"
OUTPUT_DIR="scripts/eval/predictions/"

# Check if an argument was provided
if [ -z "$1" ]; then
	echo "Error: No path to model savedir specified."
	echo "Usage: $0 <model_savedir>"
	echo " where <model_savedir> is a subdirectory of ${MODEL_DIR}"
	exit 1
fi

# Assign the first argument ($1) to MODEL_SAVEDIR
MODEL_SAVEDIR="$1"
echo "Making predictions with model in: ${MODEL_SAVEDIR}"

# Run the command
python -m seq_models.hyenaDNA.predict \
	--model_dir "${MODEL_DIR}/${MODEL_SAVEDIR}" \
	--output_dir "${OUTPUT_DIR}/${MODEL_SAVEDIR}"