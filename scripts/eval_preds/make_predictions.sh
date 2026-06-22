#!/bin/bash

MODEL_DIR="scripts/training/output/caduceus/"
OUTPUT_DIR="scripts/eval_preds/predictions/caduceus/"

# Check if an argument was provided
if [ -z "$1" ]; then
	echo "Error: No path to model savedir specified."
	echo "Usage: $0 <model_savedir> [split]"
	echo " where <model_savedir> is a subdirectory of ${MODEL_DIR}"
	echo " and [split] is one of test|val|train (default: test)"
	exit 1
fi

# Assign the first argument ($1) to MODEL_SAVEDIR
MODEL_SAVEDIR="$1"
SPLIT="${2:-test}"
echo "Making predictions with model in: ${MODEL_SAVEDIR}"
echo "Split: ${SPLIT}"

# Run the command
python -m seq_models.caduceus.predict \
	--model_dir "${MODEL_DIR}/${MODEL_SAVEDIR}" \
	--output_dir "${OUTPUT_DIR}/${MODEL_SAVEDIR}" \
	--split "${SPLIT}"
