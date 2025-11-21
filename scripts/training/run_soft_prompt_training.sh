#!/bin/bash
# This script runs the soft prompt training.
# It assumes it is being run from the project root

OUTPUT_DIR="scripts/training/output/soft_prompt"
CONFIG_DIR="scripts/training/training_configs/soft_prompt"
CONFIG_FILE="sp_dev1_str2.yaml"

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -m seq_models.hyenaDNA.train_soft_prompt \
	--config "${CONFIG_DIR}/${CONFIG_FILE}" \
	--output_dir "${OUTPUT_DIR}"
