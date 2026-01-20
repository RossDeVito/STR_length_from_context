#!/bin/bash

CONFIG_DIR="scripts/training/training_configs/soft_prompt"

# Check that two arguments were provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 <config_filename> <output_directory>"
  echo "Config files are located in ${CONFIG_DIR}"
  exit 1
fi

# Assign the first argument ($1) to CONFIG_FILE
CONFIG_FILE="$1"

# Assign the second argument ($2) to OUTPUT_DIR
OUTPUT_DIR="$2"

# Print what is running for verification
echo "Running soft prompt training with config: $CONFIG_FILE"
echo "Output directory set to: $OUTPUT_DIR"

# Confirm the config file exists
if [ ! -f "${CONFIG_DIR}/${CONFIG_FILE}" ]; then
  echo "Error: Config file ${CONFIG_DIR}/${CONFIG_FILE} does not exist."
  exit 1
fi

# Run the command
python -m seq_models.hyenaDNA.train_soft_prompt \
    --config "${CONFIG_DIR}/${CONFIG_FILE}" \
    --output_dir "${OUTPUT_DIR}"