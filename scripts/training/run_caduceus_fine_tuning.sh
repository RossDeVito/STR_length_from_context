#!/bin/bash

OUTPUT_DIR="scripts/training/output/caduceus"
CONFIG_DIR="scripts/training/training_configs/caduceus"

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Error: No config file specified."
  echo "Usage: $0 <config_filename>"
  echo "Config files are located in ${CONFIG_DIR}"
  exit 1
fi

CONFIG_FILE="$1"
echo "Running Caduceus fine-tuning with config: $CONFIG_FILE"

# Confirm the config file exists
if [ ! -f "${CONFIG_DIR}/${CONFIG_FILE}" ]; then
  echo "Error: Config file ${CONFIG_DIR}/${CONFIG_FILE} does not exist."
  exit 1
fi

# Run the command
python -m seq_models.caduceus.fine_tune \
    --config "${CONFIG_DIR}/${CONFIG_FILE}" \
    --output_dir "${OUTPUT_DIR}"
