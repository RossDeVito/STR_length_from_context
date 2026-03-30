#!/bin/bash

OUTPUT_DIR="scripts/region_permutation_importance/output"
CONFIG_DIR="scripts/region_permutation_importance/configs"

# Check if an argument was provided
if [ -z "$1" ]; then
	echo "Error: No config file specified."
	echo "Usage: $0 <config_filename>"
	echo "Config files are located in ${CONFIG_DIR}"
	exit 1
fi

# Assign the first argument ($1) to CONFIG_FILE
CONFIG_FILE="$1"

# Print what is running for verification
echo "Running region permutation importance with config: $CONFIG_FILE"

# Confirm the config file exists
if [ ! -f "${CONFIG_DIR}/${CONFIG_FILE}" ]; then
	echo "Error: Config file ${CONFIG_DIR}/${CONFIG_FILE} does not exist."
	exit 1
fi

# Run command
python -m seq_models.hyenaDNA.region_permutation_importance \
	--config "${CONFIG_DIR}/${CONFIG_FILE}" \
	--output_dir "${OUTPUT_DIR}"