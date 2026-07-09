#!/bin/bash

# All paths below (and inside the configs) are relative to the repo root, and
# `python -m seq_models...` needs the repo root on sys.path, so run from there
# regardless of where this script was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

OUTPUT_DIR="scripts/eval_preds/predictions/baseline"
CONFIG_DIR="scripts/linear_baseline/configs"

# Check that at least one config was provided
if [ "$#" -eq 0 ]; then
  echo "Error: No config file specified."
  echo "Usage: $0 <config_filename> [<config_filename> ...]"
  echo "Config files are located in ${CONFIG_DIR}"
  exit 1
fi

echo "Running $# config(s): $*"

# Run each config in sequence, stopping immediately if any one fails.
for CONFIG_FILE in "$@"; do
  echo "=== Running linear baseline with config: ${CONFIG_FILE} ==="

  # Confirm the config file exists
  if [ ! -f "${CONFIG_DIR}/${CONFIG_FILE}" ]; then
    echo "Error: Config file ${CONFIG_DIR}/${CONFIG_FILE} does not exist."
    exit 1
  fi

  # Run the command
  python -m seq_models.linear.train_and_pred \
      --config "${CONFIG_DIR}/${CONFIG_FILE}" \
      --output_dir "${OUTPUT_DIR}"
  status=$?

  if [ "${status}" -ne 0 ]; then
    echo "Error: config ${CONFIG_FILE} failed with exit code ${status}. Stopping."
    exit "${status}"
  fi
done

echo "All configs completed successfully."