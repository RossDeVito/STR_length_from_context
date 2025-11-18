""" Perform soft prompt tuning on HyenaDNA for STR length prediction. 

Requires YAML config file specifying data/model/training params.

Args:
	--config: Path to YAML config file.
	--output_dir: Directory to save outputs (models, logs).
	--cpu: Force use of CPU, even if MPS/cuda is available.
"""

import argparse
import datetime
import os
import json
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
	ModelCheckpoint,
	EarlyStopping,
	LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger

from seq_models.hyenaDNA.model import create_model
from seq_models.hyenaDNA.data import create_data_module


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train STRLengthModel")
	parser.add_argument(
		"--config",
		type=str,
		required=True,
		help="Path to the configuration YAML file."
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=".",
		help="Directory to save outputs (models, logs)."
	)
	parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force use of CPU, even if MPS is available."
    )
	args = parser.parse_args()
	

	# Load config
	print(f"Loading config from {args.config}")
	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)
	
	if config['tuning_strategy'] != 'soft_prompt':
		raise ValueError(
			"Config tuning_strategy must be 'soft_prompt' for this script."
		)
	
	# Set up logging and output directories
	experiment_name = config.get("experiment_name", "soft_prompt")
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	experiment_path = os.path.join(
		args.output_dir, f"{experiment_name}_{timestamp}"
		)
	os.makedirs(experiment_path, exist_ok=True)

	logger = TensorBoardLogger(save_dir=experiment_path, name="", version=".")
	print(f"Outputs will be saved to: {experiment_path}")

	# Create DataModule
	print("Initializing DataModule...")
	datamodule = create_data_module(config)

	# Create model
	print(f"Initializing Model...")
	model = create_model(config)

	# Define callbacks
	print("Setting up callbacks...")
	
	# Checkpoint callback
	checkpoint_dir = os.path.join(experiment_path, "checkpoints")
	checkpoint_callback = ModelCheckpoint(
		dirpath=checkpoint_dir,
		filename="best-model-{epoch:02d}",
		monitor="val_loss",
		mode="min",
		save_top_k=1,
		verbose=True
	)

	# Early stopping callback
	early_stop_callback = EarlyStopping(
		monitor="val_loss",
		patience=config.get("patience_early_stopping", 10),
		verbose=True,
		mode="min"
	)
	
	# Learning rate monitor
	lr_monitor = LearningRateMonitor(logging_interval="epoch")

	callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

	# Initialize trainer
	print("Initializing Trainer...")
	if args.cpu:
		print("--- CPU override flag set. Using CPU. ---")
		accelerator = "cpu"
		devices = 1
	else:
		print("--- Using 'auto' accelerator (GPU/MPS if available) ---")
		accelerator = "auto"
		devices = 1

	trainer = pl.Trainer(
		max_epochs=config.get("max_epochs", 100),
		accelerator=accelerator,
		devices=devices,
		logger=logger,
		callbacks=callbacks,
		gradient_clip_val=config.get("gradient_clip_val", 0.0),
		precision=config.get("precision", "32-true")
	)

	# Run training
	print("--- Starting training ---")
	trainer.fit(model, datamodule=datamodule)
	print("--- Training complete ---")

	# Run eval on test set
	print("--- Running evaluation on test set ---")
	test_results = trainer.test(datamodule=datamodule, ckpt_path="best")
	print("--- Evaluation complete ---")

	if test_results:
		# Get the first (and only) result dictionary
		test_metrics = test_results[0]
		
		# Add the best model path for reference
		test_metrics["best_model_path"] = checkpoint_callback.best_model_path
		
		json_path = os.path.join(experiment_path, "test_results.json")
		print(f"Saving test results to {json_path}")
		with open(json_path, 'w') as f:
			json.dump(test_metrics, f, indent=4)
	
	print(f"\nFind your logs in TensorBoard:")
	print(f"tensorboard --logdir={os.path.abspath(experiment_path)}")
	print(f"\nBest model saved to: {checkpoint_callback.best_model_path}")