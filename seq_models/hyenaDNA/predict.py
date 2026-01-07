""" Makes predictions using a trained STRLengthModel for a data split.

Outputs TSV with columns:
	- true_length: True STR length.
	- pred_length: Predicted STR length.

Args:
	model_dir: Directory containing trained model checkpoint and config.
	split: Data split to evaluate ('test', 'val', 'train'). Default: 'test'.
	batch_size: Batch size for evaluation. Default: use config value.
	cpu: Force CPU usage.
	output_dir: Directory to save evaluation results.
	dev: If flag is set, only predicts for 1000 examples.
"""

import argparse
import os
import yaml
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from seq_models.hyenaDNA.model import STRLengthModel
from seq_models.hyenaDNA.data import create_data_module


def parse_args():
	parser = argparse.ArgumentParser(
		description="Evaluate trained STRLengthModel on test set."
	)
	parser.add_argument(
		"--model_dir",
		type=str,
		required=True,
		help="Directory containing the trained model checkpoint and config."
	)
	parser.add_argument(
		"--split",
		type=str,
		default="test",
		help="Data split to evaluate ('test', 'val', 'train'). Default: 'test'."
	)
	parser.add_argument(
		"--batch_size", 
		type=int, 
		default=None, 
		help="Override batch size for faster evaluation."
	)
	parser.add_argument(
		"--cpu", 
		action="store_true", 
		help="Force CPU usage."
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		required=True,
		help=(
			"Directory to save evaluation results."
		)
	)
	parser.add_argument(
		"--dev",
		action="store_true",
		help="If flag is set, just predicts for 2 batches for quick dev run."
	)
	return parser.parse_args()


def create_small_dev_set(data_module, n_samples=1000):
	""" Create a small dev set for quick testing. """
	
	data_module.train_dataset.str_df = data_module.train_dataset.str_df[:n_samples]
	data_module.val_dataset.str_df = data_module.val_dataset.str_df[:n_samples]
	data_module.test_dataset.str_df = data_module.test_dataset.str_df[:n_samples]


if __name__ == "__main__":

	__spec__ = None

	args = parse_args()
	if args.dev:
		print("DEV MODE: Using first 1000 samples for quick testing.")

	# Load config
	config_path = os.path.join(args.model_dir, "config.yaml")
	print(f"Loading config from {config_path}")
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	
	# Override batch size if specified
	if args.batch_size is not None:
		config['batch_size'] = args.batch_size

	# Create data loader and get underlying dataframe
	data_module = create_data_module(config)
	
	if args.split == "test":
		data_module.setup(stage="test")
		if args.dev:
			create_small_dev_set(data_module, n_samples=1000)
		data_loader = data_module.test_dataloader()
		source_df = data_module.test_dataset.str_df
	elif args.split == "val":
		data_module.setup(stage="validate")
		if args.dev:
			create_small_dev_set(data_module, n_samples=1000)
		data_loader = data_module.val_dataloader()
		source_df = data_module.val_dataset.str_df
	elif args.split == "train":
		data_module.setup(stage="train")
		if args.dev:
			create_small_dev_set(data_module, n_samples=1000)
		data_loader = data_module.train_dataloader()
		source_df = data_module.train_dataset.str_df
	else:
		raise ValueError(f"Invalid split: {args.split}. Must be one of 'test', 'val', 'train'.")

	# Load model from checkpoint
	# Find checkpoint file in model_dir/checkpoints/*.ckpt
	checkpoint_dir = os.path.join(args.model_dir, "checkpoints")
	if not os.path.exists(checkpoint_dir):
		raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

	ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
	if len(ckpt_files) == 0:
		raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
	
	# Filter to only best checkpoints with 'best' in filename
	best_ckpt_files = [f for f in ckpt_files if 'best' in f]
	if len(best_ckpt_files) > 0:
		ckpt_files = best_ckpt_files
	elif len(ckpt_files) > 1:
		raise ValueError(
			f"Multiple checkpoint files found in {checkpoint_dir} with 'best'" 
			" in filename."
		)
	
	ckpt_path = os.path.join(checkpoint_dir, ckpt_files[0])
	print(f"Loading model from checkpoint {ckpt_path}")
	model = STRLengthModel.load_from_checkpoint(ckpt_path)
	model.eval()

	# Set up trainer
	accelerator = "cpu" if args.cpu else "auto"
	devices = 1

	# Force 32-bit precision for inference stability, 
    # even if training was mixed precision.
	trainer = pl.Trainer(
		accelerator=accelerator,
		devices=devices,
		logger=False, 
		precision="32-true"
	)


	# Make predictions
	print(f"Running predictions on {args.split} split...")
	predictions = trainer.predict(model, dataloaders=data_loader)

	all_pred = []
	all_true = []

	for batch_preds in predictions:
		pred_lengths = batch_preds['preds'].cpu().numpy()
		true_lengths = batch_preds['labels'].cpu().numpy()

		all_pred.extend(pred_lengths.tolist())
		all_true.extend(true_lengths.tolist())

	source_df['pred_length'] = all_pred
	source_df['true_length'] = all_true

	# check that lengths match
	if not np.all(source_df['true_length'] == source_df['copy_number']):
		raise ValueError("Mismatch between predictions and STR info.")

	# Create output directory
	os.makedirs(args.output_dir, exist_ok=True)

	output_filename = f"predictions_{args.split}"
	if args.dev:
		output_filename += "_dev"
	output_filename += ".tsv"
	output_path = os.path.join(args.output_dir, output_filename)
	
	print(f"Saving {len(source_df)} predictions to {output_path}")
	source_df.to_csv(output_path, sep="\t", index=False)