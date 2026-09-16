""" Makes predictions with a trained Caduceus STRLengthModel for a data split.

Writes two TSVs to --output_dir:
	predictions_{split}_by_orientation.tsv
		One row per sample (forward and reverse-complement separately), with
		columns: id, rev_comp, pred_{task}, true_{task} for each active target.
	predictions_{split}.tsv
		One row per locus after post-hoc conjoining: the native-space predictions
		of the two reverse-complement orientations are averaged (grouped by id).
		Locus metadata (chrom, str_start, str_end, motif, split) is merged in when
		available. Columns pred_length/true_length stay compatible with
		scripts/eval_preds/.

Args:
	model_dir: Directory containing the trained checkpoint (checkpoints/*.ckpt)
		and config.yaml.
	split: Data split to predict ('test', 'val', 'train'). Default: 'test'.
	batch_size: Override batch size for prediction. Default: use config value.
	cpu: Force CPU usage.
	output_dir: Directory to save prediction TSVs.
	dev: If set, cap each split to 1000 samples for a quick run.
"""

import argparse
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml

from seq_models.caduceus.data import create_data_module
from seq_models.caduceus.model import STRLengthModel

# Base directory holding trained Caduceus models (matches make_predictions.sh).
CADUCEUS_MODEL_DIR = "scripts/training/output/caduceus"


def aggregate_predictions(predictions, task_names):
	"""Flatten predict_step outputs and post-hoc-conjoin reverse complements.

	Args:
		predictions (list[dict]): per-batch outputs from `trainer.predict`, each
			with keys 'id', 'rev_comp', and 'pred_{t}'/'label_{t}' per target.
		task_names (list[str]): active target names.

	Returns:
		(per_orientation_df, per_locus_df):
			per_orientation_df has one row per sample with columns id, rev_comp,
			pred_{t}, true_{t}. per_locus_df has one row per id with pred_{t}
			averaged across orientations and true_{t} carried through.
	"""
	records = []
	for batch in predictions:
		ids = batch["id"]
		rev_comp = batch["rev_comp"]
		for i in range(len(ids)):
			rec = {"id": ids[i], "rev_comp": bool(rev_comp[i])}
			for task in task_names:
				rec[f"pred_{task}"] = float(batch[f"pred_{task}"][i])
				rec[f"true_{task}"] = float(batch[f"label_{task}"][i])
			records.append(rec)

	per_orientation = pd.DataFrame(records)

	# Post-hoc conjoining: average native-space preds across orientations.
	agg = {f"pred_{task}": "mean" for task in task_names}
	agg.update({f"true_{task}": "first" for task in task_names})
	per_locus = per_orientation.groupby("id", as_index=False).agg(agg)

	return per_orientation, per_locus


def parse_args():
	parser = argparse.ArgumentParser(
		description="Predict with a trained Caduceus STRLengthModel."
	)
	parser.add_argument(
		"--model_dir", type=str, required=True,
		help="Directory with the trained checkpoint and config.yaml."
	)
	parser.add_argument(
		"--split", type=str, default="test",
		help="Data split to predict ('test', 'val', 'train'). Default: 'test'."
	)
	parser.add_argument(
		"--batch_size", type=int, default=None,
		help="Override batch size for prediction."
	)
	parser.add_argument("--cpu", action="store_true", help="Force CPU usage.")
	parser.add_argument(
		"--output_dir", type=str, required=True,
		help="Directory to save prediction TSVs."
	)
	parser.add_argument(
		"--dev", action="store_true",
		help="If set, cap each split to 1000 samples for a quick run."
	)
	return parser.parse_args()


def find_best_checkpoint(model_dir):
	checkpoint_dir = os.path.join(model_dir, "checkpoints")
	if not os.path.exists(checkpoint_dir):
		raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

	ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
	if not ckpt_files:
		raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

	# Prefer files with 'best' in the name; otherwise require a single choice.
	best = [f for f in ckpt_files if "best" in f]
	if best:
		ckpt_files = best
	elif len(ckpt_files) > 1:
		raise ValueError(
			f"Multiple checkpoints in {checkpoint_dir} and none marked 'best'."
		)
	return os.path.join(checkpoint_dir, ckpt_files[0])


def load_model_and_data(config):
	"""Load a trained model and build the dataset for a requested split.

	Reads the model's own config.yaml, overrides a few loader-relevant keys
	from ``config``, builds the data module, and picks the dataset for the
	split.

	Args:
		config (dict): Parsed config with keys ``model_dir`` (relative to
			CADUCEUS_MODEL_DIR), ``split``, and optionally ``batch_size`` /
			``num_workers``.

	Returns:
		Tuple (model, model_config, dataset, str_df) where dataset is the
		STRLengthDataset for the split and str_df is its underlying DataFrame.
	"""
	full_model_dir = os.path.join(CADUCEUS_MODEL_DIR, config["model_dir"])

	model_config_path = os.path.join(full_model_dir, "config.yaml")
	print(f"Loading model config from {model_config_path}")
	with open(model_config_path, "r") as f:
		model_config = yaml.safe_load(f)

	# Override the loader-relevant keys from the caller's config.
	model_config["batch_size"] = config.get(
		"batch_size", model_config.get("batch_size", 16)
	)
	model_config["dataloader_num_workers"] = config.get(
		"num_workers", model_config.get("dataloader_num_workers", 4)
	)

	data_module = create_data_module(model_config)
	data_module.setup()

	split = config["split"]
	if split == "test":
		dataset = data_module.test_dataset
	elif split == "val":
		dataset = data_module.val_dataset
	elif split == "train":
		dataset = data_module.train_dataset
	else:
		raise ValueError(
			f"Invalid split: {split}. Must be 'test', 'val', or 'train'."
		)

	ckpt_path = find_best_checkpoint(full_model_dir)
	print(f"Loading model from checkpoint {ckpt_path}")
	model = STRLengthModel.load_from_checkpoint(ckpt_path, map_location="cpu")
	model.eval()

	return model, model_config, dataset, dataset.str_df


def subsample_dataset(dataset, n_loci, seed):
	"""Restrict a dataset to n_loci unique loci, keeping BOTH orientations.

	Deterministically (seeded) selects n_loci unique ``ID`` values and filters
	the dataset's str_df to the rows for those ids (forward + reverse
	complement). The dataset is filtered in place and its index reset.

	Args:
		dataset (STRLengthDataset): Dataset to subsample (modified in place).
		n_loci (int): Number of unique loci to keep.
		seed (int): Random seed.

	Returns:
		The same dataset, with a reduced str_df.
	"""
	df = dataset.str_df
	unique_ids = np.sort(df["ID"].unique())
	if n_loci >= len(unique_ids):
		print(
			f"  subsample_loci={n_loci} >= {len(unique_ids)} unique loci; "
			f"using all."
		)
		return dataset

	rng = np.random.default_rng(seed)
	chosen = rng.choice(unique_ids, size=n_loci, replace=False)
	chosen_set = set(chosen.tolist())

	dataset.str_df = (
		df[df["ID"].isin(chosen_set)].reset_index(drop=True)
	)
	print(
		f"  Subsampled to {n_loci} loci "
		f"({len(dataset.str_df)} orientation rows)."
	)
	return dataset


if __name__ == "__main__":

	__spec__ = None

	args = parse_args()
	if args.dev:
		print("DEV MODE: capping each split to 1000 samples.")

	# Load config saved alongside the checkpoint.
	config_path = os.path.join(args.model_dir, "config.yaml")
	print(f"Loading config from {config_path}")
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)

	if args.batch_size is not None:
		config["batch_size"] = args.batch_size
	if args.dev:
		config["dev_subset_n"] = 1000

	# Build the data for the requested split.
	data_module = create_data_module(config)
	data_module.setup()
	if args.split == "test":
		data_loader = data_module.test_dataloader()
		source_df = data_module.test_dataset.str_df
	elif args.split == "val":
		data_loader = data_module.val_dataloader()
		source_df = data_module.val_dataset.str_df
	elif args.split == "train":
		data_loader = data_module.train_dataloader()
		source_df = data_module.train_dataset.str_df
	else:
		raise ValueError(
			f"Invalid split: {args.split}. Must be 'test', 'val', or 'train'."
		)

	# Load the model.
	ckpt_path = find_best_checkpoint(args.model_dir)
	print(f"Loading model from checkpoint {ckpt_path}")
	model = STRLengthModel.load_from_checkpoint(ckpt_path)
	model.eval()

	# 32-bit for inference stability regardless of training precision.
	trainer = pl.Trainer(
		accelerator="cpu" if args.cpu else "auto",
		devices=1,
		logger=False,
		precision="32-true",
	)

	print(f"Running predictions on {args.split} split...")
	predictions = trainer.predict(model, dataloaders=data_loader)

	per_orientation, per_locus = aggregate_predictions(
		predictions, list(model.task_names)
	)

	# Merge locus metadata into the per-locus table when present.
	meta_cols = [
		c for c in ["chrom", "str_start", "str_end", "motif", "split"]
		if c in source_df.columns
	]
	if meta_cols:
		meta = (
			source_df[["ID"] + meta_cols]
			.drop_duplicates("ID")
			.rename(columns={"ID": "id"})
		)
		per_locus = per_locus.merge(meta, on="id", how="left")

	os.makedirs(args.output_dir, exist_ok=True)
	suffix = "_dev" if args.dev else ""
	orient_path = os.path.join(
		args.output_dir, f"predictions_{args.split}_by_orientation{suffix}.tsv"
	)
	locus_path = os.path.join(
		args.output_dir, f"predictions_{args.split}{suffix}.tsv"
	)

	print(f"Saving {len(per_orientation)} per-orientation rows to {orient_path}")
	per_orientation.to_csv(orient_path, sep="\t", index=False)
	print(f"Saving {len(per_locus)} per-locus (RC-averaged) rows to {locus_path}")
	per_locus.to_csv(locus_path, sep="\t", index=False)
