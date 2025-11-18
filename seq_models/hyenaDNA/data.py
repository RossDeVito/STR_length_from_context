""" Dataset and data loaders for STR length task with soft prompting. 

Function create_data_module() creates a PyTorch Lightning DataModule
based on the arguments in a config file used by the training script.
See create_data_module() docstring for required and optional config keys.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

from seq_models.hyenaDNA.hyenaDNA_info import PROMPT_START_ID


def create_data_module(config):
	""" Create STRLengthDataModule from config.

	Args:
		config (dict): Configuration dictionary.

	Required config keys:
		data_path (str): Path to the data file.
		hyenaDNA_checkpoint (str): Checkpoint name or path for tokenizer.
		batch_size (int): Batch size for data loaders.
		n_flanking_bp (int): Number of flanking base pairs to include
			on each side of the STR.
		n_str_bp (int): Number of base pairs of the STR to include
			on each side of the STR.
		n_prompt_tokens (int): Number of soft prompt tokens to use. Will
			be placed in the middle of the STR.
	
	Optional config keys:
		dataloader_num_workers (int): Number of workers for data loaders.
			Default: 4.
	"""

	return STRLengthDataModule(
		data_path=config["data_path"],
		tokenizer_checkpoint=config["hyenaDNA_checkpoint"],
		batch_size=config["batch_size"],
		n_flanking_bp=config["n_flanking_bp"],
		n_str_bp=config["n_str_bp"],
		n_prompt_tokens=config["n_prompt_tokens"],
		num_workers=config.get("dataloader_num_workers", 4),
	)



class STRLengthDataset(Dataset):
	""" Dataset for STR length prediction with soft prompting. 
	
	Attributes:
		str_df (pd.DataFrame): DataFrame with STR information.
		tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
		n_flanking_bp (int): Number of flanking base pairs to include on
			each side of the STR.
		n_str_bp (int): Number of base pairs of the STR to include on
			each side of the STR.
		n_prompt_tokens (int): Number of soft prompt tokens to use. Will
			be placed in the middle of the STR.
		prompt_token_ids (torch.Tensor): Tensor of prompt token IDs.
	"""

	def __init__(
		self,
		str_df,
		tokenizer,
		n_flanking_bp,
		n_str_bp,
		n_prompt_tokens,
	):
		"""Initialize dataset.
		
		Args:
			str_df (pd.DataFrame): DataFrame with STR information.
			tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
			n_flanking_bp (int): Number of flanking base pairs to include on
				each side of the STR.
			n_str_bp (int): Number of base pairs of the STR to include on
				each side of the STR.
			n_prompt_tokens (int): Number of soft prompt tokens to use. Will
				be placed in the middle of the STR.
		"""
		
		self.str_df = str_df.reset_index(drop=True)
		self.tokenizer = tokenizer
		self.n_flanking_bp = n_flanking_bp
		self.n_str_bp = n_str_bp
		self.n_prompt_tokens = n_prompt_tokens

		# Create prompt token IDs
		self.prompt_token_ids = torch.arange(
			PROMPT_START_ID,
			PROMPT_START_ID + n_prompt_tokens
		)

		self.validity_check()

	def __len__(self):
		return len(self.str_df)
	
	def validity_check(self):
		"""Check that the provided pre and post sequences are long enough.

		Assumes pre_seq and post_seq are strings all of the same lengths.
		"""

		# Check length of pre_seq and post_seq columns are as long as
		# n_flanking_bp by checking first row
		first_row = self.str_df.iloc[0]
		assert len(first_row["pre_seq"]) >= self.n_flanking_bp, (
			f"Pre-sequence length {len(first_row['pre_seq'])} is less than "
			f"n_flanking_bp {self.n_flanking_bp}"
		)
		assert len(first_row["post_seq"]) >= self.n_flanking_bp, (
			f"Post-sequence length {len(first_row['post_seq'])} is less than "
			f"n_flanking_bp {self.n_flanking_bp}"
		)

	def __getitem__(self, idx):
		"""Get item at index idx.

		Args:
			idx (int): Index of item to get.

		Returns:
			dict: Dictionary with keys 'input_ids' and 'label'.
		"""

		# Get sequences
		pre_seq = self.str_df.iloc[idx]["pre_seq"][-self.n_flanking_bp :]
		post_seq = self.str_df.iloc[idx]["post_seq"][: self.n_flanking_bp]
		
		str_start_seq = self.str_df.iloc[idx]["str_seq"][: self.n_str_bp]
		str_end_seq = self.str_df.iloc[idx]["str_seq"][-self.n_str_bp :]

		# Combine and tokenize
		start_tokens = self.tokenizer(
			pre_seq + str_start_seq,
			add_special_tokens=False,
			return_tensors='pt'
		).input_ids.squeeze(0)

		end_tokens = self.tokenizer(
			str_end_seq + post_seq,
			add_special_tokens=False,
			return_tensors='pt'
		).input_ids.squeeze(0)

		# Combine with prompt tokens in the middle and add label
		return {
			"input_ids": torch.cat(
				[
					start_tokens,
					self.prompt_token_ids,
					end_tokens
				],
				dim=0
			),
			"label": torch.tensor(
				self.str_df.iloc[idx]["copy_number"],
				dtype=torch.float32
			)
		}
	

class STRLengthDataModule(pl.LightningDataModule):
	""" DataModule for STR length prediction with soft prompting. """

	def __init__(
		self,
		data_path,
		tokenizer_checkpoint,
		batch_size,
		n_flanking_bp,
		n_str_bp,
		n_prompt_tokens,
		num_workers=4,
	):
		super().__init__()
		self.save_hyperparameters()

		self.tokenizer = None

	def setup(self, stage=None):

		# Load tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(
			self.hparams.tokenizer_checkpoint,
			trust_remote_code=True
		)

		# Load data and create df by split
		full_df = pd.read_csv(
			self.hparams.data_path,
			sep="\t"
		)

		train_df = full_df[full_df["split"] == "train"].reset_index(drop=True)
		val_df = full_df[full_df["split"] == "val"].reset_index(drop=True)
		test_df = full_df[full_df["split"] == "test"].reset_index(drop=True)

		dataset_args = {
			"tokenizer": self.tokenizer,
			"n_flanking_bp": self.hparams.n_flanking_bp,
			"n_str_bp": self.hparams.n_str_bp,
			"n_prompt_tokens": self.hparams.n_prompt_tokens,
		}

		self.train_dataset = STRLengthDataset(train_df, **dataset_args)
		self.val_dataset = STRLengthDataset(val_df, **dataset_args)
		self.test_dataset = STRLengthDataset(test_df, **dataset_args)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.hparams.batch_size,
			shuffle=True,
			num_workers=self.hparams.num_workers,
			persistent_workers=True,
		)
	
	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.num_workers,
			persistent_workers=True,
		)
	
	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.num_workers,
			persistent_workers=True,
		)