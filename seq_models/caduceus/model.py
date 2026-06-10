""" Caduceus-Ph based model for STR length / variation prediction.

- Loads the Caduceus-Ph backbone from Hugging Face (full fine-tuning).
- Adds optional learnable input tokens (prefix / STR-gap / suffix) that are new
  to the backbone, held in a SEPARATE embedding matrix from the pretrained one so
  they can be trained with a different (higher) learning rate than the backbone.
- One regression head per active target (length and/or variation), pooled from
  the sequence with attention pooling.
- Two-objective targets are balanced with homoscedastic uncertainty weighting
  (Kendall et al. 2018); single-objective is a plain loss.

create_model() builds the model from a config dict.

Note: instantiating the model loads the Caduceus backbone, which requires the
`mamba_ssm` / `causal_conv1d` CUDA kernels. Importing this module does NOT (the
checkpoint id, transforms, split embedding, pooling and loss helpers are all
usable without the backbone).
"""

import math
import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (
	AutoModel,
	AutoConfig,
	get_linear_schedule_with_warmup,
	get_cosine_schedule_with_warmup,
)
from torchmetrics import (
	MeanAbsoluteError,
	MeanSquaredError,
	MetricCollection,
)
from torchmetrics.regression import PearsonCorrCoef, R2Score, SpearmanCorrCoef


# The single Caduceus-Ph checkpoint used throughout this project. Owned here;
# data.py and others import it from this module.
CADUCEUS_CHECKPOINT = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"

# Default mapping of output_name -> source column in the data file (shared with
# data.py, which imports this).
DEFAULT_TARGETS = {
	"length": "mode_copy_number",
	"variation": "heterozygosity",
}

# Default training-space transform per output_name.
DEFAULT_TARGET_TRANSFORMS = {
	"length": "log1p",        # copy number: log(1 + n)
	"variation": "arcsin_sqrt",  # heterozygosity in [0, 1]: variance-stabilizing
}

_HALF_PI = math.pi / 2.0

_VOCAB_SIZE_CACHE = None


def get_caduceus_vocab_size():
	"""Padded vocab size of the Caduceus checkpoint (16 for the ph_131k model).

	This is the number of rows in the pretrained embedding and therefore the id
	at which the learnable tokens start. Loading only the config does NOT import
	the mamba kernels. Cached after the first call.
	"""
	global _VOCAB_SIZE_CACHE
	if _VOCAB_SIZE_CACHE is None:
		cfg = AutoConfig.from_pretrained(CADUCEUS_CHECKPOINT, trust_remote_code=True)
		_VOCAB_SIZE_CACHE = int(cfg.vocab_size)
	return _VOCAB_SIZE_CACHE


# --- Target transforms --------------------------------------------------------

def _identity(x):
	return x


def _arcsin_sqrt(y):
	return torch.arcsin(torch.sqrt(torch.clamp(y, 0.0, 1.0)))


def _arcsin_sqrt_inv(p):
	return torch.sin(torch.clamp(p, 0.0, _HALF_PI)) ** 2


# name -> (forward transform, inverse transform)
TARGET_TRANSFORMS = {
	"none": (_identity, _identity),
	"log1p": (torch.log1p, torch.expm1),
	"arcsin_sqrt": (_arcsin_sqrt, _arcsin_sqrt_inv),
}


def transform_target(transform_name, y):
	"""Map a raw target into training space."""
	return TARGET_TRANSFORMS[transform_name][0](y)


def inverse_transform(transform_name, p):
	"""Map a training-space prediction back to native units."""
	return TARGET_TRANSFORMS[transform_name][1](p)


def resolve_transforms(target_names, overrides=None):
	"""Resolve the transform name for each target (defaults + overrides)."""
	overrides = overrides or {}
	resolved = {}
	for name in target_names:
		resolved[name] = overrides.get(
			name, DEFAULT_TARGET_TRANSFORMS.get(name, "none")
		)
	return resolved


def compute_monitor_norms(df, targets, transforms):
	"""Per-task variance of the transformed target column.

	Computed in the SAME (training) space the loss uses, so it can normalize the
	fixed-weight validation monitor. Intended to be computed once on the val set
	and frozen (stored in config), not recomputed per epoch.

	Args:
		df (pd.DataFrame): rows to compute over (typically the val split).
		targets (dict): output_name -> source column.
		transforms (dict): output_name -> transform name.

	Returns:
		dict: output_name -> float variance (>= 1e-8).
	"""
	norms = {}
	for name, col in targets.items():
		if len(df) == 0:
			norms[name] = 1.0
			continue
		vals = torch.tensor(df[col].to_numpy(), dtype=torch.float32)
		tvals = transform_target(transforms[name], vals)
		norms[name] = max(float(torch.var(tvals, unbiased=False)), 1e-8)
	return norms


# --- Submodules ---------------------------------------------------------------

class SplitEmbedding(nn.Module):
	""" Embedding split across two matrices.

	`original` holds the pretrained token rows (ids < base) and is trained at the
	backbone learning rate; `extra` holds the new learnable tokens (ids >= base)
	and is trained at the higher head learning rate. This keeps the two as
	separate parameters so the optimizer can assign them different LRs (the
	alternative — one resized matrix with a gradient mask — is avoided).
	"""

	def __init__(self, original: nn.Embedding, n_new: int):
		super().__init__()
		self.original = original
		self.base = original.num_embeddings
		self.embedding_dim = original.embedding_dim
		if n_new > 0:
			self.extra = nn.Embedding(n_new, self.embedding_dim)
			# Small init so new tokens don't dominate the pretrained embeddings.
			nn.init.normal_(self.extra.weight, mean=0.0, std=0.02)
		else:
			self.extra = None

	def forward(self, input_ids):
		if self.extra is None:
			return self.original(input_ids)

		is_extra = input_ids >= self.base
		zeros = torch.zeros_like(input_ids)
		# Original lookup with extra positions masked to a safe id (0).
		orig_ids = torch.where(is_extra, zeros, input_ids)
		orig_emb = self.original(orig_ids)
		# Extra lookup with original positions masked to a safe id (0).
		extra_ids = torch.where(is_extra, input_ids - self.base, zeros)
		extra_emb = self.extra(extra_ids)
		return torch.where(is_extra.unsqueeze(-1), extra_emb, orig_emb)


class AttentionPooling(nn.Module):
	""" Attention pooling with a single learnable query (as in the HyenaDNA
	version): the query attends over the sequence and returns one vector per
	example. """

	def __init__(self, hidden_size, num_heads=4):
		super().__init__()
		self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
		self.attn = nn.MultiheadAttention(
			embed_dim=hidden_size,
			num_heads=num_heads,
			batch_first=True,
		)

	def forward(self, sequence_output):
		# sequence_output: (B, L, H)
		batch_size = sequence_output.shape[0]
		query = self.query.expand(batch_size, -1, -1)  # (B, 1, H)
		attn_output, _ = self.attn(query, sequence_output, sequence_output)
		return attn_output.squeeze(1)  # (B, H)


def uncertainty_weighted_loss(raw_losses, log_vars):
	""" Homoscedastic uncertainty weighting (Kendall et al. 2018).

	For each task, total contribution = exp(-s_i) * L_i + s_i, where s_i = log
	sigma_i^2. At s_i = 0 (sigma^2 = 1) this reduces to sum_i L_i.

	Args:
		raw_losses (dict): task -> scalar loss tensor (native training-space).
		log_vars (dict): task -> scalar log-variance parameter.

	Returns:
		(total_loss, components): components[task] = dict with 's', 'weight'
			(exp(-s)), 'weighted' (exp(-s) * L) and 'raw' (L).
	"""
	total = 0.0
	components = {}
	for name, raw in raw_losses.items():
		s = log_vars[name]
		weight = torch.exp(-s)
		weighted = weight * raw
		total = total + weighted + s
		components[name] = {
			"s": s,
			"weight": weight,
			"weighted": weighted,
			"raw": raw,
		}
	return total, components


# --- LightningModule ----------------------------------------------------------

def create_model(config):
	""" Factory for STRLengthModel from a config dict.

	Required config keys: none (sensible defaults for everything).

	Key config keys:
		targets (dict): output_name -> source column. Default DEFAULT_TARGETS.
		target_transforms (dict): output_name -> transform override.
		n_prefix_prompt_tokens / n_str_prompt_tokens / n_suffix_prompt_tokens (int).
		head_hidden_layers (list[int]), head_dropout (float).
		use_attention_pooling (bool, default True), attention_pooling_num_heads (int).
		optimizer_name ('Adam' or 'AdamW'), lr, backbone_lr, weight_decay.
		scheduler_name, warmup_steps, scheduler_patience, scheduler_factor.
		monitor_norm (dict): precomputed per-task variance for the val monitor.
		gradient_checkpointing (bool).
	"""
	return STRLengthModel(
		targets=config.get("targets", DEFAULT_TARGETS),
		target_transforms=config.get("target_transforms", None),

		n_prefix_prompt_tokens=config.get("n_prefix_prompt_tokens", 0),
		n_str_prompt_tokens=config.get("n_str_prompt_tokens", 0),
		n_suffix_prompt_tokens=config.get("n_suffix_prompt_tokens", 0),

		head_hidden_layers=config.get("head_hidden_layers", None),
		head_dropout=float(config.get("head_dropout", 0.1)),
		use_attention_pooling=config.get("use_attention_pooling", True),
		attention_pooling_num_heads=config.get("attention_pooling_num_heads", 4),

		optimizer_name=config.get("optimizer_name", "AdamW"),
		lr=float(config.get("lr", 1e-4)),
		backbone_lr=float(config.get("backbone_lr", 1e-5)),
		weight_decay=float(config.get("weight_decay", 0.01)),

		scheduler_name=config.get("scheduler_name", "ReduceLROnPlateau"),
		warmup_steps=config.get("warmup_steps", 100),
		scheduler_patience=config.get("scheduler_patience", 3),
		scheduler_factor=config.get("scheduler_factor", 0.1),

		monitor_norm=config.get("monitor_norm", None),
		use_gradient_checkpointing=config.get("gradient_checkpointing", False),
	)


class STRLengthModel(pl.LightningModule):
	""" LightningModule fine-tuning Caduceus-Ph for STR length/variation. """

	def __init__(
		self,
		targets: dict,
		target_transforms=None,

		n_prefix_prompt_tokens: int = 0,
		n_str_prompt_tokens: int = 0,
		n_suffix_prompt_tokens: int = 0,

		head_hidden_layers=None,
		head_dropout: float = 0.1,
		use_attention_pooling: bool = True,
		attention_pooling_num_heads: int = 4,

		optimizer_name: str = "AdamW",
		lr: float = 1e-4,
		backbone_lr: float = 1e-5,
		weight_decay: float = 0.01,

		scheduler_name: str = "ReduceLROnPlateau",
		warmup_steps: int = 100,
		scheduler_patience: int = 3,
		scheduler_factor: float = 0.1,

		monitor_norm=None,
		use_gradient_checkpointing: bool = False,
	):
		super().__init__()
		self.save_hyperparameters()

		self.targets = dict(targets)
		self.task_names = list(self.targets.keys())
		self.transforms = resolve_transforms(self.task_names, target_transforms)
		self.use_uncertainty = len(self.task_names) >= 2
		self.monitor_norm = monitor_norm

		# Load Caduceus backbone (base model -> last_hidden_state).
		backbone_config = AutoConfig.from_pretrained(
			CADUCEUS_CHECKPOINT, trust_remote_code=True
		)
		self.hidden_size = backbone_config.d_model
		self.caduceus = AutoModel.from_pretrained(
			CADUCEUS_CHECKPOINT,
			config=backbone_config,
			trust_remote_code=True,
		)

		if use_gradient_checkpointing:
			try:
				self.caduceus.gradient_checkpointing_enable()
			except Exception as exc:
				warnings.warn(
					f"gradient_checkpointing_enable() not supported by Caduceus: {exc}"
				)

		# Replace the plain embedding with the two-matrix split embedding.
		orig_embed = self.caduceus.backbone.embeddings.word_embeddings
		assert orig_embed.embedding_dim == self.hidden_size, (
			f"Dim mismatch: {orig_embed.embedding_dim} vs {self.hidden_size}"
		)
		n_new = (
			n_prefix_prompt_tokens
			+ n_str_prompt_tokens
			+ n_suffix_prompt_tokens
		)
		self.caduceus.backbone.embeddings.word_embeddings = SplitEmbedding(
			orig_embed, n_new
		)

		# Pooling.
		if use_attention_pooling:
			self.pooling = AttentionPooling(
				self.hidden_size, attention_pooling_num_heads
			)
		else:
			self.pooling = None

		# One regression head per target.
		self.heads = nn.ModuleDict({
			name: self._build_head(head_hidden_layers, head_dropout)
			for name in self.task_names
		})

		# Uncertainty (log-variance) params, only for multi-task. Init s = 0.
		if self.use_uncertainty:
			self.log_vars = nn.ParameterDict({
				name: nn.Parameter(torch.zeros(())) for name in self.task_names
			})
		else:
			self.log_vars = None

		self.loss_fn = nn.MSELoss()

		# Per-task validation/test metrics in native units.
		metrics = MetricCollection({
			"mae": MeanAbsoluteError(),
			"rmse": MeanSquaredError(squared=False),
			"pearson": PearsonCorrCoef(),
			"spearman": SpearmanCorrCoef(),
			"r2": R2Score(),
		})
		self.val_metrics = nn.ModuleDict({
			name: metrics.clone(prefix=f"val_{name}_") for name in self.task_names
		})
		self.test_metrics = nn.ModuleDict({
			name: metrics.clone(prefix=f"test_{name}_") for name in self.task_names
		})

	def _build_head(self, hidden_layers, dropout):
		layers = []
		in_dim = self.hidden_size
		if hidden_layers:
			for hidden_dim in hidden_layers:
				layers.append(nn.Linear(in_dim, hidden_dim))
				layers.append(nn.ReLU())
				layers.append(nn.Dropout(dropout))
				in_dim = hidden_dim
		layers.append(nn.Linear(in_dim, 1))
		return nn.Sequential(*layers)

	def forward(self, input_ids):
		outputs = self.caduceus(input_ids=input_ids, return_dict=True)
		sequence_output = outputs.last_hidden_state  # (B, L, H)

		if self.pooling is not None:
			pooled = self.pooling(sequence_output)
		else:
			pooled = sequence_output.mean(dim=1)

		# Per-task predictions in transformed (training) space.
		return {
			name: self.heads[name](pooled).squeeze(-1)
			for name in self.task_names
		}

	def _common_step(self, batch):
		preds_t = self(batch["input_ids"])

		raw_losses = {}
		native = {}  # task -> (native_pred (detached), native_target)
		for name in self.task_names:
			tname = self.transforms[name]
			target_raw = batch[name]
			target_t = transform_target(tname, target_raw)
			raw_losses[name] = self.loss_fn(preds_t[name], target_t)
			native[name] = (
				inverse_transform(tname, preds_t[name].detach()),
				target_raw,
			)

		if self.use_uncertainty:
			total, components = uncertainty_weighted_loss(
				raw_losses, {n: self.log_vars[n] for n in self.task_names}
			)
		else:
			total = sum(raw_losses.values())
			components = {
				n: {"s": None, "weight": None, "weighted": raw_losses[n],
					"raw": raw_losses[n]}
				for n in self.task_names
			}

		return total, raw_losses, components, native

	def training_step(self, batch, batch_idx):
		total, raw_losses, components, _ = self._common_step(batch)
		bs = batch["input_ids"].shape[0]

		self.log(
			"train_loss", total, on_step=True, on_epoch=True, prog_bar=True,
			batch_size=bs,
		)
		for name in self.task_names:
			self.log(
				f"train_rawloss_{name}", raw_losses[name],
				on_step=True, on_epoch=True, batch_size=bs,
			)
			if self.use_uncertainty:
				c = components[name]
				# s_i (== regularizer), effective weight exp(-s), weighted term.
				self.log(f"train_s_{name}", c["s"], on_step=True, on_epoch=False,
						 batch_size=bs)
				self.log(f"train_weight_{name}", c["weight"], on_step=True,
						 on_epoch=False, batch_size=bs)
				self.log(f"train_wloss_{name}", c["weighted"], on_step=True,
						 on_epoch=True, batch_size=bs)

		return total

	def _eval_step(self, batch, raw_losses, components, native, metrics, stage):
		bs = batch["input_ids"].shape[0]

		# Fixed-weight, native-space monitor (comparable across steps). Each task
		# loss is normalized by the frozen variance of its transformed target.
		fixed = 0.0
		for name in self.task_names:
			norm = 1.0
			if self.monitor_norm is not None:
				norm = max(float(self.monitor_norm[name]), 1e-8)
			fixed = fixed + raw_losses[name] / norm
		self.log(f"{stage}_fixed_loss", fixed, on_step=False, on_epoch=True,
				 prog_bar=(stage == "val"), batch_size=bs)

		for name in self.task_names:
			self.log(f"{stage}_rawloss_{name}", raw_losses[name],
					 on_step=False, on_epoch=True, batch_size=bs)
			if self.use_uncertainty:
				self.log(f"{stage}_s_{name}", components[name]["s"],
						 on_step=False, on_epoch=True, batch_size=bs)
				self.log(f"{stage}_weight_{name}", components[name]["weight"],
						 on_step=False, on_epoch=True, batch_size=bs)

			native_pred, native_target = native[name]
			m = metrics[name]
			m.update(native_pred, native_target)
			self.log_dict(m, on_step=False, on_epoch=True, batch_size=bs)

	def validation_step(self, batch, batch_idx):
		total, raw_losses, components, native = self._common_step(batch)
		# Uncertainty-weighted total: logged for reference, NOT for selection.
		self.log("val_loss", total, on_step=False, on_epoch=True,
				 batch_size=batch["input_ids"].shape[0])
		self._eval_step(batch, raw_losses, components, native,
						self.val_metrics, "val")

	def test_step(self, batch, batch_idx):
		total, raw_losses, components, native = self._common_step(batch)
		self.log("test_loss", total, on_step=False, on_epoch=True,
				 batch_size=batch["input_ids"].shape[0])
		self._eval_step(batch, raw_losses, components, native,
						self.test_metrics, "test")

	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		"""Per-task native-space predictions plus passthrough id / rev_comp /
		labels so downstream code can average reverse-complement pairs."""
		preds_t = self(batch["input_ids"])
		out = {}
		if "id" in batch:
			out["id"] = batch["id"]
		if "rev_comp" in batch:
			out["rev_comp"] = batch["rev_comp"]
		for name in self.task_names:
			out[f"pred_{name}"] = inverse_transform(self.transforms[name], preds_t[name])
			if name in batch:
				out[f"label_{name}"] = batch[name]
		return out

	def on_fit_start(self):
		# If monitor_norm wasn't provided, compute it ONCE from the val split and
		# freeze it. Warn so it can be copied into config for reproducibility.
		if self.monitor_norm is None:
			val_df = self.trainer.datamodule.val_dataset.str_df
			self.monitor_norm = compute_monitor_norms(
				val_df, self.targets, self.transforms
			)
			warnings.warn(
				"monitor_norm not provided; computed once from the val split: "
				f"{self.monitor_norm}. Store this in config for reproducible "
				"checkpoint selection."
			)

	def configure_optimizers(self):
		try:
			opt_class = getattr(torch.optim, self.hparams.optimizer_name)
		except AttributeError:
			raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_name}")

		# High-LR group: heads, pooling, and the new-token (extra) embedding.
		high_lr_params = []
		for head in self.heads.values():
			high_lr_params += list(head.parameters())
		if self.pooling is not None:
			high_lr_params += list(self.pooling.parameters())
		extra_embed = self.caduceus.backbone.embeddings.word_embeddings.extra
		if extra_embed is not None:
			high_lr_params += list(extra_embed.parameters())
		high_lr_ids = {id(p) for p in high_lr_params}

		# Backbone group: everything else in the backbone (incl. original
		# embedding), at the lower backbone LR.
		backbone_params = [
			p for p in self.caduceus.parameters()
			if p.requires_grad and id(p) not in high_lr_ids
		]

		param_groups = [
			{"params": high_lr_params, "lr": self.hparams.lr,
			 "weight_decay": self.hparams.weight_decay},
			{"params": backbone_params, "lr": self.hparams.backbone_lr,
			 "weight_decay": self.hparams.weight_decay},
		]
		# Uncertainty params: own group, no weight decay.
		if self.use_uncertainty:
			param_groups.append({
				"params": [self.log_vars[n] for n in self.task_names],
				"lr": self.hparams.lr,
				"weight_decay": 0.0,
			})

		optimizer = opt_class(param_groups)

		# Scheduler. Selection/plateau monitors the fixed-weight val loss.
		if self.hparams.scheduler_name == "ReduceLROnPlateau":
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer,
				mode="min",
				factor=self.hparams.scheduler_factor,
				patience=self.hparams.scheduler_patience,
			)
			return {
				"optimizer": optimizer,
				"lr_scheduler": {
					"scheduler": scheduler,
					"monitor": "val_fixed_loss",
					"interval": "epoch",
				},
			}

		if self.hparams.scheduler_name in ["linear_with_warmup", "cosine_with_warmup"]:
			try:
				total_steps = self.trainer.estimated_stepping_batches
			except Exception:
				warnings.warn("Could not get estimated_stepping_batches; using 10000.")
				total_steps = 10000

			warmup_fn = (
				get_cosine_schedule_with_warmup
				if self.hparams.scheduler_name == "cosine_with_warmup"
				else get_linear_schedule_with_warmup
			)
			scheduler = warmup_fn(
				optimizer,
				num_warmup_steps=self.hparams.warmup_steps,
				num_training_steps=total_steps,
			)
			return {
				"optimizer": optimizer,
				"lr_scheduler": {
					"scheduler": scheduler, "interval": "step", "frequency": 1
				},
			}

		if self.hparams.scheduler_name == "None":
			return optimizer

		raise ValueError(f"Unknown scheduler: {self.hparams.scheduler_name}")
