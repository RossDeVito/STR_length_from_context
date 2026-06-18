"""Tests for seq_models/caduceus/model.py.

Run with: pytest seq_models/caduceus/test_model.py -v

The pure pieces (split embedding, attention pooling, target transforms,
uncertainty-weighted loss, monitor norms) are tested on CPU without the backbone.
The full-model forward/training_step test needs the Caduceus backbone, which
requires the mamba_ssm / causal_conv1d CUDA kernels; it is skipped gracefully
when they are unavailable (i.e. always locally, runs on the GPU cluster).
"""

import math
import os
import sys

import pandas as pd
import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers.modeling_outputs import BaseModelOutputWithNoAttention

_REPO_ROOT = os.path.abspath(
	os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
	sys.path.insert(0, _REPO_ROOT)

import seq_models.caduceus.model as model_mod
from seq_models.caduceus.data import create_data_module
from seq_models.caduceus.model import (
	AttentionPooling,
	DEFAULT_TARGETS,
	SplitEmbedding,
	compute_monitor_norms,
	create_model,
	fixed_inverse_variance_loss,
	inverse_transform,
	resolve_transforms,
	transform_target,
	uncertainty_weighted_loss,
)


# --- SplitEmbedding ------------------------------------------------------------

def _split_embedding(base=16, n_new=4, dim=5):
	orig = nn.Embedding(base, dim)
	se = SplitEmbedding(orig, n_new)
	return se


def test_split_embedding_routes_ids_to_correct_matrix():
	se = _split_embedding(base=16, n_new=4, dim=5)
	# Distinct, known weights so we can tell the matrices apart.
	with torch.no_grad():
		se.original.weight.copy_(torch.arange(16 * 5).float().reshape(16, 5))
		se.extra.weight.copy_(-torch.arange(4 * 5).float().reshape(4, 5) - 1)

	ids = torch.tensor([[7, 16, 8, 19]])  # DNA, extra0, DNA, extra3
	out = se(ids)
	assert out.shape == (1, 4, 5)
	assert torch.equal(out[0, 0], se.original.weight[7])
	assert torch.equal(out[0, 1], se.extra.weight[0])
	assert torch.equal(out[0, 2], se.original.weight[8])
	assert torch.equal(out[0, 3], se.extra.weight[3])


def test_split_embedding_gradient_isolation():
	# Only original ids -> only original gets gradient.
	se = _split_embedding()
	se(torch.tensor([[7, 8, 9]])).sum().backward()
	assert se.original.weight.grad.abs().sum() > 0
	assert se.extra.weight.grad is None or se.extra.weight.grad.abs().sum() == 0

	# Only extra ids -> only extra gets gradient.
	se = _split_embedding()
	se(torch.tensor([[16, 17]])).sum().backward()
	assert se.extra.weight.grad.abs().sum() > 0
	assert se.original.weight.grad is None or se.original.weight.grad.abs().sum() == 0


def test_split_embedding_no_extra_is_passthrough():
	se = _split_embedding(base=16, n_new=0, dim=5)
	assert se.extra is None
	ids = torch.tensor([[7, 8, 11]])
	assert torch.equal(se(ids), se.original(ids))


# --- AttentionPooling ----------------------------------------------------------

def test_attention_pooling_shape_and_gradient():
	pool = AttentionPooling(hidden_size=8, num_heads=2)
	seq = torch.randn(3, 6, 8, requires_grad=True)
	out = pool(seq)
	assert out.shape == (3, 8)
	out.sum().backward()
	assert pool.query.grad is not None and pool.query.grad.abs().sum() > 0
	assert seq.grad is not None


# --- Target transforms ---------------------------------------------------------

def test_log1p_transform_round_trip():
	y = torch.tensor([0.0, 1.0, 5.0, 100.0])
	t = transform_target("log1p", y)
	assert torch.allclose(t, torch.log1p(y))
	assert torch.allclose(inverse_transform("log1p", t), y, atol=1e-5)


def test_arcsin_sqrt_transform_round_trip_and_clamp():
	y = torch.tensor([0.0, 0.25, 0.5, 1.0])
	t = transform_target("arcsin_sqrt", y)
	assert t.min() >= 0.0 and t.max() <= math.pi / 2 + 1e-6
	assert torch.allclose(inverse_transform("arcsin_sqrt", t), y, atol=1e-5)
	# Out-of-range inputs are clamped, not NaN.
	assert torch.allclose(
		transform_target("arcsin_sqrt", torch.tensor([1.5])),
		torch.tensor([math.pi / 2]), atol=1e-5,
	)
	assert torch.allclose(
		inverse_transform("arcsin_sqrt", torch.tensor([2.0])),
		torch.tensor([1.0]), atol=1e-5,
	)


def test_none_transform_is_identity():
	y = torch.tensor([1.0, -2.0, 3.5])
	assert torch.equal(transform_target("none", y), y)
	assert torch.equal(inverse_transform("none", y), y)


# --- Uncertainty-weighted loss -------------------------------------------------

def test_uncertainty_weighting_reduces_to_sum_at_s_zero():
	raw = {"a": torch.tensor(2.0), "b": torch.tensor(3.0)}
	log_vars = {"a": torch.tensor(0.0), "b": torch.tensor(0.0)}
	total, comp = uncertainty_weighted_loss(raw, log_vars)
	assert torch.allclose(total, torch.tensor(5.0))
	assert torch.allclose(comp["a"]["weight"], torch.tensor(1.0))
	assert torch.allclose(comp["a"]["weighted"], torch.tensor(2.0))


def test_uncertainty_weighting_formula():
	raw = {"a": torch.tensor(2.0), "b": torch.tensor(3.0)}
	log_vars = {"a": torch.tensor(1.0), "b": torch.tensor(0.0)}
	total, comp = uncertainty_weighted_loss(raw, log_vars)
	expected = math.exp(-1.0) * 2.0 + 1.0 + math.exp(0.0) * 3.0 + 0.0
	assert torch.allclose(total, torch.tensor(expected), atol=1e-6)
	assert torch.allclose(comp["a"]["weight"], torch.tensor(math.exp(-1.0)), atol=1e-6)


# --- Fixed inverse-variance loss (MSE + Huber) ---------------------------------

def test_fiv_mse_matches_legacy_formula():
	preds = {"a": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([0.0, 1.0])}
	targets = {"a": torch.tensor([1.5, 1.0, 4.0]), "b": torch.tensor([0.5, 0.5])}
	norms = {"a": 4.0, "b": 0.25}
	total, comp = fixed_inverse_variance_loss(preds, targets, norms)

	mse_a = torch.mean((preds["a"] - targets["a"]) ** 2)
	mse_b = torch.mean((preds["b"] - targets["b"]) ** 2)
	expected = mse_a / norms["a"] + mse_b / norms["b"]
	assert torch.allclose(total, expected)
	# 'raw' must stay the plain per-task MSE so the val_fixed_loss monitor and
	# rawloss logging are unaffected by the loss_fn choice.
	assert torch.allclose(comp["a"]["raw"], mse_a)
	assert torch.allclose(comp["b"]["raw"], mse_b)
	assert comp["a"]["weight"] == pytest.approx(1.0 / norms["a"])


def test_fiv_huber_quadratic_region_matches_mse():
	# With the x2 factor, Huber's quadratic region equals z^2, so for residuals
	# fully within delta (in std-units) the Huber total equals the MSE total.
	norms = {"a": 4.0}  # sigma = 2
	preds = {"a": torch.tensor([1.0, -2.0, 0.5])}
	targets = {"a": torch.tensor([0.0, 0.0, 0.0])}  # |z| = |r|/2 <= 1 < delta=1.5
	mse_total, _ = fixed_inverse_variance_loss(preds, targets, norms, loss_fn="mse")
	huber_total, _ = fixed_inverse_variance_loss(
		preds, targets, norms, loss_fn="huber", huber_delta=1.5
	)
	assert torch.allclose(huber_total, mse_total, atol=1e-6)


def test_fiv_huber_linear_tail_is_smaller():
	# A residual well beyond delta gets linear (robust) treatment, so its
	# contribution is strictly less than the quadratic MSE term.
	norms = {"a": 1.0}
	preds = {"a": torch.tensor([10.0])}
	targets = {"a": torch.tensor([0.0])}
	mse_total, _ = fixed_inverse_variance_loss(preds, targets, norms, loss_fn="mse")
	huber_total, _ = fixed_inverse_variance_loss(
		preds, targets, norms, loss_fn="huber", huber_delta=1.0
	)
	# 2 * (delta * (|z| - 0.5 delta)) = 2 * (10 - 0.5) = 19 vs MSE 100.
	assert torch.allclose(huber_total, torch.tensor(19.0), atol=1e-6)
	assert float(huber_total) < float(mse_total)


def test_fiv_huber_norms_none_is_weight_one():
	# norms=None -> sigma=1, so Huber acts on the raw residual with weight 1.
	preds = {"a": torch.tensor([5.0])}
	targets = {"a": torch.tensor([0.0])}
	total, comp = fixed_inverse_variance_loss(
		preds, targets, None, loss_fn="huber", huber_delta=1.0
	)
	assert comp["a"]["weight"] == pytest.approx(1.0)
	assert torch.allclose(total, torch.tensor(9.0), atol=1e-6)  # 2*(5-0.5)


def test_fiv_huber_delta_is_live():
	# Guards the single-source-of-truth wiring: passing different huber_delta must
	# change the loss (regression test against a frozen/ignored delta).
	preds = {"a": torch.tensor([5.0])}
	targets = {"a": torch.tensor([0.0])}
	t1, _ = fixed_inverse_variance_loss(preds, targets, None, loss_fn="huber", huber_delta=1.0)
	t2, _ = fixed_inverse_variance_loss(preds, targets, None, loss_fn="huber", huber_delta=2.0)
	assert not torch.allclose(t1, t2)


# --- Monitor norms -------------------------------------------------------------

def test_compute_monitor_norms_matches_manual_variance():
	df = pd.DataFrame({
		"mode_copy_number": [0.0, 1.0, 5.0, 100.0],
		"heterozygosity": [0.0, 0.25, 0.5, 1.0],
	})
	transforms = resolve_transforms(list(DEFAULT_TARGETS.keys()))
	norms = compute_monitor_norms(df, DEFAULT_TARGETS, transforms)

	len_t = transform_target("log1p", torch.tensor(df["mode_copy_number"].to_numpy(), dtype=torch.float32))
	var_t = transform_target("arcsin_sqrt", torch.tensor(df["heterozygosity"].to_numpy(), dtype=torch.float32))
	assert norms["length"] == pytest.approx(float(torch.var(len_t, unbiased=False)), rel=1e-5)
	assert norms["variation"] == pytest.approx(float(torch.var(var_t, unbiased=False)), rel=1e-5)


def test_compute_monitor_norms_empty_df():
	df = pd.DataFrame({"mode_copy_number": [], "heterozygosity": []})
	transforms = resolve_transforms(list(DEFAULT_TARGETS.keys()))
	norms = compute_monitor_norms(df, DEFAULT_TARGETS, transforms)
	assert norms == {"length": 1.0, "variation": 1.0}


# --- Fake backbone (CPU stand-in so model wiring is testable without mamba) ----

class _FakeEmbeddings(nn.Module):
	def __init__(self, vocab, d):
		super().__init__()
		self.word_embeddings = nn.Embedding(vocab, d)


class _FakeMixer(nn.Module):
	def __init__(self, vocab, d):
		super().__init__()
		self.embeddings = _FakeEmbeddings(vocab, d)


class FakeCaduceus(nn.Module):
	""" Mimics the Caduceus base model interface STRLengthModel relies on:
	`.backbone.embeddings.word_embeddings` (swapped for SplitEmbedding) and a
	forward returning `.last_hidden_state`. The forward embeds via
	`word_embeddings` so the split embedding is exercised and gradients reach the
	new-token matrix; `proj` provides "backbone" params for the optimizer split.
	"""

	def __init__(self, vocab=16, d=256):
		super().__init__()
		self.backbone = _FakeMixer(vocab, d)
		self.proj = nn.Linear(d, d)

	def forward(self, input_ids=None, return_dict=True, **kwargs):
		emb = self.backbone.embeddings.word_embeddings(input_ids)
		return BaseModelOutputWithNoAttention(last_hidden_state=self.proj(emb))


class FakeAutoModel:
	@staticmethod
	def from_pretrained(checkpoint, config=None, trust_remote_code=False, **kwargs):
		d = config.d_model if config is not None else 256
		vocab = config.vocab_size if config is not None else 16
		return FakeCaduceus(vocab=vocab, d=d)


def _fake_model(monkeypatch, **overrides):
	monkeypatch.setattr(model_mod, "AutoModel", FakeAutoModel)
	config = {
		"n_prefix_prompt_tokens": 2,
		"n_str_prompt_tokens": 2,
		"n_suffix_prompt_tokens": 2,
		"head_hidden_layers": [8],
		"head_dropout": 0.0,
		"use_attention_pooling": True,
		"attention_pooling_num_heads": 2,
	}
	config.update(overrides)
	return create_model(config)


# Valid ids: DNA in [7, 11], learnable in [16, 16 + 6).
_FAKE_INPUT_IDS = torch.tensor([
	[7, 8, 9, 16, 17, 10, 11, 18, 19, 20, 21],
	[10, 9, 8, 16, 17, 7, 11, 18, 19, 20, 21],
])


def test_model_wiring_with_fake_backbone(monkeypatch):
	model = _fake_model(monkeypatch)
	model.eval()

	# forward -> one prediction per target.
	out = model(_FAKE_INPUT_IDS)
	assert set(out.keys()) == {"length", "variation"}
	assert out["length"].shape == (2,) and out["variation"].shape == (2,)

	# predict_step applies the inverse transforms and passes id/rev_comp/labels.
	batch = {
		"input_ids": _FAKE_INPUT_IDS,
		"length": torch.tensor([5.0, 6.0]),
		"variation": torch.tensor([0.25, 0.5]),
		"id": ["STR_a", "STR_b"],
		"rev_comp": torch.tensor([False, True]),
	}
	pred = model.predict_step(batch, 0)
	assert set(pred.keys()) == {
		"id", "rev_comp", "pred_length", "label_length",
		"pred_variation", "label_variation",
	}
	preds_t = model(_FAKE_INPUT_IDS)
	assert torch.allclose(pred["pred_length"], inverse_transform("log1p", preds_t["length"]))
	assert torch.allclose(pred["pred_variation"], inverse_transform("arcsin_sqrt", preds_t["variation"]))

	# Optimizer: 3 groups (heads/pooling/extra | backbone | uncertainty), with the
	# new-token embedding in the high-LR group and the backbone in its own group.
	groups = model.configure_optimizers()["optimizer"].param_groups
	assert len(groups) == 3
	high_ids = {id(p) for p in groups[0]["params"]}
	backbone_ids = {id(p) for p in groups[1]["params"]}
	extra_w = model.caduceus.backbone.embeddings.word_embeddings.extra.weight
	assert id(extra_w) in high_ids
	assert id(model.caduceus.proj.weight) in backbone_ids
	assert groups[0]["lr"] == pytest.approx(1e-4)
	assert groups[1]["lr"] == pytest.approx(1e-5)
	assert groups[2]["weight_decay"] == 0.0


def test_single_objective_model_has_no_uncertainty_group(monkeypatch):
	model = _fake_model(monkeypatch, targets={"length": "mode_copy_number"})
	assert model.use_uncertainty is False
	assert model.log_vars is None
	out = model(_FAKE_INPUT_IDS)
	assert set(out.keys()) == {"length"}
	groups = model.configure_optimizers()["optimizer"].param_groups
	assert len(groups) == 2  # high-LR + backbone, no uncertainty group


def test_fiv_huber_model_keeps_raw_losses_mse(monkeypatch):
	# In the FIV+Huber path the per-task raw_losses (which feed val_fixed_loss)
	# must stay MSE, while the training total uses the Huber objective.
	monitor_norm = {"length": 2.0, "variation": 0.3}
	model = _fake_model(
		monkeypatch,
		loss_weighting="fixed_inverse_variance",
		fiv_loss="huber",
		huber_delta=1.0,
		monitor_norm=monitor_norm,
	)
	model.eval()
	batch = {
		"input_ids": _FAKE_INPUT_IDS,
		"length": torch.tensor([5.0, 60.0]),       # large residual -> Huber tail
		"variation": torch.tensor([0.25, 0.5]),
	}
	with torch.no_grad():
		total, raw_losses, components, _ = model._common_step(batch)
		preds_t = model(_FAKE_INPUT_IDS)

	mse = nn.MSELoss()
	mse_fiv_total = 0.0
	for name in model.task_names:
		target_t = transform_target(model.transforms[name], batch[name])
		expected_mse = mse(preds_t[name], target_t)
		assert torch.allclose(raw_losses[name], expected_mse)        # monitor unchanged
		assert torch.allclose(components[name]["raw"], expected_mse)
		mse_fiv_total = mse_fiv_total + expected_mse / monitor_norm[name]
	# Huber objective differs from the plain MSE-FIV total given the tail residual.
	assert not torch.allclose(total, mse_fiv_total)


def test_fiv_loss_validation(monkeypatch):
	with pytest.raises(ValueError):
		_fake_model(monkeypatch, loss_weighting="fixed_inverse_variance", fiv_loss="bogus")
	with pytest.raises(ValueError):
		_fake_model(monkeypatch, loss_weighting="fixed_inverse_variance",
					fiv_loss="huber", huber_delta=0.0)
	# Huber requested outside the FIV path has no effect -> warn.
	with pytest.warns(UserWarning):
		_fake_model(monkeypatch, loss_weighting="uncertainty", fiv_loss="huber")


# --- End-to-end short fit on CPU (fake backbone + real DataModule) -------------

def _write_train_val_fixtures(tmp_path):
	from seq_models.caduceus.test_data import CHROM_SEQ, STR_END, STR_START

	ref_path = tmp_path / "tv.fa"
	ref_path.write_text(f">chr1\n{CHROM_SEQ}\n")

	rows = []
	splits = ["train", "train", "train", "train", "val", "val"]
	for i, split in enumerate(splits):
		for rc in (False, True):
			rows.append({
				"ID": f"S{i}", "chrom": "chr1",
				"str_start": STR_START, "str_end": STR_END, "motif": "AT",
				"mode_copy_number": 5.0 + i, "heterozygosity": 0.2 + 0.05 * i,
				"split": split, "rev_comp": rc,
			})
	data_path = tmp_path / "tv.tsv"
	pd.DataFrame(rows).to_csv(data_path, sep="\t", index=False)
	return str(ref_path), str(data_path)


def test_end_to_end_short_fit_with_fake_backbone(tmp_path, monkeypatch):
	from seq_models.caduceus.test_data import N_FLANKING_BP, N_STR_BP

	ref_path, data_path = _write_train_val_fixtures(tmp_path)
	# Learnable-token counts must match between data and model.
	n_prefix, n_str, n_suffix = 2, 3, 4
	dm = create_data_module({
		"data_path": data_path,
		"ref_path": ref_path,
		"batch_size": 2,
		"n_flanking_bp": N_FLANKING_BP,
		"n_str_bp": N_STR_BP,
		"n_prefix_prompt_tokens": n_prefix,
		"n_str_prompt_tokens": n_str,
		"n_suffix_prompt_tokens": n_suffix,
		"dataloader_num_workers": 0,
	})
	try:
		dm.setup()
	except Exception as exc:  # tokenizer unavailable -> skip
		pytest.skip(f"Could not load Caduceus tokenizer: {exc}")

	model = _fake_model(
		monkeypatch,
		n_prefix_prompt_tokens=n_prefix,
		n_str_prompt_tokens=n_str,
		n_suffix_prompt_tokens=n_suffix,
	)

	ckpt = ModelCheckpoint(
		dirpath=str(tmp_path / "ckpts"), monitor="val_fixed_loss",
		mode="min", save_top_k=1,
	)
	trainer = pl.Trainer(
		max_epochs=1, limit_train_batches=2, limit_val_batches=2,
		accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
		callbacks=[ckpt, EarlyStopping(monitor="val_fixed_loss", mode="min", patience=1)],
	)
	trainer.fit(model, datamodule=dm)

	# The monitored metric must actually be logged, and a per-task native metric
	# present; a non-empty best_model_path proves the monitor resolved.
	assert "val_fixed_loss" in trainer.callback_metrics
	assert any(k.startswith("val_length_") for k in trainer.callback_metrics)
	assert ckpt.best_model_path != ""
	# on_fit_start computed and froze the monitor norms from the val split.
	assert set(model.monitor_norm.keys()) == {"length", "variation"}


# --- Real backbone (skip-guarded; runs only where mamba kernels exist) ---------

def test_real_backbone_forward_shapes():
	# Caduceus uses Triton/Mamba CUDA kernels, so this only runs on a GPU.
	if not torch.cuda.is_available():
		pytest.skip("Caduceus backbone needs a CUDA GPU (Triton/Mamba kernels).")
	try:
		model = create_model({
			"n_prefix_prompt_tokens": 2,
			"n_str_prompt_tokens": 2,
			"n_suffix_prompt_tokens": 2,
			"head_hidden_layers": [8],
			"head_dropout": 0.0,
		})
	except Exception as exc:  # mamba_ssm / backbone unavailable -> skip
		pytest.skip(f"Caduceus backbone unavailable: {exc}")

	model = model.to("cuda").eval()
	out = model(_FAKE_INPUT_IDS.to("cuda"))
	assert set(out.keys()) == {"length", "variation"}
	assert out["length"].shape == (2,)
