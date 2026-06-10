"""Tests for seq_models/caduceus/predict.py (CPU, no backbone).

Run with: pytest seq_models/caduceus/test_predict.py -v
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(
	os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
	sys.path.insert(0, _REPO_ROOT)

from seq_models.caduceus.predict import aggregate_predictions


def test_aggregate_predictions_flattens_and_averages_rc_pairs():
	# Two batches = the forward and reverse-complement orientations of loci A, B.
	predictions = [
		{
			"id": ["A", "B"],
			"rev_comp": torch.tensor([False, False]),
			"pred_length": torch.tensor([4.0, 6.0]),
			"label_length": torch.tensor([5.0, 7.0]),
			"pred_variation": torch.tensor([0.2, 0.4]),
			"label_variation": torch.tensor([0.25, 0.45]),
		},
		{
			"id": ["A", "B"],
			"rev_comp": torch.tensor([True, True]),
			"pred_length": torch.tensor([6.0, 8.0]),
			"label_length": torch.tensor([5.0, 7.0]),
			"pred_variation": torch.tensor([0.4, 0.6]),
			"label_variation": torch.tensor([0.25, 0.45]),
		},
	]

	per_orientation, per_locus = aggregate_predictions(
		predictions, ["length", "variation"]
	)

	# Per-orientation: one row per sample, both orientations present.
	assert len(per_orientation) == 4
	assert set(per_orientation.columns) == {
		"id", "rev_comp", "pred_length", "true_length",
		"pred_variation", "true_variation",
	}
	assert sorted(per_orientation["rev_comp"].tolist()) == [False, False, True, True]

	# Per-locus: one row per id with predictions averaged across orientations.
	assert len(per_locus) == 2
	row_a = per_locus.set_index("id").loc["A"]
	row_b = per_locus.set_index("id").loc["B"]
	assert row_a["pred_length"] == pytest.approx(5.0)   # mean(4, 6)
	assert row_b["pred_length"] == pytest.approx(7.0)   # mean(6, 8)
	assert row_a["pred_variation"] == pytest.approx(0.3)  # mean(0.2, 0.4)
	# Labels carried through unchanged.
	assert row_a["true_length"] == pytest.approx(5.0)
	assert row_b["true_variation"] == pytest.approx(0.45)


def test_aggregate_predictions_single_task():
	predictions = [{
		"id": ["A", "A"],
		"rev_comp": torch.tensor([False, True]),
		"pred_length": torch.tensor([3.0, 5.0]),
		"label_length": torch.tensor([4.0, 4.0]),
	}]
	per_orientation, per_locus = aggregate_predictions(predictions, ["length"])
	assert len(per_orientation) == 2
	assert len(per_locus) == 1
	assert per_locus.iloc[0]["pred_length"] == pytest.approx(4.0)  # mean(3, 5)
	assert per_locus.iloc[0]["true_length"] == pytest.approx(4.0)
