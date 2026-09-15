"""Shared config-loading and per-annotation-family loader helpers for
scripts in scripts/annotation_analysis/.

Annotations are described in annotation_config.yaml (see that file for the
field schema) and come in two flavors:
    - bigwig: a single continuous value per position (e.g. phyloP).
    - bed: discrete segments, optionally requiring a join to a second
      metadata file to reach the value of interest (e.g. ChromHMM, where the
      bed's segment_name must be parsed into a mnemonic and merged against
      state_annotations_processed.csv to recover `state` / `Group`).

load_annotation() is the main entry point: given an annotation type name
from the config, it returns either an open pyBigWig handle (bigwig entries)
or a loaded/joined pandas DataFrame (bed entries). Family-specific loading
quirks (column names, join-key derivation) live here as real code rather
than as free-text in the config.
"""

import os

import pandas as pd
import pyBigWig
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "annotation_config.yaml")


def load_config(path=None):
	path = path or DEFAULT_CONFIG_PATH
	with open(path) as f:
		return yaml.safe_load(f)


def resolve_path(rel_path):
	return os.path.join(SCRIPT_DIR, rel_path)


def get_entry(annotation_type, config=None):
	config = config if config is not None else load_config()
	if annotation_type not in config:
		raise KeyError(
			f"Unknown annotation type {annotation_type!r}. "
			f"Available: {sorted(config.keys())}"
		)
	return config[annotation_type]


def load_bed(entry, max_rows=None):
	"""Load a header-less bed(.gz) file using the column names declared in
	the config entry."""
	return pd.read_csv(
		resolve_path(entry["path"]), sep="\t", header=None,
		names=entry["columns"], compression="infer", nrows=max_rows,
	)


def load_bigwig(entry):
	return pyBigWig.open(resolve_path(entry["path"]))


def load_chromhmm_bed(entry, max_rows=None):
	"""Load the ChromHMM bed segments and derive the mnemonic join key.

	Segment names are "{ChromHMM state number}_{mnemonic}", but that leading
	number is the model's internal state order -- it does NOT match the
	`state` column in state_annotations_processed.csv (which uses a
	different, group-sorted numbering). Only the mnemonic is a stable join
	key between the two files, so pull it out here. Mirrors
	load_chromhmm_segments() in plot_labels_by_chromhmm.py.
	"""
	df = load_bed(entry, max_rows=max_rows)
	df["mneumonics"] = df["segment_name"].str.split("_", n=1).str[1]
	return df


def load_chromhmm_join(entry):
	return pd.read_csv(resolve_path(entry["join_path"]))


def load_chromhmm(entry, max_rows=None):
	"""Load and merge the ChromHMM bed + state metadata csv, returning a
	DataFrame with chrom/start/end/segment_name/mneumonics plus every
	column from state_annotations_processed.csv (state, Group, ...)."""
	bed_df = load_chromhmm_bed(entry, max_rows=max_rows)
	join_df = load_chromhmm_join(entry)
	return bed_df.merge(join_df, on="mneumonics", how="left")


def load_annotation(annotation_type, config=None, max_rows=None):
	"""Load an annotation by its config entry name. Returns an open
	pyBigWig handle for bigwig entries (caller is responsible for closing
	it), or a DataFrame for bed entries.

	Dispatch is by file_type + presence of join_path rather than a generic
	string-keyed registry: there are only two annotation families today
	(plain bigwig, and ChromHMM-style joined bed). A genuinely new joined-bed
	family later needs one more small loader function here plus a branch
	below, not a declarative join engine.
	"""
	entry = get_entry(annotation_type, config=config)
	if entry["file_type"] == "bigwig":
		return load_bigwig(entry)
	if entry["file_type"] == "bed":
		if "join_path" in entry:
			return load_chromhmm(entry, max_rows=max_rows)
		return load_bed(entry, max_rows=max_rows)
	raise ValueError(f"Unknown file_type {entry['file_type']!r} for {annotation_type!r}")
