"""Remove trio children from HipSTR sample union; report pop counts.

Reads:
  - Sample union file (one sample per line) from step 1.
  - 1000G 3202-sample pedigree+population file from step 2 with columns:
        FamilyID  SampleID  FatherID  MotherID  Sex  Population  Superpopulation

A "trio child" is any sample whose FatherID or MotherID is not "0", per
standard PED convention. Parents (founders) are kept.

Outputs (to --out-dir):
  - filtered_samples.txt      : Samples after trio-child removal.
  - removed_children.txt      : Samples removed because they are trio children.
  - missing_in_ped.txt        : Samples with no row in the ped file (kept,
                                but flagged for inspection).
  - population_counts.tsv     : Per-population counts after filtering.
  - superpopulation_counts.tsv: Per-superpopulation counts after filtering.
  - sex_counts.tsv            : Male / female counts after filtering.

CLI:
  --samples-file PATH  Sample union file from step 1.
  --ped-file PATH      1000G ped+pop file from step 2.
  --out-dir DIR        Output directory (created if missing).
"""

import os
import argparse
import pandas as pd


def parse_args():
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--samples-file", required=True)
	p.add_argument("--ped-file", required=True)
	p.add_argument("--out-dir", required=True)
	return p.parse_args()


def main():
	args = parse_args()
	os.makedirs(args.out_dir, exist_ok=True)

	# --- Load HipSTR sample union ---
	with open(args.samples_file) as f:
		samples = sorted({s.strip() for s in f if s.strip()})
	print(f"HipSTR sample union: {len(samples)} samples")

	# --- Load pedigree + population table ---
	ped = pd.read_csv(args.ped_file, sep=r"\s+", engine="python")
	ped.columns = [c.strip() for c in ped.columns]

	expected = {"SampleID", "FatherID", "MotherID", "Sex",
	            "Population", "Superpopulation"}
	missing = expected - set(ped.columns)
	if missing:
		raise ValueError(
			f"Missing expected columns in {args.ped_file}: {missing}\n"
			f"Got columns: {list(ped.columns)}\n"
			f"If your file uses different headers, edit this script."
		)

	# --- Cross-reference ---
	ped_in_data = ped[ped["SampleID"].isin(samples)].copy()
	missing_in_ped = sorted(set(samples) - set(ped["SampleID"]))
	print(f"Samples found in ped file: {len(ped_in_data)} / {len(samples)}")
	print(f"Samples missing from ped:  {len(missing_in_ped)}")

	with open(os.path.join(args.out_dir, "missing_in_ped.txt"), "w") as f:
		for s in missing_in_ped:
			f.write(s + "\n")

	# --- Identify trio children ---
	is_child = (
		(ped_in_data["FatherID"].astype(str) != "0")
		| (ped_in_data["MotherID"].astype(str) != "0")
	)
	children = sorted(ped_in_data.loc[is_child, "SampleID"].tolist())
	print(f"Trio children to remove:   {len(children)}")

	with open(os.path.join(args.out_dir, "removed_children.txt"), "w") as f:
		for s in children:
			f.write(s + "\n")

	# --- Filtered samples ---
	kept = ped_in_data.loc[~is_child].copy()
	filtered = sorted(kept["SampleID"].tolist())
	print(f"Filtered samples (kept):   {len(filtered)}")

	with open(os.path.join(args.out_dir, "filtered_samples.txt"), "w") as f:
		for s in filtered:
			f.write(s + "\n")

	# --- Population / superpopulation / sex counts ---
	pop_counts = (
		kept["Population"]
		.value_counts()
		.rename_axis("Population")
		.reset_index(name="count")
		.sort_values("Population")
	)
	super_counts = (
		kept["Superpopulation"]
		.value_counts()
		.rename_axis("Superpopulation")
		.reset_index(name="count")
		.sort_values("Superpopulation")
	)

	# Sex in 1000G PED is coded 1=male, 2=female. Map to labels for clarity
	# but keep "Unknown" for anything else (e.g. 0 or missing).
	sex_label_map = {1: "male", 2: "female", "1": "male", "2": "female"}
	kept["SexLabel"] = kept["Sex"].map(sex_label_map).fillna("unknown")
	sex_counts = (
		kept["SexLabel"]
		.value_counts()
		.rename_axis("Sex")
		.reset_index(name="count")
		.sort_values("Sex")
	)

	pop_counts.to_csv(
		os.path.join(args.out_dir, "population_counts.tsv"),
		sep="\t", index=False,
	)
	super_counts.to_csv(
		os.path.join(args.out_dir, "superpopulation_counts.tsv"),
		sep="\t", index=False,
	)
	sex_counts.to_csv(
		os.path.join(args.out_dir, "sex_counts.tsv"),
		sep="\t", index=False,
	)

	print("\nSex counts (post-filter):")
	print(sex_counts.to_string(index=False))
	print("\nSuperpopulation counts (post-filter):")
	print(super_counts.to_string(index=False))
	print("\nPopulation counts (post-filter):")
	print(pop_counts.to_string(index=False))


if __name__ == "__main__":
	main()