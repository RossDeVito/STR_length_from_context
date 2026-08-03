"""Compare Captum IG method / n_steps convergence across a small eval sweep.

Loads every run under SWEEP_DIR that has both meta.json and convergence.json
(see scripts/ig/configs/method_eval/ for the configs that produce these runs
on a fixed small loci subsample), reading each run's method/n_steps back out
of its own meta.json rather than hand-maintaining a list. Prints a summary
table and plots, per task, mean relative convergence delta and the fraction
of samples still above the 5% relative-delta threshold, both vs n_steps, one
line per method -- the two methods' convergence curves are directly
comparable and the diminishing-returns elbow is visible.
"""

import glob
import json
import os

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

# Which STR motif length's method_eval sweep to compare -- must match a
# scripts/ig/configs/method_eval/str{N}_*.yaml prefix / output subdirectory.
STR_MOTIF_LEN = "str3"

SWEEP_DIR = f"output/caduceus_v0/method_eval/{STR_MOTIF_LEN}"

# Figure
FIG_WIDTH = 12
FIG_HEIGHT_PER_ROW = 4.5
SAVE_PATH = None  # set to a filepath to save instead of display

# Fixed categorical color assignment (method -> color), in the project
# palette's slot order -- never cycled/reassigned by sort order. Add new
# methods here (next unused slot) rather than letting them fall back to grey.
METHOD_COLORS = {
	"riemann_trapezoid": "#2a78d6",  # categorical slot 1 (blue)
	"gausslegendre": "#008300",      # categorical slot 2 (green)
}
FALLBACK_COLOR = "#898781"  # muted ink, for any method not in METHOD_COLORS

# Human-readable display name for a task, used in the printed table and plot
# titles only (task keys themselves stay as meta.json's task_names, e.g.
# "variation", since those are what index into convergence.json).
TASK_DISPLAY_NAMES = {
	"variation": "heterozygosity",
}

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# ============================================================================
# Discover runs
# ============================================================================

run_dirs = sorted(
	d for d in glob.glob(os.path.join(SWEEP_DIR, "*"))
	if os.path.isfile(os.path.join(d, "meta.json"))
	and os.path.isfile(os.path.join(d, "convergence.json"))
)

if not run_dirs:
	raise ValueError(
		f"No runs with meta.json + convergence.json found under {SWEEP_DIR}"
	)

print(f"Found {len(run_dirs)} runs under {SWEEP_DIR}")

# ============================================================================
# Load per-run, per-task records
# ============================================================================

records = []

for run_dir in run_dirs:
	with open(os.path.join(run_dir, "meta.json"), "r") as f:
		meta = json.load(f)
	with open(os.path.join(run_dir, "convergence.json"), "r") as f:
		conv = json.load(f)

	ig_cfg = meta["ig_config"]
	method = ig_cfg["method"]
	n_steps = ig_cfg["n_steps"]

	for task in meta["task_names"]:
		rel = conv[task]["relative_delta"]
		records.append({
			"task": TASK_DISPLAY_NAMES.get(task, task),
			"method": method,
			"n_steps": n_steps,
			"n_samples": meta["n_samples"],
			"mean_rel_delta_pct": rel["mean"] * 100,
			"median_rel_delta_pct": rel["median"] * 100,
			"pct_above_5pct": rel["pct_above_5pct"],
			"pct_above_1pct": rel["pct_above_1pct"],
			"run_dir": run_dir,
		})

df = (
	pd.DataFrame(records)
	.sort_values(["task", "method", "n_steps"])
	.reset_index(drop=True)
)

# ============================================================================
# Print summary table
# ============================================================================

print(f"\n{'='*100}")
print(df.to_string(
	index=False,
	columns=[
		"task", "method", "n_steps", "n_samples",
		"mean_rel_delta_pct", "median_rel_delta_pct",
		"pct_above_5pct", "pct_above_1pct",
	],
	float_format=lambda v: f"{v:.3f}",
))

# ============================================================================
# Plot: mean relative delta and pct_above_5pct vs n_steps, per task, one line
# per method
# ============================================================================

tasks = sorted(df["task"].unique())
method_order = list(METHOD_COLORS.keys())
methods = sorted(
	df["method"].unique(),
	key=lambda m: method_order.index(m) if m in method_order else len(method_order),
)

fig, axes = plt.subplots(
	len(tasks), 2,
	figsize=(FIG_WIDTH, FIG_HEIGHT_PER_ROW * len(tasks)),
	squeeze=False,
)
fig.patch.set_facecolor(SURFACE)

panels = [
	("mean_rel_delta_pct", "Mean relative delta (%)", "Convergence vs n_steps"),
	("pct_above_5pct", "% samples > 5% rel. delta",
	 "Non-converged fraction vs n_steps"),
]

for row, task in enumerate(tasks):
	task_df = df[df["task"] == task]

	for col, (metric, ylabel, title_suffix) in enumerate(panels):
		ax = axes[row, col]
		ax.set_facecolor(SURFACE)

		for method in methods:
			m_df = task_df[task_df["method"] == method].sort_values("n_steps")
			if m_df.empty:
				continue
			ax.plot(
				m_df["n_steps"], m_df[metric],
				color=METHOD_COLORS.get(method, FALLBACK_COLOR),
				linewidth=2, marker="o", markersize=8,
				label=method,
			)

		ax.set_xlabel("n_steps", color=INK_MUTED)
		ax.set_ylabel(ylabel, color=INK_MUTED)
		ax.set_title(
			f"[{STR_MOTIF_LEN}/{task}] {title_suffix}",
			color=INK_PRIMARY, fontsize=11,
		)
		ax.grid(True, color=GRIDLINE, linewidth=0.7)
		for spine in ax.spines.values():
			spine.set_color(AXIS_LINE)
		ax.tick_params(colors=INK_MUTED)
		ax.set_ylim(bottom=0)
		ax.legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY)

plt.tight_layout()

if SAVE_PATH:
	fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
	print(f"\nSaved to {SAVE_PATH}")
else:
	plt.show()
