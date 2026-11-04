"""Render publication-ready static figures from a complete audited HR PoC.

Uses the existing main virtual environment (matplotlib/pandas). No experiment is
run and no uncertainty is recomputed here: plotted intervals are the participant
bootstrap results already published by summarize_ppg_hr_poc.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("AnyPPG", "PaPaGei-S", "PulsePPG", "SmallCNN", "MOMENT-1-small", "welch", "elgendi", "TrainMean")
FOUNDATIONS = METHODS[:3]
TRADITIONAL = ("welch", "elgendi")
LABELS = {
    "AnyPPG": "AnyPPG", "PaPaGei-S": "PaPaGei-S", "PulsePPG": "PulsePPG",
    "SmallCNN": "Small CNN", "MOMENT-1-small": "MOMENT-1-small",
    "welch": "Welch PSD \u2020", "elgendi": "Elgendi peaks \u2020", "TrainMean": "Training mean",
}
COLORS = dict(zip(METHODS, ("#0072B2", "#D55E00", "#009E73", "#202020", "#CC79A7", "#56B4E9", "#E69F00", "#777777")))
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D", "P", "v", "X", "x")))
STYLES = dict(zip(METHODS, ("-", "-", "-", "-", "--", "--", ":", ":")))
SCENARIOS = (("within", "Dalia", "Dalia"), ("within", "WildPPG", "WildPPG"), ("transfer", "Dalia", "WildPPG"))
MAE = "conditional_participant_mae_bpm"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tables(directory):
    """Refuse incomplete or altered summaries before creating any figures."""
    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    require(audit.get("status") == "complete", "Figures require a complete audited study; partial results are not publishable")
    require(audit.get("completed_learned_results") == audit.get("expected_learned_results") == 558,
            "Expected all 558 learned results before plotting")
    require(not audit.get("missing_artifacts"), "Audit still reports missing artifacts")
    tables = {}
    for name in ("summary", "strata", "paired"):
        filename = name + ".csv"
        require(digest(directory / filename) == audit["table_sha256"].get(filename), f"Audited table hash mismatch: {filename}")
        table = pd.read_csv(directory / filename, dtype={"budget": str, "level": str})
        require(not table.empty and "study_complete" in table, f"Missing completed-study rows: {filename}")
        flags = table["study_complete"].astype(str).str.strip().str.lower()
        require(flags.eq("true").all(), f"study_complete=false or invalid in {filename}")
        require(table["min_seeds_per_subject"].eq(3).all(), f"Incomplete seed coverage: {filename}")
        tables[name] = table
    return tables, audit


def select(table, scenario, budget="full", **filters):
    kind, source, target = scenario
    mask = table["scenario"].eq(kind) & table["source"].eq(source) & table["target"].eq(target) & table["budget"].eq(budget)
    for key, value in filters.items():
        mask &= table[key].eq(value)
    return table.loc[mask]


def one(table, scenario, method, budget="full", **filters):
    rows = select(table, scenario, budget, method=method, **filters)
    require(len(rows) == 1, f"Expected one row for {scenario}/{budget}/{method}/{filters}")
    row = rows.iloc[0]
    require(np.isfinite(row[MAE]) and row[MAE] >= 0, "A plotted MAE is unavailable or invalid")
    if method not in TRADITIONAL:
        require(np.isclose(row["window_coverage"], 1.0, atol=1e-12, rtol=0), "Learned-model coverage must be complete")
    return row


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 2,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
        "svg.hashsalt": "watch-lsm-wrist-hr-poc-v1",
    })


def line_style(method):
    return {"color": COLORS[method], "marker": MARKERS[method], "linestyle": STYLES[method],
            "markersize": 6, "linewidth": 1.9, "label": LABELS[method]}


def legend(fig, y):
    handles = [Line2D([], [], **line_style(method)) for method in METHODS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, y), ncol=4,
               frameon=False, fontsize=10, handlelength=2.8, columnspacing=2.0)


def finish(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = output / f"{name}.{extension}"
        metadata = {"Date": None} if extension == "svg" else None
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12, metadata=metadata)
        if extension == "svg":
            # Matplotlib emits trailing spaces inside path attributes. Keep the
            # same SVG geometry with clean, platform-independent text output.
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
        print(path.resolve())
    plt.close(fig)


def plot_label_efficiency(summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.1), gridspec_kw={"width_ratios": [1, 1, 1.12]})
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.35, top=0.81, wspace=0.48)
    fig.suptitle("Heart-rate accuracy and label efficiency", y=0.98, fontsize=16, fontweight="semibold")
    fig.text(0.5, 0.925, "Equal participant weighting; errors averaged over three seeds within each participant", ha="center", fontsize=11)
    within_max = 0.0
    for ax, scenario, title, full in zip(axes[:2], SCENARIOS[:2], ("A  DaLiA (n = 15)", "B  WildPPG (n = 16)"), ("Full\n(12)", "Full\n(12\u201313)")):
        for method in METHODS:
            values = [one(summary, scenario, method, budget)[MAE] for budget in ("3", "6", "full")]
            within_max = max(within_max, max(values))
            ax.plot(range(3), values, **line_style(method))
        ax.set_title(title, loc="left", pad=12, fontweight="semibold")
        ax.set_xticks(range(3), ("3", "6", full))
        ax.set_xlabel("Labelled source participants\n(including validation)")
        ax.set_ylabel("Participant macro MAE (bpm)")
        ax.set_xlim(-0.12, 2.12)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
    for ax in axes[:2]:
        ax.set_ylim(0, within_max * 1.12)
    ax = axes[2]
    for index, method in enumerate(METHODS):
        row = one(summary, SCENARIOS[2], method)
        ax.scatter(row[MAE], index, s=50, color=COLORS[method], marker=MARKERS[method], zorder=3)
    ax.set_title("C  DaLiA \u2192 WildPPG", loc="left", pad=12, fontweight="semibold")
    ax.set_yticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Participant macro MAE (bpm)\nSource: 12 train + 3 validation")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    legend(fig, 0.155)
    coverages = []
    for method in TRADITIONAL:
        values = [one(summary, scenario, method)["window_coverage"] * 100 for scenario in SCENARIOS[:2]]
        label = LABELS[method].removesuffix(" \u2020")
        coverages.append(f"{label}: {min(values):.1f}\u2013{max(values):.1f}%")
    fig.text(0.5, 0.045,
             "\u2020 Traditional MAE is conditional on finite estimates; available-window coverage: " + "; ".join(coverages) + ".\n"
             "Traditional outputs are reused across label budgets and transfer. Points/curves are descriptive; paired uncertainty appears in Figure 2.",
             ha="center", va="center", fontsize=9.5, linespacing=1.5)
    finish(fig, output, "hr_label_efficiency")


def plot_paired_differences(paired, output):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.32, top=0.77, wspace=0.18)
    fig.suptitle("PPG foundation models versus the small CNN", y=0.98, fontsize=16, fontweight="semibold")
    fig.text(0.5, 0.915, "Full label budgets; participant-paired difference = foundation-model MAE \u2212 CNN MAE", ha="center", fontsize=11)
    bounds = [-1., 0.]
    for ax, scenario, title in zip(axes, SCENARIOS, ("A  DaLiA", "B  WildPPG", "C  DaLiA \u2192 WildPPG")):
        n_subjects = set()
        for index, method in enumerate(FOUNDATIONS):
            rows = select(paired, scenario, foundation=method, control="SmallCNN")
            require(len(rows) == 1, f"Missing or duplicate paired contrast: {scenario}/{method}")
            row = rows.iloc[0]
            mean, low, high = [float(row[key]) for key in ("difference_bpm", "ci95_low_bpm", "ci95_high_bpm")]
            require(np.isfinite([mean, low, high]).all() and low <= high, "Invalid paired confidence interval")
            require(row["participant_bootstrap_samples"] == 10000, "Expected 10,000 participant bootstrap samples")
            require(row["min_seeds_with_pair"] == 3 and row["n_subjects"] == row["n_subjects_total"], "Incomplete participant pairs")
            require(np.isclose(row["paired_window_coverage"], 1., atol=1e-12, rtol=0), "FM/CNN comparison must cover all reference windows")
            n_subjects.add(int(row["n_subjects"]))
            bounds.extend([low, high, mean])
            ax.plot([low, high], [index, index], color=COLORS[method], linewidth=2)
            ax.plot([low, low], [index - 0.07, index + 0.07], color=COLORS[method], linewidth=1.3)
            ax.plot([high, high], [index - 0.07, index + 0.07], color=COLORS[method], linewidth=1.3)
            ax.scatter(mean, index, color=COLORS[method], marker=MARKERS[method], s=55, zorder=3)
        require(len(n_subjects) == 1, "Paired methods use different participant populations")
        ax.set_title(f"{title}\n(n = {n_subjects.pop()} participants)", loc="left", fontsize=11, pad=12, fontweight="semibold")
        ax.axvline(0, color="#555555", linewidth=1.2, zorder=0)
        ax.axvline(-1, color="#555555", linewidth=1.1, linestyle="--", zorder=0)
        ax.set_xlabel("Paired MAE difference (bpm)")
        ax.set_yticks(range(3), FOUNDATIONS)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    padding = max(0.3, (max(bounds) - min(bounds)) * 0.08)
    for ax in axes:
        ax.set_xlim(min(bounds) - padding, max(bounds) + padding)
        ax.set_ylim(2.55, -0.55)
    fig.legend(handles=[Line2D([], [], color="#555555", label="0 bpm: parity"),
                        Line2D([], [], color="#555555", linestyle="--", label="\u22121 bpm: prespecified practical margin")],
               loc="center", bbox_to_anchor=(0.5, 0.18), ncol=2, frameon=False, fontsize=10)
    fig.text(0.5, 0.06,
             "Negative differences favour the foundation model. Bars: 95% participant bootstrap intervals (10,000 resamples).\n"
             "Seeds are averaged within participants before resampling. Exploratory intervals; no multiplicity adjustment.",
             ha="center", va="center", fontsize=9.5, linespacing=1.5)
    finish(fig, output, "hr_paired_cnn_differences")


def plot_motion_strata(strata, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.31, top=0.80, wspace=0.15)
    fig.suptitle("Heart-rate accuracy across movement strata", y=0.98, fontsize=16, fontweight="semibold")
    fig.text(0.5, 0.92, "Full label budgets; within-dataset ACC-motion terciles are movement proxies, not noise labels", ha="center", fontsize=10.5)
    ymax = 0.
    for ax, scenario, title in zip(axes, SCENARIOS[:2], ("A  DaLiA", "B  WildPPG")):
        for method in METHODS:
            values = [one(strata, scenario, method, stratum="motion_bin", level=str(level))[MAE] for level in range(3)]
            ymax = max(ymax, max(values))
            ax.plot(range(3), values, **line_style(method))
        ax.set_title(title, loc="left", pad=12, fontweight="semibold")
        ax.set_xticks(range(3), ("Low", "Middle", "High"))
        ax.set_xlabel("Within-dataset ACC-motion tercile")
        ax.set_xlim(-0.1, 2.1)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Participant macro MAE (bpm)")
    axes[0].set_ylim(0, ymax * 1.12)
    legend(fig, 0.16)
    fig.text(0.5, 0.045,
             "ACC is used only for descriptive stratification, never as model input; tercile thresholds differ between datasets.\n"
             "Participants with observations in each stratum are weighted equally. \u2020 Conditional on finite estimates; see strata.csv for coverage.",
             ha="center", va="center", fontsize=9.5, linespacing=1.5)
    finish(fig, output, "hr_motion_strata")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / ".local/ppg-hr-poc/summary", help="Directory containing the complete audit and three CSV tables")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/ppg_hr_poc", help="Destination for three PNG and three SVG figures")
    args = parser.parse_args()
    tables, _ = load_tables(args.input)
    configure_style()
    plot_label_efficiency(tables["summary"], args.output)
    plot_paired_differences(tables["paired"], args.output)
    plot_motion_strata(tables["strata"], args.output)


if __name__ == "__main__":
    main()
