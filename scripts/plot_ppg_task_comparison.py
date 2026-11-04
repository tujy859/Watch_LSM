"""One research figure across tasks, preserving each metric's original unit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#0072B2", "#D55E00", "#777777")
METHODS = ("AnyPPG", "SmallCNN", "Elgendi")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def ci(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(20261104)
    boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(boot, [.025, .975])


def grouped(ax, groups, values, title, ylabel, limits=None, decimals=1):
    width = .23
    positions = np.arange(len(groups))
    for index, color in enumerate(COLORS):
        table = np.asarray(values[index], float)
        x = positions + (index - 1) * width
        y, low, high = table.T
        ax.bar(x, y, width=width * .92, color=color, zorder=3)
        finite_ci = np.isfinite(y) & np.isfinite(low) & np.isfinite(high)
        ax.errorbar(x[finite_ci], y[finite_ci], yerr=np.maximum(0, np.vstack([y[finite_ci] - low[finite_ci], high[finite_ci] - y[finite_ci]])), color="#222222",
                    fmt="none", capsize=3, linewidth=1., zorder=4)
        for xx, yy, hh in zip(x, y, high):
            if np.isfinite(yy):
                ax.annotate(f"{yy:.{decimals}f}", (xx, hh if np.isfinite(hh) else yy), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color="#202020")
            else:
                ax.text(xx, .03, "NA", ha="center", fontsize=9,
                        transform=ax.get_xaxis_transform(), color="#666666")
    ax.set_xticks(positions, groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.grid(axis="y", color="#D9DEE3", alpha=.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    if limits:
        ax.set_ylim(*limits)
    else:
        bounds = [v[2] if np.isfinite(v[2]) else v[0] for method in values for v in method if np.isfinite(v[0])]
        ax.set_ylim(0, max(bounds) * 1.22 if bounds and max(bounds) > 0 else 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", type=Path, default=ROOT / "docs/results/ppg_phase2")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/ppg_phase2")
    args = parser.parse_args()
    phase2 = args.phase2
    beat_directory = phase2 / "onsets/summary" if (phase2 / "onsets/summary").is_dir() else phase2 / "onsets"
    beat_audit = json.loads((beat_directory / "audit.json").read_text())
    sqi_audit = json.loads((phase2 / "sqi/audit.json").read_text())
    if beat_audit["status"] != "passed" or sqi_audit["status"] != "complete":
        raise ValueError("Only complete audited studies may enter the comparison figure")
    hr_path = ROOT / "docs/results/ppg_hr_poc/per_subject.csv"
    hr = read_csv(hr_path)
    beat_path = beat_directory / "summary.csv"
    beat = read_csv(beat_path)
    sqi_path = phase2 / "sqi/summary.csv"
    sqi = read_csv(sqi_path)
    strata_path = phase2 / "transfer/strata.csv"
    strata = read_csv(strata_path)
    plot_data = []
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
                         "axes.unicode_minus": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 3, figsize=(18.5, 15.2))
    def hr_values(conditions):
        values = []
        for method in METHODS:
            name = "elgendi" if method == "Elgendi" else method
            group = []
            for dataset, scenario, budget in conditions:
                people = [float(r["conditional_mae_bpm"]) for r in hr if r["target"] == dataset
                          and r["scenario"] == scenario and r["budget"] == budget and r["method"] == name
                          and r["stratum"] == "all"]
                value = ci(people)
                group.append(value)
                plot_data.append({"panel": "HR", "dataset": dataset, "scenario": scenario, "budget": budget,
                                  "method": method, "metric": "MAE_bpm", "mean": value[0], "low": value[1], "high": value[2]})
            values.append(group)
        return values
    grouped(axes[0, 0], ["DaLiA 域内", "Wild 域内", "DaLiA→Wild"],
            hr_values([("Dalia", "within", "full"), ("WildPPG", "within", "full"), ("WildPPG", "transfer", "full")]),
            "A  窗口心率 · 全标签预算", "MAE (bpm) ↓")
    grouped(axes[0, 1], ["DaLiA", "WildPPG"],
            hr_values([("Dalia", "within", "3"), ("WildPPG", "within", "3")]),
            "B  窗口心率 · 3人标签预算", "MAE (bpm) ↓")
    def beat_values(metric, duration):
        result = []
        for method in METHODS:
            values = []
            name = "MSPTD" if method == "Elgendi" else method
            for group in ("all", "rest", "squat", "step"):
                record = next(r for r in beat if r["group"] == group and r["method"] == name
                              and r["metric"] == metric and int(float(r["duration_s"])) == duration)
                value = tuple(float(record[k]) for k in ("mean", "ci95_low", "ci95_high"))
                values.append(value)
                plot_data.append({"panel": "onsets_or_prv", "dataset": "EMBC21",
                                  "scenario": group, "budget": "full", "method": name,
                                  "metric": metric, "mean": value[0], "low": value[1], "high": value[2]})
            result.append(values)
        return result
    groups = ["全部", "静息", "蹲起", "踏步"]
    grouped(axes[0, 2], groups, beat_values("f1_50ms", 0),
            "C  腕部脉搏起始点 · ±50 ms", "人工 PPG 参考 F1 ↑", (0, 1.12), 2)
    grouped(axes[1, 0], groups, beat_values("matched_rr_mae_ms", 0),
            "D  成功匹配的相邻脉搏间期", "条件 MAE (ms) ↓")
    prv_groups = ["全部\n7人", "静息\n7人", "蹲起*\n1人", "踏步\n4人"]
    grouped(axes[1, 1], prv_groups, beat_values("rmssd_ms_mae", 60),
            "E  60秒 PRV · RMSSD", "相对人工 PPG 标注的 MAE (ms) ↓")
    grouped(axes[1, 2], prv_groups, beat_values("sdpp_ms_mae", 60),
            "F  60秒 PRV · 间期标准差", "相对人工 PPG 标注的 MAE (ms) ↓")
    sqi_values = []
    for method in ("AnyPPG", "SmallCNN", "HandSQI"):
        values = []
        for metric in ("auroc", "balanced_accuracy"):
            record = next(r for r in sqi if r["method"] == method and r["metric"] == metric)
            value = tuple(float(record[k]) for k in ("estimate", "ci95_low", "ci95_high"))
            values.append(value)
            plot_data.append({"panel": "SQI", "dataset": "BUT-PPG", "scenario": "auxiliary_phone",
                              "budget": "full", "method": method, "metric": metric,
                              "mean": value[0], "low": value[1], "high": value[2]})
        sqi_values.append(values)
    grouped(axes[2, 0], ["AUROC", "平衡准确率"], sqi_values,
            "G  质量分类 · 手机辅助域", "分数 ↑", (0, 1.12), 2)
    grouped(axes[2, 1], groups, beat_values("matched_rr_coverage", 0),
            "H  相邻参考间期成功匹配率", "覆盖率 ↑ · 与 D 同时解读", (0, 1.12), 2)
    ax = axes[2, 2]
    labels, codes = ["<60", "60–80", "80–100", "100–120", "≥120"], ["lt60", "60to80", "80to100", "100to120", "ge120"]
    x = np.arange(5)
    for index, method in enumerate(METHODS[:2]):
        rows = [next(r for r in strata if r["method"] == method and r["group"] == "hr" and r["stratum"] == c) for c in codes]
        values = [float(r["gap_contribution_bpm"]) for r in rows]
        ax.bar(x + (index - .5) * .32, values, width=.29, color=COLORS[index], zorder=3)
        for code, value in zip(codes, values):
            plot_data.append({"panel": "transfer_diagnostic", "dataset": "WildPPG",
                              "scenario": code, "budget": "full", "method": method,
                              "metric": "gap_contribution_bpm", "mean": value, "low": "", "high": ""})
    ax.axhline(0, color="#333333", linewidth=.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("WildPPG 参考心率区间 (bpm)")
    ax.set_ylabel("迁移额外 MAE 的分项贡献 (bpm)")
    ax.set_title("I  跨域下降诊断 · 事后描述分析", loc="left", fontweight="bold", pad=12)
    ax.grid(axis="y", color="#D9DEE3", alpha=.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PPG基础模型在哪些任务和场景有收益？", fontsize=23, fontweight="bold", x=.055, ha="left", y=.982)
    fig.text(.055, .953, "公开冻结 AnyPPG · 小型监督 CNN · 传统算法/手工特征    |    科学评测 PoC · 2026-11-04", fontsize=12, color="#49545D")
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, ("AnyPPG + 任务头", "小型 CNN", "传统算法 / 手工 SQI 特征"))],
               loc="upper center", bbox_to_anchor=(.5, .935), ncol=3, frameon=False, fontsize=12)
    fig.text(.055, .082,
             "误差条：受试者级探索性95% bootstrap区间；未校正多重比较。各面板保留原始单位，不合成总分。\n"
             "传统对照：HR为Elgendi式峰检测（Wild输出覆盖99.13%），提点为MSPTDfast风格适配，SQI为手工特征+逻辑回归。\n"
             "C–F/H：EMBC21腕部7人、专家PPG起始点，按人留一测试；只评人工可标注区域。E/F为60秒PRV算法一致性探索。\n"
             "E/F共175个连续窗，159个为静息；*蹲起只有1人，无人群置信区间。PRV可输出覆盖均为100%，不构成临床HRV验证。\n"
             "D仅统计双方连续成功匹配的间期，需结合H覆盖率；跨设备ECG对齐受限试跑未列入主图。",
             fontsize=9.2, color="#39434B", linespacing=1.4, va="top")
    fig.subplots_adjust(left=.065, right=.98, bottom=.135, top=.89, hspace=.44, wspace=.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"ppg_task_comparison.{suffix}", dpi=180, facecolor="white")
    svg = args.output / "ppg_task_comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")
    plt.close(fig)
    with (args.output / "figure_values.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_data[0]))
        writer.writeheader()
        writer.writerows(plot_data)
    inputs = [hr_path, beat_path, sqi_path, strata_path]
    manifest = {"inputs_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "design": "Task-separated grouped bars with original units, direction arrows and participant intervals; no composite score."}
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
