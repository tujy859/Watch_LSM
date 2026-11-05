"""Redraw the formal overview using only versioned, audited summary tables.

Every panel retains its original units. No best-head selection, composite score,
or missing-model imputation is performed. Head choices follow task protocols.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/results/ppg_formal_expansion"
MODELS = ["AnyPPG", "PaPaGei-S", "PaPaGei-P", "PulsePPG", "SigmaPPG",
          "MOMENT-1-small", "AnyPPG-random-frozen", "SmallCNN", "Traditional", "Constant"]
LABELS = ["AnyPPG", "PaPaGei-S", "PaPaGei-P", "Pulse-PPG", "SigmaPPG", "MOMENT",
          "随机AnyPPG", "小型CNN", "传统/基础对照", "常数/原始带噪"]
COLORS = ["#007C83", "#387EB8", "#7161A8", "#B07620", "#AB5E83", "#638C54",
          "#8A9299", "#D55E00", "#252C34", "#A5A5A5"]


def read_table(relative):
    path = BASE / relative
    return list(csv.DictReader(path.open(encoding="utf-8"))), {
        "path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def unique(rows, **criteria):
    found = [r for r in rows if all(r[k] == str(v) for k, v in criteria.items())]
    if len(found) > 1:
        raise ValueError(f"Ambiguous plot selection: {criteria}")
    return found[0] if found else None


def value(row, keys):
    if row is None:
        return None
    mean, low, high = keys
    return {"estimate": float(row[mean]),
            "low": float(row[low]) if low and row.get(low) not in (None, "", "nan") else None,
            "high": float(row[high]) if high and row.get(high) not in (None, "", "nan") else None,
            "source_row": row}


def collect(include_pending=False):
    panels, sources = [], []
    hr, source = read_table("hr/summary.csv"); sources.append(source)
    for target, budget, scenario, title, detail in [
        ("Dalia", "full", "within", "心率 · DaLiA", "9人训练＋3人验证；15人轮流测试"),
        ("WildPPG", "full", "within", "心率 · WildPPG", "9/10人训练＋3人验证；16人轮流测试"),
        ("Dalia", "3", "within", "少标签心率 · DaLiA", "2人训练＋1人验证；15人轮流测试"),
        ("WildPPG", "3", "within", "少标签心率 · WildPPG", "2人训练＋1人验证；16人轮流测试"),
        ("WildPPG", "full", "transfer", "跨数据集心率 · DaLiA → Wild", "12人训练＋3人验证；16名新测试者"),
    ]:
        points = {}
        for model in MODELS:
            method = {"Traditional": "elgendi", "Constant": "TrainMean"}.get(model, model)
            points[model] = value(unique(hr, target=target, budget=budget, scenario=scenario,
                method=method, head="PCA16-MLP" if model in MODELS[:7] else "Task-control"),
                ("mae_bpm", "ci_low_bpm", "ci_high_bpm"))
        panels.append(dict(title=title, detail=detail, unit="MAE（bpm）↓", points=points,
                           head="训练集PCA16＋小MLP；传统为Elgendi式"))
    for dataset, title, detail, unit in [
        ("BUT-PPG", "手机PPG质量 · BUT", "50人；3,888条采集记录", "AUROC ↑"),
        ("WESAD", "腕部实验状态 · WESAD", "15人；三类实验状态", "平衡准确率 ↑"),
        ("ECSMP", "腕部刺激类别 · ECSMP", "60人；六类视频刺激，非情绪诊断", "平衡准确率 ↑"),
    ]:
        rows, source = read_table(f"classification/{dataset}/summary.csv"); sources.append(source)
        points = {}
        for model in MODELS:
            method = {"Traditional": "HandFeatures/Linear", "Constant": "TrainPrior"}.get(model,
                model + "/PCA16-Linear" if model in MODELS[:7] else model)
            points[model] = value(unique(rows, method=method), ("estimate", "ci95_low", "ci95_high"))
        panels.append(dict(title=title, detail=detail, unit=unit, points=points,
                           head="训练集PCA16＋线性分类器；传统SQI特征"))
    for dataset, target, title, detail, unit in [
        ("ECSMP", "sds_self_report_score", "健康信息增量 · SDS", "60人；连续120秒中性腕部PPG", "MAE（SDS分数）↓"),
        ("PPG-BP", "cuff_sbp_mmhg", "健康信息增量 · 收缩压", "218人；指端三个独立2秒片段", "MAE（mmHg）↓"),
        ("PPG-BP", "cuff_dbp_mmhg", "健康信息增量 · 舒张压", "218人；指端三个独立2秒片段", "MAE（mmHg）↓"),
        ("PPG-BP-age", "chronological_age_years", "辅助表征 · 年龄", "218人；年龄从全部输入协变量中移除", "MAE（岁）↓"),
    ]:
        filename = f"health/{dataset}_summary.csv"
        rows, source = read_table(filename) if (BASE/filename).exists() else ([], None)
        if source: sources.append(source)
        elif not include_pending: raise FileNotFoundError(filename)
        points = {}
        for model in MODELS:
            name = {"Traditional": "shared_control", "Constant": "train_mean"}.get(model, model)
            inputs = {"Traditional": "demographics_traditional", "Constant": "none"}.get(model,
                      "demographics_traditional_embedding")
            head = {"Constant": "constant", "SmallCNN": "cnn"}.get(model, "linear")
            points[model] = value(unique(rows, target=target, model=name, inputs=inputs, head=head),
                                  ("mae", "mae_ci_low", "mae_ci_high"))
        panels.append(dict(title=title, detail=detail, unit=unit, points=points,
                           head="人口学＋传统特征＋表征；统一线性预测器"))
    clinical, source = read_table("clinical/summary.csv"); sources.append(source)
    for dataset, target, title, detail, unit in [
        ("BIDMC", "resp_manual", "临床辅助 · 呼吸率", "46名真实患者；735个32秒片段", "MAE（次/分钟）↓"),
        ("PulseDB", "sbp", "临床辅助 · 收缩压", "100人训练/20人验证/42人测试；MIMIC子集", "MAE（mmHg）↓"),
        ("PulseDB", "dbp", "临床辅助 · 舒张压", "同一固定患者划分；非完整AAMI评测", "MAE（mmHg）↓"),
    ]:
        points = {}
        for model in MODELS:
            method = {"Traditional": "HandFeatures", "Constant": "TrainMean"}.get(model, model)
            head = "" if model in ("SmallCNN", "Constant") else "PCA16-MLP"
            points[model] = value(unique(clinical, dataset=dataset, target=target, method=method, head=head),
                                  ("mae", "ci_low", "ci_high"))
        panels.append(dict(title=title, detail=detail, unit=unit, points=points,
                           head="训练集PCA16＋小MLP；传统特征同头"))
    # Dense-task selectors are explicit to avoid pooling different durations/SNRs.
    specs = [
        ("onsets/summary.csv", {"group": "all", "metric": "f1_50ms", "duration_s": "0"},
         "逐搏 · 人工PPG起始点", "7人；±50毫秒容差；单专家参考", "F1 ↑",
         ("mean", "ci95_low", "ci95_high"), {"Traditional": "MSPTDfast"}),
        ("onsets/summary.csv", {"group": "all", "metric": "rmssd_ms_mae", "duration_s": "60"},
         "脉率变异性 · RMSSD", "7人；60秒窗口；多数静息，非ECG HRV", "MAE（ms）↓",
         ("mean", "ci95_low", "ci95_high"), {"Traditional": "MSPTDfast"}),
        ("artifact/summary.csv", {}, "逐点伪影 · DaLiA/SEGADE", "15人；4,305片段各固定中央6秒评分", "F1 ↑",
         ("f1", "ci_low", "ci_high"), {"Traditional": "HandFeatures", "Constant": "TrainPrior"}),
        ("denoising/summary.csv", {"dataset": "EMBC21", "metric": "nrmse"},
         "合成去噪 · 腕部参考波形", "7人；按人隔离；−5/5/15 dB合成噪声", "NRMSE ↓",
         ("value", "ci_low", "ci_high"), {"Traditional": "FixedBandpass", "Constant": "NoisyInput"}),
        ("denoising/summary.csv", {"dataset": "EPFL", "metric": "nrmse"},
         "外部合成去噪 · EPFL", "1人；340条噪声副本；无人口置信区间", "NRMSE ↓",
         ("value", None, None), {"Traditional": "FixedBandpass", "Constant": "NoisyInput"}),
    ]
    for filename, filters, title, detail, unit, keys, mapping in specs:
        rows, source = read_table(filename) if (BASE/filename).exists() else ([], None)
        if source and source not in sources: sources.append(source)
        elif not rows and not include_pending: raise FileNotFoundError(filename)
        points = {m: value(unique(rows, **filters, method=mapping.get(m, m)), keys) for m in MODELS}
        panels.append(dict(title=title, detail=detail, unit=unit, points=points,
                           head="冻结时序特征＋小输出头；不支持的接口留空"))
    if not include_pending:
        expected = [10]*11 + [9] + [10]*3 + [6, 6, 7, 6, 6]
        actual = [sum(item is not None for item in panel["points"].values()) for panel in panels]
        if actual != expected:
            raise ValueError(f"Incomplete plot coverage: {actual} != {expected}")
    return panels, sources


def render(panels, sources, output):
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if f in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(5, 4, figsize=(24, 25))
    for number, (ax, panel) in enumerate(zip(axes.flat, panels), 1):
        for y, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
            item = panel["points"].get(model)
            if item is None:
                ax.text(.02, y, "—", transform=ax.get_yaxis_transform(), color="#BABABA", va="center")
                continue
            x, low, high = item["estimate"], item["low"], item["high"]
            if not np.isfinite(x): raise ValueError(f"Nonfinite plotted result: {panel['title']} / {model}")
            if low is not None and high is not None:
                ax.plot([low, high], [y, y], color=color, linewidth=1.5, alpha=.85)
            ax.plot(x, y, "o", color=color, markersize=5.5)
        ax.set_yticks(range(len(MODELS)), LABELS, fontsize=9)
        ax.set_ylim(9.7, -.7)
        ax.set_title(f"{number:02d}  {panel['title']}\n{panel['detail']}", loc="left", fontsize=11.5, pad=14)
        ax.set_xlabel(panel["unit"], fontsize=10)
        ax.text(0, -0.20, panel["head"], transform=ax.transAxes, fontsize=8.2, color="#4D5661")
        ax.grid(axis="x", color="#ECEFF2"); ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9)
        if "AUROC" in panel["unit"] or "准确率" in panel["unit"] or "F1" in panel["unit"]:
            ax.set_xlim(0, 1.03)
        else:
            left, right = ax.get_xlim(); ax.set_xlim(max(0, left), right)
    fig.suptitle("PPG基础模型正式评测：跨任务、跨场景与强对照", fontsize=24, fontweight="bold", y=.99)
    fig.text(.5, .965, "点按任务预定口径汇总（分类为等人权重合并指标）；线为人级95%区间。各面板保留原始量纲；箭头表示更好方向。\n"
             "六模型用于标量/分类；逐点任务为明确支持的模型子集。灰色横杠表示未评测，绝非零分。未进行基模预训练或编码器微调。",
             ha="center", fontsize=11)
    fig.text(.5, .012, "健康增量面板：模型表征加入同一人口学＋传统特征对照；年龄输入排除年龄。临床数据不能单独证明腕部性能。\n"
             "区间为探索性且未作全任务校正；优势判断以报告中的配对检验为准。合成去噪不等于真实运动噪声恢复，EPFL只有一个原始受试者。",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=.075, right=.985, top=.925, bottom=.055, hspace=.62, wspace=.48)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(output/f"formal_overview.{ext}", dpi=170, facecolor="white")
    plt.close(fig)
    svg = output/"formal_overview.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
    (output/"formal_overview_values.json").write_text(json.dumps({"sources": sources, "panels": panels},
        indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def render_hr_heads(output):
    rows, source = read_table("hr/summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), sharey=True)
    values = []
    for ax, (scenario, target, title) in zip(axes, [
        ("within", "Dalia", "DaLiA：9人训练＋3人验证"),
        ("within", "WildPPG", "Wild：9/10人训练＋3人验证"),
        ("transfer", "WildPPG", "DaLiA训练 → Wild测试")]):
        for h, (head, label, color) in enumerate([
            ("Native-Ridge", "原始维度＋线性", "#007C83"),
            ("PCA16-Ridge", "16维＋线性", "#7161A8"),
            ("PCA16-MLP", "16维＋小MLP", "#D55E00")]):
            for y, model in enumerate(MODELS[:7]):
                row = unique(rows, scenario=scenario, target=target, budget="full", method=model, head=head)
                item = value(row, ("mae_bpm", "ci_low_bpm", "ci_high_bpm")); values.append(row)
                pos = y + (h-1)*.21
                ax.plot([item["low"], item["high"]], [pos, pos], color=color, alpha=.6)
                ax.plot(item["estimate"], pos, "o", color=color, ms=4, label=label if y==0 else None)
        cnn = unique(rows, scenario=scenario, target=target, budget="full", method="SmallCNN", head="Task-control")
        ax.axvline(float(cnn["mae_bpm"]), ls="--", color="#333333", label="CNN平均误差")
        ax.set_title(title, fontsize=12); ax.set_xlabel("HR MAE（bpm），越低越好")
        ax.grid(axis="x", alpha=.15); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(7), LABELS[:7]); axes[0].invert_yaxis()
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
    fig.suptitle("模型表征与预测器必须一起解释", fontsize=18, fontweight="bold", y=.98)
    fig.text(.5, .91, "同一训练/验证/测试划分；PCA仅在训练人拟合。三种接法完整保留，没有按测试集挑选最好结果。", ha="center", fontsize=10)
    fig.subplots_adjust(left=.13, right=.98, top=.83, bottom=.24, wspace=.25)
    for ext in ("png", "svg"):
        fig.savefig(output/f"hr_head_comparison.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    svg=output/"hr_head_comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    (output/"hr_head_comparison_values.json").write_text(json.dumps({"source":source,"rows":values},indent=2)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, default=BASE)
    args = parser.parse_args()
    if args.preview_incomplete and args.output == BASE:
        raise ValueError("An incomplete preview must use a separate local output directory")
    render(*collect(args.preview_incomplete), args.output)
    render_hr_heads(args.output)


if __name__ == "__main__":
    main()
