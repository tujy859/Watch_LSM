"""Plot preregistered linear health increments from shared audited statistics."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs/results/ppg_formal_expansion/health"
MODELS=["AnyPPG","PaPaGei-S","PaPaGei-P","PulsePPG","SigmaPPG","MOMENT-1-small","AnyPPG-random-frozen"]
LABELS=["AnyPPG","PaPaGei-S","PaPaGei-P","Pulse-PPG","SigmaPPG","MOMENT","随机AnyPPG"]


def main():
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,3,figsize=(14,5.6),sharey=True)
    selected=[]
    tasks=[("ECSMP","sds_self_report_score","腕部SDS症状分数\n60人，120秒中性PPG","SDS分数"),
           ("PPG-BP","cuff_sbp_mmhg","指端收缩压\n218人，三个独立2秒片段","mmHg"),
           ("PPG-BP","cuff_dbp_mmhg","指端舒张压\n同一批218人、相同片段","mmHg")]
    for ax,(dataset,target,title,unit) in zip(axes,tasks):
        source=json.loads((OUT/f"{dataset}_comparisons.json").read_text(encoding="utf-8"))
        values=[next(r for r in source if r["target"]==target and r["model"]==m and r["head"]=="linear") for m in MODELS]
        selected.extend(values)
        for y,r in enumerate(values):
            good=r.get("improvement_supported_exploratory_inference",False)
            worse=r.get("p_holm_six_models",1)<.05 and r["ci_low"]>0
            color="#007C68" if good else "#B35436" if worse else "#5D6B7B"
            x=r["difference_mae"]
            ax.errorbar(x,y,xerr=[[x-r["ci_low"]],[r["ci_high"]-x]],fmt="o",color=color,capsize=3,markersize=6)
        ax.axvline(0,color="#222222",lw=1,ls="--")
        ax.set_title(title,fontsize=12,pad=16)
        ax.set_xlabel(f"平均绝对误差变化（{unit}）\n小于0表示加入表征后改善",fontsize=10)
        ax.grid(axis="x",alpha=.18); ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_yticks(np.arange(len(MODELS)),LABELS,fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("加入基础模型表征，是否比人口学＋传统生理特征更准？",fontsize=17,fontweight="bold",y=.98)
    fig.text(.5,.025,"统一线性预测器；按人五折，3次初始化；误差条为探索性人级配对95%区间。\n三项主要增量比较均未证明稳定改善；橙色表示该目标内六模型Holm校正后支持误差增大。",ha="center",fontsize=10)
    fig.subplots_adjust(left=.14,right=.98,top=.77,bottom=.23,wspace=.24)
    for ext in ("png","svg"):
        fig.savefig(OUT/f"health_increment.{ext}",dpi=180,facecolor="white")
    svg=OUT/"health_increment.svg"
    svg.write_text("\n".join(x.rstrip() for x in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    (OUT/"health_increment_plot_values.json").write_text(json.dumps(selected,indent=2)+"\n",encoding="utf-8")
    plt.close(fig)


if __name__=="__main__":
    main()
