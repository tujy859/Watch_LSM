"""Render published HR evidence and the proposed expansion coverage separately.

No small-sample pilot score is placed on an efficacy chart. All assets needed
to redraw these figures are in Git; raw data and model weights are not needed.
"""
from pathlib import Path
import csv
import json
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs/results/ppg_expansion_pilot"
METHODS=["AnyPPG","PaPaGei-S","PulsePPG","MOMENT-1-small","SmallCNN","elgendi","welch","TrainMean"]
LABELS=["AnyPPG","PaPaGei-S","Pulse-PPG","MOMENT","小型CNN","传统峰检测†","传统频谱†","训练者均值"]
COLORS=["#0072B2","#56B4E9","#009E73","#CC79A7","#D55E00","#666666","#999999","#BBBBBB"]


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def save(fig,name):
    for suffix in ("png","svg"):
        fig.savefig(OUT/f"{name}.{suffix}",dpi=170,facecolor="white")
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    plt.close(fig)


def hr_figure():
    source=ROOT/"docs/results/ppg_hr_poc/per_subject.csv"
    rows=read_csv(source)
    configs=[("within","Dalia","full","A  DaLiA：9人训练、3人验证"),
             ("within","WildPPG","full","B  Wild：9或10人训练、3人验证"),
             ("within","Dalia","3","C  DaLiA：2人训练、1人验证"),
             ("within","WildPPG","3","D  Wild：2人训练、1人验证"),
             ("transfer","WildPPG","full","E  DaLiA训练 → Wild测试")]
    fig,axes=plt.subplots(3,2,figsize=(15.6,14))
    values=[]
    for ax,(scenario,target,budget,title) in zip(axes.flat,configs):
        for index,(method,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
            x=np.array([float(r["conditional_mae_bpm"]) for r in rows if r["scenario"]==scenario and
                        r["target"]==target and r["budget"]==budget and r["method"]==method and r["stratum"]=="all"])
            if len(x) not in (15,16):
                raise ValueError("Expected all published evaluation participants")
            rng=np.random.default_rng(20261105)
            ci=np.quantile(x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1),[.025,.975])
            mean=x.mean()
            ax.barh(index,mean,color=color,height=.63)
            ax.errorbar(mean,index,xerr=[[max(0,mean-ci[0])],[max(0,ci[1]-mean)]],fmt="none",ecolor="#303030",capsize=3,lw=1)
            ax.text(ci[1]+.3,index,f"{mean:.2f}",va="center",fontsize=10)
            values.append({"scenario":scenario,"target":target,"training_label_setting":budget,"method":method,
                           "mae_bpm":float(mean),"ci_low":float(ci[0]),"ci_high":float(ci[1])})
        ax.set_yticks(range(len(LABELS)),LABELS)
        ax.invert_yaxis();ax.set_title(title,loc="left",fontweight="bold",fontsize=12)
        ax.set_xlabel("心率平均绝对误差（bpm），越低越好")
        ax.set_xlim(0,34);ax.grid(axis="x",alpha=.18);ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    ax=axes[2,1];ax.axis("off")
    ax.text(.02,.96,"本图只使用已完成的正式评测",fontsize=16,weight="bold",va="top")
    ax.text(.02,.82,"• PaPaGei-P、SigmaPPG已完成小样本流程，\n   正式扩展尚待确认，因此此处没有填入分数。\n\n"
        "• 每次测试者与训练、验证者严格分开。\n   图中人数指训练和验证所使用的标签人数。\n\n"
        "• 误差条：受试者级探索性95%区间。\n   方法间优劣须看配对比较，不能只看均值。\n\n"
        "• † Wild传统方法覆盖率约99%，\n   误差为可输出窗口上的条件误差。\n\n"
        "• 所有新试跑分数均不进入本图。",fontsize=11,va="top",linespacing=1.5)
    fig.suptitle("已完成的腕部心率评测：补全模型视野",fontsize=20,weight="bold",y=.98)
    fig.subplots_adjust(left=.12,right=.96,top=.93,bottom=.06,wspace=.36,hspace=.40)
    save(fig,"completed_hr_model_comparison")
    (OUT/"completed_hr_plot_values.json").write_text(json.dumps({"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "source":"docs/results/ppg_hr_poc/per_subject.csv","values":values},indent=2)+"\n")


def coverage_figure():
    # Status refers to the current wrist/new-task protocol, not the older PPG-Bench.
    models=["AnyPPG","PaPaGei-S","PaPaGei-P","Pulse-PPG","SigmaPPG","MOMENT"]
    rows=[("腕部心率：域内、少标注人数",[3,3,2,3,2,3]),
          ("腕部心率：跨数据集",[3,3,1,3,1,3]),
          ("逐搏起始点 / 间期",[3,1,1,0,0,0]),
          ("PRV：RMSSD、间期标准差",[3,1,1,0,0,0]),
          ("手机PPG质量分类",[3,1,2,1,2,1]),
          ("点级运动伪影：SEGADE",[1,1,1,0,0,0]),
          ("配对波形去噪：合成压力测试",[1,1,0,0,0,0]),
          ("呼吸率：人工呼吸参考",[2,1,2,1,2,1]),
          ("血压：PulseDB官方划分",[2,1,2,1,2,1]),
          ("压力 / 中性 / 愉悦状态",[2,1,2,1,2,1]),
          ("六类情绪刺激",[2,1,2,1,2,1]),
          ("腕部SDS分数：健康信息增量",[2,1,2,1,1,1]),
          ("指端血压：健康信息增量",[2,1,2,1,1,1]),
          ("年龄：新受控任务头",[1,1,1,1,1,1]),
          ("SpO2：探索性预测",[2,1,2,1,2,1])]
    colors=["#EFEFEF","#FBE8BA","#B6DDE8","#74BFA4"]
    words=["暂不纳入","拟评测","小样本通路","正式结果"]
    fig,ax=plt.subplots(figsize=(15,10.6))
    a=np.array([r[1] for r in rows])
    ax.imshow(a,cmap=ListedColormap(colors),vmin=0,vmax=3,aspect="auto")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j,i,words[a[i,j]],ha="center",va="center",fontsize=10)
    ax.set_xticks(range(len(models)),models,fontsize=12)
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=10)
    ax.set_yticks(range(len(rows)),[r[0] for r in rows],fontsize=11)
    ax.set_xticks(np.arange(-.5,len(models),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which="minor",color="white",linewidth=3);ax.tick_params(which="minor",length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle("评测扩展方案：哪些已完成，哪些只跑通了小样本",fontsize=19,weight="bold",y=.98)
    fig.text(.5,.075,"绿色为既有协议的正式结果；本轮受控预测头仍只试跑。另设小CNN、传统方法、常数及随机Any对照。",ha="center",fontsize=10.5)
    fig.text(.5,.043,"本图不是性能排名。灰色表示首轮不纳入；点级伪影和去噪已验证数据，但尚未训练基模输出头。",ha="center",fontsize=10,color="#555555")
    fig.subplots_adjust(left=.29,right=.98,top=.89,bottom=.12)
    save(fig,"evaluation_coverage")
    (OUT/"evaluation_coverage.json").write_text(json.dumps({"models":models,"legend":dict(enumerate(words)),
        "rows":[{"task":r[0],"statuses":r[1]} for r in rows],"status":"proposal awaiting user confirmation",
        "completed_scope":"Green refers to previously completed protocols; new controlled PCA/linear/MLP heads have only small-sample checks"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":10,"axes.unicode_minus":False,"svg.fonttype":"none"})
    hr_figure();coverage_figure()
    print("Rendered completed-results comparison and separate proposed coverage")


if __name__=="__main__":main()
