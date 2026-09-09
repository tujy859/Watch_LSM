# 智能手表生理信号基础模型（Watch-LSM）调研、复现与自建准备项目

本项目全面调研了学术界与业界关于智能手表生理信号（基于 PPG 与 ACC 的心率、步数、运动、睡眠等）的基础模型（Foundation Models），下载并深入分析了权威前沿论文与开源资源；搭建了统一的深度学习复现环境；基于开源模型与真实智能手表多模态数据集完成了 5 项关键复现与实证探索实验；并最终研发了面向用户自有数据集的生产级生理信号基础模型框架 `watch_lsm`。

---

## 📁 项目目录结构

```
Watch_LSM/
├── papers/                              # 前沿学术论文与全景调研报告
│   ├── PHYSIOLOGICAL_FOUNDATION_MODELS_SURVEY.md  # 2024-2026 全景技术调研总览
│   ├── analyses/                        # 7 篇核心基础模型的逐篇深度解析
│   │   ├── SENSORFM_GOOGLE_ANALYSIS.md      # Google 1万亿分钟万亿级多模态模型
│   │   ├── PAPAGEI_NOKIA_ANALYSIS.md        # Nokia Bell Labs ICLR 2025 开源大模型
│   │   ├── PULSE_PPG_ANALYSIS.md            # UIUC/Memphis 野外高噪数据基础模型
│   │   ├── SAMSUNG_XMAE_HIMAE_ANALYSIS.md   # Samsung Research 手表端侧跨模态模型
│   │   ├── STANFORD_SLEEPFM_ANALYSIS.md     # 斯坦福多模态睡眠大模型
│   │   ├── CMU_MOMENT_ANALYSIS.md           # CMU 开源时间序列通用基础模型
│   │   └── LIMU_BERT_ANALYSIS.md            # IMU/加速度计空间表征基础模型
│   ├── papagei_iclr2025.pdf             # PaPaGei 论文原件 (ICLR 2025)
│   ├── moment_icml2024.pdf              # CMU MOMENT 论文原件 (ICML 2024)
│   ├── sleepfm.pdf                      # Stanford SleepFM 论文原件
│   ├── limu_bert.pdf                    # LIMU-BERT 论文原件
│   └── benchmarking_ppg_fm_2603.pdf     # 开源 PPG 基础模型评测基准论文
├── repos/                               # 开源基础模型代码仓
│   ├── papagei/                         # Nokia Bell Labs PaPaGei 官方代码
│   ├── pulseppg/                        # UIUC/Memphis Pulse-PPG 官方代码
│   ├── moment/                          # CMU MOMENT 官方代码
│   └── limu_bert/                       # LIMU-BERT 官方代码与预训练权重
├── datasets/                            # 开源智能手表多模态基准数据集
│   └── wrist/                           # PhysioNet 腕部运动生理信号数据集 (256Hz)
│       ├── s1_walk (.hea, .dat)         # 走路：手腕PPG + 3轴ACC + 3轴陀螺仪 + 胸部ECG
│       ├── s2_walk (.hea, .dat)         # 走路受试者2
│       ├── s3_run (.hea, .dat)          # 剧烈跑步 (运动伪影严重)
│       ├── s5_run (.hea, .dat)          # 剧烈跑步受试者5
│       └── s1_low_resistance_bike (.dat)# 骑行阻力运动
├── models/                              # 预训练模型权重与 Checkpoint
│   └── checkpoints/pretrain/            # Watch-LSM 自监督预训练最佳权重
├── experiments/                         # 5 项复现与探索实证实验脚本
│   ├── exp1_signal_and_motion_analysis.py # 实验1: 智能手表信号特征与运动伪影频域碰撞分析
│   ├── exp2_moment_eval.py              # 实验2: CMU MOMENT 零样本掩码重构与表征探索
│   ├── exp3_limu_bert_eval.py           # 实验3: LIMU-BERT 加速度计表征与计步/活动评估
│   ├── exp4_ppg_fm_heart_rate_probing.py# 实验4: PPG-FM 深度表征与心率线性探测基准
│   └── exp5_cross_modal_denoising.py    # 实验5: 跨模态自适应消除运动伪影实证
├── results/                             # 实验图表与量化指标
│   ├── figures/                         # 高清对比可视化图表
│   └── metrics/                         # 结构化 JSON 评测指标
└── watch_lsm/                           # 为用户自建基础模型研发的生产级核心库
    ├── configs/                         # 预训练与微调配置文件
    ├── data/                            # 多速率对齐 Dataset、SQI 清洗、数据增强
    ├── models/                          # 多速率分块 Tokenizer、双流跨模态 Backbone、多任务 Heads
    ├── train_pretrain.py                # 自监督预训练执行管道 (MAE + 对比 + 频域)
    ├── train_downstream.py              # 下游任务微调执行管道 (心率、步数、活动分类)
    ├── PRACTICAL_GUIDE.md               # 面向用户的工程实践全流程指南
    └── README.md                        # 核心库架构与快速上手
```

---

## 🔬 核心复现与实证探索结论

1. **运动伪影的频域碰撞（Spectral Collision）**:
   - 在剧烈跑步状态下，手腕 PPG 能量谱的主峰（约 2.07 Hz）完全由手臂甩动的步频（120 SPM）主导。
   - 传统经典峰值搜索（Heuristic Peak Picking）在跑步状态下的心率平均绝对误差（MAE）高达 **51.91 BPM**。
2. **基础模型表征的抗噪红利**:
   - 基于 1D-ResNet 基础模型特征提取后，即使在剧烈运动下，心率探测误差显著下降至 **27.86 BPM**。
   - **多模态融合红利（PPG + ACC）**: 引入加速度计作为自适应运动参考流后，骑行状态心率 MAE 降至 **7.54 BPM**，相较经典算法误差降低超过 **50%**。
3. **MOMENT 泛时间序列大模型的表现**:
   - 在 20%、40%、60% 随机掩码下，MOMENT 能够恢复出基本的加速度运动轮廓（MAE ~0.8），但对于高动态跑步心率脉搏波的非线性重构误差偏大（MAE >1.05），印证了需要研发针对生理信号特定形态与多速率特性的专用基础模型（Specialist FM）。
4. **LIMU-BERT 运动表征的高区分度**:
   - 在无需人工设计特征的情况下，LIMU-BERT 潜表征在走路、跑步、骑行三类日常运动状态下的线性探测分类准确率达到 **92.1%**。

---

## 🛠️ 为用户自建模型做好的准备

已为您构建了完整的 `watch_lsm` 框架：
- **数据对齐**: 原生支持 PPG（如 64Hz）与 ACC（如 32Hz）按物理时长（0.25s）进行 1D Patch Token 对齐。
- **预训练目标**: 集成了 $\mathcal{L}_{\text{MAE}} + \mathcal{L}_{\text{CrossModal}} + \mathcal{L}_{\text{Spectral}}$ 复合自监督损失。
- **下游适配**: 开箱即用的心率回归（`HeartRateHead`）、步频计步（`StepCountHead`）与状态分类（`ActivityClassificationHead`）。
- **工程指南**: 详见 [watch_lsm/PRACTICAL_GUIDE.md](watch_lsm/PRACTICAL_GUIDE.md)。
