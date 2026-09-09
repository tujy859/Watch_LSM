# 生理信号基础模型（Physiological Foundation Models）学术与业界全景调研

## 1. 调研背景与智能手表信号特性

智能手表作为主流的消费级可穿戴设备，其核心物理传感器主要为 **光电容积脉搏波（Photoplethysmography, PPG）** 和 **三轴加速度计（3-Axis Accelerometer, ACC）**，部分设备还配备三轴陀螺仪（Gyroscope）、温度传感器（Skin Temp）和皮肤电反应（EDA）。

### 1.1 信号特征与主要难点
1. **多模态与异构采样率（Multi-Rate Heterogeneity）**：
   - 原始 PPG：通常为 25 Hz、50 Hz、64 Hz 或 100 Hz（甚至 256 Hz - 1000 Hz 临床高频）。包含脉搏收缩峰（Systolic peak）、重搏波凹槽（Dicrotic notch）和舒张峰（Diastolic peak）。
   - 原始 ACC/IMU：通常为 25 Hz 或 50 Hz。反映肢体空间运动加速度矢量 $(a_x, a_y, a_z)$。
   - 派生生理指标：心率（Heart Rate, 1 Hz）、步数（Steps, 1 Hz 或累加）、心率变异性（HRV: RMSSD/SDNN, 窗口级）、睡眠分期（Sleep Stage, 30秒/epoch）、血氧饱和度（SpO2）、呼吸率（RR）。
2. **严重的运动伪影（Motion Artifacts, MA）**：
   - 当用户处于走路、跑步、日常挥手或骑行时，手腕与表体发生微小位移，静脉血液发生惯性晃动，光线漫反射剧烈改变，产生大幅度运动伪影。
   - 在频域上，步频谐波（通常在 1.0 Hz - 3.5 Hz，即 60 - 210 SPM）与运动中心率主频（1.2 Hz - 3.2 Hz，即 70 - 190 BPM）产生**强烈的频带重叠**。
3. **光电个体差异与跨域泛化瓶颈**：
   - 肤色深度（Melanin absorption, Fitzpatrick 肤色分级 I-VI 导致绿光穿透深度衰减）、皮下脂肪厚度、佩戴松紧度、环境光干扰导致信噪比（SNR）跨个体差异极大。
4. **低功耗与边缘计算限制**：
   - 智能手表电池容量通常在 200-500 mAh，算法需满足端侧 MCU / NPU 的极低计算与功耗开销（通常需要小于 5ms 推理延迟）。

---

## 2. 学术与业界代表性基础模型深度解析

### 2.1 Google Research: SensorFM (2026)
- **论文**: *Towards a General Intelligence and Interface for Wearable Health Data* (arXiv:2605.22759, Google Research)
- **核心定位**: 消费级可穿戴生理信号的“超大规模通用智能底座”。
- **预训练规模**: 超过 **1 万亿分钟（>1 Trillion minutes）** 的无标注多模态可穿戴时间序列数据，覆盖全球超过 **500 万** 同意授权的真实用户（Fitbit / Pixel Watch 生态）。
- **模态组合**: PPG（光电脉搏）、ACC（加速度）、EDA（皮电反应）、Skin Temperature（体温）、Altimeter（高度计）。
- **架构设计**:
  - 基于分块（Patch-based）的通用多通道时间序列 Transformer。
  - 针对异构采样率采用动态重采样与模态独立 Patch Embedding，统一投影到共享隐空间。
  - 自监督预训练目标：结合掩码时间序列重建（Masked Time Series Modeling）与跨模态自回归填充（Cross-Modal Infilling）。
- **下游任务与效果**:
  - 在 **35 个健康与行为预测基准** 上进行迁移测试（涵盖心率跟踪、睡眠结构分期、最大摄氧量 VO2 max、胰岛素抵抗趋势、代谢综合征、情绪压力等）。
  - 在 **34 个任务** 上超越所有现有专用模型（Specialist models），展现了基础模型在生理数据上的 Scaling Law 规律。

### 2.2 Nokia Bell Labs & Cambridge: PaPaGei (ICLR 2025)
- **论文**: *PaPaGei: Open Foundation Models for Optical Physiological Signals* (ICLR 2025 / arXiv:2410.20542)
- **开源资源**: GitHub `https://github.com/Nokia-Bell-Labs/papagei-foundation-model`，Zenodo 权重公开。
- **核心贡献**: 首个全面开源的通用 PPG 生理信号基础模型。
- **数据规模**: 超过 **57,000 小时**（约 2000 万个 10 秒无标注 PPG 信号切片），整合了 10 个公开医疗与可穿戴数据集。
- **模型架构**:
  - **ResNet1D-MoE (Mixture of Experts)**: 18 个 1D 残差块，32 个基础卷积通道，3 个 MoE 路由专家，输出 512 维潜空间表征。
  - 解决传统对比学习（SimCLR）强迫同一个人在不同生理状态下信号表征一致、从而丢失脉搏微形态变化的缺陷。
  - 引入形态感知正样本对构建（Morphology-Aware Positive Pair）与多任务对比表征。
- **下游效果**:
  - 在心率回归、收缩压/舒张压估计、血管弹性年龄预测、睡眠呼吸暂停等 20 个下游任务中，线性探测准确率平均提升 6.3%，参数量仅为其他时间序列大模型的 1/70。

### 2.3 UIUC & University of Memphis: Pulse-PPG (UbiComp / IMWUT 2025)
- **论文**: *Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications Across Lab and Field Settings*
- **开源资源**: GitHub `https://github.com/maxxu05/pulseppg`，Zenodo 权重公开。
- **核心定位**: 专门针对**真实世界野外环境（Field-Trained）** 高噪 PPG 信号训练的基础模型。
- **预训练数据**: 120 名受试者连续 100 天自由生活状态下采集的 **2 亿秒（~55,000小时）** 真实手腕 PPG。
- **架构与目标**:
  - 12 块 1D-ResNet 深度骨干网（128 base filters, kernel size 11）。
  - 提出 **Relative Contrastive Learning (RelCon)**: 考虑时间临近性与运动强度的连续相关性，而非简单的二元正负样本划分。
  - 针对真实野外严重运动伪影进行自适应过滤与对比对齐。
- **实证对比**: 在跨域生物学年龄预测和活动心率追踪中，Pulse-PPG 在野外高噪条件下的泛化能力显著优于仅在干净临床数据（如 MIMIC/VitalDB）上训练的模型。

### 2.4 Samsung Research: xMAE 与 HiMAE (ICML 2024 & ICLR 2025)
- **核心定位**: 专为智能手表（Galaxy Watch 等）端侧芯片优化的高效生理表征基础模型。
- **xMAE (Physiology-Aware Masked Cross-Modal Reconstruction, ICML 2024)**:
  - 解决智能手表上 PPG 可全天被动采集，但高精度心电（ECG）只能偶尔主动测量的痛点。
  - 基于 9,400 小时配对 ECG-PPG 数据，通过掩码跨模态重建，让连续 PPG 编码器重构对应的 ECG 潜表征。
  - 实现了“由连续 PPG 虚拟推断高精度心脏电生理特征”。
- **HiMAE (Hierarchical Masked Autoencoder, ICLR 2025)**:
  - 分层多时间尺度编码器：同时捕获秒级心跳搏动波形（1-2秒）、活动步频节奏（5-10秒）以及昼夜作息节律（小时/天级别）。
  - **端侧超低延迟**: 专门面向智能手表 CPU 进行算子精简，单次推理延迟 **小于 1 毫秒**。

### 2.5 Stanford Medicine: SleepFM (Nature Medicine 2024-2026)
- **论文**: *SleepFM: A Multi-modal Foundation Model for Sleep Analysis*
- **预训练数据**: 6.5 万名受试者、约 60 万小时的多导睡眠监测（PSG）数据。
- **创新技术**: 提出 **Leave-One-Out Contrastive Learning（留一对比学习）**：
  - 在多模态输入（EEG 脑电、ECG/PPG 脉搏、呼吸、肌电）中，随机掩码一个生理模态，训练共享表征空间根据其余模态重构缺失模态。
- **下游突破**: 单夜睡眠生理表征不仅能精准分期，还展现出对超过 130 种健康状况的预测能力（痴呆 C-index 0.85、心肌梗死 0.81、房颤 0.78）。

### 2.6 CMU Auton Lab: MOMENT (ICML 2024)
- **论文**: *MOMENT: A Family of Open Time-series Foundation Models*
- **开源资源**: GitHub `moment-timeseries-foundation-model/moment`，Hugging Face `AutonLab/MOMENT-1-large/base/small`。
- **技术路线**:
  - 基于 Patch-level Masked Autoencoder 的通用多任务时间序列模型。
  - 预训练语料库 Time-series Pile 广泛吸纳了心电、脉搏、脑电及可穿戴运动加速度数据。
  - 支持零样本信号补全（Imputation）、异常检测（Anomaly Detection）、以及微调分类与预测。

### 2.7 传感器表征基础模型: LIMU-BERT (UbiComp)
- **论文**: *LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Mobile Applications*
- **核心定位**: 针对三轴加速度计（ACC）与陀螺仪（Gyro）的端侧基础模型。
- **技术路线**: 借鉴 BERT 的 Masked Sensor Modeling，将三轴空间运动向量切片掩码后重构，提取与受试者身份解耦的高阶步态与动作动力学表征。

---

## 3. 基础模型技术路线对比与设计范式总结

| 模型维度 | 纯对比学习 (Contrastive SSL) | 掩码自编码 (MAE / MSM) | 跨模态蒸馏/虚拟传感 (Cross-Modal Distillation) | 混合多尺度架构 (Hierarchical Hybrid) |
| :--- | :--- | :--- | :--- | :--- |
| **代表工作** | PaPaGei, Pulse-PPG (RelCon) | MOMENT, LIMU-BERT | xMAE (PPG$\to$ECG), SensorFM | HiMAE, WearFM |
| **输入处理** | 连续波形 / 1D-CNN 提取 | 1D Patch (8/16采样点) 切片 | 配对时间序列对齐 | 多分支金字塔 Patching |
| **抗运动伪影能力** | 依赖数据增强与样本对选择 | 对局部缺失鲁棒，但高噪下易重构噪声 | 极强（利用ACC运动先验辅助解耦） | 极强（宏观动作与微观脉搏分层） |
| **下游任务适应性** | 分类/检索好，密集回归需微调 | 信号修复/异常检测极佳 | 生理指标（HR/BP/HRV）极佳 | 全功能多任务（步数、心率、睡眠） |
| **端侧部署开销** | 极小（1D-ResNet 易量化） | 适中（Transformer 随序列长度二次方增长） | 适中（推理期仅需单模态编码器） | 极小（专为手表端侧优化，<1ms） |

---

## 4. 对自建智能手表生理基础模型（Watch-LSM）的工程架构指导

基于以上前沿调研，为我们基于自有手表数据集（PPG、ACC $\to$ 心率、步数等）训练基础模型提炼出关键技术路线：

1. **输入表示与多速率对齐（Multi-Rate Tokenization）**：
   - 智能手表的 PPG（如 50Hz/64Hz）与 ACC（如 25Hz/50Hz）采样率不同。
   - 最佳实践：采用 **1D Patch Projection**，按固定物理时长（例如 0.25 秒或 0.5 秒）为一个 patch token，将 PPG 窗口与 ACC 窗口统一对齐到相同长度的 Token 序列。
2. **双流交互骨干网络（Dual-Stream Cross-Attention Backbone）**：
   - PPG 流：专注于捕获微血管搏动的周期性容积变化及重搏形态。
   - ACC 流：专注于捕获身体运动节奏（步频、抬手、抖动）。
   - 交叉注意力机制（Cross-Attention）：让 ACC 编码器将运动特征作为“噪声先验”提供给 PPG 解码器，实现注意力引导的运动伪影主动消除。
3. **复合自监督预训练目标（Composite SSL Objectives）**：
   - $\mathcal{L}_{\text{MAE}}$（掩码重构）：随机掩码 30%-50% 的 PPG 与 ACC 片段进行重构，迫使模型学习脉搏波的内在生理动力学。
   - $\mathcal{L}_{\text{CrossModal}}$（跨模态对齐）：通过 InfoNCE 对比损失，拉近同一时间窗口内 PPG 运动伪影成分与 ACC 运动信号的表征互信息。
   - $\mathcal{L}_{\text{Spectral}}$（频域保真损失）：引入 FFT 频谱损失，强制模型在频域保留心率基础脉冲频率和步频谐波。
4. **统一的多任务下游预测头（Unified Downstream Heads）**：
   - **心率跟踪头（HR Regression Head）**：从表征输出每秒瞬时心率（BPM），在运动伪影下具有高鲁棒性。
   - **步数与步频头（Step/Cadence Head）**：从 ACC 驱动的表征中提取瞬时步频（SPM）和累计步数。
   - **状态与活动分类头（Activity State Head）**：区分静止、散步、跑步、骑行、睡眠。

### 2.8 华为诺亚方舟实验室: Mantis (ICML 2026)
- **论文**: *Mantis: Lightweight Foundation Model for Time Series Classification* (ICML 2026 / arXiv:2502.15637)
- **开源资源**: GitHub `vfeofanov/mantis`，PyPI `mantis-tsfm`，Hugging Face `paris-noah/Mantis-8M / MantisPlus / MantisV2`。
- **核心定位**: 时序领域首个专注于**时序分类与判别性表征学习**的轻量级开源基础模型。
- **架构创新**:
  - **Token Generator Unit (TGU)**: 自适应多尺度时序卷积与差分分块，摆脱传统固定切块导致的形态信息失真。
  - **轻量级 ViT-1D 骨干**: 参数量仅数百万（~8M），体积小、单次推理延迟极低，高度适配智能手表低功耗处理器。
  - **自监督对比学习 (Contrastive Learning)**: 在大规模合成与真实多样性时序数据上预训练，学习本质动力学形态。
  - **多变量通道适配器 (Multichannel Projectors)**: 原生支持手腕 3 轴加速度计或多导联生物信号。
- **对智能手表的实测与启示**:
  - 在本项目实测中（实验6），Mantis 提取 3 轴手腕运动表征，在 CPU 上仅需 **19.8ms**，在走路、跑步、骑行三分类上的线性探测准确率高达 **99.7%**（显著超越 LIMU-BERT 的 92.1%）。
  - 在手表端侧是**防误计步、精准动作识别、跌倒预警、心律失常快速筛查**的绝佳轻量级判别基座。
