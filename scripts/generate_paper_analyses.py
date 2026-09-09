import os

analyses = {
    "SENSORFM_GOOGLE_ANALYSIS.md": """# Google SensorFM 深度技术解析

## 1. 论文与团队背景
- **标题**: Towards a General Intelligence and Interface for Wearable Health Data (arXiv:2605.22759)
- **研发团队**: Google Research & Fitbit Health AI
- **发表时间**: 2026 年

## 2. 核心规模与数据构成
- **数据总量**: 超过 1 万亿分钟（>1,000,000,000,000 分钟）真实可穿戴时间序列。
- **用户规模**: 超过 500 万经过知情同意的 Fitbit / Pixel Watch 全球真实用户。
- **输入模态**:
  1. 绿光/红外光电容积脉搏波 (PPG)
  2. 三轴重力加速度计 (ACC)
  3. 皮肤电反应 (EDA)
  4. 皮肤温度传感器 (Skin Temp)
  5. 气压高度计 (Altimeter)

## 3. 架构与自监督算法
- **分块策略 (Patch Tokenization)**: 将连续的多速率生理流统一划分为固定物理时间窗口的 Patch。针对不同传感器采用独立线性投影层（Sensor-Specific Embedding），随后拼接时间位置编码。
- **预训练目标**:
  - Masked Multimodal Modeling: 随机掩码部分模态的部分时间片段，训练多层 Transformer 联合重构。
  - Cross-Modal Infilling: 在某些模态完全未开启或缺失（如低功耗模式关闭 PPG 仅保留 ACC）时，通过隐空间表征补全缺失模态信息。

## 4. 下游任务表现
- 涵盖 35 项下游预测任务，分为 4 大类：
  1. **心血管健康**: 静息心率、运动心率、HRV、血压趋势、最大摄氧量 (VO2 Max)。
  2. **活动与能量代谢**: 步数计算、卡路里消耗、高强度运动识别、代谢综合征风险评估。
  3. **睡眠与神经节律**: 睡眠深浅分期、快速眼动期 (REM)、睡眠呼吸中断事件。
  4. **全天候压力与心理指标**: EDA 压力激增检测、情绪波动状态。
- **实验结论**: 在 35 项任务中的 34 项击败针对单一任务精心设计的专用模型，证明了“万亿分钟多模态预训练”为生理计算带来的 Scaling 泛化收益。
""",

    "PAPAGEI_NOKIA_ANALYSIS.md": """# Nokia Bell Labs PaPaGei (ICLR 2025) 深度技术解析

## 1. 论文与开源背景
- **标题**: PaPaGei: Open Foundation Models for Optical Physiological Signals (ICLR 2025)
- **团队**: Nokia Bell Labs, University of Cambridge, Dartmouth College
- **开源代码**: `https://github.com/Nokia-Bell-Labs/papagei-foundation-model`
- **模型权重**: Zenodo Record 13983110

## 2. 核心架构: ResNet1D-MoE
- **输入配置**: 10 秒长度的 1D PPG 信号，目标采样率 125 Hz（序列长度 1250 个点），每个样本进行 Z-score 标准化。
- **网络结构**:
  - 18 个 1D 残差卷积块（ResNet Blocks），基础滤波器数量 32。
  - 引入 3 个 Mixture of Experts (MoE) 专家分支，动态根据脉搏形态路由特征。
  - 最终池化输出 512 维潜空间全局表征向量。
- **创新的形态对比学习 (Morphology-Aware Pretraining)**:
  - 传统 SimCLR 对比学习在生理信号上的缺陷：若将同一个人不同时间、不同生理活动下的波形当作同一类拉近，会消除脉搏的瞬时动力学（如运动引起的波形压缩、血管紧张引起的舒张切迹变平）。
  - PaPaGei 引入了基于波形形态特征（Morphology-Guided Pair Sampling）的正负样本构建策略，确保保留心血管动力学特异性。

## 3. 下游基准测试
- 覆盖 10 个公开数据集（VitalDB, MIMIC, CAPNO, WESAD, PPG-DaLiA 等）中的 20 个下游任务：
  - 心率预测 (HR Regression, MAE ~ 1.2-3.5 BPM)
  - 收缩压/舒张压 (Systolic/Diastolic Blood Pressure, MAE ~ 5-8 mmHg)
  - 血管弹性年龄 (Vascular Age)
  - 睡眠呼吸暂停 (Sleep Apnea Detection, F1 > 0.82)
""",

    "PULSE_PPG_ANALYSIS.md": """# Pulse-PPG (UbiComp 2025) 深度技术解析

## 1. 论文与团队背景
- **标题**: Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications Across Lab and Field Settings
- **团队**: University of Memphis & UIUC (Saha, Xu, Rehg, Kumar 等)
- **开源代码**: `https://github.com/maxxu05/pulseppg`

## 2. 关键创新: 野外非受限数据训练
- **数据来源**: 120 名自由生活受试者、连续 100 天的全天候佩戴数据，有效野外 PPG 数据量超过 2 亿秒（约 5.5 万小时）。
- **解决临床数据到手表的 Domain Shift**:
  - 临床数据集（如 MIMIC、VitalDB）患者处于静卧状态，信号几乎无运动伪影，波形极度规整。
  - 手表用户处于走路、跑步、做家务、洗手等状态，光电运动伪影剧烈。在野外数据上预训练使模型对运动干扰具有天然免疫力。
- **相对对比学习 (Relative Contrastive Learning, RelCon)**:
  - 放弃离散的 Positive/Negative 二元分类损失。
  - 将时间间隔距离与受试者运动量（由 ACC 计算）作为连续相关系数，构建加权温度系数的连续对比损失。

## 3. 架构参数
- 12 层 1D ResNet 架构，基础通道数 128，大卷积核（Kernel size = 11）以捕获心动周期的长程相关性。
- 在跨域生物学年龄预测中，Pulse-PPG 达到 MAE = 9.28 岁（R2=0.388），显著优于只在干净指尖数据上训练的模型。
""",

    "SAMSUNG_XMAE_HIMAE_ANALYSIS.md": """# Samsung Research xMAE & HiMAE 深度技术解析

## 1. 业界背景: 智能手表的端侧计算挑战
- **研发团队**: Samsung Research (Health AI Lab)
- **发表会议**: ICML 2024 (xMAE), ICLR 2025 (HiMAE)
- **产品导向**: 面向 Galaxy Watch 等穿戴设备的实时、端侧生理基础模型。

## 2. xMAE: 跨模态虚拟传感
- **核心逻辑**:
  - 临床金标准 ECG 具有极高的电生理信息量，但手表上用户必须另一只手按住电极才能测量（无法全天被动测量）。
  - 手表背面的光电 PPG 可以 24 小时被动连续采集，但信息维度较低、噪声大。
- **训练方案**:
  - 利用 9,400 小时配对同步采集的 ECG-PPG 数据。
  - 将 ECG 信号大比例掩码（如 75%），输入配对的 PPG 信号，训练 Cross-Modal Transformer 重构掩码后的 ECG 潜表征。
  - 推理期：无需用户触摸测量 ECG，仅凭被动 PPG 即可推断出富含电生理信息的深层潜表征。

## 3. HiMAE: 分层多尺度端侧轻量化
- **时间尺度分层**:
  - 尺度 1 (0.5 - 2 秒): 逐拍心跳微波形特征（用于 HR、HRV、脉搏传导时间）。
  - 尺度 2 (5 - 30 秒): 步态运动与呼吸调制包络（用于步数、能量消耗、呼吸率）。
  - 尺度 3 (5 - 30 分钟): 宏观生理状态与昼夜节律（用于活动/睡眠分期、压力评估）。
- **极低功耗优化**:
  - 针对手表 ARM Cortex-M / Cortex-A 芯片进行定点量化与算子融合，单次前向推理延迟小于 1 毫秒。
""",

    "STANFORD_SLEEPFM_ANALYSIS.md": """# Stanford SleepFM (Nature Medicine) 深度技术解析

## 1. 论文与团队背景
- **标题**: SleepFM: A Multi-modal Foundation Model for Sleep Analysis
- **发表期刊**: Nature Medicine (2024-2026)
- **团队**: Stanford Medicine (Emmanuel Mignot, James Zou 教授团队)

## 2. 数据与留一对比学习 (Leave-One-Out Contrastive Learning)
- **规模**: 65,000 名受试者，约 60 万小时夜间多导生理监测（PSG）数据。
- **输入模态**: EEG（脑电）、ECG/PPG（心电/脉搏）、呼吸气流/腹带、EMG（肌电）。
- **算法核心**:
  - 在每个 30 秒 epoch 中，随机掩码一个生理模态（例如隐藏呼吸或隐藏脉搏）。
  - 共享投影隐空间通过对比损失拉近剩余多模态表征与被掩码模态表征的余弦相似度。
  - 迫使模型学习不同人体器官系统（脑神经-心血管-呼吸动力学）在睡眠过程中的深层生物耦合机制。

## 3. 远期疾病风险预测突破
- 模型不仅实现了高精度的自动化睡眠分期（Wake/REM/N1/N2/N3），而且单夜生理表征具备预测长达数年未来疾病发生风险的能力：
  - 痴呆症 (Dementia) C-index: 0.85
  - 全因死亡率 (Mortality) C-index: 0.84
  - 心肌梗死 (MI) C-index: 0.81
  - 房颤 (Atrial Fibrillation) C-index: 0.78
""",

    "CMU_MOMENT_ANALYSIS.md": """# CMU MOMENT (ICML 2024) 深度技术解析

## 1. 论文与开源背景
- **标题**: MOMENT: A Family of Open Time-series Foundation Models (ICML 2024)
- **团队**: Carnegie Mellon University (Auton Lab)
- **开源代码**: `github.com/moment-timeseries-foundation-model/moment`
- **模型权重**: Hugging Face `AutonLab/MOMENT-1-large`, `MOMENT-1-base`, `MOMENT-1-small`

## 2. 架构与分块自编码范式
- **Patch-Based Masked Autoencoder**:
  - 将 1D 时间序列切分成不重叠的 Patch（默认 patch_length=8 或 16）。
  - 随机掩码 50%-70% 的 Patch。
  - 编码器采用双向 Transformer（12-24 层），解码器采用轻量级线性投影头或微型 Transformer。
- **Time-series Pile 语料**: 涵盖生理信号（ECG, PPG, EEG）、可穿戴动作（HAR）、环境传感等 13 个领域。

## 3. 生理信号应用与局限
- **优势**: 具有强大的零样本信号插补（Imputation）、降噪重建和异常检测能力。
- **局限**: 作为一个泛时间序列模型，其 Patch Tokenizer 未针对 PPG 脉搏的重搏切迹或 ACC 三轴物理矢量正交性进行先验建模，在细粒度生理回归任务上通常需要通过 Specialist 微调才能发挥最佳性能。
""",

    "LIMU_BERT_ANALYSIS.md": """# LIMU-BERT (UbiComp) 深度技术解析

## 1. 论文与开源背景
- **标题**: LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Mobile Applications (IMWUT / UbiComp)
- **团队**: DAPO Lab
- **开源代码**: `https://github.com/dapowan/LIMU-BERT-Public`

## 2. 架构与 Masked IMU Modeling
- **输入特征**: 三轴加速度计 $(a_x, a_y, a_z)$ 与三轴陀螺仪 $(\omega_x, \omega_y, \omega_z)$ 连续 120 采样点窗口（20 Hz 采样，约 6 秒动作窗口）。
- **掩码预训练**:
  - 类似 NLP 中的 BERT，对输入序列中的某些时间片进行随机遮盖或替换为高斯噪声。
  - 通过双向 Transformer 编码器重构空间运动轨迹。
- **抗个体差异表征**:
  - 传统基于 ACC 的计步和动作识别极易受到用户身高、步幅、佩戴松紧度的过拟合影响。
  - LIMU-BERT 的自监督隐空间有效抽离了动作的时序模式与绝对幅值偏置，显著增强了跨人（Cross-Subject）泛化性能。
"""
}

out_dir = "papers/analyses"
for fname, content in analyses.items():
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"Generated {fpath} ({len(content)} chars)")
