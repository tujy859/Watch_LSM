# Nokia Bell Labs PaPaGei (ICLR 2025) 深度技术解析

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