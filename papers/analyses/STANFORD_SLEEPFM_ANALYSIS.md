# Stanford SleepFM (Nature Medicine) 深度技术解析

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