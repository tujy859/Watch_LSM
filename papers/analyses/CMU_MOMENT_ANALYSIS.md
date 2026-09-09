# CMU MOMENT (ICML 2024) 深度技术解析

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