# Google SensorFM 深度技术解析

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