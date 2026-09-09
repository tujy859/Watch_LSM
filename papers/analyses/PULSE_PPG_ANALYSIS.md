# Pulse-PPG (UbiComp 2025) 深度技术解析

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