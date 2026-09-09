# Watch-LSM: 智能手表生理信号基础模型库

`watch_lsm` 是一个专为智能手表双物理传感器（光电容积脉搏波 PPG 与三轴加速度计 ACC）量身定制的轻量级生理信号基础模型（Lightweight Smartwatch Foundation Model）库。

## 🌟 核心特性
1. **多速率物理时间分块对齐 (Multi-Rate Patch Tokenization)**:
   - 解决智能手表 PPG（64 Hz）与 ACC（32 Hz）原生异构采样率问题，按 0.25 秒物理时长精准对齐到相同序列长度的隐空间 Token。
2. **双流跨模态注意力骨干网 (Dual-Stream Cross-Attention Backbone)**:
   - 独立的 PPG 脉搏流与 ACC 运动流，通过双向交叉注意力让 ACC 运动信号作为自适应先验引导消除 PPG 运动伪影。
3. **复合自监督预训练目标**:
   - 掩码时间序列重构 ($\mathcal{L}_{\text{MAE}}$)
   - 跨模态表征对齐对比损失 ($\mathcal{L}_{\text{InfoNCE}}$)
   - 频域谐波保真损失 ($\mathcal{L}_{\text{Spectral}}$)
4. **多任务即插即用下游预测头**:
   - 运动心率估计 (`HeartRateHead`)
   - 步频与步数预测 (`StepCountHead`)
   - 活动与睡眠分期分类 (`ActivityClassificationHead`)

## 🚀 快速上手

### 1. 运行自监督预训练
```bash
python watch_lsm/train_pretrain.py --epochs 10 --batch_size 32 --lr 0.0005
```
权重将保存至 `models/checkpoints/pretrain/best_foundation_model.pt`。

### 2. 运行下游任务微调
- **心率预测线性探测 (Linear Probe)**:
```bash
python watch_lsm/train_downstream.py --task heart_rate --freeze_backbone --epochs 5
```
- **步频估计端到端微调**:
```bash
python watch_lsm/train_downstream.py --task cadence --epochs 5
```
- **日常活动/睡眠分类**:
```bash
python watch_lsm/train_downstream.py --task activity --epochs 5
```
