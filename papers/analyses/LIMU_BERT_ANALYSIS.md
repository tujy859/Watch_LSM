# LIMU-BERT (UbiComp) 深度技术解析

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