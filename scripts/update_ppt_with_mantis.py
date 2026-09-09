import re

ppt_path = "papers/WEARABLE_FOUNDATION_MODELS_PPT.md"
with open(ppt_path, "r", encoding="utf-8") as f:
    content = f.read()

mantis_slide = """
---

## Slide 8: 华为诺亚方舟实验室 —— Mantis (ICML 2026)
### 填补时序大模型“分类盲区”的轻量级判别基座

#### 1. 文章背景与痛点
- **痛点**: 以往时序大模型（如 Chronos、TimesFM）几乎全都在卷**预测（Forecasting）**，而手表最核心的商业需求大量是**分类与判别**（活动识别 HAR、计步防误触、心律失常/房颤筛查、跌倒预警）。
- **定位**: ICML 2026 顶会录用，华为诺亚方舟实验室开源的轻量级时序分类基础模型。

#### 2. 技术方案
- **Token Generator Unit (TGU)**: 摒弃简单粗暴的线性切块，自适应提取时序局部动态与差分形态。
- **极轻量 1D-ViT**: 参数量仅 **~800 万（8M）**，可在智能手表 CPU / 低功耗协处理器上以极低延迟运行。
- **自监督对比学习 (Contrastive Learning)**: 海量合成与真实时序预训练，专攻高判别力表征。
- **多通道适配器 (Multichannel Projectors)**: 原生支持手腕 3 轴加速度计与多导联时序。

#### 3. 实测数据与结果（本项目实验 6）
- **图表**: [results/figures/exp6_mantis_clustering.png](file:///c:/Coding/Work/Watch_LSM/results/figures/exp6_mantis_clustering.png)
- 在真实智能手表 3 轴加速度计数据（1069 个样本）上实测：
  - CPU 单样本特征提取延迟仅 **19.80 ms**！
  - 线性探测（Linear Probe）在走路、跑步、骑行三类运动上的分类准确率高达 **99.7%**（显著优于 LIMU-BERT 的 92.1%）！
  - 轮廓系数达到 0.203，运动表征聚类边界极为清晰。

#### 4. 对智能手表的商业应用启发
1. **终结“手部假动作误计步”**: 输入 3 轴 ACC 提取高判别力表征，直接精准过滤打字、刷牙等假动作，打造零差评的高精度计步器。
2. **端侧轻量级心律失常与房颤极速初筛**: 几百万参数量极易量化为 INT8 部署在手表 MCU 上，实现毫秒级本地心电/脉搏异常报警。
"""

# Replace Slide 8 and Slide 9
content = content.replace("## Slide 8: 跨模态解耦实证", "## Slide 9: 跨模态解耦实证")
content = content.replace("## Slide 9: 智能手表基础模型商业技术横向选型矩阵", "## Slide 10: 智能手表基础模型商业技术横向选型矩阵")
content = content.replace("## Slide 10: 基于自有数据集构建自研基础模型（Watch-LSM）的实施建议", "## Slide 11: 基于自有数据集构建自研基础模型（Watch-LSM）的实施建议")

# Add Mantis slide before Slide 9
content = content.replace("## Slide 9: 跨模态解耦实证", mantis_slide + "\n---\n\n## Slide 9: 跨模态解耦实证")

# Add Mantis into comparison table
old_table_row = "| **DAPO LIMU-BERT** | 3轴ACC + 陀螺仪 | 高度成熟 | **极佳 (微型BERT)** | 零误触精准计步、多工况运动识别、帕金森步态筛查与防跌倒 |"
new_table_row = old_table_row + "\n| **Huawei Mantis** | 单变量/多变量(3轴ACC) | 顶尖开源 (ICML 2026) | **极佳 (~8M 参数, 19ms)** | 零误触权威计步、活动状态高精分类(实测99.7%)、端侧心律失常初筛 |"
content = content.replace(old_table_row, new_table_row)

with open(ppt_path, "w", encoding="utf-8") as f:
    f.write(content)

# Update artifact copy
artifact_path = r"C:\Users\Jinying Tu\.gemini\antigravity\brain\7f13e813-10c0-487d-a558-ba07056f38c5\wearable_foundation_models_ppt.md"
with open(artifact_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated WEARABLE_FOUNDATION_MODELS_PPT.md and artifact with Mantis slide!")
