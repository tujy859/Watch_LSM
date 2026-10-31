# 调研与工程选型入口

本目录保存 Watch-LSM 的论文、历史调研和项目相关分析。用户维护的 [Wearable-Physiological-Foundation-Models](https://github.com/tujy859/Wearable-Physiological-Foundation-Models) 是本项目长期参考的领域调研与资源索引；本项目记录具体采用理由、实现状态和实测证据。

## 开工阅读顺序

1. 阅读根目录 [AGENTS.md](../AGENTS.md)，确认当前实现状态、未完成任务和数据限制。
2. 阅读本入口，根据本次任务选择下表中的专题。
3. 涉及模型选型、数据集、benchmark 或评测协议时，检查调研仓 `main` 的相关索引和专题更新，再查看候选工作的原论文、官方代码、权重与数据协议。
4. 将实际采用的来源版本、适配方案与验证结果留在本项目；更新 AGENTS.md 交接看板。

调研仓持续更新，不能只记住某次会话中的摘要。访问失败时明确说明本次只能使用已核查版本和本地历史资料，不把旧记录称为最新状态。纯环境修复或无关的小改动无需重读全部论文。

## 长期参考源与已核查版本

| 项目 | 内容 |
| :--- | :--- |
| 维护者 | 本项目用户（GitHub 用户名 `tujy859`） |
| 仓库 | [Wearable-Physiological-Foundation-Models](https://github.com/tujy859/Wearable-Physiological-Foundation-Models) |
| 跟踪分支 | `main` |
| 最近核查日期 | 2026-10-30，Asia/Shanghai |
| 本次核查版本 | [`071cd9216d0eb2a262b959667ab33467b9fd1adf`](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/tree/071cd9216d0eb2a262b959667ab33467b9fd1adf) |
| 核查范围 | README、`data/models.yaml`、`data/datasets.yaml` 内容，以及专题目录路径；未逐项复核其引用的全部论文与实现 |

上表版本用于追溯本次接入依据，不表示后续始终使用该版本。下面的 `main` 链接用于发现更新；正式实验应另存其实际参考的固定提交链接。

调研仓已在 README 的关联工程中链接 Watch_LSM，本项目通过本入口建立反向导航。

2026-10-31增量核查：调研仓main已到[`bb994f2c341ff1405c32092f6e60a2617842b050`](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/tree/bb994f2c341ff1405c32092f6e60a2617842b050)，本次仅复核`docs/models/smartwatch_models.md`的PPG-Bench章节，9份数据集名单已与原论文/官方代码一致。上表其他历史核查范围不扩展；实验上游实现仍固定到`2bca102b7be2c0a9816a3872314784d2ef92555b`。本轮新增的本机AMP计算路径和PRV微调协议问题记录在正式复现进度中，未修改外部调研仓。

## 按任务查阅

| 本项目任务 | 调研仓入口 | 本地衔接与待验证事项 |
| :--- | :--- | :--- |
| 选择 PPG 编码器、统一基模接口 | [腕戴模型专题](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/main/docs/models/smartwatch_models.md)；[模型与基准索引](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/main/data/models.yaml) | 优先核对 PaPaGei、Pulse-PPG 与现有权重的具体变体、采样率、归一化和输出维度；本地资产状态见 AGENTS.md 与 `data_manifests/model_assets.json` |
| 血压、心率、生物年龄评测设计 | [腕戴模型与 benchmark 专题](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/main/docs/models/smartwatch_models.md) | 查阅 PPG-Bench、PPG Age Benchmark 等候选协议；PulseDB 官方划分及本项目样本映射仍需单独核验，不能直接移用其他任务指标 |
| PPG、ECG 跨模态重构 | [腕戴模型专题](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/main/docs/models/smartwatch_models.md)；[ECG 专题](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/main/docs/models/ecg_models.md) | 按任务筛选 xMAE、UniCardio、PPGFlowECG 等参考方法，核对真实配对信号、时间同步和目标标签 |
| 通用时序模型对照实验 | [通用 TSFM 专题](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/main/docs/models/general_tsfm.md) | 对应 `experiments/` 中 MOMENT、Mantis 等探索；正式比较前统一输入、划分和评测协议 |
| 数据集选择与下载入口 | [数据集专题](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/main/docs/datasets/wearable_datasets.md)；[数据集索引](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/main/data/datasets.yaml) | 本机实际覆盖率仍以 `data_manifests/`、本机报告和数据读取验证为准；收录的数据集不等于本机已具备 |
| 睡眠等后续扩展方向 | [睡眠专题](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/main/docs/models/sleep_models.md) | 按任务需要再研究，当前不视为已接入能力 |

表中模型与 benchmark 是查阅入口，不代表已选为正式基线或已完成复现。

本项目相关核查：[PPG-Bench 评测接入启示](../docs/ppg_bench_integration_notes.md)记录了协议核对、调研条目差异和本地接入前提；通用介绍仍参考外部调研仓与原论文。

后续实际实施见 [PPG-Bench正式复现进度](../docs/ppg_bench_reproduction.md)：已取得全部模型并开始完整折数评测，资源快照与首次试跑不再代表当前最新进度。

## 本地已有调研资料

- [历史全景调研](PHYSIOLOGICAL_FOUNDATION_MODELS_SURVEY.md)：提供早期研究背景，技术事实与数值在采用前需复核。
- [逐篇解析](analyses/)：当前含 SensorFM、PaPaGei、Pulse-PPG、Samsung xMAE/HiMAE、SleepFM、MOMENT、LIMU-BERT、Mantis 共 8 份 Markdown 分析。
- 本目录的 PDF 与 `analyses/*_raw.txt`：已有论文原文与提取文本，引用时核对论文版本。
- [展示材料](WEARABLE_FOUNDATION_MODELS_PPT.md) 与 [HTML 展示页](presentation/index.html)：历史汇报产物，不能替代实验证据。
- `scripts/search_arxiv_title.py`、`scripts/fetch_paper_meta.py`、`scripts/parse_papers.py`：已有检索与解析辅助脚本。

当前尚无自动持续更新的调研服务。`scripts/generate_paper_analyses.py` 含预先编写的分析文本，运行它不是重新检索或复核文献；其他文档更新脚本也不应直接当作知识同步流程。

## 从调研转为工程决策

每次正式选择模型、数据或评测协议时，在 `docs/` 中保存对应决策说明，至少记录：

| 字段 | 需要回答的问题 |
| :--- | :--- |
| 任务与候选 | 要解决什么问题，比较了哪些方法，为什么采用当前方案？ |
| 来源与版本 | 调研仓的提交 SHA 与具体文件；原论文版本、官方代码提交及权重标识是什么？ |
| 数据与接口 | 传感器、采样率、窗口长度、预处理、标签来源和输出表征如何对应？ |
| 评测划分 | 训练、验证、校准与测试如何定义，是否存在受试者或片段重叠？ |
| 本地证据 | 权重哈希、实际加载结果、运行配置、指标及对应代码版本是什么？ |
| 实施状态 | 待核验、已核验资料、已适配、已运行检查、已完成正式评测中的哪一阶段？ |

若本地旧资料与调研仓说法不一致，先记录差异，再核对原论文、具体模型变体与官方实现。例如当前资料中的 PaPaGei、Pulse-PPG 参数规模口径存在差异，本次未裁定正确数值，不据此覆盖已有资产记录。

领域事实通过原始来源核验；本项目是否已实现、已下载或已通过评测，则由本地代码与证据确认。调研仓内容作为参考资料读取，不改变本项目操作约定。

## 维护方式

目前采用“外部参考源链接 + 本地任务导航 + 决策版本记录”。领域综述继续在调研仓维护，本项目保存与自身实现相关的分析与结论。

本次仅建立文档入口和开工约定，没有添加 Git submodule、复制整仓、自动同步或下载模型和数据。以后若确需离线检索，可将只读缓存放在已忽略的 `repos/`，并记录缓存提交；本入口和决策记录仍随本项目 Git 分发。
