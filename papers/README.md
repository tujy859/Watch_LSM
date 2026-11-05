# 调研与工程选型入口

2026-11-05最新实施见[四步正式扩展结果](../docs/ppg_formal_expansion_results.md)：用户已明确批准，六模型、统一预测器、随机Any、CNN及传统对照正式接入HR、分类、SDS健康增量与临床任务；逐搏/伪影/去噪使用已验证时序接口的子集，未开展预训练或编码器微调。BIDMC按46名真实患者重新隔离；PulseDB固定100/20/42人子集，非全库AAMI。SpO₂移出正式范围。最新调研仓`348a1736457050e611508fa4a324959590bb22fc`相较小样本记录`9882df0`仅网站/文档样式等变化，相关模型和数据专题未变。旧全套队列不恢复。

本目录保存 Watch-LSM 的论文、历史调研和项目相关分析。用户维护的 [Wearable-Physiological-Foundation-Models](https://github.com/tujy859/Wearable-Physiological-Foundation-Models) 是本项目长期参考的领域调研与资源索引；本项目记录具体采用理由、实现状态和实测证据。

2026-11-04用户明确当前定位为**先评测、后训练的手表PPG研究PoC**。任务、强对照、数据边界及最小里程碑建议见[评测PoC讨论稿](../docs/ppg_evaluation_poc_plan.md)；调研仓中的生产级架构建议不作为本项目当前必做项。

第一阶段已按该目标实施，实测证据与复现入口见[腕部HR评测报告](../docs/ppg_hr_poc_phase1.md)：DaLiA/WildPPG、冻结基模与传统/CNN对照、少标签和单向跨数据集评测。WildPPG参考已在独立PoC中按官方ECG函数纠正，原PPG-Bench预处理保留，后续使用时须区分两套标签版本。

第二阶段见[跨任务综合图与报告](../docs/ppg_phase2_results.md)：补齐EMBC21/Biagetti的人工腕部PPG起始点与完整BUT-PPG质量标签。DaLiA/Wild的ECG参考另做检测器核查，但跨模态毫秒配准未解决，严格ECG事件试跑只作为诊断。真实人工起始点、60秒PRV算法一致性和手机SQI分别报告，不能互相替代或直接推出临床HRV能力。

2026-11-05补充[模型覆盖、选型与Apple/Google论文对照](../docs/ppg_model_scope_and_literature_analysis.md)：区分第一阶段三个PPG基模加MOMENT、第二阶段AnyPPG案例与历史五个PPG模型版本；重点复核Apple ICLR 2024的健康信息增量证据，及LSM、SensorFM、WBM的输入层级与适配方式。本次调研仓参考为[`0cdfed1a0b552de39b30d3fcb011c6e3a2e2baf7`](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/tree/0cdfed1a0b552de39b30d3fcb011c6e3a2e2baf7)，核对`data/models.yaml`及腕戴模型、腕戴迁移、纵向可穿戴三个专题，关键结论回到原论文；未修改外部仓或运行新实验。

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

## PPG 噪声处理资源核对（2026-11-04，第二阶段实施前快照）

以下保留当时资源核查，后续已补齐BUT完整质量数据与EMBC21人工提点，当前状态以[第二阶段报告](../docs/ppg_phase2_results.md)为准。

本次核对回答“哪些资源已经进入本项目视野”，不代表启动去噪实验。检查了本仓调研、脚本、清单、当前数据目录和固定 PPG-Bench 实现；本机只读统计文件与元数据，没有重新加载全部波形或刷新评测进度。

外部调研仓参考提交为 [`229db45b5e1a4cb86c52e491da66dc5ea99c4b57`](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/tree/229db45b5e1a4cb86c52e491da66dc5ea99c4b57)，本轮读取 `data/datasets.yaml`、`docs/datasets/wearable_datasets.md` 和 `docs/models/smartwatch_models.md`。其中已收录 DaLiA、WESAD、TROIKA、BUT-PPG；这些入口未检出 EPFL Synthetic、MIMIC-III-Ext-PPG、SEGADE 或用户给出的伪影检测 benchmark。此结论限定于上述检查范围。

| 资源 | 本轮核对前的覆盖 | 当前本机资产与任务 | 距离噪声评测的缺口 |
| :--- | :--- | :--- | :--- |
| EPFL Synthetic PPG | 本仓及上述外部索引未检出 | 未发现对应数据、加载器或评测入口 | 可补成对波形重建测试，但不能用单一基础波形的多个噪声版本证明跨人泛化 |
| PPG-DaLiA | 已调研、下载并接入 PPG-Bench | 官方布局15份受试者PKL，50/64/100/125 Hz输出各15人；任务为心率、活动分类 | 未接入SEGADE逐点伪影标签；原始数据和标签对齐需另外核验；现有任务不提供干净PPG重建真值 |
| TROIKA / IEEE SPC 2015 | 调研仓已收录，本仓有[`find_troika.py`](../scripts/find_troika.py)查找入口 | 本次未发现对应数据目录或正式加载器；不在固定PPG-Bench的9数据集名单中 | 尚未接入强运动HR评测；正式采用时区分经典12人数据与竞赛其他部分 |
| WESAD | 已调研、下载并接入 PPG-Bench | 15份受试者PKL，4种采样率输出各15人；任务为状态四分类、效价和唤醒度回归 | 尚无独立去噪、质量标签或ECG参考HR评价流程；状态标签不是噪声标签 |
| BUT-PPG v2.0 | 已调研、部分下载并接入 PPG-Bench | 完整3,888行质量标注；本机830/830条Good的PPG文件对，Poor为0/3,058；预处理各采样率覆盖50人 | 当前仅有Good波形，无法直接进行Good/Poor分类；已有任务为血糖、收缩压、舒张压、SpO2回归 |
| MIMIC-III-Ext-PPG v1.1.0 | 本仓及上述外部索引未检出 | 未发现对应资产、清单或加载器；已有MIMIC-PERForm、PulseDB不等于此数据集 | 新的临床泛化候选；需独立准入、数据身份及划分核验 |
| SEGADE标注 / PPG Artifact Detection Benchmark | 本次新增关注 | 未发现本地标注、适配或运行结果 | SEGADE存在公开代码与标注文件；用户给出的benchmark当前只有README，不能直接运行 |

### 两个影响实验设计的实际缺口

**BUT-PPG已有资源只覆盖质量通过的子集。** [`download_ppg_bench.py`](../scripts/download_ppg_bench.py)与固定上游下载器都只下载 `Quality == 1` 的PPG；上游预处理再次执行同一过滤。本轮按 `quality-hr-ann.csv` 逐ID检查 `.dat/.hea` 文件对，得到Good 830、Poor 0，不能将此前“PPG-Bench数据就绪”理解为全部3,888条质量分类资源就绪。PhysioNet v2.0说明共有50人，Good/Poor依据HR是否可可靠估计；外部调研索引仍写48人，应按官方版本纠正，尚未修改外部仓。[官方说明](https://physionet.org/content/butppg/2.0.0/)

**DaLiA的现有任务不能直接代替伪影检测。** 固定PPG-Bench提交仍为 `2bca102b7be2c0a9816a3872314784d2ef92555b`，其DaLiA预处理采用8秒窗口、2秒步长，并排除低振幅与混合活动窗口；SEGADE说明采用30秒片段和逐点二值mask。需要从原始数据恢复受试者与片段位置，核验标注映射及覆盖，再建立独立评测。现有PPG-Bench处理产物仅含任务所需PPG与标签，ACC辅助方案也需另外接入。[PPG-Bench预处理](https://github.com/mannas-ai/ppg-bench/blob/2bca102b7be2c0a9816a3872314784d2ef92555b/ppg_bench/preprocess.py)、[SEGADE说明](https://github.com/chengstark/Segade)

### 新来源核验与采用边界

- [PPG Artifact Detection Benchmark](https://github.com/marcellosicbaldi/PPG_artifact_detection_benchmark/tree/b307686fb8274caa3cd531e5e0be4a8c54574607)：核对main提交 `b307686fb8274caa3cd531e5e0be4a8c54574607` 的完整Git树，只有一个 `README.md`。README描述50种算法综述、SEGADE标注、LOSO及6种比较方法，但其中列出的 `src/`、`scripts/` 和结果目录实际未发布；不能把说明当成已经可复现的实现或已核验结果。
- [SEGADE原项目](https://github.com/chengstark/Segade)：完整Git树 `f5b2ad6e180a1474bc47d6c5ec5d482edaed6797` 确实列有DaLiA训练/测试、WESAD、TROIKA的 `scaled_ppgs.npy` 和 `seg_labels.npy`，以及评测代码。本次只核对README与文件元数据，没有下载数组、验证标签内容或验证受试者独立性；不能直接假定其既有折文件符合新的LOSO协议。伪影mask也不等于干净波形真值。
- [EPFL Synthetic官方入口](https://www.epfl.ch/labs/esl/index-html/datasets/ppgdataset/)：官方搜索摘要确认一段72秒、250 Hz同步PPG/ECG基础采集记录。页面直接访问超时，本轮未复核“340条合成版本”或下载包内容。可作为成对去噪的小型补充候选，不标记为本机可用。
- [MIMIC-III-Ext-PPG v1.1.0](https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/)：官方页面确认6,189名患者、约630万个30秒片段、同步信号按可用性提供及信号质量指标，采用Credentialed Access。它是独立发布的数据资源，既有MIMIC-PERForm/PulseDB文件不能算作已经接入它，也不构成其访问授权。

按本项目现有资产，建议优先核验SEGADE到DaLiA原始时间轴的映射，其次为BUT-PPG补齐Poor子集并建立独立质量任务，再增加EPFL配对波形测试与TROIKA强运动外部测试；MIMIC-III-Ext-PPG留作临床扩展。已有WildPPG和PhysioNet wrist也可纳入运动鲁棒性设计，但历史wrist划分问题仍需修复。上述为后续候选顺序，本轮仅核查并登记，没有下载新数据、安装依赖、修改评测代码或操作既有队列。

## 维护方式

目前采用“外部参考源链接 + 本地任务导航 + 决策版本记录”。领域综述继续在调研仓维护，本项目保存与自身实现相关的分析与结论。

本次仅建立文档入口和开工约定，没有添加 Git submodule、复制整仓、自动同步或下载模型和数据。以后若确需离线检索，可将只读缓存放在已忽略的 `repos/`，并记录缓存提交；本入口和决策记录仍随本项目 Git 分发。
