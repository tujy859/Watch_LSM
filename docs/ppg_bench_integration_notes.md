# PPG-Bench 对 Watch-LSM 的评测接入启示

核查日期：2026-10-30（Asia/Shanghai）。最新状态：已完成PaPaGei-S在GPU上的加载、冻结推理及PPG-BP收缩压单折单种子线性试跑，详见 [首次试跑报告](ppg_bench_smoke_2026-10-30.md)。下文资源实查保留试跑前快照；完整论文复现仍未完成。

## 来源与版本

- 调研入口：[用户调研仓的 PPG-Bench 条目](https://github.com/tujy859/Wearable-Physiological-Foundation-Models/blob/071cd9216d0eb2a262b959667ab33467b9fd1adf/docs/models/smartwatch_models.md#ppg-bench)。
- 原文：[PPG-Bench: A Unified Benchmark for PPG Foundation Models，arXiv:2610.13435v1](https://arxiv.org/html/2610.13435v1)，提交于2026-10-09；本次按预印本对待。
- 官方代码：[mannas-ai/ppg-bench](https://github.com/mannas-ai/ppg-bench)，默认分支为 `master`，本次核查提交 [2bca102b7be2c0a9816a3872314784d2ef92555b](https://github.com/mannas-ai/ppg-bench/tree/2bca102b7be2c0a9816a3872314784d2ef92555b)，包配置版本为 `0.1.0`。
- 已读范围：论文正文与附录相关表格、官方 README、模型适配接口、任务注册表、协议常量、PaPaGei/PulsePPG 配置与编码器、部分预处理和运行记录逻辑。本次不构成完整代码审计。

## 与本项目最有关的设计

PPG-Bench 将预训练编码器、数据预处理、下游学习器和评测协议分开。模型适配器实现 `load_encoder()` 与 `prepare_batch()`，输出形状为 `[batch, embedding_dim]`；输入滤波、归一化等依照具体预训练模型处理，而不是强行共用一种预处理。

其四种评测方式为冻结编码器后接线性模型、随机森林、MLP，以及编码器与任务头联合微调。三个对照为无信息预测、确定性 PRV（脉搏率变异性）特征和 MOMENT 通用时序模型。PRV 特征提取无训练参数，其下游预测器仍需要拟合。

协议采用5个外层折与5个种子。线性/随机森林使用4折内部验证选参；神经方法使用分组隔离的验证留出与48次TPE搜索。通常以受试者为分组单位；RealWorldPPG 闭集身份识别按采集记录隔离，身份仍为共享类别，这是明确例外，不能笼统声称所有任务都跨受试者测试。

PPG-BP 的所有目标在受试者层面评分：先汇总同一受试者的多个窗口预测，再计算指标。回归采用均值聚合，分类的聚合方法在验证阶段选择。

官方运行记录包含配置、数据身份、划分、预测、搜索记录、汇总指标和缓存表征。可参考这些记录字段补齐本项目的实验可追溯性。

## 调研条目需要注意的差异

| 问题 | 原文与当前官方实现的核对结果 |
| :--- | :--- |
| 数据集名单 | 9个数据集为 PPG-DaLiA、DREAMT、WESAD、ECSMP、PPG-BP、BIDMC、BUT-PPG、RealWorldPPG、WildPPG；当前调研条目列入的 TROIKA、BAMI-1/2、Apple Heart Study 不在该名单 |
| 任务与组合数 | 共17个主要任务及9个次要任务；68是17乘4，论文两个总体比例对应主要任务；当前代码的次要任务仅运行线性和随机森林 |
| 统计解释 | 论文报告24/68没有明显优于无信息对照，57/68没有专用PPG模型同时胜过PRV与MOMENT；不解释成所有模型无效，也不把这些比例称为失败概率 |
| 模型规模 | 原文附录表7列 PaPaGei-S约5.8M、PaPaGei-P约5.0M、PulsePPG约28.5M，与本项目交接口径一致；与调研仓部分既有规模说法不一致。本次仅核对引用口径，未重新统计权重参数 |

这些差异已在本项目记录，未修改外部调研仓。论文结论限定于其已发布权重、数据、任务及搜索预算，不能由此证明一种预训练方法普遍优于另一种方法。

用户已确认调研条目的名单问题，并明确由其另行修改外部调研仓；本项目无需代为修改。

## 本地接入前提

1. **先验证已有单模态基线。** 本机已有 PaPaGei-S/P、Pulse-PPG 权重；官方已有这几种模型的适配器。PaPaGei-S 配置为125 Hz、512维，PulsePPG为50 Hz、512维。仍须用本机权重严格加载并检查输出，不能把“文件存在”视为适配成功。
2. **数据布局与覆盖率按本机审计区分。** 10月30日实查已读通PPG-BP与RealWorldPPG主要文件；PPG-DaLiA、WildPPG现有MAT不符合官方原始文件布局，标签与划分等价性仍待核验。WESAD/BIDMC现有ZIP再次确认无效，不能计为可用。详见下方资源核查。
3. **Watch-LSM 的双模态输入需要单独设计。** 当前模型需要 PPG+ACC；PPG-Bench 的标准数据接口提供单通道PPG。必须明确实现PPG单模态/缺失ACC模式，或扩展为独立的多模态实验；额外使用ACC的结果应单独报告，不能直接声称与原单模态榜单同条件。填零ACC也属于需要验证的缺失模态实验。
4. **保留 PulseDB 官方评测路径。** 此套件当前没有PulseDB任务，不能替代1,340条AAMI片段映射核验、官方训练/校准/测试划分及本项目血压评测脚本。
5. **先隔离环境再做兼容验证。** 官方要求Python3.11及以上；线性、MLP和微调支持CPU，当前RF后端需要RAPIDS cuML与受支持的NVIDIA环境。尚未验证其依赖与本项目锁定环境的组合。无需为本次阅读改装现有环境。
6. **采用前记录版本。** 官方自有代码标注GPL-3.0-only，内含模型代码保留对应许可证；具体权重和数据另有来源条款。复用时保留这些来源信息。

## 建议的实施顺序

- 第一阶段：核对PPG-BP原始布局，加载PaPaGei-S或PulsePPG，运行单任务、单外层折、单种子的线性评测，验证数据、权重与输出链路。该步骤仅用于运行检查。
- 第二阶段：为选定任务补齐5折与5种子，加入常数/无信息、PRV与MOMENT对照，保存划分、预测及指标；先建立可信的单模态结果。
- 第三阶段：扩展到PPG-DaLiA/WildPPG的运动心率与活动任务，再评估MLP、微调和Watch-LSM多模态增益。
- 同时继续独立推进PulseDB官方划分与AAMI加载，明确区分两套评测协议。

官方默认全套每个模型包含2,150次外层评测，内部还有调参训练；本机不应直接从全套运行开始。改变折数、种子或搜索预算后，需要明确标注为缩减实验，不能直接当作论文完整复现。

## 本机复现资源核查（2026-10-30，试跑前快照）

结论：已有启动部分任务的原始数据和3份PPG基模权重，但尚不具备完整论文复现所需的全部资源，也没有完成端到端benchmark运行。以下仅对应当前设备，不随Git自动迁移。

| 官方数据集 | 本机实查 | 判定 |
| :--- | :--- | :--- |
| PPG-BP | 219人标签，657份波形全部读取；655份2100点，231号的2份为4200点；各人的年龄/BMI/血压/心率数值标签无缺失 | 可优先接入；官方代码排除231号，按该规则为218人、654份波形。尚未执行完整预处理 |
| RealWorldPPG | train8.xlsx与test8.xlsx读通；1374+700条记录、35人，每条300点且有限 | 可优先接入；布局与官方加载器预期一致，尚未运行正式划分和评测 |
| PPG-DaLiA | 8份按活动组织的MAT均读取data结构，各含14或15条参与者数据；有PPG/ECG/参考标签等字段 | 有数据但非官方S1/S1.pkl等布局；跨活动受试者映射、时间线和标签等价性待核验，不能直接计为标准复现资源 |
| WildPPG | 单个WildPPG.mat，含16条cell条目的PPG/IMU/BPM等变量；0份WildPPG_Part_*.mat | 与官方逐受试者原始文件不兼容；官方流程使用胸部ECG生成HR标签，现有顶层变量未提供同名ECG字段，需核验来源与语义，不能只改名使用 |
| BIDMC、WESAD | 各自现有ZIP均未通过ZIP格式检查 | 不可用，需要恢复有效源文件 |
| DREAMT、ECSMP、BUT-PPG | 当前datasets目录未找到对应资源 | 缺失；DREAMT还涉及PhysioNet准入 |

### 模型、代码与环境

- PaPaGei-S、PaPaGei-P、Pulse-PPG共3份权重本轮重新计算SHA-256，全部匹配`data_manifests/model_assets.json`。这是本地资产一致性检查，尚未证明官方适配器严格加载及运行成功。
- 在项目模型目录、已有上游目录及默认Hugging Face模型缓存中，未发现AnyPPG、SigmaPPG、MOMENT-1-small的对应权重；`repos/moment/`只有源码，历史实验脚本中`from_pretrained`调用不能证明当前缓存存在。
- 当前`.venv`为Python3.12.7、Torch2.5.1+cu124，具备NumPy/SciPy/scikit-learn；未安装ppg-bench、Optuna及模型相关的momentfm/transformers/timm/einops等包。当前官方PRV实现依赖NumPy/SciPy，无需把缺少NeuroKit2单独当成该版本运行阻塞项。
- 项目未纳入PPG-Bench本地检出和评测入口；现有`experiments/`不等于其完整协议实现。
- 硬件实查：GTX1660SUPER 6GiB显存、约31.8GiB内存、项目所在卷约1.06TiB可用。可从CPU线性任务验证开始；神经方法的完整搜索需要再测显存和吞吐，当前没有全套耗时估计。
- 官方RF使用cuML；当前Windows venv未安装cuML/CuPy。RAPIDS的Windows支持路径为[WSL2](https://docs.nvidia.com/datascience/install/#system-requirements)，不能当作现有原生Windows环境已就绪；尚未核查本机WSL配置。改用其他RF后端会改变复现条件。

机器可读审计保存在`.local/ppg-bench-resource-audit.json`，不会提交Git。数据核查覆盖上述文件和结构，不等于全量波形质量、上游哈希或官方指标复现。本次未改原始数据、安装依赖或启动benchmark。

源码仓的`.gitignore`排除了`datasets/`、`models/weights/`、`repos/`和`.local/`；单纯克隆Watch_LSM不能获得这些本机资产。
