# Watch_LSM 开发者与 AI Agent 协作中枢 (Agent Hub & Context Anchor)

> **致接手本项目的 AI Agent**：
> 本文件是本项目的**唯一真实事实源（Single Source of Truth）**与**会话级交接中枢**。
> 任何新会话、新 Agent 接入本项目时，**必须首先阅读本文件**，以获得完整的项目上下文、已就绪资产分布、当前工作推进节点及代码规范。
> **任务交接红线**：每次完成阶段性工作后，必须主动更新本文件顶部的【一、当前动态工作交接看板】！

---

## 一、当前动态工作交接看板 (Active Handover & Task State)

*最后更新时间：2026-10-29 11:42（Asia/Shanghai，245/245修复补齐完成，完整AAMI文件清单复核通过）*

> **最新实施状态优先于历史记录**：本机已恢复原 GitHub 提交历史与 `origin`，工作分支为 `codex/multi-device-repair`。用户已明确要求使用 **venv** 并授权修复代码、环境及数据；不要再按 Conda 环境迁移设计。PulseDB 的38份修复及207份补齐已全部完成；11:42复核245/245通过，完整AAMI清单243/243通过文件大小及HDF5首尾抽样验证。正式片段映射与基准评测仍待完成。最新本机结果用 `scripts/data_status.py` 查看（`.local/` 不提交 Git）。项目级操作说明见 [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)。

> 下方任何历史数据“已就绪”项仅描述当时核查的设备，不保证其他设备完整。日期化的迁移审计报告及机器可读错误清单属于原设备记录，不随 Git 分发；共享恢复范围以 `data_manifests/` 为准。

### 1. 当前已完成工作节点 (What Has Been Done)
- [x] **10月29日项目整体状态复核**：本轮迁移的代码入口、独立venv与245份数据修复补齐已完成；现有Watch-LSM为PPG+ACC自监督及心率/步频/活动下游原型，尚未具备完整PulseDB官方划分加载与血压评测闭环。核心14项测试及四组CPU CI的通过证据对应提交 `048e52f`；本轮未重复运行未改动的测试。GitHub实时确认PR #1仍为open draft、未合并，main仍为 `78650d4`；本地未提交修改仅交接文档AGENTS.md，最新数据完成状态尚未推送远端。研究优先级为核验1,340条AAMI片段映射、正式预训练基线接入、受试者隔离评测及指标重测。
- [x] **10月29日数据修复补齐完成并复核**：单worker续传于06:00:44正常结束，AAMI作业207/207通过、失败0；结合首轮修复38份，总计245/245、48,372,845,860字节（45.05 GiB）已验证。11:42后台锁已释放，本地955份波形均通过HDF5打开检查；完整AAMI清单242份患者文件及官方测试索引共243项，文件大小及HDF5首尾波形/标签抽样均通过。报告 `.local/data-status.json`、`.local/aami-file-coverage.json`。现有原始文件未全部对照云端哈希，正式索引片段偏移语义尚未核验。
- [x] **10月29日01:16 后台状态与续传复核**：首轮于00:24结束，最终181/245份通过（修复38/38、补齐143/207），剩余64份均为 DriveFS `Errno 22` 读取失败，未新增通过文件。本地891份波形重新执行 HDF5 打开检查全部通过。已保存首轮日志到 `.local/data-transfer-first-pass*.log`，并以单worker重新启动AAMI后台任务（PID604），锁已持有，正在本地复核并跳过已验证文件后重试失败项；降低并发能否改善云盘读取仍待观察。
- [x] **21:48 本机数据进度复核**：38/38 份损坏文件已修复并验证；AAMI 缺失文件已补齐143/207份。总计181/245份通过校验，已验证约20.66 GiB/计划45.05 GiB；剩余64份中2份传输中、40份待处理、22份本轮重试耗尽（DriveFS 读取报 `Errno 22`）。后台锁仍持有且日志持续增长。本地891份波形 MAT 全部通过 HDF5 打开检查；该扫描不等于全部原有文件已对照云端哈希。
- [x] **Windows/Python 3.12 独立 CPU/GPU venv 实测通过**：CPU 构建及 CUDA 12.4 构建分别安装验证；当前 `.venv` 为 PyTorch `2.5.1+cu124`。`pip check`、14 项测试、4 份模型精确哈希、checkpoint 严格加载与 CPU/GPU 合成前后向均通过。真实 wrist 634/435 窗口完成 CPU、GPU 各一轮预训练及 CPU 一轮下游训练。结果仅用于 smoke check。公共锁已排除本机 wheel 地址和 Torch；更新用 `scripts/lock_environment.py`。构建清单明确 pin 到 `+cpu`/`+cu124`，避免 pip 将另一构建误判为已满足。
- [x] **远端工作分支与多平台检查已就绪**：已推送 `codex/multi-device-repair` 并创建 [PR #1](https://github.com/tujy859/Watch_LSM/pull/1)。首轮 Windows/Linux、Python3.10/3.12 四组 CPU CI 均通过；后续更新以 PR 当前提交的检查为准。主分支尚未合并，其他设备应先检出该工作分支。
- [x] **新设备完整 AAMI 文件清单已建立**：`pulsedb_recovery.json` 明确为本次迁移差集，不能代表空白设备的完整数据集；新设备使用 `pulsedb_aami.json` 的242份患者文件 + 官方测试索引（共243项）。测试防止把缺失差集误作完整定义；正式片段索引语义仍待加载器核验。
- [x] **数据修复已持久化运行**：使用 venv 的 Windows 后台任务按明确清单处理，日志 `.local/data-transfer.log`、`.local/data-transfer.err.log`，状态 `scripts/data_status.py`。本机6份已验证文件在切换后台时复查并跳过；旧失败片段已续传。完整作业仍未完成。
- [x] **恢复源码版本控制**：已在保留全部工作文件的前提下抓取原远端历史并重建本地索引，分支 `codex/multi-device-repair` 基于原 `main`。尚未合并任何远端分支。
- [x] **venv 与 CPU/GPU 入口已实现**：新增 `pyproject.toml`、CPU/CUDA 12.4 构建清单、安装/锁定/诊断脚本、统一路径与设备选择，以及 Windows/Linux CPU CI；安装结束必须通过实际反向步骤才能标记环境就绪。
- [x] **数据修复工具与故障保护已实现**：`scripts/sync_pulsedb.py` 支持明确 profile、断点前缀校验、流式 SHA-256、本地复读、HDF5 首尾抽查、原子替换与进程锁；5 项恢复/失败保护测试已通过。245 份文件的实际修复补齐作业已启动，尚未完成；不以启动作业代替完成状态。
- [x] **迁移代码问题已修复**：训练入口采用项目根目录/本机配置，支持显式设备、worker 与 CPU batch 默认值；预训练实际读取 YAML，微调缺失 checkpoint 时需显式 `--from_scratch`；清理旧硬编码路径与危险的默认全库同步入口。
- [x] **多设备与云端源评估完成**：确认现有远端为 `tujy859/Watch_LSM`（main）。远端 86 份文件本机均存在，除 `.gitignore` 外内容一致或仅换行不同；11 份此前已有的本地同步/下载脚本与 AGENTS.md 未同步至远端。多设备方案见 [`docs/multi_device_and_data_plan_2026-10-28.md`](docs/multi_device_and_data_plan_2026-10-28.md)。本次未修改远端或建立新 Git 历史。
- [x] **Google Drive 源已确认可访问**：连接器和经授权的 G 盘只读访问均正常；先前沙箱内无法访问盘符不代表未挂载。全量元数据扫描确认 5,361 份患者 MAT + 11 份索引，约 602.74 GiB；本机缺失 4,613 份患者文件，38 份损坏文件均存在云端更完整副本。清单位于 `docs/source_assessment_2026-10-28/`。
- [x] **修复与补齐容量已核算**：修复 38 份需完整重拷 12.23 GiB；按 AAMI 文件名映射补齐 207 份需新增 32.82 GiB，两者不重叠。组合传输约 45.05 GiB、净增约 40.71 GiB。仅评估，未批量传输或替换数据；正式索引片段映射仍待验证。
- [x] **云端源内容抽查通过**：对损坏清单中的 `p095776.mat` 完整读取云端 33,035,504 字节并计算 SHA-256，HDF5 首个 PPG 波形读取成功；本次耗时约 150.5 秒，包含云盘拉取等待，后续应先小批量测吞吐。其余源文件仅做元数据核对，未替换本地副本。
- [x] **2026-10-28 迁移核查完成**：逐个打开 PulseDB 748 份波形文件与 11 份索引；另抽查 3 份正常文件的实际 PPG/ECG/ABP 与血压标签，均成功读取。这里只验证打开与抽样内容，未执行全量哈希或全部波形读取。
- [x] **本机模型运行验证通过**：PaPaGei-S/P、Pulse-PPG、Watch-LSM 四份 checkpoint 均通过 `weights_only=True` CPU 反序列化；Watch-LSM checkpoint（epoch 2）严格加载成功，并在 GTX 1660 SUPER 6 GiB 上通过合成输入前向运行。本机已有 Conda `dl` 环境，PyTorch 2.5.1 / CUDA 12.4 可用，但缺少 `h5py`、`mat73`、`gdown`、`wfdb` 等项目依赖。
- [x] **迁移文档差异已记录**：VitalDB 50 份 NPZ 全部可读，合计 9,994 窗口，实际只含 PPG/ECG，没有 ABP；Pulse-PPG 权重实际位于 `models/weights/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl`。实验 1-6 脚本与已有结果均在，但实验 4 使用随机初始化编码器，不能作为已加载 PaPaGei 权重的实验结果。
- 以下为 **2026-10-21 源设备历史交接记录**，完整性与外置备份状态应以本机复核为准：
- [x] **基模权重已就绪**：已下载并验证 PaPaGei-S/P（ICLR 2025, 5.8M/5.0M）与 Pulse-PPG（UbiComp 2025, 28.5M Transformer）。
- [x] **长程连续配对基准已就绪**：
  - `datasets/mimic_perform/`：400 例 ICU 重症患者 10 分钟连续 125 Hz PPG+ECG+Resp 波形。
  - `datasets/vitaldb_1min/`：50 例手术患者 9,994 个无缝对齐 1 分钟连续窗口（166.6 连续小时）。
- [x] **穿戴自由生活多模态已就绪**：WildPPG（216 小时野外 4 部位 PPG + IMU）、PPG-DaLiA（8 类日常动作精修版）。
- [x] **PulseDB v2.0 核心基准已就绪（137.12 GB 本地物理资产）**：
  - 波形数据：748 例 MIMIC-III 患者全部切片（126.88 GB，经 `h5py` 100% 校验结构完整无损）；
  - 官方基准索引（11/11 全量通过校验）：包含 `Train_Info.mat`（3.58 GB, 90.2 万个切片）、`PulseDB_Info.mat`（2.51 GB 全量人口学与生理指标）、`AAMI_Test_Info.mat`（国际标准临床测试集）、`CalFree_Test_Info.mat`（无校准泛化测试集）等。
- [x] **跨设备规约收敛**：确认采用**“全项目相对路径”**机制，杜绝绝对硬编码路径，确保项目整体拷贝至任何 Windows 设备均可即插即跑。
- [x] **移动硬盘极速同步脚本已就绪**：开发了 [`scripts/sync_to_external.bat`](scripts/sync_to_external.bat) 与 [`scripts/sync_to_external.py`](scripts/sync_to_external.py)，基于 Windows 原生 16 线程 `robocopy` 自动过滤 `.venv`（3.3万个碎文件）与缓存垃圾，支持增量断点续传。
- [x] **D 盘根目录镜像备份已就绪**：已全量同步至 `D:\Watch_LSM`（139.41 GB，1,925 个核心文件，0 失败），并实测通过 HDF5 路径验证，可直接作为移动硬盘中转站或独立运行。

### 2. 接棒 Agent 下一步待办事项 (Next Steps for Incoming Agent)
- [x] **完成 AAMI 补齐及失败重试**：10月29日06:00全部完成，11:42复核修复38/38、补齐207/207通过，后台正常结束；955份波形HDF5打开检查及完整AAMI清单243项大小/首尾抽样均通过。无需再次启动下载；本机覆盖报告保存在 `.local/`。
- [ ] **扩展其他硬件与上游验证**：本机 CPU/GPU 与 Windows/Linux CPU CI 已实测；其他具体 GPU、macOS MPS 仍需按实际设备验证。上游 repos/ 历史快照还需独立版本核验，不把核心库通过视为全部基模复现完成。
- [ ] **修复历史实验解释与划分**：路径入口已处理；实验4的正式预训练权重接入、实验4/6受试者隔离划分与指标重测仍待独立研究任务，不把本轮环境恢复当成这些实验已经复现。
- [ ] **核验官方索引片段映射与有效覆盖率**：AAMI 索引1,340条记录含MIMIC674条、VitalDB666条；按 `Subj_Name` 下划线前缀映射的242份患者文件及测试索引已全部在本机通过大小与抽样验证。仍需正式核验名称和片段偏移映射，逐条确认1,340条记录可读取；文件齐全不能代替完整官方AAMI测试已复现。
- [ ] **构建标准相对路径数据加载模块**：在 `watch_lsm/data/pulsedb.py` 中编写基于 `PROJECT_ROOT` 的 PulseDB Dataset 与 DataLoader，能一键读取 `AAMI_Test_Info.mat` 与本地 748 例患者的 10 秒波形切片。
- [ ] **建立 AAMI 血压基准评测脚本**：编写 `scripts/evaluate_aami_bp.py`，加载 PaPaGei 或 Pulse-PPG 预训练表征，评估在 AAMI 标准测试集上的无校准（CalFree）与单点校准（CalBased）收缩压/舒张压 MAE 与 SD。
- [ ] **多模态 PPG+ECG 跨模态重构探索**：基于 `mimic_perform` 与 `pulsedb`，探索单模态 PPG 推理虚拟 ECG / 血压的生成网络。

### 3. 避坑指南与关键技术决议 (Critical Decisions & Caveats)
- ⚠️ **文件数和总容量不能证明完整**：迁移初查发现38份截断 MAT，实际剩余错误随后台修复变化，以本机状态为准。原有可打开文件未全部对照云端哈希。BIDMC/WESAD 的 `.zip` 也不是有效 ZIP，不能当作可用数据集。
- ⚠️ **不要直接复用现有指标作为泛化结论**：实验 4 未加载官方预训练权重；实验 4/6 随机划分具有重叠的滑窗，存在数据泄露风险。已有指标只是历史探索产物。
- ⚠️ **切勿盲目全量下载 PulseDB**：PulseDB v2.0 全量 5,374 个未压缩 MAT 文件实际高达 **602.74 GB**！当前本地下载的 137.12 GB 精选集已包含上百万个样本，足够训练与评测。不要一次性全下以免撑爆磁盘。
- ⚠️ **严格使用相对路径**：代码中严禁写入类似 `r"C:\Coding\Work\Watch_LSM\..."` 的绝对路径！统一使用 `Path(__file__).resolve().parents[...]` 定位项目根目录。
- ⚠️ **防止数据泄露（Data Leakage）**：训练和评测必须严格遵循 PulseDB 官方给出的 `Train_Info.mat` 与 `AAMI_Test_Info.mat` 等索引划分，严禁以片段（Segment）为粒度随机切分训练测试集。

---

## 二、项目使命与技术路线 (Mission & Architecture)

- **核心定位**：聚焦于**智能手表/穿戴端生理信号（PPG、ECG、ACC）多模态基础模型**的自研、表征迁移与评测。
- **主力攻坚下游任务**：
  1. **无袖带连续血压估计（Cuffless Blood Pressure Estimation）**：以连续动脉血压（ABP）为真值金标准，冲击 AAMI/ISO 81060-2 医疗器械临床标准（$\text{ME} \le 5\text{ mmHg}, \text{SD} \le 8\text{ mmHg}$）；
  2. **心血管生物学年龄（Vascular / Biological Age）**：基于脉搏波传导阻抗推测心血管生理老化程度；
  3. **剧烈运动抗伪影心率与节律追踪**：利用 3D 加速度计对抗运动伪影与频域碰撞。

---

## 三、环境依赖与运行规约 (Environment & Conventions)

- **Python 环境**：每机独立 `.venv`，Python 3.10-3.12；不继承系统 site-packages，不复制环境目录。
  - Windows 解释器：`.venv/Scripts/python.exe`；Linux/macOS：`.venv/bin/python`。
  - `python scripts/bootstrap_env.py --backend cpu`，Windows/Linux NVIDIA 可选 `--backend cu124`。
  - 运行时 `--device auto/cpu/cuda/cuda:0/mps`；本机挂载路径、硬件偏好写入被忽略的 `watch_lsm.local.toml`。
  - 核心依赖以 `pyproject.toml` 为准；平台公共依赖锁定与 torch 构建清单分开，完整本机 freeze 保存在 `.local/`。
- **代码路径定位标准范式**：
  ```python
  from pathlib import Path

  # 统一以当前脚本/模块所在位置推算项目根目录
  PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 假设脚本在 scripts/ 下
  DATASETS_DIR = PROJECT_ROOT / "datasets"
  MODELS_DIR = PROJECT_ROOT / "models"
  ```
- **Git 版本控制红线**：
  - 严禁将 `datasets/`、`models/weights/` 等任何大于 10 MB 的二进制数据文件提交进 Git；
  - 所有的索引清单以极小的 JSON/CSV 保存在 `data_manifests/` 或 `docs/` 下。

---

## 四、本地已就绪资产全景大账本 (Verified Local Assets)

### 1. 预训练基模权重 (`models/weights/`)
| 模型名称 | 来源/发表 | 参数量 | 预训练数据规模 | 权重路径 |
| :--- | :--- | :---: | :---: | :--- |
| **PaPaGei-S** | ICLR 2025 (Nokia Bell Labs) | 5.8M | 400k 小时穿戴 PPG | `models/weights/papagei/papagei_s.pt` |
| **PaPaGei-P** | ICLR 2025 (Nokia Bell Labs) | 5.0M | 脉搏波形态专家 | `models/weights/papagei/papagei_p.pt` |
| **Pulse-PPG** | UbiComp 2025 (UIUC) | 28.5M | 百万级连续脉搏波 | `models/weights/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl` |

### 2. 本地数据集大账本 (`datasets/`)
| 数据集名称 | 存放路径 | 规模与内容 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **PulseDB v2.0** | `datasets/pulsedb/` | **137.12 GB 本地物理资产**<br>- 748 例 MIMIC-III ICU 患者连续对齐切片<br>- 11/11 份全量官方评测与训练索引（全部通过 HDF5 校验） | 连续血压金标准评测、AAMI 医疗标准认证、跨人群无校准泛化测试 |
| **MIMIC-PERForm** | `datasets/mimic_perform/` | 400 位重症监护患者，每人 **10 分钟连续未切片** 125 Hz PPG+ECG+Resp（共 268 MB） | 长程连续时序自监督、呼吸率估计、PPG 跨模态生成 ECG |
| **VitalDB 1-min** | `datasets/vitaldb_1min/` | 50 例手术患者，**9,994 个 1 分钟窗口**（166.6 小时 PPG+ECG，没有 ABP；连续块经筛选，不能假定整例无间断） | 长窗口表征与 PPG/ECG 配对研究 |
| **WildPPG** | `datasets/wildppg/` | 216 小时自由生活非受限多部位 PPG + 3D IMU（602 MB） | 穿戴真实运动抗噪、多身体部位（腕/胸/头/踝）域适应 |
| **PPG-DaLiA** | `datasets/ppg_dalia/` | 8 类本地活动文件（坐/走/桌上足球/骑行/楼梯/办公/驾车/进餐）手腕 PPG+ECG+ACC（367 MB） | 运动状态分类、运动伪影消除 |
| **PPG-BP** | `datasets/ppg_bp/` | 219 例患者，含收缩压/舒张压、高血压、糖尿病、脑梗死等 10 项临床疾病标签 | 临床慢病分类与下游少样本微调 |

---

## 五、Agent 交互与协作行为规范 (Rules for Agents)

1. **接盘第一步**：每次接到任务，通读本文件，明确当前处于哪一个工作节点；
2. **代码编写规范**：
   - 保持文档与注释完整；
   - 永远使用相对路径；
   - 输出数值与时间范围时，严禁使用单半角波浪号（使用 `-` 或 `～`），转义文本中的美元符号（`\$`）；
3. **交接最后一步**：在向用户交付成果或会话结束前，**必须自动更新本文件【一、当前动态工作交接看板】中的状态与下一步待办**，为下一位 Agent 铺平道路。
