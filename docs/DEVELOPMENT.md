# 多设备开发、venv 与数据恢复

## 新设备安装

使用 Python 3.10-3.12。每台电脑创建自己的 `.venv`，不复制其他电脑的 Python 环境。

本轮修复在 `codex/multi-device-repair`（[PR #1](https://github.com/tujy859/Watch_LSM/pull/1)）。合并前用下列分支安装；合并后可改用默认 `main`。

```powershell
git clone --branch codex/multi-device-repair https://github.com/tujy859/Watch_LSM.git
cd Watch_LSM
python scripts/bootstrap_env.py --backend cpu
```

Windows/Linux NVIDIA 设备选择 `--backend cu124`。本项目先固定到 PyTorch 2.5.1，分别安装官方 CPU/CUDA 12.4 wheel。安装工具使用项目本地缓存与临时目录，并执行 `pip check`。`--backend auto` 根据可用的 `nvidia-smi` 选择；默认 `cpu` 避免在普通电脑上下载 CUDA 包。

安装结束前会运行合成模型的前向、反向和优化步骤。缓存齐全时可用 `--torch-wheel <已核验wheel>` 与 `--wheelhouse <完整依赖目录>` 离线安装；已有独立 venv 会先检查解释器与隔离状态，再复用，不覆写正在被后台任务使用的 Python 可执行文件。

macOS 使用 `--backend cpu` 安装 PyPI 构建；运行时 `auto` 可选择可用的 MPS。当前安装和测试方案明确区分：CPU/CUDA 为本机验证目标；MPS 为可选择路径，未在本机验证。较新的 NVIDIA 架构或其他加速器不应假定兼容该基线，需要另行选择和测试 PyTorch 版本。

选择构建与选择运行设备是两个不同步骤。CUDA 构建仍可运行 CPU 任务；CPU 构建无法通过 `--device cuda` 获得 GPU 支持。显式请求不可用的设备会报错；`auto` 才会自动选择可用设备。

Windows 无需激活环境即可运行：

```powershell
.venv\Scripts\python.exe scripts/doctor.py --device cpu --train-step
.venv\Scripts\python.exe scripts/doctor.py --device cuda --checkpoint --train-step
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS 对应解释器是 `.venv/bin/python`。`--checkpoint` 要求本机已经有已登记的 Watch-LSM checkpoint，纯源码 clone 上不使用该参数即可完成合成数据检查。

## 配置与路径

复制 `watch_lsm.local.example.toml` 为 `watch_lsm.local.toml`，填写本机配置。真实配置、`.venv`、`.local` 不提交 Git。

- 共享配置、manifest 中使用相对路径。
- 默认数据、模型、结果分别为项目根目录下的 `datasets`、`models`、`results`。
- 本机可以通过配置设置另一个磁盘的数据目录、Google Drive 的实际挂载目录、batch size 和默认设备。
- 环境变量 `WATCH_LSM_DATASETS_DIR`、`WATCH_LSM_MODELS_DIR`、`WATCH_LSM_RESULTS_DIR`、`WATCH_LSM_PULSEDB_SOURCE` 可覆盖对应路径。
- 训练入口支持 `--device auto/cpu/cuda/cuda:0/mps`、`--batch_size` 与 `--num_workers`。Windows 默认 worker 数为 0。显存不够时减小 batch size；CPU 和 GPU 不要求相同配置。

训练入口不依赖当前工作目录：

```powershell
.venv\Scripts\python.exe -m watch_lsm.train_pretrain --device cuda --epochs 5 --batch_size 16
.venv\Scripts\python.exe -m watch_lsm.train_downstream --task heart_rate --device cpu --epochs 5
```

预训练入口实际读取 YAML 的模型、采样、损失和主要训练参数；CLI 覆盖对应项。checkpoint 记录模型配置，后续微调自动恢复。缺少 checkpoint 时微调默认失败，需要随机初始化时显式使用 `--from_scratch`。

Wrist 入口沿用早期小样例的记录划分，不是正式跨人评测协议；真实训练的一轮 smoke check 只证明管道能运行，不证明泛化指标。

## 源码与大资产同步

源码通过分支、提交、GitHub PR 同步。开发前 pull，完成后提交到自己的工作分支并推送，其他设备拉取同一提交。不要用整目录同步覆盖正在开发的代码。

大资产保持在 Git 外，按 `data_manifests` 恢复到每机本地。现有上游 `repos/` 是历史复制的代码快照，缺少可证明的 commit 元数据；当前核心库不依赖它们。MOMENT、Mantis、PaPaGei 等独立复现实验还需专门的上游环境与版本核验，不能将核心 venv 验证称为所有上游实验复现成功。

已存在于 Git 历史的二进制不因 `.gitignore` 自动消失；本轮只移除将来提交中应外置的资产，保留本地文件，不重写远端历史。Wrist 小型示例数据仍留作现有训练示例，完整研究数据按清单管理。

`data_manifests/model_assets.json` 登记 4 份现有模型的确切大小与 SHA-256。`python scripts/restore_model_assets.py --verify-only` 检查本机；`--source-root <离线项目副本>` 可从其他设备副本校验恢复。自研旧 checkpoint 暂可从已存在的 Git 历史恢复，作为迁移过渡；PaPaGei/Pulse-PPG 的原始记录链接用于溯源，不把尚未核对的上游文件冒充本机精确快照。缺少这些权重时，核心合成输入检查仍可运行。

共享的包兼容范围放在 `pyproject.toml`，CPU/CUDA 构建在 `requirements/torch-*.txt`。已验证的操作系统/Python 公共依赖锁定文件放 `requirements/locks/`，不包含 torch；每机完整 freeze 写入 `.local/`，不能将某机 CUDA freeze 直接当作其他平台安装清单。

更新锁定版本时，在通过验证的 venv 中运行 `python scripts/lock_environment.py --write-shared`。该工具读取包名/版本，排除 Torch 与加速器组件；不要直接把含本机 wheel 地址的 `pip freeze` 结果提交为公共锁。

## PulseDB 修复与补齐

`data_manifests/pulsedb_recovery.json` 是本次迁移快照的修复差集：38 份损坏文件、207 份 AAMI 所需缺失患者文件，以及可选的 11 份官方索引。该差集适用于已有复制资产的本机，不能当作空白设备的完整数据集定义。

全新设备使用 `data_manifests/pulsedb_aami.json`：完整列出 242 份患者文件与官方测试索引，不假定另一台设备已有的 35 份患者文件会自动存在。新设备对应命令：

```powershell
.venv\Scripts\python.exe scripts/sync_pulsedb.py --manifest data_manifests/pulsedb_aami.json --profile aami --workers 2
.venv\Scripts\python.exe scripts/data_status.py --manifest data_manifests/pulsedb_aami.json
```

在本机配置中填写 `paths.pulsedb_source` 后：

```powershell
.venv\Scripts\python.exe scripts/sync_pulsedb.py --profile repair-aami --dry-run
.venv\Scripts\python.exe scripts/sync_pulsedb.py --profile repair-aami --workers 2
.venv\Scripts\python.exe scripts/data_status.py --verify-hdf5
```

Windows 后台传输：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_data_repair.ps1
Get-Content .local/data-transfer.log -Tail 20
```

`repair` 只修复 38 份，`aami` 只补齐 207 份，`indices` 用于校验复制 11 份索引。旧的 `sync_pulsedb_from_gdrive.py`、`download_pulsedb_full.py` 已路由到同一安全入口，不再默认遍历全库下载。没有自动全库同步选项。

每个文件都先写 `.part`；中断重启时核对已传前缀后续传。源数据流 SHA-256、本地复读 SHA-256、大小及 HDF5 首尾波形与标签检查通过后，才原子替换正式文件。源文件不改写，校验失败不删除旧文件。重复运行时复查已验证文件哈希并跳过。全局进程锁避免多个修复进程写同一目标。

每机传输状态、失败原因、源流哈希保存在 `.local/pulsedb-sync/`。源哈希证明复制一致性，不能替代上游已发布的真实性校验和；目前只有一份源文件有已固定的独立检查哈希。全量波形质量与全部样本语义不属于首尾抽查的保证范围。

AAMI 清单完成后，仍需要正式的 PulseDB 加载器核验 `Subj_SegIDX`、标签、训练/校准/测试划分。当前修复不会随机重划数据，也不会生成声称达到医疗标准的评测结果。
