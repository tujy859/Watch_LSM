# PPG-Bench 正式复现进度与运行说明

2026-10-30（Asia/Shanghai）阶段记录。用户已授权补齐所需模型和数据、运行完整评测；优先使用本机，后续需要时再评估AutoDL等租赁资源。当前尚未完成整篇论文复现，也未开通付费服务器。

## 已验证的结果

全部5个专用PPG模型及PRV、MOMENT共7个对象的权重/接口已就绪，均在GTX1660SUPER上通过真实PPG输入的前向，以及小批量任务头/编码器反向与优化步骤。PRV没有编码器参数，只训练预测头。

PPG-BP的7个对象均已完成线性评测：每项任务5种子、5个外层折，保留4折内部选参及受试者级评分。以下两个血压目标的全部均值，在两位小数上与[论文表1](https://arxiv.org/html/2610.13435v1#S5.T1)一致：

| 对象 | 收缩压MAE（mmHg） | 舒张压MAE（mmHg） |
| :--- | ---: | ---: |
| AnyPPG | 13.3227 | 8.2089 |
| PaPaGei-S | 14.9159 | 8.6957 |
| PaPaGei-P | 14.6018 | 8.5549 |
| PulsePPG | 13.7696 | 8.4880 |
| SigmaPPG | 14.9464 | 8.6271 |
| PRV | 16.3979 | 8.7676 |
| MOMENT-1-small | 13.8866 | 8.5401 |
| 训练集均值对照 | 16.3129 | 8.7898 |

这些结果验证了该数据集的这一评测方式，不代表其他数据集、随机森林、MLP和全量微调已经复现。当前实况运行 `python scripts/ppg_bench_status.py`；原始结果保存在`.local/ppg-bench-reproduction-py312/ppgbp/`和`.local/ppg-bench-reproduction-extra/ppgbp/`。

可随Git查看的全部70项模型/任务汇总见[summary.csv](results/ppg_bench_ppgbp_linear/summary.csv)，逐项核查证据见[audit.json](results/ppg_bench_ppgbp_linear/audit.json)。已检查1,750项评测的五折五种子覆盖、218人划分、训练/测试受试者无交集、各折测试集不重叠、预测受试者身份与有限值；SBP/DBP的MAE已从保存的预测按受试者取均值后重算一致。校验记录含产物集合哈希；原始划分和预测文件仍留本机，不随Git发布。

完整7对象、26任务、规定方法/种子/折数对应15,050个外层评测（每个对象2,150个）；次要任务仅用线性/RF。一个神经评测内部还可能包含3次验证重复与各48次搜索，不能把外层计数当作训练次数。

## 资源与环境

小型来源清单：[ppg_bench_assets.json](../data_manifests/ppg_bench_assets.json)。已有PaPaGei/PulsePPG权重哈希继续见`data_manifests/model_assets.json`。新增AnyPPG、SigmaPPG、MOMENT权重已校验，Sigma/MOMENT权重SHA-256与作者Hugging Face LFS元数据匹配。

| 数据集 | 本阶段状态 |
| :--- | :--- |
| PPG-BP | 已按官方规则预处理218人/654片段，线性评测已运行 |
| RealWorldPPG | 已预处理35人/2074条采集记录，线性评测运行中 |
| PPG-DaLiA | 新下载官方逐受试者原始数据，15人、4种采样率已预处理，线性评测运行中 |
| BIDMC | 官方106份CSV已补齐，53人、4种采样率已预处理 |
| WESAD | 作者原始数据已补齐，15人、4种采样率已预处理 |
| WildPPG | 官方原始归档已补齐，16人、4种采样率已预处理；不使用早期汇总MAT替代 |
| BUT-PPG | 830对有效质量采集记录文件已补齐，按50名受试者分组、4种采样率已预处理 |
| ECSMP | 原始PPG/EEG与标签已补齐，60人、4种采样率已预处理；按官方解析规则跳过039（缺失SDS）及042/048/064（视频起点数异常） |
| DREAMT 2.1.0 | 等待用户账号和该项目数据协议授权 |

新增原始数据均保存在`datasets/ppg_bench_raw/`，不覆盖早期变体与无效ZIP。预处理在`datasets/ppg_bench_processed/`，包含50/64/100/125 Hz版本。下载完成不等于预处理或评测完成；机器状态和后续校验优先于本表快照。

截至本阶段核查，8个无需新增账号授权的数据集的32份采样率输出均已发布上游预处理元数据；DREAMT仍未就绪。状态报告的`preprocessing`字段逐数据集、逐采样率列出就绪状态和划分组数，RealWorldPPG的2074组为采集记录，不能解释为2074名受试者。

最终环境为独立`.local/ppg-bench-venv`，Python3.12.7、Torch2.5.1+cu124；项目原`.venv`未改装。MOMENT旧发行版固定NumPy1.25.2，与PPG-Bench的NumPy>=1.26要求冲突，因此使用固定作者源码`momentfm0.1.5`及Transformers4.54.1，具体提交与直接依赖见清单及`requirements/ppg-bench*.txt`，本机完整freeze位于`.local/ppg-bench-repro/environment-freeze.txt`。这是需要记录的环境差异，不能把不兼容的旧发行包强装后宣称环境一致。

新设备按[环境与资产恢复说明](ppg_bench_setup.md)重新创建venv、获取固定源码并校验离线权重；不要复制当前环境目录。

## 神经评测与平台差异

- 原始冻结MLP协议的一项收缩压外层评测已完成：48次候选搜索、3次验证重复，单折单种子MAE15.2036 mmHg。正式5折5种子MLP队列已启动；相同身份的已完成试跑任务通过完整运行身份比对后复用。
- 原始AMP微调在本机第一项候选出现NaN，保留在`.local/ppg-bench-finetune-capacity/`及对应日志，未跳过失败后伪装成功。
- FP32、单候选并行的诊断中，同一候选的3次验证重复已完成。该诊断只搜索1个候选，不计入完整复现。另已启动保持48次搜索的本机FP32微调队列，输出`.local/ppg-bench-neural-fp32/`；其精度与串行TPE调度差异单独记录，不计为原始AMP设置完成。
- `scripts/ppg_bench_neural_worker.py`可显式记录候选并行数、评估批量、精度与诊断预算；训练批量候选、学习率、epoch和正式搜索次数不会被隐式缩减。FP32结果不得与原始AMP结果混为同一配置。
- Windows下已完成的Optuna数据库句柄可能导致工作目录改名失败。启动器会在子进程退出后确认结果及预测文件齐全，再发布并按上游机制续跑；不改动已完成预测或指标。相关缺失预测/未知任务保护测试已添加。
- 原生Windows没有cuML，当前进程没有管理员权限且未安装WSL。官方RF仍待兼容Linux环境；不能改成scikit-learn RF后直接当作论文后端已复现。

## 本机命令与后台任务

```powershell
# 实况：模型、公开下载、评测计数与未完成项
.venv\Scripts\python.exe scripts/ppg_bench_status.py

# 下载公开数据，保留已完成文件和断点
.local\ppg-bench-venv\Scripts\python.exe scripts/download_ppg_bench.py

# 指定已补齐的数据预处理；已验证输出按上游元数据跳过
.local\ppg-bench-venv\Scripts\python.exe scripts/run_ppg_bench.py preprocess --datasets bidmc wesad

# 完整折数与种子的线性任务；按对象/数据集独立保存，可续跑
.local\ppg-bench-venv\Scripts\python.exe scripts/run_ppg_bench.py run --datasets bidmc wesad --methods linear
```

不要对同一输出目录重复启动队列；启动器使用进程锁。若需要变更配置或精度，使用新的输出目录。当前后台任务已在运行，开始新任务前先看状态。

当前活动输出：

- `.local/ppg-bench-reproduction-py312/`：5个对象的PPG-BP/RealWorldPPG线性队列。
- `.local/ppg-bench-reproduction-extra/`：SigmaPPG、MOMENT的对应线性队列。
- `.local/ppg-bench-auto/`：新数据下载完成后，自动预处理和执行7对象线性队列；当前处理Dalia。
- `.local/ppg-bench-neural-mlp/`：PaPaGei-S完整折数MLP队列。
- `.local/ppg-bench-neural-fp32/`：有明确精度差异的本机微调队列。
- `.local/ppg-bench-repro/`：下载、模型/梯度验证、环境和故障日志。

所有后台Python进程仅在本机持续开机时运行；关闭电脑后需按状态恢复。这里没有创建定时自动化，也没有开启服务器费用。

`scripts/continue_ppg_bench_mlp.py --after-pid <现有MLP启动器PID>`会等待当前PaPaGei-S队列退出，再在同一输出目录继续8份已就绪数据、全部7对象的MLP评测；保留原48次搜索及5折5种子，已完成任务按上游身份检查复用。其状态见`continuation.json`，不与当前MLP作业争用GPU。PID只用于本次等待，不应在重启后直接沿用旧值。当前已启动这一后续队列；DREAMT和RF不包含在其中。

## DREAMT需要用户完成的步骤

1. [登录或找回PhysioNet账号](https://physionet.org/login/)，没有账号则注册。
2. 打开[DREAMT2.1.0](https://physionet.org/content/dreamt/2.1.0/)，在页面底部按提示签署该项目数据使用协议。该版本页面只列出注册账号和项目协议，没有列CITI或credential培训要求。
3. 确认能访问`data_64Hz`目录。不要在聊天或Git里提供密码；下载凭据应仅在本机安全设置。

自动续跑器要求数据目录存在上游下载器完成后生成的`source.json`。手动下载到一半时只有若干CSV不会触发新的队列；完整下载后再按官方清单核对并预处理。

为匹配benchmark使用2.1.0，不自动替换成更新版本。RF环境、原始AMP微调数值稳定性和完整计算预算仍需处理，整篇复现保持未完成状态。
