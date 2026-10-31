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

10月31日新增BIDMC阶段结果：AnyPPG与PaPaGei-S各完成2任务×25折种子，共100项线性评测。已核查53人划分无交集、预测身份和有限值，并从窗口预测重算MAE；四组均值和Student-t 95%置信区间均与[论文表1](https://arxiv.org/html/2610.13435v1)在两位小数上一致：AnyPPG呼吸率2.56±0.18、SpO2 2.85±0.17；PaPaGei-S分别2.44±0.08、2.54±0.06。精确数值及证据见[AnyPPG](results/ppg_bench_bidmc_linear_partial/AnyPPG.csv)、[PaPaGei-S](results/ppg_bench_bidmc_linear_partial/PaPaGei-S.csv)和[100项核查记录](results/ppg_bench_bidmc_linear_partial/audit.json)。BIDMC其余对象仍在运行，本段不是全部模型已完成的声明。

7对象、26任务、规定方法/种子/折数的组合原先按15,050个名义报告格估算。10月31日进一步核验发现，PRV编码器参数数目为0，上游`build_finetune_model`明确拒绝无参数编码器，因此不能给它安排17任务×25折种子的425项编码器微调。排除这部分后，独立计算计划为14,625项；425个PRV报告格另列为协议核验项，不自动计为完成。[论文表10/11](https://arxiv.org/html/2610.13435v1)的PRV均值和置信区间一致，这支持复用MLP对照的推断，但本项目尚未把该推断当作已确认的逐折复用规则。启动器会将PRV微调标记为`requires_prv_protocol_clarification`，不会伪造训练。次要任务仅用线性/RF；一个神经外层评测内部还可能包含3次验证重复与各48次搜索，外层计数不能当作训练次数。

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
| DREAMT | 用户已完成注册/申请并指定2.2.0；直连和网页登录验证通过，但文件清单仍403，尚未下载数据。用户表示可暂缓，已退出密码会话；2.2.0将独立保存 |

新增原始数据均保存在`datasets/ppg_bench_raw/`，不覆盖早期变体与无效ZIP。预处理在`datasets/ppg_bench_processed/`，包含50/64/100/125 Hz版本。下载完成不等于预处理或评测完成；机器状态和后续校验优先于本表快照。

截至本阶段核查，8个无需新增账号授权的数据集的32份采样率输出均已发布上游预处理元数据；DREAMT仍未就绪。状态报告的`preprocessing`字段逐数据集、逐采样率列出就绪状态和划分组数，RealWorldPPG的2074组为采集记录，不能解释为2074名受试者。

最终环境为独立`.local/ppg-bench-venv`，Python3.12.7、Torch2.5.1+cu124；项目原`.venv`未改装。MOMENT旧发行版固定NumPy1.25.2，与PPG-Bench的NumPy>=1.26要求冲突，因此使用固定作者源码`momentfm0.1.5`及Transformers4.54.1，具体提交与直接依赖见清单及`requirements/ppg-bench*.txt`，本机完整freeze位于`.local/ppg-bench-repro/environment-freeze.txt`。这是需要记录的环境差异，不能把不兼容的旧发行包强装后宣称环境一致。

新设备按[环境与资产恢复说明](ppg_bench_setup.md)重新创建venv、获取固定源码并校验离线权重；不要复制当前环境目录。

## 神经评测与平台差异

- 原始冻结MLP协议的一项收缩压外层评测已完成：48次候选搜索、3次验证重复，单折单种子MAE15.2036 mmHg。正式5折5种子MLP队列已启动；相同身份的已完成试跑任务通过完整运行身份比对后复用。
- 原始AMP微调在本机第一项候选出现NaN，保留在`.local/ppg-bench-finetune-capacity/`及对应日志，未跳过失败后伪装成功。
- 10月31日进一步定位：使用原先失败的PaPaGei-S收缩压候选、seed42/fold0，仅在外层训练集内划分417个训练窗口和105个验证窗口。cuDNN9.1的FP16路径在`basicblock_list.17.conv2.conv`首先出现非有限输出；对相同输入和权重，原生CUDA FP16和cuDNN FP32均有限，输出最大绝对值分别6.16797和6.16845，二者最大绝对差0.002214。关闭cuDNN、保持AMP的单候选，以及FP32对照，都完成15轮验证。证据见[同层计算对比](results/ppg_bench_amp_diagnostic/kernel-comparison.json)及[15轮精度对照](results/ppg_bench_amp_diagnostic/precision-comparison.json)。这将问题定位到本机这一cuDNN/FP16计算路径，尚不能推断其他显卡或软件版本的行为；全程未使用外层测试集选方案。
- 已另行启动原生CUDA卷积的AMP试跑：PaPaGei-S收缩压、seed42/fold0，仍保留48次候选搜索、3次验证重复和15轮上限，尚待完成。输出`.local/ppg-bench-amp-native-pilot/`含独立`backend-profile.json`及`execution-profile.json`，不并入原始cuDNN或FP32的结果计数。该试跑只改变卷积后端、候选并行数和评估批量；不从通过单候选诊断推断完整微调已复现。
- FP32、单候选并行的诊断中，同一候选的3次验证重复已完成。该诊断只搜索1个候选，不计入完整复现。另已启动保持48次搜索的本机FP32微调队列，输出`.local/ppg-bench-neural-fp32/`；其精度与串行TPE调度差异单独记录，不计为原始AMP设置完成。
- `scripts/ppg_bench_neural_worker.py`可显式记录候选并行数、评估批量、精度与诊断预算；训练批量候选、学习率、epoch和正式搜索次数不会被隐式缩减。FP32结果不得与原始AMP结果混为同一配置。
- `scripts/run_ppg_bench.py --convolution-backend native --neural-trial-workers 1`通过独立后端启动器设置`cudnn.enabled=False`，其余精度仍由`--precision`明确控制；改变后端时必须使用新输出目录，不替换已运行配置。
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
- `.local/ppg-bench-auto/`：已就绪数据的7对象线性队列；当前并行处理Dalia与BIDMC，后续BUT-PPG/ECSMP/WESAD/WildPPG排队。
- `.local/ppg-bench-neural-mlp/`：PaPaGei-S完整折数MLP队列。
- `.local/ppg-bench-neural-fp32/`：有明确精度差异的本机微调队列。
- `.local/ppg-bench-amp-native-pilot/`：保留AMP、改变卷积后端的完整搜索单外层试跑，独立计数。
- `.local/ppg-bench-repro/`：下载、模型/梯度验证、环境和故障日志。

所有后台Python进程仅在本机持续开机时运行；关闭电脑后需按状态恢复。这里没有创建定时自动化，也没有开启服务器费用。

线性续跑器支持`--parallel-datasets 2`。本次在保留Dalia子队列的情况下替换旧协调进程，以`--adopt dalia=<已核实的活动启动器PID>`接管等待，不重新训练Dalia；BIDMC进入第二个并行位置。并行只涉及独立数据集，模型内部划分、搜索和指标不变。`--adopt`必须使用当次核实的PID，不能从旧交接或上次开机盲目复制。协调器确认全部7对象的目标scope完成后才标记数据集完成，进程退出本身不能作为成功证据。状态中已就绪但暂未运行的数据标记`queued`。

`scripts/continue_ppg_bench_mlp.py --after-pid <现有MLP启动器PID>`会等待当前PaPaGei-S队列退出，再在同一输出目录继续8份已就绪数据、全部7对象的MLP评测；保留原48次搜索及5折5种子，已完成任务按上游身份检查复用。其状态见`continuation.json`，不与当前MLP作业争用GPU。PID只用于本次等待，不应在重启后直接沿用旧值。当前已启动这一后续队列；DREAMT和RF不包含在其中。

## DREAMT 2.2.0下载与账号验证

用户于10月31日告知已注册账号、完成数据集申请，并明确要求下载[DREAMT2.2.0](https://physionet.org/content/dreamt/2.2.0/)。该版本官方说明修正了ACC与PSG数据对齐，可能影响睡眠标注；与此前固定的2.1.0不是同一数据版本。下载保存到`datasets/ppg_bench_raw/dreamt-2.2.0/`，不覆盖`dreamt/`或静默触发原2.1.0队列。后续需明确记录版本差异及预处理位置，再开展2.2.0评测。

采用官方[MIT-LCP/physionet](https://github.com/MIT-LCP/physionet)客户端0.2.1，源码固定`54ebd21308c368a9d1e8cfd8a673b111cacead0e`，独立环境`.local/physionet-venv`，不修改benchmark环境。新设备需先获取该固定源码并安装到新venv；本机完整依赖清单见`.local/physionet-environment-freeze.txt`。

当前Windows默认路由经过EFan TUN，因此只设置`--noproxy`不能证明绕过VPN。`scripts/download_dreamt_direct.py`使用忽略系统代理的Requests会话，将IPv4源地址与`IP_UNICAST_IF`一起绑定到指定物理网卡；预检读取实际socket的本地地址和网卡索引。官方客户端下载函数更换重试adapter时，也保留该绑定。拒绝代理参数与跨源重定向，直连失败时停止，不回退VPN；不修改全局路由或VPN配置。WLAN直连PhysioNet的HTTPS预检已实测200。

Windows交互启动（本机网络接口为WLAN，其他设备可用`-InterfaceAlias`指定）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_dreamt_download.ps1
```

程序先验证直连，再在该终端隐藏输入密码。密码只留在当前进程内存，不写入命令参数、环境变量、聊天或状态文件。优先下载`data_64Hz/`，随后继续其余文件；沿用官方SHA-256、续传和失败重试。数据文件访问被拒绝时，程序进一步验证普通网页登录；若登录成功但文件仍403，则等待用户在网页确认该版本协议/权限后按回车重试，不自动签署协议，不将登录成功当作数据权限已获批。

下载状态位于`.local/dreamt-download-status.json`，取得权限后才会生成`.local/dreamt-2.2.0-manifest.json`校验清单。`source.json`仅在全部下载成功后生成。本轮Basic认证及普通网页登录后的文件清单请求均返回403，网页登录已确认成功；尚不能下载该版本文件，也不能从登录成功推断已具备项目访问权限。用户表示该数据集可暂缓，已退出等待重试的密码会话，未下载数据、未保存密码或Cookie。后续需要时在新本机终端输入凭据并核验项目协议状态。

对当前Watch-LSM研究优先级的判断：DREAMT对睡眠分期和夜间PPG+ACC迁移有价值，100Hz文件还包含同步PSG/ECG等参考信号；但没有本项目血压评测所需的连续ABP标签，也不是剧烈运动抗伪影数据。当前可优先推进其余8份数据与PulseDB血压闭环；完整PPG-Bench报告须明确暂缺睡眠分期任务。官方目录是`data_64Hz`和`data_100Hz`：前者仅穿戴信号及睡眠标签，后者额外包含PSG信号；不能把重采样后的统一频率理解为所有传感器的原生采样率。

自动续跑器要求数据目录存在上游下载器完成后生成的`source.json`。手动下载到一半时只有若干CSV不会触发新的队列；完整下载后再按官方清单核对并预处理。

用户指定的2.2.0按独立版本下载；原论文结果对照仍须说明数据版本不同。RF环境、AMP适配试跑和完整计算预算仍需处理，整篇复现保持未完成状态。
