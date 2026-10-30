# PPG-Bench 首次 GPU 与线性评测试跑

日期：2026-10-30（Asia/Shanghai）。本次完成单任务、单种子、单外层折运行检查，不属于完整论文复现。

## 策略与版本

- 直接使用已有 PaPaGei-S 官方预训练权重；不重新预训练基础模型，不微调编码器。冻结表征后，在训练数据上拟合 Ridge 线性预测器。
- 使用官方 PPG-Bench 的预处理、模型适配器、内层选参与受试者级评分，未修改上游实现。
- 上游版本：[`mannas-ai/ppg-bench@2bca102b7be2c0a9816a3872314784d2ef92555b`](https://github.com/mannas-ai/ppg-bench/tree/2bca102b7be2c0a9816a3872314784d2ef92555b)，本机检出于 `repos/ppg-bench/`。
- 权重：`models/weights/papagei/papagei_s.pt`；SHA-256：`79d68671e51bdc951fee853e5a8241099e38148ff26ed888b843849332e0b988`，运行前后文件哈希一致；官方适配器严格加载成功。
- 配置：[ppg_bench_papagei_s.yaml](../watch_lsm/configs/ppg_bench_papagei_s.yaml)，125 Hz、512维表征。
- 独立环境：`.local/ppg-bench-venv/`，不继承系统包；原项目`.venv`未改装。Python3.12.7、Torch2.5.1+cu124、NumPy1.26.4、SciPy1.14.1、scikit-learn1.5.2、Optuna4.9.0。两个环境的`pip check`均通过；完整实验环境版本保存在本机输出目录。

## 数据与预先指定的划分

任务为 PPG-BP 收缩压回归（`ppgbp-sysbp`），使用 `linear`，种子42，外层fold0。没有根据测试结果更换种子或折。

官方规则排除231号受试者后，218人、654份片段完成预处理。1000 Hz、2100点原始片段经官方重采样产生125 Hz、263点输入。训练174人/522份片段，测试44人/132份片段，受试者交集为空。

保留官方4折分组内部验证，用训练部分选择Ridge正则强度，选中`alpha=100`。测试前按每名受试者对其3份片段预测取均值，再计算指标。

## 实测结果

| 方案 | 测试集MAE（mmHg） | 测试集RMSE（mmHg） |
| :--- | ---: | ---: |
| 冻结PaPaGei-S + Ridge | 15.2132 | 18.4483 |
| 训练集均值常数预测 | 16.5888 | 19.6669 |

这个单折中MAE减少约1.38 mmHg。仅说明当前小规模流程能工作，不能据此宣称统计显著、达到血压临床标准或复现论文汇总指标。单折单种子不报告跨折/跨种子置信区间；本次也没有运行PRV、MOMENT或其他PPG模型。

GPU实查为GTX1660SUPER，编码器位于`cuda:0`。实际输入`[32, 1, 263]`，输出`[32, 512]`，全部有限；编码器参数量5,785,612，冻结推理前后模型状态一致。线性拟合运行在CPU。

单批GPU前向预热后约19.66 ms；PyTorch峰值已分配显存58.38 MiB、已保留显存68 MiB。该显存数值不包括完整驱动上下文；该批次计时不包括CPU滤波与数据传输，不能代表其他窗口长度、微调或全套评测开销。

官方预处理约4.82秒；官方单任务评测命令约6.49秒（含该任务全部表征提取、内层选参及评分，不含环境安装和独立GPU检查）。

## 警告与结果交叉检查

调参日志含4条Ridge矩阵条件数警告，未修改官方搜索空间或屏蔽警告。对最终选定`alpha=100`的预测器，用同一特征以float64重新拟合后，与官方输出的最大窗口预测差异约0.000119 mmHg，平均差异约0.000016 mmHg；该诊断没有替换官方结果。保存预测重新计算的受试者级MAE与结果JSON一致。

上游权重加载还输出PyTorch关于未来`weights_only`默认值的提示。本次使用此前已经核验并匹配哈希的本地权重，没有改动上游代码。

## 本机产物与重复运行

详细本机产物位于被Git忽略的`.local/ppg-bench-smoke/`：

- `gpu-check.json`：真实GPU输出、冻结状态、显存、划分和权重哈希。
- `verification.json`、`result-crosscheck.json`：运行状态与预测交叉检查。
- `results/summary.csv`、`results/run.json`：结果与运行身份。
- `results/jobs/`：原始划分、测试预测和官方指标。
- `processed/`：预处理后的PPG-BP数据。
- `environment-freeze.txt`、`preprocess.log`、`benchmark.log`：环境与日志。

已准备好本机独立环境、固定上游源码与本地资产后，可从项目根目录重复官方命令：

```powershell
.local\ppg-bench-venv\Scripts\python.exe -m ppg_bench preprocess ppgbp --fs 125 --raw-root datasets/ppg_bp --output .local/ppg-bench-smoke/processed
.local\ppg-bench-venv\Scripts\python.exe -m ppg_bench run --model watch_lsm/configs/ppg_bench_papagei_s.yaml --datasets ppgbp-sysbp --processed .local/ppg-bench-smoke/processed --methods linear --seeds 42 --folds 0 --device cuda --batch-size 32 --output .local/ppg-bench-smoke/results
```

相同输出目录会跳过已完成任务；新实验或要实际重新执行时应选择新输出目录。上述命令假定本机环境与资产已存在；Git克隆不会带上这些被忽略的目录。

下一阶段可在授权范围内补齐5折5种子及PRV/MOMENT等对照。本次没有启动全套实验。
