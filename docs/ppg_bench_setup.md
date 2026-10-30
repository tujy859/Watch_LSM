# PPG-Bench独立环境与资产恢复

本页对应[正式复现进度](ppg_bench_reproduction.md)中已验证的Python3.12 / CUDA12.4环境。以下为新设备操作，不要在当前后台队列运行时重装环境。项目核心`.venv`与benchmark的`.local/ppg-bench-venv`分开。

## 固定外部源码

在项目根目录执行；已有目录先检查未提交修改，不要覆盖。采用的提交由`data_manifests/ppg_bench_assets.json`记录：

```powershell
git clone https://github.com/mannas-ai/ppg-bench.git repos/ppg-bench
git -C repos/ppg-bench checkout --detach 2bca102b7be2c0a9816a3872314784d2ef92555b
git clone https://github.com/Ngk03/AnyPPG.git repos/anyppg
git -C repos/anyppg checkout --detach 661b877aa96eac3bba320a462cb2a3bfea991103
git clone https://github.com/ZonghengGuo/SigmaPPG.git repos/sigmappg
git -C repos/sigmappg checkout --detach b1cc8d90da3878958b890c6e352cd3ae2a644005
git clone https://github.com/moment-timeseries-foundation-model/moment.git repos/moment-benchmark
git -C repos/moment-benchmark checkout --detach 38f7310ad594100747ca2a8357e9c7ca7d323e0e
```

## 创建环境

需要本机独立安装的Python3.12及适配的NVIDIA驱动。在项目根目录：

```powershell
py -3.12 -m venv .local/ppg-bench-venv
.local/ppg-bench-venv/Scripts/python.exe -m pip install --upgrade pip
.local/ppg-bench-venv/Scripts/python.exe -m pip install -r requirements/ppg-bench-cu124.txt
.local/ppg-bench-venv/Scripts/python.exe -m pip install -r requirements/ppg-bench.txt -e repos/ppg-bench -e repos/moment-benchmark
.local/ppg-bench-venv/Scripts/python.exe -m pip check
```

Linux使用`python3.12 -m venv`和`bin/python`；上述CUDA构建仍须与实际驱动兼容。这里不安装cuML：论文RF要求上游`rf`额外依赖所固定的Linux/cuML后端，需要在相应Linux环境单独验证。不要在Windows中安装一个不同RF实现后混入原始结果。

不要安装`ppg-bench[moment]`替代上述组合：上游extra的Transformers4.40.2与当前MOMENT源码要求不一致。已验证的兼容选择为固定MOMENT源码0.1.5和Transformers4.54.1；这项环境差异在正式报告中保留。

## 恢复权重和数据

代码仓不分发原始数据、权重或外部源码。从已校验设备的项目目录离线复制可使用已有恢复工具，`--source-root`指定旧项目根目录，目标路径仍由本项目推导：

```powershell
.venv/Scripts/python.exe scripts/restore_model_assets.py --source-root <旧项目根目录> --only papagei_s.pt --only papagei_p.pt --only checkpoint_best.pkl
.venv/Scripts/python.exe scripts/restore_model_assets.py --manifest data_manifests/ppg_bench_assets.json --source-root <旧项目根目录>
.venv/Scripts/python.exe scripts/restore_model_assets.py --verify-only --only papagei_s.pt --only papagei_p.pt --only checkpoint_best.pkl
.venv/Scripts/python.exe scripts/restore_model_assets.py --manifest data_manifests/ppg_bench_assets.json --verify-only
```

`<旧项目根目录>`是需替换的说明占位符。恢复工具先校验大小和SHA-256，再原子发布文件；没有离线源时，从两个清单中的作者链接获取对应文件后执行校验。PRV没有预训练权重；MOMENT包含权重和配置两份文件。

公开数据使用`scripts/download_ppg_bench.py`，预处理及评测使用`scripts/run_ppg_bench.py`，命令见正式进度页。DREAMT须先由用户获得2.1.0项目访问权限。数据必须采用benchmark原始布局，早期活动MAT/汇总MAT不能直接替代官方原始归档。

## 检查与续跑

`python scripts/ppg_bench_status.py`汇总下载、四采样率预处理和已完成外层评测。`scripts/run_ppg_bench.py`使用固定上游提交及输出锁；上游还核对模型、数据、版本、设备和协议身份，身份改变时使用新输出目录。

跨设备迁移时保留原始结果作为历史证据；上游身份含绝对路径和文件时间信息，直接复制结果到另一台设备不保证能原地续跑。不要手工修改运行身份来绕过检查，也不要将重复输出相加作为新评测次数。

完整PPG-BP线性结果可重新核查：

```powershell
.venv/Scripts/python.exe scripts/audit_ppg_bench_ppgbp.py
```

核查通过仅说明这一完整子集；全篇完成状态仍由其余数据、方法、折数与配置核验决定。
