# 2026-11-06只读诊断

对应[原因分析与下一步建议](../../ppg_advantage_limitations_and_next_steps.md)。这些是对已完成实验的事后复核，不是新训练结果。

- `sds_readonly.json`：标签汇总、按既有训练人计算的常数对照、保存的PCA保留方差、预测分布、验证曲线及中性段时间覆盖。没有个人行或个人标识。
- `denoising_readonly.json`：已保存预测的噪声分层/幅度/相关性、参考频带、重采样边界与直流分量的线性核验。部分统计按公开S1–S7编号汇总；不含原始波形。
- `evidence_manifest.json`：共享证据和只读诊断脚本的SHA-256及额外独立复核。原正式目录的95份哈希证据保留，不能把旧执行审计通过当作参考构造没有缺陷。

源数据、权重与逐窗预测仍在本机`.local/ppg-formal-expansion/`，不随Git分发。两份诊断的输入路径及源哈希用于确认复算对象；它们不是自动下载清单。

使用已有benchmark环境，在仓库根目录执行：

```powershell
.local/ppg-bench-venv/Scripts/python.exe scripts/diagnose_ppg_sds_readonly.py --output .local/sds-diagnostic-new.json
.local/ppg-bench-venv/Scripts/python.exe scripts/diagnose_ppg_denoising_readonly.py --output .local/denoising-diagnostic-new.json
```

Linux用该设备独立环境的解释器替换上述Windows路径。省略`--output`则输出JSON；指定的输出必须不存在，SDS还要求位于`.local`内。脚本只读取已保存数据/权重数组并做数值运算，无优化器、预训练、编码器前向或GPU依赖。

SDS重建705份已保存最佳预测，最大差4.023×10⁻⁶分，记录MAE最大差3.981×10⁻⁷；源数据、705份预测及60份时间元数据哈希匹配。去噪诊断由独立NumPy/SciPy实现复算，并再次按168窗核验DC边界响应恒等式，最大差1.44×10⁻¹¹。重新运行的统计与首次诊断一致；诊断执行入口元数据更新时不要求整份JSON与旧临时入口字节相同。

没有重拟合模型或替换正式输入、预测、指标、图。已知去噪问题的修复与重评仍待下一步实施。
