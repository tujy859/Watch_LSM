# Watch_LSM 开发者与 AI Agent 协作中枢 (Agent Hub & Context Anchor)

> **致接手本项目的 AI Agent**：
> 本文件是本项目的**唯一真实事实源（Single Source of Truth）**与**会话级交接中枢**。
> 任何新会话、新 Agent 接入本项目时，**必须首先阅读本文件**，以获得完整的项目上下文、已就绪资产分布、当前工作推进节点及代码规范。
> **任务交接红线**：每次完成阶段性工作后，必须主动更新本文件顶部的【一、当前动态工作交接看板】！

---

## 一、当前动态工作交接看板 (Active Handover & Task State)

*最后更新时间：2026-10-31（Asia/Shanghai，本会话收尾；23:03主队列3979项，FP32另计69项；交接已整理，既有评测保留运行，临时预览服务关闭）*

> **下一会话先读**：[2026-10-31会话交接](docs/session_handover_2026-10-31.md)。其中记录两条研究工作线、23:03更新快照、已有证据和问题、后台续接入口，以及PulseDB可以逐项验收的小里程碑。下方18:04及更早数字属于历史快照，不当作实时进度。

> **最新实施状态优先于历史记录**：本机已连接 `origin`，PR #1已合并，合并提交为 `30aef2c`；本地当前分支为 `main`，跟踪 `origin/main`。用户明确指定后续默认直接在 **main** 修改、提交和推送，不自动创建工作分支、worktree或PR。每台设备使用独立 **venv**，不要按Conda环境迁移设计。PulseDB的38份修复及207份补齐已全部完成；11:42复核245/245通过，完整AAMI清单243/243通过文件大小及HDF5首尾抽样验证。正式片段映射与基准评测仍待完成。最新本机结果用 `scripts/data_status.py` 查看（`.local/` 不提交Git）。项目级操作说明见 [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)。

> 下方任何历史数据“已就绪”项仅描述当时核查的设备，不保证其他设备完整。日期化的迁移审计报告及机器可读错误清单属于原设备记录，不随 Git 分发；共享恢复范围以 `data_manifests/` 为准。

### 1. 当前已完成工作节点 (What Has Been Done)
- [x] **10月31日本会话收尾与知识沉淀**：用户要求本session先收尾并整理可复用信息，未开展新实现或新实验。新增`docs/session_handover_2026-10-31.md`，汇总PPG-Bench/PulseDB边界、既有审计证据、精度/后端/PRV协议限制、DREAMT暂缓、远程优先普通文档、main/独立venv/采集记录术语等约定，并将PulseDB拆成索引映射、按需加载与划分检查、最小预测三个可验收阶段。23:03:13北京时间按已发布结果核实主队列3979/14625（线性3745、MLP234）；ECSMP成为第5份全部7对象线性完成的数据，Dalia142/350、WESAD278/525。MLP为PaPaGei-S的PPG-BP175项与RealWorldPPG25项、AnyPPG的PPG-BP34项；FP32微调69项及原生AMP单折1项另计。小型快照`docs/results/ppg_bench_handover_2026-10-31.json`，原18:04文件不覆盖。只核查结果存在性和进程，不新增指标审计；已有评测队列保持运行，机器可读进程表在`.local/ppg-bench-repro/session-close-processes.json`。核验命令后仅关闭本会话创建的55688临时预览服务（原PID34832及启动器3208），未停止评测或删除任何数据；重启后不得沿用旧PID。
- [x] **10月31日PulseDB未完成正式评测的原因复核**：本机源码检索确认现有PulseDB工具以清单、同步、修复和只读检查为主，核心Dataset仍为wrist PPG+ACC，`watch_lsm/data/pulsedb.py`和`scripts/evaluate_aami_bp.py`均不存在。PulseDB并非未曾读取：10月29日既有覆盖报告确认AAMI清单243项文件大小及HDF5抽样通过；但1340条正式索引的片段映射、按需加载、官方训练/校准/测试隔离与预测闭环未完成，训练/校准数据覆盖亦不能由AAMI测试文件清单推定。前期38份修复/207份补齐已结束，之后开发重心转到不含PulseDB的PPG-Bench，独立实现没有继续落地。已在简表补充，不把现状归结为完全没读数据、权限限制或必须租服务器。本轮仅核查代码和既有报告，未重复扫描数据、启动PulseDB训练或改动现有队列。
- [x] **10月31日PPG-BP与PulseDB概念澄清**：核对两份原始论文与固定上游预处理，明确PPG-BP为桂林219人/657段2.1秒指尖PPG及电子血压计标签，不含连续ABP；benchmark排除231号后218人/654段。PulseDB为MIMIC-III/VitalDB的同步PPG+ECG+ABP数据，公开全集5361人/5245454个10秒片段，非本机下载覆盖数。已在`docs/ppg_bench_progress.md`补充区别；PPG-BP的1750项线性结果不能视为PulseDB已评测，后者正式片段映射及完整评测闭环仍待完成。本轮不改动数据、训练队列或评测计数。
- [x] **10月31日改用普通文档交付进度**：用户确认会话目录文件名引用仍未在远端显示，并要求改用文档等简单形式；内联可视化的端到端问题仍未解决，不再声称已修复。新增`docs/ppg_bench_progress.md`，以普通表格展示4种方法、9数据集/26任务、完成数、适配微调和已审计范围，并从完整报告链接；交付GitHub文档链接及回复内核心表格，不依赖本地路径、localhost或内联组件。沿用2026-10-31 18:04:53固定快照，不将旧计数称为实时进度。本轮只整理文档，不改动评测队列、数据、权重或依赖。
- [x] **10月31日远端查看时的可视化读取引用修正**：用户报告`Error: Invalid visualization read request`并说明可能在远程查看。只读核查本机Codex26.1027.3384.0安装包中的读取实现：支持以纯文件名在按会话ID日期划分的`visualizations/YYYY/MM/DD/<threadId>/`中定位；项目绝对路径在缺少工作区映射时会被拒绝，Windows路径在POSIX路径规则下也可能在校验阶段被拒绝。已将现有修订HTML原样复制到本会话专用目录、文件名`ppg-bench-progress-fixed.html`，两份SHA-256一致；本次内联引用使用该文件名，不继续重发项目绝对路径。从安装包提取原读取方法及边界校验逻辑，在隔离测试宿主中验证文件名读取成功、36单元格完整，并复现上述两类绝对路径拒绝；证据`.local/visualizations/read-request-verification.json`明确记录IO宿主为测试替身、尚未验证远端端到端展示。远程查看是可能触发条件，不能断言已确认用户实际执行宿主或前端请求路由。此前localhost预览仅适用于执行电脑，不作为远程入口；未修改应用本体、权限、评测代码或实验结果。
- [x] **10月31日进度可视化显示修正**：用户截图显示聊天中的嵌入区整块空白，本地独立预览可显示，不能仅凭截图确认宿主加载失败的具体原因。已将矩阵、默认数量、模型明细和时间改为直接写入首屏内容，脚本仅增强点击与选择恢复；初始化兼容立即执行/文档加载，忽略无效保存状态。完成状态改为中文“完成”，避免320px下100%断行。修订片段位于`.local/visualizations/ppg-bench-progress-fixed.html`，并同步修订此前线程可视化文件；新引用使用项目内ASCII路径。736px/320px无横向溢出、36单元格选择/数量/单选状态、异常保存状态和移除交互脚本后的完整首屏均已验证，证据`.local/visualizations/verification.json`及预览PNG。另准备仅监听127.0.0.1的独立预览`http://127.0.0.1:55688/`，不是公开部署，重启后需重新启动。保持2026-10-31 18:04:53进度快照，不将本轮显示修复当作新评测进度，也未改动活动worker、数据、权重或环境。
- [x] **10月31日18:04任务与方法进度快照**：按声明的run.json及最终发布结果逐任务/模型/方法/种子/折去重统计，主队列3569/14625项：线性3412/4550、MLP157/2975，RF及默认cuDNN/AMP微调均0。PPG-BP1750/1750、BIDMC350/350、BUT-PPG700/700、RealWorldPPG175/175线性完成；Dalia132/350、ECSMP305/350，WESAD/WildPPG排队，DREAMT按用户意见暂缓。MLP157项均为PaPaGei-S的PPG-BP主要任务，单对象157/175；全部7对象范围157/1225。另有FP32微调48项及原生CUDA/AMP1项，不并入主队列。更新`docs/ppg_bench_reproduction.md`，明确四方法含义、9数据集17主要+9次要任务、平台差异及结果发布/指标审计区别；机器可读快照`docs/results/ppg_bench_progress_2026-10-31.json`。覆盖率24.4%不是计算用时进度；本轮只检查已发布结果及文档，不重新训练、改动活动worker或安装依赖。
- [x] **10月31日WSL重新核查**：当前用户的`wsl --version/--status/--list --verbose`均明确报告未安装；当前账号无Lxss发行版注册项，无匹配WSL/Ubuntu/Debian等Appx与包目录，`.wslconfig`和`C:/Program Files/WSL`不存在。结论限定为本机当前无可用WSL，不推断用户历史上从未配置。未执行安装、启用系统特性或重启；证据`.local/wsl-audit-20261031.json`。
- [x] **10月31日DREAMT直连下载器已准备**：按用户指定2.2.0获取官方MIT-LCP/physionet客户端0.2.1，提交`54ebd21308c368a9d1e8cfd8a673b111cacead0e`，安装至独立`.local/physionet-venv`。默认路由实测经过EFan TUN；新增下载器忽略系统代理，将源IP及IP_UNICAST_IF绑定到WLAN物理网卡，读取实际socket确认后获得HTTPS200；重试adapter也保留绑定，禁止跨源/代理回退。不改VPN或全局路由。新增`scripts/download_dreamt_direct.py`、交互启动器`scripts/start_dreamt_download.ps1`及10项认证/网卡/路径保护测试，全部36项项目测试通过，下载venv pip check通过。
- [ ] **10月31日DREAMT文件权限未通过，用户同意暂缓**：用户已注册账号并申请数据；首次官方Basic清单请求被拒绝，第二次普通网页登录确认成功，但DREAMT2.2.0的SHA256清单仍403。不能将账号登录成功当作项目文件权限已生效；用户确认网页可登录并表示该睡眠数据集可暂缓，已在本机窗口退出等待重试。未下载任何DREAMT数据、未保存密码或Cookie，已无下载Python进程；状态`.local/dreamt-download-status.json`为`deferred_by_user`。后续仅在用户需要时再恢复授权核验，不再反复要求注册/输入密码。2.2.0修正ACC/PSG对齐，计划保存到`datasets/ppg_bench_raw/dreamt-2.2.0/`，不覆盖原2.1.0输入或混用结果。DREAMT适合后续睡眠分期/夜间多模态研究；当前血压/年龄/剧烈运动主线可先推进其余数据，完整PPG-Bench仍须标记缺睡眠任务。其他评测进程仍在运行，本轮核查正式结果3563项。
- [x] **10月31日新增BIDMC指标复现与核查**：AnyPPG和PaPaGei-S各完成呼吸率/SpO2两任务×5种子×5折，共100项线性评测；53人训练/测试组无交集、五折五种子、预测身份与有限值、窗口MAE重算全部通过。四组均值及Student-t 95% CI均与论文表1保留两位小数一致：AnyPPG为2.56±0.18 / 2.85±0.17，PaPaGei-S为2.44±0.08 / 2.54±0.06。精确数值、原始结果/预测集合哈希和运行清单哈希已保存`docs/results/ppg_bench_bidmc_linear_partial/`。正式完成数已超过2050，FP32另计；其余BIDMC对象和原生AMP完整HPO仍运行，不把本阶段当作全篇完成。
- [x] **10月31日AMP数值路径定位**：用原PaPaGei-S收缩压失败候选、seed42/fold0的417训练/105验证窗口定位到`basicblock_list.17.conv2.conv`，本机Torch2.5.1+cu124/cuDNN9.1在FP16下首先出现非有限输出。同层相同输入和权重改用原生CUDA FP16或cuDNN FP32均有限，输出最大绝对值6.16797/6.16845、两者最大差0.002214。原生AMP与FP32单候选均完成15轮；不涉及外层测试集，不等于完整HPO已完成。新增`scripts/diagnose_ppg_bench_amp.py`，证据保存在`docs/results/ppg_bench_amp_diagnostic/`。未更改上游源码、权重或安装新依赖。
- [x] **10月31日完整搜索AMP单折试跑已完成**：原生CUDA卷积、保留AMP的PaPaGei-S收缩压seed42/fold0试跑在`.local/ppg-bench-amp-native-pilot/`完成1/1外层评测，保持每次48候选、3次验证重复、15轮上限；3份数据库均48项终态，无失败或运行中候选，COMPLETE/PRUNED分别13/35、15/33、15/33。单外层汇总MAE16.5163 mmHg，不能与论文全折均值直接比较。收尾状态phase=complete、exit_code=0，已恢复并发布Windows工作目录、刷新官方CLI汇总。独立`backend-profile.json`及`execution-profile.json`保留后端、精度、候选并行数差异，不并入默认cuDNN或FP32计数；默认cuDNN/FP16问题仍未解决，其他任务/对象稳定性待验证。
- [x] **10月31日线性并行与进度**：替换旧线性协调器（24208/20624已退出），保留活动Dalia启动器12264继续计算，不重复训练；新协调器PID35804以`--parallel-datasets 2 --adopt dalia=12264`运行，BIDMC启动器38328进入第二位置。协调器支持明确接管并核验全部7对象scope，进程退出本身不作为成功证据；状态已区分queued与缺数据。BIDMC AnyPPG完整50项已完成，PaPaGei-S继续；正式结果计数已超过2030，FP32适配结果单独计数。新增6项队列/后端保护测试与原20项测试均通过，PRV跳过策略实际CLI检查通过。
- [ ] **10月31日全套计划与PRV协议澄清**：官方PRV编码器实际0参数，微调构建器拒绝无参数编码器；论文表10/11的PRV均值与置信区间一致，支持复用MLP对照的推断，但未确认逐折复用规则。此前15050是名义报告格数，不能全当独立训练；扣除PRV的17×25=425个微调报告格后，独立计算计划改为14625，425项复用/协议核验单列，不自动计完成。启动器标记`requires_prv_protocol_clarification`，不会启动不存在的PRV编码器训练。本轮增量核查用户调研仓main `bb994f2c341ff1405c32092f6e60a2617842b050`的PPG-Bench章节，数据名单已校正；benchmark仍固定`2bca102...`。DREAMT权限、Linux/cuML RF及其余全部折数仍待完成。
- [x] **10月31日完全访问与临时目录复核**：当前收到的执行配置已为danger-full-access/never，与用户所述完全访问一致；上一轮工具配置为workspace-write，界面与当时执行配置为何不一致尚无证据，不归因于用户修改设置。此前测试报错的3个系统临时目录（tmpv5qa7b8k、tmpxa0n_ylu、tmpmlc65fqt）复核均已不存在，清理时间/机制未确认。本轮在普通用户Temp新建独立测试目录，读写及自动清理均通过；不需要迁移临时目录。项目内5份中断评测归档全部保留，其中3份线性工作目录为空、MLP归档3份文件320892字节、FP32归档5份文件666284字节，合计987176字节；不与系统测试临时目录混淆，未修改原始数据、模型权重或已完成评测结果。
- [x] **10月31日进度复核与中断恢复**：实际系统只读检查确认原6个Python队列均已退出，日志最后更新于10月30日20:40左右；退出原因未确认，不能声称一直在运行或归因于权限变化。按已发布的结果文件核实为1859项正式评测（PPG-BP线性1750、RealWorldPPG线性68、Dalia线性29、PaPaGei-S MLP12），另有FP32适配微调2项，不纳入原始精度统计。发现上游status.json比最终结果目录滞后，已修改`scripts/ppg_bench_status.py`按run.json中的已发布结果计数，排除.working及未知任务，保留原摘要计数供比较；新增3项测试，全部20项测试通过。
- [x] **10月31日本机队列已恢复**：将5份未完成任务目录完整归档到`.local/ppg-bench-interrupted/2026-10-31/`，已完成结果未改动；记录`.local/ppg-bench-repro/interrupted-20261031.json`。已获准在受限环境外恢复后台进程：主线性PID4728、补充线性31804、自动线性24208、MLP28676、FP32微调6040、MLP后续队列27172（等待新MLP PID28676）。进程映射及日志见`.local/ppg-bench-repro/resume-20261031.json`和`20261031-101225-*.log`；恢复后神经搜索数据库持续更新，6份入口错误日志为空。旧PID均失效；新PID也仅对本次开机有效。DREAMT权限、Linux/cuML RF、原AMP NaN仍未解决，不能声称完整复现完成。
- [x] **10月31日操作权限澄清与统计修复保存**：上一轮为受限操作权限，系统进程检查和系统临时目录测试曾被拦截；获准在受限环境外重试后成功。这与benchmark依赖损坏是不同问题；后续优先将临时文件放项目`.local/`，不要把操作权限限制笼统称为评测环境故障。当前已恢复完全访问，状态统计修复、新增3项测试及交接更新一并按main流程保存；已检查远端无更新。本轮没有重装依赖、重新下载数据或租用服务器。
- [x] **10月30日源码与后续MLP队列已保存**：上述资源接入、核查脚本及PPG-BP汇总已以`314a782`推送main。新增`scripts/continue_ppg_bench_mlp.py`，本机PID22356等待现有MLP启动器PID31908退出后，在同一`.local/ppg-bench-neural-mlp`目录续跑8份已就绪数据、全部7对象的原始MLP协议；已完成任务按上游运行身份复用，不额外并行占用GPU。状态`continuation.json`、日志`continuation.log`及`.local/ppg-bench-repro/mlp-continuation*.log`；关机后PID失效，重启时须先查活动进程再恢复，不照抄旧PID。该队列不含DREAMT/RF，也不代表原始AMP微调已解决。
- [x] **10月30日20:30资源补齐与结果核查**：BUT-PPG的830对有效质量采集记录文件已补齐并按50名受试者完成预处理；ECSMP最终60人、WildPPG16人也均完成预处理。除DREAMT外的8份数据、50/64/100/125 Hz共32份输出全部就绪。新增`scripts/audit_ppg_bench_ppgbp.py`核查1750项完整线性结果的5折5种子、218人覆盖、受试者无交集、测试折互斥、预测身份/有限值及血压MAE重算，全部通过；70项汇总及产物集合哈希已保存到`docs/results/ppg_bench_ppgbp_linear/`。补充`docs/ppg_bench_setup.md`固定源码/独立环境/离线权重恢复说明，恢复工具现支持新增模型清单；4份新增权重/配置SHA-256复核通过。17项项目测试、benchmark环境pip check通过，上游源码未改动。实况计数已超过1840项，仍为部分评测，原始AMP/FP32分开记录。
- [x] **10月30日正式复现资源与阶段结果**：5个专用PPG模型+PRV/MOMENT共7对象均通过真实PPG的GPU前向和小批量反向/优化检查。新增模型来源及哈希见`data_manifests/ppg_bench_assets.json`。PPG-BP的7对象、10任务、5种子×5折共1750个线性外层评测完成，收缩压/舒张压的全部均值与论文表1在两位小数上一致；其余线性任务及PaPaGei-S完整折数MLP继续后台运行。Dalia、BIDMC、WESAD、WildPPG、BUT-PPG、ECSMP官方原始数据均已补齐并完成4采样率预处理。最终使用原独立`.local/ppg-bench-venv` Python3.12，已补兼容的固定MOMENT源码0.1.5/Transformers4.54.1；不再使用此前准备但未完成的Python3.11环境，项目`.venv`未改装。17项测试通过。运行及完整限制见`docs/ppg_bench_reproduction.md`，实况用`python scripts/ppg_bench_status.py`；不能将此阶段等同于全篇复现完成。
- [ ] **10月30日神经评测与外部条件**：原AMP微调首个候选在本机产生NaN，保留失败结果；FP32单候选诊断的3次验证重复通过，正式48次搜索的串行FP32队列已另行启动，精度与TPE并行调度差异单独记录，不计作原AMP设置已复现。Windows的Optuna数据库句柄导致完成目录改名失败已通过外层恢复完成文件/原CLI续跑处理，相关完整性保护测试通过。DREAMT2.1.0仍待用户登录注册并签署该项目协议；当前非管理员Windows未安装WSL，原cuML RF后端仍待Linux环境。用户偏好先用本机，确需时再评估AutoDL等租赁，尚未产生云服务器费用。
- [x] **10月30日正式复现授权与范围**：用户明确要求补齐全部必要模型和数据、复现原工作，已覆盖此前“不自动运行全套”的限制；采用开源预训练权重，完整协议包含下游MLP和全量微调，不从头预训练基础模型。Python3.11和旧MOMENT包的准备方案已被上方最终Python3.12兼容环境取代；全部模型与8份公开数据已完成补齐，当前以顶部最新状态为准。DREAMT2.1.0当前要求为注册用户签署该项目数据协议，页面未列CITI或credential要求；用户尚未提供权限。优先本机，未开通付费服务器。
- [x] **10月30日PPG-Bench首次GPU试跑通过**：按用户要求使用现有开源PaPaGei-S权重，不预训练/微调编码器，仅拟合Ridge预测头。独立`.local/ppg-bench-venv`安装固定版本官方源码（`repos/ppg-bench`，提交`2bca102b7be2c0a9816a3872314784d2ef92555b`）；项目原`.venv`未改装，两个环境pip check均通过。官方预处理完成218人/654片段；seed42/fold0训练174人522片段、测试44人132片段，受试者无交集。GPU严格加载及冻结推理成功，输入32×1×263、输出32×512且有限，权重文件运行前后哈希一致；GPU提取表征后CPU完成官方4折内部选参，alpha100。单折收缩压MAE15.2132 mmHg，常数基线16.5888 mmHg；仅为流程运行检查，不是完整论文复现或临床结论。原始结果`.local/ppg-bench-smoke/`，报告`docs/ppg_bench_smoke_2026-10-30.md`，可复用配置`watch_lsm/configs/ppg_bench_papagei_s.yaml`。调参有4条矩阵条件数警告，最终alpha100模型float64交叉诊断最大窗口预测差约0.000119 mmHg，未替换官方结果。未运行PRV/MOMENT、全折全种子或其他任务。
- [x] **10月30日PPG-Bench资源实查**：用户确认调研仓名单错误将自行修改。当前本机PPG-BP的219人标签及657份波形已读通，231号2份波形为4200点，官方代码排除该受试者后为218人/654份；RealWorldPPG两份XLSX已读通35人/2074条300点有限记录。PaPaGei-S/P、Pulse-PPG三份权重重新计算SHA-256均匹配本地清单。PPG-DaLiA为8份活动MAT（各14或15条参与者数据），WildPPG为单份16条cell结构MAT，均非官方直接加载布局，语义等价性未核验；BIDMC/WESAD ZIP再次确认无效，DREAMT/ECSMP/BUT-PPG未找到。项目及默认缓存未发现AnyPPG/SigmaPPG/MOMENT-1-small权重；现有venv未安装ppg-bench/Optuna等依赖。硬件GTX1660SUPER 6GiB、内存约31.8GiB；完整RF需另外解决RAPIDS的Linux/WSL2环境。结论为资源足以推进部分任务接入，但完整论文复现尚未就绪。明细`docs/ppg_bench_integration_notes.md`、本机报告`.local/ppg-bench-resource-audit.json`；本轮未安装、改数据、运行benchmark或验证官方指标。
- [x] **10月30日PPG-Bench核查**：从用户调研仓进一步阅读arXiv:2610.13435v1及官方 `mannas-ai/ppg-bench`（master提交 `2bca102b7be2c0a9816a3872314784d2ef92555b`）README和部分实现，新增 `docs/ppg_bench_integration_notes.md` 并从调研入口链接。确认17个主要+9个次要任务，68组结论仅对应17乘4；调研条目的9份数据集名单与原文/代码不一致，已在本地记录但未修改外部仓。官方含PaPaGei/PulsePPG适配器、受试者分组评测与简单对照；闭集身份识别按采集记录隔离是例外。当前标准接口为单通道PPG，Watch-LSM需明确单模态/缺失ACC方案或独立多模态扩展；套件没有PulseDB任务，不能替代官方AAMI工作。建议先做PPG-BP单任务线性运行检查再扩展。本轮未安装、下载数据、执行benchmark或验证适配成功。
- [x] **10月30日调研知识入口与术语约定**：用户指定生理信号 recording 统一称“采集记录”，已加入下方行为规范并修正上一条交接措辞。核查用户维护的 `tujy859/Wearable-Physiological-Foundation-Models`，main参考提交为 `071cd9216d0eb2a262b959667ab33467b9fd1adf`；新增 `papers/README.md` 作为调研入口，连接已有综述、8份解析与外部模型/benchmark/数据集索引，并按任务映射本地实现和待验证项。根README与开工阅读规范已接入；后续选型、数据与评测设计需检查相关源更新并记录具体采用版本。本轮没有复制整仓、安装工具、自动同步或下载模型/数据，未逐项复核外部论文；PaPaGei/Pulse-PPG规模口径差异保留待核验。
- [x] **10月30日架构与辅助工具评估**：本地main已执行 `git pull --ff-only origin main`，无远端更新；检查Git跟踪的48份Python文件，其中核心包12份。当前为PPG+ACC研究原型，包含数据预处理、多采样率tokenizer、4层跨模态backbone、自监督损失及心率/步频/活动任务头；PulseDB官方片段加载器与血压评测闭环仍未实现。新增确认：两个训练入口按采集记录前3/后2划分，受试者s1同时出现在训练和验证；Dataset预处理后全部驻内存且未保留受试者/片段身份；下游YAML尚未接入训练入口，预训练部分配置项尚未消费。建议优先补数据协议、防泄漏验证、统一模型适配和实验记录，再渐进接入Ruff及轻量依赖图；重型CodeGraph暂非优先事项。本轮只做静态核查及工具官方文档核对，未安装工具、改训练代码或重跑训练/测试。
- [x] **PR #1已合并并切换main开发**：合并前最新PR提交 `a8bd7da` 的Windows/Linux、Python3.10/3.12四组CPU CI全部通过；GitHub以merge方式合入main，保留原提交历史，合并提交 `30aef2c`。本地已切换并跟踪 `origin/main`，配置 `pull.ff=only`；README及开发指南已改为同一开发者多设备直接main的流程。未删除历史工作分支。后续设备开始前先拉取，换设备前提交推送，避免覆盖另一台设备的工作。
- [x] **10月29日项目整体状态复核**：本轮迁移的代码入口、独立venv与245份数据修复补齐已完成；现有Watch-LSM为PPG+ACC自监督及心率/步频/活动下游原型，尚未具备完整PulseDB官方划分加载与血压评测闭环。核心14项测试及四组CPU CI的通过证据对应提交 `048e52f`；本轮未重复运行未改动的测试。GitHub实时确认PR #1仍为open draft、未合并，main仍为 `78650d4`；本地未提交修改仅交接文档AGENTS.md，最新数据完成状态尚未推送远端。研究优先级为核验1,340条AAMI片段映射、正式预训练基线接入、受试者隔离评测及指标重测。
- [x] **10月29日数据修复补齐完成并复核**：单worker续传于06:00:44正常结束，AAMI作业207/207通过、失败0；结合首轮修复38份，总计245/245、48,372,845,860字节（45.05 GiB）已验证。11:42后台锁已释放，本地955份波形均通过HDF5打开检查；完整AAMI清单242份患者文件及官方测试索引共243项，文件大小及HDF5首尾波形/标签抽样均通过。报告 `.local/data-status.json`、`.local/aami-file-coverage.json`。现有原始文件未全部对照云端哈希，正式索引片段偏移语义尚未核验。
- [x] **10月29日01:16 后台状态与续传复核**：首轮于00:24结束，最终181/245份通过（修复38/38、补齐143/207），剩余64份均为 DriveFS `Errno 22` 读取失败，未新增通过文件。本地891份波形重新执行 HDF5 打开检查全部通过。已保存首轮日志到 `.local/data-transfer-first-pass*.log`，并以单worker重新启动AAMI后台任务（PID604），锁已持有，正在本地复核并跳过已验证文件后重试失败项；降低并发能否改善云盘读取仍待观察。
- [x] **21:48 本机数据进度复核**：38/38 份损坏文件已修复并验证；AAMI 缺失文件已补齐143/207份。总计181/245份通过校验，已验证约20.66 GiB/计划45.05 GiB；剩余64份中2份传输中、40份待处理、22份本轮重试耗尽（DriveFS 读取报 `Errno 22`）。后台锁仍持有且日志持续增长。本地891份波形 MAT 全部通过 HDF5 打开检查；该扫描不等于全部原有文件已对照云端哈希。
- [x] **Windows/Python 3.12 独立 CPU/GPU venv 实测通过**：CPU 构建及 CUDA 12.4 构建分别安装验证；当前 `.venv` 为 PyTorch `2.5.1+cu124`。`pip check`、14 项测试、4 份模型精确哈希、checkpoint 严格加载与 CPU/GPU 合成前后向均通过。真实 wrist 634/435 窗口完成 CPU、GPU 各一轮预训练及 CPU 一轮下游训练。结果仅用于 smoke check。公共锁已排除本机 wheel 地址和 Torch；更新用 `scripts/lock_environment.py`。构建清单明确 pin 到 `+cpu`/`+cu124`，避免 pip 将另一构建误判为已满足。
- [x] **远端修复与多平台检查已就绪**：修复经 [PR #1](https://github.com/tujy859/Watch_LSM/pull/1) 合并至main；合并前最新提交的Windows/Linux、Python3.10/3.12四组CPU CI均通过。其他设备直接使用main。
- [x] **新设备完整 AAMI 文件清单已建立**：`pulsedb_recovery.json` 明确为本次迁移差集，不能代表空白设备的完整数据集；新设备使用 `pulsedb_aami.json` 的242份患者文件 + 官方测试索引（共243项）。测试防止把缺失差集误作完整定义；正式片段索引语义仍待加载器核验。
- [x] **数据修复后台作业已完成**：使用venv的Windows后台任务按明确清单处理，日志 `.local/data-transfer.log`、`.local/data-transfer.err.log`，状态 `scripts/data_status.py`。245份修复补齐全部通过，后台正常结束。
- [x] **恢复源码版本控制**：已在保留全部工作文件的前提下抓取原远端历史并重建本地索引，分支 `codex/multi-device-repair` 基于原 `main`。尚未合并任何远端分支。
- [x] **venv 与 CPU/GPU 入口已实现**：新增 `pyproject.toml`、CPU/CUDA 12.4 构建清单、安装/锁定/诊断脚本、统一路径与设备选择，以及 Windows/Linux CPU CI；安装结束必须通过实际反向步骤才能标记环境就绪。
- [x] **数据修复工具与故障保护已实现**：`scripts/sync_pulsedb.py` 支持明确profile、断点前缀校验、流式SHA-256、本地复读、HDF5首尾抽查、原子替换与进程锁；恢复/失败保护测试已通过。245份实际修复补齐作业已完成并复核。
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
- [x] **完成PPG-Bench最小评测试跑**：PaPaGei-S + PPG-BP收缩压单任务/单折/单种子，GPU提取特征、CPU线性拟合已通过；见`docs/ppg_bench_smoke_2026-10-30.md`。只验证冻结开源编码器路线，不代表自研Watch-LSM或全套benchmark已复现。
- [ ] **执行已授权的PPG-Bench正式复现**：先推进已就绪8份数据、5份专用PPG模型与PRV/MOMENT对照，保留原5折5种子、内部选参和4类下游协议。优先本机；DREAMT已注册并验证网页登录，但2.2.0文件仍403，用户同意暂缓；RF需解决Linux/cuML环境，后续更大算力再评估租赁。继续检查`.local/ppg-bench-repro/`；原始数据、权重和实验产物不入Git，PulseDB官方评测独立保留，未补DREAMT前不能宣称全篇复现。
- [ ] **后台队列接棒与未完成项**：先运行`python scripts/ppg_bench_status.py`，查看预处理、`automatic_continuation`、`mlp_continuation`与`adapted_backend_pilot`；不要重复启动仍持锁的队列。8份公开数据下载与预处理已完成，无需重复下载；截至23:03，`.local/ppg-bench-auto`的BIDMC/BUT-PPG/ECSMP完成，Dalia/WESAD部分完成，WildPPG排队。正式线性输出在`.local/ppg-bench-reproduction-py312`与`...-extra`，MLP及7对象后续队列在`.local/ppg-bench-neural-mlp`，FP32适配微调在`.local/ppg-bench-neural-fp32`。PaPaGei-S的PPG-BP/RealWorldPPG MLP均完成，全部公开数据MLP续跑已在运行。原生AMP完整搜索单折已完成，后续先验证更多任务/对象再扩展，保持后端结果隔离；新增线性结果待扩大独立指标审计。仍需其余所有折/种子/任务、PRV报告复用规则、DREAMT与Linux/cuML RF，不能声称全篇完成。
- [ ] **按调研入口推进具体选型**：后续开工先读 `papers/README.md`；涉及基模、数据集、benchmark或评测协议时，查阅用户调研仓相关最新内容，再核验原论文和官方实现，记录采用的提交与本地证据。现有历史分析无需整体复制或覆盖；持续自动同步尚未实施。
- [ ] **统一研究数据与实验契约**：先明确样本的受试者ID、记录ID、片段位置、采样率、标签来源及官方split；修复现有wrist入口s1跨训练/验证的问题，为PulseDB加入按需读取及划分交集检查。统一基模输入/embedding接口，核对配置项实际消费情况，保存每次运行的有效配置、Git版本、数据划分与权重哈希、种子及指标。辅助工具接入仅为建议，尚未实施。
- [x] **完成 AAMI 补齐及失败重试**：10月29日06:00全部完成，11:42复核修复38/38、补齐207/207通过，后台正常结束；955份波形HDF5打开检查及完整AAMI清单243项大小/首尾抽样均通过。无需再次启动下载；本机覆盖报告保存在 `.local/`。
- [ ] **扩展其他硬件与上游验证**：本机 CPU/GPU 与 Windows/Linux CPU CI 已实测；其他具体 GPU、macOS MPS 仍需按实际设备验证。上游 repos/ 历史快照还需独立版本核验，不把核心库通过视为全部基模复现完成。
- [ ] **修复历史实验解释与划分**：路径入口已处理；实验4的正式预训练权重接入、实验4/6受试者隔离划分与指标重测仍待独立研究任务，不把本轮环境恢复当成这些实验已经复现。
- [ ] **核验官方索引片段映射与有效覆盖率**：AAMI 索引1,340条记录含MIMIC674条、VitalDB666条；按 `Subj_Name` 下划线前缀映射的242份患者文件及测试索引已全部在本机通过大小与抽样验证。仍需正式核验名称和片段偏移映射，逐条确认1,340条记录可读取；文件齐全不能代替完整官方AAMI测试已复现。
- [ ] **构建标准相对路径数据加载模块**：在 `watch_lsm/data/pulsedb.py` 中编写基于 `PROJECT_ROOT` 的 PulseDB Dataset 与 DataLoader；先完成官方索引到实际片段的逐条核验，再按需读取本机已验证覆盖的患者文件，不硬编码早期748人的规模。验收步骤见本次会话交接。
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
  - 用户于2026-10-29明确指定：本项目仅一位开发者在多台设备上使用，后续默认直接在 `main` 修改、提交和推送，不自动创建工作分支、worktree 或 PR；另有用户明确要求时再采用其他流程。
  - 每台设备开始前检查工作区并执行 `git pull --ff-only origin main`，切换设备前提交并推送；保留未提交修改，遇到分叉先检查并解决冲突，不强推覆盖其他设备的提交。
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
| **PulseDB v2.0** | `datasets/pulsedb/` | 2026-10-29核查：955份患者波形文件通过HDF5打开检查；完整AAMI文件清单243项通过大小及首尾抽样。此前748份/137.12 GB属于补齐前历史规模；正式片段映射及训练/校准覆盖仍待核验 | 血压估计、官方AAMI数据划分评测与跨人群无校准泛化研究 |
| **MIMIC-PERForm** | `datasets/mimic_perform/` | 400 位重症监护患者，每人 **10 分钟连续未切片** 125 Hz PPG+ECG+Resp（共 268 MB） | 长程连续时序自监督、呼吸率估计、PPG 跨模态生成 ECG |
| **VitalDB 1-min** | `datasets/vitaldb_1min/` | 50 例手术患者，**9,994 个 1 分钟窗口**（166.6 小时 PPG+ECG，没有 ABP；连续块经筛选，不能假定整例无间断） | 长窗口表征与 PPG/ECG 配对研究 |
| **WildPPG** | `datasets/wildppg/` | 216 小时自由生活非受限多部位 PPG + 3D IMU（602 MB） | 穿戴真实运动抗噪、多身体部位（腕/胸/头/踝）域适应 |
| **PPG-DaLiA** | `datasets/ppg_dalia/` | 8 类本地活动文件（坐/走/桌上足球/骑行/楼梯/办公/驾车/进餐）手腕 PPG+ECG+ACC（367 MB） | 运动状态分类、运动伪影消除 |
| **PPG-BP** | `datasets/ppg_bp/` | 219 例患者，含收缩压/舒张压、高血压、糖尿病、脑梗死等 10 项临床疾病标签 | 临床慢病分类与下游少样本微调 |

---

## 五、Agent 交互与协作行为规范 (Rules for Agents)

1. **接盘第一步**：每次接到任务，通读本文件，明确当前处于哪一个工作节点；开始或恢复开发时同时阅读 [调研与工程选型入口](papers/README.md)。用户维护的 [Wearable-Physiological-Foundation-Models](https://github.com/tujy859/Wearable-Physiological-Foundation-Models) 是长期领域参考源；涉及模型选型、数据集、benchmark与评测协议时按入口查阅相关最新资料，采用前核对原论文/官方实现并记录参考提交。访问失败时明确所用旧版本及限制，不把调研收录等同于本地实现完成；
2. **代码编写规范**：
   - 保持文档与注释完整；
   - **术语约定**：生理信号的 recording / record 统一称“采集记录”，record list 称“采集记录列表”，不称“录音”；受试者、采集记录与切片窗口是不同层级，描述划分时须明确层级；
   - 永远使用相对路径；
   - 输出数值与时间范围时，严禁使用单半角波浪号（使用 `-` 或 `～`），转义文本中的美元符号（`\$`）；
3. **交接最后一步**：在向用户交付成果或会话结束前，**必须自动更新本文件【一、当前动态工作交接看板】中的状态与下一步待办**，为下一位 Agent 铺平道路。
