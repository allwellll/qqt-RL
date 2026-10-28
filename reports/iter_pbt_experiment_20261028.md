# ITER-PBT-V1 持续优化（2026-10-28）

用户授权 `.hermes/iter_pbt_plan_0045.md` 覆盖此前补齐配对实验要求，目标为持续训练更强模型。旧 COMBAT 已启动候选运行至 u320，取消未启动 legacy seed52/53。课程/塑形 seed53 在授权时已正式运行，按“正在跑的7条继续”保留；原配对结果保留，不补配对。旧 R2/O5/储备暂停不变，不提交、不推送、不发布。

## 冻结执行协议

主协议 `runs/iter_pbt_v1/PROTOCOL.json`；异步/初始化歧义在执行前写入 `protocols/amendment01_async_selection.json`。8 slot，每卡独立 env2048、rollout128、minibatch8192、epochs1、DAgger/KL=0。每 chunk 80 次 PPO 更新，新增20,971,520真实环境步、2560 Adam步；40/80更新保存完整params、Adam、环境、aux、RNG、ring、fixed actor。持续接续，用户暂停优先。

初始候选来自已完成 COMBAT u320；新dev分数齐备后按资格和分数排序。分数未齐时允许确定性的临时候选初始化并扰动，不能把顺序称为能力排名。公共固定对手使用 seed51共同u160父点，PBT后续完整保存fixed actor；这项新种群定义明确记录，不冒称与所有旧候选固定对手完全一致。

每40更新请求生产JS Hunter困难/普通/简单各64局与原生JAX自然64局categorical，dev seed20261101；两路CPU各4核，与GPU线程核段分离。JS沿用已冻结64出生坐标，dev使用新游戏/动作随机流；这不代表新空间覆盖。奖励塑形与原生事件/引擎按现有COMBAT实现；JAX Hunter仍是近似训练对手，不冒称生产JS。

选择分=3×困难安全击杀局+2×普通安全击杀局+简单安全击杀局−2×所有难度死亡局，局级死亡去重。JAX同split自然交付≥共同父90%才可作donor。完整8slot cohort齐备后，后2位可被前2位中合格者替换，分差≥6。异步分数迟到不阻塞训练；在后续80更新边界消费完整cohort，同一loser/cohort只消费一次。

允许探索lr×{.8,1.25}、entropy×{.8,1.25}、shaping{0,.1,.2,.3}、课程{.25,.4,.6}及hard/current各±.1对手质量转移，非负且总和为1。属于整合优化配方，不是单变量因果实验。

新dev最佳请求旧seed20261001保留确认：JS各64局，JAX304局categorical及greedy；需保留分数超过先前确认最佳（首次为父点）且两模式交付≥父点90%，才写BEST.json。反复使用固定保留集会产生自适应选择偏差，确认结果仅作描述性检查，不称独立最终统计推断。

## 技术验收与运行记录

- 选择/扰动、完整cohort技术重试归属、缓存、owner暂停/通知、原对手配比与league序列化恢复相关测试26项通过（`tests/RELATED_GREEN.log`）；导入与语法检查通过。
- 首批smoke在任何PPO更新前因aux.kind int8/int32 ABI不一致被JAX scan拒绝。原尝试和日志保留；amendment02记录最小int32修复，失败chunk不计有效训练量，新的chunk编号从同完整父点重试。后续真实更新、fresh恢复与吞吐以原始marker为准，不以进程存在代替。
- 同期修复旧COMBAT controller终态通知重复unix字段异常，保留worker和终态checkpoint，不重训已完成run。
- 持久入口：`.venv/bin/python runs/iter_pbt_v1/WATCHDOG.py`；OWNER.lock/EVAL_OWNER.lock/WATCHDOG.lock保证唯一性，重启后采用存活worker；slot谱系、chunk、SHA、超参分别见LINEAGE.jsonl、protocols、STATUS.json。
- legacy整批复用候选使用用户50%吞吐标准被动重判，原85%判定保存在SMOKE_FROZEN85.json；未达50%跳过，不改训练源码和已执行更新。

当前过程尚不能证明PBT能力提升；完整新dev与保留结果未齐，不宣称新最佳。

## 首批实际执行证据

`runs/iter_pbt_v1/GPU_RESUME_ACCEPTANCE.json` 记录4条正式run、源checkpoint与双SHA、真实Actor/Critic非零量、fresh恢复和3次GPU连续实测；京ME发送返回0。slot02/04/05/07从完整u320继续到u326/u330（该收据时刻），暖态约20,082–20,605环境步/秒，真实更新每次262,144环境步、32 Adam步。后续进度由末尾被动监控更新。

legacy候选暖态14,093环境步/秒，基线比例0.69578，按用户50%标准接受。恢复后的league NumPy `.at`错误经RED复现→`j.asarray`最小修复→GREEN；u162双SHA与fresh恢复后已提交u163。额外审计发现固定父对手恢复不能跟随移动checkpoint，现绑定原注册u160并持久化frozen actors。旧失败、源文件、收据均保留；不重跑技术smoke。修复见COMBAT amendment06。原生状态与pool恢复的额外技术变更不冒称行为实验收益。

新dev共同父选择分21，初始control seed51/52/53分别23/−1/16；各自然JAX交付64/64。空间/对手/席位覆盖有限且交付饱和，不能将此视为生产JS自然完整局能力证明。尚无保留确认新最佳。

同slot编译缓存以相对链接复用内容寻址条目，不移动运行缓存；首次编译和fresh进程恢复会暂时降低GPU利用率，如实记阶段。技术尝试编号与有效80更新generation分开，避免部分slot零更新重试导致无法形成8slot cohort。追加真实工作量账本不混donor历史。看门狗采用已存活owner及worker，阶段感知错误退出并通知；被动监控每分钟写本报告，训练不等文档。


## 首轮终态与自动接续验收

`runs/iter_pbt_v1/AUTO_CONTINUATION_ACCEPTANCE.json` 已逐条核验首批4个u400终态state/policy双SHA、每条恰好80个有限非零更新、20,971,520新增环境步及2560 Adam步。至少3个后继chunk已真实非零更新至u402，证明持久owner实际接续而非仅launch。后继沿用原配方，因为完整8slot dev cohort尚未齐，不用缺失分数做exploit。有效generation与技术尝试编号分开；同slot缓存复用已发生。

首轮含首次采样编译、新进程恢复间隙、保存的端到端速度与暖态每更新速度在同一收据分列；端到端从首次collect开始计时，不含此前解释器启动。不能把约2万的暖态更新速率当作包含这些间隙的全墙钟效率。


## 08:55 评估滞后修复验收

用户授权 `.hermes/pbt_eval_fix_0855.md`；实现见 amendment10 与 `runs/iter_pbt_v1/EVAL_FIX_ACCEPTANCE.json`。调度改为每slot仅领取最新已保存40/80点，未启动旧点写 SKIPPED_SUPERSEDED，历史文件与在途评估保留；初始候选旧请求不再挤占队列。每个评估4个JS分片并发，全局28个跨进程锁令牌；原生JAX新任务限制2路。20个评估job核段为120–143、160–183，与GPU训练32–55、72–111分离，CUDA不可见，BLAS单线程。恢复时先采用全部存活评估再分配空核，避免重复job或核段冲突。

指定 `slot05_c000005_u000720` 的 holdout（seed20261001）最高优先，真实分片已产出并校验；现存完整SUMMARY为124分，与指令现场171不一致，以原始文件为准，不宣称该点当前最佳。其他未启动holdout只领取当前最高合格dev候选，在途旧确认完成。holdout未完成，不预称确认或提升。

PBT淘汰只消费完整8slot cohort，且有效generation对所有当前slot均在0–2个chunk内；不按可能继承倒退的PPO update判断新鲜度。无新鲜完整分数继续原配方训练。监控新增每slot最新保存点、已评分点与chunk滞后；当前历史滞后仍约12–14，不把调度修复冒称评估已追齐。

TDD RED保留于 `tests/EVAL_FIX_RED.log`，GREEN及相关32项通过。仅向三个controller发送重载信号；切换中slot04_c22与slot06_c18的阶段入口因旧controller hash竞态在更新前失败，双SHA验证u2002/u722后保留失败收据并从原子状态接续，未重跑smoke。现已将可热更controller/eval provenance独立为orchestration_sha256，训练source_sha256仍强制不变。GPU3 c18 smoke与fresh恢复PASS，formal有限且Actor/Critic非零，u1700以上继续推进。对同run暖态更新的前后周期比较及全部SHA见验收收据；多条周期仍约13秒，短样本不证明长期吞吐。训练源、奖励、网络、配方未改；旧source绑定备份在 `eval_fix_source_bindings_before`，仅controller/eval文件hash重绑定。


## 09:20 原生整局评估共卡迁移（执行中）

授权 `.hermes/pbt_eval_gpu_native_0920.md`，不改训练worker、奖励/网络/超参；amendment11注册独立原生GPU评估owner。每GPU最多一路，关闭预分配、BFC比例0.03（当前设备约4.3GiB），另按进程显存6GiB保护；CPU JS保持28路。dev64、holdout每mode304局整批，复用已完成CPU分片而不重跑。每slot仅chunk末点排新评估，中点继续保存但169个历史中点请求不再领取。

冻结前8暖周期每卡基线分别约20,576/20,487/20,156/20,117/20,573/20,408/20,320/20,175环境步/秒；共卡时连续3个新暖周期中位值下降>5%则自动减半并发并让评估在scan边界释放，训练无停机信号。编译/恢复冷周期单列，不把编译进程当有效评估。

首批默认GPU矩阵精度对照：同checkpoint slot05_c000005_u000720、seed20261101、固定最前8个episode。CPU/GPU均交付8/8、死亡0/8、自然终局8/8；放泡数5/8局不一致，首tick策略概率已不同。默认精度对照先BLOCKED，保留 `native_gpu/PARITY_GATE_DEFAULT_PRECISION_PRESERVED.json`；最高矩阵精度完整局复核正在执行，不先将该差异称为纯物理错误或能力退化。相关35项测试PASS。最终切换、GPU显存/吞吐、耗时及滞后验收另见后续收据。


### GPU迁移对照与上线

默认与highest矩阵精度均做了完整8局同checkpoint对照，交付/死亡/完整终局均相同（8/0/8），5局放泡数不相同。固定初态探针的obs/global状态哈希、mask、actor keys完全一致，Gumbel最大差1.19e−7；显式bf16原生网络在两后端logit最大差移动0.08322、能力0.07542。用共同Gumbel加两条轨迹的概率，8/8首次动作分岔均精确重建（`FIRST_ACTION_DIVERGENCE.json`）。按用户允许“结果一致或差异可解释”，PARITY_GATE为PASS_WITH_EXPLAINED_BACKEND_NUMERICS；不是逐轨迹严格等价，不把CPU/GPU混为精确配对。原严格BLOCKED结果保留。

共卡队列已实际启用，指定holdout与更高已评分候选各304局整批、最新dev64局整批。原CPU评估只终止自身独立session，保留所有原子完成分片；新JS merger复用这些分片，不触碰训练进程。共卡显存采样约0.7GiB，首次编译单列。GPU0三暖周期吞吐19,131/19,802/19,044，相对冻结20,576触发保护，并发上限8→4，评估在scan边界协作退出；训练继续。首批8局最高精度完整对照约325.7秒，含两次约百秒级冷编译，后续128tick scan约3秒；不能据此推称304局点耗时，正式点完整耗时待产物。

持续证据：`native_gpu/STATUS.json`、`REPORT.json`、`SAMPLES.jsonl`、`CONCURRENT_SAMPLES.json`；训练每卡基线与有效更新在 `BASELINE.json`/主STATUS，评估点完成耗时在各 `native/NATIVE_SUMMARY.json`。目标lag≤2目前尚未达成，不报告追平。相关37项回归PASS（`tests/NATIVE_GPU_GREEN.log`）。复现：`.venv/bin/python -m scripts.iter_pbt_native_parity`；持续入口仍为 `.venv/bin/python runs/iter_pbt_v1/WATCHDOG.py`，新增native owner被唯一锁保护。


### 首个正式GPU点评估完成

`slot05_c000023_u002160` 的JS192局与原生GPU自然64局已原子完成，SUMMARY SHA验收通过，原生交付64/64、死亡1/64，当前选择分29（不据此称提升）。slot05最新保存与评分同c23，lag=0；其余slot仍有历史滞后，整体≤2目标未达。原生GPU耗时528.418秒，其中首两scan冷编译/执行218.568+232.021秒，后9个暖scan合计约30.3秒，余下含初始化/轨迹压缩/被动指标汇总。冷启动整点8.8分钟不能宣称优于现场估计CPU约7分钟；需后续缓存复用点判断收益。

验收 `native_gpu/ACCEPTANCE.json` 保存各卡冻结基线/新三周期中位吞吐、原始marker/hash、当前活跃owner与评估。已观察的评估显存约0.7–1.1GiB，未达6GiB；自动8→4保护已实触发，训练八slot仍在，未发送停机信号。指定holdout仍在完成304×2，完整终态由持久owner异步消费，未预报结果。

<!-- ITER_PBT_AUTO -->

## 持久监控最新快照

| slot/GPU | run | 阶段 | update | 环境步/秒 | finite | 最新保存 | 已评分 | 滞后chunk |
|---|---|---|---:|---:|---|---|---|---:|
| 3 | slot03_c000042 | formal | 3611 | 20119 | True | slot03_c000041_u003600 | slot03_c000021_u002000 | 20 |
| 4 | slot04_c000046 | formal | 3926 | 20486 | True | slot04_c000045_u003920 | slot04_c000024_u002240 | 21 |
| 5 | slot05_c000046 | formal | 3924 | 20387 | True | slot05_c000045_u003920 | slot05_c000025_u002320 | 20 |
| 7 | slot07_c000045 | formal | 2857 | 20152 | True | slot07_c000044_u002800 | slot07_c000024_u001200 | 20 |
| 2 | slot02_c000045 | formal | 3847 | 19977 | True | slot02_c000044_u003840 | slot02_c000025_u002320 | 19 |
| 1 | slot01_c000042 | formal | 2558 | 20678 | True | slot01_c000041_u002480 | slot01_c000022_u000960 | 19 |
| 0 | slot00_c000043 | fresh_restore | 2722 | 20458 | True | slot00_c000042_u002720 | slot00_c000021_u001040 | 21 |
| 6 | slot06_c000041 | formal | 2613 | 20090 | True | slot06_c000040_u002560 | slot06_c000021_u001040 | 19 |

实际新增工作量见 `runs/iter_pbt_v1/WORK_TOTALS.json`；谱系计数不混入本slot新增步数。连续利用率/功率见 `GPU_SAMPLES.jsonl`，正式首更新与恢复SHA见 `GPU_RESUME_ACCEPTANCE.json`。

共卡原生评估：启用=True，并发上限=0，活跃=0；分段耗时/显存/训练吞吐见 `native_gpu/SAMPLES.jsonl` 与各点评估 `native/NATIVE_SUMMARY.json`。

最新保留确认：`runs/iter_pbt_v1/formal/slot05_c000006/u000760`，dev分=139，保留分=145
