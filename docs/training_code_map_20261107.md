# 训练代码与封存索引（20261107）

当前状态：**训练全部停止，恢复必须用户重新明确授权**。七臂worker和owner已退出，GPU0–7计算进程为空、显存0MiB，最终checkpoint逐项见 `reports/cx_training_stopped_20261107.md`。不得执行历史文档里的持续接续命令；不得为保存、诊断或新评测重开GPU。原来等待1200预算结束的授权已被立即停止覆盖。

## 有效入口与持久停止策略

|入口/策略|当前用途|限制|
|---|---|---|
|`model_entry.py`|通用私有配置执行器|执行payload之前检查可版本管理的`training_launch_policy.json`和私有统一停止状态；拒绝训练角色、GPU配置与CUDA环境；CPU任务清除继承GPU环境|
|`model_training.py`|历史训练通用入口|当前禁用，不能当恢复命令|
|`model_evaluation.py`、`model_evaluation.js`|CPU评估、导出、复验|使用私有参数文件；不触发训练|
|`model_diagnostics.py`|CPU测试、只读审计、文档整理|当前不得运行GPU诊断/优化器更新|
|`training_launch_policy.json`|可入版本库的统一默认关闭策略|即使没有本地`.hermes`或私有状态过期，也不能自动恢复训练|
|`.hermes/TRAINING_STOP_STATE.json`|本机最终身份、checkpoint、资源释放回执|私有，不入公开仓库；不是新授权来源|
|`scripts/model_launch.py`|通用入口/私有配置映射|只定义命令，不构成GPU授权|

真实目录不搬动；本机外显别名只作兼容。正常入口不使用`r.py`、`a.py`、`q`或暴露具体训练模块的`-m`命令。密钥、私有配置、绝对checkpoint路径及实际进程身份不进入公开代码、commit或Release。

## 物理版本、奖励、任务和原卡位

这些是**停止时的历史运行配置**，不是继续执行指令。单卡DUEL/FTN2/ATK实际2048env×128tick、8192minibatch；SP实际3072env×128tick、12288minibatch。同步双卡是单策略global2048/8192、每卡1024/4096、每Adam前`pmean`，不把独立run称为同步加速。

|臂|原GPU|最后实际物理|任务/奖励|最后完整update|
|---|---|---|---|---:|
|L2_DUEL_SYNC2|0/1|bun-native-v2（旧编排回退）|50%完整抢包+50%空手击杀；同步梯度|20048|
|L2_DUEL25|2|bun-native-v3-items|25%击杀专项；完整局奖励保持|20844|
|L2_DUEL75|3|bun-native-v3-items|75%击杀专项；完整局奖励保持|20903|
|L2_FTN2|4|bun-native-v3-items|跨tick延续信用替代同步CHAIN，c门内一次安全前沿事件|22292|
|L2_ATK|5|bun-native-v3-items|原主动进攻配方，运包/道具机制保持|25623|
|L2_DUEL|6|bun-native-v3-items|50%完整抢包+50%空手击杀|20905|
|L2_SP|7|bun-native-v3-items|完整抢包、自对弈联赛，原奖励保持|24155|

新物理DUEL_SYNC2曾准入并保存u020988；其后继因协议兼容失败退出，旧owner自动回退到旧namespace。新旧点全部保留；最终真实active为旧物理，不能用新物理初始准入覆盖这一事实。精确源码/协议SHA、最后实际配置与原始奖励回执索引见 `runs/code_review_20261107/FINAL_CONFIGURATION_INVENTORY.json`。

`scripts/cx_items_rules_v2.py`、`cx_items_native_v2.py`实现v3供给与全图合法掉落；`cx_items_build_v2.py`生成私有冻结副本，`cx_items_install_v2.py`核验后安装。冻结来源为 `runs/items_alignment_20261107/v2/ACCEPTED_RULE.json`，原协议不关闭SHA校验。806原砖率0.582011保持、开局补24、预期道具总量71.142891；专项零供给基线补15，专项后续砖掉落0。属性上限、堆栈数量守恒及不足时待落队列保留；专项不因吃道具额外奖励。

`scripts/cx_duel_reward_v1.py`只按真实唯一kill +16、death -16、自死额外-8结算；受困、空爆、放泡数、泛idle和专项运包均不付。`cx_duel_mixed_v3.py`隔离完整局mode0与专项mode1，完整局导出固定mode0。场景由`cx_duel_scenarios_v1.py`声明：open_chase、narrow_block、crossroads、corner_pillar、brick_break、multi_exit、staggered_chain、items_slide，变化地图、能力、距离、时序及道具，train/dev/holdout分离。场景文件路径与SHA取各实际protocol/REWARD_RECEIPT，不能以相似文件名替代。

`cx_ftn2_reward_v1.py`组合`cx_fountain_continuation_v2.py`和`cx_safe_frontier_v1.py`。死亡/受控作废待付，空爆、同批同tick、回家堆泡、敌死不付；放敌拿包回杀循环撤销待付并回收既付进攻信用。完整局携包/道具队列保持。安全掩码与短DAgger实验属于单独历史分支，未据开发中的总体误拦率或合成吞吐冒称已挂到这些最终臂。

## 评估、导出与线上交付

`cx_items_candidates_v1.py`冻结七臂模型、生产JS引擎和seed；`cx_items_candidate_worker_v1.js`使用真实生产物理、采样器和bot，独立回放核对。`cx_items_candidate_games_v1.js`只声明专项初态/独立分集；`cx_items_candidates_report_v1.py`核验RESULT/review、hash、样本量和分组，输出私有READY。`cx_items_candidate_cpu_v2/v3.py`只调CPU核数，不改变已冻比赛。缓存只在完整protocol/model/engine/seed匹配时复用。

已完成856局v3评测，argmax和τ0.3分开，网页每对手每方式8局，并非64局。推荐可选DUEL u019902 τ0.3（近身击杀，网页自死更多、全局静止仍高）和SP u023302 τ0.3（猎手与喷泉，非已证实近身或运包优势）。最后新保存点未经同等独立评测，不因更新号更大替换已验证候选。

线上由网页CX管理，现为bun-native-v4-facing。v3的856局证据与网页owner的v4小样本兼容快筛分开，不能跨版本冒称同一成绩。两可选已发布，默认仍G_frozen τ0.3；发布与浏览器核验 `.hermes/WEB_CANDIDATE_PUBLISHED.json`，公共详细证据由网页owner自己的报告归档。旧Review按旧物理只读解释。

## 已废弃或暂停的入口

- `cx_items_transition_v2.py`、`cx_items_resume_v3.py`：历史迁移/后继owner，已终止，不可重启。
- `cx_stop_after_active_finalize_v1.py`：仅适用于已撤销的等待预算终点模式，当前会拒绝启动；不得补跑fresh。
- `cx_duel_owner_v*`、`cx_duel_sync_owner_v*`、`cx_duel_sync_successor_v1.py`、`cx_ftn2_budget_owner_v1.py`、`cx_ftn2_successor_20261107.py`、旧league supervise/switch脚本：历史来源，均不构成恢复授权。
- `cx_items_*v1`训练迁移、早期DUEL v1/v2以及FLOW安全掩码开发版：保留追溯，不作为当前运行标准。
- 旧GPU67任务、stop_after_active报告、历史报告中的“持续训练/1200后继”均已被最新立即停止覆盖；不删除历史checkpoint、协议、录像或runs来“清理”。

本机任何再次训练，必须先获得新的明确用户授权，再由owner审阅停止策略、资源范围、规则版本、奖励、精确协议及完整恢复流程。没有提供可直接复制的自动恢复命令。

## 非破坏性复核与集成边界

CPU已通过7项真实启动闸门测试、69项FTN2结算/防hack测试和5项DUEL纯结算测试。`git diff --check`通过；源码、资产、协议和最后COMMITTED的SHA均复核。复核只更新当前审计产物，不重写封存protocol/测试回执。

训练树有大量既有实验与网页修改；本次不整树add、reset、搬目录或push。交付按明确文件清单及SHA，由网页集成owner非破坏性review后合main。不能覆盖main自己的AGENTS、现行v4网页规则/模型目录或旧Review，不纳入`.hermes`、`local_config`、runs/checkpoints、私有软链及凭证。完整主仓库全测由集成owner在其实际合并HEAD执行并记录Actions/Pages SHA。
