# JE09 H8 Route PBRS 终态报告（v35/v36）

实验 `JE09-STAGE-H8-ROUTE-PBRS-20261017-v35` 从不可变 H8 v34 g32 raw Actor/Critic/对手池独立启动，唯一训练变化为生产网格路线势函数 PBRS，固定 `lambda=0.05`。课程、对手、PFSP、DAgger、网络、PPO/GAE、gamma=.995、评估协议均保持。正式固定预算完整跑完32代、67,108,864 fresh ticks；32×128 updates finite/nonzero，g32 Actor/Critic update L2=2.61813/2.91340，final fresh CPU restore成功。训练累计约1,908.8秒（31.8分钟）；GPU已释放。训练源基线提交 `7572dc90de6d03bf5e90282e9ecd6e8bd5c23657`，训练源文件哈希见run内 `SOURCE_SEAL.json`。

Smoke为真实8卡g1、2,097,152 ticks、128 updates，Actor/Critic finite/nonzero、8副本exact，DAgger noneligible label=0；fresh CPU restore及fresh GPU exact replay通过。正式首个STATE亦为g1/2,097,152 ticks/128 updates，Actor/Critic update L2=1.78380/2.37421。聚焦PBRS测试7/7通过。

g0/4/8/12/16/24/32 assist-off评估队列全部完成、failed=0。每个checkpoint包含固定32主评16局、独立holdout16局、continuous四任务各16局、web task四类各16局、Stage A/B主评与holdout各16局、Tactical v2 64局、Hunter hard/normal/easy各64局，以及拓扑四任务各16局；共544局/checkpoint。下表评估墙钟为该checkpoint `COMPLETE_WITH_BUN_TASKS`、continuous、A/B、Tactical、Hunter汇总器实际记载秒数之和；fixed32和topology分片计入局数，分片自身未提供独立wall_seconds，不能把汇总时间误称完整CPU占用时间。

| Checkpoint | Eval wall (s) | Fixed32 main pickup/capture | Holdout pickup/capture | Drop-return pickup/25%/50%/home/capture | Full2400 pickup/capture | Carry-home capture | Near steal |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0 | 571.1 | 16/16, 0/16 | 16/16, 0/16 | 16/16/0/0/0 | 0/16, 0/16 | 16/16 | 15/16 |
| g4 | 582.9 | 16/16, 0/16 | 16/16, 0/16 | 16/16/0/0/0 | 0/16, 0/16 | 16/16 | 15/16 |
| g8 | 591.0 | 16/16, 0/16 | 16/16, 0/16 | 16/16/6/2/2 | 0/16, 0/16 | 16/16 | 15/16 |
| g12 | 597.8 | 16/16, 0/16 | 16/16, 0/16 | 16/16/0/0/0 | 0/16, 0/16 | 16/16 | 16/16 |
| g16 | 597.6 | 16/16, 0/16 | 16/16, 0/16 | 16/16/2/1/1 | 0/16, 0/16 | 16/16 | 16/16 |
| g24 | 571.0 | 16/16, 8/16 | 16/16, 8/16 | 16/16/1/1/1 | 0/16, 0/16 | 16/16 | 14/16 |
| g32 | 572.7 | 16/16, 0/16 | 16/16, 0/16 | 16/16/7/6/6/2 | 0/16, 0/16 | 16/16 | 14/16 |

Drop-return列来自独立1200tick持续桥接协议，字段为 first pickup/pickup-return/25%/50%/己方半场/capture；g32实数为16/16、16/16、7/16、6/16、6/16、2/16。该协议的drop为预声明因果trap fixture，不是模型自然击杀证据。固定32主评/holdout的toward、arrival、first pickup、pickup-return全部七点16/16，但g24之后的capture退回0；完整2400自然链七点均首拾0/16、capture0/16。拓扑600tick drop fixture g32首拾16/16、capture0/16；carry near/mid各16/16，far2/16。Route-break 16局中仅1局safe-win，且其场景未证明有效开墙形成了稳定目标路长收益，故不记作有效开墙成功。

**Reward与训练曲线**

从32条真实训练事件按 `component_names` 与 `reward_components` 对齐求和：route shaped reward总计-1,440.963；其中正分量+3,281.750、负分量-4,722.713，32代每代route reward均非零。累计combined_total为2,847,457.344，route净值约-0.0506%。route potential delta总计-28,819.261；拓扑gain事件18,608、INF→finite 151,535、phase transition347,916、invalidated64,589、terminal correction94,235、安全拒绝23,438。以上都是训练批次和日志分量总和，不代表单局收益或评测成功数。mean reward从g1 .07456逐步降至g32 .04178，中间最低g9 .03677，后段略回升；不能只以通过状态报告训练效果。

**父点对照与结论**

父点H8 v34 g32的同协议drop-return为pickup16/16、25%1/16、50%0/16、home0/16、capture0/16；本实验g32为16/16、7/16、6/16、6/16、2/16，诊断性返程进展有所改善，但固定32捕获两seed组仍0/16，full自然局没有变化。父点与Treatment Tactical v2均64局：surviving causal kill由12/64降到9/64，steal由9/64（父点值按记录为0.140625）升至14/64，own-bomb defeat均0。Hunter每难度均64局：hard surviving causal kills 9→1，normal 8→5，easy 10→9；Treatment deaths hard/normal/easy均0，父点为2/0/1；Treatment own-bomb defeats均0，父点hard有2例。总体显示进攻效能退化而死亡安全未恶化，不能只看存活。

综合判断：固定限时主目标和自然full无改善，g24的固定capture短暂8/16但g32回落，路线训练净reward为负，Tactical/Hunter攻击保持下降。当前单变量证据不足以支持按λ=.05继续同配置训练；保留实验实现与回放用于诊断，不自动启动后续训练。评估原始产物位于run namespace下 `authority_je09_v1/`，训练reward原始事件位于 `formal/bun392_bridge_pfsp8/events.jsonl`。

## v36 连续格内势能（attempt02）

实验ID `JE09-STAGE-H8-ROUTE-PBRS-CONTINUOUS-20261017-v36`。首次技术 attempt namespace `runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36/` 因8卡smoke OOM失败，`optimizer_updates=0`，保留为失败证据、不计作训练模型。有效 attempt02 namespace 为 `runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36_attempt02/`，从不可变 H8 v34 g32 raw checkpoint独立启动，不从v35续训。固定 λ=.05、γ=.995；唯一变量是把路线势能的整数格距离替换为连续位置到可达生产网格的加权路径距离。32代、67,108,864 fresh ticks完成，8副本一致、finite/nonzero，4096 Adam steps/设备，训练墙钟1,918.95秒，fresh CPU restore通过。g0/4/8/12/16/24/32各544局评估完整。

下表包含fixed32、1200tick drop-return、full2400、Tactical v2、Hunter hard/normal/easy与carry near/mid/far。每点总计544局。carry远中近任务是初始化持包诊断，不等于自然取包。评估队列全体任务并行的统一起止墙钟未记录；表中计时为各评估artifact记录的wall/elapsed秒数求和，并行阶段可能重叠，只作可复现的评估工作量记账，不解释为去重后的实际墙钟。

| 点 | 局数 | 计时加总(s) | fixed首拾主/holdout；capture主/holdout | drop首拾/返程/25%/50%/己方半场/capture | full首拾/capture | carry_far | Tactical kills | Hunter hard/normal/easy kills |
|---|---:|---:|---|---|---|---:|---:|---|
| g0 | 544 | 648.5 | 16/16,16/16；0/16,0/16 | 16/16,16/16,1/16,0/16,0/16,0/16 | 0/16,0/16 | 3/16 | 12/64 | 9/64,8/64,10/64 |
| g4 | 544 | 676.3 | 16/16,16/16；0/16,0/16 | 16/16,16/16,4/16,3/16,3/16,0/16 | 0/16,0/16 | 2/16 | 10/64 | 2/64,7/64,9/64 |
| g8 | 544 | 651.0 | 8/16,8/16；0/16,0/16 | 8/16,8/16,0/16,0/16,0/16,0/16 | 0/16,0/16 | 3/16 | 18/64 | 0/64,8/64,13/64 |
| g12 | 544 | 653.6 | 16/16,16/16；0/16,0/16 | 16/16,16/16,6/16,5/16,5/16,4/16 | 0/16,0/16 | 2/16 | 18/64 | 1/64,5/64,8/64 |
| g16 | 544 | 635.1 | 16/16,16/16；0/16,0/16 | 16/16,16/16,0/16,0/16,0/16,0/16 | 0/16,0/16 | 2/16 | 10/64 | 0/64,4/64,9/64 |
| g24 | 544 | 675.5 | 16/16,16/16；8/16,8/16 | 16/16,16/16,0/16,0/16,0/16,0/16 | 0/16,0/16 | 2/16 | 14/64 | 2/64,5/64,10/64 |
| g32 | 544 | 655.5 | 16/16,16/16；0/16,0/16 | 16/16,16/16,0/16,0/16,0/16,0/16 | 0/16,0/16 | 1/16 | 15/64 | 1/64,8/64,10/64 |

表中drop列依次是首拾、pickup-return、25%、50%、己方半场、capture；fixed32主评与holdout的首拾/pickup-return均相同。父点H8 v34 g32的1200tick drop-return为首拾/返程16/16、25%=1/16、50%=0/16、己方半场0/16、capture0/16。v35 g32为16/16、7/16、6/16、6/16、2/16；v36 g32变为16/16、0/16、0/16、0/16、0/16。因而连续格方向信号没有改善拾取后的限时返程交付。v35/v36 g32相同carry诊断near/mid/far分别为16/16、16/16、2/16与16/16、16/16、1/16；这些是持包初始化能力，不是完整实战链。v35/v36原始逐局carry拓扑JSON保存在 `reports/je09_bun_20261017/route_pbrs_v36_carry_diagnostics/`。

自然full2400在全部七个checkpoint均首拾0/16、capture0/16；g32没有自然完整对局成功。另一个真实网页full任务也是16局无steal/capture。不能把carry、预置包或drop fixture结果计入自然full。route-break任务没有经归因证明的稳定有效开墙收益；训练期间拓扑gain事件不等于有效开墙成功。

战斗方面，v36 g32 Tactical v2 surviving causal kills 15/64，高于v35 g32的9/64及H8父点12/64。Hunter hard/normal/easy为1/64、8/64、10/64；相较v35 g32的1/64、5/64、9/64，normal/easy略升、hard持平；相较H8父点9/64、8/64、10/64，hard显著偏低。v36 Tactical trade为2/64、own-bomb defeat为0/64；Hunter死亡hard/normal/easy为0/64、1/64、0/64，normal自炸1/64。上述是64局筛查，不作显著性主张。

正式事件32代route shaped reward合计-1,250.406（正项+2,336.445、负项-3,586.851），combined total 2,792,851.469，route净项约-0.0448%；route potential delta -25,008.114。topology gain 18,326、INF→finite 150,386、phase transition 348,288、invalidated 63,429、terminal correction 97,321、安全拒绝23,845。mean reward从g1 0.07427降至g32 0.03815，g9最低0.03725。路线塑形优势相对原始任务优势的逐样本/逐checkpoint对照未记录，不能声称优势证据已证明有效。

v36网页自然局录像已加入 `web/replays.json`，ID为 `bun392-je09-route-pbrs-continuous-v36-g32-seed-2026101736`，来自真实g32 Actor、生产 `web/sim.js`、seed 2026101736、Tactical v2、300 ticks。模型放泡16次、对手4次，模型死亡0、对手死亡2，最终未决。逐动作确定性重模拟通过。模型JSON SHA256 `3643e9e2bbf4b2c48dc40a8fff5e2fb2191811ff9df23e0849c6aa80a39bf4b6`，raw policy/checkpoint SHA256 `e2dd2cb9890abf5753b34adcdb14abc93b3f977229c29326fd2033e584a73945`，回放SHA256 `570066f6e04db5bec231dcb843b750d65aa82e33be68abfd769e8d9f9458f4a0`。这段录像是自然对局但没有赢，不构成full运包成功证据。原始checkpoint在attempt02的 `formal/bun392_bridge_pfsp8/gen000032/`，导出在 `authority_je09_v1/bun392_bridge_pfsp8_g000032/model/`。

v36结果说明连续格势能计算和完整预算训练已完成，但自然full没有变化，fixed32最终capture仍为0，drop-return后段和carry_far相较v35下降；Tactical与normal/easy战斗有回升，hard Hunter仍低于父点。保留实现、checkpoint和录像用于分析；不自动重训或发布。

**网页回放交接**

`route_pbrs_replay_g32/`包含Actor `actor_model.json`/`actor.onnx`/`actor_policy.pt`、同代Critic `critic_params.npz`、`training_config.json`、真实固定seed逐tick轨迹`trajectory.json`和`recalculation_proof.json`。轨迹由g32网页Actor与生产`web/sim.js`生成，seed 2026101732、1200 tick、assist-off；它明确是掉包返程诊断fixture，不能展示为full自然成功。独立复算检查1,200条轨迹状态连续，逐tick`delta=.995*phi_after-phi_before`及`shaped=.05*delta`最大残差0。Actor JSON SHA256 `7bdfea4b1f8d3cc05a931cf2760ac8d5edc64494f49d0050dcf7b2079f336a2c`；Critic NPZ SHA256 `3174b62f39ea9d56f206c917d2107a8f5e31bda2fe693e88630f72715c0604d8`；轨迹SHA256 `a98bcdaee0e3c5ca73049f0bae42fcfd571da083737a929aba00c3c4673d388c`。Critic导出从g32同代`state.pkl`的`live.params.critic`逐数组验证相等，源checkpoint SHA见handoff。
