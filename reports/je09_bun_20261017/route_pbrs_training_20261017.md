# JE09 H8 Route PBRS 终态报告

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

**网页回放交接**

`route_pbrs_replay_g32/`包含Actor `actor_model.json`/`actor.onnx`/`actor_policy.pt`、同代Critic `critic_params.npz`、`training_config.json`、真实固定seed逐tick轨迹`trajectory.json`和`recalculation_proof.json`。轨迹由g32网页Actor与生产`web/sim.js`生成，seed 2026101732、1200 tick、assist-off；它明确是掉包返程诊断fixture，不能展示为full自然成功。独立复算检查1,200条轨迹状态连续，逐tick`delta=.995*phi_after-phi_before`及`shaped=.05*delta`最大残差0。Actor JSON SHA256 `7bdfea4b1f8d3cc05a931cf2760ac8d5edc64494f49d0050dcf7b2079f336a2c`；Critic NPZ SHA256 `3174b62f39ea9d56f206c917d2107a8f5e31bda2fe693e88630f72715c0604d8`；轨迹SHA256 `a98bcdaee0e3c5ca73049f0bae42fcfd571da083737a929aba00c3c4673d388c`。Critic导出从g32同代`state.pkl`的`live.params.critic`逐数组验证相等，源checkpoint SHA见handoff。
