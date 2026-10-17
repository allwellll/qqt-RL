# 路线回放导出接口 v1

本接口只转换已采集/已独立复算的结果，不训练、不加载 GPU、不根据 Reward 或事件猜测路线阶段。现有 `qqt.diagnostic-replay/v1` 保留原始 `advantages`、`rewards`、DAgger 与动作；新增可选 `meta.route` / `route`。旧文件不改字节，缺字段显示“未记录”。fixture 仅在测试目录及浏览器拦截响应中使用，不进公开 catalog。

## 数学与时间

转移 tick t 指 `s[t-1] -> s[t]`，与原 Reward/A1 共用整数 tick。

- `route_potential_delta = gamma * phi_after - phi_before`。
- `route_shaped_reward = lambda * route_potential_delta`。lambda 是训练最终选定的路线系数，不是 GAE lambda；gamma 必须来自绑定配置。
- Phi 已包含阶段 offset 和边界处理，不能额外叠加开墙奖励或 terminal 奖励。`terminal_correction` 是 nullable boolean，表示该转移包含终态边界归零；修正后的 `phi_after` 必须为 0。`terminal` 另列记录真实终态。
- 原始任务 A1 保持原协议；塑形 Critic A1 单独 `Q1=r_shaped_total+gamma*V_shaped(next)`，减合法联合动作的策略加权均值。`r_shaped_total` 是完整塑形训练 ledger 奖励，不是仅路线奖励。terminal next V=0。move×ability（含道具）的联合概率合法域重归一；固定同一生产状态、对手动作、RNG快照和有状态 ledger。
- 塑形 ΔA 只跨相邻有效决策，首项、缺口后为 null。不得把原始 Q/A 加路线奖励后充当塑形 Critic。
- 短期炸弹/火焰/人物/胶等不混入静态路长。`production_weighted_grid_v1` 表示生产硬墙不可达、空格进入cost1、可破坏砖进入cost `1+3*brick` 的精确加权网格最短路。

## 训练侧输入

调用 `scripts/export_diagnostic_route.js` 的 `convert(replay, source, independentlyRecomputed)`，或 CLI：

```sh
node scripts/export_diagnostic_route.js \
  --replay runs/route-export/base-replay.json \
  --input runs/route-export/collected.json \
  --recomputed runs/route-export/independently-recomputed.json \
  --output runs/route-export/route-replay.json \
  --report runs/route-export/export-report.json
```

输入 `schema = qqt.route-export-input/v1`，`replay_identity` 绑定输入录像的原始 identity；只有以下顶层字段：

| 字段 | 内容 |
| --- | --- |
| `meta` | 下述来源/配置元数据；不允许路径、任意附加字段 |
| `events` | 按 tick 严格递增的稀疏路线点观测；缺字段转换为 null，不向后延用 |
| `shaped_results` | 未测时 `[]`；真实塑形 Critic 的逐决策一步反事实结果 |
| `task_results` / `task_advantage` | 可选原始任务 Critic 反事实结果/现有 A1 元数据。已有原始测量时只校验，不覆盖；旧局尚无测量时按原始 A1 协议导出 |

`meta` 必填：`schema=qqt.route-potential/v1`、`kind=production`（测试只能 `fixture`）、`status=recorded`、`gamma`、`lambda`、`source_run`（仅脱敏短机器标识）、UTC `generated_at`、`distance_metric=production_weighted_grid_v1`、`checkpoint_sha256`、`policy_sha256`、`simulator_sha256`、`reward_ledger_sha256`、`route_implementation_sha256`、`shaped_critic=null` 或独立证据对象。source/recomputed 规范 JSON 对象 hash 和测量/缺测计数由转换器生成，无需上游填写。checkpoint/policy/simulator 必须与录像绑定；完整训练 receipt、配置、源代码及重算证据在正式导出时另附安全证据摘要。

每个 `events` 元素只允许以下字段，标量缺测填 null（不可用 0 冒充）：

| 字段 | 编码 / 语义 |
| --- | --- |
| `tick` | 已执行转移的整数 tick，1..episode ticks |
| `route_phase` | `NONE`, `B_STOCK`, `A_ENEMY_DROP`, `OWN_RECOVERY`, `C_RETURN`, `D_DELIVERY` |
| `target_type` | `enemy_cage`, `loose_bun`, `home_cage` |
| `target_provenance` | `enemy_stock`, `enemy_carried_drop`, `own_recovery`, `held_enemy_bun`, `own_cage`；A/B 阶段必须匹配来源 |
| `target_id` | 上游物体稳定身份。转换后只公开 episode域 SHA-256 `target-…`，不保留原文；无目标时三个 target 字段同时 null |
| `phi_before`, `phi_after` | 已含 offset/边界处理的真实有符号势能，finite 或 null |
| `route_potential_delta`, `route_shaped_reward` | 上游真实值，转换器校验公式和独立复算，不能只给 reward 后逆推 Phi |
| `distance_before`, `distance_after` | 非负精确加权路长或字符串 `INF`（JSON 禁止 Infinity）；null=未测 |
| `topology_region_before`, `topology_region_after` | 脱敏短机器键或 null；不能包含路径/身份 |
| `topology_gain` | 非负诊断量或 null，不额外相加进 reward |
| `topology_evidence` | 正增益必需 `[before_map_sha256, after_map_sha256, safety_evidence_sha256]`；其他可 null |
| `phase_transition`, `invalidated`, `terminal_correction`, `terminal` | nullable boolean；失效/安全拒绝时不能声称拓扑正增益 |
| `safety_reject_reason` | 短机器键或 null；现有中文映射：`unsafe,self_death,self_trap,teammate_harm,rescue_blocked,cage_blocked,lost_carried_bun,unreachable,target_invalidated,window_closed`。未知键醒目标为未翻译 |

上游其他字段必须先在适配层显式转换，不能直接倾倒训练 dict。训练侧阶段 ID 与公开枚举的转换必须依据真实状态机，禁止从事件/reward 猜测。相邻有测量的 tick 还校验前一 Phi_after 与后一 Phi_before 一致；有缺口时不声称整个 episode 的望远镜账本已验证。

只有真实塑形 Critic 训练/CPU parity/反事实均验证后才能填 `meta.shaped_critic`：

```json
{
  "schema": "qqt.route-shaped-a1/v1",
  "trained_on_route_reward": true,
  "checkpoint_sha256": "<同录像不可变checkpoint hash>",
  "critic_sha256": "<真实塑形Critic参数hash>",
  "reward_ledger_sha256": "<绑定路线训练ledger hash>",
  "counterfactual_sha256": "<完整私有反事实复核摘要hash>",
  "action_scope": "joint_move_ability",
  "legal_actions": "<真实生产mask和联合动作口径>"
}
```

`shaped_results` 每项：`{tick,taken:[move,ability],candidates:[{action:[move,ability],legal,probability,reward,nextValue,terminal}]}`。必须涵盖该决策所有生产合法联合动作、冻结对手动作/RNG/ledger；工具检查动作域/唯一性/执行动作、排除非法项、合法概率重归一、terminal和首项/缺口 ΔA。全量候选 Q/logits/状态只作私有输入，不进入输出。`task_results` 用相同逐决策结构，绑定 `task_advantage` 的原始 Critic/gamma/evidence；有原始 A1 时拒绝不一致结果。

`--recomputed` 应由独立 CPU 重放器从不可变 checkpoint + 生产 Sim + 路线 ledger 重建。转换器核对两份结果的量化列、配置与来源，并保存两份输入对象的 `SHA-256(JSON.stringify(input))`（按输入键序，无格式空白）；它本身不代替独立推理/RNG克隆/路由真实性验证。两份相同文件不能证明独立性，正式发布必须另审计复算程序、seed、tick样本、模型parity和receipt。源码中的合成 fixture 更不代表真实训练效果。

## 输出、紧凑性和发布门禁

`route` 为版本化量化列式对象：`scale=1e-6`；Phi/ΔPhi/路线reward/两种塑形Q/A/ΔA为整数列（未知 null）；`targets` 字典去重，`target` 列引用索引。其他枚举/距离/标志仍为稀疏列，所有列长度一致、tick唯一递增。塑形列名为 `shaped_critic_q1_taken`, `shaped_critic_q1_policy_mean`, `shaped_advantage_a1`, `shaped_advantage_delta`, `shaped_legal_count`。不存在逐tick完整state、全动作Q或logits。只保存观测点，不把 run 汇总铺成逐tick数据；查找只精确命中 tick。

identity 纳入新增 route 和 meta.route；旧局没有字段时 identity 不变。原始动作/Reward/状态hash与 catalog 文件不被转换器修改。CLI 使用 exclusive-create，拒绝覆盖文件；fixture 输出到 web 和任何 catalog 输出均被拒绝。报告包含完整文件和路线列各自 raw/gzip(level9)大小、hash、测量/缺测计数。网页仍逐局懒加载，没有额外全量 bundle。

正式发布还需要：预声明 seed×场景×对手矩阵、路线专项+完整局、不可变路线训练checkpoint/Actor/同代塑形Critic及 CPU parity、训练gamma/lambda配置与receipt、生产JS Sim/真实Bot/真实reward和路线ledger一致性、动作重推/终态/Reward和/statehash、固定tick独立反事实复算、目标/开墙/安全来源证据和隐私检查。页面 fixture 测试通过不解除这些门禁。

## 当前交接状态

训练侧设计报告说明旧 H8 v34 g32 未加入本次路线塑形。只读训练总账最新记录实验 `JE09-STAGE-H8-ROUTE-PBRS-20261017-v35` 为预注册/实施中；本轮检查时没有该实验的可读不可变正式 checkpoint 或完成的真实路线导出。本接口与 UI 可先提交，最终路线录像、main/Pages 发布和线上同SHA验收等待上述真实 artifact。网页任务不启动/停止训练，不使用 GPU，不写训练目录。
# g32 诊断交接语义

真实 v35 g32 交接使用 `scripts/import_route_pbrs_g32.js SOURCE_PACKAGE [WEB_OUTPUT]`，仅 CPU；显式设置 `CUDA_VISIBLE_DEVICES=''` 与可用的 `ORT_NODE_MODULE`。首次导入记录实际 UTC 时间，复算沿用已发布 receipt 的 `generated_at`；可用 `ROUTE_GENERATED_AT` 固定该时间以重生成相同字节。

`meta.route.semantics=diagnostic-post-target-uncorrected/v1` 只允许明确 `diagnostic=true` 的生产诊断场景：分类和目标取转移后的真实状态，前后距离按这个相同目标独立重算；末 tick 保留源导出器的非零 Phi，`terminal=true, terminal_correction=false`。这不是训练 PBRS 的终态归零语义，不能把该路线辅助奖励称作训练总 Reward。未声明该版本的原协议仍严格要求 terminal Phi=0。

本交接未包含原环境与训练奖励账本，使用 `reward_status=unrecorded`、`training_reward_status=unrecorded`、`summary.reward=[null,null]` 和空 rewards；显示未记录，禁止当零收益。原始/塑形反事实优势均为空列；Critic 参数来源校验不等于 Q/A 计算。独立审计 receipt 记录原始包所有 SHA、1200 次动作重推/状态/最短路/Phi 复算和缺测范围。目标 ID 按 episode 域 SHA 脱敏；拓扑增益、安全拒绝、失效未采集时保持 null。阶段切换从生产状态连续序列计算，不推断教师信号。

播放器只按允许的模拟器哈希加载 `review-sim/route-v35.js`（原生产源码字节不变），使用现有生产 renderer，不用轨迹快照冒充权威 Sim。旧录像继续使用当前生产 Sim；模型张量无需下发浏览器，页面资源同源，Git 仅存动作/量化路线列/周期 hash/小型审计证据。
