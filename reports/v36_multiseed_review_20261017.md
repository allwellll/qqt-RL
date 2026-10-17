# v36 attempt02 g32 八局 Review 预声明与验收（20261017）

## RED 契约（生成任何新对局之前冻结）

- 基线 `640b5a7`；仅当前 worktree、单线程代理、CPU 离线诊断，不启动训练或其他代理。
- 固定 raw g32 checkpoint SHA-256：`e2dd2cb9890abf5753b34adcdb14abc93b3f977229c29326fd2033e584a73945`，training state SHA-256：`6bc988b647d4e0fad8dcef66dd47bbc8d3b4ea7dd75c3a8b2530c9b01fba55bd`。
- 以下八局全部保留，不按输赢、失败种类、路线或优势结果筛选，不替换 seed，不从额外候选局中择优。异常也记为失败或阻塞，不静默跳过。

| seed | 场景 | 对手 | 上限 |
| --- | --- | --- | --- |
| 2026101780 | native 原生砖块、糖泡、道具 | bun.hunter/normal | 2400 tick |
| 2026101781 | native 原生砖块、糖泡、道具 | bun.hunter/normal | 2400 tick |
| 2026101782 | open 清除可破坏砖，其余原生规则 | bun.hunter/normal | 2400 tick |
| 2026101783 | open 清除可破坏砖，其余原生规则 | bun.hunter/normal | 2400 tick |
| 2026101784 | carry_home 预置持敌包、清砖、1HP | bun.hunter/normal | 300 tick |
| 2026101785 | carry_home 预置持敌包、清砖、1HP | bun.hunter/normal | 300 tick |
| 2026101786 | tactical_v2 生产 Tactical Arena 清砖、无道具 | bun.tactical_v2 | 2400 tick |
| 2026101787 | tactical_v2 生产 Tactical Arena 清砖、无道具 | bun.tactical_v2 | 2400 tick |

Actor 使用同一导出、greedy 与生产合法联合 move×ability mask。场景初始化、规则开关与终局必须明确；持包初始化不能声称自然抢包成功。每个动作记录双方动作、状态 hash、事件、奖励分量、路线阶段/目标/连续势能前后/PBRS、实际可证明训练信息。缺测保留 null 和原因。

严格优势定义：`Q1=r+gamma*(terminal?0:V(next))`，`A1=Q1(taken)-sum(pi_legal*Q1)`，合法联合概率归一，全部候选共享同 checkpoint、对手动作、RNG 与隐藏奖励账本前态。独立原始任务 Critic 与塑形 Critic 必须分开；跨口径 `A_shaped-A_raw` 与时间差 `A(t)-A(t-1)` 分开，首步/断档为 null。Reward、Value、Logit 不得冒充 A1。

优先审计并使用冻结 SOURCE_SEAL、COMMITTED、训练 trajectory/critic/ledger。自然录像若无法严格计算，检查真实保存训练轨迹能否另导出明确标注的训练轨迹录像；没有真实轨迹时报告证据与阻塞，不把新生成评估局冒充历史训练局。

发布仅新增 `web/review-replays/` 与 `web/review-catalog.json` 的录像入口；不向旧 replays 链路新增。必须逐动作重新推理、确定性状态复算、schema/hash/HTTP、Review 与全量测试、隐私扫描，再普通快进 push main，等待精确提交 SHA Pages success，在线逐条校验 catalog、HTTP200、hash 和深链。

## 审计与最终结果

已实际观察缺生成器的 RED 失败；跨日执行结果与用户后续放宽的近似诊断口径见 `reports/v36_multiseed_review_20261018.md`。本文件中的八seed与不择优契约保持冻结。
