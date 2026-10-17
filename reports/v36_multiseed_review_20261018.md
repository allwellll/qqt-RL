# v36 attempt02 g32 八局 Review：当前奖励离线近似诊断

## 已冻结的范围与口径调整

八个 seed 在任何新局生成前写入 `reports/v36_multiseed_review_20261017.md` RED 契约，并实际观察缺生成器的 RED 失败。基线 `640b5a7`。矩阵为 2026101780–2026101787，原生、开放、预置持包返程、Tactical 各两局；全部保留，不择优、不换 seed、不修改动作。全部使用同一 raw v36 attempt02 g32 checkpoint 和真实生产模拟器/对手。没有启动其他代理、训练或模型服务。

本轮用户随后明确允许采用仓库当前 reward/路线逻辑在真实 v36 状态上近似重算。最终字段统一标为 `offline_recomputed_current_reward`；当前函数和可用旧依赖字节码混合加载，逐文件记录 SHA-256。该标记不能表述为 training-time reward。严格 A1 规则仍保持。

## 来源审计

- raw policy/checkpoint SHA-256：`e2dd2cb9890abf5753b34adcdb14abc93b3f977229c29326fd2033e584a73945`。
- sealed state SHA-256：`6bc988b647d4e0fad8dcef66dd47bbc8d3b4ea7dd75c3a8b2530c9b01fba55bd`；重新计算整个文件，与 COMMITTED 一致；raw Actor 与 state replica0 数组精确一致。
- v36 同代塑形 Critic 有 75 个参数叶，8 副本精确相等、全部 finite；规范张量摘要 SHA-256：`b7f9e0bb09e3f4b4d0ca6ca912269b549b3b56a1c795209ab78ed1fb8ff6bbf3`。没有独立原始任务 Critic。参数存在不代表严格 Q/A 已测。
- state 的 live 字段含当前 states/aux/params/optimizer/pool 等，不含历史逐动作 rollout；完整 namespace 未找到 trajectory/rollout 文件。原有 `route_pbrs_replay_g32/trajectory.json` 是 v35，不能用于冒充 v36 训练轨迹。
- 不可用的环境 namedtuple 类仅在 checkpoint 清单审计时使用中性 tuple 占位；参数 ndarray 正常读取并独立比较。占位状态没有送入 Critic、模拟器或录像。
- 当前奖励重算使用真实 Web 逐 tick 状态、bun-update 前状态、原生掉包/死亡/拾取事件、爆炸 owner 和方向覆盖；生产 CPU safety 检测；当前 `je09_stage_h8_pbrs_ops.combined_reward`、连续路线 PBRS。aux 在声明初态冷启动；scene 映射 native=8/open=0/carry_home=2/tactical=0；terminal 来自真实 Web Sim.done；danger_exposure 用当前 selected-action exposure 的 any 近似；bridge window=120/scale=2。以上均为近似诊断假设，不是训练一致性证明。

## 八局原始结果

| seed | 场景 / 对手 | tick | 终态 | 规则失败标签 | 模型近似总 reward |
| --- | --- | ---: | --- | --- | ---: |
| 2026101780 | 原生 / Hunter normal | 1771 | 负 | 窗口未切换、对局失利、未交付 | -12.043708 |
| 2026101781 | 原生 / Hunter normal | 2145 | 胜、真实交付 | 窗口未切换、返程偏航 | 39.343877 |
| 2026101782 | 开放 / Hunter normal | 552 | 负 | 对局失利、未交付 | 1.500000 |
| 2026101783 | 开放 / Hunter normal | 809 | 负 | 对局失利、未交付 | 1.300000 |
| 2026101784 | 预置持包返程 / Hunter normal | 92 | 胜、交付 | 无上述规则失败 | 1.437500 |
| 2026101785 | 预置持包返程 / Hunter normal | 107 | 胜、交付 | 无上述规则失败 | 1.000000 |
| 2026101786 | Tactical Arena / Tactical v2 | 2400 | 平 | 窗口未切换、未交付 | 0.817124 |
| 2026101787 | Tactical Arena / Tactical v2 | 2400 | 平 | 窗口未切换、未交付 | 1.004600 |

原生胜局是本次声明样本内的观测，不推翻先前不同评估协议的统计结论。持包初始化两局不能算自然抢包成功；Tactical 清砖、无道具和默认 Sim 规则与原生场景分别标明。失败检测不是完备行为分类；没有观察到标签不能证明无问题。

## 字段、覆盖率与限制

共 10,276 个真实动作。每 tick 都保存双方原始动作、合法 move/ability mask、真实 Actor 策略概率、状态前后 SHA-256、随机流 draw count、原生 info hash、事件，以及近似当前 reward 的 20 个分量、双方近似总 reward、当前路线 phase/target/来源、连续 Phi 前后、delta、PBRS、终态修正。零分量也在逐 tick 诊断中明确输出；页面奖励轨使用等价稀疏非零表示。路线项已计入近似总 reward，不再重复叠加。

动作/状态/事件与当前近似奖励、路线字段覆盖率均为 100%。训练时 Reward、严格原始 A1、严格塑形 A1、跨 Critic 差与相邻严格 ΔA 覆盖率均为 0%，字段保持 null。未计算 `approx_value` 或 `approx_A1`，没有把 reward/value/logit 改名成优势。因没有保存的 v36 历史训练 trajectory，本次没有另造“训练录像”。

当前账本在 seed 2026101781、tick 714 输出 Phi 跨步不连续：前一步 Phi_after 为 0.1352008879，本步 Phi_before 为 0.1239508688，跳变 -0.0112500191。保留原值并标为 `approx_phi_jump` / `approx_phi_discontinuity`；当前 tick 的 PBRS 仍由该 tick 的真实重算输入满足公式。已核对真实状态：tick 713 砖数从35降至34，砖格185被清除，当前 route_safety_rejected=1、topology_gain=0；该步重基准保持 Phi_after，下一步恢复当前距离后出现跳变。该现象与当前 unsafe_map_change 分支只替换 base_next、下一步重新读取当前距离的行为吻合。它是可定位的近似账本异常候选，不能从近似桥接单点证明训练因果。

## 验证与发布

已完成八局原始双方策略/真实对手的完整独立重推理，动作、状态和事件逐 tick 精确一致；当前奖励在独立 Python 进程八局全量复算精确一致；新 schema/hash/HTTP/伪造 A1/错误分量与 scope 拒绝测试通过。Review 回归通过，当前路线/课程 Python 测试14项通过；桌面/手机共16条真实深链、逐局三个 tick 数值、刷新定位和零 HTTP/console/page error 通过。`npm test` 全量 Web/榜单/Review 回归、单独 `npm run test:review` 与隐私扫描全部通过。隐私扫描仅覆盖 git ls-files 与本次待提交文件；用户标识、内网地址、机器PID和含标识绝对路径均为零。发布提交后将按精确SHA等待 Pages，并将最终SHA、部署及八局线上HTTP/hash/深链结果保存至忽略的 `runs/replay_review_20261017/v36_release_acceptance_20261018.md`，避免为记录提交自身SHA再触发部署。


## 定位建议与复现入口

- 开放两局分别获得 +1.5 / +1.3 的近似 reward，来源完全是 settled_shaping，但均未交付且输局；可优先检查战斗辅助收益与最终任务结果的脱节。
- 原生失败局 settled_kill +16、settled_shaping +5.8，随后 combat_clawback -21.8、global_safety -12；总值 -12.043708，页面可逐事件定位结算、追回与死亡顺序。
- 原生胜局交付在 tick2145；仍有窗口未切换和返程偏航标签；其收益大头是 settled_kill +32，路线净项约 -0.081123。这只是当前近似分解，不能把差值解释为训练因果或模型优势。
- 势能异常深链：`review.html#replay=h8-v36-g32-native-2026101781&tick=714`。页面显示原始动作/事件、20项近似分量、当前目标、连续Phi与跨步跳变，严格优势轨保持空。

复现流程为 `scripts/audit_v36_review_inputs.py` → `scripts/generate_v36_multiseed_review.js`（CPU ONNX，八局全量复算两次）→ `scripts/export_v36_reward_trace.js`（只在忽略的 runs 下输出私有全状态）→ `scripts/recompute_v36_current_reward.py`（八局，另进程 `--verify`）→ `scripts/finalize_v36_current_review.js` → `npm test` / `npm run test:review` / `scripts/verify_v36_review_browser.js`。Python 使用 `<PYTHON>`，ONNX/Playwright/Chromium 外部依赖分别通过已有环境变量指定中性位置。近似重算还依赖本机已有、已hash的旧Python字节码；纯Git检出不能单独恢复这些缺失源码，所以公开结果是已核验导出，不宣称干净检出能从头重训或重建训练ledger。
