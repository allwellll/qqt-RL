# 面朝穿点规则与 P1 快筛（20261107）

## 各端穿泡机制核实

| 端 | 核实结果 | 代码依据 |
| --- | --- | --- |
| JAX 训练模拟器 | **未实现充能/放泡获得穿泡机制**。状态只有移动方向、滑行等，无 touch/pass 权限。存活泡始终进入 blocked。 | `jax_bomb/bun_env.py:474`、`jax_bomb/je07_bun_env.py:477`；底层 `jax_bomb/je05_core_env.py:1153`、`jax_bomb/jax_env.py:1150`。另一个训练工作树的对应 Python 目录只读检索同样未发现实现。 |
| Python Tactical | **未实现穿泡**。它是动作决策器，不是独立穿泡物理；规划使用地形邻接和泡的时间危险信息，没有充能/权限状态。 | `jax_bomb/bun_rule_bot.py:177`、`:271`、`:346`；`jax_bomb/bun_tactical_family.py:28`、`:56`。 |
| 旧 JS 网页真人原生移动 | **已实现**。只有移动碰撞探针遇泡才 `_nativeTouch`；放泡要求严格 `500 < charged < 600` 且 `since < 100`，原地松手不能持续刷新接触。 | `web/sim-bun-native-v2.js:2302`、`:2316`、`:2406`、`:2429`。 |
| 旧 JS 模型/bot 评估 | **没有启用这套穿泡计时**。只调用逻辑 `step`，其移动 blocked 始终包括泡；没有调用真人 `frameStep`，也不推进 nativeMove 时钟。 | `web/sim-bun-native-v2.js:1228`；`scripts/cx_web_temperature_worker_v1.js:5`、`:29`；`scripts/resume_gpu67_eval_v1.js:71`。 |

因此，不在训练端新增穿泡机制：训练分布和在训 GPU 臂保持原样，现有模型没有通过训练端这套机制学习过穿泡。网页加入新规则后，对两个冻结模型重跑 CPU P1 对比，不能把网页能力变化表述成训练提升。本任务没有调用 CC，没有修改另一 CX 的文件，也没有停止它的进程。

## 新规则与版本边界

新增独立冻结引擎 `web/sim-bun-native-v2-face.js`，规则名 `bun-native-v2-face`。旧 `web/sim-bun-native-v2.js` 字节保持原值，SHA256 为 `593b50e8278b5e544132f86dc4011c4dc06d0de1166c8ff8e751d9ac4f553154`。保留更早的所有 Review 引擎和映射。

“面朝”使用权威 `lastMoveDir`，当前非空移动输入先更新朝向；强制滑行沿用实际滑行方向。渲染在新规则下使用同一朝向。目标必须是角色中心主格正前方相邻格的存活泡，不能是脚下或斜对角泡。沿用 40px 原生格、±19px 两个前缘角点，再向前探测 3px 贴近带；两角跨行/列时，只取中心主格对应的行/列。墙、砖、越界或前缘尚未靠近均不成立。

每帧/每 tick（含 idle）观察面朝目标：同泡只刷新 touchLast，转向/走开保留 touchStart；转回同泡继续原时钟；面朝新泡重置 touchStart 并转移目标。真人时钟沿用 frameStep；逻辑模型和所有注册 bot 按 25ms 子观察推进每个 100ms tick 的充能，位移仍用原来的 10Hz solver，真人实时移动标记避免双计时。逻辑策略首次在第一个 25ms 观察点取得目标，持续面朝后 600ms tick 开始时 charged=575ms，可落入原来的开区间。

放泡成功时，必须仍面朝该目标，并保留严格 `500 < charged < 600`、`clock - touchLast < 100`。passDuration 仍按原速度穿行一格计算；原生的泡后空间检查保持原样，逻辑移动和 legalMask 同步应用该检查。计时/权限、目标和移动小数余量写入新版本快照，恢复不丢精度。模型参数、τ=0.3、推理编码、温度算法、地图、奖励、道具和爆炸规则不变。

Review 按 simulator SHA 选择原引擎，新版本增加单独映射；旧映射、录像身份和旧物理不变。新规则默认选择、build-info ruleset_version、快照 ruleset_version 和排行榜 cohort 使用 `bun-native-v2-face`。旧分榜继续保留，不混入新成绩。

## 圆环

前景线宽统一 5 个画布像素，9px 深色下描边提供两侧各 2px 的轮廓，半透明底轨。满圈实/虚线直接依据是否仍面朝目标；“面朝泡就开始计时，不用顶”图例已更新。离开后继续计时、换泡转移、600ms 过期的生命周期保留。

## 测试与真实 Chromium

- `web/test_pass_facing_rule.js`：静止面朝、无方向输入满圈放泡并穿越、背对拒绝、转回保留计时、换泡重置、500/600ms 开区间、斜角与主格、3px 贴近边界、逻辑策略的时钟/碰撞/mask、快照与小数余量恢复、泡后阻挡、无穿泡时其他物理逐 tick 一致。
- 旧引擎 SHA 不变，Review 版本选择与旧逻辑逐帧对比通过；完整历史 D Review 的 15 个固定种子及全部既有 Review 测试通过。来源兼容白名单仅精确增加新引擎映射，仍要求可逆恢复到原发布源码，不放宽其他变更。
- `npm test` 全部通过；SQL/PGlite 迁移测试覆盖三版分榜隔离、旧成绩保留、幂等、规则不匹配重试拒绝及权限；`git diff --check` 通过。
- 本地真实 Chromium 桌面 1280px 与手机 390px 均 PASS。原地全程未按方向键，160ms 时进度 32%，512ms 时满圈 ready，位置始终 `[5.5,5.5]`；仅按真实 Space / 手机触屏放泡获得权限，再按真实移动输入穿越至 x=7.1；权限按时过期，无页面错误、无横向溢出。
- 已实际查看桌面增长中与手机满圈截图。截图为本地验收产物：`.hermes/facing-local-1280-growing.png`、`.hermes/facing-local-1280-ready.png`、`.hermes/facing-local-1280-active.png`、`.hermes/facing-local-1280-passed.png` 及对应 `390` 文件。另以纯显示夹具核对砖块邻近背景和火焰上的圆环：`.hermes/facing-local-1280-brick.png`、`.hermes/facing-local-390-flame.png`，均已实际查看；这些对比图不改变真实对局物理。

## CPU P1 快筛

同一个既有网页 P1 categorical 种子协议，每组 64 局，固定模型在 seat=1。G_frozen+τ0.3、L2_S05g u019302+τ0.3，各跑新旧规则，对手为 Tactical v2、猎手 hard/normal/easy、协作猎手 hard，共 1,280 局。Tactical 使用网页实际清砖近距场，其他四组完整 806 地图；native items/traps、2400 ticks 上限、模型与 bot RNG 与旧协议一致。

使用 CPU ONNX（每 session 单线程），前三个真实输入与同权重 JS 参考比较，误差门限 0.002；沿用实际网页的 R3 编码、合法掩码与温度采样，未加部署边界辅助。每局保存动作和状态 SHA 检查点，并用独立新模拟器回放逐检查点验证。穿泡次数同时记录“放泡获权限次数”和“已获权限后进入存活泡格次数”，不把脚下泡的普通离开算成穿泡。

评估仅用非训练 NUMA 0/1 的空闲 CPU，nice 10，未占用 GPU。发现新快照恢复余量问题后保留了被取代的本地结果，只重跑新规则分组，最终表必须对应最终引擎 SHA。

每格均为同一套 64 局。交付沿用旧网页快筛的 `same_custody_delivery` 口径（全部来源）；死亡为该模型死亡事件总数。完整逐局结果与源码/模型 SHA 见 `reports/cx_pass_facing_rule_evaluation_20261107.json`。

| 模型 τ0.3 / 对手 | 旧 W/L/D | 新 W/L/D | 交付 旧→新 | 死亡 旧→新 | 穿泡授权/实际穿入 旧→新 |
| --- | --- | --- | --- | --- | --- |
| G_frozen_tau0.3 / bun.tactical_v2 | 63/0/1 | 63/0/1 | 36→36 | 0→0 | 0/0→0/0 |
| G_frozen_tau0.3 / bun.hunter@hard | 61/1/2 | 61/1/2 | 60→60 | 16→16 | 0/0→0/0 |
| G_frozen_tau0.3 / bun.hunter@normal | 63/0/1 | 63/0/1 | 63→63 | 2→2 | 0/0→0/0 |
| G_frozen_tau0.3 / bun.hunter@easy | 64/0/0 | 64/0/0 | 64→64 | 3→3 | 0/0→0/0 |
| G_frozen_tau0.3 / bun.coop_hunter@hard | 63/1/0 | 63/1/0 | 62→62 | 9→9 | 0/0→0/0 |
| u019302_tau0.3 / bun.tactical_v2 | 64/0/0 | 64/0/0 | 38→38 | 1→1 | 0/0→0/0 |
| u019302_tau0.3 / bun.hunter@hard | 62/1/1 | 62/1/1 | 62→62 | 8→8 | 0/0→0/0 |
| u019302_tau0.3 / bun.hunter@normal | 63/1/0 | 63/1/0 | 63→63 | 5→5 | 0/0→0/0 |
| u019302_tau0.3 / bun.hunter@easy | 63/0/1 | 63/0/1 | 60→60 | 15→15 | 0/0→0/0 |
| u019302_tau0.3 / bun.coop_hunter@hard | 57/3/4 | 57/3/4 | 56→56 | 13→13 | 0/0→0/0 |

两模型对协作猎手·困难的净胜差均为 **0**，没有触发下降超过 5 局的停止发布门槛。所有对手的 W/L/D、交付和死亡也完全一致。新旧规则模型/bot 穿泡使用均为 0：本轮自然对局未覆盖策略主动利用新机制；真人机制由针对性测试与真实浏览器穿越验收覆盖，不据此宣称模型学会穿泡。

1,280 局全部完成独立动作回放与状态检查点校验；另使用既有被动 source observer 再回放 1,280 局，核对交付/死亡统计且每个 SHA 检查点不变。CPU ONNX 与 JS 真实输入的最大误差为 1.767781104611288e-05。

## 排行榜、发布和隐私

线上只读 contract 核实当前仅接受 `stock-window-v1` 和 `bun-native-v2`。已准备 `supabase/migrations/20261107010000_pass_facing_ruleset.sql`，本地 SQL 真实执行测试通过；没有线上管理凭据，已请求本机凭据路径或由维护者应用迁移。新版本分榜后端未就绪时不能发布默认新规则，以免提交被拒绝。

**当前未发布**：表现门槛通过，但线上排行榜尚未应用新 cohort 迁移，阻止完整发布。已授权的代码、测试、真实本地浏览器和评估工作均已完成；不是等待再次确认常规 push 权限。

实现分支：`feat/pass-facing-rule-20261107`；代码提交：`2c003316b8e1cf8900a8e5719a717bb6bf4f8ae2`。正常本地提交完成，未 push main，未 force。完整逐局评估证据 JSON 已随代码提交；本报告为本地交付回填。

最终只读核对，远端 main 和公开 Pages build-info 的 SHA 均仍为 `b42f8d12bf711977940ad642dfc901fac1e34394`；这是已完成的 OSS 发布版。[已有 Actions 42251660818](https://github.com/allwellll/qqt-RL/actions/runs/42251660818) 的 head_sha 与该 main SHA 一致，`completed / success`。本次新规则没有触发 Actions/Pages 部署，也没有把本地浏览器结果冒称线上新规则验收。后端迁移应用并回读确认新规则 cohort 后，仍需正常 push main、核对新 SHA 的 Actions/Pages 和线上真实浏览器验收。

最终隐私扫描：已追踪文件、暂存新文件和本报告，私有标识/内网 IP/机器 PID 检查 0 项；额外扫描既有四个 OSS 凭据完整值及两个 AK ID 前缀，0 命中；commit 元数据使用中性身份。`git diff --check` 通过。交付不含凭据或本机私有绝对路径，所有运行和模型大文件仅留本地。
