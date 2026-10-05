# 备用泡补充与原版出生保护验证 20261005

## 范围与基线

- 同一 worktree：`cx_bot_cooperation_20261004`，分支同名。起点与本轮 fetch 的最新 main 均为 `5ecb5c9ee5725b8262cbd140158fb1065c0fed62`。
- 不重复上一轮方向键、大鸟与死亡掉落工作。本轮只扩展协作猎手备用泡、安全检查和网页原生出生/复活保护。
- 对照从 Git 加载 `5ecb5c9` 的协作策略，双方使用当前同一模拟器、地图、道具规则及 seed。这样隔离 bot 改动，不把无敌规则恢复误算为 bot 提升。

## 原版依据

- 来源：`kuuhaku1314/qqtang` v1.3.2 源码与同版本官方 `QQTang-Local.zip` 的 `runtime/client-patched`。上游 Go 源码并不直接实现出生保护，不能仅用变身恢复常量推断出生时长。
- 官方 `Client.exe` SHA-256：`670af4f24b3db6bac030a61d628cedc8e4522468f4ef4cb88a2239ad6b946f74`。
- 角色初始化函数 `0x5afdfa`：`0x5aff67 push 0xbb8`，`0x5aff72 call 0x5b5202`；参数 **3000ms**，保护对象为 actor `+0x318`。客户端本地角色权限检查后安装保护，网页模拟器统一仲裁所有角色。
- 复活协议 `NOTIFY_PLAYER_RELIVE = 0x0FBA`，源文件 `internal/protocol/game/game_event.go`。客户端处理器 `0x603770` 验证 schema，`0x603813 -> 0x5c0b08` 切换状态，通过 vtable `+0x28` 进入 `0x5c0edd`。其混淆指令路径最终调用同一计时器 `0x5b5202`，参数仍 **3000ms**。复活不是简单套用变身时长。
- `scripts/probe_native_spawn_protection.py` 使用 pefile/unicorn 对本地 PE 进行指令仿真，固定客户端哈希；仅替代状态管理器的前态 ID getter，其余计时器参数准备原样执行。前态 ID 0/9 均在 2458 条指令后到达保护 setter，参数 3000、actor 偏移 `0x318`。这是一段状态入口的指令证据，不是完整原生 Windows 对局验证。
- 保护 setter `0x5b5202` 设置时长与 active flag，加载 `effect/flash2.eff`；`0x5b5244` 更新时长至 0 后撤销 active，`0x5b528f` 仅 active 时循环绘制。伤害门 `0x5ad175` 检查此保护对象，因此普通爆炸伤害不能进入糖泡。
- 上游 `internal/game/battleengine/active_item.go:254` 在场地接触门检查 `actorHarmProtected`，位于 action 41/42/43 分发之前：出生保护也阻止香蕉皮/慢慢胶接触触发，不消耗地面道具。普通拾取、主动放泡与移动照常；伤害免疫不是穿泡或上墙权限。
- `flash2.eff` 使用 `magic139`、100ms 帧间隔和 1500ms 动画段；客户端保护计时与动画段独立。实际资源 `object/magic/magic0139.img`，100x113、原点(-48,-90)、15帧。不是 `baohuzhao.eff` 的道具罩，也不能把 1500ms 动画长度当保护时长。

## 实现

- 原生 `nativeTrap` 模式出生/复活设 30tick（10Hz = 3000ms）。`invuln` 继续供判伤和 bot 读取；新增独立来源计时 `spawnProtection` 供光环渲染和快照，死亡清除；旧回放缺此字段时不补造光环。
- 原版 `magic0139` 位图导出为 `web/assets/native/protection.png`，记录帧数、锚点、100ms 帧间隔、1500ms 循环和3000ms保护元数据；按原始宽高比/坐标换算绘制。PNG SHA-256：`cb9944101da08d2c617bc93b06160971e0be6bc4b52bf6489fba630979ea980e`。
- 训练/模型默认路径没有出生保护，新字段不进入其快照哈希。4组原生金样有意更新，8组训练金样与基线逐项一致。
- 备用泡不提高在场上限（仍至多4、受实际角色容量约束）。从零布置需留3个容量；补已有未连接的己方泡需再留2个容量，保留连接泡额度。
- 目标由观察到的短期行进趋势及通往己方基地/己方搬包者的前几格路径决定。路线推测仅限附近、至少两个有价值路线格，不因远处理论路径到处空埋；优先真实当前/短期射线目标与墙角。
- 第二枚与锚泡双向均不可直接相连，连接点需可步行到达且同时接上两枚。补充已有泡只选剩余13..27tick、距离5格内的己方泡，不能临爆后才开始漫长布置。
- 到达连接点等待最早锚泡剩余不超过12tick，并在落连接泡前重新确认目标当前/短期路径进入组合火线。目标临时换道可在准备窗口继续等待，但临爆时无目标便取消；不是强行每轮补一个连接泡。
- 救援、搬包、滑行、无容量、场地堵塞、锚泡消失、目标失效、8tick未推进或计划到期均取消，重新预置冷却12tick；不让计划无限占用任务。护送、拦截、回防可用准备好的连接机会，但仍受搬包路线安全门约束。
- 每次实际新增泡用全场可见泡重新预测，检查本人首步、可用终点余量、连续坐标撤离，保留3tick余量；对队友同样检查，并保留已声明首步。未声明/停留的队友，包括真人，不假设其主动逃生：其当前格不能新增火线。
- 预先检查未来组合的本人/队友连续撤离；实际每次落泡再检查。回程当前格、前行格、预计到达时间和整条路径不能新增火线或被泡阻断。连续撤离搜索避开会触发移动状态的场地道具。
- 临爆移动增加连续坐标筛选；格子搜索判“必死”时，再检查中心格伤害语义下的真实首步，避免有实际逃路却原地不动。有限候选存在性检查不是形式证明，也不能预知对手下一次放泡/改变路线。

## 自动化与网页

- `web/test_spawn_protection.js`：出生/复活30tick、本人/敌方/友方爆炸免疫、到期首个tick恢复糖泡判定、香蕉/慢慢胶免疫及到期接触、普通拾取和主动放泡允许、普通泡仍碰撞、死亡清除、快照独立复制/过期/旧格式兼容、训练路径不变与素材元数据。
- 协作测试保留真实两枚分散预置、晚连接、三泡同tick连爆困敌及本人/队友存活；新增已有泡真实补充、搬包/滑行取消、未声明队友拒绝新增火线、连续坐标真实躲过自己即将爆炸的泡。
- `npm test` 21个脚本通过，`git diff --check`通过。`scripts/verify_reserve_spawn_browser.js` 可复用，使用 `PLAYWRIGHT_MODULE`/`CHROMIUM_PATH`/`WEB_URL`。
- 桌面1440x1000和手机390x844：真实 app 默认协作猎手；tick0/8/19落第一枚、第二枚和连接泡，连接时锚泡12tick，实际三泡同tick连爆困敌，两bot全程存活。真实 canvas >1000种采样色，无JS/资源错误、无横向溢出。
- 两端光环像素对照45196、动画差异46759、复活光环差异15239；回放恢复差异0、光环到期差异0。人工查看桌面/手机截图，光环随角色正确显示。旧发布版真实页面复现出生0tick、复活10tick。
- 完整806真实网页：seed19、2v2，真人pid0不操作、另三人为协作猎手；加速真实app tick，不修改地图/初始道具/终局。1151tick正常结算、比分[0,1]；94次真实放泡、16次新/第二预置、6次补充、6次连接攻击、23枚提前连爆、2死亡、0自困、0友困，81砖剩8。这只证明完整网页流程可执行，不是公平纯bot胜率评估。
- 本地证据目录：`runs/bot_reserve_spawn_20261005/`，JSON/截图为本地产物；固定seed JSON保存全部源码SHA256和候选受困事件，便于复查。

## 固定 Seed 对照

- 所有样本：806完整地图不清砖，原生道具/糖泡/出生保护，困难，2v2，上限2400tick；每seed双方换边；新版或基线协作猎手各自面对相同旧猎手，不是新版直接打旧版。总数及每局时长都保存在JSON，放泡理由不是实际命中。
- 初筛 `screen.json`（2seed各换边）：新0胜2负2平，基线2胜0负2平；新自困3/友困1、旧1/0。第二预置40次但空埋明显，未当提升证据。
- `confirmation.json`：2seed换边，新2胜1负1平、旧1胜0负3平；新自困4/友困3、旧1/0。`release.json`：新1胜2负1平、旧0胜2负2平；新10/1、旧8/3。两者均未满足安全目标，加入连续首步筛选、场地道具规避与撤离余量。
- `safety_trace.json`：1seed换边新旧都0胜2负；新自困6/友困1，旧4/2。受困事件显示部分角色在中心格可逃时被保守格子搜索判必死；新增实际复现回归及连续坐标后备逃生。
- `final.json`记录中间策略（运行期间后续补入静止队友保护，按文件哈希区分）：新旧各1胜1负2平，新自困4/友困3、旧2/1。不可冒称最终发布评估。
- **最终冻结策略 `final_release.json`**：独立seed起点2026104907、2seed各换边，新旧各4局均**0胜3负1平**。

| 指标 | 新版 | 基线5ecb5c9 |
|---|---:|---:|
| 实际放泡 | 549 | 480 |
| 进攻意图泡（非挖砖理由） | 475 | 400 |
| 新泡当前射线覆盖敌人 | 204 | 167 |
| 实际提前连爆泡 | 232 | 193 |
| 第一枚分散预置 / 第二枚 | 55 / 17 | 12 / 3 |
| 已有泡补充 | 15 | 0 |
| 分散预置同时在场的player-tick | 429 | 79 |
| 预置泡在场的bubble-tick | 2466 | 450 |
| 预置泡实际被提前连爆 | 18 | 0 |
| 自然到期时射线没有敌人的预置泡 | 68 | 15 |
| 连接攻击落泡 / 堵路屏障泡 | 27 / 41 | 8 / 30 |
| 死亡 / 自困 / 友困 | 3 / 1 / 2 | 8 / 6 / 1 |
| 搬包tick / 护送tick | 550 / 457 | 184 / 162 |
| 实际道具拾取 | 72 | 93 |

- “射线没有敌人”是浪费风险代理，可能仍炸砖或形成迫走效果，不是精确无意义泡数；连接理由也不等于完成三泡突袭。自动真实场景证明三泡突袭执行，自然对局数据另列。
- 最终策略SHA256：`e7aef0245df8d1b918e19a0e8c3139266783744e34a53d1fa29ae2022cd58977`；模拟器SHA256：`db2131d262df536eeedd0458e39c9f9b309848c1c1912a46f811ec01e9058b5d`。
- 本轮单人3seed共4800次决策与发布版完全一致；训练8组金样未变。
- 新旧4局总时长分别6538/6620tick，实际放泡每1000队伍tick约83.97/72.51；该归一化只解释节奏，仍不是有效命中率。
- 最终评估JSON SHA256：`8aefcadef097a3f8444314a820bd03f3ed798bb8dae2867b0275368e921e28eb`；最终浏览器检查JSON：`ddd7ae35091538e274eaccedc020c7a16a56d220c0a03aaa06c422fe52529a36`。
- 结论：补充/分散/临爆连接实际执行更频繁，这4局自困和死亡较少，但友困更高、道具拾取更少，胜负相同。样本不足以证明稳定胜率提升或绝无误困；空炸代理计数仍说明有优化余地。

## Python 回归与发布

- Python全套初次因系统环境缺JAX无法收集；随后在临时venv安装项目`.[test]`，CPU模式收集201例并完整运行，最终 **192通过、9失败**，并非全绿。
- 已定位辅助头维度既有失败：`tests/test_aux_heads.py` 的 `test_aux_heads_present_only_when_requested`、`test_aux_forward_shapes`、`test_aux_bce_gradient_flows_into_backbone` 共3项失败。测试按3种放泡动作检查15维辅助输出，现有模型默认2种动作、10维辅助输出；在未改动main `5ecb5c9`使用同环境完整运行该文件，同样3失败/2通过。本任务不修改训练模型契约。
- 最终再次执行 `npm test`，21个脚本全部通过；再次运行桌面/手机/完整地图浏览器验证，结果及JSON哈希与前述一致。原版指令探针也再次通过。
- 其余失败：`test_counterfactual_batch_equivalence.py` 3项（动作维度3/2不一致）、`test_jax_bots.py::test_collect_rollout_with_jax_bot_pool`（同类维度不一致）、`test_eval_transformer_checkpoint_sweep.py::test_summary_paired_deltas_and_spawn_consistency`（临时测试环境缺matplotlib）、`test_training_modules.py::test_prepare_enables_repaired_v7_seed_namespace`（batch-size测试期望4，现有配置1）。
- 在未修改main `5ecb5c9`、同一venv/CPU环境，全量收集201项后用 `-k` 只执行上述9项，**9失败、192未选中**，失败名单完全一致。仅指定部分文件会改变收集时的配置副作用，不能替代这个基线核对。未将既有训练问题混入本轮网页行为修改。
- 补装matplotlib后，checkpoint汇总绘图专项 **1通过**。没有重复耗时约42分钟的全套，因此准确结果仍是全套192/9、补齐依赖后该专项通过；剩余8项为已复现的main训练问题。
- 后台全套及基线复现进程均已退出，无遗留评估进程；保留本轮网页预览服务供检查。

## 复现命令

```bash
npm test
git diff --check
node scripts/eval_bot_cooperation.js --pairs 2 --seed 2026104907 --max-steps 2400 --sizes 2 --reference-ref 5ecb5c9 --out runs/bot_reserve_spawn_20261005/final_release.json
node scripts/verify_reserve_spawn_browser.js
JAX_PLATFORM_NAME=cpu /tmp/cx_reserve_spawn_test_venv/bin/python -m pytest -q
python scripts/probe_native_spawn_protection.py /path/to/official/v1.3.2/Client.exe
node scripts/verify_reserve_spawn_pages.js <full-deployed-commit-sha>
```

- 浏览器检查需要 Playwright 与 Chromium，当前机器通过 `PLAYWRIGHT_MODULE`、`CHROMIUM_PATH` 指向已有安装；依赖库/字体分别使用 `LD_LIBRARY_PATH`、`FONTCONFIG_FILE`。本地服务 `http://127.0.0.1:8080/`，发布页面可通过 `WEB_URL` 指定。
- 原版探针需要 `pefile`、`unicorn`；素材导出需要原版资源及 Pillow。Pages检查从Git credential helper读取凭证，不在源码或输出中记录token。
