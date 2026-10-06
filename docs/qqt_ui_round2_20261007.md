# QQ堂 UI 第二轮与对手道具修复（20261007）

## 范围与交互

从 `5fb4dea` 继续原 `cx_bot_cooperation_20261004` worktree，按 `.hermes/qqt_ui_round2_contract.md` 实施。用户确认 SQL 与 submit-result Edge 已部署；本轮不修改或部署任何后端。Pages 测试只读取真实榜单，所有结算及资料写入口均 mock。

- 终局只提供「提交战绩」和面板右上角 ×，保留原 Canvas 胜负、比分、时间、服务端排名及 R 提示，没有第三个再来一局按钮。
- 状态：终局校验并暂存 → 显式提交 → 提交中 → 成功或失败可重试。关闭仅隐藏 Canvas 结果与操作面板，点击终局画布重新打开；关闭不授权提交。R 可开新局，未授权赛果保留本地队列；刷新也保留，默认不提交。高阶设置「提交待存战绩」是跨局/刷新后的明确补交入口。原队列上限20局、7天有效期及本地存储不可用提示保持不变。
- 本地 approval 标志不发给服务器；并发共享请求，成功后禁用提交，同一赛果重试始终沿用相同 match ID 与完整 payload，服务端幂等边界未改动。旧版已经授权的失败队列仍可重试；排行榜「刷新 / 重试」不能授权新草稿。
- 战绩提交中拦截 R/重开/设置触发的 reset。升级资料出现后，× 禁用、R 拦截，必须成功保存资料或显式跳过（保存后允许关闭/R；再次编辑则重新保护未保存内容）；资料请求尚未完成时 R 继续拦截。手机资料区位于 Canvas 下方，桌面资料区位于操作面板下方，不覆盖按钮。R 的输入聚焦、repeat/长按、修饰键和异步重开 gate 保留。
- 常显仅排行榜、人物、操作说明和两个入口按钮（公告、高阶设置）。完整设置清单：重开、补交暂存战绩、音效、地图、对局模式、队伍、策略、模型信息/下载进度/状态、离线录像选择/播放/重播/倍速/进度/状态、运行状态，均在高阶设置内。
- 每次打开高阶设置输入 `demaxiya`；错误显示游戏风格提示并清空输入。关闭或 Escape 锁定，再打开重新验证，刷新锁定；不存储或记录密码。这是前端便捷门。焦点在密码输入框、解锁后的关闭按钮和入口间按流程恢复；弹窗/设置输入拦截游戏键盘。
- 公告采用静态纯文本与 mailto，保留合同原意，支持 ×、Escape、合理遮罩关闭，原生 modal 加 Tab 首尾循环保障焦点，限制视口高度并允许正文滚动。

## 道具根因与规则

固定 seed `20261007` 最小复现保存于 `runs/qqt_round2_20261007/root-cause.json`。在真实基线代码和修复代码上执行相同场景：

| 场景 | 基线 | 修复 |
| --- | --- | --- |
| 对手在慢慢胶原格未离开，玩家踩入 | status=0，道具仍为2 | status=1，道具消费为0 |
| 已启用慢慢胶，真人在两个10Hz tick间高速跨越整格 | x=6.225，status=3，道具仍为2 | x=5.375，status=1，道具消费为0 |

根因是 `fieldArmed` 被当作所有人的总开关，以及接触仅在逻辑 tick 检查终点。Bot 与玩家共用 `_placeHeldItem`，没有不同放置路径或隐藏的敌人/team免疫。修复拆分启用和接触：未离开放置格时只保护放置者，对手可以立即触发；已启用后的放置者/队友仍按原设计触发。生成保护条件保持不变。

真人在原25ms像素移动子步检查实际中心接触，当子步进入慢慢胶即使用减速计算后续移动；香蕉触发后在同帧后续子步更新方向和滑行碰撞状态。Bot 逻辑移动按碰撞解算后的轴向路径检查跨过的格，覆盖自定义600px/s香蕉高速穿越。道具原子消费一次；同tick重叠按稳定player ID顺序，包含对手同tick放在玩家起始格的接触；仅豁免放置者从未启用的自身道具格离开。接触/状态均在原snapshot中，未增加非持久化接触队列或改变回放schema。香蕉和慢慢胶同源回归均补充，未修改无关玩法和黄金fixture。

## 验证与审查

- 完整 `npm test` 通过，日志 `/tmp/qqt-round2-npm-20261007.log`。新增 `test_field_items`、`test_settlement_actions`、`test_ui_panels` 纳入npm/CI；原12例逐tick金样、176例Bot parity、生成保护、原版物理、回放、全部排行榜/SQL/Edge本地测试保留通过。SQL测试为本地PGlite执行，不是远端部署或正式数据库写入。
- `git diff --check` 通过。
- `scripts/verify_round2_browser.js` 真实 Chromium 1440×1000、390×844通过，证据 `runs/qqt_round2_20261007/browser-local/checks.json` 和截图。覆盖真实Canvas/Sim/素材/榜单只读、胜负平超时、显式提交、关闭重开、受控422失败重试、完整payload一致、真实鼠标/触屏长按、Enter重复、R repeat/并发、资料升级及保存保护、重开/刷新草稿补交、错误密码/每次重锁、公告×/Escape/遮罩/Tab焦点循环。公告在视口内，升级表单不盖提交按钮，无横向溢出或JS异常；唯一console error为有意mock的422。
- 终局通过真实Sim超时逻辑确定性推进，对手道具通过真实放置和frameStep复现；不宣称自然完整真人对局统计。结算/资料RPC均 mock，不宣称后端写入E2E。实际远端榜单读取另由线上无mock只读脚本验收。
- 独立只读审查 `round2_readonly_review` 完成，无发布阻断；独立执行3组新增Node测试、复核完整npm日志、浏览器checks与截图。增量审查复现了同tick起始格被过宽豁免的遗漏，已缩小为放置者离开保护并补固定seed回归。审查的香蕉同帧滑行状态建议已采纳；长按/Enter补充真实浏览器输入，资料区和操作面板避免重叠。

## 发布与收尾证据

push main 自动触发 Pages，不需要人工审批。最终发布后验收证据保存到以下本项目忽略目录，避免为记录自身commit不断新增提交：

- `runs/qqt_round2_20261007/pages-final.json`：本地head、远端main、push工作流最终success、线上build-info和18个关键资源逐字节一致（含新增ui_panels.js）。
- `runs/qqt_round2_20261007/browser-pages/checks.json`：同最终源码的线上双视口四项需求复验，全部写入口mock。
- `runs/qqt_round2_20261007/pages-readonly/checks.json`：无网络mock的真实榜单读取、仅脱敏IP、exact-head、无写请求/异常/溢出。
- `runs/qqt_round2_20261007/release_status_20261007.md`：实际工作流、分支同步、两处工作树清洁及本任务预览关闭结果。

执行环境：`PLAYWRIGHT_MODULE=/tmp/qqt-playwright/node_modules/playwright`，`LD_LIBRARY_PATH=/tmp/qqt-libs/lib/usr/lib/x86_64-linux-gnu`。本地预览仅本任务 `PORT=8091 node web/server.js`，验收后关闭，不终止其他任务进程。
