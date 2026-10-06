# Supabase 用户排行榜 20261005

## 实现

- 前端新增匿名浏览器身份：随机 `player_id` + 32字节 capability secret，写入 `localStorage`；不把 IP 作为身份、不上传回放、不依赖匿名Auth。增量 migration/Edge 网关部署后，榜单预留脱敏 IP 字段，但在平台未证明转发头可信前保持禁用。
- 新增昵称和胜利宣言设置，昵称1–24字、宣言最多80字，拒绝控制字符和尖括号；榜单使用 `textContent` 渲染。
- 真人对局结束自动提交一次结算；观战、回放、模型对局不计分。失败时结算留在本地最多20局，7天内可重试，不影响重开和游戏运行。
- 排行榜等级是局外进度：胜利+3，平局/失败+1，每10点升一级。排名为等级降序、达到本级累计游戏时间升序、胜场降序、胜率降序、稳定UUID升序，Top20。
- 首次 migration 位于 `supabase/migrations/20261005140000_leaderboard.sql`；已执行项目的增量文件为 `supabase/migrations/20261005173000_leaderboard_ip_and_best_win.sql`，增加最佳胜利耗时排序、可空脱敏 IP 字段与 Edge-only 写入 RPC。
- Pages 构建生成当前提交的 `build-info.json`，公开配置只含 URL 与 publishable key。
- 本局结算画面显示胜/负/平、本局耗时和服务端 `match_rank`；主榜精简为排名、昵称、最佳胜利用时、宣言、脱敏 IP。没有 `match_rank` 的旧 RPC 只显示“排名待数据库升级”。
- 已部署旧 schema 的本局排名增量为 `supabase/migrations/20261006090000_match_ranking.sql`；它按同结果、模式、地图、对手、难度分 cohort，每位其他玩家取最佳记录，并使用严格优于/劣于样本计算并列排名和百分位。

## 数据库状态

- 用户已在 Supabase SQL Editor 执行 migration。当前 REST 复核确认 `qqt_leaderboard` 和 `qqt_submit_result` 返回 200，错误 capability secret 返回 401/`42501`，`players` 与 `match_results` 直接访问返回 404/`PGRST205`。
- 用户提供的初步证据保存在 `runs/leaderboard_20261005/remote-live-e2e.json`；本任务使用独立随机身份再次完成有效提交、相同 payload 幂等、错误 secret 拒绝、排行榜回读和基表访问拒绝，证据在 `runs/leaderboard_20261005/remote-live-e2e-independent.json`。
- `ddlExecutedByThisTask:false` 表示本任务没有数据库管理权限，未声称由本任务执行 DDL；没有使用、记录或输出数据库密码。此前 `remote.json` 的未初始化结论是 migration 执行前的历史探测，不能代表当前远端状态。
- 使用 PGlite 实际执行同一 migration 的本地兼容回归通过：匿名RPC调用、重复幂等、错误凭证拒绝、长度/类型/枚举/时间限制、每身份限流、基表 SELECT/INSERT/UPDATE/DELETE 拒绝、RLS、升级事件和多重排序均通过。这不是远端 Supabase E2E。

## 验证

- `npm test`：原有网页全套 + 新排行榜单测和 PGlite SQL 回归全部通过。
- mock 浏览器：桌面1440×1000、手机390×844均通过真实 app 结算钩子、单次提交、刷新身份稳定、离线重试、恶意字符串纯文本显示、输入框不抢游戏键盘、无横向溢出。
- 正式 Pages 非 mock 浏览器证据：`runs/leaderboard_20261005/browser-live-real/checks.json`，桌面 1440×1000 实际提交一次临时结算并从远端回读，手机 390×844 刷新后回读同一榜单；两种视口均检查 profile、canvas、无横向溢出和无 JS 错误。
- mock 浏览器回归仍保留在 `runs/leaderboard_20261005/browser/checks.json`，用于离线失败、重试、转义和幂等 UI 路径；它不再被当作远端数据库 E2E。
- npm 日志：`runs/leaderboard_20261005/npm-test-final.log`，SHA256 `3ff5a06f3c875727be8affc289a988a03a25fd9433322b4128bb5002273b3874`。
- Pages 部署运行 [37267080842](https://github.com/allwellll/qqt-RL/actions/runs/37267080842) 对提交 `2fb4f5f35099778fdd08a8f9a75ed8d0826585a1` 的 `test-and-build`、`deploy` 均成功；`runs/leaderboard_20261005/pages-implementation.json` 核对了线上 16 个资源（含 `build-info.json` exact-head）逐字节一致。
- 线上浏览器证据 `runs/leaderboard_20261005/browser-online/checks.json`：桌面和手机 mock RPC、身份刷新、离线重试、恶意文字和布局通过；同一页面不加 mock 时对当前未初始化 Supabase显示明确降级状态，游戏仍运行。该文件包含线上 `remoteDatabaseE2E:false`，SHA256 `63f246137b97062a49f22df7fd4190e33dd26e1d4945b4db18b0b61a4c9c9ca0`。
- 最终文档提交的 Pages 精确 SHA 和资源证据另存到 `runs/leaderboard_20261005/pages.json`；文档提交不改变前述已验证游戏资源。

## 部署后操作

1. 若重置测试环境，在 Supabase SQL Editor 执行完整 migration，并确认四表 `relrowsecurity=true`、匿名读取基表被拒绝、两个 RPC 存在。
2. 运行 `node scripts/verify_leaderboard_remote.js` 和桌面/手机浏览器脚本；脚本只输出测试 UUID，不输出 capability secret。
3. 管理端执行脚本证据中的精确 cleanup SQL 后，再次读取排行榜确认临时身份消失。本任务没有管理连接，因此测试记录是否清理必须以人工执行结果为准。

本轮测试均使用昵称 `Hermes远端验收`。管理连接不可用，以下记录**尚未清理**；执行前请核对 UUID，删除玩家会按外键 cascade 删除其战绩、进度和升级事件：

```sql
delete from qqt_private.players where player_id in (
  '1cbd8dd8-1980-4b3f-9892-83164f27d5a1'::uuid,
  '4465444b-09ea-4043-ac85-d159f45ec4fb'::uuid,
  '3c162283-4dc5-4c30-b23c-83079e8d9e27'::uuid,
  'ad141562-447e-4562-9175-bf5b53256d75'::uuid
);
```

客户端排行榜仍是可信度有限的测试榜：拥有本地凭证的客户端可以伪造自己的赛果，随机身份可绕过每身份限流。正式公平榜需要服务器签名赛果、Auth和网关级限流。当前未证明 Edge 转发 IP 头不可由调用者伪造，因此不记录 IP，榜单 IP 显示为“—”；Supabase 平台访问日志的保留由平台配置决定。
