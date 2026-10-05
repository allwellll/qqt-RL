# Supabase 用户排行榜 20261005

## 实现

- 前端新增匿名浏览器身份：随机 `player_id` + 32字节 capability secret，写入 `localStorage`；不使用IP、不上传回放、不依赖匿名Auth。
- 新增昵称和胜利宣言设置，昵称1–24字、宣言最多80字，拒绝控制字符和尖括号；榜单使用 `textContent` 渲染。
- 真人对局结束自动提交一次结算；观战、回放、模型对局不计分。失败时结算留在本地最多20局，7天内可重试，不影响重开和游戏运行。
- 排行榜等级是局外进度：胜利+3，平局/失败+1，每10点升一级。排名为等级降序、达到本级累计游戏时间升序、胜场降序、胜率降序、稳定UUID升序，Top20。
- migration 位于 `supabase/migrations/20261005140000_leaderboard.sql`：`qqt_private` schema、players/match_results/player_progress/level_events、RLS、无基表匿名权限、两个 SECURITY DEFINER RPC、字段限制、幂等 `client_match_id`、凭证校验、并发锁和每身份限流。
- Pages 构建生成当前提交的 `build-info.json`，公开配置只含 URL 与 publishable key。

## 数据库状态

- **DDL 未在远端执行。** 直接数据库地址仅解析到IPv6，本机无IPv6路由；按项目区域猜测的两个新加坡 pooler 候选在5432/6543均连接超时，不能确认属于该项目。
- Supabase REST 可访问Auth settings，但 `qqt_leaderboard`/`qqt_submit_result` 返回 `PGRST202`，`players`/`match_results` 返回 `PGRST205`，证明远端当前没有这些函数和表。管理 API 需要个人 access token；当前会话没有可用管理凭据。未使用、记录或输出数据库密码。
- 证据：`runs/leaderboard_20261005/remote.json`，其中 `initialized:false`、`ddlExecutedByThisTask:false`、`validSettlementE2E:false`。
- 使用 PGlite 实际执行同一 migration 的本地兼容回归通过：匿名RPC调用、重复幂等、错误凭证拒绝、长度/类型/枚举/时间限制、每身份限流、基表 SELECT/INSERT/UPDATE/DELETE 拒绝、RLS、升级事件和多重排序均通过。这不是远端 Supabase E2E。

## 验证

- `npm test`：原有网页全套 + 新排行榜单测和 PGlite SQL 回归全部通过。
- mock 浏览器：桌面1440×1000、手机390×844均通过真实 app 结算钩子、单次提交、刷新身份稳定、离线重试、恶意字符串纯文本显示、输入框不抢游戏键盘、无横向溢出。
- live 未初始化 Supabase：桌面和手机均显示明确“数据库尚未初始化”状态，游戏仍可重开和绘制，无JS错误；没有把404误报成排行榜成功。
- 浏览器证据：`runs/leaderboard_20261005/browser/checks.json`，SHA256 `88e1790e90b34ceaf2f9ffa732c439b6a3ab161dee27ccf5c3a00b6720558a78`。
- npm 日志：`runs/leaderboard_20261005/npm-test-final.log`，SHA256 `3ff5a06f3c875727be8affc289a988a03a25fd9433322b4128bb5002273b3874`。

## 部署后操作

1. 在 Supabase SQL Editor 执行完整 migration，确认四表 `relrowsecurity=true`，并确认匿名读取基表被拒绝、两个RPC存在。
2. 用隔离随机测试身份验证有效提交、相同payload幂等、错误secret拒绝、恶意字段拒绝、排行榜排序，然后按 UUID cascade 清理测试玩家。
3. 再运行 `node scripts/verify_leaderboard_remote.js` 和桌面/手机浏览器脚本；只有这一步完成后才可声称远端数据库 E2E。

客户端排行榜仍是可信度有限的测试榜：拥有本地凭证的客户端可以伪造自己的赛果，随机身份可绕过每身份限流。正式公平榜需要服务器签名赛果、Auth和网关级限流。IP未写入数据库；Supabase平台访问日志的保留由平台配置决定。
