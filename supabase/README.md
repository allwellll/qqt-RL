# QQT 测试排行榜部署

## 初始化

在项目 `ozfdtqtlwrqywwfdxfac` 的 Supabase Dashboard → SQL Editor 执行完整文件
[`migrations/20261005140000_leaderboard.sql`](migrations/20261005140000_leaderboard.sql)。
这是首次建表 migration，只执行一次；失败会回滚事务。用户已执行该 migration；当前远端复核证据见 [`runs/leaderboard_20261005/remote-live-e2e-independent.json`](../runs/leaderboard_20261005/remote-live-e2e-independent.json)。

若使用 CLI，先从 Dashboard → Connect 复制该项目真实 session/transaction pooler host，
用户名通常为 `postgres.<project-ref>`，session 端口5432、transaction端口6543，必须验证项目实际配置。
direct host `db.ozfdtqtlwrqywwfdxfac.supabase.co` 当前仅有IPv6。本机无IPv6路由；不能从project ref确定pooler区域。
管理API查询 `GET /v1/projects/<ref>/config/database/pooler` 需要个人access token，publishable key不具备该权限。
仅将临时数据库密码通过交互提示/受保护的环境传给初始化进程；勿写入脚本、连接URL、历史、Git、日志或Pages。

## 增量排行榜与 IP 记录

旧初始化 migration 已执行的项目只执行一次 [`migrations/20261005173000_leaderboard_ip_and_best_win.sql`](migrations/20261005173000_leaderboard_ip_and_best_win.sql)，不要重跑首次建表文件。它为玩家增加可空的 `ip_display/ip_hash`，并重建排行榜 RPC：`best_win_duration_ms` 是真实 `result='win'` 对局的最短游戏内用时；无胜局为 `null`，前端显示“—”。主排名口径是等级、达到本级累计游戏用时、最佳胜利耗时、胜场、胜率，最后用内部 UUID 稳定打破平手。

PostgREST 与 Edge Function 收到的转发请求头目前没有已验证的“调用者不可伪造”平台保证，因此匿名 RPC 不接受 IP 字段，结算网关也**不读取或记录 IP**。数据库字段及 service-role-only 写入 RPC 仅为未来接入已证明可信、具备严格标准 IP 解析器的基础设施预留；当前旧记录保持 `null`。

部署（service-role key 只进入 Supabase secrets，不进 GitHub Pages）：

```bash
supabase link --project-ref ozfdtqtlwrqywwfdxfac
# 首次 migration 已通过 SQL Editor 执行时，先标记历史，避免 CLI 重跑建表：
supabase migration repair 20261005140000 --status applied
supabase db push
supabase secrets set CORS_ALLOWED_ORIGINS='https://allwellll.github.io'
supabase functions deploy submit-result --no-verify-jwt
```

也可在 Dashboard → Edge Functions 新建 `submit-result`，粘贴同名目录代码并设置 secret。`CORS_ALLOWED_ORIGINS` 是逗号分隔的精确 origin 白名单；未列出的浏览器来源返回 403，并设置 `Vary: Origin`。当前成功响应始终保留有效结算进度并额外返回 `network_metadata_recorded: false`。函数尚未部署时，Pages 会回退直连 RPC，结算仍可用但不记录 IP。

### 本局排名增量（已部署旧 schema 继续执行）

在完成上述 IP/最佳胜利 migration 后，再执行一次
[`migrations/20261006090000_match_ranking.sql`](migrations/20261006090000_match_ranking.sql)。它把既有结算校验函数保留在 `qqt_private`，公开同名 RPC 先调用原校验/幂等逻辑，再返回 `match_rank`。比较 cohort 固定为相同 `result`、`mode`、`map_id`、`difficulty` 与 `opponent`；每名其他匿名玩家取最佳记录，当前对局取当前耗时。胜局按耗时升序，失败/平局按耗时降序；相同耗时竞争排名并列（`rank = 严格更优样本数 + 1`），`percentile = 严格更差其他玩家数 / 其他玩家总数 × 100`，单样本为0。旧 RPC 响应没有 `match_rank` 时，网页只显示“排名待数据库升级”，不从 Top20 推算。

该 RPC 仍处理未签名客户端赛果，排名只能作为测试榜指标；同一 `client_match_id` 重试不新增样本，排名按请求时数据库样本重算。迁移会短暂锁定提交函数，应在低流量窗口执行。增量文件可重复执行。seed 与提交 SHA 留作审计，但不划分 cohort，因此不同 seed/版本的客户端结果仍可能不可完全公平比较。

## 浏览器与排名口径

Edge 的 `SUPABASE_URL` 和 `SUPABASE_SERVICE_ROLE_KEY` 使用平台内置环境变量；不要将它们写到 Pages 或命令历史。Dashboard 部署必须同时上传 `index.ts` 和 `handler.mjs`。CORS 只是浏览器来源约束，不能阻止伪造 Origin 的脚本请求；当前没有可靠 IP 限流，新匿名身份可绕过每身份限制。公开使用前还需平台全局配额/限流与服务器签名赛果。

当前 IP 收集关闭。未来可信入口启用前，需批准并配置 30 天元数据保留策略，例如管理端每日运行：

```sql
update qqt_private.players set ip_display=null, ip_hash=null, ip_recorded_at=null
where ip_recorded_at < clock_timestamp() - interval '30 days';
```

这不是已部署的定时任务；平台访问日志由 Supabase 项目配置另行管理。

[`../web/leaderboard_config.js`](../web/leaderboard_config.js) 只包含公开项目URL和publishable key。
不使用数据库密码、service-role key或IP身份。不依赖匿名Auth开关。
浏览器生成随机UUID和32字节随机凭证，保存到localStorage；只向结算RPC发送凭证，数据库只保存SHA-256哈希。
排行榜RPC不返回ID、哈希、原始对局、时间戳或网络信息。清除浏览器存储会产生新身份，不提供恢复/跨设备登录。

已有游戏没有角色经验等级；新增的是**排行榜等级**，不改变游戏战斗属性。
真人完成对局胜利+3点，平局或失败+1点，10点升级：`level = 1 + floor(points/10)`。
仅完成对局计时累加；重开/中止、回放、模型观战不计分。两个地图及三种队伍模式共用测试榜，模式/对手/难度留作审计。
主榜排名固定为等级降序、达到本级累计游戏用时升序、最佳胜利用时升序（无胜局最后）、胜场降序、胜率降序、内部UUID升序，Top20。等级统计单独保留，不塞入五列主表；本局 cohort 排名与主榜排名口径不同。
一级达到时间为0；“本次升级”是从上一次升级到本次升级所累加的游戏时间。
客户端总用时也单独留存，但**数据库自行累加每局游戏时间**作为榜单时长，不信任客户端进度或升级时间。
`level_events` 保存跨级时间与对应对局；一次最多+3点，不跨多个等级。

客户端局内tick时间100ms/tick、真实墙钟时间、累积完成对局时间均上报；单局限制1秒至240秒，
真实时间不超过1小时，且不少于游戏时间75%，完成时间不超过7天且不能领先服务端5分钟。
Pages workflow生成 `build-info.json`，结算使用对应提交SHA；本地为 `dev`，build-info读取失败同样回退dev。
网络故障不会阻断游戏；localStorage最多保留20局待提交，通过刷新/重试按钮重发原payload。
超过7天的待提交记录在重试时丢弃；重复 `client_match_id` 仅接受相同玩家和相同payload，重复不加分。

## 权限、隐私与限制

四张基表位于不对PostgREST暴露的 `qqt_private` schema，启用RLS且不提供客户端policy。
撤销PUBLIC/anon/authenticated的schema和基表权限，只授予两个 `SECURITY DEFINER` RPC的execute。
RPC固定空search_path、完整限定表名，验证字段类型、长度、枚举、范围和时间，
串行锁定玩家与对局ID并校验凭证，单身份10秒内最多1次、每天最多60次新结算，重试先做幂等检查。
公开显示昵称/宣言全部通过textContent，输入限制字符长度、尖括号和控制字符。

这是**未由权威服务校验的客户端测试榜**。有凭证的玩家仍能伪造自己的胜负、时间、模式和版本，
新建随机身份能绕过每身份限流；限制只能阻止简单误提交和重复/冒名，不能保证公平排名。
升级包括失败/平局进度，会存在刷局风险；将来正式榜应接入服务器签名赛果、Auth和网关限流。
凭证作为匿名capability，知道玩家UUID不能修改其记录；持有凭证可以提交该身份赛果。

浏览器不上传 IP。纯 PostgREST 与当前 Edge 转发头都不能可靠证明公网 IP，因此当前禁用 IP 记录。Supabase 基础设施可能保留访问日志，榜单字段保持为空。
不要将玩家UUID或hash结果公开到排行榜，勿保存无必要隐私或回放正文。

## 验证与清理

本地：`npm ci && npm test`。SQL测试使用PGlite实际执行migration和anon角色调用，
这是本地Postgres兼容回归，**不能替代Supabase远端REST/RLS验证**。

SQL Editor可检查：

```sql
select relname, relrowsecurity from pg_class
where relnamespace = 'qqt_private'::regnamespace and relkind = 'r';
select * from public.qqt_leaderboard();
set role anon;
select * from qqt_private.players; -- 应拒绝访问
reset role;
```

远端专项：`node scripts/verify_leaderboard_remote.js`，使用一次性随机身份提交有效 payload、重发幂等、验证错误凭证、确认 Top 与 RLS，输出不含凭证。该脚本不能删除测试数据。
浏览器专项：`WEB_URL=https://allwellll.github.io/qqt-RL/ node scripts/verify_leaderboard_browser.js`，需要安装 Playwright Chromium；脚本会在桌面视口提交一次临时记录，并在手机视口回读榜单。
清理固定测试UUID（从本地测试结果获取）会cascade删除对应战绩、进度和升级事件，不要按IP或昵称误删真实玩家：

```sql
delete from qqt_private.players where player_id = '<test-player-uuid>'::uuid;
```

重新验证排行榜无测试记录。日常只清match_results会破坏统计语义；应按玩家cascade清理，或在明确重置全榜时用SQL Editor事务清空四表。
SQL migration不会自动删除现有业务数据，不包含数据库密码。
