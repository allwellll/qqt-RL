# QQT 测试排行榜部署

项目：`ozfdtqtlwrqywwfdxfac`。Pages 与数据库 / Edge 分开部署。

## 当前部署状态（20261006）

本轮只读核验远端排行榜已有 `player_ip` / `best_win_duration_ms` 字段；新的 `qqt_update_profile` 和 Edge 均404。本局排名增量是否已执行没有只读证据；本轮没有 Supabase 管理 token、CLI 或数据库管理连接，**没有执行远端 migration 或部署 Edge**。Pages 发布不代表 SQL / Edge 已上线。旧库结算继续工作；未返回本局排名时显示“排名待数据库升级”；没有本局升级证明时，不显示资料提交区。

## SQL Editor 直接执行

在 Dashboard → SQL Editor （已有 IP / 最佳胜利列的当前项目）粘贴整个
[`manual_upgrade_20261006.sql`](manual_upgrade_20261006.sql)。这是排名增量与本轮新增量按顺序合并的可执行 SQL，**不包含首次建表或已生效的 IP / 最佳胜利增量**。排名增量可安全重复执行，随后新 wrapper 保留原排名逻辑。不要重跑首次建表文件。

如果已经应用部分增量，只执行尚未应用的文件：

1. [`20261005173000_leaderboard_ip_and_best_win.sql`](migrations/20261005173000_leaderboard_ip_and_best_win.sql)
2. [`20261006090000_match_ranking.sql`](migrations/20261006090000_match_ranking.sql)
3. [`20261006220000_upgrade_profile_and_raw_ip.sql`](migrations/20261006220000_upgrade_profile_and_raw_ip.sql)

第三个 migration 可以重复执行，保留旧 validator、队列 payload fingerprint、幂等与比赛排名；增加升级证明、独立资料提交和原始 IP。

- `qqt_submit_result(jsonb)` 的 `match_upgraded` 根据该局 `level_events` 判定，同时返回 `client_match_id`。胜 +3、负/平 +1，10点升级。只在真人完整对局的本局确认升级后显示表单；回放、观战和中止不提交。
- `qqt_update_profile(uuid,text,uuid,text,text)` 验证匿名凭证与属于该玩家的升级事件，立即更新昵称和宣言并返回服务端最终资料同步浏览器缓存；空宣言原样传到 SQL 并保留服务端旧值，后续旧队列不能覆盖已提交资料。不再通过重复赛果保存资料。
- 私表 `qqt_private.players.raw_ip` 为 `inet`，保存最近一次有效来源原始 IP。Edge 按 `x-forwarded-for` 第一项、`cf-connecting-ip`、`x-real-ip` 顺序选首个有效地址。用户已接受可伪造，不把这些字段当作身份认证。
- 新二参数 `qqt_record_player_ip(uuid,text)` 仅授予 service_role，数据库生成 `ip_display`：IPv4 `123.*.*.89`，IPv6 `2001:*:*:42`。旧三参数 writer 停用。公开排行榜和资料/结算 RPC 不返回原始地址；Edge 只返回脱敏值与记录成功标志。

SQL Editor 验证：

```sql
-- 以下在管理端可读原始地址；切勿把查询结果贴到公开日志或 Pages。
select column_name, data_type from information_schema.columns
where table_schema='qqt_private' and table_name='players'
  and column_name in ('raw_ip','profile_saved','ip_display');
select has_function_privilege('anon','public.qqt_record_player_ip(uuid,text)','EXECUTE') as anon_ip_write,
       has_function_privilege('service_role','public.qqt_record_player_ip(uuid,text)','EXECUTE') as edge_ip_write;
-- 期望 false / true。
select * from public.qqt_leaderboard(); -- player_ip 必须为脱敏值；没有 raw_ip / UUID / secret。
set role anon;
select raw_ip from qqt_private.players; -- 期望 permission denied。
reset role;
```

## Edge 部署（SQL 成功后）

Dashboard → Edge Functions 新建 / 更新 `submit-result`，同时上传
[`index.ts`](functions/submit-result/index.ts) 和 [`handler.mjs`](functions/submit-result/handler.mjs)。入口为 `index.ts`。

在项目 Edge secrets 设置 `CORS_ALLOWED_ORIGINS=https://allwellll.github.io`。`SUPABASE_URL`、`SUPABASE_SERVICE_ROLE_KEY` 使用 Supabase 平台内置变量，禁止写进网页、Git、SQL 文件或命令历史。仅有 publishable key 无法代替管理权限。

已登录有管理权限的 Supabase CLI 可从仓库根目录执行：

```bash
supabase link --project-ref ozfdtqtlwrqywwfdxfac
supabase secrets set CORS_ALLOWED_ORIGINS='https://allwellll.github.io'
supabase functions deploy submit-result --project-ref ozfdtqtlwrqywwfdxfac --no-verify-jwt
```

本步骤只部署 Edge；SQL Editor 已执行后不需要 `supabase db push`，避免 CLI 迁移历史不一致重跑旧基线。若要维护 CLI 历史，应按管理端已执行的真实版本分别 `migration repair <version> --status applied`。

Edge 验证：Pages 真人完成一局后，浏览器 Network 中 `submit-result` 应200；`network_metadata_recorded=true` 且 `ip_display` 仅脱敏；管理端确认私表 `raw_ip is not null`，公开 RPC 只有脱敏字段。没有有效请求头或 metadata 写入失败时，成功结算仍返回200、`network_metadata_recorded=false`，不会重复加分。Edge 尚未部署或暂时不可达时，前端同 payload 回退直连结果 RPC，直连不采集 IP。

然后用独立身份分别验证非升级与升级：只有带本局匹配 `client_match_id` 的 `match_upgraded=true` 显示表单；填写资料后 `qqt_update_profile` 返回 `saved=true`，公开榜单立即回读昵称/宣言。

## 排名、权限与验证范围

主榜仍五列：排名、昵称、最佳真实胜利用时、宣言、脱敏 IP。数据库按等级、达到本级累计游戏用时、最佳胜利用时、胜场、胜率排序。

本局排名比较相同结果、模式、地图、对手、难度；每名其他玩家取其最佳记录，本人取当前局。胜局越快越好，负/平越久越好；并列 rank=严格更优人数+1，百分位=严格更差其他玩家/其他玩家数，单样本0。

四张业务表均在私有 schema 中开启 RLS，客户端不能直接访问。SECURITY DEFINER RPC 使用空 search_path、完整限定表名、凭证验证和锁。匿名 secret 只留在浏览器、私有提交请求及服务端校验链路，数据库只存哈希。网页昵称/宣言使用 textContent；长度、控制字符、尖括号校验继续执行。

赛果仍为未签名客户端测试榜，客户端可伪造赛果和转发头；CORS 不代替身份验证。原始 IP 只用于用户已授权的记录，不参与认证或公平性判断。

本地 `npm test` 包括 PGlite 实际执行迁移、升级胜负平边界、幂等、凭证、旧队列保护、原始 inet 保存和脱敏/ACL；不能替代远端部署验证。真实 Chromium 测试：

```bash
node scripts/verify_settlement_overlay_browser.js
node scripts/verify_upgrade_profile_browser.js
```

可通过 `WEB_URL` 指向本地或正式 Pages。这两个脚本明确 mock 数据库 RPC，游戏 Sim / Canvas / WebAudio 及素材真实加载，不新建远端玩家。只读线上数据库状态另存本轮报告。

## 历史 Hermes 测试身份清理

没有管理删除权限时，只提供准确 UUID 清理 SQL；不按昵称/IP模糊删除。本轮专项浏览器未写远端记录。SQL Editor 可直接执行 [`cleanup_test_players_20261006.sql`](cleanup_test_players_20261006.sql)，按历史报告和验收产物中的精确 player_id 汇总；这是独立人工清理步骤，没有执行证据。既有测试身份材料见 [`docs/qqt_ui_release_20261006.md`](../docs/qqt_ui_release_20261006.md) 最新追加章节。
