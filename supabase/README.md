# QQT 测试排行榜部署

## 初始化

在项目 `ozfdtqtlwrqywwfdxfac` 的 Supabase Dashboard → SQL Editor 执行完整文件
[`migrations/20261005140000_leaderboard.sql`](migrations/20261005140000_leaderboard.sql)。
这是首次建表 migration，只执行一次；失败会回滚事务。远端执行状态以当天验证报告为准，仓库内有 SQL 不代表远端已建表。

若使用 CLI，先从 Dashboard → Connect 复制该项目真实 session/transaction pooler host，
用户名通常为 `postgres.<project-ref>`，session 端口5432、transaction端口6543，必须验证项目实际配置。
direct host `db.ozfdtqtlwrqywwfdxfac.supabase.co` 当前仅有IPv6。本机无IPv6路由；不能从project ref确定pooler区域。
管理API查询 `GET /v1/projects/<ref>/config/database/pooler` 需要个人access token，publishable key不具备该权限。
仅将临时数据库密码通过交互提示/受保护的环境传给初始化进程；勿写入脚本、连接URL、历史、Git、日志或Pages。

## 浏览器与排名口径

[`../web/leaderboard_config.js`](../web/leaderboard_config.js) 只包含公开项目URL和publishable key。
不使用数据库密码、service-role key或IP身份。不依赖匿名Auth开关。
浏览器生成随机UUID和32字节随机凭证，保存到localStorage；只向结算RPC发送凭证，数据库只保存SHA-256哈希。
排行榜RPC不返回ID、哈希、原始对局、时间戳或网络信息。清除浏览器存储会产生新身份，不提供恢复/跨设备登录。

已有游戏没有角色经验等级；新增的是**排行榜等级**，不改变游戏战斗属性。
真人完成对局胜利+3点，平局或失败+1点，10点升级：`level = 1 + floor(points/10)`。
仅完成对局计时累加；重开/中止、回放、模型观战不计分。两个地图及三种队伍模式共用测试榜，模式/对手/难度留作审计。
排名固定为等级降序、达到本级累计游戏用时升序、胜场降序、胜率降序、内部UUID升序，Top20。
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

浏览器不上传IP。纯PostgREST提供的request headers不能可靠证明公网IP，本迁移不猜测/保存伪造IP，
没有客户端IP字段，也不提供未部署的Edge Function作为既成能力。Supabase基础设施可能保留访问日志。
需要IP风控时另部署可信Edge网关，明确平台保证的来源header、hash secret、保留时限和隐私说明。
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

远端专项：`node scripts/verify_leaderboard_remote.js`，只做只读检查和应拒绝的输入探测，输出不含凭证。
建表后还需使用隔离随机测试身份提交有效payload、重发幂等、验证错误凭证、确认Top与RLS，再由管理端清理。
清理固定测试UUID（从本地测试结果获取）会cascade删除对应战绩、进度和升级事件，不要按IP或昵称误删真实玩家：

```sql
delete from qqt_private.players where player_id = '<test-player-uuid>'::uuid;
```

重新验证排行榜无测试记录。日常只清match_results会破坏统计语义；应按玩家cascade清理，或在明确重置全榜时用SQL Editor事务清空四表。
SQL migration不会自动删除现有业务数据，不包含数据库密码。
