# QQT UI 与结算验收 20261006

## 实现与审查

- 沿用同一 `cx_bot_cooperation_20261004` worktree，未 reset/restore/clean 现有修改。
- 默认角色为 `maomao`，素材 `sprites.json` 和原版导出脚本确认 roleId=9。保存显式选择到 `qqt.character`，已有有效选择优先，存储不可用时游戏继续。
- 所有阵亡玩家在权威 `bunSpawnPos` 显示阵营环、身份与向上取整秒倒计时。真人死亡只去饱和/压暗地图，HUD 和标记不使用滤镜；观战/回放显示阵营编号；复活当 tick 清除。render 边界 save/restore，滤镜不泄漏。
- 空投逻辑仍在3 tick落地，视觉缓存最多动画5 tick；拾取/销毁后取消残余动画。未更改黄金 fixture，原12例逐 tick回归通过。死亡掉落为原逻辑放入道具队列，没有延长生成/拾取/碰撞等待。
- 主榜仅五列：排名、昵称、最佳胜利用时、宣言、脱敏IP；手机隐藏IP列。等级等汇总仅保留独立进度，不混入表格。
- 独立结算画面显示胜/负/平、耗时、服务端名次/样本/超过比例，以及安全再来一局。旧响应显示“排名待数据库升级”，离线显示等待结算，无Top20臆算。
- 比较同结果、模式、地图、对手、难度。每名其他匿名玩家取最佳记录，当前玩家取本局记录；胜局越快越好，失败/平局越久越好。并列 rank=严格更优人数+1；percentile=严格更差其他人数/其他人数×100，单样本0。seed和提交版本仅作审计，测试榜不宣称公平。
- Edge 不读取任何来源 IP 请求头，metadata_recorded=false。service-role仅在Edge平台内使用；私有原校验函数不允许anon/authenticated/service_role直接执行。CORS限制Pages origin并Vary，顶层异常不泄露服务端细节。CORS不能代替全局反滥用限流。

## 验证证据

- 全套 `npm test`（含旧schema历史数据增量升级、RLS、服务角色ACL、胜负平cohort、并列、百分位、独立玩家去重、幂等、XSS、离线/未升级、422不回退、503/提交后断线同payload回退）通过。日志在 `/tmp/qqt-npm-final-20261006.log`。
- 本地 Chromium 桌面1440×1000、手机390×844：mock错误路径与非mock远端旧RPC实际提交/回读通过；证据 `runs/leaderboard_20261006/browser-local/checks.json`。
- 正式 Pages 同两种视口的mock错误路径、非mock实际提交/回读通过；`runs/leaderboard_20261006/browser-pages/checks.json`，无JS异常或横向溢出。结算为全地图806超时分支人为推进，不是自然完整人类对局。
- 定向线上浏览器无网络mock验证：毛毛默认、显式皮皮刷新保存、2v2四玩家死亡标记、真实Canvas filter恢复、复活标记消失、结算再来一局：`runs/leaderboard_20261006/ui-pages/checks.json`。场景由真实Sim权威方法确定性推进，非自然游戏表现统计。
- 首次发布 `bf34d678dbd004f5ff5cc908fefdccaa97345d98` 的Pages workflow [37401316356](https://github.com/allwellll/qqt-RL/actions/runs/37401316356)成功，16关键资源逐字节一致。最终收尾提交exact-head另存 `runs/leaderboard_20261006/pages-final.json`。

## 外部阻塞和执行步骤

远端已执行首次建表，两个增量 SQL 未执行，Edge入口仍404。IP线上未生效且即便部署当前Edge也保持禁用，直到可信平台来源可证明；服务端本局排名与最佳胜利字段待数据库升级。前端Pages发布不受此阻塞影响。

SQL Editor依次执行，不重跑首次建表：

1. `supabase/migrations/20261005173000_leaderboard_ip_and_best_win.sql`
2. `supabase/migrations/20261006090000_match_ranking.sql`

Edge CLI部署见 `supabase/README.md`。函数需要同目录 `index.ts`、`handler.mjs`，平台内置服务角色环境变量，CORS_ALLOWED_ORIGINS=https://allwellll.github.io；所有秘密禁止进入Git或Pages。

本轮真实验收身份未清理，管理端精确执行：

```sql
delete from qqt_private.players where player_id in (
  '6a0231d5-5905-4125-a291-11f36dcb83e1'::uuid,
  'e48b877c-15f8-4b84-ba43-a71162add0dd'::uuid,
  '45ab592f-7202-4b88-9818-585c1f1a8929'::uuid
);
```

历史四个验收身份的清理SQL在 `docs/leaderboard_20261005.md`，同样没有管理执行证据。未使用或输出数据库密码。本轮预览PID608327/608328已停止，浏览器finally关闭，最终状态再次核查。

最终重跑证据位于 `runs/leaderboard_20261006/browser-final-local`、`ui-final-local`，正式Pages最终head验收位于 `browser-final-pages`、`ui-final-pages`。真实验收脚本每次创建的身份精确清理SQL位于对应目录 `cleanup.sql`，最后一轮线上身份请一并执行该文件清理；本任务没有管理删除权限。
