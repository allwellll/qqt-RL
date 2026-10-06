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

## 右侧菜单精简追加验收（20261006）

- 继续原 workspace/worktree，从 `958a00e` 修改；未新建、reset 或进入 goal。
- 右栏首块为排行榜，主表仍仅排名、昵称、最佳胜利用时、宣言、脱敏 IP。桌面和手机均显示五列，资料设置折叠；删除等级/身份/加分/算法/Edge/权限等长说明和结算 cohort 长文，保留状态、错误、数据库升级提示。实际排序与数据库 RPC 不变，后台边界仍见本文件前文和 Supabase 部署文档。
- 新身份昵称为 `QQT玩家·本地abc…def`：UUID 首尾各三个十六进制字符，显式“本地”，不呈现为 IP。旧有效昵称（包括旧默认“QQT玩家”）全部保留，因为无法区分旧默认和主动同名。显式保存后 `nickname_auto=false`，不被后续元数据覆盖。
- 只有有效结算响应明确给出 `network_metadata_recorded=true` 且 `ip_display` 是合法已脱敏 IPv4/IPv6 才可替换自动昵称；不采用原始 IP、浏览器自报 IP、其他排行榜行或无证明的值。当前 Edge 始终返回 false，因此线上仍使用本地短标识；没有修改 SQL/Edge 或可信 IP 边界。
- 按钮改为暗色低饱和、较小内边距；手机点击高度至少44px，保留焦点指示。资料表单、错误和重开仍正常。
- 新增自动化覆盖稳定本地昵称、旧昵称保留、禁用/畸形/原始 IP 拒绝、明确可信脱敏元数据、显式昵称优先、顶部五列和长文移除。全套 `npm test` 通过，日志 `/tmp/qqt-menu-npm-20261006.log`；独立只读审查无阻断，`git diff --check` 通过。
- 新增 `scripts/verify_menu_browser.js`。本地 Chromium 1440×1000 / 390×844 真实加载页面和远端榜单（无网络mock），默认本地标识、保存刷新后自定义昵称、五列可见、紧凑按钮、无页面/表格横向溢出、无JS错误通过。结算截图为确定性 UI 状态验证，未制造服务端排名或提交赛果；另一个明确mock场景验证长文本/XSS五列布局。证据 `runs/leaderboard_20261006/menu-local/checks.json` 和对应截图。
- 正式 Pages 同视口重复验收证据 `runs/leaderboard_20261006/menu-pages/checks.json`；exact-head workflow与关键资源证据 `runs/leaderboard_20261006/pages-menu-final.json`（以发布后生成的实际结果为准）。
- 本轮浏览器只读远端数据库，不新建远端测试记录、不添加或执行任何删除/清理动作。已有测试身份由用户另行处理。Supabase增量SQL与Edge仍未部署，服务端本局排名/IP不宣称上线。
- 截图脚本追加等待加载遮罩 `opacity=0`，避免把淡出中的加载层误当最终页面。全套测试再次通过，正式Pages与本地截图重取；没有改动游戏运行或网络提交逻辑。
