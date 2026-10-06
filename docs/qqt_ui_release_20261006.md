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

## 死亡彩色压暗与结算 R 重开追加（20261006）

- 开始时原 worktree 干净，HEAD/main/origin-main 均为 `9d09fae`。未新建 workspace/worktree、reset 或进入 goal。
- 死亡等待复活仅在地图区域覆盖透明黑色，删除 grayscale/brightness 滤镜及灰色层，保留原 RGB 比例；HUD、复活标记与右栏不压暗。alive 恢复当帧取消压暗，无过渡延迟；观战/回放不额外压暗。
- Canvas 外层 save/finally restore 保留，并为地图内部 translate/save 增加 try/finally；新增真实 renderer 抛错/恢复后调用者 filter 不泄漏回归。
- 结算按钮显示“再来一局 · 按 R”。R 与按钮调用相同 guarded reset；忽略输入/textarea/contenteditable、repeat/已按住 R 与 Ctrl/Meta/Alt，合并异步重开期间的重复请求，过期结算按钮不能重开替换后的新局。
- 真实浏览器快速三局暴露上一局刷新排行榜尚未结束时，新局结算会留队的问题。修复 drain 刷新后继续处理新入队赛果；新增延迟 read 下两局各提交一次回归，幂等 ID/payload/数据库逻辑不变。
- 单元覆盖存活/死亡/复活、渲染颜色与 filter、异常 save/restore、键盘R/焦点/repeat/修饰键、异步重开合并/报错恢复、快速两局队列；全套 npm test 与 diff --check 通过。日志 `/tmp/qqt-color-restart-npm-20261006.log`。
- `scripts/verify_color_restart_browser.js` 在桌面1440×1000/手机390×844真实加载完整806地图、资产、QQT.Sim和Canvas渲染。结果提交明确mock，实际不会写数据库；通过真实超时分支确定性推进胜/负/平，不是自然完整真人对局统计，也不是数据库排名E2E。
- 本地证据 `runs/leaderboard_20261006/color-restart-local/checks.json`：每视口3625个彩色地形采样，平均RGB缩放比例差约0.00586、亮度约0.544；HUD像素变化0、复活后与原帧像素变化0、调用者sepia filter恢复、观战压暗值0。死亡、复活、胜/负/平R结算截图已保存；各结果焦点不重开、R重复+按钮连点只有一次重开，上一局不重复提交，无JS错误/横向溢出。
- 审查补修：重开/回放重置递增 sessionRevision，每个异步 Bot/model 决策 await 后检查局面版本与 Sim 身份，过期动作直接取消；地图加载时暂停 tick。浏览器真实延迟猎手 adapter 决策跨越 R 重开，确认新局 tick=0、无旧泡且不提交旧赛果。HUD 像素检查从 BOARD_OFFSET+H*CELL（810px）开始，覆盖完整底部 HUD，两视口变化均为0。
- 独立只读 Agent `color_restart_review_final` 审查完成，无发布阻断；独立重跑 controls、death_overlay、web_feel、leaderboard 定向测试及 diff --check 通过，复核双视口证据。此前 CLI 审查因重连未完成，已停止，不将其算作通过。
- 正式Pages证据发布后生成于 `runs/leaderboard_20261006/color-restart-pages/checks.json`；exact-head和线上关键文件（包括新增核验 controls.js）证据 `runs/leaderboard_20261006/pages-color-restart-final.json`。
- 可信IP记录仍禁用；增量SQL/Edge未部署，不宣称服务端本局排名/IP上线。本轮没有新增数据库写入或清理动作，既有记录由用户另行处理。

## 主任务合同 A/B/C/D 续做（20261006，最新状态以本节为准）

### 基线与产品实现

- 从 `6c554c7456764657a64f9be9db405ac23f06ac49`、原 `cx_bot_cooperation_20261004` 工作区续做；未 reset、新建 worktree 或撤销其他任务修改。Bot 快筛、10%半身位和香蕉墙角实现没有重新开发。`.hermes/` 本地恢复状态原样保留并加入忽略规则。
- A：删除终局独立可见 HTML 文本层及“再来一局”按钮/点击入口。原 Canvas 使用统一原终局字体、颜色、描边与行距绘制结果、运包成功/失败或超时原因、正确比分、游戏内耗时、服务端 rank/total/percentile；无服务端排名诚实显示待数据库升级。只留“按 R 再来一局”，R 仍走原安全 restart gate。HTML 仅保留不可见的屏幕阅读器镜像，标题可见仅一次。
- B 音效：代码中的“释放”是主动使用香蕉皮/慢慢胶，把持有道具放在动作指定格。成功动作的 `itemReleased` 仅增加事件信息，不改变随机数、物理或确定性状态。复用 `assets/snd/吃道具音效.wav` 与原音量/静音路径；同 tick 拾取+释放合并一次，无物品/被阻挡不响。回放从真实 step 重建事件，观战按监听玩家播放，seek 原 silent 路径保留。
- B 资料：右栏完全移除昵称/宣言控件。只在真人完整对局、服务端证明本局升级（`match_upgraded=true` 且匹配 `client_match_id`）后显示结算资料区。首次无已保存昵称时留空并必填；后续沿用浏览器缓存，刷新不会重现旧升级表单。独立资料 RPC 立即保存，空宣言保留旧值，后续可修改；跳过和 R 不受网络保存阻塞。桌面资料区位于结算 Canvas 内下方，手机接在 Canvas 下方的同一 stage，避免遮挡与越界。长度、控制字符、尖括号及 textContent/XSS 校验继续执行。
- C：新 migration 增加私表 `raw_ip inet` 和 `profile_saved`。Edge 接受有效转发头 IP（按用户要求接受可伪造），结算成功后经 service-role-only RPC 保存原始值；SQL 生成脱敏展示。公开榜单、结算和资料 RPC 不返回原始 IP，Edge 使用响应字段白名单；无有效来源或 metadata 写入失败不回滚已成功的结算。IP 不参与身份认证。旧三参数 IP writer 停用。
- 原赛果 validator、凭证验证、payload fingerprint、幂等及 cohort 排名保留。资料 RPC 使用与原赛果相同玩家 advisory lock，已提交资料不会被缓存的旧赛果资料覆盖，不通过重提赛果保存宣言。

### 验证与独立审查

- 完整 `npm test` 通过，日志 `/tmp/qqt-main-npm-20261006.log`。新增 PGlite 测试实际执行下述准确 SQL Editor bundle，并重复新增 migration；覆盖胜/负/平升级边界、非升级禁止提交、凭证/玩家/升级事件归属、重复赛果、宣言空值/覆盖、旧队列保护、IPv4/压缩/映射IPv6 inet与数据库脱敏、私表和 writer ACL。
- 真实 Chromium 1440×1000、390×844：Canvas 单标题、胜/负/平、运包成功/失败、超时比分、排名有无、无重开按钮、输入聚焦/重复/长按 R gate 和无横向溢出通过。最终本地证据 `runs/qqt_main_20261006/terminal-final-local/checks.json`。
- 同双视口验证首次昵称+宣言、缓存、刷新、空宣言保留、修改宣言、尖括号拒绝、双提交合并、跳过/R和右栏无控件；实际加载原 WAV、真实 AudioBufferSourceNode.start，成功释放一次、失败释放不响、静音成功释放不创建音源。观战和真实离线录像不触发表单或提交。最终证据 `runs/qqt_main_20261006/upgrade-final-local/checks.json`。
- 浏览器使用真实网页、Sim、Canvas、地图/角色/音频资源；结果/资料 RPC 明确 mock。终局通过真实超时逻辑确定性推进，运包视觉分支单独设置权威终态；不作为自然完整真人对局或已部署数据库/Edge E2E 证据。本轮没有新建远端玩家。
- 独立只读审查按恢复合同执行；审查结果和正式发布核验在收尾追加。

### 远端实测状态与准确人工材料

旧章节“所有增量均未部署 / IP 主动禁用”已不适用于本轮代码和远端实测状态。本轮只读 `qqt_leaderboard` HTTP200，包含 `player_ip`、`best_win_duration_ms` 列；`qqt_update_profile` HTTP404/PGRST202、Edge `submit-result` OPTIONS404。本局排名增量是否已执行没有只读证据，不宣称其已部署。证据 `runs/qqt_main_20261006/remote-deployment-status.json`。

本机没有可用 Supabase 管理 CLI、管理 token 或数据库连接，**本轮没有部署 SQL / Edge，没有清理历史测试身份**。Pages 发布不等于这些后端已上线。没有本局升级证明时，线上资料区保持隐藏；无 rank 时 Canvas 显示“排名待数据库升级”。

管理端 SQL Editor 可直接粘贴 [`supabase/manual_upgrade_20261006.sql`](../supabase/manual_upgrade_20261006.sql)：只含排名增量与本轮新增量，跳过首次建表和已生效的 IP/最佳胜利增量。排名增量可安全重复执行，新增 wrapper 随后恢复完整排名/升级/资料逻辑。已有最新 migration 的项目不需要重新执行旧增量。新 migration 为 [`20261006220000_upgrade_profile_and_raw_ip.sql`](../supabase/migrations/20261006220000_upgrade_profile_and_raw_ip.sql)。

SQL 成功后，按 [`supabase/README.md`](../supabase/README.md) 更新 Edge 的 `index.ts` / `handler.mjs`，设置精确 CORS origin，使用平台内置 service-role 变量。人工数据库、Edge、匿名公开返回、真实升级与私表 IP 验证步骤均写在该文档；没有把任何数据库/service-role凭据放入仓库、日志或 Pages。

历史 Hermes 测试身份已从既有报告与生成的 cleanup.sql 汇总为7个准确 player_id，管理端独立执行 [`supabase/cleanup_test_players_20261006.sql`](../supabase/cleanup_test_players_20261006.sql)。只按 UUID cascade 清理，不按昵称或 IP 模糊删除；本轮未执行删除。

### 发布核验材料

自动 Pages 由 push main 触发，无额外审批。`scripts/verify_pages_exact_head.js` 要求远端 main、当前 HEAD、push 事件工作流 success、线上 `build-info.json.commit` 相同，并逐字节比较17个关键资源（包括 HTML/CSS/JS、角色素材、原吃道具 WAV）。结果保存 `runs/qqt_main_20261006/pages-final.json`。正式 Pages 双视口证据分别保存 `terminal-pages` 与 `upgrade-pages`；最终收尾核验在发布后完成。

- 独立只读审查 `qqt_final_readonly_review` 发现多标签缓存陈旧时，空宣言被前端替换成缓存旧值，可能覆盖服务端新宣言。已修复为真实空字符串传给 SQL；资料 RPC 回传服务端最终昵称/宣言以同步缓存。新增多标签等效的陈旧缓存回归，准确人工 SQL bundle 同步更新并实际重跑通过。审查复核结果在最终收尾追加。

- 独立只读复审确认空宣言阻断已解决，当前没有发布阻断；实际执行的 manual SQL 与两份源 migration 完全一致。复跑三个定向测试和 diff --check 通过。完整 npm 与最终本地双视口浏览器通过；正式发布核验接续执行。

### 正式 Pages 发布与收尾

- 功能提交 `8ac4d417ead8f35902f29542d71f94479640717e` 已发布。Git 智能推送受代理 CONNECT503 / TLS 中断影响；通过 GitHub Git API 上传必要 blob/tree/commit，逐层核对 SHA 与本地完全一致，非强制更新 main。push 事件自动触发 [Pages workflow 37480663383](https://github.com/allwellll/qqt-RL/actions/runs/37480663383)，最终 success。
- `build-info.json.commit` 与该 head 一致，17个关键资源逐字节相同，SHA256证据 `runs/qqt_main_20261006/pages-feature.json`。本地 main/origin/main 已同步该 head，canonical main 工作树干净。
- 正式网页 1440×1000 / 390×844 Canvas 终局专项通过：`runs/qqt_main_20261006/terminal-pages/checks.json`。升级资料和实际 WebAudio 专项正式页面证据保存到 `upgrade-pages/checks.json`，与本地相同明确 mock 数据库 RPC，不将其宣称为已部署后端 E2E。
- 新增 `scripts/verify_qqt_pages_readonly_browser.js` 在两种视口无网络 mock 读取实际远端榜单，确认五列/仅脱敏IP/右栏无资料控件/无重开按钮/无横向溢出/无JS异常/线上提交SHA正确；冻结游戏 tick 避免结算，验证无结果或资料写入。证据 `runs/qqt_main_20261006/pages-readonly/checks.json`。
- 本任务启动的本地预览服务 PID258922 已正常关闭（exec exit143）；没有终止其他任务进程。SQL、Edge和历史玩家清理仍未执行，准确人工材料见上节。
- 包含本节与只读验收脚本的最后收尾提交继续触发自动 Pages，最终 head/workflow/build-info/资源一致性及全部分支状态保存在 `runs/qqt_main_20261006/pages-final.json` 和 `runs/qqt_main_20261006/release_status_20261006.md`；这两份发布后证据不写入源码，避免为了记录自身SHA不断新增提交。

- 正式 Pages 升级/资料/音效专项已完整通过双视口，`upgrade-pages/checks.json` 确认缓存刷新、空宣言保留、重复提交 gate、跳过/R、观战/真实回放排除、实际 WebAudio source 与静音、无JS异常。加上终局和无 mock 只读验收，三份正式站证据均为完成状态；产品代码不再变化，仅追加收尾报告与只读验收脚本。
