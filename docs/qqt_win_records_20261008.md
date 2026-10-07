# QQ堂逐胜局排行榜：本地实现与人工升级交付

日期：20261008。基线及未发布的 origin/main：`a3d7de9fbbe88ce159705da5f668772e5545dde7`。
合同：`.hermes/qqt_leaderboard_multi_records_current_highlight.md`，含用户对可编辑随机昵称初值的修正。
本项完成本地提交后停止，等待用户人工 SQL；没有 push、Pages 发布、Edge 部署或正式数据库写入。

## 实现与数据边界

- 新私表 `qqt_private.win_records` 按真实胜局保存唯一快照，外部记录句柄为独立随机 UUID；不公开玩家 UUID、match UUID、secret、hash 或 raw IP。
- 原 Round 4 验证器、不可改名、身份 capability、match 归属、收据幂等、空感言保留、结算玩家最佳排名和旧读写 RPC 均沿用。
- 新胜局首次赛果保存感言快照，首份有效资料 receipt 更新该局；重试不创建重复行或重写快照。乱序旧局可更新自己的快照，但不覆盖较新玩家资料。
- 历史真实胜局各一行，感言/IP 为 NULL。历史昵称使用迁移时昵称，无法重建更早昵称史；不复制当前玩家感言/IP 来伪造历史。
- service-role `qqt_record_match_ip(uuid,uuid,text)` 校验 match 归属，复用旧严格 inet 解析/脱敏，只冻结第一份该局 IP。转发头不用于身份认证；没有新版 Edge 或直连兼容 RPC 时逐局 IP 允许为空。
- 排名由服务端按胜利用时、服务端接收时间、随机 record_id 升序计算全局 row_number。仅 Top20 加最多一个本人榜外最新胜局；最新依持久化胜局累计游戏时长、接收时间、私有 match UUID 排序。
- 新公开 `qqt_win_leaderboard()` 不接受 secret。新本人 `qqt_my_win_leaderboard(uuid,text)` 验证 capability 后仅输出安全布尔 mine/latest，未知身份只返回公开榜单且不创建玩家。
- 前端读取版本化刷新，拒绝迟到读取覆盖；严格验证8字段、脱敏 IP、归属标记和 Top20/榜外行边界。五列、我的/最新文本徽标、单一省略行使用 textContent 渲染。
- 新用户默认输入为 `QQT玩家` 加三个无歧义安全随机字符，独立草稿持久化；删除后缀、改名和清空都不补回，保存用户实际输入。昵称注册后沿用不可改名合同。
- loss 无资料区域/请求，win/draw 资料流程、成功提交/X 一次自动新局、失败不重开、仅重试资料与幂等继续有效。

## SQL 文件与发布顺序

完整 SQL（169行），逐字一致：

- `supabase/migrations/20261008090000_win_records.sql`
- `supabase/manual_win_records_20261008.sql`

SQL SHA256：`f006421a4042b3eeba09c6a242a755c12cbd49ef9dd9507d8d5b2ce30fa766ba`。
前置为20261006排名/IP升级及20261007资料receipt升级。SQL是带事务的增量，可重入，不重跑初始schema。
范围为私表/索引、保留旧核心的赛果与资料包装、service逐局IP writer和两个新读取RPC/RLS/ACL。
HTML包含完整升级、前后只读核验、停止条件及Round4函数兼容回退SQL；回退不删除快照，回退期新增胜局重升时按历史NULL补齐。

顺序：本地验证提交 → 用户执行SQL → 真实只读验证新RPC → 经授权部署新版Edge及push前端 → 同head Pages/exact-head及线上双视口验收。
本次没有跨过用户执行SQL的边界，未进行依赖新RPC的线上验收。

## TDD 与实际测试

证据根目录：`runs/qqt_win_records_20261008/`（运行产物已忽略，不入源码提交）。

- RED：`tdd-sql-red.log` 缺少新RPC；`tdd-nickname-red.log` 原身份/IP式默认不符合可编辑三字符；`tdd-rows-red.log` 原客户端不接受逐局envelope；`tdd-edge-red.log` IP仍仅绑定player；`tdd-extra-row-red.log` 客户端接受孤立榜外行。
- `npm run test:win-records`：客户端、Edge、PGlite定向测试通过，日志 `directed-final.log`。覆盖多局/空感言/历史NULL/同昵称不同身份/乱序资料/不可改名/伪造secret或match/真实rank26/Top20内外/并列/重入/幂等/XSS/RLS/ACL/IP隐私。
- 完整 `npm test` exit0，包含新增tests，日志 `npm-final.log`。
- `scripts/verify_win_records_browser.js`：真实 Chromium `149.0.7827.55`，1440x1000、390x844，真实网页/Sim/Canvas/素材，全部外部请求拦截，mock RPC由本地PGlite真实SQL执行。每视口4份mock赛果、5次mock资料请求、30条本地胜局；没有正式数据库写入、未知远端请求或pageerror。
- 新浏览器合同：默认稳定、删后缀/清空刷新不补回、真实保存`QQT玩家`；同身份三局感言不串局、空保留、同昵称不误认；Top20内无省略/重复，榜外展示真实rank30及唯一省略行；刷新/旧响应保护；资料失败只重试资料；loss禁止写、draw不伪造胜局。24字昵称/80字感言/最长脱敏IPv6全部td无溢出。证据 `browser-local/checks.json`、`1440/390-inside.png`、`outside.png`、`longest.png`。
- 既有结算双视口通过：`browser-round4/checks.json`。浏览器测试增加实际mock资料挂起信号等待，消除早于请求到达的断言race；保留成功/X只重启一次、失败不重开、冻结重试、连点/Enter保护。
- 延迟素材加载双视口：`browser-loading/checks.json`，初始化/慢素材无错误loss提示，真实loss与清理正常，零写入。
- Bot双视口：`browser-bot/checks.json`，真实错峰进攻/撤离轨迹及Canvas回归通过，远端写mock。
- HTML：`scripts/verify_win_records_sql_doc.js` / `html-checks.json` 验证嵌入SQL逐字一致，实际PGlite前检→升级→后检→回退→重升；双视口浅色无外部请求/溢出/pageerror。截图 `browser-local/1440/390-sql-doc.png`、`sql-doc-top.png`。
- 独立只读审查 `/root/round4_review` 最终无阻塞，独立定向tests与diff check通过；额外PGlite回退/重升实验验证原快照/句柄/IP不变、回退期新增局历史NULL、重升后新局正常；独立检查最长字段截图及HTML字节/hash一致。
- `git diff --check` 通过。所有测试服务与浏览器由脚本finally关闭，不终止其他任务进程。

## OSS 交付

本地文档：`docs/20261008-qqt-win-records-sql-upgrade.html`，单文件浅色、自包含。
使用 `/mnt/jpfs/afs/wangyaqi/.hermes/skills/oss-static-deploy/SKILL.md` 的上传流程；包装器缺pypinyin，按技能允许的英文规范文件名临时副本+oss.sh回退。

外网URL：
http://jdh-nlp.s3.cn-north-1.jdcloud-oss.com/agent_model/wangyaqi49/20261008-qqt-win-records-sql-upgrade.html

OSS API HEAD/GET均200，Content-Type `text/html`，26203字节，与本地完全一致。
HTML SHA256：`da62062ebba4348a227981989dddbaf6dcfed3c034cc7f3ec61c2712f8192a3b`。
真实回读证据：`oss-readback.json`；上传输出：`oss-upload.log`。
公网HTTP连接从当前机器超时；这是上传器返回的外网URL，API回读已验证，不能声称本机已完成公网HTTP验收。
