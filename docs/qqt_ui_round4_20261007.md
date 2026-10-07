# QQ堂 Round 4：昵称不可变与每局宣言即时写入（20261007）

基线 `fcf3dd558ac85b9930dafd54cf8e2d4c8be3e074`，工作目录 `/mnt/jpfs/afs/wangyaqi/code_room/cx_bot_cooperation_20261004`。当前 CX 实际调用 shell 完成预检，`hostname -I` 输出 `10.186.23.100 `（末尾空格），`pwd` 为上述目录，分支 `cx_bot_cooperation_20261004`。接续开始时的全部 dirty 前端、SQL、测试草稿，未 reset、checkout 覆盖或丢弃修改。预检未调用子代理；用户随后明确授权独立只读审查代理。

## 最终合同与实现

- 首次玩家显示必填“昵称（首次玩家）”与宣言，一个提交按钮。通过 `qqt_get_profile(uuid,text)` 返回的版本 2 合同确认身份是否注册；未确认时禁止提交，不以本地昵称缓存猜测。首个经凭证验证、完整且有效的赛果初始化昵称与宣言，后续结果和资料请求都不能改名，包括尚未 `profile_saved` 的历史玩家。
- 已注册玩家的昵称输入在 DOM 中隐藏、禁用且不再 required，显示服务端确认的只读昵称。资料请求固定 `p_nickname=null`。任何本人已完成 match 都可更新宣言，升级局与非升级局使用相同合同；空宣言保留服务端旧值。
- 资料意图在首次授权时按 match 固定，并在网络请求前持久化。相同请求幂等；同一 match 更换意图会拒绝。写入期间的后续宣言编辑保留到下一局，不在当前已接受的 match 上再次变更意图。首次已冻结昵称始终以服务端注册结果为准。
- 赛果成功的收据持久保存；资料失败、重新打开旧卡片或刷新恢复后只重试资料，不重复提交赛果。更早授权失败的赛果队列仍保持原始完整 payload 指纹。旧 Round 2 queue-only 存储继续可恢复，旧 Round 3 待编辑昵称不再作为改名请求。
- 赛果/资料成功使旧排行榜读取失效，最新服务端读取才可替换榜单；身份读取也使用版本保护，旧读取不得覆盖刚保存的宣言或注册状态。刷新失败与资料写失败分离；写成功后只读刷新失败可单独重试。
- 资料收据采用冻结的累计游戏时长、接收时刻与 match UUID 的次序判定较新局，防止离线旧局晚到及旧收据重试回滚新宣言。此顺序只保护本人资料，并不把客户端时长提升为可信赛果证明。

## SQL 与安全边界

新增 `supabase/migrations/20261007120000_match_profile_updates.sql` 和完全相同的 `supabase/manual_profile_update_20261007.sql`，只要求已部署的 20261006 ranked/IP 升级，不重建基线 schema，不替换原 IP writer。整个增量为一个事务，可重复执行。新增私有 `profile_receipts`；全部五张私表开启 RLS，匿名/authenticated/service_role 无直接收据权限。公开资料读写仅授予 anon/authenticated，赛果沿用 Edge/service_role 合同。

保留匿名 secret 哈希校验、player/match 归属、advisory lock、原赛果 payload 指纹、长度/控制字符/尖括号限制与排名逻辑。网页资料使用 textContent。原始 IP 仍在私表，二参数 writer 仍仅 service_role 可执行，公开 RPC 只含脱敏 IP；新资料 RPC 不返回原始 IP、UUID 或 secret。

新前端要求版本 2 资料合同；未升级、缺失或不兼容的资料 API 不会被当作更新成功。必须先升级数据库并确认 PostgREST 新 RPC，再安排 push 与 Pages 验收。

## 实际验证

- 完整 `npm test` 最终 exit 0，日志 `runs/qqt_round4_20261007/npm-test-final.log`。默认套件保留游戏、Bot、回放、Edge、旧 SQL 回归，并纳入本轮 SQL 与资料边界测试。
- 定向 `web/test_settlement_card.js`、`web/test_upgrade_profile.js`、`web/test_settlement_profile_round4.js` 均通过：首次/老玩家、旧缓存不能确认身份、旧 RPC 版本拦截、首局注册昵称权威性、普通局更新宣言、空值保留、快速连点、冻结意图、提交中编辑、失败后刷新恢复仅重试资料、旧卡片收据和旧存储迁移、迟到榜单/身份读取保护。
- `scripts/test_profile_update_round4_sql.js` 用真实 PGlite 执行初始基线、20261006 升级、本轮 manual 与 migration（重复执行），并实际执行 HTML 中的前置和升级后只读 SQL。覆盖升级/非升级、首次注册、历史未 profile_saved 昵称保护、伪造 match/他人 match/错误 secret、相同请求与冲突意图、空宣言、旧/迟到局收据、XSS、RLS/ACL、原始 IP 保存与公开脱敏。manual/migration/HTML 嵌入 SQL 逐字一致。日志 `runs/qqt_round4_20261007/sql-test.log`。这些均为本地数据库测试，未连接或写入正式 Supabase。
- `scripts/verify_round4_browser.js` 实际运行 Chromium `149.0.7827.55`，1440×1000 与 390×844。真实 Sim/Canvas/素材，由真实 Sim 超时生成确定性完整赛果。context 默认拒绝所有远端请求，仅允许本地素材；所有 Edge/RPC 请求均 mock，无未 mock 的远端请求。
- 浏览器覆盖首次昵称必填、老玩家只读昵称与隐藏禁用输入、每局宣言立即回读、刷新页面后服务端 mock 仍返回新资料、长按/连点/Enter 幂等、关闭/重开/草稿恢复、资料失败后重载只重试资料、冻结资料重试、空宣言、升级局更新、迟到榜单读取与刷新失败恢复。两视口 pageerror 均为空；console 仅故意 mock 的 422/503，无非预期错误。卡片不遮挡榜单刷新，无横向溢出。实际截图人工查看通过。
- 浏览器证据 `runs/qqt_round4_20261007/browser-local/checks.json`；截图 `1440-first-card.png`、`1440-first-success.png`、`1440-returning-card.png`、`1440-final-returning-card.png`，以及同名 `390-` 版本。浏览器日志 `runs/qqt_round4_20261007/browser.log`。
- 离线浅色 HTML 实际用 Chromium 打开，两视口无外部请求、无脚本错误、无横向溢出，四个代码块与 SQL 文件核对通过。修复了手机上长 SQL 文件路径导致的横向溢出。证据 `runs/qqt_round4_20261007/html-checks.json` 与 `browser-local/1440-sql-doc.png`、`390-sql-doc.png`。
- 独立只读审查：`round4_review` 实际审阅合同、完整状态机、旧/新 SQL、测试、HTML 和浏览器证据，并独立执行排行榜与定向测试；最终审查通过，未发现需修复问题或提交阻塞；审查记录 `runs/qqt_round4_20261007/independent_review_20261007.md`。未允许其改文件、正式 SQL、push 或发布。
- `git diff --check` 已通过；提交前再次执行并确认。

## 人工升级与后续发布顺序

单文件浅色离线人工文档：`docs/qqt_round4_sql_upgrade_20261007.html`。包含前置核对、完整可复制升级 SQL、ACL/RLS/资料合同只读检查、新 PostgREST RPC 示例及明确发布顺序。

**本地验证提交 → 用户人工执行 SQL → 验证新 RPC 返回版本 2 → 再 push 与 Pages exact-head 验收。** 本轮仅做到干净本地提交，没有正式执行 SQL，没有 push，没有发布。最终提交 SHA 与干净树结果记入忽略目录 `runs/qqt_round4_20261007/local_status_20261007.md`，避免报告引用自身 SHA 的递归提交。

## 同日追加：失败局提示、胜利感言与自动新局

以下为用户随后覆盖的最新交互与发布要求；上文是首次本地交付的历史记录。继续基线 `0884d901bacde4d4cf760e603173ea714386ea51` 的同一 worktree，保留中断前全部有效 dirty 修改。接管时前次自有测试/浏览器均已结束，未并发重复启动，也未终止其他任务进程。

- 玩家看到的资料名称统一为“胜利感言”，包括表单、占位文本、排行榜列和资料状态/校验文案。没有更改 `victory_message`、`p_victory_message`、存储键或既有玩家资料，也没有 SQL/Edge 修改。
- `loss` 隐藏整个资料表单和只读昵称，禁用输入与提交；原结算区域只显示“小伙子，再沉淀沉淀吧”（无首尾空格）。核心方法同时拒绝失败局资料请求，刷新恢复与程序触发 submit 也不产生 `qqt_update_profile`。去掉失败局“提交后查看排名”的无效邀请；原游戏胜负、时间及物理逻辑保留。没有新增失败局自动赛果写入。
- 胜利和平局的资料提交流程保留。完整提交成功后自动开新局；结果失败、资料失败或请求尚未完成时不重开。资料已成功但榜单刷新失败时，依然开新局，榜单可单独重试读取。
- X 关闭结算卡后自动开新局，对 win/draw/loss 均有效；关闭不授权或提交资料/赛果，原未完成草稿仍持久保存。自动新局与 R 共用 `requestRestart`/restart gate；结算动作在提交与异步新局期间互斥。重复 submit、X、Enter 只产生一次提交和一次新 Sim，清理后的旧隐藏控件不重开新局。
- 刷新恢复的旧结算卡可能覆盖在运行中的 Sim 上，点击 X 或成功提交仍须开新局，因此结算回调使用普通重启保护，而非只允许已结束 Sim 的限制。

TDD 实际证据位于 `runs/qqt_round4_loss_20261007/`：

- `tdd-red.log` 与 `tdd-loss-baseline-red.log`：新 loss 测试在原 Round 4 实现实际 RED；后者仅在内存加载 Git 基线，未覆盖 worktree 文件。
- `tdd-copy-red.log`、`tdd-submit-red.log`、`tdd-close-red.log`：实现前分别确认旧文案、成功未重开、X 未重开断言失败。
- `tdd-loss-caption-red.log`：失败局仍展示无效提交邀请的断言 RED。
- `tdd-loss-layout-red.log`：真实 Chromium 发现手机紧凑 loss 卡片的 44px X 超出底边，补按钮边界断言确认 RED；手机卡片增加 56px 最小高度后通过。同样断言纳入本地和线上最终双视口验收。
- `tdd-green.log`、`tdd-restart-green.log`：最小实现后 GREEN。新 `web/test_loss_settlement.js` 和 `web/test_settlement_restart.js` 纳入完整 npm test，覆盖首次/已有昵称、loss 资料守卫、刷新恢复、win/draw、资料失败仅重试资料、待完成与重复动作、成功/X 后新局和旧事件保护。
- `npm-test.log`：完整 npm test；本地 SQL/PGlite、身份/幂等/RLS/IP 隐私回归继续执行。
- `browser-local/checks.json` 与截图：真实 Chromium 149，1440×1000 与 390×844，真实 Sim/Canvas/素材。用包子库存触发真实超时胜负，按实际新 Sim 实例数确认每次动作只新建一局。全部远端请求默认拒绝或 mock，无正式写入。覆盖 loss 首次/老玩家/重载、win/draw 提交、X 新局、提交失败与待完成不重开、连点/长按/Enter、资料重试和榜单竞态；截图人工检查无溢出或遮挡。console 仅故意 mock 的 422/503，无非预期错误。
- 独立只读审查 `round4_review` 通过：实际执行两个新定向测试及 `git diff --check`，检查结算回调时序、刷新恢复、重复动作、玩家文案/字段边界、双视口证据及截图，未发现提交阻塞；没有改文件或执行数据库写入。审查记录 `independent_review_20261007.md`。

发布前用随机未注册身份只读探测正式 `qqt_get_profile`，实际 HTTP 200、`profile_contract_version=2`、`registered=false`，不创建玩家/赛果/资料。证据 `profile-contract-readonly.txt`。用户本次明确授权本地提交后 push main 与 Pages 验收；正式数据库写入仍被禁止。

最终提交 SHA、push、同一 head Pages 工作流/build-info/18 个关键资源逐字节检查、线上双视口只读验收与干净树事实统一记录在 `runs/qqt_round4_loss_20261007/release_status_20261007.md`；发布验收证据分别为 `pages-final.json`、`browser-pages/checks.json`、`pages-readonly/checks.json`。线上功能验收继续全部 mock 远端请求；只读验收放行真实资料/榜单读取且在网络层拦截所有写入口，冻结游戏 tick，禁止污染排行榜。
