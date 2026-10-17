# 低质量对局回放诊断验收（20261017）

## H8 新批次预声明与平滑根因审计

- 新任务起点 `41b16967f33fc2c6538cc5b717ef6263b0cd67f0`，远端与当前分支一致且干净；保留此前审计。仅只读已完成 H8 v34 g32 的固定 COMMITTED、policy 与生产 Web 导出，不读取 LATEST 或训练中 checkpoint，不使用 GPU。
- 在运行新批次前冻结 24 个 seed：2026101740–2026101751（前 6 原图 Hunter normal，后 6 清砖图 Hunter normal）；2026101760–2026101771（前 6 原图 Hunter hard，后 6 原图 Tactical v2）。上限 2400 tick，真实 greedy 联合动作，禁止修改动作。沿用原先六类失败规则，每类最早 seed 优先，再按 seed 补足最多 6 局；缺少的失败类型如实报告，不追加看结果挑 seed。
- H8 g32 policy SHA-256 `b491758761b32872555555f08cf4e42f9cab120215501509a2bdbf905e64ce11`；CAMPAIGN_COMPLETE completed/actual_generation=32，COMMITTED passed/generation=32；生产导出 ONNX SHA-256 `0c4df18e853a833476afa8e18fce660645ed4075974025e051eae3f2f59972fa`。
- 先线上采样真实生产画布，963 帧，CSS 比例恒定 0.7；双方实际 drawImage 坐标全部整数，旧测试只验证 fraction 变化。前 200 tick 模型移动 185 次中 58 次、对手移动 80 次中 15 次被吸附，含正常贴墙或离开泡泡。无相位重复重置证据。原始坐标证据保留 `runs/replay_review_20261017/motion-baseline/`。
- critic 审计：冻结 worker 保存 policy.pt=host.params.actor，训练 critic 为独立参数树；Web exporter 仅导出该 actor。actor 内 head_wv 不证明是训练 critic。当前生产导出缺少可验证的 trained critic，因此本批 A1 / taken Q / policy mean Q / advantage delta 一律 null，绝不由 reward/logprob 估算。生产 Sim 的随机性均来自 seeded rng，可从起始随机流与 draw count 恢复；reward 另有跨 tick ledger，未经同状态 ledger 分支验证也不可宣称反事实等价。本批不伪造 rollout GAE。
- 新生成器移除重复 trap timer 注入，直接采用生产 `_getLaggedInputs` 的 bun24-trap-timer-v2 编码；历史 H7 文件保持原样，历史动作重推证据仍对应其当时生成器版本。

### H8 与浮点渲染验收

- 原因已从线上实际 drawImage 坐标复现：人物/阴影坐标取整，旧 AABB 门控把相邻实体和脚下泡泡当作整段不可插值；并非只在 tick 刷新或 CSS 比例变化。修复向生产 renderer 传入独立浮点 `visualPos`，不修改 Sim。沿用时间戳累积器与 rAF，前瞻状态只用于安全人物位置；生产 `_tryMove` 校验直线全部中间点，真实碰撞截停、侧向校正、死亡/复活/携包/交付/控制/放泡及变化格仍吸附。真实策略停顿、反向及状态断点仍会可见，不人为平滑动作或事件。
- 旧线上 SHA 的生产坐标断言真实触发 RED（整数精灵丢弃小数位置）；新页面双方均在同一安全 tick 画出 8 个严格单调的中间锚点，包含非整数位置，权威状态不变。桌面 H7 同录像前后对照各采 963 帧，CSS 比例均为 0.7；新 H8 手机 390×844 采 483 帧，比率恒定 0.411111，双方都出现真实中间绘制坐标。相位未在同 tick 重置。`motion-red/`、`motion-baseline/`、`motion-h7-green/`、`motion-h8-mobile/` 保存详细证据。
- 页面显示实际 FPS / 模拟 10 tick/s / 插值 alpha；暂停、seek、单步清零 alpha，保持整数 tick、精确局内时间及 URL 定位。蓝色对手、中文 Reward、授权记忆、标注与深链均保留。关键事件轨按 tick 合并死亡/控制/包子/失误标记，每个显示 R 数值；悬浮与详情列非零中文分量。同 tick 多事件共享该 tick 证据，不声称因果分摊，不重复求和。放泡仍是权威状态事件，沿用此前关键事件轨的筛选范围。
- 优势定义为联合 move×ability 的策略加权一步优势：`Q1=r+gamma*V(next)`，`A1=Q1(taken)-Σ pi_legal Q1`，合法候选策略概率重归一；终局 next value 为零。`gamma=0.995` 来自冻结 H8 plan。metadata 记录计算 schema、checkpoint / policy export / simulator hash、critic=null、合法动作口径及缺测原因。本批 `advantages` 为版本化稀疏量化列式空记录，读取任何 tick 的 Q/A/delta 均为 null；没有导出全 Q 表、logits 或伪造轨迹优势。未来有效记录必须校验 A1 公式、量化误差、相邻 delta、首项/断档 delta=null 及反事实证据 hash。
- 预声明 24 局全部 CPU/JS 真实重跑；按原规则选出 6 局，覆盖 enemy_drop_unpicked / window_closed_empty / return_deviation / self_bomb。未观察到 cage_no_delivery / stagnation；没有为覆盖失败类型改动作。6 局再次真实推理，双方逐 tick 动作、状态 hash、终局完全一致，ONNX/JS logits 首次最大误差 `0.000005449342688645942`。独立 Python CPU 进程复算 6 局的生产 Reward ledger 完全一致；仍明确它是冻结 ledger 在真实 Web 轨迹上的重建，不是训练 episode telemetry。
- 全量重新生成后 catalog 与 6 局文件字节完全一致。旧 6 局文件字节不变，catalog 默认 H8，可切换历史 H7，旧 replay+tick 深链和标注仍绑定原始文件 hash。每局独立懒加载，静态 metadata 一次，最小动作、稀疏 Reward、空优势列和周期 hash，无逐 tick 完整 state 或全动作 Q 表。

| 新 Replay seed / 地图 / 对手 | ticks | raw bytes | gzip bytes |
| --- | ---: | ---: | ---: |
| 2026101760 / 原图 / hard | 2400 | 43881 | 8635 |
| 2026101740 / 原图 / normal | 1783 | 41637 | 9037 |
| 2026101742 / 原图 / normal | 2400 | 53502 | 10135 |
| 2026101741 / 原图 / normal | 2400 | 43788 | 8313 |
| 2026101745 / 原图 / normal | 2094 | 48562 | 9612 |
| 2026101750 / 清砖 / normal | 2344 | 46618 | 9511 |
| 合计 | | 277988 | 55243 |

- catalog（含历史摘要）11358 / gzip 1765 bytes。筛查明细见 `reports/replay_screen_h8_20261017.json`，逐局 provenance/hash 见 catalog 与录像 metadata；尺寸按实际 UTF-8 与 gzip level 9 测量。
- `npm test` 生产全量回归通过；`npm run test:review` 覆盖新旧 12 局共 312 项原协议校验，另含浮点严格单调/断点吸附、19 项 Reward 翻译、A1 手算/策略加权/非法动作/终局/首项与断档增量/稀疏 schema/hash、DAgger 与授权回归。真实 Chromium 桌面与手机通过播放/暂停/FPS/alpha、4x、seek/逐步、事件 Reward、优势缺测、标注持久化/编辑/删除/导入/导出、深链及授权记忆；pageerror / console error / HTTP error 为零。证据在 `browser-h8-local/`，发布后同 SHA 证据另存 `release-h8.json`、`browser-h8-public/` 及 `release_h8_acceptance_20261017.md`，不覆盖既有审计。
- 提交范围 `git ls-files`（含本次已暂存文件）隐私扫描 0 项，`git diff --check` 通过；未修改生产模拟器或训练目录，未调用 GPU。

## 冻结协议

- 独立分支 `feat/je09-replay-review`，初始 HEAD `6e5f69698889187e1ed07e89ce05bd4d0a475128`，初始工作区干净。
- 仅 CPU 离线推理；不读取 H8 checkpoint，不修改训练目录，不停止训练。
- 最新完成 H7 v33 g32，checkpoint SHA-256 `779fa8a20587f2df7e6e34242ce23cfba16183108fe57c4618e22b4b247ccc63`；COMMITTED 与 CAMPAIGN_COMPLETE 均通过。
- 预声明 seeds：2026101700–2026101711，共 12 局。前 6 局原始 806 图，后 6 局明确声明的清砖 806 图；对手均为真实 Hunter normal；最长 2400 ticks，原始动作不修改，跑到模拟器终局。
- 预声明失败规则：真实 enemy drop 后 120 tick 未拾；运输窗口关闭仍在空手运输阶段；持敌包返程距离连续倒退；到己笼仍不交付；自身炸弹死亡；连续 100 tick 未移动。每类优先最早 episode，最多 6 局，其他已运行 episode 摘要保留。
- 本批为定性诊断，不作为能力提升或统计显著性证据；不会人为制造掉包/持包/死亡。
- 首批 12 局仅观察到“窗口关闭仍空手”，其他类别不伪造。补充运行前声明固定 seeds 2026101720–2026101731：前 6 局真实 Hunter hard、后 6 局真实 Tactical v2，均原始 806、相同 H7 g32、上限 2400 tick；相同失败规则与最多 6 局筛选上限。连续返程倒退按连续改变的距离样本计数，等距 tick 不重置方向。
- 高阶设置是静态页面 UI 门控，不是服务端鉴权，无法防御可读源码的攻击者。只保存版本化授权标记与不可逆校验摘要，密码不写入存储；清存储或授权规则升级后重新验证。

## 验证结果

### 真实录像与边界

- 24 局均为冻结 H7 g32 的 CPU greedy policy 与真实 Hunter / Tactical 对手，运行至模拟器终局。原始 episode ID、seed、模型/对手/模拟器/地图/checkpoint SHA 保存在协议与筛查报告中。没有修改动作，未访问训练中的 H8 checkpoint。
- 最终按预声明规则选出 6 局：观察到 `window_closed_empty` 和 `return_deviation`；未观察到 `enemy_drop_unpicked`、`cage_no_delivery`、`self_bomb`、`stagnation`。窗口标签表示对手死亡/控制窗口结束时仍空手，不声称测得模型内部 FSM。胜局也可能有具体失误，不把全部入选局称作败局。
- 6 局重新真实推理，双方逐 tick 动作、周期状态 hash 和终态完全一致；ONNX 与生产 JS forward 的首次 logits 最大误差为 `0.000002533833360462978`。
- Reward 为冻结 H7 原始 ledger 作用于真实 Web 观测轨迹的重建值，沿用 CPU safety detectors 与原生事件，桥接参数 `window=120, scale=2`。19 个 component 每 tick 求和与原函数 reward 一致；新进程复算稀疏 reward 一致。它不是训练 episode telemetry，也不声称训练/Web 轨迹完全相同。教师 eligible/query/accept/reject 未采样，页面明确显示“未测量”。
- 复用生产 `QQT.Sim` 与 `QQTVisual` 画布。协议每 tick 仅双方四个最小动作，reward 仅非零 component delta，附关键事件；每 200 tick 和终态存状态 hash。Seek 的完整状态只在浏览器内存由生产模拟器重建，不写入录像。目录轻量，每局独立懒加载，未打进 JS bundle。

| Replay（seed） | ticks | 原始 bytes | gzip bytes |
| --- | ---: | ---: | ---: |
| native-2026101700 | 1380 | 26254 | 4377 |
| native-2026101705 | 2359 | 41066 | 5317 |
| native-2026101701 | 2400 | 37912 | 5056 |
| native-2026101702 | 1426 | 26898 | 4346 |
| native-2026101704 | 2400 | 42638 | 5449 |
| native-hard-2026101720 | 2400 | 37411 | 4824 |
| 合计 | | 212179 | 29369 |

尺寸使用实际 UTF-8 文件与 gzip level 9 测量。逐局公开 SHA/大小见 `web/review-catalog.json`；全部 24 局筛查与重新推理证据见 `reports/replay_screen_20261017.json`。

### 本地验收

- `npm test` 全量生产回归通过；最终 `npm run test:review`：6 局、156 项 schema RED/GREEN、非法动作/长度/reward 和/hash/seek/标注检查通过；设备授权单元测试通过，HTTP smoke 与 `git diff --check` 通过。
- Chromium 无网络 mock：1440×1050 与 390×844，生产画布采样 1660 种颜色、Lucide 图标 134 个非透明采样像素，移动端无横向溢出、控制重叠；诊断页 pageerror / console error / HTTP error 均为零。
- 播放/暂停时间保持、seek、双向单步、4x、缩放、reward 与曲线、标注持久化/编辑/删除/导出/导入、错误 hash 拒绝、复制定位与 URL hash 深链均通过；精确定位验证为 `tick 123 · 00:12.300`。仅选中录像被请求，切局懒加载通过。
- 高阶设置真实浏览器覆盖：首次提示 → 错误密码无授权 → 正确验证 → 刷新免输 → 新 tab 免输 → 主动忘记后重新提示 → 授权版本与验证规则变化失效 → 新浏览器 context 重新提示。原密码明文从当前源码移除；localStorage 只存版本、验证规则摘要与标记。存储不可用时只保留当前页面内存授权。
- 本地详细证据保存在忽略目录 `runs/replay_review_20261017/`：`inference-verification.json`、`browser/checks.json`、`browser/desktop.png`、`browser/mobile.png`。权重、逐帧原始 trace、浏览器标注和本机配置均不提交。
- 额外运行的旧 `verify_round2_browser.js` 在结算关闭后点击画布重开的断言失败；原始脚本配合原始 `ui_panels.js` 复核也在同一断言失败，属于已有验收缺口。本次设备授权流程已通过独立真实浏览器验证；不将旧脚本声称为全量通过，不扩大修改结算流程。
- 提交前 tracked + 待提交文件隐私扫描为 0 项；当前受追踪内容中原密码明文为 0 处。3 个既有浏览器验证脚本改从 `QQT_SETTINGS_PASSWORD` 环境读取，旧文档移除密码明文并保留原验收语义。未改写历史提交。

公开目标：<https://allwellll.github.io/qqt-RL/review.html>。提交后按实际发布 SHA 等待 Actions/Pages 成功，再校验公开文件字节/hash/尺寸并运行线上真实浏览器；最终发布记录保存在 `runs/replay_review_20261017/release_acceptance_20261017.md`，避免为记录自身 SHA 再次部署。

## 视觉与可读性追加验收

- 基于已发布 `5d2c0f1` 同步分支，保留以上录像、筛查与真实性审计。6 局录像和 catalog 字节不变，未读取新 checkpoint，未使用 GPU 或触碰训练工作区。
- 仍复用生产 `QQTVisual` renderer。核对仓库原生角色资源后，没有可直接使用的蓝色毛毛 skin；通过 renderer 可选 palette 对现有毛毛帧的红色区域缓存换蓝，保留肤色、白色和透明度。未复制外部仓库资产。模型 P0 保持原生毛毛；对手 P1 为蓝色毛毛，画布头顶、标题与侧栏同时标注“模型 / 对手”，不依赖颜色辨认。默认游戏不传 palette / 标签时沿用原行为。
- 时间轴四行常驻“阶段轨 / 关键事件轨 / Reward 轨 / 我的标注轨”，每行有简短含义和悬浮说明，行首独立于横向滚动区，缩放和横向滚动不移走标签。
- 默认目标 60 fps，可选 30 / 60 / 120 fps，显示实际刷新采样与 10 tick/s 逻辑频率。rAF 只在相邻整数 tick 间插值安全人物位置；权威模拟器、动作、reward、事件和物件仍读取当前整数 tick。暂停、seek、单步清零插值。死亡、复活、控制、携包、道具、HP、移动状态、保护阈值、异常位移、终局、碰撞盒接触实体或变化格均保守吸附，防止穿墙或虚构动作；可随时关闭平滑插值。
- 新增显式 19 项 reward 中文名称及释义，覆盖全部 6 局双方 metadata 与非零记录；右栏、曲线选项和时间轴 tooltip 使用中文，英文原名可选或悬浮查看。未知项显示“未翻译（英文键）”并醒目标记，映射覆盖测试失败，不静默回退英文。
- `npm test` 全量生产回归通过；`npm run test:review` 的 156 项协议检查、19 项翻译完整性、palette、连续/不连续状态边界、终态 hash 和设备授权回归通过。真实第一局 2760 个双方转换中，1064 个允许插值、1696 个保守吸附，终态 hash 不变。
- 无网络 mock 的 Chromium 桌面 1440×1050、手机 390×844 验收通过：默认 60 fps、rAF 多帧插值、同 tick 权威状态不变、控制断点不插值、4x 平滑播放与整数 tick 暂停、精确 seek / 双向单步、时间轴固定行首、中文 reward、红色模型及蓝色对手真实画布像素检查。授权记忆/撤销/版本失效、标注持久化/编辑/删除/导出/导入、错误 hash 拒绝、定位复制和深链全部通过；无横向溢出，诊断页 pageerror / console error / HTTP error 均为零。
- 本轮本地截图、帧数和像素证据在 `runs/replay_review_20261017/browser-visual-local/`；同 SHA 发布证据与线上浏览器结果分别保存为 `release-visual.json`、`browser-visual-public/` 和 `release_visual_acceptance_20261017.md`，不覆盖原始审计。显示帧率为目标值，实际刷新取决于设备和浏览器。

## DAgger 追加验收

- 从干净且与远端一致的 `aae0565` 继续；该 SHA 的 Actions/Pages 成功，19 个公开资源字节/hash/尺寸一致。保留原生模型、蓝色对手、60 fps 插值、中文 reward、授权记忆、标注和深链。本次不改变模拟器、renderer、录像或录像 catalog；训练目录仅只读质量 JSON、已完成 COMMITTED 元数据及 smoke 日志，未读取 H8 checkpoint、访问 GPU 或操作训练进程。
- 六局本身没有逐 tick DAgger telemetry。新增本局摘要、右栏及固定中文“DAgger 轨 / 教师与梯度采集”，均明确“本局逐tick DAgger：未记录”，不会把 reward、事件、阶段或 run 汇总转换成采集记录。H7 汇总通过 COMMITTED 的 policy SHA 与六局 checkpoint 精确关联；H8 smoke 和 H6 历史参考明确与本局/本 checkpoint 无关。
- `web/review-dagger/` 独立小 catalog 与三份懒加载汇总，另有按需下载的原始质量快照和白名单数值日志投影。所有 source run/path 为项目相对路径，保留来源 schema、原始字节 SHA-256、来源生成时间与转换时间。H8 funnel 本身没有 schema/生成时间，保留 null；投影标明自身 schema 与原始日志 hash，并使用实际末代日志时间，不伪造原 artifact 时间。

| 来源 | 范围 | 候选 | proposed | strict accepted | Actor 梯度查询 | 接受率 | 最终监督覆盖 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H7 v33 g1–g32 | 同模型 run，formal + smoke 快照 | 2342320 | 695980 | 389036 | 2970 | 55.90% | 0.13% |
| H8 v34 attempt06 g1–g4 | smoke；严格 A 链验收失败 | 196554 | 139554 | 53261 | 296 | 38.17% | 0.15% |
| H6 v28 g1–g32 | 历史参考；accepted 为 query 下界 | 2035270 | 1102974 | 995127 | 995127 | 90.22% | 48.89% |

- H8 无完整 DAGGER_QUALITY，smoke artifact 与四代实际日志逐项核验：stock theft 50、enemy-carried drop 44、enemy-origin pickup 0、enemy-path query 0、own-recovery pickup 2、A 课程 Actor query 26。26 与全 smoke 梯度查询 296 分开显示。原 funnel 的 `passed=true` 与 `strict_declared_acceptance=false` 原样保留；页面以严格条件明确显示 FAILED，缺少 native loose pickup / enemy-origin pickup / enemy-path query。己包回收不能满足敌源要求。`a_loose` 分段沿用来源未区分的聚合，不声称 A 真实敌源追拾成功。
- H6 分段 accepted 为 A 71549 / B 371152 / 返程 540012 / 交付 12414，污染 0；全部从 JSON 重读。H7/H8/H6 的明确定位和拒标细因、教师短反事实存活/进展、专家一致率保留 null，显示“未记录”；合法率仅显示 artifact 或实际日志计数支持的数据。固定语义 fixture 不冒充模型识别精度。
- 来源快照 SHA-256：H6 quality `aab926efa2cb1ca461b88d8123190af6de54fb08e4dc39b46aa4dfd419e164a0`；H7 quality `65d5f16f6bf850aaefd039ce054fb200e3b3cc6d7b9beab36dbbcb94d613b184`；H7 COMMITTED `d283f52d0ce8be4d1b472d62ede1d55008213f43732d9a0844a8c8a6c1f52aeb`；H8 funnel `25875950b5e2efd7b1e05c7bb2fe9ab884ebd3719db96fa45c2ad0ce0dd1a513`；H8 原日志 `2e947f3bbd06012ba18595e03291ef077c9341ef4b3af0428dfd16e22356b76e`。公开投影保留原日志 hash，完整日志不进入网页 bundle。
- DAgger catalog 为 1086 / gzip 557 bytes；H7 汇总 3055 / 1264、H8 汇总 3306 / 1344、H6 汇总 2677 / 1095 bytes。6 局录像仍为 212179 / gzip 29369 bytes，逐局大小/hash不变。
- 回放 v1 增加可选 `dagger_events`：按 tick/player 唯一有序的点观测，仅允许 tick/player、片段枚举、nullable boolean 的 candidate / explicit_localization / teacher_proposed / strict_accepted / actor_query / actor_gradient、拒标原因机器键及最小 `[move, ability]` 动作集合。禁止非法长度、动作、重复、未知字段、不一致接受/梯度标志。缺字段与非记录 tick 均保持未知，不延续上一记录。存在该字段时纳入 replay identity hash；旧录像身份兼容。A 来源未区分与 `A_ENEMY_DROP_RECOVERY` 明确区分，不凭泛 A 标签认定真实敌源掉包。
- `npm test` 全量生产回归通过；`npm run test:review` 包含真实来源快照/hash、转换数值、缺字段 null、own/enemy 分离、不推断、稀疏 schema RED/GREEN 与 DAgger identity 篡改拒绝，另用只读原始目录复核来源字节 hash。原有 156 项协议及 19 项 reward 映射回归通过。
- 无网络 mock 的 Chromium 桌面/手机通过真实质量数值、未记录状态、默认 60 fps / 4x 插值、断点吸附、整数 tick 暂停/seek/单步、蓝色像素与双方文字、授权、标注和深链回归。稀疏轨点击使用隔离的合成协议测试记录，真实录像未注入或持久化测试数据；点击生产轨道按钮确实跳到 tick 321。手机表格横向滚动保留类别列，页面无横向溢出；诊断页 pageerror / console error / HTTP error 为零。
- 新证据保存在忽略目录 `runs/replay_review_20261017/browser-dagger-local/`、`browser-dagger-public/`、`release-dagger.json` 与 `release_dagger_acceptance_20261017.md`。发布必须等待同一最终 SHA 的 Actions/Pages，再验证页面、录像/catalog、DAgger catalog/汇总/来源文件 HTTP 200 与本地字节/hash/尺寸一致，不覆盖前述审计证据。

## 真实 Critic A1 与多场景追加验收

从干净且同步远端的 `6d1e7a7184aed66f8e00e9279b49f899f9ef50bc` 继续，保留前述 H7、H8、视觉、DAgger 和授权审计。全程 CPU，读取已完成 H8 v34 formal g32 的不可变文件与封存代码；没有训练新模型、使用其他代 Critic、读取可变 checkpoint 或操作训练进程。

### Critic 与优势真实性

- 校验 COMMITTED、policy.pt、ema_policy.pt、state.pkl 与 plan hash。选择 raw policy.pt，确认它与 state.live.params.actor replica 0 逐元素一致；EMA 同架构、不同权重，未参与本批次。Critic 从同一 state.live.params.critic 提取，75 个参数叶、8 副本完全一致；没有从 Actor 的未使用 value head 取值。
- 原始 checkpoint policy SHA 为 `b491758761b32872555555f08cf4e42f9cab120215501509a2bdbf905e64ce11`；state SHA 为 `f2fcf9c44401b924776e7e7e990bceebe1660fb1397de30d1476416766716ec7`；Critic 规范参数 SHA 为 `7002b1b4f40d87021462d69a4bd3594964698ce2608592c5c4e17db6d1d5cf85`。公开安全提取证据在 `web/review-evidence/h8-g32-critic.json`，保留 EMA、网络和 plan hash。
- 运行真实训练 Transformer BF16 forward 与 128 bins 的 HL-Gauss value decode（范围 -20 到 20）。输入为生产 encodeObsJAX 的 24×13×15 与 bun24-trap-timer-v2 state24，使用原有固定通道缩放，没有另加 running normalization。3 个固定生产观测在原 checkpoint 与提取模型上的 value 完全相同：5.153899669647217、5.152127742767334、5.185464382171631，最大误差 0。
- 定义严格为 `Q1=r+0.995*V(next)`、`A1=Q1(taken)-sum(pi_legal*Q1)`，terminal next value=0。gamma=0.995 从真实 plan 及其 receipt hash 重读。动作口径为合法 move×ability 联合动作，ability=2 已包含模型道具决策；每个合法联合分支固定同一对手动作、完整生产 Sim 状态、RNG draw 位置与有状态 reward ledger。取实际生产 ONNX FP32 actor 的 masked move/ability softmax 乘积，再对合法联合动作重归一；执行动作为同 checkpoint 的 greedy policy。该比较分布是生产导出策略分布，不声称 ONNX FP32 logits 与训练 BF16 logits 逐位相同。
- tick t 的数值对应 s[t-1]→s[t] 的动作与 Reward。ΔA 使用相邻有效决策之差；首项与缺口后为 null。开局 tick 0 不存在已执行决策，显示未记录；对手没有本 checkpoint 的优势，绝不套用模型数值。新批次 5,806 个决策均测得 A1，缺测 0。
- 所有 tick 的 taken 反事实分支完整 snapshot/info 与原录像一致，全部非零 Reward 分量一致。全量第二次计算的量化列和反事实摘要 hash 完全一致，双方原 reward ledger 又分别整局重算通过。从原始 state.pkl（不用提取 pkl）重载 Critic，以 seed xor 0xA100 的固定 PCG64 抽取每局 8 个随机 tick，另含开局、终局及所需前驱，共复算 112 个 tick 的全部合法分支；Q、均值、A 与 ΔA 最大误差为 4.997314446697487e-07，小于量化半单位 0.0000005。证据为 `reports/replay_a1_verification_20261017.json` 与公开副本。

### 冻结矩阵与文件大小

原生关卡资源审计只确认一张完整可复现的 Bun 底图 806；其他项目的 242 个真实关卡涉及不同玩法，不把它们套成未经验证的 Bun 地图。本批次采用三种真实规则/出生场景，共享底图，页面明确显示规则、qqt_id、map/scene hash 与区分标识：原生抢包；生产 eval 已支持的开放清砖；封存 je05_web_tasks.setup(carry_home) 的清砖、1HP、持敌包返程任务。最后一项初态与原始封存函数的生产 snapshot 完全相同。场景协议及来源代码 hash 单独保留，不伪称三张不同地图。

`reports/replay_a1_matrix_20261017.json` 在生成前冻结 seeds 2026101900–2026101905、场景和真实 Hunter normal 对手。全部预声明局保留，无按结果手选、无修改动作。观察到原生局窗口未切换，开放局两局对手胜，返程两局交付获胜；没有给未观察到的失败类型造标签。原有历史录像继续提供其他已观测失败证据。新批次为默认目录，历史 12 局按需显示，仍按单局懒加载。

| 场景 / seed | ticks | 终局 | raw bytes | gzip bytes |
| --- | ---: | --- | ---: | ---: |
| 原生抢包对战 / 2026101900 | 2400 | 平局 | 124965 | 44278 |
| 原生抢包对战 / 2026101901 | 2400 | 平局 | 131448 | 45577 |
| 开放抢包对战 / 2026101902 | 291 | 对手胜 | 22682 | 8918 |
| 开放抢包对战 / 2026101903 | 512 | 对手胜 | 33414 | 13037 |
| 持包返程任务 / 2026101904 | 100 | 模型胜 | 12688 | 5225 |
| 持包返程任务 / 2026101905 | 103 | 模型胜 | 12639 | 5261 |
| 合计 | 5806 | | 337836 | 122296 |

大小按实际 UTF-8 与 gzip level 9 测量。公开协议只增加量化列式 Q、策略均值、A、ΔA、合法动作数（scale=1e-6）及 provenance；没有逐 tick 完整 state、候选 Q 表或 logits。私有原 trace、Critic 参数和反事实分支只保存在忽略目录。六局再次从真实 ONNX 策略与 Hunter 重新生成，双方动作、状态 hash 和终态完全一致，Reward 和与优势复算通过。

### 页面与本地验收

- 画面后第一块为精确时间、定位、播放/暂停、单步、速度和主进度条，紧接完整七轨时间轴；DAgger 卡片移到时间轴之后。桌面和手机 DOM 与实际坐标均验证无间隔卡片，画布底部与播放区顶部相等。行名常驻，优势密集点在缩放后保持轨道紧凑，数值标签及 tooltip 可点击定位。
- 模型头顶默认显示三行 A、带正负号的“优势增量 ΔA”、模型 P0；对手保留蓝色与“对手 P1”。文字跟随插值浮点位置，边界夹紧并避免双方标签相叠；手机按 Canvas 实际 CSS 尺寸放大，ΔA 实际字号至少 10px。可关闭优势叠加。暂停、seek、单步与深链按整数 tick 同步头顶、右栏与时间轴数值。
- `npm test` 全量生产回归通过；追加 renderer 改动后 visual parity、death overlay、airdrop/death 与 `npm run test:review` 通过。后者含 18 局当前/历史共 462 项协议检查，19 分量中文映射、连续/不连续状态、手算联合优势/非法动作排除/概率归一/terminal/首项与缺口 delta、合法数/来源证据 schema RED/GREEN、三个场景 production Sim/hash、来源函数初态、设备授权与 DAgger 来源真实性回归。
- 无网络 mock 的 Chromium 桌面 1600×1100、手机 390×844：三个场景逐一播放，固定 tick 50 的头顶、右栏与时间轴 Q/A/ΔA 和文件一致，真实 ΔA 随动作变化，优势轨点击定位、开关、深链刷新、授权免输通过。每次 1.4 秒播放约 87 帧；确定性生产 renderer 的优势标签 6 个中间屏幕坐标严格单调，权威状态不改。手机无横向溢出，诊断页 console/pageerror/HTTP 错误为 0。
- 完整浏览器回归另覆盖首次/错误密码无授权/正确/刷新/新 tab/撤销/版本与校验规则变化，蓝色实际像素与中文身份，Reward/事件证据去重，60fps、4x、断点吸附、整数 tick 暂停/seek/单步，标注持久化/编辑/删除/导入导出/错误 hash 拒绝，复制定位及深链，DAgger 未记录与真实历史汇总。证据位于忽略目录 `runs/replay_review_20261017/browser-a1-local/`、`browser-a1-regression-local/`、`motion-a1-local/`；截图包含 desktop/mobile 各场景及画布局部。公开裁切截图为 `web/review-evidence/screenshots/desktop.png` 与 `mobile.png`，保留真实页面像素，不加入标注或模拟内容。

发布前扫描 git ls-files 与全部待提交文件，公开 evidence 不包含机器身份、凭据、绝对路径、内网地址或 PID。提交后按同一 head SHA 等待 Actions/Pages 成功，再核对公开 catalog、录像、DAgger、Critic/A1 evidence HTTP200、字节/hash/尺寸及线上真实浏览器。发布验收记录保存在忽略目录 `release-a1.json`、`browser-a1-public/`、`browser-a1-regression-public/`、`motion-a1-public/` 与 `release_a1_acceptance_20261017.md`，不为记录自身 SHA 追加部署。
