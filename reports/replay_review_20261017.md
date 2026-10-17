# 低质量对局回放诊断验收（20261017）

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
