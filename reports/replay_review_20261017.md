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
