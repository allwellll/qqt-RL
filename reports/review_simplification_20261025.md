# Review 录像诊断页精简验收 · 2026-10-25

当前工作区已完成实现与本地验收，未提交、push 或发布。初始工作树干净，基线为 `7128071`。生产链路是 `web/index.html` → `web/review.html` → `web/review.js` / `web/review_protocol.js`；Pages 直接打包 `web/`。旧版 `web/app.js` 录像入口仅只读核对，未修改。

## 页面结果

- 桌面左列只显示录像、原有人物头顶信息、当前 tick/时间、播放/暂停和可拖动时间横轴。录像上方黄色说明、状态/策略/优势/路线详情、多轨诊断、标注及 DAgger 面板均退出观看布局，无空白卡片。
- 右列只显示当前玩家的即时奖励和累计奖励走势，保留必要的玩家切换，曲线游标随当前帧更新，也可点击曲线定位。
- 录像选择改为顶部默认折叠入口，保留模型、场景、seed 搜索与真实录像选择；选中后自动收起。
- 手机改为录像、播放横轴、奖励依次排列，无横向溢出；完整曲线可向下滚动查看。
- 原有诊断 DOM 绑定保留在不参与布局、不可聚焦的隐藏容器中，原始诊断资产、解析、hash 校验、头顶标注以及 A1/ΔA 计算均未改动。缺测仍为“未记录”；不可用录像显示“无数据”，不残留上一局奖励曲线。即时奖励保留“离线复算”来源标识，不冒充历史训练 reward。

代码改动：`web/review.html`、`web/review.css`、`web/review.js`。定向浏览器回归：`scripts/verify_review_minimal_browser.js`。

## 实际验收

使用现有真实资产 `web/review-replays/je10-g8-metrics-recomputed-natural-2026101800.json`，浏览器按目录 SHA-256、地图、模拟器及间隔状态 hash 校验后播放，未修改或替换响应数据。所用 Chromium 版本、资产 SHA、逐帧标签、奖励数值和曲线游标坐标见 [checks.json](review_simplification_20261025/checks.json)。

真实 Chromium 分别以 **1440×1000** 桌面与 **390×844** 手机视口运行，均通过：

- 原生 range 指针拖动、播放后帧推进、暂停后帧冻结、曲线点击定位。
- tick 387 即时奖励 `0.500`，tick 388 即时奖励 `0`，与录像 reward 分量之和一致；曲线游标与当前 tick 精确对应。
- 当前帧人物身份、A 与 ΔA 实际绘制到 canvas，数值与原有解析器逐帧一致；复活期间使用既有倒计时显示。
- 玩家切换后奖励正确切换；录像选择可用，选中后列表收起；刷新保留深链 tick。
- 黄色描述与冗余面板不可见；桌面两列、手机纵向布局均无横向溢出。
- 已被准入规则拒绝的真实历史深链显示不可用，奖励“未记录”、曲线“无数据”；重新选择有效录像后恢复正确数据。
- 页面错误、HTTP 错误、远程请求、非 GET 请求均为 **0**。未访问正式赛果或资料写入接口。

截图已逐张查看：

- [桌面 1440×1000](review_simplification_20261025/desktop.png)
- [手机 390×844](review_simplification_20261025/mobile.png)
- [手机完整页面](review_simplification_20261025/mobile-full.png)

既有检查：`npm run test:review` **exit 0**，包含协议/hash、Reward/Critic/A1、缺测、时轴、头顶展示、人物运动、键盘、DAgger、授权 UI 与 Python diagnostic A1 检查；`node web/test_replay.js`、`node web/test_http.js`、JS 语法检查和 `git diff --check` 均通过。完整 Review 与 Chromium 输出见 [validation.txt](review_simplification_20261025/validation.txt)。浏览器首次启动缺少动态库，已仅在外部临时目录补齐依赖后完成验收；未改项目依赖。

浏览器回归可通过 `PLAYWRIGHT_MODULE`、`CHROMIUM_PATH`、`EVIDENCE_DIR` 指定外部工具及证据位置，脚本自行启动并关闭本地只读静态服务器。页面加载数据、奖励及图像均为真实本地资产；仅为浏览器测试初始化现有设备授权标记，不改变产品授权实现。

## 边界与最终检查

所有既存 run/checkpoint 保留；未启动、停止或变更训练，未改训练策略、奖励函数、原始录像或诊断账本；未写入正式赛果/资料。未出现启动更新菜单，无需执行 Skip until next version。

最终 diff 已审查。隐私检查边界仅为 `git ls-files -z` 的追踪文件与本次待提交文件，不扫描未追踪 run/checkpoint 或本地编排。用户标识、私人绝对路径、内网 IP、机器 PID 检查结果见 [privacy.json](review_simplification_20261025/privacy.json)；本地 `AGENTS.md` 已被根级 `/AGENTS.md` 规则忽略且未被追踪。

本次范围无未完成项；修改与证据留在现有工作区，未 push/main 发布。
