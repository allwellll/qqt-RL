# 手机触控修复交付（20261107）

基线为候选模型正式发布 `cd9cddf0af110f7825a31181e519e08ee1b8921f`；手机改动独立提交，正常快进发布，不改训练、物理或历史 Review。手机发布 SHA：`c2ed1cb26b4c9546c7e8554f10dceae2eb451c74`；已正常快进推送 main，线上结果在验收后回填本报告。

## 根因与实现

原触控按钮使用较小尺寸，缺少局部 Safari callout、选中、contextmenu 和拖拽防护；单一道具键固定发送第一槽。放大原覆盖式控件会遮挡画布。原窄屏分界 820px 也漏掉 844px 手机横屏。

- `web/style.css`、`web/index.html`：控制区置于画布外；仅控制区及按钮禁选、禁 callout/拖拽，局部取消 contextmenu/selectstart/dragstart。按钮最小 56CSS px，宽度充足时 64px；900px 内及 coarse pointer 使用触控布局，横屏画布与控制区并排；配置 safe-area 和 viewport-fit，不禁止页面缩放。
- `web/controls.js`：保留 pointer capture、逐指释放、cancel/lostcapture 清理；同键多指只触发一次初始动作。按住动作键不重复使用道具。
- `web/app.js`：三个快捷按钮显示真实槽位编号、香蕉皮/慢慢胶图标和数量或空槽；直接发送原生槽位 action，经既有权威道具放置路径生成实物。不是先选中后确认。库存超过三槽时有可见换页入口，依次 1–3、4–6、7；换页不重写库存。
- 当前库存物理只有香蕉皮、慢慢胶两种道具。三键对应三个槽位，不发明第三种道具。末页无对应槽的按钮为空；主键盘 Digit1–7 仍为绝对槽号，小键盘 5123 保持独立移动。
- blur、页面隐藏、重开与输入框焦点沿用统一清理；普通说明文字和昵称输入正常编辑选择。

## 验证与证据

`npm test` 完整通过（含 v4、物品、拾取音效、模型加载、排行榜兼容和旧 Review）；布局最终调整后补跑 controls、UI panels、HTTP 与 diff 检查通过。

真实 Chromium 触摸事件验收脚本：`scripts/verify_mobile_controls_browser.js`。覆盖 320×844、360×800、390×844、844×390；每尺寸实际墙钟长按方向超过两秒，无 fake clock。逐帧世界位置证明持续移动、松手停止；无选中文本，受信任右键 contextmenu 被局部取消。每个按钮命中至少 56px，控制区可见、页面无横向溢出且不覆盖画布。

多指验证方向＋放泡、方向＋三个槽位，抬起动作指仍持续移动，放泡只触发一次；槽位 1/2/3 的权威事件、cell、fieldItem、owner 与 action[6] 匹配，只有对应数量减一。三槽夹具为香蕉皮×3、慢慢胶×5、香蕉皮×7，验证的是三槽映射。空槽无事件、无扣减、无假拾取音效。人工七槽边界夹具确认 4–7 可达，第七槽真实放置且其它槽保持；没有宣称正常游戏会生成七种道具。

另覆盖捕获后滑出/滑回、真实 touchCancel、blur 与 visibilitychange 清理、昵称编辑不劫持、小键盘 NumLock 关闭别名移动/keyup停止且不消耗库存、主键盘 Digit2 真实放置胶水。visibilitychange 使用隐藏状态事件夹具，未宣称实体手机切应用。

桌面 1280 与触屏 390 的 v4 回归：半格/整格明显间隙、松键原地计时、零碰撞触发，满圈放泡并真实穿过。新候选与 G_frozen τ0.3 桌面/手机选择、OSS SHA 核验、worker-wasm 推理与 v3/v4 说明均通过；默认保持 G_frozen。

浏览器范围：Linux Chromium 的真实 CDP 触摸、手机 viewport；不是实体 iOS，也没有声称 Safari 系统 callout 或真机 safe-area 实测。已实现 Safari CSS 防护，实体 iOS 系统菜单仍需真机补验。

复现：安装 Playwright Chromium 及运行依赖后，执行 `node scripts/verify_mobile_controls_browser.js`；线上设置 `MOBILE_PUBLIC_URL=https://allwellll.github.io/qqt-RL/`。`PLAYWRIGHT_MODULE` 可指向本地 Playwright 包。

证据目录：`reports/cx_mobile_controls_20261107/`，包含触摸世界状态 JSON、尺寸截图、v4 与模型回归结果。隐私检查限定 tracked 文件与本次交付产物，不扫描本地训练产物；密钥不进入源码、前端、报告或日志。

## 发布验收

正常快进 push main：`c2ed1cb26b4c9546c7e8554f10dceae2eb451c74`。

[同 SHA Actions/Pages](https://github.com/allwellll/qqt-RL/actions/runs/42347826789)：test-and-build、deploy、workflow 均最终 success。Pages 的 build-info.commit 与该 SHA 一致，当前 ruleset 为 bun-native-v4-facing；CSS、HTML、app、controls、模型 manifest/说明、各版物理与旧 Review 引擎均逐字节回读一致。证据：`cx_mobile_controls_20261107/mobile-release-actions.json`、`cx_mobile_controls_20261107/mobile-release-public-readback.json`。

线上四尺寸（320×844、360×800、390×844、844×390）真实触摸复验全部 PASS；实际连续长按分别 2164、2171、2166、2165ms，按钮最小 56–64px。多指放泡及三个槽位、空槽、七槽换页、cancel/失焦/页面隐藏处理、普通文字与输入、主键盘/小键盘隔离均重验通过；截图已查看且无覆盖或横向溢出。线上 v4 桌面/手机半格及整格空隙起计、零接触、满圈放泡并穿过也 PASS。证据：`cx_mobile_controls_20261107/mobile-public-browser.json` 及四尺寸 public-layout 截图、`facing-range-public-browser-results.json`。

本次手机部署后的候选回归也全部 PASS：1280 桌面与 390 手机各选择 DUEL/SP，真实 worker-wasm 推理各至少三次、OSS verified/SHA 通过，原 v3 证据与当前 v4 快筛说明均可见；回切 G_frozen 仍 τ0.3、非 greedy，无页面错误或溢出。证据：`cx_mobile_controls_20261107/specialists-public-browser.json` 及两张模型截图。

远端 main 最终只读核验仍为 `c2ed1cb26b4c9546c7e8554f10dceae2eb451c74`。完整 tracked 隐私扫描与最终交付 JSON/报告的路径、私有标识、内网地址、PID、凭据检查均 0 命中。最终线上结果回填本地报告，不为回填触发额外部署。

## 最新授权与 READY 复核（20261107）

用户最新明确：立即停止训练，撤销全部后继，覆盖此前等待active段完成的安排；未来重新授权前不得续训、启动新训练或后继。训练owner已交付worker归零、八卡释放与COMMITTED保存回执；持久停止闸门由训练 CX 管理；网页交付侧不写训练工作树、不更改或终止其进程，不请求任何新训练。后续选模只允许读取既有 checkpoint/READY 并评估。

本次只读复核 READY 仍为 DUEL u019902 τ0.3、SP u023302 τ0.3，两候选 JSON/ONNX SHA 与已发布 manifest 完全相同，没有新的待消费候选，不重复导出上传或重跑能力评测。证据：`cx_mobile_controls_20261107/latest-web-ready-check.json`。现有长短板标注、默认 G_frozen τ0.3、手机修复均已实际部署，不等待训练或全面胜出。

当前 main 为 `c2ed1cb26b4c9546c7e8554f10dceae2eb451c74`；同 SHA Actions/Pages 最终 success，部署后桌面/手机候选 OSS 校验与真实推理均 PASS。此次复核只回填本地报告，不为无代码变更重复发布。未将旧 READY 中历史性的 training_continues 字段解释为当前续训授权。

发布回执 `.hermes/WEB_CANDIDATE_PUBLISHED.json` 已生成，并按用户明确要求回写训练侧同名交付文件；包含两候选资产身份、OSS、同 SHA Actions/Pages、桌面/手机真实加载与默认保持说明。训练侧仅写该交付回执，未触碰训练源码、停止状态或进程。
