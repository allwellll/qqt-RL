# 面朝首泡距离修复（20261107）

紧急修复独立版本 `bun-native-v4-facing`，默认使用，保留 v3 更多道具、全图死亡掉落、OSS 模型、小键盘和拾取音效；排行榜继续暂缓，不提交新成绩，不迁移或删除旧数据。训练运行源码未改。

## 根因与之前误验收

线上基线 `e04646ef74f44dccc3380e91fba8bbe49a66d9a3` 实际加载链为 index → v2/capture → v2-face/capture → v3 → ruleset_selector 默认 v3。只读取 v2 的 `_nativeTouch` 无法代表实际运行引擎。但 v3 继承的 face 判定确有问题：仅检查邻格，并把 19px 身体前缘向前探测 3px，仍要求极小接触间隙。旧夹具人物和泡相邻中心，身体距泡格边界约 1px，误把它通过当作远距离不用挤的证据。此前验收遗漏了明显空隙用例。朝向保留、idle frame 推进本身已有实现，并非仅规则名错误或 v3 漏加载。

## 精确定义

- 人物主行/主列取 `floor(position)`，按保留的最后方向扫描前方格；不要求当前有输入。目标为中心轴向距离最多 3 格的第一个泡，排除同格、斜角和背后。
- 墙、砖、箱体 pushable、cover 遮挡，不能隔墙激活。首泡遮住后泡，不无限远选择。判定不调用 segment collision、身体前缘或接触距离。
- 面朝观察通过原生 `touchCell/touchStart/touchLast` 生命周期保存；空闲帧持续观察。转向/离开不重置原泡时钟，换目标转移，原生过期后圆环消失。
- 原生开放窗口仍为严格 `500 < charged < 600ms`、最近观察 `<100ms`。真实放泡成功时必须仍面朝同一目标。圆环和放泡授权共用 `_nativePassReady`，不只画圈。
- 放泡所得权限绑定 `passCell`，只允许穿该目标，保留远侧合法性判断。期限为 `ceil(max(1,轴向中心距离格数)*40*1000/原生像素速度)` ms，覆盖有限接近时间；目标消失或超时失效，不授予其他泡权限。
- `ITEM_RULE_CONFIG.version` 仍是 `bun-native-v3-items`，供给、掉落和上限未变。新玩法/Review 使用 v4 独立引擎 SHA；旧 v3/v2-face/v2 和更早归档物理字节不改。

引擎：`web/sim-bun-native-v4-facing.js`；SHA256：`5b4f93d0f994cbabc9ef2012e6d9b4062c031f5b801a7b3847778950af372fad`。

## 本地验收

`npm test` 全套成功，包括旧 Review、控制、模型、道具、音效与新 `web/test_pass_facing_range.js`。新测试覆盖四方向、半/整/两格空隙、3格边界、最近/墙砖箱遮挡/背后/斜角、严格时间窗、实际穿越、target-only 权限、模型时钟、snapshot、旧v3反例和物品函数/seed布局一致性。空闲充能时将两类 collision 函数替换为抛错探针仍通过。

真实 Chromium：桌面 1280 键盘、手机 390 触摸，分别以人物 x=4.85、4.35，泡中心 x=6.5 起始，先真实转向32ms再松键。释放后身体与泡边界净空分别 **0.575、1.075格**；31个16ms空闲帧位置完全不变、movement probe 不增长、起计至放泡前碰撞命中0；512ms 实圈 ready 与物理 ready 同为真。真实 Space/触摸放泡得到 cell81 权限，再真实方向输入逐帧经过泡格，最终 x=8.675/8.175、权限已到期。未直接调用授权函数。测试清理 owner/fuse 配对，避免旧夹具遗留泡变成爆炸，截图可见完整空隙。

- [逐帧位置/朝向/touch/真实圆环/穿越证据](cx_pass_facing_range_20261107/facing-range-local-browser-results.json)
- [桌面整格空隙增长截图](cx_pass_facing_range_20261107/facing-range-local-1280-whole-cell-growing.png)
- [手机整格空隙满圈截图](cx_pass_facing_range_20261107/facing-range-local-390-whole-cell-ready.png)
- [物品/拾取声音/默认模型回归](cx_pass_facing_range_20261107/facing-preservation-audio-local-browser.json)：正常图24/对拼15；上限10/8/2.4；真实增强；全图死亡掉落守恒；人类和模型拾取均仅一次实际播放，满库存不响，静音有效，AudioContext running，4音频200；默认 G_frozen τ0.3 在正常/对拼均运行。
- [旧Review真实播放](cx_pass_facing_range_20261107/face-old-review-local.json)：8条录像逐一校验、seek终点与实际播放通过，仍用冻结 SHA `f6e346f3022640556efc2a988eb3b9b8e90abd5776af6ca1b4b1b0ee2acd6abc`。

浏览器复现：安装 Playwright 与 Chromium 后，运行 `node scripts/verify_facing_range_browser.js`。`PLAYWRIGHT_MODULE` 可指定本机模块位置，`PASS_PUBLIC_URL` 可指定线上地址；输出到本地 `.hermes/`，浏览器仅观察物理和渲染，Supabase 请求被拦截避免写入合成成绩。

没有重跑既有1280局，也不将其当作新距离规则的模型性能证据；本次紧急变更按用户明确授权修正玩法。

## 发布核验

待正常 push 后，回填同 SHA Actions/Pages 和线上桌面/手机结果。
