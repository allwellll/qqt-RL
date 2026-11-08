# 穿点圆环对准窗口中点（20261108）

在自己复活预览版本 `65655b739cbf3bd07129987b591d8c264aa5a284` 完成同SHA Pages及线上验收后，独立顺序实现。

只修改 `web/visual_renderer.js` 的显示进度分母，从 `PASS_MIN_MS` 改成 `(PASS_MIN_MS + PASS_MAX_MS) / 2`，由现有常量推导550ms。颜色渐变随同一个progress变化；kind/ready、实虚线、触点生命周期和所有物理逻辑均未修改。复用该函数的显示使用同一映射；Review原先隐藏的视角继续隐藏。

## 最小验证

```sh
node web/test_bubble_passage_visual.js
node web/test_pass_facing_range.js
node web/test_pass_facing_rule.js
PLAYWRIGHT_MODULE=<PLAYWRIGHT_MODULE> node scripts/verify_pass_ring_midpoint_browser.js
python3 scripts/audit_combat_training_path.py
git diff --check
```

上述检查通过。真实Chromium加载生产渲染器、原生v4模拟器和实际美术素材，记录Canvas实际圆弧绘制角度及模拟器状态：

|计时|圆环进度/实际绘制比例|ready|
|---|---|---|
|500ms|0.909091|false|
|500.5ms|0.91|true|
|549ms|0.998182|true|
|550ms|1|true|
|575ms|1|true|
|599ms|1|true|

原物理使用严格开区间，精确500ms不授权；此边界已在旧测试中固定，本次没有把它改成闭区间。进入窗口后的500.5ms，圆环只有91%，实际放泡成功、passActive为true，随后真实穿过目标泡，位置从[5.5,4.5]到[5.5,8.5]。550ms满圈后仍保留到600ms前的窗口，600ms过期行为不变。

浏览器为真实Chromium，采用确定性模拟器frameStep推进到指定毫秒并检查真实Canvas画弧，不冒称真机iOS或人工毫秒级按键。为避免额外29MB模型下载，浏览器用例直接加载生产模拟器/渲染器和素材，没有模型评测。截图和阶段收据在忽略目录 `.hermes/pass-ring-midpoint-local.png` / `.hermes/pass-ring-midpoint-local.json`。

隐私扫描覆盖Git追踪内容及本次报告/浏览器脚本，零发现后正常commit/push main；等待同SHA Actions/Pages成功，线上用相同生产模块用例核对版本、画弧与500ms后实际穿泡。最终发布收据存忽略目录，并在消息回报SHA及Actions结果。不 force，不调用CC，不动训练，无资产SHA回读。
