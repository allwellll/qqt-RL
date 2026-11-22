# 网页运包默认模型 · 2026-11-22

网页正常运包模式当前默认加载 `dcu-courier-u000168`，显示名为“DCU 运包强化 · u168（运包默认）”。该模型来自 `runs/carry-joint-u170-u170-20261122e/courier-s01/u000168`，是当天已发布运包对拼录像中的运包方版本。

- 训练更新：u168
- 训练环境步：176,160,768
- 训练 Actor SHA256：`610f94ae33b15de4fcb8393a74590dc985c482e3eecc225da5e782f7ebf50b97`
- 网页 JSON SHA256：`57d59ad04deb81f33f5c9ff2bd9053fd07f5dff338ebb56074f2bbece5ff682f`
- 网页 ONNX SHA256：`09bcfd68ba8fa94429a3bcd4afb8f7caf50fa1bb43dc17bd2def5a9321accd88`
- 导出：固定 `mode=0`，38 维训练状态投影到网页 37 维状态；JS/ONNX 最大绝对误差 `2.31e-5`，合法动作选择一致。
- Release：<https://github.com/allwellll/qqt-RL/releases/tag/dcu-courier-u000168-20261122>

旧的 SP u023302 和 DCU 运包 u587 仍保留在模型菜单中，便于回退和人工复核。默认配置位于 `web/model_inference_defaults.json`，模型目录位于 `web/models.json`。
