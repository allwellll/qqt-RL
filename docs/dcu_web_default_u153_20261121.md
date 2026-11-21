# u153 网页对拼默认模型发布（2026-11-21）

用户要求把 u153 发布为网页纯击杀模式默认。训练仍在继续；u153 是训练中途检查点，
不是本轮最终模型，也没有与当前 u930 默认做同规则的 48 局对照。

## 网页入口与证据

- 网页入口：[纯击杀模式](https://allwellll.github.io/qqt-RL/)
- 对局录像：[u153 固定对手 12 局](https://allwellll.github.io/qqt-RL/dcu-replays/training-u000153-20261121.html)
- 模型 ID：`dcu-kill-u000153-tau03`
- 纯击杀模式默认切至 u153；u840 和 u930 保留为可选历史模型。
- 检查点：`runs/duel-pool-8card-e2048-mb2048-20261121/u000153`
- Actor SHA256：`c22aefaece05b7073ed2fec90c72e08178d9982dcca45e5e45d641cf935c55fe`
- 固定对局：DUEL、SP、G 各 4 局，双方座位均衡，共 12 胜。该模型在训练池对手上的小样本，
  不能解释为稳定胜率，也不能证明相对 u930 进步。

## 导出验收

网页 Actor 使用原始 FP32 权重，仅投影掉固定为零的 mode 输入行。投影逐张量精确相同；
ONNX 对源 FP32 的最大误差 `2.10e-5`。17 个 JS/ONNX 输入最大误差 `1.04e-5`，
合法贪心动作完全一致。训练时使用 JAX BF16，网页推理使用相同投影权重的 FP32，
两者不逐位相同。

导出、校验与发布收据位于 `runs/dcu-web-u000153-20261121-v1/`：
`EXPORT.json`、`PARITY.json`、`RELEASE.json`、`PAGES_PUBLISH.json`、
`LOCAL_BROWSER.json` 和 `PUBLIC_BROWSER.json`。
模型 JSON/ONNX 只放在带内容哈希的 GitHub Release 资产；仓库只提交目录与模型证据，
Pages CI 下载后按大小和 SHA256 校验。
