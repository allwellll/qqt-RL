# DCU 收敛复核与最新网页发布 · 20261115

本轮训练已在 u001030 自然结束，270008320 环境步，未重启训练。
用户询问是否还在变强，并要求更新新录像及较好模型。
结论：这三个检查点在固定网页规则下不是稳定收敛，u930 表现较好，u1030 明显退步。
网页纯击杀默认选择 u930，不能用最后一步替代较好检查点。

## 同协议复核结果

JAX BF16 推理、温度 0.3，固定 DUEL_latest/SP_latest/G_frozen，
种子 2026111601 至 2026111608，双席、每局 1800 tick；每点 48 局。
本次源码、对手 SHA、种子与规则相同。u840 额外重复 48 局，动作、状态 SHA
及终局统计 48/48 均与原评测相同。三个新导出均独立重放全部 86400 tick 通过。

| 检查点 | 胜/平/负 | 对 DUEL 胜/平/负 | 击杀/死亡 | 自死次数 |
|---|---|---|---|---|
| u840 | 44/1/3 | 12/1/3 | 377/74 | 12 |
| u930 | 45/1/2 | 15/1/0 | 479/47 | 11 |
| u1030 | 19/18/11 | 9/4/3 | 48/296 | 258 |

选型顺序：完整局总积分（胜 1、平 .5），再看 DUEL 积分、自死次数、净击杀，
完全相同时保留原默认。u930 是本次三个点中的推荐点，不宣称全程最优。
同种子逐局比较 u930 对 u840：4 局积分更好、41 局相同、3 局更差，合计仅多 1 分；
整体胜率优势仍是小样本。u930 对 SP/G 各 15 胜 1 负，u840 原先各 16 胜，
所以不能说对所有对手都变强。u930 自死次数略低，但自死占死亡比例更高
（11/47 对 12/74），不能把比例说成下降。

三点选型复用 8 个种子，不是独立 holdout；SP/G 也参与训练。
详细统计与源码身份：[复核报告](../reports/dcu_convergence_20261115.json)。

## 如何理解训练日志

训练混合对手的击杀/死亡比：u741–840 为 3.64，u931–1030 为 4.28；
同阶段 Critic loss 约 3.060/3.069，熵约 .578/.603。
但当前与历史自对弈对手会刷新，指标不等于固定网页对战能力。
末段训练 K/D 上升、真实网页固定对手表现下降，同时成立；
不能凭 loss 平稳认定收敛，也不能凭训练 K/D 上升认定持续变强。

目前只证实后段网页对战退步，尚未定位是训练优化、对手分布、JAX/JS 规则差异
或其他因素导致。后续训练前应先排查规则/观测/掩码一致性，加入定期固定对手
完整局评测与最佳点保留。不得自动继续 u1030 并默认其更强。

## 新录像与网页模型

网页：<https://allwellll.github.io/qqt-RL/>，纯击杀默认 `dcu-kill-u000930-tau03`。
推荐录像：<https://allwellll.github.io/qqt-RL/dcu-replays/u000930-20261115.html>。
末轮行为记录：<https://allwellll.github.io/qqt-RL/dcu-replays/u001030-20261115.html>。
旧 u840/u330 录像保留。录像是检查点重跑的对战，不是训练现场轨迹。

u930 Actor SHA256：
`1b1079d213a971ed7013739868d3b9848484ecf48ade25418142e16fb3a06d27`。
与此前同样仅删除输入恒零的 mode 行，源 FP32 到 ONNX 最大误差约 1.53e-5，
17 组 JS/ONNX 输入最大误差约 1.86e-5，合法贪心动作一致，部署张量精确相同。
网页同权重 FP32 不与训练 BF16 逐位相同，因此本表不是浏览器 FP32 胜率声明。

JSON/ONNX 在内容寻址 Release：
<https://github.com/allwellll/qqt-RL/releases/tag/dcu-kill-u000930-20261115>。
权重不入 Git，Pages CI 从 Release 下载六模型共十二份资产并校验大小/SHA。
生产 main 仅整合本次网页目录、配置、测试、证据和录像，不合并训练分支的模拟器。

## 接手入口

- `runs/dcu-convergence-20261115/RESULT.json`：三点选择与 u840 精确复测。
- `runs/kill-eval-u000930-20261115-convergence`：u930 48 局、Actor、协议、重放/浏览器回执。
- `runs/kill-eval-u001030-20261115-convergence`：末轮退步的同协议原始记录。
- `runs/kill-eval-u000840-20261115-repeat`：控制组复测，动作/状态/终局完全重复。
- `runs/dcu-web-u000930-20261115-v1`：导出、parity、Release、Pages 与桌面/手机回执。

使用 [既有导出与验收方法](dcu_web_default_20261115.md)，把 source/output 改为本页目录。
`scripts/verify_dcu_kill_default_browser.js` 默认读取当前本地配置中的目标模型，
也可设置 `DCU_EXPECTED_MODEL_ID` 显式指定；线上仍须实际匹配目标 ID 和元数据。
只查看提交/上传回执不代表部署成功，最终应确认 Actions、线上 build-info、文件 SHA
和 PUBLIC_BROWSER.json。

发布 main 提交：`6d2157fa679b526cce5ece6bf3ec01f4cb9a612d`。
此前 u840 发布说明与回执保留为历史；新的生产默认和录像以本页及本轮回执为准。

## 用户叫停时的最终交接状态

用户因机器时限要求“先干到这里”，已停止后续发布，不自动继续。
已上线 main `6d2157fa679b526cce5ece6bf3ec01f4cb9a612d`，
[Actions 43715340219](https://github.com/allwellll/qqt-RL/actions/runs/43715340219) success；
线上 build-info 与 9 份发布网页文件大小/SHA 全部核验通过。
u930 默认模型在线桌面/手机 worker-wasm、切换运包、切回纯击杀与刷新均通过，
u930/u1030 两份新录像也在线通过桌面/手机播放、状态 SHA、跳转和换局检查。
[最终回执](../reports/dcu_latest_publication_20261115.json)。

**未完成事项：生产网页计分归因同步。** 最后核对发现生产模拟器 SHA
`442176b87da4b8a7a3f399efe9fa586dcc67395d9211fdfa4c9120b346cedc09`，
评测/训练分支模拟器 SHA
`89165636ba43b667e68e03bd2985410a2fe2c808c63b11aa692b050468ae2c45`。
差异是纯击杀的延迟被困死亡、接触击杀、同时击杀与最后一 tick 计分归因；
动作/移动物理未修改。旧生产源码在 `tests/test_kill_credit.js` 上确实失败，
本地修正版通过，证据为本轮 `SCORE_REGRESSION.json`。网页与录像的计分口径
因此尚未完全一致；本页前面的胜负表来自评测修正版，不应当作旧生产计分结果。

用户叫停前，`.cache/publish_dcu_kill_credit.py` 已准备但停在 getpass，
现已中断；未输入 token、未创建任何新的远程计分修复提交。
本地 `package.json` 已将既有 `tests/test_kill_credit.js` 加入 npm test，
后续如继续，应审查并只向 main 同步修正版模拟器、回归测试、package 与对应源码指纹，
等待新 Pages success，再重验默认模型。不要整支合并 dcu 到 main。
本次最新修改保存为本地 dcu 提交，尚未推送远程 dcu；后续接手先检查 git log/status。
