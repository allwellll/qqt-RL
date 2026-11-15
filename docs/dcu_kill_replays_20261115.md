# 新模型对弈录像（2026-11-15）

观看入口：<https://allwellll.github.io/qqt-RL/dcu-replays/>。
本次固定录像：<https://allwellll.github.io/qqt-RL/dcu-replays/u000330-20261115.html>。

按用户要求只展示新模型，不做训练前后对照。页面可选对局、暂停、逐 tick 前后移动、
拖动进度、0.25–8 倍速，以及跳到下一次击杀/死亡。手机也可查看。
HTML 内嵌游戏渲染素材、冻结的模拟器和动作，下载后可离线打开；不需要下载权重或访问推理服务器。

## 本次录像与效果口径

从 `runs/kill-jax-8card-e512-20261115/u000330` 冻结 Actor，
对应本轮已训练 86,507,520 环境步；导出时校验了 Actor 和训练源码 SHA256，
候选权重另存于导出目录，训练检查点后续滚动淘汰不会改变这批录像。

在当前网页 v4 引擎的空笼击杀模式下，使用固定 DUEL/SP/G 为对手，
种子 `2026111501,2026111502`，双方座位、温度 .3，每局完整 1800 tick。
这是用检查点重新对弈的录像，训练现场的 JAX 轨迹此前没有录制，不能事后还原为现场录像。
每对手 4 局只用于快速了解行为，不能据此断言稳定胜率或相对过去的进步。

| 固定对手 | 胜 / 平 / 负 | 击杀 / 死亡 | 自死 |
|---|---:|---:|---:|
| DUEL | 2 / 1 / 1 | 22 / 14 | 0 |
| SP | 2 / 1 / 1 | 14 / 5 | 0 |
| G | 3 / 1 / 0 | 29 / 6 | 2 |

合计 12 局：7 胜、3 平、2 负，击杀 65、死亡 25、自死 2。
不筛选胜局；页面保留全部 12 局，默认按预声明种子、对手和座位顺序展示。

## 复现与更新

实际导出目录为 `runs/kill-replays-20261115-01`。主要产物：

- `PREDECLARED.json`：运行前固定的种子、对手、模型/代码哈希与 CPU 资源。
- `candidate_actor.pt`、`SOURCE_CHECKPOINT.json`：私有的候选权重及来源，不发布到 Pages。
- `replays.json`：动作、击杀/死亡事件、每 180 tick 状态 SHA256、终局统计。
- `watch.html`：约 1.3 MB 的完整离线播放器。
- `RESULT.json`、`BROWSER_CHECK.json`：完整对局导出和桌面/手机浏览器验收。
- `PAGES_PUBLISH.json`：发布提交、公开路径和文件哈希，以状态字段区分提交与部署成功。

导出只使用 CPU，最多 8 个 CPU 核，既有八卡训练继续运行。没有修改训练指纹覆盖的源码。
无需安装新的训练依赖；播放验收的 Playwright/Chromium 位于项目 `.tools/replay-preview`，
与系统和 `.venv` 训练依赖分离，不纳入 Git。

```bash
cd /root/private_data/qqt_rl
source scripts/dcu_env.sh
# 输出目录必须是新目录；省略 --checkpoint 时在开始时冻结当前 LATEST
python -u -m scripts.export_dcu_kill_replays \
  --out runs/kill-replays-NEW \
  --checkpoint runs/kill-jax-8card-e512-20261115/uXXXXXX
# 不推理，只独立重放全部动作并核对状态哈希、终局与逐事件统计
.tools/node/bin/node scripts/verify_dcu_kill_replays.js runs/kill-replays-NEW/replays.json
```

以后更新可给相同的固定对手/种子组生成新检查点录像；如只看新模型，不必加载旧模型做对照。
原始权重保留在持久项目目录，公开站点仅放动作/素材/页面。不要覆盖同名固定点录像，
为新检查点生成新文件并更新目录入口。

## Pages 发布边界与验证

现有 Pages 通过 main 的 push 自动部署。此次基于远程 main
`64eb579af03de007cb59a4bab71aaf740a8458de`，只新增两个静态文件：
`web/dcu-replays/index.html` 和 `web/dcu-replays/u000330-20261115.html`。
没有将训练分支整体部署为线上游戏；训练工具与交接记录仍在 `dcu`。
Pages 发布提交为 `ce1cb3fbeff79782e607f7c6c3631ad569dc5d2c`。

验收包括 21600 tick 的独立 Node 重放与统计复算、桌面/手机浏览器的状态 SHA256、
播放暂停、跳转终局、换局、画布尺寸及布局。公开地址已核对同一发布提交的
Actions 成功、`build-info.json` 和两个文件 SHA256，并完成公开内容的桌面/手机浏览器检查。
容器 Chromium 默认网络直连曾超时；公开内容浏览器检查使用容器常规 HTTP 转送，
保留生产安全响应头，状态 SHA256 和交互通过，传输范围在报告中明确记录。
实际验收状态见 [发布记录](../reports/dcu_kill_replays_20261115.json)。

训练接手仍优先阅读 [2026-11-15 交接文档](dcu_training_handover_20261115.md)。
