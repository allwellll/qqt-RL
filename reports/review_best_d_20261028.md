# 当前最佳模型 D 的 Review 诊断录像交付 · 2026-10-28

已在新分支 `agent/review-best-d-20261028-v2` 完成本地实现与重新验收。该分支直接基于 main `7128071`，未继承旧交付提交。Review 新增 D（ITER-PBT-V1 slot05 u720）模型条目及 **15 局、10,156 tick** 固定 seed 录像，成功、失败和平局全部保留；模型毛毛、对手蓝色毛毛，D 的可见播放轴为中文。右侧保持当前帧即时奖励和累计奖励折线。

v2 本地实现及证据 commit：`172194d705d3d0171f3a78669fb04ff2628498a4`。本报告由后续交付文档 commit 收录；可用 `git log -2 --oneline` 查看两次本地提交。旧分支 `agent/review-best-d-20261028` 保留在 `7977dc856787cae888c10a6f20bde31184ac2f56`，未改写或删除。未 push、未合并 main、未创建 tag/Release。网页默认模型、默认选项、旧 `web/replays.json`、旧录像和模拟器源码均未改动。当前普通网页模型目录未新增 D 发布条目；Review 初始录像仍为原首条，模型筛选初始仍为“全部模型”。

## v2 模型文件与 Git 大小修正

39MB `model.json` 已排除在 v2 的提交及可达新增历史之外。运行时仍只读取录像、目录及小型凭证，真实浏览器记录的该模型文件请求数为0。原文件保留在本地 `.hermes/best-d-private/model.json`，由现有 `.gitignore` 的 `.hermes/` 规则忽略；也可使用 `BEST_D_MODEL=<EXTERNAL_MODEL>`，Python验证器另支持 `--model <EXTERNAL_MODEL>`。没有添加模型下载或游戏模型发布入口。

`source-receipt.json` 现在记录 `model_sha256` 和 `model_bytes`。生成、组装与测试统一绑定该SHA，并与Actor凭证和录像目录交叉核对。本地模型存在时先核对文件SHA及大小，Python验证器再做checkpoint到JSON逐张量比对；缺失时明确输出 **SKIP**，`export_tensors_exact=null`、`export_tensors_status=SKIP`、完整Actor验证 `passed=false`，只将实际执行的数值检查标为通过。npm Review测试始终区分“小凭证/历史证明核对”和“本次精确张量核对”，不将后者未执行冒充PASS。哈希不匹配会报错，不按缺失跳过。

已分别执行模型存在与缺失两条Python路径、生成器输入检查和组装路径，并验证错误哈希被拒绝。[model-input-checks.json](review_best_d_20261028/model-input-checks.json) 保存各检查的真实状态。15局的actions、rewards、events、checks、summary、advantages与旧分支逐项完全一致，仅源代码hash及派生录像身份/文件hash随脚本修正更新，见 [branch-comparison.json](review_best_d_20261028/branch-comparison.json)。

v2 全部Git追踪/暂存文件最大为 **7,958,608 字节**（十进制约7.959MB，约7.590MiB），路径 `web/review-replays/h8-v36-g32-tactical_v2-2026101786.json`。检查阈值采用更严格的10,000,000字节；所有文件均不超过10MB。扫描也检查基线之后新提交的可达blob，排除旧对照分支，见 [asset-sizes.json](review_best_d_20261028/asset-sizes.json)。旧分支保留其大模型文件作对照；新分支没有继承它。

v2重新执行且exit 0：`npm test`（本地模型存在）、`npm run test:review`（显式指定不存在的本地模型，确认SKIP行为）、两个真实浏览器验收、隐私扫描。日志已更新为本次重新运行结果；场景结果和已披露的bf16/fp32差异保持原结论。

## 模型核对与数值证据

来源训练项目为 `cx_je03_admission_curriculum_20261009`，以下 checkpoint 路径均相对该项目根：

- `runs/iter_pbt_v1/formal/slot05_c000005/u000720/policy.pt`
- policy SHA-256：`ed5a2b1860e8256fa34ac5a5830ff9b5c81e67a0b33d5c8a73101b3149796663`，**使用前核对一致**。
- 配套 `state.pkl` SHA-256：`eb90f66fa52e53f5988742043cefaed0d8b3a9a471e3993c1f13f173e44bc341`，与 `COMMITTED.json` 一致；state Actor 与 policy Actor 每个叶子、全部副本完全一致；配套 Critic 张量 finite，摘要见凭证。
- 原生 `scripts/export_bun_web_model.py` 导出的 `75` 个 fp32 Actor 张量与 checkpoint **逐项完全一致**，编码 `bun24-trap-timer-v2`；未加非 Bun 输入适配器。
- 本次 JSON SHA-256：`ab6fa204a1c867081c71950de3ff472701bd8351c89fc12e385449c346d7266a`。
- 录像使用与该 checkpoint 配套的 CPU/WASM ONNX，SHA-256：`919b117bb07f8c9b03528b7c8b214b923af2160de7009f007800f528428e5f06`；先校验来源模型 checkpoint SHA、JSON 全部张量和图文件 SHA。
- 每局预声明 tick 0/1/50/199（不足则只取存在的 tick），共 **57 个状态**。网页 JS、WASM 与同网络定义的 **JAX fp32 导出参考**最大绝对误差 `3.957748413e-05`，阈值 0.002，greedy 动作差异 **0/57**。
- **原始训练 JAX 使用 bf16，不能声称与网页 fp32 完全一致**：同样 57 个输入上，最大 logits/Value 绝对差 `0.6683850288`，greedy 动作差异 **3/57**。JAX fp32 参考仅把原网络 backbone 显式 dtype 改为 fp32；未更改结构、权重、输入或采样状态。原训练 bf16 另行实测并完整保留差异。

详细证据：[actor-parity.json](../web/review-evidence/best-d-20261028/actor-parity.json)、[source-receipt.json](../web/review-evidence/best-d-20261028/source-receipt.json)。Critic 仅核对来源与张量，未用它冒充已完成 A1 计算。

按编排决定采用 u720；`BEST.json` 的 u760 未使用，原因是任务给出的 u760 argmax 交付只有 277/304。历史 u720 holdout 的 JS Hunter 难／中／易 safe_win 为 18/27/32（各64局），JAX categorical／greedy 交付为 304/302（各304局）。这些是来源项目已有结果，本次没有启动其 GPU 评估，也没有把它们混作以下 Web 实测结果。

## 固定场景、seed 与结果

生成前写入 [matrix.json](../web/review-evidence/best-d-20261028/matrix.json)，所有 15 局全部保留，未按结果筛选。Actor greedy、每 tick 推理、assist-off，使用生产 `web/sim.js` 与现有真实 Hunter adapter。

| 场景 / 对手 | 局数 | 预声明 seed | 胜 | 负 | 平 | 模型死亡事件 | 模型交付事件 |
|---|---:|---|---:|---:|---:|---:|---:|
| 806清砖对拼 / 困难 | 3 | 2026102800, 2026102801, 2026102802 | 0 | 0 | 3 | 0 | 0 |
| 806清砖对拼 / 普通 | 3 | 2026102810, 2026102811, 2026102812 | 3 | 0 | 0 | 0 | 0 |
| 806清砖对拼 / 简单 | 3 | 2026102820, 2026102821, 2026102822 | 3 | 0 | 0 | 0 | 0 |
| 806带砖原生整局 / 普通 | 3 | 2026102830, 2026102831, 2026102832 | 0 | 3 | 0 | 0 | 0 |
| 预置持包返程 / 普通 | 3 | 2026102840, 2026102841, 2026102842 | 3 | 0 | 0 | 0 | 3 |

清砖对拼：双方1血、300 tick、固定出生 `[5.5,5.5]` / `[5.5,9.5]`，清除可破坏砖，关闭抢包终局，采用原生对拼规则而非糖泡抢包规则。胜负口径为：模型有死亡则负，否则独占归因击杀对手则胜，否则平；保留模拟器原始包子 `winner`，以独立 `summary.outcome` 表示对拼结果。

原生整局：保留806砖块、糖泡与道具，2400 tick 上限，不预置携包。三局都到2400 tick，按生产模拟器包子规则判负；模型没有死亡但没有交付，不能把存活当作胜利。持包返程：沿用 Review `carry_home` 初始化，清砖、1血、预置敌包；分别在85、91、80 tick交付。

失败标签使用既有自动事件规则，附加 `match_loss` / `no_delivery`：三局原生局全部为“对局失利、未交付”；中／易清砖六局的既有规则还标出“窗口未切换”，这是观察事件标签，不改变对拼胜负。全部逐局 tick、reward、双边死亡/交付、标签、大小和 SHA 见 [verification.json](../web/review-evidence/best-d-20261028/verification.json)。

## 每帧奖励与优势边界

对全部10,156 tick复用 Review 当前 `diagnostic_reward.py` / `recompute_review_metrics.py` 的 JE10加法账本、安全适配器和终态aux更新，在生产 Web 真状态/事件上 CPU 离线复算；23项加法分量逐帧校验总和，零奖励通过稀疏记录的缺省0表达。新进程独立重算全部15局，分量及总和完全一致。相关源文件 SHA 均封存在录像，所有10,156 tick已测。

页面标题显示“即时奖励 · 离线复算”，其 tooltip 和录像元数据写明：**这是现行 Review 口径，初始aux归零，不是 ITER-PBT 历史训练账本或 rollout reward**；终局仍有危险时保持未结算余额，不伪造额外tick。右侧仍只有即时奖励、玩家切换和奖励曲线；原诊断DOM与计算绑定保留在隐藏区。

raw A1、shaped A1、ΔA均未计算、明确为“未记录”。没有用 Reward、Value 或 Logit充当优势，也没有构造V=0的伪反事实。为保留旧完整指标准入，新条目放在 `review-catalog.json` 的 `evaluation` 扩展中，独立校验 Reward-only 范围；原22条完整Reward/Critic诊断、原准入器及其测试均保留，D不会被标为“完整Critic合格”。

## 测试与真实浏览器

- `npm test`：exit 0，含网页物理、Hunter、Worker、模型目录、HTTP、赛果/资料UI和完整Review回归。
- `npm run test:review`：最终资产与来源哈希更新后再次运行，exit 0；包含新增 `web/test_review_best_d.js`。
- `node scripts/verify_review_minimal_browser.js`：真实 Chromium，1440×1000 / 390×844，原精简交互、拖动/播放/暂停、曲线点击、刷新深链、无效旧深链无残留奖励均通过。
- `node scripts/verify_best_d_review_browser.js`：本地HTTP+真实 Chromium，两视口均通过默认保留、D筛选15局、三类场景hash验证、播放/暂停、即时奖励/折线/游标联动、中文播放轴、毛毛/蓝色皮肤、缺测A1、无横向溢出；页面错误、HTTP错误、非本地请求、非GET请求均为0。浏览器显式禁用GPU。
- Actor checkpoint / state / JSON精确性、JAX fp32/JS/WASM数值核对通过；训练bf16差异另记，不掩盖。
- 奖励独立重算：15/15通过；协议终态、间隔状态hash和往返seek均通过；篡改模型身份、奖励范围、缺测状态、元数据身份会被拒绝。
- `git diff --check` 通过；默认目录、旧入口、旧 `replays.json`、生产模拟器和Hunter相对基线字节未变。

完整成功日志：[validation.txt](review_best_d_20261028/validation.txt)；机器证据：[D浏览器checks.json](review_best_d_20261028/browser/checks.json)、[原精简页checks.json](review_best_d_20261028/minimal-browser/checks.json)。浏览器工具借用外部现成Playwright/Chromium，缺失的共享库只补入外部临时工具目录；未改项目依赖。

截图已查看，以下均来自真实本地资产：

- [桌面D模型与全部15局](review_best_d_20261028/browser/desktop-selector.png)、[手机D选择](review_best_d_20261028/browser/mobile-selector.png)
- [桌面清砖对拼](review_best_d_20261028/browser/desktop-open.png)、[手机清砖对拼](review_best_d_20261028/browser/mobile-open.png)
- [桌面原生整局](review_best_d_20261028/browser/desktop-native.png)、[手机原生整局](review_best_d_20261028/browser/mobile-native.png)
- [桌面持包返程](review_best_d_20261028/browser/desktop-carry_home.png)、[手机持包返程](review_best_d_20261028/browser/mobile-carry_home.png)
- [原精简页桌面](review_best_d_20261028/minimal-browser/desktop.png)、[手机](review_best_d_20261028/minimal-browser/mobile.png)、[手机完整页](review_best_d_20261028/minimal-browser/mobile-full.png)

## 新增与修改文件

本地实现commit的新增文件逐项如下（包含任务开始已复制的精简页报告/截图/验证脚本，保留原证据）：

- `reports/review_best_d_20261028/browser/checks.json`
- `reports/review_best_d_20261028/browser/desktop-carry_home.png`
- `reports/review_best_d_20261028/browser/desktop-native.png`
- `reports/review_best_d_20261028/browser/desktop-open.png`
- `reports/review_best_d_20261028/browser/desktop-selector.png`
- `reports/review_best_d_20261028/browser/mobile-carry_home.png`
- `reports/review_best_d_20261028/browser/mobile-native.png`
- `reports/review_best_d_20261028/browser/mobile-open.png`
- `reports/review_best_d_20261028/browser/mobile-selector.png`
- `reports/review_best_d_20261028/minimal-browser/checks.json`
- `reports/review_best_d_20261028/minimal-browser/desktop.png`
- `reports/review_best_d_20261028/minimal-browser/mobile-full.png`
- `reports/review_best_d_20261028/minimal-browser/mobile.png`
- `reports/review_best_d_20261028/privacy.json`
- `reports/review_best_d_20261028/asset-sizes.json`
- `reports/review_best_d_20261028/model-input-checks.json`
- `reports/review_best_d_20261028/branch-comparison.json`
- `reports/review_best_d_20261028/validation.txt`
- `reports/review_simplification_20261025.md`
- `reports/review_simplification_20261025/checks.json`
- `reports/review_simplification_20261025/desktop.png`
- `reports/review_simplification_20261025/mobile-full.png`
- `reports/review_simplification_20261025/mobile.png`
- `reports/review_simplification_20261025/privacy.json`
- `reports/review_simplification_20261025/validation.txt`
- `scripts/best_d_model_input.js`
- `scripts/generate_best_d_review.js`
- `scripts/publish_best_d_review.js`
- `scripts/recompute_best_d_rewards.py`
- `scripts/verify_best_d_actor.py`
- `scripts/verify_best_d_review_browser.js`
- `scripts/verify_review_minimal_browser.js`
- `scripts/verify_review_privacy.py`
- `scripts/verify_review_asset_sizes.py`
- `web/review-evidence/best-d-20261028/actor-parity.json`
- `web/review-evidence/best-d-20261028/matrix.json`
- `web/review-evidence/best-d-20261028/source-receipt.json`
- `web/review-evidence/best-d-20261028/verification.json`
- `web/review-replays/d-slot05-u720-carry_home-normal-2026102840.json`
- `web/review-replays/d-slot05-u720-carry_home-normal-2026102841.json`
- `web/review-replays/d-slot05-u720-carry_home-normal-2026102842.json`
- `web/review-replays/d-slot05-u720-native-normal-2026102830.json`
- `web/review-replays/d-slot05-u720-native-normal-2026102831.json`
- `web/review-replays/d-slot05-u720-native-normal-2026102832.json`
- `web/review-replays/d-slot05-u720-open-easy-2026102820.json`
- `web/review-replays/d-slot05-u720-open-easy-2026102821.json`
- `web/review-replays/d-slot05-u720-open-easy-2026102822.json`
- `web/review-replays/d-slot05-u720-open-hard-2026102800.json`
- `web/review-replays/d-slot05-u720-open-hard-2026102801.json`
- `web/review-replays/d-slot05-u720-open-hard-2026102802.json`
- `web/review-replays/d-slot05-u720-open-normal-2026102810.json`
- `web/review-replays/d-slot05-u720-open-normal-2026102811.json`
- `web/review-replays/d-slot05-u720-open-normal-2026102812.json`
- `web/review_evaluation.js`
- `web/test_review_best_d.js`

交付文档另新增 `reports/review_best_d_20261028.md`。修改已有文件：

- `package.json`
- `web/review-catalog.json`
- `web/review.css`
- `web/review.html`
- `web/review.js`
- `web/review_protocol.js`

`node_modules`、`web/vendor` 均为原有本地软链，没有加入Git；`.hermes/`、AGENTS规则、私有trace/原始观测输入没有发布。

## 隐私、剩余风险与停点

隐私扫描仅限 `git ls-files -z` 与本次暂存文件，未遍历未追踪run/checkpoint/编排。扫描结果见 [privacy.json](review_best_d_20261028/privacy.json)：用户标识、本机私人绝对路径、内网IP、机器PID均0个实质发现；已人工区分历史表头中home/capture漏斗字段与游戏player id。AGENTS被根级规则忽略且未追踪。

剩余风险：

1. 网页fp32与原训练JAX bf16有已测精度与动作差异（3/57），不得把本次录像称为训练执行逐步复现。样本一致性也不能证明全状态一致性。
2. 原生Web整局本批0/3交付，与历史JAX holdout表现差距明显；本次只交付诊断录像，未修模型、输入、物理或默认设置，未将差异归因成已证实根因。
3. 奖励为当前Review离线复算，有Web/JAX安全适配近似和终局pending限制；D原始训练reward、raw/shaped A1、ΔA未记录。持包预置成功不能当作自然整局成功。
4. 场景各3局，是固定诊断样本；清砖出生固定，不等于64局随机spawn holdout，也不作统计泛化。
5. 精确张量复验依赖忽略路径或 `<EXTERNAL_MODEL>` 中的本地模型。干净clone没有该文件时会明确SKIP，不声称本次已做精确张量比对。网页播放不依赖此文件；任何后续上线仍需另行指令。

本次仅CPU计算；未启动/停止/占用训练GPU，未在来源 `runs/iter_pbt_v1` 下写入任何文件，未终止其他任务进程，历史run/checkpoint保持原样。本地测试服务器和本任务浏览器均已关闭。完成交付后停下等待指令。
