# 主训练路径整合报告（2026-10-28）

本次在 `integrate/main-training-path-20261028` 分支完成整合，基于 main `7128071`；来源基线为 `4757906`。所有实现、测试环境和报告均位于本 worktree。未 push、未合并到 main、未创建 tag 或 Release；未修改来源目录、来源 runs 或 Review 工作区，未启用 GPU。

## 范围与引用闭包

从 38 个主干 run 提取了 55 份有效 plan、LAUNCH、SOURCE_SEAL、PROTOCOL、ACCEPTANCE 等元数据。整合链路覆盖 it831 → Bun392 → JE03 retention/selfplay → JE05 → JE06/07/08 → JE09 v5/v7/v8 → JE10 r01…r19 causal_clean → PPO-SCALE A/B → COMBAT-MIX C → ITER-PBT D。

`closure.json` 逐文件保存来源 run、证据路径、引用链、依赖边、来源与整合 SHA-256。最终闭包 474 件：scripts 280、tests 154、jax_bomb 18、web 13、qqt_rl 8、pyproject.toml 1。闭包包括 main 已有且无须修改的文件，不能把 474 当作本次新增文件数。

依赖识别覆盖静态 Python import、动态模块/路径、JS require 和评估器调用；测试采用反向覆盖关系关联实现。源码清单中仅列名或独立启动命令中出现的支线入口经复审排除。`dagger_rl*`、`combat_item*`、`retention*`、`expert_*` 中保留的模块均为主干所需依赖，不代表引入整条支线实验。

PPO-SCALE 原先从独立 r15 CPU 入口取源码并三次替换后 exec；本次把等价展开结果保存为 `scripts/ppo_scale_eval_functions.py`，保留实际评估函数及原模块 globals 语义（续训会重绑定 B/R/notify），排除 r15 独立 runner/train/bank/report。展开前后 AST 等价检查与续训路径回归已通过，见 `ppo_eval_equivalence_20261028.json`。

## 三方合并与兼容处理

- `jax_bomb/bun_env.py`：来源与 main 内容原本一致，已包含 main 的因果初始化修复。CPU 验证发现 bucket 审计仍使用旧 `split(key, 11)[9]`，修正为现行 `_fresh` 的 `split(key, 13)[11]`；初始化随机流、环境转移及奖励均不改动。
- `web/review.js`、`web/review.html`：保留 main 默认模型、默认选项、完整奖励/Critic 证据、多模型比较、admission 拒绝原因；加入来源本地录像导入及无 decision 的诊断显示。清除模型过滤器只发生在用户显式导入后。`review_protocol.js` 冲突为格式差异，main 已有完整实现，保留 main。
- `docs/training_experiment_ledger.md`：合并冲突两侧完整条目，主干与支线历史记录均保留；活动/完成/暂停状态是记录当时状态，不代表本次重新核验实时训练。
- 两份 JE10 训练报告：保留 main 完整内容，附加来源完整历史记录。冲突图片来源版另存 `_source_20261028.png`，不覆盖 main 图。
- `scripts/je10_diagnostics.py`、`scripts/je10_evaluate.py`：合入增量的命名空间、种子、库存资格与并行评估功能，保留 main 逻辑。main 的 `je10_route_funnel.js` 被公开凭证 SHA 绑定，保持原文字节；增强版本独立为 `je10_route_funnel_current.js`，实时消费者使用此版本。
- `je09_stage_h5c_ops.py`、`prestock_joint_replay.py`：来源的 custody-age/return-window 扩展无当前主干消费者，保留 main，避免破坏历史凭证。`je10_supervise.py` 和既有 recipe 回归测试继续保留 main 已安装 recipe 的语义。
- JE06/07/08 recipe 在 import 时读取前代 runs seal 的依赖改为继承前代静态 `SOURCES`；只读导入不再要求外部 runs。执行历史实验仍需相应外部资产。默认 Python 使用当前解释器。

逐文件决定见 `merge_audit.json`；报告资产来源 SHA 和冲突处理见 `documents.json`。

## 文档资产与排除

选入主干报告资产 44 件：14 Markdown、9 JSON、19 PNG、2 指定 HTML。该数包含 main 中已存在且相同的资产。完整排除路径与理由见 `exclusions.json`：共 324 条（含通配排除规则），已知大小合计 328,103,705 字节；其中 59 个候选在最终 import 闭包复审中被进一步排除。

- 录像目录 `reports/je09_bun_20261017/route_pbrs_replay_g32/` 的所有资产均排除，合计约 328 MB。
- 所有 runs payload、checkpoint、pkl、pt 均不复制；只读取允许范围的元数据以追踪入口。
- 单件超过 5 MiB 的报告资产、录像、非请求的日志不纳入。
- 仅支线启动/源码清单涉及且不被主干 import 的入口和测试排除（包括无消费者的 campaign_gpu_batch_sweep/campaign_gpu_followup）；rules-v2 parity round2/4/5 面向另一个环境协议，未为这些测试引入支线规则。
- `node_modules`、`web/vendor` 是本地依赖目录，原有未追踪状态保留；`.hermes`、AGENTS 指引、临时虚拟环境及原始测试日志均不提交。

## 验证方法与修复

正式 pytest 使用独立 CPU 环境，JAX/JAXLIB 0.11.2，与来源运行环境一致；各文件在新的 Python 进程执行，四个并发、每进程四核。设置 `CUDA_VISIBLE_DEVICES=`、`JAX_PLATFORMS=cpu`、`JAX_PLATFORM_NAME=cpu`、`JAXBOMB_RULE=bun`，OMP/OPENBLAS/MKL 线程各 1。没有运行训练入口或 GPU 验收。

```sh
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAXBOMB_RULE=bun \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  <PYTHON> -m pytest -q --tb=short <TEST_FILE>
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu npm test
node <CLOSURE_JS_TEST_FILE>
```

pytest 运行覆盖现存 164 个 `tests/test_*.py`，包含 main 原有回归与闭包测试。`pytest_final.json` 保存逐文件结果与 skip 原因。npm 完整链包括 runtime 构建、网页回归、leaderboard 及 Review；另运行闭包选中的 13 个 JS 测试脚本。原始日志不发布，以免带入本机路径或机器信息。

修复内容包括 pytest helper 搜索路径、缺失的 JE08 原生状态 fixture、JAX tracing 前显式导入 safety、已退休携包 reset 的旧断言、通过真实 native steal 构建 critic context、ONNX 测试改用已声明的 WASM 单线程推理，Stage C 使用临时 declaration、默认 authority 命名空间的 recipe 断言，counterfactual batch=1 的既有配置断言，以及补齐 prestock 梯度 fixture 的 kind/preparation 字段（保留真实 actor 更新与零 critic 梯度断言）。需要真实 run/checkpoint、或被范围审查排除的支线文件的测试在缺资产时明确 skip，未伪造外部训练证据。

**执行边界偏差**：最初批次 runner 设置了 900 秒 timeout，Python subprocess 超时逻辑自动结束了两批由本任务启动的 pytest 子进程（初始 batch 2 和 batch 8，各约 900 秒）。这不符合“不杀任何进程”的字面约束。未执行手动 kill，未影响来源训练进程；后续正式 runner 无 timeout，所有新启动验证均等待自然结束。初始环境 JAX 0.5.3 的中间失败、选集收紧后已移出文件的收集错误和这次超时不作为最终结果，也未被隐藏为成功。

## 尚存风险

1. 没有执行 GPU 训练、8 卡一致性或真实 checkpoint 的完整历史验收；CPU 测试不能证明训练质量或复现所有 run。
2. 历史 seal 对应历史代码，来源仍有实时变化；本次元数据与代码来自各自读取时刻，不能宣称当前代码 SHA 等于历史 seal。
3. `scripts/je09_stage_h8_audit.py` 在来源与 main 均不存在，既有 H8 standalone smoke launcher 仍引用它。当前 JE10/PPO/PBT 代码测试不调用该历史分支，无法凭空恢复缺失实现。
4. JE08 PFSP8 只有 validation 元数据，没有根 plan/seal；依赖链由 PFSP4/H200 续训元数据补充支持。
5. 历史 `je05_web_hunter.js` 依赖外部 runs 中的 onnxruntime-node，完整历史 CPU 复现仍需配齐该依赖与 checkpoint/bank/plan；为保留历史凭证，本次没有改写该旧 loader。
6. 原有 `cx_screen_round.py` 引用缺失的 cx_seed_average，位于本次闭包之外，未扩大范围修理。

## 最终结果、隐私与提交

正式 pytest 共 **829 项：816 通过、13 跳过、0 失败、0 错误**，164 个测试文件均有最终结果。四个修复前失败文件在修改后整文件复跑通过，`pytest_final.json` 的 `corrected_reruns` 保留前后计数及修复原因。13 个 skip 为 11 项缺少未纳入的 run/bank/checkpoint 证据，另 2 项涉及被排除的 expert 支线入口（其中一项同时需要 run 资产）；逐项原因见该 JSON，未计入通过。

完整 `npm test` 退出码 0，展开执行 68 条命令（包含 runtime 构建和 Python 诊断回归，不将 68 当作断言数量）；额外闭包 JS 测试 13/13 脚本退出码 0。`web_tests_20261028.json` 保存命令、脚本收据及日志摘要 SHA。`environment_20261028.json` 记录 Python/Node/依赖版本，JAX backend 与所有设备 platform 均为 cpu。

隐私检查边界为 `git ls-files -z` 与本次明确选中/暂存的文件；未扫描其他未追踪 run/checkpoint/编排目录或 AGENTS 指引。检查用户标识（大小写不敏感，二进制也检查）、本机绝对路径、RFC1918 内网地址、机器 PID 字面量。语义性指标 `/home/capture` 和 `/data/checkpoint`、通用系统工具路径经复核排除假阳性。最终收据见 `privacy_scan_20261028.json`；扫描无命中。AGENTS、`.hermes`、node_modules、web/vendor 均未纳入提交；已有追踪的 runs 资产保持原样，本次没有新增或修改 runs 文件。

代码/测试提交：

- `1ad44788b17cb6edd76bf33abde1d90769b12628` — Integrate main training runtime and evaluation dependencies
- `4f26c54a032e3e4d2863b4f53ba6be097dbf882a` — Exclude unused capacity sweep entrypoints from integration
- `8d6a29c57c458bd33e160d404c96d39007cc339d` — Add CPU regressions and portable mainline test fixtures
- `1bb4aaae7e492dccd2f8ef33d889011fdedc064b` — Align critic context regression with native custody reset

文档、实验账本、报告资产与整合审计提交为 `ec4d98495e33eb2d7a15d81d2f9f8cd2b9fac1a8` — Document mainline experiments and integration validation。本次最终收据更新该提交的 SHA、diff 统计和扫描收据。最终收据提交自身 SHA 不写入自身内容，使用 `git log -1 --format=%H -- reports/main_integration_20261028/REPORT.md` 查询。

与 main 起点的最终差异按目录统计如下（A 为新增，M 为修改，无最终删除）：

| 目录 | 新增 | 修改 | 合计 |
|---|---:|---:|---:|
| docs | 0 | 1 | 1 |
| jax_bomb | 0 | 1 | 1 |
| pyproject.toml | 0 | 1 | 1 |
| reports | 33 | 2 | 35 |
| scripts | 137 | 5 | 142 |
| tests | 71 | 4 | 75 |
| web | 1 | 2 | 3 |

统计收据：`integration_stats_20261028.json`。与 main `7128071` 的最终差异为 **258 files changed, 85498 insertions(+), 27 deletions(-)**。

交付时只保留既有未追踪依赖 node_modules、web/vendor。本任务测试进程均已结束；不 push，完成后停下等待指令。
