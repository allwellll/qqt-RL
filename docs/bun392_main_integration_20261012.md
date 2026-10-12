# Bun392 主线整合与验收（20261012）

基于最新 `origin/main` 的 `a2606154df6b1ab4d118fc89cbff3a6d2028741a` 整合，
代码提交 `2006268`。实验取证截止分支提交 `22b079e`，审计其相对 main 的全部
248 个未合并提交和1274个差异文件，逐文件决策及完整提交列表存
`reports/je03_bun_stage2_20261012/main_integration_audit_20261012.json`。
采用文件白名单和局部适配，没有合并整个历史分支。

## 最终结论

完整结果：
[验收报告](../reports/je03_bun_stage2_20261012/bun392_final_acceptance_20261012.md)，
[完整曲线与证据hash](../reports/je03_bun_stage2_20261012/bun392_final_acceptance_20261012.json)。

Stage1教师KL＋45%战斗重采样组合减轻了无保持组的安全退化，组合内不能拆归因。
实际续训起点是当时冻结的teacher g2048；teacher g4096是另一个值得独立确认的候选。
Stage1历史技术AB明确 `stage1_passed=false`；Stage2来自用户随后明确授权，
不能倒写成第一阶段已通过自动晋级门槛。

Stage2两臂均完成4096代、每臂2,147,483,648 fresh ticks，14个检查点pair和终态
hash全部重新验证。动态12%联赛未形成稳定三档共同优势；动态g3072保留为本轮
最有希望的扩大样本确认候选，不能默认4096最佳，也未获得部署晋级。

下表每格为 safe/death/self/trade，每档Hunter及Tactical每臂64局：

| checkpoint | hard | normal | easy | Tactical |
| --- | --- | --- | --- | --- |
| 共同g0 | 2/1/0/0 | 4/1/0/0 | 10/5/1/2 | 21/3/0/1 |
| control g3072 | 0/1/1/0 | 3/2/1/1 | 12/4/3/0 | 22/1/0/1 |
| dynamic g3072 | 0/1/0/0 | 4/0/0/0 | 23/3/0/1 | 24/0/0/0 |
| control g4096 | 3/1/0/0 | 2/3/0/1 | 14/1/1/0 | 18/0/0/0 |
| dynamic g4096 | 0/1/0/0 | 3/4/3/0 | 16/2/1/0 | 19/0/0/0 |

动态终点对固定终点Hunter总safe持平19/192、死亡+2、自炸+3、Tactical+1。
g3072的优势主要来自easy，hard仍弱，full任务另有2局自炸。Stage2全曲线
carry capture14/16、near steal16/16，route/full capture/steal/return均0，
未见完整Bun任务进展。每档64局只是快筛，单seed、多检查点选择不提供最终显著性证明。

## 保留范围与冲突处理

- 原生Bun392迁移：仅复制共享14平面、11global语义；保留原生Bun24/global24及三能力。
- retention_v2候选奖励、分项与课程遥测、固定g0教师KL、战斗重采样。
  **lesson路由会覆盖名义合成奖励**，保留实际执行语义，不声称候选所有尺度已生效。
- 保守独立Critic PPO、平衡角色、recent/history版本快照与完整恢复、真实计数。
- CPU网页Hunter＋残余结算、Tactical、Bun任务及native安全probe；双锁恢复owner、
  nested SPAWNS修复、原子结果、hash验证、可选通知失败不影响评估。
- 迁移、KL梯度隔离、非零更新、fresh-process恢复、出生manifest、锁和篡改测试；
  只读最终曲线汇总与结论。

main的网页Hunter/页面/排行榜、package、workflow和其他现有更新全部保留，
没有用实验分支旧版本覆盖。历史评估协议与网页源码hash随最终JSON保留；
新main上的Hunter代码已前进，**新跑分必须新seal/protocol，不与旧曲线混报**。
历史run应使用其原始实验worktree和seal恢复，不能用本次可移植性修改绕过旧seal。

未纳入早期Admission/B1/B2/Reward/非Bun实验链、历史vendor源码、旧接管/kill脚本、
实时PID/锁/通知/启动receipt、runs/checkpoints/日志、.hermes控制文件及模型权重。
历史开源实现仅作为已完成审计的证据；本次代码来源都是当前实验worktree。
旧qqt-gpu-sim仅借用其已安装Python解释器执行测试，没有复制其源码。
网页模型目录没有改变，本次发布不替换任何在线模型。

## 可移植性与使用

旧checkpoint默认硬编码路径改为显式外部输入：`BUN_IT831_CHECKPOINT`、
`BUN_IT4000_CHECKPOINT`；新迁移namespace可用`BUN_MIGRATION_RUN`，
Stage2用`BUN_SELFPLAY_RUN`和`BUN_SELFPLAY_PARENT_RUN`。
CPU评估的64条冻结seed/spawn协议数据独立存`data/evaluation/bun_hunter_spawns_v1.json`，
可用`BUN_EVAL_SPAWNS`指定不同manifest，但新manifest必须新协议，不覆盖旧结果。
ONNX Runtime需自行安装`onnxruntime-node`或用`QQT_ONNX_RUNTIME`指定已安装模块。
export另需PyTorch/ONNX；不把这些可选评估依赖强加给网页部署。

CPU默认继承调用者核额度；可用`BUN_TACTICAL_CPUS`和`BUN_TASK_CPUS`明确隔离，
owner最多8个Hunter shard并发，probe最多3并发；所有评估子进程CUDA空、JAX CPU。
唯一Stage2入口：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
  python -m scripts.je03_bun_selfplay_authority_recovery_20261012 owner
```

本轮整合没有启动新训练，也没有再次启动已完成的评估owner。

## 验证

40项机制/恢复/权威回归全部通过；另外32项reward/curriculum/export相关测试通过，
其中2项恢复测试与前者重合，共70项不同Python测试。教师KL对Critic梯度为0，
Actor/Critic真实CPU更新均finite且非零，参数/Adam/EMA/RNG/league及合成环境标记
在fresh-process下一更新digest一致。后者是小batch optimizer恢复回归，
不冒称本轮新跑了生产GPU或完整物理环境训练；历史完整环境恢复见原run证据。

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAXBOMB_RULE=bun \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  tests/test_je03_bun831_20261011.py tests/test_je03_bun_stage1_20261011.py \
  tests/test_je03_bun_selfplay_20261012.py \
  tests/test_je03_bun_selfplay_authority_recovery_20261012.py \
  tests/test_tactical_spawn_manifest.py tests/test_summarize_bun_stages.py \
  tests/test_bun_retention_update_restore.py -rA

# 上述恢复文件与此组重合；用于新旧reward/curriculum/export兼容性。
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAXBOMB_RULE=bun python -m pytest \
  tests/test_bun_retention_update_restore.py tests/test_bun_reward_shaping.py \
  tests/test_bun_curriculum.py tests/test_export_bun_web_model.py

node --test tests/test_bun_onnx_authority.js tests/test_je03_native_bun_probe.js
npm ci --no-audit --no-fund
npm test
```

7项新增Node测试和main全套npm测试通过；初次npm测试因本地未安装既有PGlite依赖
失败，npm ci后完整重跑通过，不更改依赖或跳过测试。所有纳入Python/JS做AST/node
语法检查及git diff --check。main当前网页代码上的真实8局hard CPU smoke完成，
9.443秒、safe1/8，300tick＋残余hazards清零，ONNX对网页原模型logits parity通过；
这仅是兼容性smoke，不与旧64局结果比较能力。

## 发布与回滚

代码和证据文档分提交发布；无force push，无发布历史改写。
代码可用`git revert 2006268`回滚，文档提交可独立revert；本次不修改checkpoint或在线模型。
发布后按最终head SHA核验GitHub Actions和Pages终态，最终SHA/链接记录在本轮交付回复
及实验worktree的同日发布receipt中；waiting/in_progress不算成功。
