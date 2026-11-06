# G 对拼 → 联赛 v2 训练与评估路径（2026-11-06）

本次把实际运行入口的 import/require、动态源码替换与配置内模块名推导出的代码闭包放入 main。代码入库不代表恢复训练、改线上默认或放行实验。历史产物、外部模型与已冻结协议仍由各自 owner 管理。

## 谱系与当前结论

- G 自然局联赛由 `combat_specialist_g_league_train` 驱动，经 `multi_region_g_train_v1` 到独立 `combat_regions_train_v2`。独立区域线从 u18102 开始，70%自然配方、30%原生区域 reset；保留真实 Bank 总量25%、两席均衡、原地图与三技能接口。u18262 为 G_frozen。
- G_frozen → `cc_train_v1` 自模仿各系 → `cc_league2_train` 的 peer 联赛。L2_S05g 从 L2_S05 u19142 换 seed2026619142，固定 bot floor、SIL=.5，到健康点 u019302。固定权重为 `[0,.025,.05,.25,.20,.05,.275,.10,.05]`，progression threshold=1.01 表示关闭自动变权；不把最初动态联赛配方当作此点配方。
- L2_S05i（seed2026649302，u019382）和 L2_S05j（seed2026659302，u019462）均从同一 u019302 完整 checkpoint 换 seed，属于配方复现。独立候选 holdout 尚未完成时不得宣称晋升。
- FTN2 也是从健康 u019302 分叉，seed2026709302。初始160更新，后继累计1200更新按独立冻结预算执行。它增加 continuation-frontier v2 奖励，与原配方复现分列。FTN、ATK、SP 为实验支线；本次不将任何支线登记成新最佳。
- 网页 u9809 是同期五槽发布线；G/联赛使用三技能，不能把 u9809 画成 G 的父 checkpoint。当前图同时呈现优化与发布里程碑，并用切换线说明两条路线。

原始数值、配方关键字段、来源文件 SHA 与摘录见 [时间轴证据](../reports/mainline_timeline_evidence_20261106.json)。u019302 的原双座 argmax=50胜/18负/20平，独立复测=95/44/37（176局，score=.644886）；同批 G argmax=.5。categorical复测=.465909，低于同批 G 的 .522727。旧复测规则采用旧 categorical .386+SE 门槛，因此 PASS 不表示采样方式也胜过同批参照。

## 入口与最小代码边界

完整逐文件清单、SHA 与依赖边在 [闭包 manifest](combat_league_v2_dependency_manifest_20261106.json)。该 manifest 包含已存在于 main 的依赖，不是所有文件都在本次新增。审计只读取源码，不启动 worker。

| 环节 | 入口/主要模块 | 作用 |
|---|---|---|
| G 训练 | `combat_specialist_g_league_train`、`combat_specialist_g_owner` | 完整 Actor/Critic/Adam/env/aux/RNG/league 保存、smoke、fresh restore |
| 多区域 | `multi_region_g_*`、`combat_regions_{prepare,starts,train,owner}_v2` | 原生合法起点、课程迁移、独立 namespace、固定预算 |
| 联赛 | `cc_prepare_v1..v4`、`cc_train_v1..v4`、`cc_supervise_v1..v3`、`cc_snapshots_v1` | 历史自模仿配方与快照来源 |
| 联赛 v2 | `cc_league2_prepare{,_g}`、`cc_league2_supervise`、`cc_league2_train`、`cc_league2_lib` | peer current/ring 与 G_frozen fallback；40更新保存、80更新评估/变权边界 |
| 奖励变体 | `cc_reward_import_hook`、`cc_attack_reward_v2`、`cc_fountain_reward_v1` | overlay 后安装真实 reward 补丁，写实际加载回执 |
| FTN2 | `cx_ftn2_prepare_v1`、`cx_ftn2_train_v1`、`cx_ftn2_budget_owner_v1` | 独立有限预算、原始状态继承和 continuation-frontier 奖励 |
| 强评估 | `cc_strong_eval_v1`、`cc_strong_retest_v1`、`cx_strong_holdout_v1`、`cx_strong_watch_v1` | 冻结 checkpoint、同批参照、双座分模式与新seed复测 |
| 诊断 | `cc_strong_metrics_fountain_v1.{py,js}`、`cx_idle_diagnosis_v1`、`cx_idle_replay_v1.js` | SHA核对的物理重放、宽/严格喷泉、可行动静止与c类分解 |
| 温度 | `cx_temperature_{eval,report,inference,worker}_v1`、`cx_web_temperature_*` | 两动作头按温度缩放后合法mask采样；P1网页五对手单列 |
| 运行核心 | `r3_items_*`、`r4_k4_*`、`bomb_threat_*`、`jax_bomb`、`web/sim-*` | 原生观测/奖励、collector/GAE/Adam ABI、规则与实际 import 依赖 |

`model_training.py` / `model_evaluation.py` / `model_diagnostics.py` 及 Node 的 `model_evaluation.js` 是通用进程入口；真实模块和参数放在被忽略的 `local_config/`。不需要历史短文件名。`scripts/model_launch.py` 默认使用当前解释器和当前仓库；可用 `PYTHON` 选择解释器、`QQT_PROJECT_ROOT` 指定指向同仓库的可选别名，不依赖特定兄弟工作树。

## 命令与外部输入

在仓库根执行；`<PYTHON>` 是安装了 JAX/NumPy/Optax/pytest 的解释器。checkpoint 的 pickle 只允许已核对 SHA 的可信训练产物。导出/渲染另需现有导出器使用的 torch、onnx、onnxruntime、Pillow，按实际使用入口安装；这些不是产物入库的理由。

```bash
<PYTHON> scripts/audit_combat_training_path.py
<PYTHON> -m pytest tests/test_combat_league_training_path.py
<PYTHON> -m scripts.cc_reward_import_hook_test
<PYTHON> -m scripts.cc_attack_reward_v2_test
<PYTHON> -m scripts.cc_fountain_reward_v1_test
<PYTHON> -m scripts.cx_ftn2_reward_test_v1
<PYTHON> -m scripts.cx_ftn2_native_test_v1
<PYTHON> -m scripts.cc_league2_sp_lib_test
node scripts/cx_temperature_test_v1.js
node scripts/cx_idle_replay_test_v1.js
npm ci
npm test
```

这些命令使用 CPU；主训练入口仍拒绝 GPU 训练静默回退 CPU。FTN2 collector 的完整 checkpoint shape 测试 `cx_ftn2_collector_test_v1` 另需健康 checkpoint、Bank、spawn、冻结协议和完整 league bank，不作为无数据 checkout 的普通单测。

训练前，先由实验 owner 把 `<EXTERNAL_RUN_ARCHIVE>` 中匹配的输入恢复到协议声明的仓库相对 `runs/` 路径，核对 `COMMITTED.json` 的 state/policy SHA，以及 Bank、spawn、区域 train/dev/holdout manifests 与 league 固定历史。协议仍需独立冻结的 common/arm/seed、来源及源码 SHA。`combat_regions_freeze_v2` / `prepare_v2` 的外部输入根改由 `QQT_SOURCE_ROOT` 传入；前者是历史源码采集工具，会复制文件，只可用于隔离的复现树，当前 main 不需再次执行它。

main 保留了出生 RNG 审计修复、解释器选择及网页边界物理修复。旧协议的源码 SHA 与 main 不完全相同，**不能直接复用旧协议并关闭校验**；须审查差异后冻结新协议，独立 namespace、smoke 与完整 fresh restore 通过才可正式训练。本次没有重跑 GPU，也没有将旧协议伪造为当前源码的验收证据。

以下为通用入口配置形状，保存为本地 `local_config/training.json` 后执行；`gpu` 必须是当前另行授权的设备，示例值不是资源授权：

```json
{
  "module": "scripts.cc_league2_train",
  "args": ["runs/<EXPERIMENT>/protocol.json", "<ARM>", "<SEED>", "--smoke"],
  "gpu": 0,
  "env": {"PYTHONUNBUFFERED": "1"}
}
```

```bash
<PYTHON> model_training.py local_config/training.json
# 同 protocol/arm/seed 独立新进程 --verify-restore，通过后去掉 flag 执行正式有限预算。
```

`cc_league2_prepare NEW_ARM SOURCE_ARM SIL_COEF [SOURCE_UPDATE]` 保留从已核对完整状态派生协议的接口；`prepare_g` 则从 G_frozen 派生。原始 i/j 的 seed、权重与 source 都记录在证据 JSON 中；不要只更换 arm 名却遗漏独立 collector/update RNG。

强评估配置示例（由冻结协议提供88局/模式或原始44场景分母）：

```json
{
  "module": "scripts.cc_strong_eval_v1",
  "args": ["point", "L2_S05g@u019302", "--seats", "0,1", "G_frozen"],
  "env": {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "CC_STRONG_CPUS": "0"}
}
```

```bash
<PYTHON> model_evaluation.py local_config/evaluation.json
# 复测配置换 module=scripts.cc_strong_retest_v1，args=["run","L2_S05g@u019302","G_frozen"]。
# 温度配置换 module=scripts.cx_temperature_eval_v1，args=[]，提前冻结输入与新namespace。
```

CPU列表也须由 owner 选择。脚本保留原实验的默认槽位/锁；部署到另一机器时应显式配置，不因文档示例占用已有训练资源。

## 物理版本、统计与选模

- 新评估默认当前 `bun-native-v2`。`native_eval_simulator_v1.js` 把实际 simulator SHA 写入重放，并要求诊断与生成时同 SHA。旧结果重放时，在通用配置的 env 中设置 `QQT_EVAL_SIMULATOR=legacy`，使用已在 main 的精确归档 `web/review-sim/bun-native-v2-20261104.js`；不改历史录像、不放宽 hash 断言。同一评估进程不混用两种物理，新旧结果分别命名/报告。
- 原强评估每席每模式44局，双席每模式88局；retest_v1 新seed2026612000–2087，44场景×2复制×双席，每模式176局。模式、席位、地图/区域、seed、规则SHA和对手模式必须分列；相同场景或双座不是独立地图。
- 固定对手 G_frozen 始终 argmax。同checkpoint换seed复测，并加入同批 G_frozen 自对弈参照。原样保留旧复测规则，但新选模不能用旧 `.386` 当作跨seed不变的采样基线。
- 先看胜负与真实交付，再看因果杀、携包失败、自死、静止与喷泉，报告 seed-cluster 配对区间。选择后的最大值是探索性结果；不凭邻近两个 checkpoint 通过冒充同点复测，不把放泡数量当能力。
- 网页快筛模型固定 P1，真实 Tactical / Hunter hard-normal-easy / Coop hard 五对手；与实际线上 u9809 及同批 G 分开对照。新规则/源码SHA不匹配不能复用旧局。最终默认和模型资产由独立发布 owner 决定。

## 防 hack 与已知短板

奖励不按空放泡数或 idle 处罚直接付钱。FTN2 第一个有效批次只建序列，同tick/同batch不付 continuation，gap≤24，后续 `.01×min(streak−1,4)`，每局最多20次，同步连锁支付0。frontier 只在本来存在的 c 类机会、模型所选动作原生可存活、后态和扫掠路径安全且严格推进时支付一次 .03；有包、道具、敌死、终局、危险与让包均拒绝。攻击正奖励经过30tick隔离；让包时撤销未付攻击项，追缴已付攻击项且不回填cap，禁取包→让包→击杀循环套利。原生道具/custody 语义不随支线改写。

u019302 的 greedy 长静止依然明显；温度.3降低长静止但 P1 强评估/默认 Coop hard 可能回退，不能称全面改善。区域课程 center 无合法起点的覆盖缺口必须保留；单图强评估不代表真人泛化。SIL v3/v4、SP、FTN、ATK 与FTN2都不能仅因代码合入就视为超过健康源。GPU finite、真实非零更新、完整恢复与能力评估是不同验收项。
