# DUEL/SP 最小迁移包交付（20261107）

用户最新指令为“不用验证，直接上传”，覆盖此前等待SP及干净目录验收的要求。**仅 DUEL 本机 CPU 加载检查通过，SP 与干净目录未验证。** 本轮不跑GPU训练、collector rollout或优化器更新，不恢复/重建旧state.pkl。没有启动训练/后继、没有CC/force，没有终止未知进程。检查停止自启验证进程时其已结束，没有额外发送进程信号。

## Release 和代码

[training-bundle-20261107](https://github.com/allwellll/qqt-RL/releases/tag/training-bundle-20261107)。19个运行payload分别上传，另带MANIFEST.json与SHA256SUMS，共21个Release资产；没有再上传重复的压缩归档。

运行payload合计 **275,698,799 bytes**。下载器按清单放到 `data/portable-training-20261107/`，源码在main的 `training_runtime/items_v3/` 和 `scripts/portable_training.py`。数据、policy与Bank只在Release，不进Git。正常推送的main提交及最终Actions/Pages回执见本报告末尾和本地交付收据；Release tag固定本次代码版本。

正式说明：[docs/continue_training_on_new_machine.md](../docs/continue_training_on_new_machine.md)，README及docs索引已有入口。包含硬件/显存、Python/JAX/CUDA来源环境、clone、逐文件下载/SHA、目录、两线开训示例、模型保存方式、优化器从零与对手池重建预期、常见问题和重新授权要求。当前tracked停止策略false保持有效。

## 最小闭包及历史边界

原ACCEPTED_RULE包含9个私有模块，原协议还固定约300个溯源源码/产物。并非clone main即可直接运行原历史入口。本次仅把新入口实际选择的4个封存模块补入Git：两份JAX物理、regions混合任务和DUEL混合任务；保留其原字节SHA。canonical历史sim代码独立封存，不覆盖网页生产物理。私有训练wrapper/FTN/SYNC分支不属于本次新入口，不纳入运行映射。

生成可移植ACCEPTED_RULE和两个新协议，原协议不改。全部路径仓库相对，manifest按文件列大小/SHA/用途，源码关键路径另列SHA；无旧机器绝对路径和授权/PID/续跑编排。两份真实Bank只保留实际读取的states/aux/attribute_highwater，座位0/1为25/16条，保留原高水位和奖励历史；没有伪造carrier或复制额外训练样本。

模型9个：DUEL最后点u020905与网页点u019902、SP最后点u024155与网页点u023302；5个实际对手锚点G_health(u019302)、G_j(u019462)、G_i(u019382)、G_frozen(u018262)、SP_fixed(u004002)。G_frozen是本次Cfixed/SP回退的实际运行依赖，因此包含。旧own ring不打包，按用户授权从起点重建；C是明确的封存fallback，不冒称已接入动态跨线peer。

旧K1 u004642、bt_slot01 u003680、bt_slot05 u002320、env2048 u000160、r3_a u002749是原协议中的历史policy/lineage引用。新入口不读取这些模型，因此不把它们当成运行缺口，不上传无关历史。当前source tree缺少其中若干历史点，不伪造替代。需要的9份actor均找到且按来源COMMITTED或既有固定对手manifest核对SHA；未缺实际运行actor。

新入口明确不是旧full-state继续：actor继承，critic确定性随机初始化、Adam从零、环境/RNG fresh、pool重建。只保存policy.pt与POLICY.json及日志，不保存完整状态/优化器；每40update和最后保存，固定预算结束不续跑。GPU更新路径本轮未执行，不把CPU加载或文件完整性等同于GPU训练已验收。

## 本机与跳过项

此前DUEL本机CPU检查输出PASS_CPU_ONLY：起点20905、9模型、6对手槽、4私有模块、Bank25/16、4fresh环境、obs[4,25,13,15]/global[4,38]、推理finite；CPU设备，rollout_calls=0、optimizer_updates=0、state_pickle_reads/writes=0。

按最新要求，SP不作为已验证交付，干净clone目录未执行，GPU未执行。只保留用户要求不可跳过的隐私扫描和Release上传后SHA256回读；不启动新一轮或补跑验证。

## 19个运行文件

详细SHA256与用途见Release [MANIFEST.json](https://github.com/allwellll/qqt-RL/releases/download/training-bundle-20261107/MANIFEST.json) 和 [SHA256SUMS](https://github.com/allwellll/qqt-RL/releases/download/training-bundle-20261107/SHA256SUMS)。Release文件名以 `--` 表示包内目录。

| 包内相对路径（省略data/portable-training-20261107/） | bytes | 用途 |
|---|---:|---|
| ACCEPTED_RULE.json | 2643 | runtime protocol/rule |
| canonical/web/assets/maps/levels.json | 21765 | frozen v3 canonical map table actually read by JAX |
| datasets/BANK.json | 460 | runtime protocol/rule |
| datasets/SCENARIOS.json | 748392 | DUEL train/dev/holdout scenario definitions |
| datasets/p0_bank.pkl | 1501210 | runtime genuine Bank states/aux/highwater |
| datasets/p1_bank.pkl | 961210 | runtime genuine Bank states/aux/highwater |
| datasets/train_spawns.json | 1801 | native own-side spawn pairs |
| datasets/train_starts.json | 135432 | regional reset curriculum |
| models/DUEL_latest/policy.pt | 30257447 | DUEL new-round starting actor |
| models/DUEL_web/policy.pt | 30257447 | published web actor, optional restart point |
| models/G_frozen/policy.pt | 30257447 | SP C fallback and both Cfixed anchors |
| models/G_health/policy.pt | 30257447 | DUEL fixed G opponent |
| models/G_i/policy.pt | 30257447 | DUEL Cring fallback |
| models/G_j/policy.pt | 30257447 | DUEL Ccurrent fallback |
| models/SP_fixed/policy.pt | 30255879 | SP fixed G opponent |
| models/SP_latest/policy.pt | 30257447 | SP new-round starting actor |
| models/SP_web/policy.pt | 30257447 | published web actor, optional restart point |
| protocols/DUEL.json | 5214 | runtime protocol/rule |
| protocols/SP.json | 5217 | runtime protocol/rule |

## 发布收据

Release上传后SHA256回读、tracked与Release隐私扫描、同SHA Actions/Pages结果待发布完成回填；不以此段提前宣称成功。
