# R3-sp吞吐护栏修订与自踩胶审计（20261031）

授权 `.hermes/STEER_sp_guard_amend_gpu3_20261031_1600.md`。GPU3独占恢复u3040，原seed2026323001、640新增更新（167772160真实步）、55%自对弈与45% bot及奖励/超参不变。

## 执行状态

23项护栏/完整恢复编排定向测试与2项自踩胶分组测试通过。已重载唯一owner/watchdog；未向任何训练worker发信号。新attempt `runs/r3_selfplay_ramp_20261031/attempts/gpu3_guard1600/` 保留旧暂停与收据。完整恢复、smoke、fresh restore与首正式更新均已通过，640预算仍在执行。

BT slot00/01/05/06/07下降超过15%，R3-a/b与R2-b/c超过5%，均须三个不重叠5更新窗口才请求SP原子暂停。当前恢复收据 `GUARD1600_TASK_CONSUMED.json`、冻结修订 `GUARD_AMENDMENT03.json`。R2-a暂停、BT03 BLOCK、BT04退休均保留。

## 持包自踩自己胶（只读）

使用已有768完整自然JS局中的R3-a u3600 / hunter hard / P0，64局；输入16个outcome分片SHA均与原COMPLETE一致。事件条件type=2、owner=player=0、carried>=0；按整局是否至少一次分组，66次不是66局。

|分组|局数|胜/平/负|整局死亡数|死亡/局|持包自踩胶次数|自伤|
|---|---:|---|---:|---:|---:|---:|
|有持包自踩自己胶|15|15/0/0|4|0.2667|66|0|
|无持包自踩自己胶|49|44/3/2|9|0.1837|0|0|

合计64局，59胜3平2负、13次死亡、66次持包自踩胶、自伤0。此样本未显示自踩胶组胜负更差；其整局死亡/局较高，但不能把死亡归因给胶。分组来自整局事后接触，受持包/存活暴露时长与状态选择影响，不能证明自踩胶有益、无害或致死因果，不改奖励。

原始逐局seed/接触tick/死亡与输入SHA：`runs/r3_selfplay_ramp_20261031/audit/guard1600/OWN_GLUE_AUDIT.json`；复现 `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m scripts.r3_sp_own_glue_audit_20261031`。

## 首正式更新与参照吞吐

u3040完整恢复、两更新smoke、u3042 fresh restore、u3043正式原子更新均已有实证，详见下方验收。

<!-- WALLTIME_CORRECTION -->

## 吞吐统计分母纠错

`AMENDMENT04_REFERENCE_WALLTIME.json`保留原pre-SP更新锚和原始错估计器快照。R3-b共享卡29/60秒交替，原u3827五更新总吞吐6316.51 env/s，u3839为6382.34（+1.04%），单步速率中位数却从9038.13降为8017.66（−11.29%）。按实际总步/总cycle墙钟计算同5更新吞吐，并与原锚相同奇偶阶段窗口比较，避免交替调度导致假报警；三个不重叠5更新窗及BT15%/R2-R3 5%均不变。仅重载owner/watchdog并接管存活smoke，未向训练worker发信号。

<!-- GUARD1600_FORMAL -->

## 正式更新验收

GPU3独占，u3040完整恢复与u3042 fresh restore均PASS；首正式u3043 finite、Actor/Critic非零：{'actor': 1.0213328497950536, 'critic': 1.5764455031844489}，262144真实on-policy步。
state SHA `9c72c960084989ab0aa4b47ef66e4f951df73ee900b730202ad9ebc9b5350665`；policy SHA `8cbbc12055090596b749428ba54dab3c1acd5adea44d551833e4f0a60e1261db`。
首周期env/s 989.184（含新进程冷编译，非暖态稳定吞吐）；最新已观察u3043，固定终点u3680。

|参照线|同阶段基线env/s|当前5更新总步/总墙钟env/s|比例|最低比例|连续越界窗|
|---|---:|---:|---:|---:|---:|
|runs/bomb_threat_v1:0|16743.72|16261.13|0.9712|0.85|0|
|runs/bomb_threat_v1:1|20152.25|20143.38|0.9996|0.85|0|
|runs/bomb_threat_v1:5|19387.95|19355.91|0.9983|0.85|0|
|runs/bomb_threat_v1:6|待暖态基线|待5更新|未可比|0.85|不预判|
|runs/bomb_threat_v1:7|4320.16|4358.38|1.0088|0.85|0|
|runs/bun_native_v2_20261030:b|15221.38|15569.99|1.0229|0.95|0|
|runs/bun_native_v2_20261030:c|3774.99|3780.58|1.0015|0.95|0|
|runs/r3_items_route_20261030:a|5595.18|6328.99|1.1312|0.95|1|
|runs/r3_items_route_20261030:b|6316.51|5514.09|0.8730|0.95|0|

吞吐比例是时间窗相关观测；同5更新总步/总cycle墙钟与原pre-SP锚的相同奇偶阶段比较，修正交替调度中位数偏差；参照run变更后才建立新基线。收据 `runs/r3_selfplay_ramp_20261031/GUARD1600_FIRST_FORMAL_ACCEPTANCE.json`；640预算尚未完成，由唯一owner及watchdog继续。

## 暖态进展与新增CPU复核

GPU3最近五个更新[3071, 3072, 3073, 3074, 3075]全部finite，1310720真实步/75.382秒=17387.6 env/s；首正式冷编译周期不混入暖态。收据 `runs/r3_selfplay_ramp_20261031/WARM_UPDATE_ACCEPTANCE_20261031.json`。38项focused测试通过。

16:20新指令已落实纯CPU复核 `runs/r3_ownside_recheck_1620_20261031/PROTOCOL.json`，u3040三档既有768局及SP父greedy128局复用；新增1920局。holdout扩展为逐模式每席128局，原每模式64/席的共同前缀单列；候选R3-a u3760/R3-b u3600三档各256局。无训练控制/发布副作用。

## 后续资源护栏终态（非预算完成）

R2-b连续三个不重叠5更新窗口低于0.95，已按原合同只暂停SP；u3127完整原子保存，state SHA `00b1c960b745e2e58db7a6347adc41fe5382888bdaa7be5a465eb2c97083a4af`、policy SHA `c6df56805ac7f5d184a3aa0568b066d57c4b0ed64b9ff2e6fa18666fdd1197a3`复算一致，next-step token存在，worker已退出，未发信号。新增87更新/22806528步，剩553更新/144965632步，不能标640完成。GPU3因此空闲；原R3/R2-b等继续。收据 `runs/r3_selfplay_ramp_20261031/GUARD1600_BOUNDARY_STOP_ACCEPTANCE.json`。只证明保存哈希与token，未额外执行u3127 fresh restore。吞吐越界是时间窗观测，不证明由SP或CPU评估单独造成；不自动清闸或扩预算。
