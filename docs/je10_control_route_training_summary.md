# JE10 v38 击杀信用与搬包路线训练总结

本页长期维护 JE10 的目标、方法、固定评估结论和继续训练所需条件。**训练事件增加、某个检查点较好、技术运行成功，分别不等于固定泛化能力、整轮稳定提升或完成研究目标。** JE10 尚未解决原墙地图取库存后的完整返程，也没有跨原生地图泛化证据。

<!-- JE10-STATUS:BEGIN -->

**证据读取截至 2026-10-18 13:33:24 UTC+8**。这是读取快照，不代表此后训练静止。

| 身份 | 检查点 / fresh ticks | 固定能力与状态 |
|---|---|---|
| 网页当前发布综合最佳 | r01 attempt04 g8 raw / 16,777,216 | 既有发布规则选中的折中点；未随后续轮自动更新 |
| 最近完整轮 | r06 g128 / 268,435,456 | 8个预声明固定点评估全部COMPLETE；full交付1/16、drop5/16、Tactical17/64 |
| 正在训练轮 | r07，STATE g68 / 142,606,336 | finite/nonzero=True/True；最新COMMITTED g68 / 142,606,336，与评估点分列 |
| r07 最新完整固定评估 | g32 / 67,108,864 | Hunter难/中/易0/2/20（各64）；Tactical16/64；八类交付4/4/11/16/16/10/0/2（各16） |

当前轮已完成固定点：g0, g4, g8, g16, g32；未完成的预声明点评估不进入结论。八路线顺序为 death/sugar/drop/near/mid/far/recovery/full。

<!-- JE10-STATUS:END -->

本次只读证据来自独立训练工作区中的 JE10 产物；文中使用实验名和仓库相对的产物名，不包含本机路径。数字及出处哈希保存于[冻结证据摘要](../reports/je10_evidence_20261018.json)，全部固定点、曲线和本次验证记录见[当日训练报告](../reports/je10_training_20261018.md)。历史报告和账本中的“当前”属于写入当时的状态，本页以带截至时间的状态块为准。[实验总账](training_experiment_ledger.md)保留全项目历史；[网页发布报告](../reports/je10_best_model_review_20261018.md)保留既有模型选择和录像验收。

## 目标与范围

训练目标是把“制造敌人死亡/糖泡窗口 → 识别可获取目标 → 在窗口内持续推进 → 真实取包 → 准备/执行返程 → 回笼交付”连成闭环，同时保持战斗、安全和对手恢复时的合法切换。首控信用专门解决糖泡已经由模型造成、最后接触爆破却掩盖了原施害者的问题；路线学习专门解决首步看似正确但随后偏航、开墙不足、取包后走不回来的问题。

所有正式轮只覆盖 Bun806 的原墙/开墙状态、双席位、出生和距离桶。原墙与开墙在固定库存协议中分别绑定 team0/team1，存在席位与几何混杂，不能称独立地图对照。预置死亡、糖泡、掉包或持包窗口是合法 reset fixture；它们验证窗口之后的策略，不能证明模型自然制造了该窗口。只有 `natural_full2400` 是原生完整局。

首轮父点为已发布 JE09 Stage C g8 raw Actor、匹配 Critic 和完整对手池。r02 起依次继承前一轮最终 g128 raw Actor/Critic/pool；每个新 run 重置 Adam、RNG、环境、FIFO、统计计数，EMA 从 raw 初始化。它是连续策略迭代，**不是同父点、多随机种子的严格单变量因果实验**。某轮只改一个配方因素，仍不能消除父策略、状态分布和随机数的变化。

## 架构、预算与真实暴露

- Actor 为 Transformer：observation `24×13×15`、global 24、embedding 392、4 层、4 heads；move 5 × ability 3，共 15 个联合动作，Critic 独立。路线 phase/target 和弱对手目标元数据不进入 Actor 输入。
- 每轮 8 设备 × 1024 环境 × 256 tick，单代 2,097,152 fresh ticks；128 代共 268,435,456 ticks。每代每设备 128 次 optimizer updates，minibatch 4096，单 epoch，gradient accumulation 1，有效全局 Actor/Critic batch 各 16,384。
- PPO：learning rate `7.5e-5`、clip `.1`、gradient clip `.5`、entropy `.005`、value coefficient `.5`、Adam epsilon `1e-8`、EMA `.995`；GAE 的 γ `.995`、λ `.95`。
- 固定评估点 g0/4/8/16/32/64/96/128；恢复 checkpoint 每 4 代。正式预算不受中间能力成绩控制；导入、finite/nonzero、保存恢复、源码/副本完整性属于技术有效性检查。CPU 评估落后不阻塞 GPU 后继。
- r01 reset 权重为 combat `.40` / real-drop bridge `.25` / natural full `.35`，其他原生路线独立质量为零。r02 起为 `.35/.20/.35/.10`；最后 `.10` 由原生路线课程分配。**这些是 reset 抽样权重，不是训练 tick 占比。**

r06 全轮有效决策 tick 为 261,678,722，其中 full 200,966,768（76.7990%）、combat 35,384,524（13.5221%）、bridge 10,416,314（3.9806%），其余为路线课程。这说明完整局长 horizon 会显著改变实际暴露。上述分母来自 r06 g128 冻结整轮诊断的 `training.scene_valid_ticks`，不能将标称 35% combat 当实际战斗份额。

动态对手的底层 family 采样为 responsive rule `.30`、g0 `.20`、stage1 teacher `.20`、PFSP league `.30`。league 含 current、6 个固定历史成员、4 个 ring 槽；current 每代更新，JE10 生效的 `ring_interval=16`，每 16 代轮换一个槽。plan 内继承的 `opponents.history_interval=128` 和部分旧课程描述不是 JE10 生效字段，须核对顶层 plan 与源码。

natural-full 弱课程仅修改敌方合法动作，不注入死亡、掉包或 custody。弱 episode 比例每 8 代按 `.50/.35/.20/.10` 降低；角色为 courier/guard。carry assist 固定 `.1`，评估关闭。底层抽到 current 并不代表最后由 current 出动作：弱角色和 assist 可能覆盖它，必须同时报告覆盖前和覆盖后的来源。

<!-- JE10-EXPOSURE:BEGIN -->

| 点与来源口径 | current | ring | fixed league | 分母 |
|---|---:|---:|---:|---|
| r06 g128 底层league内部条件份额 | 9.5318% | 33.5582% | 56.9100% | 78,030,097 底层league ticks |
| r06 g128 最终实际action source | 6,314,149 (2.4129%) | 22,249,541 (8.5026%) | 37,389,080 (14.2882%) | 261,678,722 alive-learner有效决策ticks |
| r07 g32 底层league内部条件份额 | 5.9249% | 42.1807% | 51.8944% | 19,624,840 底层league ticks |
| r07 g32 最终实际action source | 819,376 (1.2466%) | 6,028,767 (9.1720%) | 7,246,190 (11.0241%) | 65,730,223 alive-learner有效决策ticks |

<!-- JE10-EXPOSURE:END -->

r01 未完整记录覆盖后的 action source；缺项保留未知，不能从 sampled family 或弱角色的 visited ticks 反推出精确真实来源。有限 current/ring 暴露也不能证明适应了当前强对手。

## 首次糖泡主信用与爆破残余信用

实现见 [je10_control_credit.py](../scripts/je10_control_credit.py)。只在真实 alive、未受控 → 受控边沿触发首控，来源矩阵为 `[batch,victim,actor]`；要求唯一敌对施害者，拒绝自伤、队友、来源歧义、terminal 和重复 tick。身份包含 episode、victim、life、entry，不能靠多次观察同一糖泡重复支付。

首次糖泡主信用为 **4**；已受控敌人的真实主动接触/爆破死亡仅支付 **.25** 残余给该次唯一敌对 trigger，原首控 owner 保留在独立归因账本。超时死亡不是主动爆破。单局每个 actor gross 上限 **8**，死亡扣回尚存 balance，扣回不返还 gross 预算。该机制区分“谁制造窗口”与“谁完成最后触发”，并不把接触者改写成首控施害者。

v38 通过 `native_trap_popper` 接入原生接触来源；受控死亡关闭继承的 settled kill **16**，避免 `4 + .25 + 16` 的重复信用。非受控直接战斗击杀仍保留 **16**；交付 **24** 及其真实 hazard settlement 仍是主要任务奖励。单元 ledger、native source 适配及与父奖励的接线均有 focused 测试。

## 连续拓扑 PBRS 与阶段目标

路线增量为 `λ × (γ Φ_after − Φ_before)`，JE10 λ `.5`、γ `.995`。基础场来自当前拓扑最短距离，距离截断到 64；目标身份切换用 offset 连续衔接，不在切换边界凭空增加潜能，terminal 的 Φ 归零。无效目标和拓扑变化有明确重基准/冻结规则。gross 正项、负项、净额与终局奖励分开看，不能仅因路线净额为负就判学习信号无效。

r01/r02 的名义 `.18` 端点不总是原生 steering 的实际端点；r03 使用已有的五个 native 一步端点选择最大正进度合法安全集合。随后 r03 固定诊断发现，旧连续距离在约 `2e-6` 位移处跳变 `.640002`，会让 PBRS 和 teacher 对微小移动给出错误方向信号。r04 用共享三角插值场修复：每格以中心划分四个三角形，邻接可通行格共享角点，断开的对角组件不共享，保留格中心距离，加入 `.125` 的小幅居中项。它是连续规划势能，**不是精确连续测地距离或学习 Q**。[连续场实现](../scripts/continuous_route_metric.py)供 PBRS、route teacher 和 native progress certificate 共用。

| 阶段 | 激活目标和退出规则 | 主要限制 |
|---|---|---|
| COMBAT_CONTROL | 自由敌人且 learner 空手；不强行运输 | 该阶段 DAgger 仅审计，不新增战斗回放 query；战斗沿 PPO/expert/retention |
| A 掉包/己方回收 | 同 episode、有效 nonce、真实归因 loose 对象 | 自己 origin 的包是回收，不能当作敌方包交付 |
| B 敌笼库存 | 敌人死亡/受控且有真实库存 | 原墙下可达敌笼不等于取包后能返程 |
| CARRY_RETURN | 已有真实 custody，目标为己方笼 | 不再受旧 120 tick 截断；死亡、失包、终局结束 custody |
| CAGE_DELIVERY | 已进入己笼的交付阶段 | 与 DAgger kind 编码不同，不能仅凭 kind 推断 PBRS phase |
| PRESTOCK_NAV / OPEN（r06 起 phase 5/6） | 取库存前选走廊 staging/revision，准备安全开路 | 不改变 Actor 输入；有限 witness 不是全路径规划保证 |

敌人恢复后空手关闭运输；已持包时，敌人自由且 L1 距离 ≤4 会切回近威胁处理。r05 将 PBRS 的“归因己方 loose → 拾回 → 回己笼”链与 teacher 对齐，修复“teacher 追己方包而 PBRS 指库存”以及“己方拾包后 PBRS NONE”的矛盾。目标身份包含 origin/nonce，不能把重复拾回当新任务刷奖励。

r06 把取库存前的通路准备接到同一 PBRS 与 joint DAgger 目标：区分 ready、preparing、navigation、opening，保存 revision 与 objective，使用受限 40 步原生开路 witness、安全条件及合法 move×ability 集合。持包无法放泡，因此要先准备通路；有限 witness 未覆盖所有推箱、威胁、开墙和地图状态。机制可达不代表模型掌握。

## Set-valued oracle、learner-state DAgger 与偏航

每代由当前 learner 的真实动作生成新状态，oracle 仅提供标签，不替换 `env.step` 的动作。覆盖 scene1..9 的真实访问状态、窗口和完整 custody 生命周期；候选数、bounded selection、native-safe acceptance、首偏航 query、实际进入 FIFO/回放的数目必须分列。旧字段 `geometrically_reachable` 实际是 selected，不能用它计算几何可达率；未记录的几何候选分母为 null。

movement loss 为 `−log Σ π(a)`，求和只含合法 oracle 集合，不要求等优方向之间均匀分配概率。r06 联合监督扩展为 move5×ability3 的 accepted-pair 概率总量；普通 movement 标签保留所有合法 ability。危险、INF、无正进度、拓扑变化、死亡/受控 learner 等状态按安全规则拒绝，goal 上保留合法 wait。native 一步安全/进度证书不是未来全局价值。

每设备 FIFO 容量 **2048**，每代最多加入 **256**；microbatch 取 **64**，current/history 各半，缺失回退到可用侧，无效 padding 梯度严格为零。首偏航按 drop、stock、return、delivery 分阶段分配每 rollout 各 **4** 个 verifier 槽。查询可能跨 rollout 重复同一 episode，不能称独立机会数。

r07 只改 joint DAgger 的回放分配：每半 32 有组时为 return12 / drop4 / ready-stock4 / navigation10 / opening2；缺组只在本半重新分配，没有 preparation 标签时与旧 uniform 采样逐位一致。不改奖励/监督外系数、FIFO 新增量/容量、query、安全验证、网络/PPO、scene mix。重复上采样 opening 不会凭空增加独立开路机会。

“偏航”必须带口径：

1. **固定 JS 旧 oracle 偏航**：在有可观测合法 oracle 机会时，实际动作不在集合；按目标身份/阶段保留首次偏离。它仍用冻结旧可观测场，不自动等于新训练场的错误。
2. **native 当前场偏航**：在同状态、同原生 steering、安全/目标身份下重算各方向进度，判断实际方向是否比可接受动作差；也不是 Q 最优性证明。
3. **训练 capped 片段**：有限记录的 learner 首偏航样本，仅供定位，不能作为总体偏航率。DAgger kind4 是 own-recovery；PBRS phase 见 `trace.route_phase`，kind4 不是 CAGE_DELIVERY。

固定 Review 保留状态前后 hash、原动作、事件、phase/target/Φ。旧 JS reward-GAE 使用零后续值近似，不能改名为训练奖励、真实 Critic advantage 或反事实 Q。r07 增加 selected device/tick/env 的真实 `collector_gae_advantage`、value target/estimate、behavior logp；它是 Critic-backed rollout GAE、在每设备 PPO normalization 之前，也不是动作 Q。网页 r01 g8 另有明确 `Q1=r+γ×0` 的一步反事实 A1 近似，不能与训练长程优势混用。

## 固定异步评估协议

GPU worker 与全局唯一 CPU/JS authority 独立。每点只有 `COMPLETE.passed=true`、policy 身份与全部 shard/SUMMARY/DIAGNOSIS 哈希吻合，才能进入本页数据；尚未完成、失败、旧 INVALID_METRICS/INVALID_FIXTURE 不计。当前快照协议 SHA-256 为 `f2fcf50022fbb61e9630855ca3d08ccef4b1d390d89a783a5bdf998b9205a936`，所有纳入 v38 轮一致。

| 协议 | 样本和执行 | 能说明什么 |
|---|---|---|
| Python Tactical v2 | 每点 64，CPU JAX，固定 seed20261001，greedy，300 tick；无 web residual | 独立战斗保持；不能与 JS 或训练 bot 混算 |
| production JS Hunter | hard/normal/easy 各64；同 frozen64 spawn cells，seed20261001，greedy，300+64 残留结算 | 清砖、HP1 的战斗快筛；native_trap/items=false，零糖泡事件不表示无糖泡能力 |
| production JS 路线 | 下述八类各16，seed2026101800..2026101815，team=game%2，Hunter normal，assist-off | 窗口后路线及原生完整局；每项保留独立分母 |

| 路线任务 | 初始化/时限 | 解读边界 |
|---|---|---|
| native_death_stock | 合法敌方死亡库存窗口，1200 tick | fixture 非模型制造击杀；原墙/开墙各8 |
| native_sugar_stock | 合法敌方糖泡60 tick窗口，1200 | fixture 非自然首控能力 |
| real_drop | 归因 trap 后由原生 step 真掉包，1200 | 测追包/返程，初始 trap 不算模型击杀 |
| carry_near / carry_mid / carry_far | 预置真实规则允许的持包返程，1200 | 无需再拾包；不含自然找窗口/取包 |
| recovery_threat | 敌糖泡8 tick后恢复，1200 | 路线与近威胁切换，仍是 fixture |
| natural_full2400 | 原生 reset，2400 tick上限，真实终局 | 唯一完整局；16局仍不足以证明广泛泛化 |

报告窗口→目标的 episode 共现数、首方向一致/有机会局数、持续 oracle_matches/opportunities、progress_ticks/target_ticks、拾包/16、交付/16、恢复切换/机会、安全死亡和自伤。SUMMARY 中历史 `oracle_agreement`、`progress_per_target_tick` 是逐局比值之和；总体率必须由原始计数重新计算，零分母保持 null。pop 包含初始化 fixture 或超时，不能直接除以新首控做因果转化率。

## r01 至当前的方法和结果

下表为每轮**最终点或最新完整点**；不是逐轮最优点。完整八点曲线、[冻结绘图输入](../reports/je10_curve_data_20261018.json)、全部自伤和出处见当日报告，不择优删点。Hunter 三档和 Tactical 各分母64；D/S/Drop/Near/Mid/Far/Recovery/Full 的交付各分母16。

<!-- JE10-ROUNDS:BEGIN -->

| 轮次/所报检查点 | 相对前轮的方法 | Hunter难/中/易 / 各64 | Tactical /64 | D/S/Drop/Near/Mid/Far/Recovery/Full，各16 |
|---|---|---|---:|---|
| r01 g128 | 首次受控4/残余.25原生接线；PBRS .5；DAgger .2与完整custody（整合） | 1/5/15 | 5 | 3/4/11/16/15/11/4/8 |
| r02 g128 | reset mix改为.35/.20/.35/.10 | 4/1/11 | 10 | 3/6/11/16/16/13/3/5 |
| r03 g128 | 复用native一步端点纠正合法安全最大正进度集合 | 2/5/9 | 9 | 5/5/9/16/16/13/3/4 |
| r04 g128 | PBRS/oracle共享连续三角插值场 | 1/2/5 | 10 | 6/3/13/16/15/9/4/4 |
| r05 g128 | 己方loose→拾回→回笼的PBRS目标链对齐 | 2/4/10 | 8 | 6/6/10/16/16/13/5/7 |
| r06 g128 | 共享prestock corridor目标，phase5/6与native-safe joint DAgger | 1/3/11 | 17 | 1/1/5/16/16/7/0/1 |
| r07 g32（未完轮） | current/history内分阶段回放20 ordinary/10 navigation/2 opening | 0/2/20 | 16 | 4/4/11/16/16/10/0/2 |

<!-- JE10-ROUNDS:END -->

r01 是多项整合配方：首控/残余、PBRS `.05→.5`、DAgger `.05→.2`、生命周期和原生接线共同变化，不能把效果归给其中一项。r02 是场景混合修正；r03 是 native 一步 progress set；r04 是共享连续场；r05 是己方回收目标链；r06 是统一 prestock 准备机制；r07 是阶段回放分配。后续轮沿用前轮所有已安装因素。

## 较有效的手段及其证据强度

- **整合路线训练产生局部能力**：相同固定协议下，父点 r01 g0 real_drop 为 0/16、natural full 2/16、far 8/16；r01 g8 为 8/16、6/16、14/16。near/mid 仍为16/16。这是联合配方下的观察；Hunter hard 从2/64变0/64、Tactical17/64变12/64，存在保持退化。
- **先验证原生端点、再修 teacher**：r02→r03 的改动直接排除原生负进度方向并保留安全等优集合，r03 g8 sugar7/16、drop13/16、full8/16，但末点分别5/16、9/16、4/16；标签正确性修复没有自动带来稳定长期提升。
- **连续场修复有明确数值依据**：边界连续、格中心保持、断连对角、INF、原生 steering 居中等 focused 测试支撑 r04 修复；r04 final drop13/16 高于父点9/16，但 full4/16相同、far13/16降9/16、Hunter总量16/192降8/192。不能把某列改善称整体成功。
- **目标链一致性是必要机制修复**：r04 capped g1..16 冻结证据有35条 own-recovery 标签面对 stock PBRS、2条 return 面对 PBRS NONE；r05 统一链后终点 full7/16（父4/16）、sugar6/16（父3/16），但 drop13/16降10/16、Tactical10/64降8/64，原墙交付仍未解决。
- **阶段计数能约束后续试验**：r06 g64 FIFO opening 仅60条、greedy11/60、accepted-pair mass均值 `.187832`；g128 opening56条、17/56、mass `.330983`，return4032条、1906/4032、mass `.476660`。全轮 opening accepted1924/query1032，而 navigation accepted750467；支持 r07 测试回放份额，不能据此证明梯度竞争或因果归因。

以上固定16局/64局为反复查看的探索性样本，未用独立大样本确认，不提供显著性或跨地图泛化结论。网页 r01 g8 的“综合最佳”是**当时37个已完成候选**按零观测自伤优先、四项几何均值折中选出的发布点，不是对当前所有 checkpoint 重新全局搜索后的结论；本次不更新模型、Release 或录像。

## 当前能力边界、失败与退化

**取到包不等于走得回来。** r06 g128 natural full 的拾包4/16、交付1/16；窗口→目标7/7只是 episode 共现，首方向3/5、持续758/1145、进度785/2105、恢复切换11/14。drop 首方向16/16、拾包16/16，仍只交付5/16；因此只报“首步正确”会漏掉主要失败。

r06 final 原墙 death/sugar/recovery 三层分别拾包8/8、交付0/8；返程 target tick为2688/2087/2323，oracle机会0/187/0。death 原墙的死亡已为0仍交付0，说明存活率不能替代路线能力。其8局和开墙team1的8局绑定，不能把差异完全归因给墙体。

r06 prestock 机制没有稳定改善固定路线：natural full 从父7/16降到1/16、far13/16降7/16、recovery5/16降0/16；Tactical终点8/64升17/64也不能覆盖这些退化。g96 Hunter normal曾有自伤2/64；终点自伤零不代表整轮无安全回落。

r06 末状态8192个 learner rows 中 prep_active428、ready101、preparing327，其中 staged13、clear8、long3/place2；这是 rollout-end witness 前提分布，不是全轨迹覆盖或安全接受率。FIFO分布随 learner 改变，greedy fit 和 mass 是 in-sample 拟合，不是固定 holdout；没有 exact Actor input 相同的冲突，不等于总体无冲突。

r06 全轮 control gross343451.75/net258465，route正72225.692/负−134363.487/net−62137.796，终局capture项3606192；这些是训练 learner trace 累计奖励。full 内 own_window197189、own_kill132083、capture61534，在200966768 full有效ticks上分别可归一；capture为306.190次/百万full ticks，绝不是JS完整局交付成功率。训练 bot、弱对手和长局分布与固定评估不同。

固定 Review seed2026101800 反复暴露返程错误：r06 final drop 动作前81步向左，当前场 gain−.257496，而右+.337502/下+.237499；nearby FIFO同terrain/base/origin且L1≤1只有14条、exact Actor input匹配0。它支持执行/分布缺口，不证明同输入监督矛盾，也不能证明训练从未访问。旧 JS 某些“偏航”经新连续场复算为合法最优，维护时必须保留两种标签。

技术失败要单列：v37 初次尝试因合法库存拾取未延续 custody，在正式更新前判无效；第二次在数值smoke后遇宿主变量覆盖/日志流程错误，正式更新为0；attempt03 因原生接触残余来源遗漏与受控死亡重复支付16而技术中止。它们不能用于能力比较，也没有成为 v38 的父点。测试接线修复比纯ledger单测更重要。旧 baseline_prefetch 的 INVALID_FIXTURE、旧 v1 的 INVALID_METRICS 不进入固定结果表。

## 后续方向

先按 r07 的实际每半回放计数、opening/return fit、固定路线推进与战斗/安全保持检验阶段分配机制。current/history fallback 和重复标签必须如实计数，不能把64个回放槽称64个独立状态。当前后备 r08 已预声明同配方、同128代预算、父r07最终g128 raw Actor/Critic/完整pool，train seed2176341172、technical2775223227；这是声明，未运行，允许基于完成点另准备最小可验证配方。

再分别检查取包前的通路准备覆盖、同原生状态目标/安全/Actor可观测性、返程执行缺口与真实强对手暴露。优先针对固定失败状态做机制诊断，不一次同时改 mix、query、奖励系数和 Actor 输入；若需要因果结论，应另设同父点对照和独立 seed/holdout。跨地图、双席位解混杂和大样本确认仍需独立实验。文档工作不安装配方、不触碰运行中的训练源或进程。

## 已纳入代码、依赖闭包与发布缺口

[逐文件源码清单](../reports/je10_source_manifest_20261018.json)记录每个文件的 source/published SHA-256、直接依赖及 required_by；[当日报告的集成清单](../reports/je10_training_20261018.md#源码集成与验证)提供逐文件阅读入口。源码先双次只读比对稳定字节，对在r07 seal中的文件核对封印，再复制。四项历史测试文件调整为当前已安装配方的验收，另将跨测试文件的scalar fixture内联，清单保留原哈希以追溯；数值源码没有因文档集成改变。Review的历史审计测试补充了版本区分：固定v35的H5c源码身份保持原哈希，当前JE10文件另以本次manifest校验；录像、审计JSON和模型未改。

纳入 JE10 credit/route oracle/ops、continuous/native certificate、own-recovery alignment、prestock目标/原生teacher/joint FIFO、phase-balanced replay、JS Hunter/control/route漏斗、完成点分析/画图和证据摘要工具，以及它们直接需要的 JE04–JE09 ops、scenes、环境等模块。旧命名是当前 import 依赖，不表示把旧实验脚本批量发布。生产JS simulator、Hunter和Review沿main已有版本；Review协议使用main已发布版本，不移植训练工作区的网页脏改动。`web/sim.js` 中 try/catch 的可选 TimeAStar 插件不参与Bun Hunter，不属于必需闭包。

**没有纳入完整训练启动/交接链** `je10_worker.py`、`je10_campaign.py`、`je10_supervise.py`、Python异步owner `je10_evaluate.py` 及各轮源码改写 installer。它们的 recipe/authority/bootstrap 链继续依赖未发布的历史编排脚本、绑定旧namespace/训练资产的副作用和外部输入；把孤立入口放入main会暗示并不存在的完整可启动性。公开的是可导入数值组件、可独立调用的JS评估和只读诊断，**不是开箱即跑的128代训练系统**。Python Tactical的冻结结果可审计，JE10全套自动评估owner不能由本次main独立重建。

完整续训还需在独立训练环境准备可信父Actor/Critic/完整pool、g0/stage1/retention引用、return replay、regions、封印plan及每轮saved-source，并恢复checkpoint中的prestock/FIFO/对手/custody/RNG等完整状态。只用policy权重无法恢复训练。旧源码checkpoint的验证必须使用对应saved-source，不能拿当前源码替代。run目录、权重、缓存、锁/PID和临时产物均不进Git。

## 复现与长期维护

在干净checkout安装项目依赖（JAX需匹配CPU/CUDA环境，曲线额外需要matplotlib）。本次验证全部在本分支使用CPU，没有启动正式训练或固定对局评估owner。

```bash
# 不导入训练，不加载权重：核验本次逐文件闭包
python3 scripts/verify_je10_source_manifest.py

# 从已冻结摘要重生成全表、状态块和账本（不访问训练源）
python3 scripts/render_je10_training_report.py

# 重画已完成轮曲线（此例r06，需要matplotlib）
python3 scripts/plot_control_route_curve.py \
  reports/je10_curve_data_20261018.json <OUTPUT_DIR>/r06.png --round 6

# 只读抓取新的COMPLETE证据；输出必须在只读source tree之外
python3 scripts/summarize_je10_evidence.py \
  --source-root <READ_ONLY_TRAINING_TREE> \
  --output <OUTPUT_DIR>/je10_evidence.json

# CPU导入/focused检查（需先安装pyproject声明的依赖）
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAXBOMB_RULE=bun \
  python3 -m pytest -q tests/test_je10_control_credit.py \
  tests/test_je10_native_credit_wiring.py tests/test_je10_custody.py \
  tests/test_je10_route_oracle.py tests/test_je10_evidence.py \
  tests/test_phase_balanced_joint_replay.py
node tests/je10_control_funnel_test.js

# 具有可信已导出模型时才运行；写入独立输出目录，不指向训练目录
node scripts/je10_route_funnel.js <EXPORTED_MODEL_JSON> \
  <OUTPUT_DIR>/route.json <REGIONS_JSON>
node scripts/je10_web_hunter.js --model <EXPORTED_MODEL_JSON> \
  --difficulty hard --spawns <FROZEN_SPAWNS_JSON> --start 0 --end 8 \
  --seed 20261001 --checkpoint-sha <POLICY_SHA256> --out <OUTPUT_DIR>/hunter.json
```

维护时先确定新的截至时间，读取最新 STATE、LATEST/COMMITTED、RUN_TERMINAL、SUCCESSOR_RECIPE/NEXT_RECIPE_PROPOSAL、固定 SUMMARY/COMPLETE、训练报告和账本；完整点必须逐shard验哈希，active STATE只能标记采样状态。更新快照、当日后缀报告、完整曲线和本页状态/方法表，同步账本索引。不要复制仍由writer修改的报告或partial；用冻结数据重新生成归档。源码有变化时重新核验封印与逐文件闭包并运行focused检查，不能只重写hash掩盖行为变更。机器路径、用户标识、内网IP和机器PID禁止进入提交。若外部冻结产物已不在，已提交摘要仍可核对数字和出处哈希，但不能独立重新执行原训练或重建全部逐局轨迹。
