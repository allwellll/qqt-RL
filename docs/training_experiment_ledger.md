# 训练优化实验总账

> 2026-10-28 集成说明：保留 main 与来源分支的全部实验记录。下文“当前／正在／已暂停”为各条目记录当时的状态，后续状态以较晚日期条目为准。

<!-- JE10-LEDGER:BEGIN -->

### JE10 v38 击杀信用与搬包路线（2026-10-18，持续研究）

证据截至 **2026-10-18 13:33:24 UTC+8**。长期解释与复现边界见[JE10训练总结](je10_control_route_training_summary.md)，全部固定点/曲线见[当日报告](../reports/je10_training_20261018.md)，哈希及分母见[冻结摘要](../reports/je10_evidence_20261018.json)。

本条选择性整合训练账本的JE10方案与逐轮终态，保留本账本既有实验。网页当前发布仍为r01 attempt04 g8 raw；最近完整轮为r06 g128。

正在训练r07：STATE g68，COMMITTED g68；最新完整固定评估g32。运行状态与固定能力分列，不把训练事件当泛化。

r01整合首控4/主动残余.25、受控legacy16去重、连续PBRS .5、set-valued learner-state DAgger .2及完整custody；r02只改reset mix为.35 combat/.20 bridge/.35 full/.10其他原生路线；r03 native端点progress set；r04共享连续场；r05己方回收目标链对齐；r06统一prestock corridor与joint teacher；r07分阶段current/history回放。每轮8×1024×256、128代/268,435,456 fresh ticks，g0/4/8/16/32/64/96/128固定协议不变。

r06终态natural full1/16、drop5/16、far7/16、recovery0/16，对比父点7/16、10/16、13/16、5/16均退化；Tactical从8/64升17/64，Hunter难/中/易1/3/11（各64）。原墙库存三层pickup8/8仍delivery0/8；不使用战斗末点或JAX训练事件覆盖路线失败。阶段回放是后续诊断假设，顺序迭代不构成同父单变量因果。

按用户覆盖策略，JE10不设置能力/admission/promotion门禁；中间诊断不改变健康固定预算或阻塞后继，技术有效性另查。r08是同配方同预算连续后备预声明，尚未运行；具体下一因素须由完成点证据支持。完整启动/历史bootstrap和外部资产未入main，公开源码边界见总结，不能假称开箱可续训。

v37技术无效attempt、INVALID_FIXTURE/INVALID_METRICS不纳入v38能力表。网页发布选择、最新完整轮与进行轮不得混称综合最佳。

<!-- JE10-LEDGER:END -->

> 2026-10-18 更正：r11 g18已EXCLUDED/INVALID。旧“真实取包后无路／动态封路／预开返程通道”推断全部撤回；其末状态分母未验证真实取包链。当前r12_causal_clean从r10 g128全新初始化，删除无事件learner携包并保留合法返程DAgger×2；详见文末因果初始化修复记录。

### JE10 v38 原生信用接线修正后的整合长训（2026-10-18，r01–r03固定终态，r04正式8卡训练中）

实验 ID：`JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r01-attempt04`；namespace：`runs/je10_bun_20261018/control_route_v38_r01_attempt04`；父点沿用已发布JE09 Stage C g8 raw Actor/Critic/pool，选择依据见下方v37审计，不继承技术无效v37 checkpoint或旧v36 WIP。train seed `353284308`，technical seed `3378785851`，seed按实验身份+role派生。固定8×1024×256，128代/268,435,456 fresh ticks，128 updates/代/设备；reset抽样权重40%combat/25%bridge/35%full（不是有效tick占比），其他预置路线scene独立质量为0，地图仅Bun806原墙/开墙。保持PBRS .5、set-valued learner-state DAgger .2、固定carry assist .1、原战斗保留项，仍是整合配方，不声称单变量能力归因。

v37生产集成错误修复：native contact popper通过明确`native_trap_popper`进入残余ledger；受控死亡不再积累legacy settled kill16，仅已有首次控制4与主动残余.25，单局gross cap8、死亡clawback不退款；非受控直接战斗击杀16保留，交付24保持主导。新增4项RED复现native接触来源/超时/受控去重/直接战斗保持，连同原回归39项GREEN。真实8卡数值smoke已启动，有限非零更新及保存恢复通过后立即转正式128代，不等待CPU、网页、报告或人工批准。

**无能力/admission/promotion门禁，持续迭代**：仅导入、focused TDD、真实非零更新、NaN/OOM、checkpoint保存恢复检查技术有效性；短点无改善不停止固定run。仅技术故障、硬安全恶化或用户停止可中断。固定128代终态先以`nvidia-smi`核验8卡compute client全部释放，立即启动预声明`control_route_v38_r02`（同128代、继承完整最终raw Actor/Critic/pool）；随后独立确认后继正式g1 Actor/Critic有限非零、8副本一致和checkpoint hash，再主动报告。r01已完整完成g128/268,435,456 fresh ticks，g1–g128全部finite/nonzero、未提前停；compute4751.549秒、wall9949.416秒。8卡compute clients释放凭据已核验，r02已通过最小8卡技术smoke/fresh restore并正式训练，g1已真实非零更新。r01固定g0/4/8/16/32/64/96/128异步评估全部COMPLETE。终态后CPU旧入口的当前源码seal检查失败已保留；独立saved-source恢复核验127份源码、完整state及next RNG通过，不影响训练。禁止以能力曲线影响训练预算。

跨run唯一CPU authority通过`runs/je10_bun_20261018/evaluation.lock`串行，后继GPU不等待该锁；当前owner已验证global_unique。异步唯一CPU/JS owner、锁与marker沿`je10_evaluate.py`：`authority_je10/evaluator.lock`、`OWNER.json`、`QUEUE.json`、每点`COMPLETE.json`和最终`EVALUATION_COMPLETE.json`；g0/4/8/16/32/64/96/128，恢复checkpoint每4代。Python Tactical v2 64、真实production JS Hunter hard/normal/easy各64，seed20261001/冻结64spawns/greedy/max300+64；严格区分JAX训练bot。固定路线8tasks×16seed，详见下方v37完整协议；各点报告机会分母、父点/前一点变化、保留指标、current/ring/fixed底层tick与实际弱课程暴露。冻结Hunter快筛保留原协议native_trap=false/native_items=false，糖泡机会不适用；糖泡链取独立原生route/full，禁止零机会混算。被动observer只绑定实际Actor推理sim，排除bot规划克隆，kill计数与权威结果parity。固定控制漏斗含原控制施害归因、final_pop、active_contact_pop、持久因果击杀/存活，并单列production native causalKill；expiry-tick接触的主动计数保守缺失，不混作训练奖励。

Review是唯一录像诊断入口，可在现有网页导入每task固定首seed的`REVIEW_FIXED_<task>.json`；精确reset-fixture与完整重放hash校验。动作/状态hash/事件/阶段目标势能及明确标注的JS重建reward-GAE近似，不能冒充训练奖励或反事实Q。真实learner首偏航保留`FIRST_DEVIATIONS_*.json`，hash范围明确为observation/global。评估产物在run中，不入Git。

迭代规则：全段曲线及事件机会归一化→奖励占比→首次方向偏离→oracle冲突/覆盖→DAgger learner-state实际分布→势能分辨率/阶段→战斗安全保持→对手暴露，证据排序后最小改动；不佳继续训，改善补最弱漏斗并防遗忘。候选修改不能作为停训门禁，也不热改运行训练源。正式g1真实Actor/Critic更新已复核finite/nonzero、8副本exact及policy hash。r01 g1–g128均通过finite/nonzero；g0/4/8/16/32/64/96异步完整评估已完成，最终g128也已COMPLETE，无failed shard；CPU唯一owner已承接r02。

后继预声明补充（只替换行政owner，不改运行中r01配方/源码/计划）：固定r01完成后先启动r02：父点r01最终raw Actor/Critic/pool，预算同128代；唯一数值因素为reset scene mixture `.35 combat/.20 bridge/.35 full/.10 other native routes`。随后fresh restore和训练g1正式finite/nonzero确认。后继owner在前run终态后安装host-only偏航hash单行提取、verifier audit和最终动作实际source provenance；绝不热改r01。r01曲线显示路线局部改善但Hunter/Tactical保持弱，且奖励内control/route/终局分别拆分；当前仅完整g64证据见当日报告。r02之后的改动会依据全曲线与解混杂诊断独立选择，不把r02混合比例误写成能力晋级门禁。

### JE10 v38 r02 后继（2026-10-18，固定128代已完成）

实验 ID：`JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r02`；namespace：`runs/je10_bun_20261018/control_route_v38_r02`。父点r01最终g128 raw Actor/Critic/pool，train seed `1881716240`、technical seed `1618573730`，均按新身份派生。8卡固定128代/268,435,456 fresh ticks；唯一数值配方改动为reset场景混合`.35 combat/.20 bridge/.35 natural full/.10 other native routes`，补零独立路线初始化质量；实际tick分布另计。reward/PBRS/DAgger系数、query预算、网络/PPO保持；终态后安装的单行hash/verifier audit/实际对手provenance只作诊断。smoke只查数值更新与保存恢复，通过后立即formal，不等评估。现已正式g1/2,097,152 ticks提交，Actor/Critic更新L2=3.584486/2.357491，finite/nonzero、8副本exact及policy hash核验，正式非零marker已写；监督继续至128代。

异步沿固定g0/4/8/16/32/64/96/128，每4代保存恢复点；Python Tactical v2与JS Hunter各64及8路线各16，仍同冻结协议。全局唯一CPU锁承接r01最终点，不阻r02 GPU；COMPLETE后才消费。预声明r03在r02固定终态、8卡释放后立即继承g128并启动同128代，无能力门禁；后续具体最小配方须在终态前完成proposal/TDD并另封印，不能热改健康run。

诊断口径校正：H6内部将label第8列改为selected，因此既有audit九列实际是`eligible_active → selected_for_verifier → safety_and_geometry_accepted → first_deviation_query`及`selected_rejected`、四个accepted phase。r02宿主字段中`geometrically_reachable`/`reachable_rejected`是沿袭误命名，须按上述含义解释；真正几何候选覆盖未独立记录，保留null，不将抽样当可达性。本run数值源码不热改；后继宿主命名修正另记。

### JE10 v38 r03 原生一步oracle修正（2026-10-18，固定128代已完成）

实验 ID预定`JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r03`；父点r02最终g128 raw Actor/Critic/pool，固定8卡128代/268,435,456 fresh ticks，train seed `1639578875`、technical seed `3501906343`（新身份+role SHA256派生）。reset混合保持r02 `.35/.20/.35/.10`；唯一因素为严格路线oracle改用已存在的五个native一步端点，选所有合法安全的最大正连续进度集合，保留goal wait，INF/无进度/拓扑变化拒绝。不增env.step、不改credit/PBRS/DAgger系数或query预算/网络/PPO，不接管learner。证据是固定Review g8错误等价动作及g32遗漏等优动作，focused候选/边界/交接20项RED→GREEN，尚未进入r02。原墙返程零可达oracle仍需独立开路阶段，不声称本改动解决。异步checkpoint/owner/marker与固定快筛沿r02，r02健康固定run照常完成、核验8卡释放即启动，不等最终报告/能力阈值。r01全8点曲线及证据排序见当日报告末节；r02曲线继续诊断并准备后续最小改动。

### JE10 v37 首次受控信用与路线整合（2026-10-18，技术无效尝试已原子停止）

以下保留v37原始预声明作为历史。attempt03最终在g3/6,291,456 ticks因native信用接线错误原子保存后技术中止，不作能力实验结论；旧v37自动后继已撤销，当前唯一GPU主线是上方v38。未恢复v36 stash。

实验 ID：`JE10-CONTROL-ROUTE-INTEGRATED-20261018-v37-r01`；历史技术无效namespace：`runs/je10_bun_20261018/control_route_v37_r01_attempt03`。初始`control_route_v37_r01`因新增RED发现reset-window库存拾取后custody断链，首更新前终止，仅技术失败，optimizer_updates=0；保留其g0/源码/日志。有效attempt02已修复，独立seed/cache。父点选择已发布 `web-bun392-je09-stage-c-g8-v1` 对应 `runs/je09_bun_20261015/stage_c_return_set_pfsp8_v8/formal/bun392_bridge_pfsp8/gen000008` 的 raw Actor、匹配 Critic 和完整 pool，fresh Adam/RNG/env。依据：受控掉包交付主评15/16、holdout13/16，Tactical17/64，Hunter hard/normal/easy2/3/29（各64）；已发布 transport1536 Hunter3/1/32、carry15/16，但缺少同级受控完整返程证据；v36 full首拾/交付0/16、hard1/64。优先保留g8返程，承认hard弱项和小样本局限，不声称父点全面最优。旧v36 WIP stash不恢复，不启动其他代理。

**整合配方，非单变量因果实验**：首次正常→原生糖泡受控4分，主动爆破残余0.25分；episode/victim/life/entry稳定身份、唯一施害者、edge去重、队友/自伤拒绝、死亡扣回、总gross上限8且扣回不退款；原终局交付24主导。连续拓扑PBRS λ从.05改为.5；DAgger .05→.2，采用合法oracle集合概率总量损失，覆盖learner真实访问scene1..9、完整持包生命周期，不再120tick截断；按目标阶段独立首偏航查询，危险状态拒绝，敌人恢复后空手关闭运输，持包遇近威胁切换。Actor输入/实际采样动作不被oracle替换。原生地图集只有Bun806；本轮包含原墙体/开墙、多出生及距离桶，不能声称已完成跨原生地图泛化。

固定预算：8×1024环境×256tick/代，128代/268,435,456 fresh ticks，128 Adam updates/代/设备，train seed `3399754560`（SHA256实验身份+role前4字节大端）；技术seed独立。40% combat、25%真实drop bridge、35%自然full；carry assist固定.1，评估关闭；真实场景比例、弱课程比例和league实际暴露按tick记录。current每代更新，ring每16代轮换，fixed六历史成员保留。网络/PPO/GAE沿已有生产配置，γ=.995、λ_GAE=.95；完整plan与源hash在run封印。

**无能力/admission/promotion门禁，连续迭代**：仅导入、focused RED→GREEN、真实非零更新、NaN/OOM、checkpoint保存恢复最小技术smoke；不等CPU评估、网页测试、报告或批准。短checkpoint不改善不能停；只有技术故障、硬安全恶化或用户停止可中断。固定run结束先启动预声明`control_route_v37_r02`（继承r01最终raw Actor/Critic/pool，同128代固定预算），再做最终restore、汇总和报告；后继可按同规则接r03。健康训练不因诊断暂停，不用旧行为smoke门禁。

异步协议：独立唯一CPU/JS owner `scripts/je10_evaluate.py`，锁`authority_je10/evaluator.lock`；跟随g0/4/8/16/32/64/96/128早中晚最终点；逐点`COMPLETE.json`，队列`QUEUE.json`，全部终态`EVALUATION_COMPLETE.json`。每checkpoint固定统一seed20261001、greedy、相同双seat/出生、max300+64危险残留结算：Python Tactical v2 64、production `web/sim.js` + JS Hunter hard/normal/easy各64，严格区分JAX训练bot。固定路线seed2026101800..2026101815，死亡敌笼/原生糖泡敌笼/真实掉包/near-mid-far返程/恢复威胁/自然full2400，各16；原机会分母、阶段转化、首oracle动作、持续推进、拾包/返程/交付及安全独立报告。每点相对g0父点及前一点变化；不可将不同simulator列混算。动态对手current/ring/fixed实际tick及比例随训练记录。

固定能力漏斗：首次糖泡归因→主动爆破→击杀/存活；死亡/受控窗口→目标选择→首个oracle一致动作→持续推进→拾包→返程→交付。诊断顺序固定为信号可达性/事件机会、奖励占比、首次偏航、oracle覆盖冲突、DAgger learner-state分布、势能分辨率/相位、战斗保持、对手暴露。唯一Review诊断链使用固定seed次序的首个失败，不择优，保留动作/状态hash/事件/奖励分量/阶段目标势能及明确标注的零值函数reward-GAE近似优势分解；不得冒充反事实Q。训练真实首偏航片段保存`diagnostics/FIRST_DEVIATIONS_*.json`，网页完整状态hash须经production Review重放补齐。

迭代决策：固定run完整曲线、机会归一化漏斗、首偏航和DAgger覆盖/战斗安全保持共同排序根因；最终不佳时提出下一轮最小可验证改动、focused TDD→技术smoke→有效长训；改善则补最弱漏斗并保持已学能力。默认同配方后继维持GPU，变更另设实验身份/封印，不热改运行中代码。每个固定run终态、技术阻塞和完成主动报告。关键结果更新`reports/je10_training_20261018.md`。

进展：信用12项、路线/oracle9项、真实custody集成1项、既有PBRS回归11项及宿主worker回归2项，共35项。attempt02真实8卡smoke g1已完成2,097,152 ticks、128更新，Actor/Critic更新L2=5.65596/2.62807、finite/nonzero、8副本exact，checkpoint已提交；其后宿主诊断数组覆盖预算变量导致退出异常，正式更新为0。attempt03只修复宿主变量命名与JSON日志写入顺序，复用该有效数值smoke并完成fresh CPU restore后已启动正式8卡128代训练；不重复数值smoke，不以能力作门禁。CPU/JS评估独立owner已启动；正式g1及评估结果待回填。原attempt02与旧stash完整保留。


本轮覆盖限制：除combat/bridge/full外的预置路线scene目前独立采样质量为0；oracle支持scene1..9不等于实际覆盖所有初始化课程。将按真实learner-state机会与首偏航证据决定后继是否最小调整场景混合，当前健康run照常完成。

### JE09 Route Potential PBRS H8 g32（2026-10-17，已完成）

实验 ID：`JE09-STAGE-H8-ROUTE-PBRS-CONTINUOUS-20261017-v36`。首次技术 attempt namespace `runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36` 因8卡 smoke OOM终止，`optimizer_updates=0`，只保留失败证据，不作为训练结果。有效独立 attempt02 namespace `runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36_attempt02` 从不可变 H8 v34 g32 raw Actor/Critic 与完整对手池启动，不从 v35 g32 续训。唯一变量为将既有严格路线 PBRS 的整数格距离替换为连续位置到当前格/四个轴向可达候选格的生产加权路径距离；固定 `lambda=0.05`、`gamma=.995`，其余阶段、身份、offset、反刷、安全、课程、PFSP、DAgger、网络、PPO/GAE、seed体系与评估协议保持。固定32代/67,108,864 fresh ticks，g0/4/8/12/16/24/32；正式更新 finite/nonzero，8副本一致，fresh CPU restore通过。聚焦连续格、格边界和生产地图语义测试通过；训练墙钟1,918.95秒。每个checkpoint同协议544局，逐点评估结果和任务计时见 `reports/je09_bun_20261017/route_pbrs_training_20261017.md`。

v36训练事件汇总：32代 `route_shaped_reward=-1,250.406`（正项+2,336.445、负项-3,586.851；combined_total=2,792,851.469，净route约-0.0448%），`route_potential_delta=-25,008.114`；topology gain 18,326、INF→finite 150,386、phase transition 348,288、invalidated 63,429、terminal correction 97,321、安全拒绝23,845。mean reward从g1 0.07427降到g32 0.03815，中间最低g9 0.03725。训练日志没有记录原始任务优势与路线塑形优势的逐样本/逐checkpoint对照，优势变化证据为“未记录”。

结果：v36 g32 fixed32主评/holdout均首拾与pickup-return 16/16、capture均0/16；1200 tick受控drop-return首拾/返程16/16，但25%/50%/己方半场/capture均0/16。v35 g32同drop-return链也首拾/返程16/16，25%=7、50%=6、己方半场=6、capture=2（均/16），故v36未保住其返程后段优势。v35/v36的自然full2400各checkpoint首拾/capture均0/16。统一carry拓扑诊断中v35→v36 g32 near=16→16、mid=16→16、far=2→1（每桶16局）；这属于预置持包能力，不是自然取得敌包。战斗：Tactical surviving causal kills v35→v36为9→15/64；Hunter hard/normal/easy为1/5/9→1/8/10（各64局），相较H8父点9/8/10，v36 hard仍低。路线开墙有效收益未证明：route-break任务16局未形成可归因的稳定最短路收益。综合结论：连续势能实现进入训练，但没有改善自然完整对局或固定限时交付，drop-return后段和carry_far相对v35变差；Tactical/normal/easy较v35回升，hard Hunter弱于父点。保留代码和真实回放用于分析，不据此宣称任务解决，也不自动续训。详见当天报告与 `web/replays.json` 中的g32固定seed真实自然对局录像。

实验 ID：`JE09-STAGE-H8-ROUTE-PBRS-20261017-v35`。独立 namespace：`runs/je09_bun_20261017/stage_h8_route_pbrs_v35`；父点固定为 H8 v34 `formal/bun392_bridge_pfsp8/gen000032`。只读父点 SHA256：`policy.pt=b491758761b32872555555f08cf4e42f9cab120215501509a2bdbf905e64ce11`、`state.pkl=f2fcf9c44401b924776e7e7e990bceebe1660fb1397de30d1476416766716ec7`。Control 为该父点原奖励历史评估，不启动第二训练臂；Treatment 唯一新增变量为严格路线 PBRS。网络、PPO、GAE、gamma=.995/GAE lambda=.95、课程、PFSP、DAgger、对手池和评估协议冻结；fresh Adam/RNG/env，正式固定32代/67,108,864 fresh ticks。

按用户最新指令固定 λ=0.05，不做系数扫描；行为效果和路线信号幅度不作为正式启动或固定预算停止门槛。新 experiment/run/cache/train/eval seeds 派生并与历史身份互斥。smoke 仅验证 8 卡生产更新 finite/nonzero、8设备 exact、DAgger非eligible污染为0、fresh CPU restore与GPU exact replay；仅技术故障可停止固定32代预算。正式训练启动后异步观察既有 assist-off 协议，完整预算结束后评估路线专项、full、返程与Hunter安全。

2026-10-17完成回填：代码/配置锁定 λ=0.05，未做系数扫描。聚焦测试 `tests/test_je09_route_pbrs.py` 7/7通过。真实8卡 smoke g1=2,097,152 fresh ticks、128 updates，Actor/Critic finite且非零、8副本exact，DAgger noneligible label=0；fresh CPU restore及fresh GPU replay下一更新state digest exact。正式固定32代/67,108,864 fresh ticks完成，32×128 updates finite/nonzero，g32 Actor/Critic更新L2=2.61813/2.91340，final CPU restore通过，GPU释放。训练墙钟约1,909秒；异步assist-off评估g0/4/8/12/16/24/32全部通过完成，无失败；每点16局/每协议分母见当天报告。

Reward事件实际汇总：`route_shaped_reward=-1,440.96`（累计总combined reward 2,847,457.34的约-0.0506%）；正分量+3,281.75、负分量-4,722.71，32代均有非零route reward。路线阶段事件347,916、拓扑有效gain18,608、INF→finite 151,535、安全拒绝23,438、目标失效64,589、terminal归零94,235。mean reward从g1 .07456降至g32 .04178（中间最低 .03677）；路线奖惩并非单向正向密集优势。

七个checkpoint固定32-step主评/holdout均toward/arrival/first_pickup/pickup_return=16/16。capture主评/holdout仅g24为8/16，其余g0/4/8/12/16/32为0/16；1200tick drop-return诊断g32为首拾/返程16/16、25%=7/16、50%=6/16、己方半场6/16、capture2/16。carry_home 16/16；near-steal 14–16/16。full2400七点首拾/capture均0/16。route-break 16局仅1局出现有效safe_win，开墙并未形成稳定收益。7点每点Tactical64局、Hunter hard/normal/easy各64局；g32 Tactical surviving causal kill 9/64，父H8 v34 g32为12/64；g32 Hunter hard/normal/easy surviving causal kills=1/5/9，对照父点=9/8/10（各64局），own-bomb defeat均为0，双方评估未见严重自炸或死亡计数。但安全存活不能抵消攻击效能明显回落。

固定协议g32相对父点：受控drop-return进度由父点25%=1/16、50%=0、capture=0升至7/16、6/16、2/16；fixed32两边capture仍0/16，唯一g24中间点主评/holdout均8/16后g32回落至0。full仍未改善。结论：路线PBRS信号已真实进入production reward/GAE/PPO且累计含正负项，但当前λ=.05设置没有稳定提高固定限时交付或自然full；路线训练reward净负且Tactical/Hunter hard攻击保持回落。值得保留实现和回放用于分析，不建议按此单变量配置直接续训；按用户要求不自动启动新训练。

真实网页模拟器回放包位于 `reports/je09_bun_20261017/route_pbrs_replay_g32/`：包含g32 Actor JSON/ONNX/raw policy、从同代state精确导出的Critic-only NPZ、配置、真实production `web/sim.js` 固定seed 1,200 tick状态/动作/route轨迹及独立PBRS算术复算证明。此轨迹使用明确声明的drop-return诊断fixture，不代表自然full成功。完整结果与逐checkpoint分母/耗时见 `reports/je09_bun_20261017/route_pbrs_training_20261017.md`；最终网页侧交接见本机 `.hermes/route_pbrs_final_handoff.md`。

2026-10-17网页追加诊断：固定真实seed `2026101732` 的tick177不是有效返程，而是上下周期2振荡；位置为 `[y,x]=[1.80,9.64]→[1.98,9.64]`，move1向下，下一tick以55.92%选择move0向上（向下11.74%）。整格距离38、Phi=.421875不变；连续向下两步可降距37，无碰撞校正或目标错位。tick177连续61tick未下降，保守确认振荡143–590；完整窗口147–207、1200动作重推、244合法一步分支和来源hash见 `web/review-evidence/route-g32-return.json`。两代Critic存在，但该诊断外部Tracker因果预置与训练scene9 native归因、隐藏reward aux、terminal/阶段PBRS语义尚无严格同口径适配；原始/塑形Q/A及跨Critic差仍null，不能用路线reward冒充。未启动或改变训练；下一步需要完整版本化状态/info/ledger桥和独立反事实复算。页面增加失败区间、实际Actor概率/候选路径及缺测说明。详见 `reports/replay_review_20261017.md` 的tick177审计。原生winner0来自存包超时，敌包交付0，不是自然完整链成功。

> 本文档是训练实验的唯一长期索引。任何会改变训练分布、优化目标、对手、场景、样本规模、训练规模、超参数或评估协议的动作，都必须在实验启动时新增记录，并在结果产生后原位回填。聊天结论、run 日志和单独报告不能替代本总账。

## 1. 记录规则

### 1.1 每次实验必须记录

1. 实验 ID、日期、Owner、状态和变更类型。
2. 假设与唯一主要改动；组合实验必须列出所有改动及其已有单项证据。
3. 代码 commit、初始化 checkpoint、run 路径和完整关键配置。
4. 对手/场景/课程分布、环境数量、设备数量、训练步数或 iter、seed。
5. 固定评估协议：checkpoint、seed、局数、座位、出生、对手、最大步数和输出 JSON。
6. 训练吞吐与完整墙钟；失败重启、smoke 和正式训练分开统计。
7. 原始指标、相对基线变化、结论、是否晋升和下一步。

### 1.2 禁止事项

- 不得只写“效果更好”；必须附原始计数或率及评估产物路径。
- 不得把不同 seed、地图、对手、座位或局数的结果直接混算。
- 不得用训练分布内胜率代替跨对手泛化结论。
- 不得把场均放泡数增加单独解释为能力提升。
- 进行中的实验不得提前写成成功；未完成字段统一标为“待回填”。
- JE10起按用户明确覆盖策略：不设置能力/admission/promotion门禁。评估用于诊断、固定协议对照和后继配方设计，不阻塞启动/续训；旧实验中的历史门槛记录仅为当时事实。

### 1.3 状态

`设计中` → `smoke` → `训练中` → `评估中` → `完成/中止`。

## 2. 统一评估口径

当前统一快速筛选协议固定为四类对手，每个 checkpoint、每类对手 64 局：

- `bun.tactical_v2`：冻结 Python Tactical v2；
- `bun.hunter@hard`：真实网页猎手困难；
- `bun.hunter@normal`：真实网页猎手普通；
- `bun.hunter@easy`：真实网页猎手简单。

固定 seed `20261001`、相同出生、greedy、`max_steps=300`，逐类报告 surviving non-trade kill、被Bot击杀、own-bomb death、mutual death、W-L 与 bombs/game。网页猎手必须运行 `web/sim.js` + `web/bun_hunter_bot.js`，禁止用设备端 JAX 同名 Bot 代替。Tactical 与网页猎手是不同模拟器/协议，只能在同一对手列内比较。

JAX Bot ladder 可作为训练专项能力的旁路诊断，但不能替代上述四Bot统一快筛。64局按预声明协议提供快速诊断，不能推断统计显著性；JE10不设置晋升或能力门禁，也不等待扩样才续训。

## 3. 实验索引

| ID | 日期 | 主要优化动作 | 状态 | 关键结果 | 决策 |
|---|---|---|---|---|---|
| EXP-20260929-01 | 2026-09-29 | 危险监督、Reward 重构与安全 teacher 审计 | 完成（部分后续未回填） | 证明自炸主要是学习策略问题；旧 control 虽把自炸降至 0.8%，但放泡降至 0.43、可避免死亡升至 21.1%，属于被动崩溃 | 不晋升旧 control；安全提升必须同时保持进攻活性 |
| EXP-20260930-01 | 2026-09-30 | Reward V2、enemy-threat reward、逃生分析提速 | 完成 | 训练吞吐约 42.3k→121.2k SPS；短点可接近基线，但长训退化 | Reward V2方向保留；采用短训选点，不以最终点自动晋升 |
| EXP-20261001-01 | 2026-10-01 | 分难度设备端 JAX Bot 池 | 完成 | 新2500对JAX Bot平均p=0.939、强Bot p=0.878；历史it4000为0.658/0.190 | 专项能力显著提升，但需独立泛化门槛 |
| EXP-20261001-02 | 2026-10-01 | 出生位置多样化 | 已实现并进入长训 | 与Reward V2、Bot池组合进入8卡run；尚无单独消融 | 组合结果不可归因到该动作，后续需单项A/B |
| EXP-20261001-03 | 2026-10-01 | 8卡、2048环境、长训规模扩大 | 训练中/评估中 | 约20万SPS；截至15:09为12932/60000，最新完整ckpt it12500 | 继续训练；任何ckpt不得仅凭JAX Bot结果晋升 |
| EXP-20261001-04 | 2026-10-01 | 新增危险残局/随机场景族 | 实现完成，验证中 | 6类场景已接入；专项测试、4卡smoke和短A/B仍在进行 | 未评估，不得宣称改善泛化 |
| EXP-CX-20261002-01 | 2026-10-02 | GAE lambda .95/.99 | 完成 | 四 Bot 64 局快筛 0/24，最高 C .219 < 基线 .281 | 不晋升；延长信用窗口无稳定收益 |
| EXP-CX-20261002-02 | 2026-10-02 | 去除战术泡安全结算奖励 | 完成 | 64 局 0/24，最好 C .281 仅持平，normal 自炸超限 | 不晋升；减少退化不等于攻击转化 |
| EXP-CX-20261002-03 | 2026-10-02 | P1 current/frozen it4000 | 完成 | 64 局 0/24，最高 C .219 | 不晋升；冻结对手未消除网页负迁移 |
| EXP-CX-20261002-04 | 2026-10-02 | actor top25%/均匀采样 | 完成 | 64 局 0/24，最好 C .141 | 不晋升；均匀采样的相对改善仍低于基线 |
| EXP-CX-20261002-05 | 2026-10-02 | 单次临爆压力奖励 0/.4 | 完成 | 64 局 0/24，后期四格自炸上升 | 不晋升；早期相对改善不能掩盖末期安全退化 |
| EXP-CX-20261002-06 | 2026-10-02 | shared/detached 价值梯度 | 完成 | 64 局 0/24，攻击强度不足 | 不晋升；梯度干扰诊断不等于收益 |
| EXP-CX-20261002-07 | 2026-10-02 | 50% JAX hunter_hard 训练对手 | 完成 | 64 局 0/24 | 不晋升；JAX 强对手不能代替真实 JS hard |
| EXP-CX-20261002-08 | 2026-10-02 | 清砖训练几何 | 完成 | 64 局 0/24，自炸降低但 Tactical 攻击退化 | 不晋升；防御改善不能以攻击塌缩换取 |
| EXP-CX-20261002-09 | 2026-10-02 | 清砖后强规则对手占比 | 完成 | 64 局 1/24；原点 512 局通过，独立训练 seed 未复现 | 不晋升；单点通过只触发复训 |
| EXP-CX-20261002-10 | 2026-10-02 | 正向塑形尺度 1.0/.05 | 完成 | 64 局 3/24；原点 512 局通过，独立重训综合 CI 跨零且 Tactical 退化 | 不晋升；放泡增加不能代替击杀 |
| EXP-CX-20261002-11 | 2026-10-02 | 中等正向塑形 .3/.6 | 完成 | 64 局 3/16；原点和独立重训均未过 512 局 Tactical 门槛 | 不晋升 |
| EXP-CX-20261002-12 | 2026-10-02 | env/minibatch 同步扩大 4 倍 | 完成 | 64 局 4/16；原点 512 局通过，独立重训 Tactical 退化 | 不晋升；低采样噪声未带来跨 seed 稳定 |
| EXP-CX-20261002-13 | 2026-10-02 | mixed/native 出生分布 | 完成 | 64 局 5/16；原点 Delta C +.365，独立重训 -1.055 | 不晋升；重训四格自炸激增 |
| EXP-CX-20261002-14 | 2026-10-02 | hard/graded 规则对手难度 | 完成 | 64 局 2/16；原点/重训 Delta C +.254/+.156，重训 Tactical 下界 -.105 | 不晋升；综合分不能掩盖关键对手退化 |
| EXP-CX-20261002-15 | 2026-10-02 | Adam 前全局梯度裁剪 0/5 | 完成 | 原点/重训 512 局 Delta C +.238/-.049，重训 Tactical 下界 -.152 | 不晋升；单 seed 曲线不能证明泛化 |
| EXP-CX-20261002-16 | 2026-10-02 | 猎手相邻格反制泡预判 | 完成 | 64 局 3/16；原点 Delta C +.266，重训综合下界 -.008、Tactical -.174 | 不晋升；局部 JAX 机制未覆盖完整 JS 猎手 |
| EXP-CX-20261002-17 | 2026-10-02 | PPO 后导出 EMA .95 | 完成 | 原点 512 局综合 CI 跨零，重训 Tactical 下界 -.082 | 不晋升；EMA 不是收益保证 |
| EXP-CX-20261002-18 | 2026-10-02 | 固定两 seed 等权参数平均 | 完成 | 64 局 5/16；原点/重训 Delta C +.105/+.002，Tactical 均失败 | 不晋升；参数平均不能替代复训 |
| EXP-CX-20261002-19 | 2026-10-02 | 全参数/仅策略价值头适配 | 完成 | 64 局 6/16；原点/重训 Delta C +.197/-.107，重训 Tactical -.143 | 不晋升；冻结特征未稳定保留攻击 |
| EXP-CX-20261002-20 | 2026-10-02 | 存活击杀终局奖励 12/24 | 完成 | 原点/重训 Delta C +.252/+.102，重训 Tactical 下界 -.0586 | 不晋升；停止单向加奖 |
| EXP-CX-20261002-21 | 2026-10-02 | current/frozen 学习型 P1 | 完成 | 原点/重训 Delta C +.156/+.182，Tactical 均失败 | 不晋升；综合分不足以放行 |
| EXP-CX-20261002-22 | 2026-10-02 | 击杀奖励 24/48 | 完成 | 原点/重训 Delta C +.172/+.100，重训 Tactical 失败且 hard trade 18 | 不晋升；不继续加码 |
| EXP-CX-20261002-23 | 2026-10-02 | kill24 下 EMA .95/.99 | 完成 | 原点/重训 Delta C +.234/+.168，重训 Tactical 下界 -.0625 | 不晋升；逐格保底仍失败 |
| EXP-CX-20261003-24 | 2026-10-03 | 学习型 P1 sampled/greedy 解码 | 完成 | 原点 Delta C +.141；重训 -.260、Tactical 下界 -.09375 | 不晋升；更多瞬时威胁未转化为真实击杀 |
| EXP-CX-20261003-25 | 2026-10-03 | HL-Gauss 价值标签边界投影 | 完成 | 两次 512 局 Delta C +.205/+.156 均通过数值门槛；hard S 均 0/512 | 综合安全收益可复现；不能宣称 hard 攻击改善 |
| EXP-CX-20261003-26 | 2026-10-03 | 按对手分组等预算 top25% | 完成 | 64 局 global/balanced 3/8、1/8；原点 Tactical 512 局下界 -.05078125 | 不晋升；严格门槛未通过，不改选重训点 |
| EXP-CX-20261003-27 | 2026-10-03 | 后半 P1 固定为 CX-25 强 reference | 完成 | 64 局 current/strong-ref 2/8、1/8；原点/重训 Delta C +.145/+.031 | 不晋升；重训综合 CI 跨零、Tactical 下界 -.1328 |
| EXP-CX-20261003-28 | 2026-10-03 | 安全放泡按敌方逃路缩减加 margin reward .4 | 完成 | 64 局两臂各 2/8、全部 hard S=0；原点/重训 Delta C +.150/-.230 | 不晋升；原点属于 control，不能归因于新奖励 |
| EXP-CX-20261003-29 | 2026-10-03 | trap16 路线目标场与四 Bot attack trace | 完成 | 两次 512 局确认的 Delta C CI 均大于零；it101 快筛 0/4 通过，hard/normal 未改善 | 不按原配置长训；CX-30 门控也未提供延长证据 |
| EXP-CX-20261003-30 | 2026-10-03 | trap16 对手混合：50% vs 75% JAX hunter_hard | 完成 pilot | mixed-hard 8/16 保存点通过、3/4 seed 有至少两个通过点；hard S 仍为 0，normal S 仅少量出现，末端 Tactical 退化 | 不启动 401；不因快筛通过直接追加训练或 512 局确认 |
| EXP-CX-20261003-31 | 2026-10-03 | JAX/网页主动击杀机会与 45-tick 配对审计 | 完成诊断 | JAX hard 96 次安全放泡反事实均无 surviving kill；网页近距/逃路缩减事件也为 hard/normal 0→0 | 不训练；先寻找网页时间展开 forced 且自身可逃的单变量机制 |
| EXP-CX-20261003-32 | 2026-10-03 | 网页 forced 几何见证与 JAX reset window pilot | 完成，阴性 | 人工格位可见 52/52 机制见证；两条 21-update 模型真实 hard/normal 均 0/64，JAX 标签与网页时序不一致 | 不追加训练或 512 局；先对齐标签和正常出生可达性 |
| EXP-CX-20261003-33 | 2026-10-03 | JAX 对手梯度 ladder 对照 | 完成，阴性 | 两 seed ladder 的真实 hard/normal 均 0/64，Tactical 低于 control；代理 kill/death 信号未转化 | 不扩大对手梯度；保留为负结果诊断 |
| EXP-CX-20261003-34 | 2026-10-03 | trap-search 局部候选格诊断 | 完成，阴性 | 两 seed 真实 hard 均 0/64，normal 仅单点 1/64；无跨 seed 非换命击杀 | 不合入代理实现，不启动长训 |

CX-01 至 CX-28 的实验假设、唯一变量、证据等级和失败模式可按 ID 在[经验索引](cx_experiment_lessons.md)检索；关键机制诊断、复现与启动踩坑见[CX-01 至 CX-28 报告](../reports/cx01_28_diagnostics_20261003.md)。各轮原始 run 产物仍在隔离工作区，未纳入 Git；64 局结果均不能替代 512 局确认。

## 4. 详细记录

### EXP-20260929-01：危险监督与安全进攻联合修复

- 变更类型：Reward / 辅助监督 / 评估与teacher审计。
- 背景：学习策略出现自炸、被动和不逃生。
- 动作：danger特征进obs；actor auxiliary supervision；奖励只鼓励非trade击杀、逃生与安全结算，惩罚自炸、trade和可避免死亡；锁定Python/JS安全teacher parity。
- 关键证据：训练分布4747个存活态中逃生动作可用率100%，平均可生存动作8.91；规则Bot在清砖/带砖各128局均0自炸，说明主要问题不是teacher无路可逃。
- 旧实验对照（128局）：

| arm | surviving causal kill | own-bomb death | bombs/game | avoidable danger death |
|---|---:|---:|---:|---:|
| base | 5.5% | 28.9% | 1.02 | 8.6% |
| control | 4.7% | 0.8% | 0.43 | 21.1% |
| safe_medium | 3.9% | 0.8% | 0.31 | 12.5% |
| safe_high | 3.9% | 11.7% | 0.54 | 9.4% |

- 结论：单纯压低自炸会导致“不放泡也不逃”的被动崩溃，不能作为晋升标准。
- 来源：canonical工作区中的`reports/danger_curriculum_optimization_20260929.md`（canonical工作区历史未提交报告；后续应迁入版本库）。

### EXP-20260930-01：Reward V2 + threat

- 变更类型：Reward / 性能优化。
- commit：`5d991b1`（Reward与逃生分析），报告 `c0c4b43`。
- 初始化：历史 `runs/overnight_tf_v2/phase2_it4000.pt`。
- 公共训练：Transformer 192×4，1024 env，128 step，flee ratio 0.3，danger_arena=1。
- Reward V2：安全放泡base 0.06、tactical placement 0.6、forced kill 2.0、resolution 1.0、enemy threat 0.4；自炸/换命时屏蔽正向战术结算。
- 性能：V2路径优化前42,340 SPS，优化后121,202 SPS（2.86×）；V2+threat约116,899 SPS。
- 固定评估：Tactical v2，seed 20261001，1024局，greedy，max_steps=300。

| checkpoint | surviving kill | own-bomb | mutual | bombs/game | 相对it4000结论 |
|---|---:|---:|---:|---:|---|
| 历史it4000 | 41.8% | 7.3% | 0.7% | 16.6 | 基线 |
| sweep threat0.4 it100 | 41.3% | 6.2% | 1.7% | 25.8 | 击杀持平、进攻更活跃，换命略升 |
| long threat0.4 lr1e-4 it200 | 43.8% | 6.2% | 1.7% | 30.0 | 点估计更高但在噪声内 |
| long threat0.4 lr1e-4 it1500 | 25.0% | 7.4% | 1.6% | 22.8 | 长训显著退化 |
| long threat0.4 lr3e-4 it1500 | 11.5% | 25.7% | 1.2% | 22.8 | 严重退化 |

- 结论：Reward V2能防止攻击性塌缩，但未证明比历史it4000更强；lr 1e-4、≤300 iter并密集评估优于盲目长训。
- 来源：`docs/reward_v2_experiment_report.md`。

### EXP-20261001-01：分难度JAX Bot池

- 变更类型：新Bot / 对手数量与难度增加。
- commit：`75619e0`。
- Bot档位：dodge_easy、dodge、bomber_easy、hunter、hunter_hard，并保留legacy_flee。
- 训练分布：后续8卡run使用70%同参自对弈、30%分难度JAX Bot池。
- 固定评估：seed 20261001，每档1024局，danger_arena，max_steps=300。

| checkpoint | 六档平均p | hunter p | hunter_hard p | 强Bot平均p |
|---|---:|---:|---:|---:|
| 历史it4000 | 0.658 | 0.239 | 0.140 | 0.190 |
| dp8 it2500 | 0.939 | 0.925 | 0.831 | 0.878 |
| dp8 it6000 | 0.930 | 0.923 | 0.795 | 0.859 |
| dp8 it8000 | 0.938 | 0.927 | 0.844 | 0.886 |
| dp8 it10000 | 0.930 | 0.932 | 0.786 | 0.859 |
| dp8 it12000 | 0.916 | 0.884 | 0.752 | 0.818 |

- 结论：专项对战能力大幅提升，it8000是当前强Bot点估计最佳；但12k已有回落，说明增加训练数量并非单调收益。
- 结果：`runs/v2_jaxbot_dp8_20261001_102237_job-t3vswpexou-master-0/eval_live_20261001_new/jax_bot_ladder_1024.json`。

### EXP-20261001-02：出生位置多样化

- 变更类型：场景/出生分布。
- commit：`870b7dd`；8卡入口接入commit `6946024`。
- 动作：增加danger_arena出生位置采样桶，降低固定出生关系过拟合。
- 当前证据：native路径回归已通过；动作与Reward V2、Bot池一起进入8卡run。
- 缺口：尚无“只改出生分布”的固定seed消融，不能将组合训练效果单独归因于该动作。
- 决策：保留实现；后续若场景优化有效，需设计spawn-only对照。

### EXP-20261001-03：8卡长训规模扩大

- 变更类型：训练规模 / 对手池 / Reward / 出生多样化组合。
- 入口：`scripts/train_v2_jaxbot_dp8.sh`。
- 相关commit：`634c31e`（8卡入口）、`6946024`（出生多样化接入）、`036c2f4`（解释器修复）、`1e323bc`（前台持有训练任务）。
- 初始化：历史 `phase2_it4000.pt`。
- 配置：8设备、2048 env、minibatch 4096、60000 iter、每500 iter保存；70%同参自对弈 + 30% JAX Bot池；Reward V2；出生多样化。
- run：`runs/v2_jaxbot_dp8_20261001_102237_job-t3vswpexou-master-0`。
- 训练状态快照（2026-10-01 15:09 CST）：12932/60000，约1.30–1.33秒/iter，约19.7万–20.2万SPS；最新完整checkpoint `final_it12500.pt`。本条为进行中快照，最终墙钟待回填。
- Tactical v2探索评估：seed 20261001，每点256局，固定相同出生，greedy，max_steps=300。

| checkpoint | surviving kill | own-bomb | danger→death | mutual | safeDet | tactResolution | bombs/game |
|---|---:|---:|---:|---:|---:|---:|---:|
| 历史it4000 | 37.1% | 5.1% | 7.0% | 0.4% | 99.5% | 91.9% | 16.0 |
| dp8 it2500 | 22.7% | 23.0% | 28.9% | 1.2% | 95.7% | 90.4% | 12.4 |
| dp8 it6000 | 16.0% | 15.6% | 33.6% | 1.2% | 93.3% | 85.6% | 24.2 |
| dp8 it8000 | 16.4% | 22.3% | 41.8% | 2.7% | 92.4% | 91.6% | 34.8 |
| dp8 it10000 | 21.9% | 22.3% | 40.6% | 2.7% | 92.9% | 88.7% | 27.8 |
| dp8 it12000 | 26.2% | 16.4% | 37.9% | 2.3% | 93.4% | 82.5% | 40.7 |

- 当前解释：256局探索结果显示所有新增点的Tactical surviving kill仍低于历史基线，且自炸/危险死亡明显更高。该次启动误将`--device gpu`当作设备值，JAX无可用CUDA后退到CPU；因此只用于组内筛点，不能与历史GPU结果混算。
- 1024局GPU确认：固定seed 20261001、相同出生，仅评估历史it4000、探索候选it12000和明显回归点it8000。历史it4000 surviving kill 41.8%、自炸7.3%、danger→death 9.4%；it8000分别15.4%、13.1%、33.2%；it12000分别17.9%、20.6%、50.5%。相对基线的配对差均显著：it8000 kill −26.4pp [−30.1,−22.7]；it12000 kill −23.9pp [−27.8,−20.0]。
- 决策：当前无可晋升checkpoint；1024局确认已足以判定“训练Bot专项进步、独立泛化与安全退化”，无需继续放大该比较样本。
- 探索结果：`.../eval_live_20261001_new/tactical_256/summary.json`。
- 确认结果：评估worktree下`runs/eval_dp8_confirm_20261001/tactical_confirm_1024_gpu/summary.json`；完整checkpoint路径、SHA256和审计摘要在同目录上级`audit_report.json`。

### EXP-20261001-04：危险残局与随机场景族

- 变更类型：新场景 / 训练分布多样化。
- 分支：`agent/train-scene-opt`；commit `568609b`。
- 假设：专项Bot池导致对手过拟合；通过预生成并验证的多种危险局面，让模型学习逃生、走廊处理、连锁爆炸、主动攻击和攻防冲突。
- 场景：native、escape、corridor、chain、attack、conflict。
- 实现：设备端JAX/JIT生成与采样；接入reset/auto-reset和DP训练；场景标签不进入模型观测；新增固定seed场景评估器和A/B入口。
- 当前状态：实现、合法性审计、4卡DP smoke、两个独立训练seed的短A/B与评估均完成。
- 4卡 smoke：control与scenario均退出0；warmup约111秒；control中位约110.4k SPS，scenario中位约109.5k SPS。
- 大样本seed-1结论：scenario显著降低自炸与danger→death，但Tactical v2存活击杀下降；scenario final仅6.3%存活击杀（历史it4000为41.8%），虽自炸为0%、danger→death为2.8%，但属于过度保守和攻击塌缩。
- 第二seed的64局场景复核方向一致：scenario final在留出场景对hunter/hunter_hard存活约84.1%/83.8%，高于control约65.0%/64.4%，但native对hunter存活击杀仅7.8%（control 18.8%），对hunter_hard为0%。
- 统一四Bot 64局快速矩阵进一步确认：scenario两seed自炸普遍较低，但对网页猎手困难/普通几乎无存活击杀；结果见`reports/four_bot_matrix_64_20261001/`。64局只作筛点，不替代既有1024局证据。
- 决策：不晋升当前scenario checkpoint，不启动当前配置长训。下一轮降低scenario比例，并加入native/JAX Bot进攻能力保持约束。
- 晋升门槛：场景合法性与可解性通过；旧native路径不回归；4卡吞吐可接受；Tactical v2/held-out场景安全与击杀改善，同时JAX Bot专项能力不显著下降。

### EXP-CX-20261003-29：trap16 路线目标场

- 状态：完成短程训练、独立复现和四 Bot 确认；未启动长程训练。
- 变更类型：训练对手策略 / 评估协议。
- 假设：在半数 JAX `hunter_hard` P1 槽位中，为路线目标加入距敌三格内、至多 16 个可达安全候选，可提升安全进攻。
- 训练：control/trap16 各 4 seed，101 updates；起点 it4000，projected critic、kill reward 24、reward shaping 0.6、EMA 0.95、512 env x 64、top 25%、32 PPO epoch、lr 3e-4。全部正式任务 `rc=0` 且 checkpoint 有限。
- 快筛：固定 seed 20261001、每 Bot 64 局、真实 JS hunter、300 tick。trap16 it50 为 2/4 通过，it101 为 0/4；两条已通过轨迹从 C=0.547 到 0.422、C=0.516 到 0.250。
- 确认：每 Bot 512 局、eval seed 20264011、bootstrap seed 20264012、10,000 paired bootstrap。原候选 Delta C +0.142578，95% CI [0.060498, 0.226562]；独立重训 +0.101562，95% CI [0.009766, 0.193359]。
- 行为：tactical surviving-kill 为 201/230/229（基线/原候选/重训），自炸为 28/8/1；hard 为 4/0/0，normal 均为 0，收益不能解释为高难攻击转化。
- 决策：不批准原配置长训。预注册的四 seed、两臂、201-update 对手混合配对试验已在 CX-30 完成，未提供扩至 401 updates 的证据；后续必须先有与真实网页机制对齐的新单变量信号。
- 证据：`runs/cx20261003_29_trap_routes/`、`runs/cx20261003_29_screen/screen.json`、`runs/cx20261003_29_confirm_eval/`。运行产物不纳入版本控制；完整设计见 `reports/cx29_trap_routes_20261003.md`。

### EXP-CX-20261003-30：对手混合主动击杀 pilot

- 状态：完成四 seed、两臂、201-update 训练和 64 局四 Bot 快筛；未启动 401-update 或 512 局确认。
- 假设与唯一变量：固定 CX-29 trap16 及全部训练配置，只把 P1 JAX `hunter_hard` 比例从 50% 提高到 75%；reference 保持 50% 作为配对参照。
- 训练：两臂各 4 个固定 seed，初始化为 historical it4000，保存 it50/100/150/200/201；8 个任务均 `rc=0` 且 checkpoint 有限。
- 快筛：固定 seed 20261001、每 Bot 64 局、真实网页 hard/normal/easy、Tactical v2、同出生、300 tick 和 attack trace。reference 通过 1/16，mixed-hard 通过 8/16；mixed-hard 的 seed 20265002/3/4 各有至少两个通过点。
- 行为限制：mixed-hard 在 it50/100/150/201 的 hard+normal S 总计为 0/1/4/3（512 局合计），hard S 各保存点均为 0；it201 Tactical S/B/P 下降到 82/14.520/0.746。结果支持改变对手混合再评估，但不支持把它写成 hard 攻击改善。
- 决策：不启动 401-update 长训，不合入一次性训练入口，也不因 64 局快筛通过自动启动新的训练 seed 或 512 局确认。后续必须先完成真实网页机制机会审计并在新的单变量设计中得到跨 seed hard/normal 信号。详细结果见 `reports/cx30_active_kill_20261003.md`。

### EXP-CX-20261003-31：主动击杀机会审计

- 状态：完成 JAX/JIT 与真实网页 45-tick 配对诊断；未训练新模型。
- 唯一变量：只改变审计首步动作，比较 actor 原动作与安全放泡反事实；不改变训练或评估模型。
- 结果：JAX hard 三 seed 共 96 次实际放泡，surviving kill 对照/干预均为 `0/0`；网页 hard/normal 近距和 Hunter 逃路缩减事件也均为 `0→0`，部分干预增加死亡。计数是相关 tick，不是独立局率。
- 决策：不把静态 pressure/宽泛 threat 作为训练奖励，不启动长训。完整协议与限制见 `reports/cx31_attack_opportunity_20261003.md`。

### EXP-CX-20261003-32：网页 forced 几何见证与 reset window pilot

- 状态：机制见证和 21-update 短训完成；未做 512 局确认或长训。
- 假设与唯一变量：将网页清砖 map 806 的人工 forced 格位混入 reset；control/window 的名义差异为 reset 比例，但两臂共同改变了奖励，且 seed 未配对，因此不作严格因果归因。
- 结果：人工起点规则移动单泡为 `52/52` surviving kill；历史模型为 `44/52` 且 trade/死亡增加。正常出生四 Bot 64 局中，两条 window 模型 hard/normal 均 `0/64`，Tactical 为 `26/64`。
- 决策：人工见证不代表正常出生收益；先修正 JAX 静态标签与网页时间展开差异。详见 `reports/cx32_geometry_windows_20261003.md`。

### EXP-CX-20261003-33：JAX 对手梯度 ladder 对照

- 状态：完成 2 seed × 2 arm、21-update 训练与真实四 Bot 64 局快筛；未晋升。
- 假设与唯一变量：只将 JAX 对手池从 `dodge+legacy_flee` 改为包含 `bomber_easy/hunter/hunter_hard` 的 ladder；control 与 ladder 各两个 seed。
- 结果：control seed Tactical S 为 `28/30`、ladder 为 `23/22`；两臂真实 hard/normal 均 `0/64`，easy 仅有噪声差异。训练侧 hunter kill/death 信号不等于网页转化。
- 决策：不扩大对手梯度；保留精炼结论和长训门控见 `reports/cx33_opponent_ladder_20261003.md`。原始矩阵不入 Git。

### EXP-CX-20261003-34：trap-search 局部候选格诊断

- 状态：完成 2 seed、control/treatment 配对短训和真实四 Bot 64 局快筛；未晋升、未启动长训。
- 假设与唯一变量：在 hard JAX 代理动作中加入局部候选格的 danger/逃生空间搜索；训练通过 inline monkeypatch 使用隔离分支实现，生产训练入口未改变。
- 评估：固定 seed `20261001`、同出生、greedy、300 tick、Tactical v2 与真实网页 hunter `hard/normal/easy`，每格 64 局。
- 结果：trap-search 两个 seed 的 hard surviving non-trade kill 均为 `0/64`；normal 只有 `trapsearch_s20340001_final` 为 `1/64`，其余 treatment 点为 `0/64`。Tactical treatment 短点最高 `32/64`，但 control 也达到 `30/64`，不能据此归因。
- 决策：代理 JAX 的 kill/death 信号未转化为跨 seed 网页 hard/normal 击杀；不合入隔离实现，不扩展训练预算。详细表见 `reports/cx34_trapsearch_20261003.md`。

## 5. 新实验追加模板

复制本节到“详细记录”末尾，并同步更新“实验索引”。

```text
### EXP-YYYYMMDD-NN：实验名

- 状态：设计中 / smoke / 训练中 / 评估中 / 完成 / 中止
- Owner：
- 变更类型：新Bot / Reward / 场景 / 数量规模 / 超参数 / 评估协议 / 其他
- 假设：
- 唯一主要改动：
- 代码commit：
- 初始化checkpoint及SHA256：
- run路径：
- 训练配置：设备；env；step；iter；minibatch；lr；seed；保存频率
- 对手/场景/课程分布：
- 墙钟：smoke；失败/重启；有效训练；评估；总计
- 吞吐：
- 评估协议：checkpoint；seed；局数；座位；出生；对手；max_steps；输出JSON

| checkpoint/arm | 专项能力 | 泛化能力 | 自炸 | trade | danger→death | bombs | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | | | | | | | |
| treatment | | | | | | | |

- 相对基线变化：
- 结论与归因：
- 决策：晋升 / 不晋升 / 需要更大样本 / 中止
- 下一步：
- 已知限制：
```

## 6. 维护约定

- 每次启动新实验：先创建实验ID和“设计中”条目，再启动训练。
- 每次保存评估结果：立即追加指标和产物路径，不等聊天收尾。
- 每次停止、失败或重启：记录原因和墙钟，不能只保留成功段。
- 每次提交训练相关代码：PR/commit说明中引用实验ID。
- 每次模型晋升：在对应条目中写明被替代checkpoint、依据和可回滚路径。


## Bun392 持续训练能力保持冲突根因审计（20261011）

## Bun392 Retention v2 启动（20261011）

基于上述审计和 commit `862303f`，新建独立 namespace `runs/je03_bun_stage1_20261011/bun392_retention_v2`，实验 ID `JE03-BUN392-RETENTION-V2-20261011-v1`。两臂从同一冻结 `bun392_gen0.pkl` 启动：control 为 retention_v2 + 战斗优先课程，teacher 为相同底座加冻结 g0 actor KL=.08 与 45% 战斗 replay。奖励把安全击杀/存活置于主导量级，任务进度有界、一次性且无 crate/brick/explore 奖励；对手冻结为 rule_combat .35 + old .65，动态自对弈关闭。

CPU smoke/recovery 两臂均已通过 finite、非零 Actor/Critic 更新、四副本一致和 fresh-process digest 一致。旧 `bun831_native_tracks_v1` 在 control g1113、recipe g2253 更新边界安全退出，旧 checkpoint/评估/失败证据保留，`SUPERSEDED_20261011_retention_v2.json` 明确这是用户授权的替换而非能力失败伪造。GPU smoke/正式 owner 随后在 8 卡空闲后启动，CPU Hunter/Tactical/Bun-task 评估异步运行，不作为 GPU 启动或扩训阻塞门禁。设计与奖励逐项见 `reports/je03_bun_stage1_20261011/bun392_retention_v2_design.md`。

实际启动证据：GPU owner PID（本地记录）；正式 worker control PID（本地记录）（GPU0-3）、teacher PID（本地记录）（GPU4-7）。smoke/recovery 首次正式更新均 finite、非零且 replica exact；control 首个正式事件 g1 actor/critic L2=9.0233/15.4688，teacher g1=8.2185/15.8791。随后现场读到 control g24、teacher g22，fresh 吞吐约 47.4k/47.9k ticks/s、11.58/11.70 optimizer updates/s，显存约 30-32 GiB/card，8卡均有持续计算。独立 CPU retention owner PID（本地记录）；旧 evaluator KeyError（错误假设 teaching tracks）已保留并由新 owner 修复隔离，不影响训练。

对 `JE03-BUN831-NATIVE-TRACKS-20261011-v1` 做独立只读审计，未停止或修改健康 GPU/CPU 进程。新报告与机器结果：`reports/je03_bun_stage1_20261011/bun392_retention_conflict_analysis_v1.md`、`bun392_retention_conflict_analysis_v1.json`；诊断脚本：`scripts/je03_bun392_retention_conflict_analysis_20261011.py`。覆盖 base g0/32/64/128/192/256/512/768、recipe g0/32/64/128/192/256/512/768/1024/1536 的 Hunter hard/normal/easy safe/death/self/trade/bombs、Tactical、carry/steal/route/full 任务、训练 reward/done/kill telemetry、实际 optimizer accounting、Actor/Critic/输入尾部/heads 漂移。

关键事实：共同 Bun392 g0 Hunter safe `19/192`、Tactical `22/64`；base g32 已降至 Hunter `1/192`、Tactical `16/64`，recipe g32 为 `7/192`、`14/64`。recipe 任务却在 g128 达到 carry/steal `16/16`，base 到 g512 carry `16/16`，而 Hunter 未恢复；base g512 Hunter death/self `142/192`、`68/192`。训练 mean reward 随早期点上升（base .044→.066，recipe .060→.148），不能当作能力提升。实际 Adam 为 base 128/gen、recipe 32/gen，Top25 原始 telemetry 的 128/gen 已由 `accounting_v1` 取 checkpoint 校正，旧 raw 保留。

可证伪假设排序：A Bun 任务奖励与 Hunter 目标错位最强；B 课程/冻结对手分布冲突；C PPO 累计更新遗忘；E 新 Bun 输入/道具通道表征扰动；F Critic/共享梯度可能放大但独立 Critic 两臂仍退化，不能作为唯一解释；D Top25 只解释 recipe 特有捷径，不能解释 base 共同退化。现阶段结论是“任务局部学会、Hunter 战斗/安全目标遗忘或未被优化”，不是训练完全失败；无 entropy/KL 训练遥测，已明确标缺失。后续仅做 CPU 离线 reward 分项、lesson/opponent 频率、固定 batch KL/漂移和 Top25 构成反事实，不设 GPU 阻塞门禁、不热改当前配方、不启动第二阶段。

## Bun392 保守动态自对弈续训接管（20261012）

最终CPU monitor_v2已实际完成两臂真实非零finite更新、每代测试league refresh及fresh-process完整恢复，监测进程自然退出；最终SOURCE_SEAL所有源码hash复核一致。完整JSON证据在`reports/je03_bun_stage2_20261012/selfplay_cpu_validation_20261012.json`，不作GPU阻塞门禁。接管实现提交851d4bd，唯一handoff494076持续WAIT_PARENT_ATOMIC_EXIT，当前候选2048；正式起点与GPU首更新将在实际4卡释放时生成。

耐久调度实装：tmux `je03-bun392-selfplay12-v2-20261012`、launcher494073、唯一handoff494076，flock已现场验证拒绝第二writer。plan SHA `4e89f524c9e428da48417403f0326cc049d6f5e4a36ed2f216f6c322c4b4f5f5`，当前WAIT_PARENT_ATOMIC_EXIT、candidate2048、assigned空、checks_block_launch=false。8项CPU测试通过；两臂真实CPU nonzero+fresh-process restore digest相同且passed，refresh=1监测证明recent→history滚动存取；另真实网络同物理状态概率响应诊断通过（最大变化.930254、KL1.967522，仅功能诊断，不作训练稳定性结论）。最终快照monitor_v2并行复验。新正式self-play尚未占GPU，父retention两worker440471/441350仍健康占满8卡；不伪造PID或起点hash。安装证据在 `reports/je03_bun_stage2_20261012/selfplay_handoff_installed_20261012.json`，首次释放自动生成共同 `INITIALIZATION.json/SOURCE_SEAL.json` 并直接启动formal，技术检查不阻塞。

用户授权阶段2准备，但当前 retention_v2 control/teacher 继续健康训练至4096，未中断；旧CPU owner继续3072/4096。新实验 `JE03-BUN392-SELFPLAY12-20261012-v2` 的CPU准备产物位于 `runs/je03_bun_stage2_20261012/bun392_selfplay12_v2`，handoff owner将在旧worker/owner原子退出、实际4卡释放后自动接管，不等待CPU评估或技术READY。共同保守底座：LR7.5e-5、clip.1、1epoch/64 updates-gen、global clip.5、g0 Actor KL.10、45%战斗重采样；两臂只在12%对手槽位区分冻结Stage1与每128代更新的recent/history神经网络联赛，规则/g0/Stage1其余池相同，learner角色50:50。起点按教师曲线先比较相邻点hard/normal共同安全击杀再比较稳定效用，当前偏好g2048，不默认4096；迟到CPU不阻GPU。实际reward披露：现有retention_v2构造被lesson路由覆盖，新A/B冻结当前语义，不把未生效候选奖励当实验变量。新增每代KL、熵、梯度、Actor漂移、对手/lesson ticks和league版本遥测；连续三点明确恶化才原子替换同卡recovery。六项self-play TDD及control CPU nonzero/fresh restore已通过，dynamic CPU monitor并行中。详见 `reports/je03_bun_stage2_20261012/bun392_selfplay_plan_20261012.md`。

## Stage2 CPU 权威评估故障恢复（20261012）

旧 CPU evaluator 在取 manifest 时误用 Bun831 wrapper 的 `SPAWNS` 属性，实际常量位于嵌套 authority，启动抛 AttributeError；handoff 15秒重试未解决。旧 `cpu_owner.log`/protocol/产物保留。新增独立 `authority_stage2_recovery_v2` 及恢复入口，保持训练 SOURCE_SEAL 不变，现场 seal 验证通过。唯一新 CPU owner PID（本地记录）（tmux `je03-stage2-cpu-authority-recovery-20261012`）同时持有新旧评估锁，旧重试不能成为 writer；CUDA 空、JAX CPU，CPU96–127与trainer128–135隔离。13项相关CPU测试通过。GPU worker（本地记录）继续健康4+4，取证时g2077/g1649、约71.2k/65.3k fresh ticks/s，未中断。

同冻结 Hunter64/档300+64 residual、Tactical64及 Bun task16/类，配对优先g0/512/1024/1536/共同2048，再补早期点及追随3072/4096。任务只增加换命/安全击杀事件观察，idle诊断局限明确；真实Hunter依旧网页路径。新owner逐checkpoint及shard hash校验、原子PAIR、异步不阻GPU；终态以新EVALUATION_COMPLETE/OWNER_FINISHED为准，唯一通知为每PAIR一次jdme-notify claim。根因、完整协议、启动与结果记录在 `reports/je03_bun_stage2_20261012/selfplay_cpu_authority_recovery_20261012.md` 和run下 `RECOVERY_RECEIPT_20261012.json`。

本轮已取得真正首个完整两臂 `PAIR_g000000.json`（新authority_stage2_recovery_v2），两臂独立重评同Stage1 teacher2048 policy SHA fe61db9e…，Hunter hard/normal/easy各64：safe2/4/10、death1/1/5、self0/0/1、trade0/0/2；Tactical64 safe21/death3/self0/trade1。四类任务每类16：carry capture14、near steal16、route/full capture0；full death/self1，所有task trade0，两臂全指标一致，Hunter pair差全零。control/dynamic墙钟240.611/240.568秒，配对481.179秒；唯一g0通知发送成功，owner自动开始g512。现场GPU原worker继续g2141/g1708、70.9k/65.1k fresh ticks/s；已有10个待评pair约80分钟CPU工作量，未来2048/3072/4096跟随共同提交。数据与协议hash/进程receipt存同名20261012 JSON报告；64局只作快筛，不宣称第二阶段能力已提升。

## Bun392 Stage1/Stage2 全曲线终态验收（20261012）

Stage1/Stage2两臂均实际完成4096代，Stage2每臂2,147,483,648 fresh ticks，control/dynamic有效训练8.487/9.191小时、墙钟8.658/9.375小时。Stage2全部14个pair、所有checkpoint state/policy、评估shard、TRAINING_COMPLETE、EVALUATION_COMPLETE及OWNER_FINISHED hash已重新验证；所有训练/评估相关进程退出。独立只读汇总脚本 `scripts/summarize_bun_stages.py` 输出 `reports/je03_bun_stage2_20261012/bun392_final_acceptance_20261012.md` 和同名JSON，保留完整两阶段曲线和证据hash，无run/checkpoint/log入库。

Stage1教师KL+45%战斗重采样组合明显改善无保持对照的安全退化，但两项共同变化不能拆归因。实际Stage2起点是当时已冻结的teacher g2048，Hunter safe2/4/10、death1/1/5、self0/0/1、trade0/0/2、Tactical21。Stage1 teacher g4096为另一个探索确认候选：safe0/4/18、death/self/trade全0、Tactical24；不回填成历史已知最优。

Stage2动态12%联赛没有形成稳定三档共同优势。最值得扩大样本确认的动态候选为g3072：Hunter safe0/4/23，death1/0/3、self全0、trade0/0/1、Tactical24；但full任务death/self2，不能声称安全非劣已通过。g4096退为safe0/3/16、death1/4/2、self0/3/1、trade全0、Tactical19。固定g4096 safe3/2/14、death1/3/1、self0/0/1、trade0/1/0、Tactical18；动态相对固定总safe持平19/192、death+2/self+3/Tactical+1。保留g3072，不默认终点最好。Stage2所有点carry capture14/16、near steal16/16，route/full capture/steal/return全0，Bun任务仅局部保持，无新完整任务进展。单seed、每档64快筛、多点选择偏差，不得作为最终显著性/晋级/部署结论；旧协议网页Hunter native_items/native_trap关闭的局限与独立native安全probe必须同时披露。


主线白名单整合（20261012）：基于最新origin/main a260615，代码提交2006268；完整248提交/1274差异文件审计见main_integration_audit_20261012.json。main网页/部署和现有更新全部保留，未携带runs/checkpoints/logs/.hermes/历史vendor。70项不同Python测试、7项新增Node及main全套npm测试通过；真实main CPU Hunter8局兼容性smoke通过。最终候选不自动部署，不改网页模型目录；详细整合、依赖、恢复和回滚见docs/bun392_main_integration_20261012.md。

## Bun392 g3072 网页模型发布（20261012）

完成Stage2 dynamic g3072的COMMITTED、raw/EMA/state、plan及全部权威分片hash复核。独立CPU同冻结协议EMA追评137.02秒：Hunter1/5/16（22/192）、Tactical18/64；原始策略0/4/23（27/192）、Tactical24/64。EMA安全性略好（full死亡/自炸0/0对raw2/2），但综合击杀快筛较低，故本轮网页选择raw，不套用raw指标到EMA。所有64局/档仅快筛，不作显著性结论；困难档0、easy贡献为主、route_break/full任务未突破。

新Release `web-bun392-selfplay3072-v1`，网页名称“Bun392动态自对弈3072”。原生Bun392输入/动作保持，无adapter；确定性JSON两次字节一致，与权威模型全部packed权重一致，8组JS/ONNX最大误差5.72e-6。当前main三档Hunter各8局smoke0/0/3且危险结算完整，与冻结曲线分开。12项Python测试、完整npm回归、本地浏览器真实下载校验及16tick/16次有限推理通过。来源、SHA、Release资产、EMA比较、限制与发布后核验标准见 `docs/bun392_model_release_20261012.md` 和同日机器报告。保留旧5个网页模型；Pages从各目录Release URL下载并校验后同源加载。

按用户新隐私边界脱敏已追踪文档中的本机身份路径/IP/PID；不清理未追踪run/checkpoint/编排，不重写Git历史。新增仅本地AGENTS规则并以根级`/AGENTS.md`忽略；扫描覆盖tracked＋本次明确public文件。未启动训练、未改历史checkpoint。

发布恢复：Actions `38265081242` / head `7265ee5` 的test-and-build因首次Release下载HTTP500失败，deploy未执行。下载器改为最多5次、仅HTTP429/500/502/503/504与超时/断连重试，1/2/4/8秒指数退避并尊重Retry-After；超过60秒等待预算明确失败。永久错误/大小/hash错误仍硬失败，不跳过模型。26项确定性相关测试通过，完整静态站继续校验全部6模型；必须用新提交部署并核验同head最终success及公网真实加载，不以rerun旧workflow替代代码修复。Release五个资产均已通过无需凭据的公开下载与完整SHA复核。


## Bun392 网页推理性能修复（20261012）

不停止JE04 GPU训练或CPU评估，网页独立在main实施专用Worker+同源ONNX Runtime Web1.23.2单线程SIMD WASM。原392纯JS p95 183.4ms，Worker/WASM27.8ms；16原生状态最大误差2.3222e-5、greedy0差异，32固定seed完整轨迹0差异。完整装配站两模式各10秒100逻辑tick/100完成，推理p95 14.8/36.0ms，rAF约16.7ms，无稳定窗口longtask。模型raw、Bun编码/动作/greedy及10Hz不变，不降低tick、不替换模型。JSON/ONNX下载和SHA/大小硬校验，运行时回退明确与性能成功区分；WebGPU无实际adapter，不声称测得更快。25项Python相关测试及网页回归，部署按精确head终态核验。详见 `docs/bun392_web_inference_performance_20261012.md` 和同日JSON报告。

## JE05 raw g1536网页模型发布（20261013）

按已完成的同协议权威结果选择JE05 `bun392_combat_transport_pfsp8` g1536 raw，而非正在训练的JE06或JE05终点。Hunter hard/normal/easy3/1/32（36/192）、死亡/自炸/换命3/0/1；Tactical22/64且安全项全0，原生carry15/16、near steal15/16（死亡1、无条件抢包1）。g3072同为36/192，但Tactical17、自炸3；JE04 g768 normal9且更安全，却原生carry0，g2048 Tactical26但Hunter死亡6/自炸1/换命2。全部候选route/full300以及600/900tick旁路未capture。g1536是综合Pareto折中，不宣称所有维度支配或统计显著提升，64/档仅快筛。

raw SHA `16ba817d53ef1f3ed4cf6039ff58f33f1f0de4adc69b3578047c5c3cdfbda719`；COMMITTED/raw/EMA、完整state、对应Critic、8副本和权威shard均核验。EMA未同协议评估，不套用raw成绩，发布实际被评估的raw。原生24/global24/392×4/15动作不变；网页按显式`bun24-trap-timer-v2`元数据兼容糖泡倒计时，旧模型编码不变。

JSON与ONNX独立双导出字节一致、packed权重exact，Python FP32/JS/ONNX八个合法双席位状态最大误差4.77e-6、合法greedy动作一致；JAX训练BF16漂移单列诊断（最高约0.113），不冒称逐值等同FP32权威。27项Python发布/下载测试、完整npm网页回归、本地Chromium WASM worker实际1v1及三档Hunter300tick+危险结算smoke通过。Release `web-bun392-transport1536-v1`与网页首项“Bun392战斗运输1536”保留全部旧模型，资产JSON/ONNX均内容寻址；不上传state/优化器/日志。远端SHA与精确head的最终部署另留验证证据。报告见 `reports/model_releases/bun392_transport1536_release_20261013.md`及同名JSON。

发布只在canonical main进行，JE06独立supervisor持续正常8卡训练，未停止、热改或污染其run；未发布JE06早期点。隐私边界严格为Git追踪/待提交与Release内容，AGENTS本地忽略，旧run/checkpoint保留。

最终线上验收：修复head `195993d` 的Actions `38416916939` test-and-build/deploy均最终success，公网build-info与完整目录匹配。Chromium以worker-wasm实际加载，原生1v1完成16 tick/16推理，Hunter三档300 tick及危险结算通过，无页面错误或写请求。Release五项资产与Pages JSON/ONNX完整回读大小/SHA一致。首次网络超时失败证据保留，技术smoke不计能力成绩。

Release五项资产认证API完整回读、Pages JSON/ONNX公开完整回读大小/SHA一致；首个发布head `abea27f` Actions `38411683710`两job最终success且公网目录匹配。首次Chromium真实加载因单连接约17KB/s触发worker20分钟超时，未推理，失败不作成功。追加有界8并发256KiB分段下载与确定性错误/完整性测试，最终仍要求完整SHA，推理/策略/10Hz语义不变；新head与公网浏览器终态继续核验。


## JE10 因果清理接续与用户暂停终态（20261019）

本条覆盖此前持续训练/自动接续政策：用户明确要求安全暂停、总结并合入main。r19正在执行代通过worker原生停止标记完成g58原子checkpoint及STATE后退出；atomic_stop=true、121,634,816 fresh ticks、剩余70代，不标COMPLETE/FAILED。supervisor先阻断，r20未启动，trainer/evaluator/consumer退出、8卡compute contexts0，active marker stale。恢复必须显式用户指令。原dirty树与run证据完整保留。

撤回旧“真实取包后无路／动态封路／预开额外返程通道”根因。真实取包存在可逆入站通路是规则不变量；假不可达来自库存fixture直接置入敌笼组件、无真实入站，death/sugar/recovery各16题有8题初态home INF，不能作为真实返程分母。scene2/3/4/6、旧lesson1/11和scene-ID custody特权及缺少原生拾取证明的return监督已清理，合法2058行及真实库存/drop/contact链保留。r11g18 EXCLUDED_INVALID排除所有能力结论；干净链从r10g128→r12…r19，合法返程DAgger×2。focused及相关测试覆盖16384固定seed空手初始化和真实取包链。

r12/r13清理后的合法返程×2，r14第二不同格返程query、r15 error优先、r16回放暴露遥测、r17实际replay128、r18认证返程移动豁免旧策略移动KL、r19 FIFO4096，均按固定128预算及唯一独立CPU/JS固定评估接续（r19因用户指令暂停）。每点Tactical Python v2及JS Hunter难/中/易各64、八路线各16，g0/4/8/16/32/64/96/128固定seed；没有能力门禁，中间分数不控制停训。原生地图只有806，不能声称多地图泛化。

r12–r18各8点、r19前5点共61个COMPLETE逐shard验证。最近完整r18g128：Hunter0/2/7、Tactical5，八交付0/3/12/16/15/12/3/2；r19最后完整g32：Hunter2/1/7、Tactical6，6/3/11/16/16/12/5/6，自然拾8/交6，但Tactical自伤4。r19g58未评估，不以训练reward推断能力。历史发布r01 attempt04g8保留其当时37点折中身份，不自动宣称今日全局最优。所有点/专项并列最高、动态对手实际占比、归因/路线漏斗及根因证据见[20261019报告](../reports/je10_training_20261019.md)、[长期总结](je10_control_route_training_summary.md)。

独立集成worktree基于最新origin/main，选择已安装因果清理/DAgger/FIFO/训练控制/评估代码和相应测试；不发布未安装快照候选或一次性审计脚本。保留上游多模型Reward/Advantage回放；仅添加只读结果页。run/checkpoint/cache/.hermes/机器信息不入Git；外部bootstrap依赖明确说明。完成测试/隐私扫描后push main，等待同SHA Pages并实测HTTP/内容，交付收据在本地忽略目录保存。

集成验证：347项选定依赖用例与9项证据/曲线用例分批复核，历史假设/fixture包遮蔽修复后的相关重跑通过；npm全量及Review、JS control funnel、桌面/手机61点页面与筛选验证通过。diff及隐私扫描通过；旧文档一处本机目录脱敏。未安装候选及历史边界草稿排除，详见20261019报告验证附录。

终态交接技术补充（20261018）：行政owner经focused TDD RED→GREEN（6项含既有后继规则）后独立更新，当前GPU worker/CPU evaluator未中断，运行数值source/plan seal复核exact。固定run原子终态后用真实GPU清单及compute client查询验证8卡释放，立即启动预声明后继，不等待最终评估。`RUN_TERMINAL.json`先记录终态和交接；后继完成最小技术smoke/fresh restore即正式训练，并仅在formal g1 checkpoint提交、8副本一致、Actor/Critic均finite且真实非零、policy hash核验后写`FORMAL_NONZERO_CONFIRMED.json`并主动确认启动。整段分析报告在此后进行；这都是技术有效性证据，不是能力门禁。后继的行政/CPU诊断源码独立封印，避免无关报告维护污染数值训练seal。

后继宿主诊断修正预声明（20261018）：只在r01固定128代worker退出后，新run准备阶段安装状态切片后再D2H拷贝的hash采样，避免逐样本hash对整批观测发起host提取请求（JAX可能缓存首次转换，不将源码重复请求冒充已测量的重复物理传输）；hash语义保持exact。追加现有context verifier audit的candidate/selected/accepted/first-deviation-query及片段分项日志，区分抽样、安全/几何过滤与首次偏航配额。r01未热改，numerical seal仍exact；9项focused技术交接/采样测试GREEN。两项均只影响宿主诊断，不改奖励、oracle、动作、梯度；r02唯一配方因素仍为预声明场景混合。

场景混合口径补充（20261018）：所有combat/bridge/full/routes预声明值均指reset抽样权重，不能称作实际训练tick份额。不同horizon和终止速度导致g15有效tick分布：combat272398、bridge123476、full1669009（约13.2%/6.0%/80.8%）。这为战斗保持下降提供暴露分布假设，尚非单变量因果结论；继续固定run并按每点实际计数报告。r02原预声明reset混合.35/.20/.35/.10不临时改变；最终曲线再决定后续最小可验证调整。

实际对手曝光后继补充（20261018）：原r01记录是底层抽样current/ring/fixed，受weak-role/assist覆盖影响，不回填伪造精确实际份额。r02仅在前run固定终态后增加诊断provenance：在reset更新kinds之前捕获最终动作来源，按assist优先、weak role其次、原family最后归类，并保存被选league slot及learner-alive有效位；不改任何动作/RNG/reward/loss。context既有trace79:92与label48:60保持，新增三列之后的verifier audit由末尾相对索引读取。12项focused技术测试GREEN，覆盖override排除、终态安装边界、reset前捕获及audit offset。CPU唯一owner于g16完成、g32尚未到达的空闲间隔更新，只增加可选的`executed_after_overrides`汇总，旧点COMPLETE保持；r01该字段缺失明确为不可得，非零比例。后继source/plan与独立EVAL seal分别记录。数值worker与当前r01封印未变，无能力门禁。

固定run `runs/je10_bun_20261018/control_route_v38_r01_attempt04` 完成128代/268,435,456 ticks；compute=4751.549s、wall=9949.416s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r02`；本次scene-mixture选择依据 `Zero independent native route initialization support; reserve .10 mass while retaining .35 natural full and combat.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。

原生一步候选补证（20261018）：JAX生产转向primitive在Bun806开墙固定位置`[6.28,5.11]`、速度1.3上独立复现旧`.18` oracle下/左并列，但真实下gain=-.39、左+.39；新证书仅左。focused候选/边界/交接共21项GREEN；r02运行源码不变。行政owner已只替换自身加载r03提案，原GPU/CPU worker未变、numerical seal exact。r02 g0全部固定评估COMPLETE，与r01 g128的Hunter/Tactical及完整route对象exact相同（原始pickle序列化hash不同，不能冒称byte exact）。

r02持续监督补证（20261018）：g8/16,777,216 fresh ticks全部finite/nonzero、恢复点已提交，固定预算继续；CPU g4正在运行、无failed shard。父点继承不只依赖行为：r02 g0与r01 g128导出Actor flat权重及tensor schema exact；原始policy pickle文件hash不同，明确区分。r02数值source/plan和准备r03 certificate源码hash均exact。

r02 g8实际曝光（20261018）：有效tick比例combat16.58%、bridge3.99%、full72.89%、其他路线约6.54%，其他课程独立曝光已非零，但不视为能力改善。最终执行source累计current/ring/fixed-league占全部有效tick1.03%/6.59%/9.59%，fixed-g0/stage1为12.04%/11.60%、responsive18.60%、weak-courier/guard28.92%/10.13%、assist1.50%；均为override后真实source且计数守恒。audit active2981288、selected185252、accepted103004、first-query1740，accepted片段drop44522/stock3363/return55102/delivery17；不同阶段/抽样分母保留，不把query/active当可达率。

r02 g4固定点诊断（20261018，训练已越过g16）：Hunter难/中/易8/3/17、Python Tactical8；相对本轮父点+7/-2/+2、+3。路线交付death5/sugar5/drop4/near16/mid16/far10/recovery4/full3（各16），相对父点+2/+1/-7/0/+1/-1/0/-5。掉包team0/1拾取3/8与8/8、交付各2/8，两席都是open geometry；首方向16/16并不防止后续偏航。固定seed2026101800首偏航动作前tick3/结果帧4，A_ENEMY_DROP、distance31.17，oracle左实际下，重建route-.003048828、zero-value GAE .002883419/前帧.003098989、delta-.000215570。full首次控制32/唯一归因9/final pop32/active22/因果kill9/surviving8，拾取7/16交付3/16、death6/self0。恢复switch28/32是目标关闭observer计数，不能当持包远敌也必须combat的指标。audit active1322390、selected90282、accepted48517、first-query861；独立几何分母未记录。g4早期能力取舍不触发任何停训或promotion门禁，g8评估已异步开始。

r02 g8窗口耗尽诊断（20261018，固定Review seed2026101800，不择优）：真实drop全局8tasks中该game0没有pickup；A_ENEMY_DROP从tick2进入，到动作前tick100仍distance7.00、tick101仍6.61，结果帧102切COMBAT（敌恢复），最终441tick结束且pickup/delivery均0。前段结果帧4距31.17、20距32.22、40距30.27；这40tick只净推进约1.34，之后正常推进也来不及在原生窗口内抵达。首偏航动作前tick3仍oracle左、实际下，重建route-.003048828、zero-value reward-GAE .002356804/前帧.002601206、delta-.000244403。g8固定掉包team0/1拾取2/8与8/8、交付1/8与5/8；首步一致16/16仍无法保证窗口内持续推进。该证据支持先修原生进度标签而非延长窗口/奖励补偿；敌恢复后空手关闭运输符合当前规则，不把合法切换当失败动作。

r02 g8完整点：Hunter难/中/易1/3/16、Tactical7；delivery death4/sugar6/drop6/near16/mid16/far9/recovery4/full5（各16）。相对g4为-1/+1/+2/0/0/-1/0/+2，父点为+1/+2/-5/0/+1/-2/0/-3。natural full首次控制39/唯一归因8/final pop39/active34/因果kill8/surviving7、pickup8/16、delivery5/16、death5/self0。中间结果只用于诊断和后继提案，固定run不提前停。

r02固定预算四分之一（20261018）：g32/67,108,864 fresh ticks checkpoint已原子提交，passed/devices8/replicas-exact均确认，g1–g32 finite/nonzero，训练已继续至g33。CPU完整点0/4/8已COMPLETE，g16运行、无failed shard，g32顺序排队；不等CPU、不因早期掉包/full回落停止。r03已预声明单因素native-step oracle修正，终态后才安装。

跨run因果边界补充（20261018）：每个后继按预声明继承最终raw Actor/Critic及完整pool，fresh Adam/RNG/env，DAgger FIFO重新建立；`retention_parent.pkl`实际重置为该新父Actor（c.prepare写initial），冻结g0/stage1来源保持。旧继承模板中的“unchanged historical retention reference”并非当前实际输入，不能沿用该文字声称参考权重不变。r02在这一固定run初始化机制下只改reset scene mixture，r03候选在同机制下只改strict native-step oracle；顺序训练缺少同父点对照，不声称单因素因果效果。运行数值source/plan不热改，后继声明应明确实际初始化语义。

r02 g16完整点（20261018）：Hunter难/中/易3/9/15、Tactical10，较g8+2/+6/-1、+3；相对父点+2/+4/0、+5。路线delivery death4/sugar4/drop8/near16/mid16/far11/recovery5/full6，各16；相对g8为0/-2/+2/0/0/+2/+1/+1，父点为+1/0/-3/0/+1/0/+1/-2。drop team0/1拾取8/8与5/8、交付5/8与3/8，双seat开墙，弱席再次变化；natural full拾取11/16、交付6/16、death7/self0，控制首进入56/唯一归因19/final pop56/active49/因果kill19/surviving17。固定real_drop seed2026101800首次偏航移至C_RETURN动作前tick81/结果帧82，目标己笼、distance33.90，oracle上/右、实际左；重建route-.001353906、zero-value GAE-.013705910/前帧-.014153949、delta+.000448039。早期回落后部分恢复，支持照常完成预算，未证明最终改善或单因素因果。audit active6166835、selected381054、accepted211124、first-query3455，仍缺独立几何分母；累计自然full事件按原机会记录。CPU已进入g32，GPU健康继续。

r02 g32完整点（20261018）：Hunter难/中/易5/3/8、Tactical11，相对父点+4/-2/-7、+6；较g16+2/-6/-7、+1。路线delivery death2/sugar3/drop14/near16/mid15/far13/recovery4/full8（各16），较g16-2/-1/+6/0/-1/+2/-1/+2，父点-1/-1/+3/0/0/+2/0/0。drop拾取16/16、两席交付6/8与8/8；natural full拾取10/16、交付8/16、death3/self0，控制first39/归因14/pop39/active33/causal14/surviving13。固定real_drop seed2026101800首偏航C_RETURN动作前tick81/结果帧82，distance33.06、oracle下/右、实际左，重建route-.002621094、zero-value GAE-.022013309/前帧-.015623705、delta-.006389604。full持续oracle一致2515/3314、progress2501/6302；与pickup/delivery机会分母分列。
实际有效tick combat13.79%、bridge3.83%、full76.88%、其余路线5.51%；新增reset路线质量不等于充分combat保持。override后current/ring/fixed-league占有效tick1.285%/7.293%/11.815%，分别848982/4819968/7808448，分母66089737；weak-courier/guard14582533/5222432、assist1065671另计。audit active11504333、selected774326、accepted444451、first-query6966，phase accepted drop201523/stock29058/return213617/delivery253。旧原墙stock不可达与战斗取舍未解决；中期路线改善仍按原固定run继续，CPU下一点64。

r02固定Review相位切换补充（20261018）：real_drop seed2026101800在g16有C_RETURN→COMBAT5次、反向5次（317tick）；g32有16次与15次（801tick），包含可能的交付/终局退出，不能直接叫错误切换。新增只读curve统计每个固定Review的phase ticks及phase transitions，不运行新评估、不改协议。距离≤4的威胁代理是否在墙体遮挡或边界附近造成返程抖动，是待结合原生状态核查的独立假设；不据此热改当前run，也不与r03原生标签单因素混改。

r01全曲线可视化补充：`reports/je10_r01_curve_20261018.png`，仅消费完整固定点，不挑checkpoint，不作为训练门禁。

r02固定预算中点（20261018）：g64/134,217,728 fresh ticks已原子提交，devices8/replicas_exact确认，g1–g64全部finite/nonzero，无failure marker。CPU唯一owner已自动开始g64固定Tactical/JS Hunter三档各64及路线漏斗；GPU继续剩余64代，无能力/admission/promotion门禁，不等待CPU图表或报告。

DAgger first-query边界审计（20261018）：`first_fragment_deviations`在每次1024tick rollout内以全零failed开始，并在episode done清零；没有把每阶段failed位跨rollout checkpoint持久化。因此准确语义是“rollout内各episode片段首偏航”，跨rollout继续的同一episode可再次query。此前简称每episode首偏航配额需以此限定，不能把query数当独立完整episode或唯一事件数。该事实不改变credit事件账本的稳定ID/去重，也不构成NaN/无效run故障；当前健康固定run继续，后继若调整采样须独立预声明，不能与r03原生端点标签混改。

r02 g64中点完整评估（20261018）：Hunter难/中/易2/6/11、Tactical5；相对父点+1/+1/-4、0，前点g32-3/+3/+3、-6。路线delivery death4/sugar3/drop10/near16/mid16/far14/recovery5/full6（各16），较父点+1/-1/-1/0/+1/+3/+1/-2，较g32+2/0/-4/0/+1/+1/+1/-2。drop拾取16/16、两席交付8/8与2/8；natural full拾取10/16、交付6/16、death4/self0，控制first33/归因10/pop33/active28/因果kill10/surviving9。死亡stock原墙team0组合8/8拾取，C_RETURN target2123ticks但oracle机会0、交付0/8；开墙team1拾取7/8、交付4/8。这是结构性监督缺口的延续，不宜靠增加query解释。
固定real_drop seed2026101800首偏航又回到A_ENEMY_DROP动作前tick3/结果帧4，distance31.17、oracle左实际下，route重建-.003048828、zero-value GAE .018986222/前帧.018320164、delta+.000666058；该局最终仍拾包并交付，说明首偏航不等于整局失败，需联合后段漏斗。返程→对抗/反向各8次，不直接认定为错误切换。训练累计control gross/net150393.5/120148.25，route正34007.199/负-59053.497/net-25046.296、capture终局1896480，终局继续主导。audit active21208001、selected1561153、accepted936708、first-query14835，accepted drop412912/stock95183/return427928/delivery685。override后current/ring/fixed league占有效tick1.881%/8.517%/13.008%；相对各点独立曝光作诊断，不作为停训阈值。GPU固定128代继续，CPU下一点96。

r02固定晚期点（20261018）：g96/201,326,592 fresh ticks已提交，8副本exact；g1–g96有限且真实非零，无technical/owner failure。CPU唯一owner已自动进入g96固定Tactical、真实JS Hunter三档64及8路线漏斗；GPU继续最后32代，固定终态后先交接预声明r03，无能力门禁。

r02 g96晚期完整点（20261018）：Hunter难/中/易2/5/9、Tactical11，相对父点+1/0/-6、+6，前点g64为0/-1/-2、+6。路线delivery death5/sugar6/drop7/near16/mid16/far14/recovery4/full6（各16），相对父点+2/+2/-4/0/+1/+3/0/-2、前点+1/+3/-3/0/0/0/-1/0。drop仍pickup16/16、两席交付2/8与5/8，natural full pickup8/16、delivery6/16、death8/self0；控制first38/归因8/pop38/active30/causal8/surviving8。固定real_drop seed2026101800首偏航C_RETURN动作前tick80/结果帧81，distance33.88、oracle下实际左，route重建+.000209375、zero-value GAE-.017043125/前帧-.017299858、delta+.000256733；COMBAT422/A79/C_RETURN495 ticks、返程/对抗双向各25次。不能以即时折扣PBRS为正证明该动作有高长期Q，仍需原生进度标签及相位状态联合诊断。
训练累计control gross/net232554/184266.75，route正49374.133/负-87858.958/net-38484.824、capture终局2938176，终局继续主导。audit active30992729、selected2347853、accepted1430201、first-query22232；accepted drop624355/stock161106/return643606/delivery1134。override后current/ring/fixed league占有效tick2.109%/8.993%/13.301%。路线与战斗仍有取舍，r02固定预算继续，r03预声明最小native端点标签修正不变；最终完整曲线与后续最弱漏斗需在交接后综合。

r02固定终态与先交接（20261018）：完整128代/268,435,456 fresh ticks，compute4356.956秒、wall5210.326秒、atomic_stop=false；8卡compute client为0的release receipt已核验，随即启动预声明r03新身份最小8卡smoke。r03尚未确认正式非零更新；r02最终CPU评估同时启动，不阻GPU后继。独立snapshot终态restore通过128份保存源码、plan、完整state/字段/finite与next RNG，无复现r01旧当前seal误拒绝。r03声明已清除旧模板smoke能力验收文字，准确记录1代技术smoke、128代预算、train1639578875/technical3501906343、实际fresh优化器/FIFO/新parent保留参考。正式非零确认后再综合r02最终完整曲线。

固定run `runs/je10_bun_20261018/control_route_v38_r02` 完成128代/268,435,456 ticks；compute=4356.956s、wall=5210.326s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r03`；本次scene-mixture选择依据 `Fixed production Review g8 has a proposed native step with -.39 gain versus +.39 alternative; g32 omits an equal +.39 direction. Reuse existing native safety-verifier endpoints to correct the legal set before changing budgets or reward.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。

r03正式真实启动（20261018）：smoke Actor/Critic L2=3.977050/2.254528，2,097,152ticks、finite/nonzero、8副本exact、fresh CPU restore通过后立即formal。正式g1/2,097,152ticks已原子提交，Actor/Critic L2=3.660164/2.223232、finite/nonzero、8副本exact及policy hash核验，FORMAL_NONZERO_CONFIRMED已写且主动通知；固定128代继续。CPU唯一owner已承接g0，r02全部8个固定点COMPLETE且snapshot restore通过，终态和后继正式receipt均齐备。r03严格DAgger标签现包含原生进度筛选，audit的accepted语义较r02增加native progress条件；冻结JS评估oracle保持旧协议，方向一致率不能等同新证书覆盖率。

相位恢复候选TDD（20261018，未安装）：固定Review154–160的4.0099/3.8299周期2序列已写纯状态RED→GREEN四项，定义近威胁distance≤4立即切对抗、已进入后需distance>5才恢复返程，敌死亡/糖泡受控立即解除，self死亡/受控/失包清空carrier记忆。仅纯函数`carrier_phase_hysteresis.advance`通过，不含环境/PBRS/DAgger一致接线、跨rollout保存恢复或效果证明；r03未引用该模块，后续若完整曲线支持须独立proposal/TDD/封印后在固定终态安装。r03 g0全部固定指标与r02 g128 exact相同，CPU下一点4，GPU第二代首次编译继续，未中断。

相位恢复候选接线进展（20261018，仍未安装）：固定Review阈值抖动的纯状态逻辑及PBRS前后endpoint配对已GREEN；真实原生transition的reward-state/reset树结构、JIT rollout分段pickle恢复、两席独立状态及终局清理已通过。审计发现旧return teacher独立于近敌相位，候选现把该teacher与PBRS、learner-state DAgger接到同一carrier gate，新增一致性RED后正在复核GREEN。候选只作为固定终态后的安装器，不修改r03任何封印数值源码；r03 g14、29,360,128ticks，finite/nonzero，唯一CPU正在g4且无failed shard。行政交接支持已准备，尚未替换当前owner或声明最终后继配方；不把测试当能力门禁。

r03 g4固定异步点（20261018）：唯一CPU owner已COMPLETE并进入g8；GPU已过g15且健康，不等待评估。Hunter难/中/易2/2/7（各64），Python Tactical9/64；相对父点分别-2/+1/-4及-1。路线交付death2、sugar2、drop13、near16/mid16/far13、recovery2、full4（各16），相对父点-1/-4/+2/0/0/0/-1/-1。drop首方向16/16、持续3353/5465、pickup16/16、delivery13/16；full窗口→目标13/13、首方向8/10、持续1377/1912、pickup9/16、delivery4/16、death5/self0。full首次控制29/归因12/finalpop29/active26/causal12/surviving11，初始化受控pop另计，不混分母。原墙stock拾取8/8、return target2065ticks、oracle0、delivery0/8，结构缺口未消失。固定Review掉包首偏航动作前80/帧81，C_RETURN距33.93，oracle下/实际左，近似route+.000207422、reward-GAE-.022039126、前帧差-.0000148453；return→combat10/反向9。训练control gross5560.75/net5334.75、route+2225.867/-2811.159、capture127968；DAgger active1389703/selected89829/accepted54194/firstquery1450（本run accepted含native progress）。最终执行current/ring/fixed league占有效ticks2.457%/5.362%/10.949%，底层条件13.126%/28.919%/57.955%分列。早期drop改善但full/部分战斗未改善，只诊断、不停训或择优。

r04预声明（20261018）：新ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r04`，父点为r03固定最终g128 raw Actor/Critic/完整pool，train seed3688535036、technical seed1017883248，8卡固定128代/268,435,456 fresh ticks。唯一配方因素是“统一持包对抗/返程相位状态与恢复滞后”，同时把旧return teacher的权限与PBRS及learner-state DAgger对齐；近敌distance≤4立即对抗，进入后直到distance>5才恢复，敌死亡/受控立即恢复，self死亡/受控/失包/终局/reset清理。保留r03原生端点证书，reset mix .35 combat/.20 bridge/.35 full/.10 native routes，奖励/DAgger系数及query预算/网络/PPO不变。Actor不读phase记忆，非recurrent，不能把辅助状态接线通过当作策略已会稳定切换。23项focused RED→GREEN含真实原生transition/reset树、JIT分段保存恢复、两席状态、三种监督一致性及仅固定终态后安装边界，凭据与候选source hash存r03 `PHASE_CANDIDATE_TDD.json` / `NEXT_RECIPE_PROPOSAL.json`。行政owner仅自身重载attach，原GPU/CPU进程未变，数值source/plan seal exact；未提前安装或中断r03。

后继启动与异步协议继续无门禁：r03健康照常完成；固定终态核验8卡释放后立即安装已准备r04，1代最小技术smoke/fresh restore后直接正式训练，formal g1 finite/nonzero/8副本exact/hash后主动确认。恢复checkpoint每4代；CPU唯一全局owner按0/4/8/16/32/64/96/128跟随，Tactical v2及真实JS Hunter难/中/易各64、八类固定路线各16，固定seed/出生/Review链路不变。COMPLETE为消费marker，EVALUATION_COMPLETE为全固定点评估终态，报告与评估均不阻塞GPU；中间点只准备下一proposal。r03最终整段曲线在r04正式有效更新后综合，继续按最弱漏斗及保留指标准备后继，不因短点不佳停健康run。fresh Adam/RNG/env/DAgger FIFO与新parent retention参考照实记录，顺序继承不能声称同父对照因果。

原墙开路候选准备（20261018，未安装）：`route_frontier.plan`四项RED→GREEN仅证明离散规划：已有开放绕路优先；仅断连时选当前位置可达、加权最短第一块brick之前的开放staging等优集合；硬墙不造路、不同spawn独立；真实开墙后恢复home目标。它不授权走进brick、不授予放泡奖励，尚未有合法安全joint move/ability证书、learner-state查询/能力损失接线及完整TDD，因此不纳入r04，不能称为已解决原墙stock返程。后续在健康GPU主线持续期间准备这一最强结构缺口。

r03 g8固定异步点（20261018）：CPU已COMPLETE并顺序进入g16，GPU继续g24/50,331,648ticks，未停。Hunter难/中/易1/3/7（各64）、Tactical10/64；相对父点-3/+2/-4及0，相对g4 -1/+1/0及+1。路线交付death4/sugar7/drop13/near16/mid16/far12/recovery6/full8（各16），父点变化+1/+1/+2/0/0/-1/+3/+3，前点+2/+5/0/0/0/-1/+4/+4。drop首方向16/16、持续3229/4772、pickup16/16、death1/self0；full首方向6/12、持续1750/2435、pickup10/16、delivery8/16、death6/self0，首次控制37/归因14/finalpop37/active31/causal14/surviving14。原墙stock仍pickup8/8、return target2889tick、oracle0、delivery0/8，不能把总体stock改善冒充该组合已会返程。固定掉包Review首偏航动作前80/帧81，C_RETURN距33.88、oracle下/实际左，route近似+.000209375、GAE-.017710178、相邻变化+.000220212；return→combat19/反向18，比g4切换更频繁。full首偏航1266/帧1267、距29.84、oracle下/实际上、近似GAE变化-.000530248。累计control gross15420.5/net13549.75，route+5292.538/-7231.046/net-1938.508，capture223080；DAgger active3208134/selected184541/native-safe-progress accepted111523/query2553。最终执行current/ring/fixed league占有效ticks3.100%/3.212%/10.864%；底层条件19.248%/17.244%/63.507%另列。早期full/恢复局部恢复且战斗保留仍分化；保持固定128预算，不依据这一点停训或择优。

r03 g16固定异步点（20261018）：COMPLETE后消费，CPU顺序进入g32；GPU已继续g36/75,497,472ticks，无健康run中断。Hunter难/中/易4/3/5（各64）、Tactical10/64，父点变化0/+2/-6及0，前点+3/0/-2及0。交付death5/sugar2/drop12/near16/mid16/far15/recovery6/full7（各16），父点+2/-4/+1/0/0/+2/+3/+2，前点+1/-5/-1/0/0/+3/0/-1。drop首方向16/16、持续2683/3497、pickup16/16、death3/self0；full首方向6/11、持续2559/3398、pickup9/16、delivery7/16、death7/self0，首次控制43/归因18/finalpop42/active32/causal17/surviving17。原墙stock拾取8/8、返程target4150tick、oracle0、delivery0/8。固定Review掉包首偏航动作前64/帧65，A追包距6.95、oracle上/实际左，近似route+.001291016、GAE+.012498867、相邻变化-.001399281；full首偏航820/帧821，stock距8.2571、oracle左/实际右，route-.007041406、GAE+.000349188、相邻变化-.433294498（JS重建近似，非训练Q）。累计control gross33433/net27734.5、route+10892.468/-15980.843、capture427392；DAgger active6541176/selected380064/native-safe-progress accepted229031/query4658。实际执行current/ring/fixed league占有效ticks2.079%/5.234%/10.255%，底层条件12.611%/29.013%/58.376%分列。

新根因证据（20261018，候选未安装）：同一r03 g8固定Review seed2026101800动作前80精确hash匹配重放，掉包返程位置[5.94,9.82]，实际左native gain+.18、下+.82；full动作前1266实际上gain-.18、下+.18。保留原对手/ability的一步fork只读诊断不构成额外评估owner，非长期Q/完整安全证书。对同一固定地图场在[5.999999,9.82]与[6.000001,9.82]，移动2e-6产生旧距离jump .640002；JAX oracle与PBRS均RED复现。旧“连续”五邻格connector随floor候选变化并不C0连续，不能继续声称全域连续。独立连续场候选以共享角点+cell-center分片插值处理可通行边界、隔离对角断连、保留中心网格值与goal0；初版角点平均虽连续，却在1260原生无危险near-center状态产生44处无正进度方向，已用新增RED捕获，尚未选入后继。r04相位声明当前保留；只有连续场的方向信号与PBRS/oracle一致接线正确后，才会单独更新后继提案并保留旧声明证据，不与相位混改。当前r03固定预算照常完成。

r03 g32固定异步点（20261018）：全部COMPLETE、CPU等待g64；GPU继续g45，未以能力门禁停训。Hunter难/中/易4/0/10（各64）、Tactical8/64；相对父点0/-1/-1及-2，前点0/-3/+5及-2。交付death4/sugar4/drop9/near16/mid16/far12/recovery5/full6（各16），父点+1/-2/-2/0/0/-1/+2/+1，前点-1/+2/-3/0/0/-3/-1/-1。drop首方向8/16、持续2571/3458、pickup16/16、death4/self0；full首方向5/11、持续1634/2173、pickup10/16、delivery6/16、death4/self0，首次控制28/归因17/finalpop28/active23/causal17/surviving17。原墙stock仍pickup8/8、return target1941tick、oracle0、delivery0/8。固定Review掉包首偏航动作前1/帧2，A追包距31.61、oracle下/实际左，近似route-.003031641、GAE+.049530753、相邻变化+.002711809；本固定full game0全2400ticks均COMBAT，无路线机会且无首偏航，不能称其路线完美或换seed择优。累计control gross78197.75/net61142，route+19873.596/-31878.373，capture904848；DAgger active12025146/selected773324/native-safe-progress accepted480201/query9307。实际current/ring/fixed league占有效ticks2.889%/6.626%/10.874%，底层条件14.421%/32.017%/53.562%分列。

连续场候选追加证据（20261018，未安装）：初版角点均值与goal相邻角点归零造成对齐动作零/负进度，失败audit原件保留。修正为连接分量内最小值+.5、仅纯goal角点0、cell中心三角插值并加.125连续子格居中代价（格中心值不变，原生转向需要先对齐）；不是exact continuous geodesic或Q。双笼各4706个碰撞合法开放子格、两种速度1/1.3及原墙可达区57位置/笼均有正进度原生动作，dense audit v3记录；开放双笼各700起点、500步纯原生motor oracle全部到目标，无zero-progress，旧metric也700/700，故只证明候选信号可达而非模型或策略效果提升。四项基本场测试后追加native-alignment RED→GREEN，真实PBRS/终局归零与后继边界focused检查正在固定hash收据。

r04后继提案更新（20261018，取代上文相位配方、尚未安装）：在r03健康运行期间完成更近因的数值证据与22项focused RED→GREEN：旧JAX oracle及PBRS在真实固定Review格边界出现.640002/2e-6距离跳变。后继新ID/父点/seed/8卡128预算/场景mix/固定CPU协议保持，r04唯一因素改为“PBRS与路线oracle共用连续拓扑场”，包括局部连通角点min+.5、cell中心三角插值、.125连续原生转向居中代价。保留原生一步证书和既有相位阈值，不混入hysteresis。奖励及DAgger外系数、query预算、网络/PPO不变；field是规划potential非精确geodesic/Q。focused涵盖连续边界、对角阻隔、网格中心值、真实native对齐、stock/drop/home一致接线、真实reward transition/terminal phi0、固定终态与8卡release安装边界，完整hash凭据为r03 `CONTINUOUS_CANDIDATE_TDD.json`，22项全GREEN。原相位声明原件保留为 `PHASE_RECIPE_PROPOSAL_SUPERSEDED.json`，其实现与TDD保留供后续单独实验。行政owner再次仅自身attach重载新proposal，原GPU/CPU owner未重启，r03 numerical seal exact；不等能力评估、不热改健康run。

连续场限制与比较：无危险原生motor路径双笼各700/700达目标，旧metric也700/700，不能当模型能力提升。新的连续场按当前open/weighted静态图采样，无法替代安全开墙目标或ability监督；原墙stock仍是独立最弱结构漏斗。冻结JS统一评估继续使用旧场/阈值以保持协议可比，该oracle含已知floor边界跳变，首方向不一致须结合实际native端点诊断，不把旧JS一致率当新教师覆盖率。终态后先r04有效formal g1确认再综合r03最终完整曲线，下一轮继续最弱漏斗而不机械重复mix。

连续场持包速度补充（20261018）：原生移动在受控/持包状态采用absolute72，等效基础speed .6；已补0.5/.6/.65/1/1.3五档、双笼开放4706位置/笼及原墙可达57位置/笼，均存在正进度native一步，原始audit独立保留。相同固定Review g8掉包动作前80场，旧gain下+.82/左+.18/右-.06，候选下+.127499/左-.157497/右+.217503；full动作前1266旧上下-.18/+.18，候选-.162498/+.157501，实际上仍为反向。候选会改变teacher偏好，不能称为仅oracle无行为语义变化或将旧JS方向一致率直接当新teacher一致率；仍须安全证书和完整学习曲线。

r03固定中点（20261018）：g64/134,217,728 fresh ticks已原子提交，passed/devices8/replicas_exact均核验，已完成代数全部finite/nonzero。CPU唯一owner自动开始g64统一快筛及固定路线漏斗，GPU继续后64代；r04连续场提案source hash复核exact。不中断、不等CPU、不设能力门禁。固定终态仍先8卡release→r04技术smoke/正式真实非零→完整分析。

r03 g64固定异步点（20261018）：COMPLETE后消费，GPU已继续g76；CPU等待g96。Hunter难/中/易3/2/11（各64）、Tactical6/64；父点-1/+1/0及-4，前点-1/+2/+1及-2。交付death4/sugar1/drop12/near16/mid16/far10/recovery7/full2（各16），父点+1/-5/+1/0/0/-3/+4/-3，前点0/-3/+3/0/0/-2/+2/-4。drop首方向16/16、持续3043/3718、pickup16/16、death1/self0；full窗口→目标11/11、首方向3/8、持续1439/1789、pickup7/16、delivery2/16、death8/self0，首次控制40/归因17/finalpop40/active32/causal17/surviving16。固定full game0仍全2400tick COMBAT、无路线机会，无首偏航不是路线成功。drop首偏航动作前80/帧81，C_RETURN距33.93、oracle下/实际左，近似route+.000207422、GAE-.016118577、相邻变化+.000309305。

必须更新原墙结论：此前早期点零返程oracle不代表此后永久为零。g64 death-stock原墙组合pickup8/8、return target2847tick、oracle553、match277、progress274，来自固定game4/8/12共三局（188/181/184机会），仍delivery0/8、death6/self0。sugar-stock原墙组合也有一局185次机会，仍0/8交付。固定game0精确Review重放核验：初始81块brick到最终73，动作前34/137/176/220/333共五次真实opening，包含敌方仍死亡时34处开一砖；该局return844tick但oracle0，说明“有开墙动作/事件”仍不等于目标连通或有效返程。未对非固定game4/8/12另造择优录像，也未仅从机会数推断开墙施害者；其机会来源需以后精确provenance区分。

累计control gross161590/net120992.75、route+34880.944/-60018.421、capture1867152；DAgger active21382482/selected1559975/native-safe-progress accepted1012582/query19035。实际current/ring/fixed league占有效ticks3.111%/7.685%/12.659%，底层条件13.483%/32.504%/54.013%分列。native full每百万有效ticks窗口1069.843、kill629.345、库存/native steal417.076、custody367.204、capture296.003，训练JAX收益仍未转成JS full稳定改善。该点继续暴露安全、目标有效开路及跨sim保持问题；不提前停健康run，r04仍先修复已实证的不连续方向信号，后续独立解开路/安全漏斗。

固定Review开墙归因补证（20261018）：只重放r03 g64固定game0/seed2026101800的唯一production Review Player，1200tick动作前后hash逐项exact；观察原生resolver返回source并保留穿过4tick brick linger的来源，未生成新对局或择优录像。8块实际移除砖中，cell83由learner的cell68糖泡在动作前30爆炸覆盖、34开放，owner0/cause mask1；其余7块由enemy糖泡覆盖（133→137、172→176、216→220、329→333），owner1/cause mask2。各次移除前后己笼连通分量均仅8格，learner到己笼距离均INF；所以“存在真实开墙”并不意味着有效返程开路，且本固定样本大部分开墙来自对手。凭据 `runs/je10_bun_20261018/control_route_v38_r03/REVIEW_G64_NATIVE_STOCK_TERRAIN_CAUSES.json`；这是simulator source/chain provenance，不是安全ability证书，也不能推广归因到其他seed或g64另三局机会。r03已过g84仍finite/nonzero、数值source/plan seal exact；不热改配方或停训。

监督目标冲突诊断（20261018，未改运行配方）：仅消费r03 g64 COMPLETE之前的1024条按遇到顺序截取偏航样本，kind3返程283条中252条route即时负、30条正，kind1敌掉包297条中160条正、kind2库存237条中84条正、kind4己掉包207条中139条正；样本每代first16、有设备/时间偏序及同episode重复，不能当总体错误率或“第一唯一偏航”。kind4中200条同时PBRS phase1 STOCK，7条NONE。源码与构造合法原生状态复现：敌死亡、有库存、己方归因掉包在左时，DAgger kind4指左，PBRS选敌笼stock上/右；是目标身份冲突而非同一目标势能分辨率问题。凭据 `LEARNER_DEVIATION_G64_SUMMARY.json` / `OWN_RECOVERY_OBJECTIVE_CONFLICT.json`位于r03 run；后者是明确构造状态、不是额外评估或录像、不冒称真实首偏航hash复现。该缺口可影响保留/目标选择，列为后续独立最小因素候选；r04只修连续场，暂不把目标优先级、相位与开墙混改。即时正PBRS并不必然证明反向刷分（折扣/offset和native等优集合不同），近似advantage也非因果Q。

r03 g96固定异步点（20261018）：统一COMPLETE后消费，GPU继续g107，CPU等待最终g128；无能力门禁。Hunter难/中/易1/4/11（各64）、Tactical6/64；相对父点-3/+3/0及-4，相对g64 -2/+2/0及0。交付death5/sugar6/drop8/near16/mid16/far10/recovery4/full6（各16），相对父点+2/0/-3/0/0/-3/+1/+1，前点+1/+5/-4/0/0/0/-3/+4。drop首方向16/16、持续3019/4335、pickup16/16、death2/self0；full窗口→目标14/14、首方向8/10、持续2182/2643、pickup10/16、delivery6/16、death10/self0；恢复切换38/47。full首次控制48/唯一归因16/finalpop48/active39/causal16/surviving15，初始化fixture pop不混归因分母。原墙death-stock拾取8/8、return1933tick、oracle16/match8、delivery0/8、death1；sugar-stock原墙return2283tick、oracle0、delivery0/8，不能把总体sugar恢复冒充原墙路线解决。

固定Review g96 real_drop首偏航动作前81/帧82，C_RETURN、距33.11、旧oracle下/右、实际左，route重建-.002623047、零值reward-GAE-.008556670、相邻变化-.005650900；return→combat5/反向5。full固定game0仍全2400tick COMBAT，无路线机会、不能称路线成功。唯一Review端点fork保留原对手及ability，分别real_drop81实际左旧gain-.18/新-.157501、death-stock1184实际上旧-.18/新-.127499、recovery1198实际左旧-.179996/新-.337498；候选均提供正进度替代方向，非长期Q或完整安全证书。原始逐hash端点与场比较为r03 `REVIEW_G96_PHASE_NATIVE_ENDPOINTS.json` / `REVIEW_G96_PHASE_FIELD_COMPARISON.json`。

累计control gross244790.5/net181504.5、route+49861.058/-87848.231/net-37987.174、capture2811264，终局仍主导。DAgger active30748993/selected2346336/native-safe-progress accepted1546319/first-query28545；真实几何候选独立分母未记录，query跨rollout可重复同episode。实际override后current/ring/fixed league占有效ticks2.832%/8.572%/13.029%，底层条件11.893%/34.760%/53.347%分列。JAX natural-full每百万有效ticks窗口1008.744、kill622.539、库存/native steal436.176、custody384.990、capture310.874；与JS自然完整对局不混合。前段局部改善尚未稳定，r03照常完成，r04连续场单因素预声明保持。

后续目标一致性纯候选4项RED→GREEN（20261018）：`route_objective_consistency.align_recovery`复用既有反origin drop距离缓存，对齐已归因且当前存在的己方掉包目标，负identity命名空间与enemy/stock/carry不碰撞；敌方掉包及持包返程优先，native窗口关闭/自身受控或死亡/持包/foreign credit/stale nonce拒绝。覆盖双席位、目标切换offset连续及terminal phi0。它没有installer或production transition/reset接线，没有进入r04；TDD收据 `OBJECTIVE_CANDIDATE_TDD.json`保留初始RED和一次测试期望修正。待完整曲线与r04信号修正后的实际覆盖再决定是否单独选择，不能将候选pure GREEN当能力门禁或完成原墙开路。

目标一致性候选补充（20261018）：追加第5项真实native env.step构造transition检验，现有己包teacher指左，原PBRS选stock，对齐后左移的同一归因目标势能上升，5项全GREEN（`OBJECTIVE_CANDIDATE_GREEN_v3.log`及更新hash收据）。这仍无production installer/full reset接线、未进入r04，不改变既有连续场后继。

固定run `runs/je10_bun_20261018/control_route_v38_r03` 完成128代/268,435,456 ticks；compute=4397.280s、wall=5240.626s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r04`；本次后继配方选择依据 `Fixed r03 g8 Review and actual JAX/PBRS focused RED reproduce .640002 metric jump for 2e-6 motion. Correct the direction signal shared by PBRS, route teacher and native-step DAgger before separately changing phase state; full curves show current recipe has not stabilized across tasks.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。

r03完整固定run终态综合（20261018，在r04正式g1真实非零后）：128代/268,435,456 fresh ticks完整完成，8卡release收据compute_clients0；最终snapshot恢复passed、129份保存源码核验一致。已先接r04，正式g1 Actor/Critic update L2=4.130642/2.328964（各8副本exact）、finite、policy hash及COMMITTED核验。r04当前共享连续场单因素，train3688535036/technical1017883248，固定8卡128代；恢复每4代，唯一CPU owner按0/4/8/16/32/64/96/128独立评估，不等能力阈值。没有恢复v36 stash，没有commit/stage/run入Git。

| g | JS Hunter难/中/易（各64） | Python Tactical（64） | death/sugar/drop交付（各16） | near/mid/far（各16） | recovery/full（各16） |
|---:|---|---:|---|---|---|
| 0 | 4/1/11 | 10 | 3/6/11 | 16/16/13 | 3/5 |
| 4 | 2/2/7 | 9 | 2/2/13 | 16/16/13 | 2/4 |
| 8 | 1/3/7 | 10 | 4/7/13 | 16/16/12 | 6/8 |
| 16 | 4/3/5 | 10 | 5/2/12 | 16/16/15 | 6/7 |
| 32 | 4/0/10 | 8 | 4/4/9 | 16/16/12 | 5/6 |
| 64 | 3/2/11 | 6 | 4/1/12 | 16/16/10 | 7/2 |
| 96 | 1/4/11 | 6 | 5/6/8 | 16/16/10 | 4/6 |
| 128 | 2/5/9 | 9 | 5/5/9 | 16/16/13 | 3/4 |

最终相对父点：Hunter -2/+4/-2、Tactical-1；death/sugar/drop +2/-1/-2，near/mid/far0/0/0，recovery/full0/-1。相对g96：Hunter+1/+1/-2、Tactical+3；death/sugar/drop0/-1/+1、far+3、recovery/full-1/-2。整段drop11→13→13→12→9→12→8→9、full5→4→8→7→6→2→6→4；没有稳定跨场景提升，不能选g8最佳点冒充最终或因短点停止。完整曲线 `reports/je10_r03_curve_20261018.png`。

最终固定漏斗：full窗口→目标12/12、首方向6/10、持续1543/1881、进度1576/5083目标tick、pickup9/16→delivery4/16、恢复切换33/37、death10/self0；首控39→唯一归因8，finalpop39/active28/因果kill8/存活8，fixture与新控制严格分母分开。drop首方向16/16、持续2803/3737、pickup16/16→delivery9/16、death3/self0。原墙death/sugar/recovery均pickup8/8、return1967/3263/2231tick、oracle0、delivery0/8；该组合仍是结构缺口，早期g64确有部分开路机会的事实保留。近/中返程16/16保持，far13/16、death2/self0；carry fixture初始持包无拾取阶段。JS Hunter hard最终self1/death_episode2，normal trade1、easy self0；Python Tactical death1/self0，均独立诊断。

最终唯一Review：掉包首偏航动作前64/帧65，A_ENEMY_DROP距6.95、旧oracle上/实际左；即时重建route+.001291016、零值reward-GAE+.012801261、相邻变化-.001382725；native实际左旧gain+.29/连续候选+.68125，上+.39/+.73125，说明这是相对最优方向偏差而非必然无推进。首返程偏航80实际左旧+.18/新-.157497，候选右+.267498/下+.215000。保留原对手及ability的一步fork与hash见r03 `REVIEW_G128_PHASE_NATIVE_ENDPOINTS.json` / `REVIEW_G128_PHASE_FIELD_COMPARISON.json`，不是长期Q/完整安全证书。fixed full game0在914tick结束、全COMBAT无路线机会，不能把无首偏航解读成路线成功或换seed。

最终奖励分解：entry317358.5/residual8883.25/clawback-84750.25，control gross326241.75/net241491.5；残余仅gross约2.72%。route+64854.928/-115765.287/net-50910.360；capture终局3778128，仍大于control gross约11.6倍。负net本身不能证明PBRS无用或应直接加系数，须连同offset/折扣/terminal与首次偏航解读。DAgger active40172960→selected3132993→native-safe progress accepted2075890→first-query37574，accepted drop1033398/stock212755/return828451/delivery1286；既非独立几何覆盖，也非唯一episode数。

最终实际current/ring/fixed league占有效ticks2.848%/8.598%/13.515%，底层条件11.632%/34.244%/54.125%分列，覆盖所有128代。实际场景tick combat13.668%、bridge4.158%、full76.589%，reset .35/.20/.35不能冒充实际占用。JAX full每百万有效ticks：window976.615、kill616.487、steal447.412、custody393.469、capture319.078、真实loose pickup18.086；训练事件增长没有稳定转成JS自然完整局，不能合并Python/JS/JAX分母。

证据排序根因1：Static route-field discontinuity and nominal geometry misrank native endpoints. 最小后续工作：r04 already running: shared continuous field only, existing native certificate retained; whole fixed run finishes regardless of intermediate performance. 限制：Sequential final-parent continuation, not same-parent randomized causality.

证据排序根因2：Disconnected native-wall return has no useful reachable movement objective or goal-directed ability teaching. 最小后续工作：Prepare reachable first-brick staging plus legal safe joint ability witness and learner-state supervision; planner-only GREEN is insufficient to install. 限制：Seat and stock geometry are confounded; do not call this a seat effect or infer every opening owner from opportunity counts.

证据排序根因3：Phase switching and distinct target identities can create inconsistent supervision. 最小后续工作：Use r04 fixed-phase deviations to select one phase or target-consistency factor; do not combine them or add reward coefficients. 限制：Capped learner samples are not population failure rates; no production alignment installer or full transition/reset integration yet.

证据排序根因4：Sparse stage-stratified query coverage and actual opponent/reset exposure can limit transfer and combat retention. 最小后续工作：Inspect rejected/accepted first-deviation stages and threat conditions before changing bounded query budget or scene mixture; keep simulator populations separate. 限制：Independent geometric candidate coverage not logged, same episode may reappear across rollout chunks; query/candidate alone does not locate rejection cause.

r05后继预声明：ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r05`，父r04最终g128 raw Actor/Critic/完整pool，train seed2340928073、technical seed4008119116，8卡固定128代/268,435,456 ticks，恢复每4代；快筛与固定八路线、seed/greedy/64 spawns、唯一Review、owner/COMPLETE marker与早中晚最终点完全沿用。数值单因素候选保持reset .35/.20/.35/.10；若证据选择曝光混合因素，须在安装前单独写明唯一变化。按r04已完成固定点及r03整段证据准备最小proposal/TDD，r04健康照常完成，终态先8卡release→后继技术smoke→正式真实非零，再综合最终评估；无admission/promotion/能力门禁。运行级前瞻收据为r04 `SUCCESSOR_PREDECLARATION.json`。

分析分母修正：window→target改为有window且有target的episode交集/有window episode数；初始持包、无window的target不再进入分子，避免carry_far出现>1伪比例。该统计只能表示episode共现，非逐window事件因果转化；两项RED→GREEN，r01/r02/r03完成曲线已重算，原始训练/评估产物未改。

目标一致性接线候选（20261018，r04未热改）：8项focused全GREEN，凭据r04 `OBJECTIVE_RECIPE_CANDIDATE_TDD.json`，包括native PBRS reward/terminal phi0、原reset树结构与仅前run128终态且8卡released可安装的AST边界。installer只包装PBRS目标身份及复用已有反origin距离缓存，不新增env.step/actor输入/aux字段，不改reward/DAgger系数或预算；敌方掉包和持包返程优先级保持，stale/foreign/absent/closed window拒绝。仍保留weighted PBRS与brick-blocked oracle地图语义差异，不能承诺全球同一安全动作集。尚未接入自动relay或选择为r05；r03旧pure候选收据是当时版本，当前测试抽出共享native fixture后的hash及全部8项新收据保留于r04。r04继续唯一连续场配方、当前g7健康，g4唯一CPU快筛中。

原墙结构根因进一步定位（20261018，未改r04）：原生 `je07_bun_env.legal_mask` 的can_place_bomb明确要求bun_carried<0，step也把持包者放泡强制清零。候选native witness首轮正例把learner设为持敌包，所有placement被拒绝；这是发现非法前提，不应放松合法mask或改变游戏规则。先前g64 learner自泡在拾包后34tick开放，也来自此前空手放置/30tick爆炸，并不证明持包能放泡。因此当原墙stock5tick即拾包且返程图断连，事后仅给frontier放泡标签不可能可达；需在合法控制窗口、空手阶段先准备能返程的走廊，再取库存包，或等待已真实放置的开墙事件建立通路。原candidate正例已改为空手并追加“持包必须无放泡标签”；原失败日志保留为r04 `FRONTIER_NATIVE_WITNESS_GREEN_v2.log`。该约束比泛称“能力head缺监督”更近因，应升为原墙后续proposal核心；不热改r04、不设能力门禁。

r04 g4固定异步点（20261018，COMPLETE后消费，GPU继续g22）：Hunter难/中/易4/2/7（各64）、Tactical8/64，父点变化+2/-3/-2及-1；交付death7/sugar4/drop7/near16/mid16/far7/recovery4/full5（各16），父点+2/-1/-2/0/0/-6/+1/+1。drop旧冻结JS首方向8/16、持续3212/4915、pickup16/16、death3/self0；full窗口→目标9/9、首方向4/8、持续1134/1596、pickup6/16→delivery5/16、death5/self0，恢复切换20/24。full首次控制24/唯一归因8/finalpop24/active21/因果kill8/存活8。原墙death-stock pickup8/8、return2896tick、oracle0、delivery0/8；fixed game0返程→对抗45/反向45，原墙recovery42/41，需检查阈值抖动与真正威胁，不能取消近敌对抗。

必须结合新场解读冻结JS一致率：固定掉包game0首偏航动作前1，旧JS oracle下、实际左；保留原对手/ability的真实native端点，新连续场gain左+.341251/下+.241251，旧场左-.389999/下+.390001，说明这处“旧协议不一致”实际符合新field方向。返程动作前80实际左仍新gain-.157497（右+.267498/下+.215000），是真实新场偏航。fixed full game0 stock2162实际等待、新gain0而替代方向正；return2192实际下，新gain+.157497优于右+.027500，旧场却下-.18/右+.18。不能直接将旧JS oracle首方向8/16当新teacher错误率；需分开方向场协议与实际交付/安全指标。逐hash唯一Review证据r04 `REVIEW_G4_PHASE_NATIVE_ENDPOINTS.json` / `REVIEW_G4_PHASE_FIELD_COMPARISON.json`；GAE仍为JS重建近似，非训练Q。

截至g4训练：control gross6155/net5780.75、route+2100.147/-2640.917、capture125040。DAgger active1355153/selected88205/native-safe-progress accepted50802/first-query2902；实际override后current/ring/fixed league占有效ticks2.304%/7.491%/8.911%，底层条件12.560%/40.704%/46.736%分列。局部full/death改善与远返程/drop退化并存；不以短点停训，固定128预算继续。

原生开路候选进展（20261018，未安装）：frontier planner7项及40步实际native见证2项全GREEN，r04 `FRONTIER_PREPARATION_CANDIDATE_TDD.json`固定source hash及RED/失败/最终GREEN日志。见证枚举5个合法初始move+place与4个直线逃生续行（阻塞时wait）；只接受自方所放泡覆盖并实际移除所需第一砖、40步内自身存活不受控/无自伤、敌始终受控、泡/火焰/残骸全部结束、固定原起点weighted topology严格改善的joint集合。正例空手，持包拒绝、短控制窗口拒绝、既有危险场拒绝、无目标/无目标砖拒绝。仅clear场且窗口>40、固定native RNG与有限续行，不是对活跃对手/所有RNG的通用安全证明，仍没有learner-state挑选、能力loss、PBRS准备相位或relay接线。下一步目标是拾库存前准备合法返程走廊，不是在持包状态授予非法放泡。

拾库存前走廊纯planner候选（20261018，未安装）：4项新RED→GREEN并联既有frontier7项，共11项通过，凭据r04 `PRESTOCK_PLANNER_CANDIDATE_TDD.json`。ready要求当前位置、真实库存笼与己笼在开放图连通；近敌笼但返程断连时先home frontier，准备时排除库存笼以防过早拾取；处于home分量则先forward frontier，已有绕路或真实开墙后才direct stock。inactive/硬墙断连不造标签。native eligibility、learner-state查询、PBRS/DAgger/ability接线尚未完成，不声称已解决原墙问题。

r04 g8固定异步点（20261018）：Hunter难/中/易2/3/10、Tactical8，各64；相对父点0/-2/+1及-1，相对g4为-2/+1/+3及0。death/sugar/drop交付4/3/7，near/mid/far16/16/9，recovery/full2/4（各16）。full窗口→目标13/13（episode共现）、首方向6/11、持续1441/1911、pickup11/16→delivery4/16，death3/self0、恢复切换37/43；首次控制44/归因12/pop44/active40/因果kill12/存活12。drop旧JS首方向8/16、持续2854/4299、pickup16/16、death2/self0；旧JS方向场缺陷仍保留，不能把它等同新teacher。原墙death-stock返程3549ticks、oracle0、交付0/8，recovery返程3631ticks同样0机会/0交付，尚需拾包前开路准备。

截至g8训练control gross16098.5/net13657.75、route+4986.755/-6876.400、capture终局209616，终局继续主导。DAgger active3150393/selected182029/native-safe-progress accepted105420/first-query5613；accepted stock仅3046，独立几何候选分母未记录，不将selected当全部几何覆盖。override后current/ring/fixed league占有效tick1.887%/7.552%/7.652%，底层条件10.843%/45.484%/43.673%分列。固定full game0全2400ticks COMBAT，无路线机会，不能将“无首偏航”当路线成功。固定drop动作前1左仍须结合新field端点解读；返程动作前80为后段核查点。所有中间结果只诊断，r04健康固定128代继续。

r05最小因素已具体预声明（20261018，尚未安装）：保持既定父r04最终g128、train2340928073/technical4008119116、8卡128代、reset .35/.20/.35/.10及固定快筛；仅将PBRS己方真实归因掉包目标与既有learner-state DAgger teacher对齐，复用反origin距离cache，保留连续场与native一步证书，不改系数、query预算、Actor输入或环境规则。直接证据是合法native状态teacher左而PBRS敌笼上/右，以及r03 capped样本207条own-recovery中200条PBRS STOCK；该抽样非总体频率。18项focused TDD全GREEN（原8项接线+继承、install-before-prepare、plan/declaration/source seal、技术交接），收据r04 `OBJECTIVE_RELAY_CANDIDATE_TDD.json`，具体`NEXT_RECIPE_PROPOSAL.json`。取库存前开路与相位hysteresis保持独立候选，未混入r05。当前r04固定连续场run继续，不等待能力阈值；终态先release→r05技术smoke→formal真实非零，再综合最终评估。

拾包前候选新增native接线与移动标签13项GREEN（20261018，未入训练）：两席检查self空手/自由/alive、敌死或原生受控窗口、真实敌库存、非terminal和已有loose优先级；准备途中排除库存接触，到staging中心不教wait。候选endpoint检查支持砖旁子格居中，仍必须通过真实native速度/安全筛选；没有能力loss、learner-state replay或生产相位接线。凭据r04 `PRESTOCK_NATIVE_CANDIDATE_TDD.json`，未将纯planner测试通过当作能力或安装门禁。

r04 g16完整固定点（20261018）：Hunter难/中/易5/2/17、Tactical6，各64；相对父点+3/-3/+8及-3，前点g8+3/-1/+7及-2。交付death3/sugar4/drop7/near16/mid16/far14/recovery5/full3，各16；相对父点-2/-1/-2/0/0/+1/+2/-1，前点-1/+1/0/0/0/+5/+3/-1。drop窗口目标16/16、首方向16/16、持续2780/3984、pickup16/16→delivery7/16、death2/self0。full窗口目标9/9、首方向3/5、持续788/994、pickup5/16→delivery3/16、death2/self0，恢复18/23；首控制23/归因10/pop23/active19/因果kill10/存活10。原墙death-stock返程2318ticks、oracle0、交付0/8；近中返程保持与远程恢复尚未转成full稳定改善，健康固定run继续。

唯一固定Review g16掉包首偏航动作前15实际下：旧场下/左gain各+.389999，新场下+.288748高于左+.266251，是旧协议不一致而非新field负进度；近似route+.002818359、reward-GAE .045088528，前帧差-.000364997。返程动作前80实际左仍新gain-.157497，右+.267498/下+.215000；近似JS route+.000207422、reward-GAE-.015708581，不能用旧JS折扣reward正值证明真实推进。phase返程→对抗/反向各6次，固定full game0全COMBAT无路线机会，均不择优替换seed。端点hash及原对手/ability保留于r04 `REVIEW_G16_PHASE_NATIVE_ENDPOINTS.json` / `REVIEW_G16_PHASE_FIELD_COMPARISON.json`。

截至g16：control gross34803.75/net28679.5，route+10410.906/-15376.869，capture终局401760；DAgger active6523026/selected376941/accepted221572/first-query10937，accepted drop115682/stock7402/return98400/delivery88，独立几何分母未知。override后current/ring/fixed league占有效ticks1.468%/8.323%/7.776%，底层条件8.532%/48.224%/43.244%分列。行政owner已仅重载attach以支持r05目标一致性dispatch，GPU worker marker exact、CPU owner未重启、数值source/plan seal exact；凭据r04 `ADMINISTRATIVE_OBJECTIVE_RELOAD.json`。

r04 g16真实learner snapshot候选覆盖（20261018，只读、不添评估对局）：以已COMPLETE的g16原始state.pkl并核验SHA，沿训练固定learner seat=environment%2的8192个rollout-end状态，空手/自由/alive/合法敌死亡或糖泡窗口/真实敌库存/无优先loose/非terminal共969。固定device/environment顺序前64个，ready8、forward frontier47、home frontier9，移动候选57；staged+clear+window>40为0，实际40步witness查询0、接受0。不能把这一受限快照的0查询当全轨迹0机会，也不能把57/64当无偏总体覆盖率；该证据支持先让空手learner学会合法staging并增加准备阶段日志，不能宣称现有ability witness已在训练分布可用。没有替换seed或启动额外失败录像链；保存于r04 `PRESTOCK_SNAPSHOT_G16_AUDIT.json`，candidate源码hash及完整native状态hash均保存，数值训练不受影响。

拾包前joint候选补证（20261018，未安装）：开路逃生途中提前拾包的RED已复现，native witness现要求完整40步保持空手（仅起步空手不足），原持包/短窗口/危险/错误目标拒绝仍GREEN。联合7项focused TDD通过：3项实际native witness、4项joint set-mass loss/权限测试；联合move×ability集合保留合法配对，不以两个head的独立边缘集合制造非法交叉动作；任何一个等优合法joint均可，不强制均匀；inactive/空集/非法ability梯度有限且为零；移动推进与staging开路仅有一个监督owner。凭据r04 `PRESTOCK_JOINT_CANDIDATE_TDD.json`。新源码尚未进入任何数值训练依赖；还缺与PBRS准备目标的统一接线、受限learner-state查询及replay/optimizer，不能把loss单测通过写成已训练能力。g16快照收据保留的是当时witness源码hash，历史结果不改写。

r04 g32完整固定点（20261018）：JS Hunter难/中/易0/2/7、Python Tactical8，各64；相对父点-2/-3/-2及-1，较g16为-5/0/-10及+2。路线交付death2/sugar2/drop9/near16/mid15/far11/recovery2/full3，各16；父点-3/-3/0/0/-1/-2/-1/-1，前点-1/-2/+2/0/-1/-3/-3/0。drop首方向16/16、持续3237/4676、pickup16/16→delivery9/16、death1/self0；full窗口目标11/11、首方向5/10、持续1079/1676、pickup7/16→delivery3/16、death6/self0、恢复29/34。full首控制34/归因12/pop34/active27/因果kill12/存活12；原墙death-stock仍pickup8/8、return3578ticks、oracle0、交付0/8。未见不同场景稳定提升，仍照常完成固定128代，不把低safe win当能力停训门禁。

固定Review g32掉包首方向动作前6实际左，新field gain+.406252高于下+.241251，旧field左/下各+.730001；不以冻结JS首偏航直接判新teacher错误。返程动作前82实际左，新-.157497而右+.267498/下+.215000，是真实新场偏航；JS重建route+.000176953、zero-value reward-GAE-.018437273（非训练Q），返程/对抗双向各5次。hash、原对手/ability一步fork及每阶段首偏航保存在r04 `REVIEW_G32_PHASE_NATIVE_ENDPOINTS.json` / `REVIEW_G32_PHASE_FIELD_COMPARISON.json`。

g32累计control gross79093/net61979，route+19318.824/-30951.029，capture终局855864；DAgger active12134748/selected770010/native-safe-progress accepted470750/first-query22414，accepted drop246103/stock27810/return196605/delivery232。实际override后current/ring/fixed league占有效tick1.262%/8.821%/10.403%，底层条件6.566%/44.424%/49.010%分列。r04 g1–g32固定截断样本之外没有择优seed：已完成g16边界内256条偏航样本中35条own-recovery全部PBRS STOCK，另两条return teacher活跃但PBRS NONE；native真实拾回自己包的RED确认原phase只覆盖敌包，拾回自己包后PBRS会冻结。后继目标链候选正在补足这一同一目标一致性缺口，保留原近威胁外层gate及terminal/reset，尚未热装进r04。

r04 g32 learner snapshot候选可达性（20261018，checkpoint SHA核验、未增加评估对局）：8192个真实rollout-end learner状态，合法空手库存窗口406；固定前64个ready11、home frontier9、forward frontier44，移动候选61，staged+clear+window>40仍0，因此witness queries/accepted均0。与g16分列，不能以两次截断快照外推全轨迹频率或声称ability已覆盖；r04 `PRESTOCK_SNAPSHOT_G32_AUDIT.json`保留状态hash/选择顺序/源码hash。接下来的准备阶段必须协调移动目标与PBRS，并为实际到达staging后的合法joint提供查询和replay，单纯提高原stock奖励或query总数不能让持包者合法放泡。

r05目标链最终候选补全（20261018，未安装，替代上方仅空手目标版本）：当前r04 g1–g16固定截断256条accepted偏航中，35条own-recovery均PBRS STOCK；两条return teacher活跃但PBRS NONE。真实native wait拾回己方掉包后，DAgger kind3仍可用而原PBRS phase0的RED已复现。修正同一目标链：空手时归因own-loose目标与teacher一致，真实持己方包后沿现有home cache继续CARRY/DELIVERY势能；原enemy-drop/carry优先级、近敌外层gate、phase/identity offset连续与terminal清零保持。20项focused全GREEN，收据r04 `OBJECTIVE_CHAIN_CANDIDATE_TDD.json`、`OBJECTIVE_CHAIN_GREEN.log`。`NEXT_RECIPE_PROPOSAL.json`已更新helper hash/证据/20项TDD，旧仅空手提案保存`OBJECTIVE_EMPTY_TARGET_PROPOSAL_SUPERSEDED.json`，不改写历史收据；scene mix、seed、父点、128代预算、固定异步协议及系数/query/Actor输入全部沿既定预声明。r04现有数值源码/plan未改，行政owner无需再次重启，终态读取已更新提案；无能力门禁。

r04固定中点g64（20261018）：134,217,728 fresh ticks checkpoint已原子提交，passed/devices8/replicas_exact核验，g1–g64全部finite/nonzero；唯一CPU owner已自动开始预声明g64完整快筛，GPU继续后半程。没有technical/owner failure，不设能力门禁，终态仍先核验8卡release并启动已预声明r05。

取库存前目标选择候选（20261018，未安装）：推箱占位RED确认原native adapter会把尚未真实移开的推箱当通路，已修正为当前hard包含pushable，仅实际transition打开后重算。14项planner/native/frontier测试GREEN；另外3项选择/势能测试GREEN：可达且拓扑未变时保留staging集合，目标/阶段边沿才推进revision，不因持续准备重复生成身份；真实opening切direct stock；terminal/reset全清；准备PBRS使用同一stage距离并保持offset边界连续，不覆盖已有drop/carry目标。17项收据r04 `PRESTOCK_SELECTION_CANDIDATE_TDD.json`。selection_state和phase5/6都尚未进入生产reward/DAgger，仍缺统一的native标签、查询配额、replay/optimizer接线，不能以独立模块GREEN声称完成路线能力。

r04 g32候选原生端点核查（20261018，未入训练）：保留同一checkpoint及固定64个state的选择顺序，每状态5个movement、ability0、原生敌wait、RNG6106做一步fork（非新对局/失败录像）。名义.18步长移动标签61/64，按实际native端点正进度集合59/64，17个集合不同；此批提前拾包moves涉及状态0，staged-clear-long仍0，无40步ability查询。证据r04 `PRESTOCK_SNAPSHOT_G32_NATIVE_AUDIT.json`，对所有状态保留hash与结果，未择成功样本。这里仅一步alive/free/无自伤/非terminal、当前合法mask和进度/拓扑过滤，没有额外time-expanded survivability，因此不是完整时间安全证书；生产准备接线必须继续合并既有bun_safety与native endpoint verifier，不能直接把这59条当安全teacher入训。结果支持保留原生端点计算，不用固定速度近似替代。

r04 g64中点完整评估（20261018，COMPLETE后消费）：Hunter难/中/易5/1/11、Tactical10，各64；相对父点+3/-4/+2及+1，前点g32+5/-1/+4及+2。交付death4/sugar4/drop8/near16/mid16/far14/recovery5/full5，各16；父点-1/-1/-1/0/0/+1/+2/+1，前点+2/+2/-1/0/+1/+3/+3/+2。drop首方向旧JS8/16、持续3129/4755、pickup16/16→delivery8/16、death3/self0、恢复16/18；full窗口目标11/11、首方向6/11、持续1261/1539、pickup12/16→delivery5/16、death3/self0、恢复26/33，首控33/归因7/pop33/active28/因果kill7/存活7。原墙death-stock pickup8/8、return1980tick、oracle0、delivery0/8，结构性返程缺口仍在。

唯一固定Review g64掉包首步1实际左仍新场+.341251优于下+.241251；返程80实际左新-.157497，右+.267498/下+.215000，是实际负进度。natural full首偏航C_RETURN1990实际下，新gain+.085266虽推进但小于右+.309639，属于非最大方向；旧场下-.139900/右+.180000。JS近似route-.001087504、zero-value reward-GAE+.000374341、前帧差+.0000204952，不代表训练Q。逐hash原生端点/新旧场对比保存在r04 `REVIEW_G64_PHASE_NATIVE_ENDPOINTS.json` / `REVIEW_G64_PHASE_FIELD_COMPARISON.json`，固定seed2026101800，不择优。

g64训练control gross159493.25/net120991.75，route+34089.489/-59129.749，capture终局1795200；actual tick combat13.771%/bridge3.989%/full76.574%，与reset权重分列。JAX full每百万有效full ticks窗口1076.912、kill653.660、真实loose pickup15.529、custody366.080、capture288.445，只作同JAX人口归一化，不与JS/Python混算。DAgger active21774474/selected1556585/accepted1000956/first-query46773；override后current/ring/fixed league占有效ticks1.745%/9.366%/12.415%，底层条件7.470%/40.926%/51.603%另列。

分相位机会覆盖新增分析（20261018）：4项RED→GREEN，r04 `PHASE_COVERAGE_TDD.json`，r01–r04已完成点重算且原评估/训练未改。g64 drop active1402536/accepted507370/query16161；stock active9753985/accepted91950/query8361；return active10617386/accepted401070/query22200；delivery active567/accepted566/query51。accepted/active为drop36.175%、stock0.943%、return3.777%、delivery99.824%的实际已认证标签覆盖，包含bounded选择与安全/进度筛选，绝不能叫独立几何覆盖率或全体候选的安全通过率；分相位selected/geometric未记录，仍null。四phase active/accepted/query分别与全量audit严格守恒，scope均为所有native route scenes；不能除以仅natural full visited，query跨rollout可重复episode。库存信号薄弱仍是重要漏斗，不宜只看总候选多就说监督充分。

g64真实snapshot准备候选：8192个状态中405个合法空手库存窗口，固定前64个ready14、forward45、home5，名义/原生一步候选均63/64，集合不同12，提前拾包状态0；staged+clear+长窗口0，witness查询0。仍为截断snapshot和一步非完整时间安全诊断，保存r04 `PRESTOCK_SNAPSHOT_G64_NATIVE_AUDIT.json`。中点full与部分战斗回升尚不等于全曲线稳定改善，r04健康继续固定128代，r05目标链配方不混入这些准备候选。

后继初始化语义补全（20261018，仅元数据，数值未改）：r05继承r04最终raw Actor/Critic和所有pool成员权重；不是整个训练state resume。除已记录fresh Adam/RNG/env/DAgger FIFO外，EMA重新设为raw父点，funnel/course计数、pool_counts/PFSP统计及distance_stage0也重建；完整pool指成员权重完整，不代表PFSP胜率历史继承。既有各轮初始化方式一致，此处补全透明口径；实际每轮对手source及比例仍按执行tick报告。r04 `NEXT_RECIPE_PROPOSAL.json`增加initialization_scope，helper hash、系数、配方/预算/seed不变，dispatch收据hash同步。隐私扫描在618个tracked文件及18个明确准备文件未见本机绝对路径、用户标识或内网IP；run/checkpoint/本地编排未纳入扫描，未stage或commit。

r04 g64固定返程偏航的FIFO覆盖补证（20261018，仅只读诊断）：从唯一production Review的game0/seed2026101800导出真实Actor观测，使用bun24-trap-timer-v2编码并精确核验state hash；只比较同一已完成g64 checkpoint的bounded FIFO，不新增对局或训练评估样本。5706条有效标签中return2500条。real_drop动作前80（约[5.89,9.82]，实际左）有13条同terrain/pushable/base、同持包origin且位置曼哈顿距离≤1的return标签，目标质量为下5/右8，接受实际左0；最近同图样本距.169897。完整Actor输入exact匹配0，其他全局/危险/对手特征仍可能不同，因此不能称监督冲突，但已不能以“该返程区域完全无teacher几何覆盖”解释重复负进度。natural_full动作前1990相同几何筛选0条，需分任务看分布缺口。收据r04 `REVIEW_G64_FIFO_COVERAGE.json`，明确bounded FIFO快照不是整段轨迹/总体覆盖率，未把固定评估状态入训；r04继续健康固定预算。

r04 g64 FIFO拟合补证（20261018）：同checkpoint raw Actor在CPU对上述固定Review输入重算，greedy与录像动作均一致（drop左概率.929852；full下.506060、右.445253），编码/推理核验通过。有效FIFO内kind1/2/3/4 greedy命中分别1174/1622、770/1241、1549/2500、232/343，对应target概率均值.6664/.6066/.6047/.6743；这些是当前参数对缓存训练标签的样本内拟合，非holdout能力。掉包返程附近13条几何匹配teacher样本，仅6条greedy落入target、mean target probability .506187；最近同图样本teacher右，但当前Actor wait概率.992265。进一步支持检查既有监督的有效拟合/与其他loss竞争，而非只增加oracle机会计数。危险/对手/全局差异仍未消除，不能据此认定同输入矛盾或决定系数；r05仍只做预声明目标链对齐，未混入权重/采样改动。

r06连续预算预声明（20261018，r05尚未启动）：ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r06`，父r05最终g128 raw Actor/Critic/所有pool成员权重，train2236293928/technical3840110319，8卡128代/268,435,456 fresh ticks，恢复每4代。固定CPU/JS唯一owner继续0/4/8/16/32/64/96/128、既有Tactical/Hunter各64与八路线各16、固定seed/Review/COMPLETE协议。先具体声明“保持r05目标链配方、无新增数值因素”的连续后备方案，scene mix .35/.20/.35/.10不自动改动；r05已完成点若支持更小可验证因素，只在准备完整proposal/TDD后于固定终态前替换，并保存旧声明。连续后备收据预置于r05 `SUCCESSOR_PREDECLARATION.json` / `NEXT_RECIPE_PROPOSAL.json`，不创建plan或启动第二GPU owner。终态照常先8卡释放→后继最小技术smoke/fresh restore→正式真实非零，再综合最终曲线；不等最终CPU、不设能力/admission/promotion门禁。

r04全部已完成早/中固定点FIFO拟合（20261018，非新增评估）：g4/8/16/32/64中同一drop返程偏航位置约[5.89,9.82]，当前Actor的左概率分别.962167/.965362/.965528/.995130/.929852；FIFO几何匹配标签数7/14/17/16/13，均无左目标，附近样本greedy命中4/9/12/8/6。全return FIFO命中1337/1707、2388/3179、1784/2713、1746/2687、1549/2500，target概率均值.774886/.736980/.649960/.632287/.604666。各点FIFO分布随learner改变，不能把这一序列直接视为同分布遗忘曲线；但“多个固定点持续负方向且附近已有未充分拟合teacher”的证据稳定存在。r04各点 `REVIEW_G*_FIFO_COVERAGE.json`保存完整checkpoint/脚本hash、nearest样本和限制，无评估样本入训；g8/g16/g32 full固定game0无C_RETURN首偏航时保持缺项，未补选其他录像。

r04 g96晚点完整评估（20261018，COMPLETE后消费，GPU继续固定128代）：JS Hunter难/中/易1/4/12、Python Tactical6，各64；相对父点-1/-1/+3及-3，较g64 -4/+3/+1及-4。Hunter hard/normal/easy自伤0/0/1，Tactical自伤1；路线self均0，未把零星安全回落隐去或设能力门禁。路线交付death3/sugar2/drop10/near16/mid16/far13/recovery4/full6，各16；父点-2/-3/+1/0/0/0/+1/+2，前点-1/-2/+2/0/0/-1/-1/+1。full窗口→目标14/14为episode共现，首方向6/13、持续oracle1907/2665、实际进度1944/4699、pickup11/16→delivery6/16、death6/self0、恢复27/34；首控34→唯一归因16→pop34/active27→因果kill16/存活14。drop旧JS首方向8/16、持续oracle3302/5098、实际进度3308/5162、pickup16/16→delivery10/16、death1/self0、恢复16/16。原墙death/sugar/recovery均pickup8/8→delivery0/8，返程target3415/2587/3424ticks且oracle全0，持包无法放泡的结构缺口仍在。

固定Review g96掉包首步1实际左新场gain+.341251，高于下+.241251；返程80实际左-.157497，右+.267498/下+.215000，仍是真实负进度。该步JS重建route+.000207422、zero-value reward-GAE-.018775030，较前帧+.000163864；新旧场方向差异证明不能把这份重建当作实际JAX reward/Q。full stock1382实际左新+.678775优于下+.143950，full return2091实际右+.202499优于下+.182598，均不能用旧JS偏航判新teacher错误；full return近似GAE-.022286768、前帧差-.0000944232。drop返程/对抗6/5次、full2/1次。原生一步端点/field/状态hash/事件仍走唯一Review固定game0/seed2026101800，存r04 `REVIEW_G96_PHASE_NATIVE_ENDPOINTS.json` / `REVIEW_G96_PHASE_FIELD_COMPARISON.json`。g96 FIFO返程1542/2407 greedy命中、target概率均值.623210；附近9条teacher命中6，固定drop仍左概率.686383，full返程右概率.950722且无同图邻近FIFO样本，分布覆盖与拟合须分任务判断。

g96训练累计control gross240518.5/net180761.5，route+49142.610/-87513.987，capture终局2764344仍主导；DAgger active31580979/selected2343179/native-safe-progress accepted1530202/first-query70709。drop/stock/return/delivery accepted768640/154945/605728/889，active2185002/13059501/16335585/891，accepted/active约35.178%/1.186%/3.708%/99.776%，包含bounded选择、非独立几何覆盖。actual ticks combat13.695%/bridge4.005%/full76.685%；JAX full每百万有效full tick窗口1012.573、kill637.980、loose pickup16.834、custody382.633、capture306.063。override后current/ring/fixed league占alive-learner有效tick1.700%/9.698%/13.089%，底层league条件7.022%/40.344%/52.634%另列，不混JS/Python/JAX分母。

分析口径修正（20261018，6项focused RED→GREEN）：原始SUMMARY的oracle_agreement/progress_per_target_tick是逐局比值之和，不是总体率，原始收据与历史评估不改写。CURVE_ANALYSIS一直按原始计数重算正确总体率，现增normalized_route_delta_parent/previous，零机会差值保持null。g96 drop持续oracle率相对父点-10.236pp、较g64 -1.034pp；full为-10.474pp/-10.379pp；不能使用原始summed-rate字段的delta。r01–r04已完成点均重算，6项TDD及hash收据r04 `NORMALIZED_DELTA_TDD.json`，数值训练/当前CPU协议和源封印未改。

r04 g96全部rollout-end snapshot可达性补证（20261018，区别于此前前64截断）：8192个真实learner末状态中375个合法空手库存窗口全部按固定顺序分析，ready69、forward frontier225、home frontier81，名义移动标签353；staged+clear+长窗口仍0，native opening witness查询0。306/375处于需要预取包准备的断连目标，但这是单个末状态快照，不能推成全轨迹81.6%或说原生安全teacher已覆盖353条。此证据支持先让learner在空手时走到可开路位置，而非仅增加尚不可达的放泡标签；候选仍未接生产查询/回放/optimizer。收据r04 `PRESTOCK_SNAPSHOT_G96_AUDIT.json`（本次limit8192覆盖所有eligible，旧点仍原截断口径）。

g96信用分量：首次控制正奖励233936，主动爆破残余6582.5（gross240518.5的2.737%），clawback-59757；事件计数entry75333/residual49657与实际支付因上限/去重/回收规则分列，不能把计数乘系数当实际支付。capture终局2764344仍显著高于两者总正信用；数字不证明战斗能力已学成。r04 numerical source/plan seal再次exact，当前analysis/本地只读诊断未进入数值配方。

后续开路候选回放schema补证（20261018，r05 smoke期间、未安装）：7项focused RED→GREEN。统一5×3合法joint payload对旧movement teacher严格保留loss与movement梯度，ability梯度为零；只允许库存准备替换原标签，drop/carry权限保持；准备状态若没有安全证书则抑制原stock标签，不能回退教提前拾包；非法-only集合无标签、合法joint配对不拆独立marginal。收据r05 `JOINT_PAYLOAD_CANDIDATE_TDD.json`。仍缺生产collector/共用query配额、FIFO/optimizer及目标相位统一接线；单测GREEN不代表已具备开路能力。当前r05 source/plan封印无此模块、GPU配方未改。

固定run `runs/je10_bun_20261018/control_route_v38_r04` 完成128代/268,435,456 ticks；compute=4414.817s、wall=5281.152s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r05`；本次后继配方选择依据 `Current r04 capped g1-g16 learner deviations reproduce 35 own-recovery labels with stock PBRS and 2 return labels with PBRS NONE. Native RED reproduces both own-loose target conflict and own-origin pickup followed by frozen PBRS. Align this own-recovery drop->home objective chain before separate corridor/ability work.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。

r04完整固定终态与有效后继（20261018，在r05正式g1真实非零后综合）：128代/268,435,456 fresh ticks完整结束，128次均finite/nonzero，最终8副本exact；compute4414.817s、wall5281.152s。8卡释放compute_clients0后立即安装并启动r05；最小smoke Actor/Critic L2=3.226400/2.229073，保存/fresh CPU恢复通过；正式g1 L2=3.146591/2.156716，8副本exact、finite、checkpoint/hash及FORMAL_NONZERO_CONFIRMED核验。r04保存源码131份与最终完整state/fresh CPU恢复passed；r05 source/plan seal exact。继续固定8卡128代、无门禁；独立CPU/JS唯一owner已跟随g0。

| r04固定代数 | JS Hunter难/中/易（各64） | Python Tactical64 | death/sugar/drop交付（各16） | near/mid/far（各16） | recovery/full（各16） |
|---:|---|---:|---|---|---|
|0|2/5/9|9|5/5/9|16/16/13|3/4|
|4|4/2/7|8|7/4/7|16/16/7|4/5|
|8|2/3/10|8|4/3/7|16/16/9|2/4|
|16|5/2/17|6|3/4/7|16/16/14|5/3|
|32|0/2/7|8|2/2/9|16/15/11|2/3|
|64|5/1/11|10|4/4/8|16/16/14|5/5|
|96|1/4/12|6|3/2/10|16/16/13|4/6|
|128|1/2/5|10|6/3/13|16/15/9|4/4|

完整曲线图：`reports/je10_r04_curve_20261018.png`。drop晚段交付改善，far和Hunter末点回落、full最终未高于父点，尚无不同场景稳定提升证据。最终Hunter难/中/易相对父点-1/-3/-4、较g96 0/-2/-7；Tactical+1/+4。最终各路线交付父点delta death+1/sugar-2/drop+4/near0/mid-1/far-4/recovery+1/full0，前点+3/+1/+3/0/-1/-4/0/-2。Hunter自伤各0，Tactical自伤1/64；全部路线self0，不把安全和攻击效能混为一个成功指标。

最终full窗口→目标12/13为episode共现，首方向10/12、持续oracle2043/2738、进度1904/5990、pickup11/16→delivery4/16、death6/self0、恢复36/43；首控44→归因15→pop44/主动40→因果kill15/存活14。drop旧JS首方向8/16、持续oracle3399/5223、进度3567/5348、pickup16/16→delivery13/16、death0/self0、恢复16/16。原墙death/sugar/recovery各pickup8/8、delivery0/8，返程target2223/2890/2422tick，oracle72/56/64；末点有部分开墙机会，不能再说每点返程oracle恒0，但尚未转成交付。

唯一固定Review最终drop首方向1实际左新gain+.341251优于下+.241251；return80实际左-.157497，右+.217503/下+.127499。JS重建route+.000209375、近似GAE-.022021878、前帧差-.000015854；full stock864实际wait而下+.650537/左+.419106，return938实际下-.157501而右+.157497，近似route-.003877065/GAE-.042946054/前帧差-.001279700。drop返程/对抗14/13次，full1/0次；不把phase切换次数等同根因。state hash/动作/事件/阶段/目标/势能/近似优势见r04 `REVIEW_G128_PHASE_NATIVE_ENDPOINTS.json` / `REVIEW_G128_PHASE_FIELD_COMPARISON.json`。最终FIFO返程1472/2320 greedy命中、target概率均值.633536；附近12条几何匹配命中8，固定drop仍左概率.900300。full返程下概率.910542、无同图邻近FIFO样本。不能声称完整Actor输入冲突、全轨迹未覆盖或同分布遗忘。

最终训练control正信用322755（entry314029/residual8726）、净241506.5；route+64045.663/-115878.066，capture终局3724272仍主导。DAgger active41341568/selected3129800/native-safe accepted2055512/first-query93249；drop/stock/return/delivery accepted1029922/218580/805789/1221，active2959408/16302831/22078106/1223，stock覆盖仅1.341%且包含bounded选择。JAX full每百万有效full ticks窗口979.505、kill629.824、loose pickup17.474、custody391.086、capture314.325；不与JS/Python换分母。实际current/ring/fixed league在override后占有效ticks1.838%/9.472%/13.658%，底层条件7.386%/38.609%/54.005%分列。

证据排序与最小改动已写r04 `FULL_CURVE_ROOT_CAUSE_REVIEW.json`：1）取库存前未准备真实返程走廊，原生持包不能放泡；2）己方掉包→拾回返笼PBRS与teacher目标链冲突，已由20项TDD支持并单因素装入r05；3）已有标签拟合不足且full/drop覆盖缺口不同；4）stock标签薄、实际战斗/对手暴露与reset权重不等。r05先修可具体审查的目标链，取包前准备的生产joint回放/optimizer仍在单独准备，不混入当前run。r06已预声明连续128代后备，不等待最终报告或阈值；继续监督r05完整固定点并按新证据准备最小可验证下一因素。此序列无同父随机对照，不能将前后曲线当单变量因果证明；额外地图泛化仍未验证。

r04最终旧离线返程标签一致性审计（20261018，只读、非重标注）：固定short1062/long996条历史训练记录全部分析；根据Actor输入重建静态terrain/base、当前名义.18端点field（缺完整native state/ledger，非实际端点或安全证书）。与当前名义集合完全不相交均0条；short10条旧集合包含至少一个负进度成员，但仍有新集合可接受成员，long为0。r04最终Actor命中旧集合short1062/1062、long996/996；新名义集合1024/1062与983/996，旧集合概率均值.997540/.991865。drop固定偏航邻近同图long只有3条，旧目标下.5/右2.5，新均右；没有支持“旧离线标签在该区域教向左”的证据，不据此删除离线回放或盲增系数。相比learner FIFO返程1472/2320，提示历史固定数据拟合强而新learner-state分布拟合不足，但两者难度/分布不同，不构成loss竞争的因果证明。收据r04 `FROZEN_RETURN_REPLAY_G128_AUDIT.json`；完整记录hash及所有不相交索引，未选样本、未改训练数据。下一准备仍优先真实取包前走廊的统一目标/联合监督接线，保持r05固定128代健康运行。

r05基线连续性（20261018，g0 COMPLETE后）：g0 raw Actor的canonical tree（所有leaf值/shape/dtype/结构）与r04最终g128 exact，全部Hunter/Python Tactical/八路线完整结果逐项exact；policy.pkl序列化文件SHA不同，未误记成权重变化或SHA相同。凭据r05 `PARENT_BASELINE_PARITY.json`。没有把新seed初始化或PBRS目标链当成g0策略改善。正式训练继续。联合FIFO候选新增3项RED→GREEN，逐字段验证普通movement标签的旧FIFO、随机选择、current/history各半回放、全部旧统计bitwise一致；joint pairs与obs/合法mask经环形覆盖/历史采样仍严格配对，illegal-only不入有效回放。r05 `JOINT_REPLAY_CANDIDATE_TDD.json`；尚无生产native collector/query/目标相位统一接线及真实optimizer梯度收据，未安装。

统一准备状态候选（20261018，未安装）：3项native/JIT focused RED→GREEN，双席共用所选stage及revision，持续相同物理拓扑不抖动；真实opening连通后切回原stock；基于native真实loose存在、routing/drop nonce一致和credit选择优先级，不依赖可能滞后一帧的DAgger loose缓存；自方死亡/受控/terminal全清，窗口关闭去除权限，pickle后JIT恢复逐字段一致。潜在方向只替换stock，新增phase5/6沿既有offset连续账本；hard不可开断连无stock回退。收据r05 `PRESTOCK_STATE_CANDIDATE_TDD.json`。尚未接生产reward、原生安全联合teacher及查询/optimizer；r05当前数值不变。

r05 g4固定点诊断（20261018，COMPLETE核验后消费）：Hunter难/中/易4/1/12、Python Tactical11（各64），相对父点及前一点分别+3/-1/+7、+1；Hunter自伤均0，Tactical0。交付death4/sugar2/drop12/near16/mid16/far7/recovery3/full5（各16），父点及前点delta -2/-1/-1/0/+1/-2/-1/+1。full窗口→目标11/11为episode共现，首方向5/6，持续1076/1319、进度1075/3918，拾包6/16→交付5/16、death4/self0，恢复26/32；首控33→归因11→pop33/主动27→因果kill11/存活8。drop首方向16/16，持续2677/3545、进度2673/3563、pickup16/16→delivery12/16、death3/self0。原始控制fixture与新因果控制分开，零机会率保持null。

唯一固定Review game0/seed2026101800补证：drop动作前3旧JS首偏航实际左，新field gain+.341251高于下+.241251，不能判作新teacher偏航；返程82实际左-.157497，右+.267498/下+.215000，真实返程反向仍在。death-stock1084实际下-.202503、上+.243210；recovery970实际左-.157391、上+.172501。全部保持记录中的对手/ability，只从已验证production Review重放做五个一步端点，无额外评估对局；近似奖励/GAE与state hash见r05 `REVIEW_G4_PHASE_FIELD_COMPARISON.json`，不是JAX真实奖励或Q。

截至g4训练control gross7481.75/net7101.75，route+2205.285/-2780.061，capture终局124992仍主导。DAgger active1386961/selected89167/native-safe accepted51643/firstquery2066；stock accepted1151/active747707=.154%，包含bounded选择，独立几何分母未记录。override后current/ring/fixed league占有效ticks1.727%/6.823%/10.087%，底层league条件8.912%/36.445%/54.643%，严格分列。中间改善与退化只诊断；r05继续固定128代，r06连续后备声明有效，候选取库存前开路尚未安装。

r05 g8固定点诊断（20261018，COMPLETE后）：Hunter难/中/易0/5/11、Python Tactical12（各64），相对父点-1/+3/+6、+2，相对g4 -4/+4/-1、+1；四项自伤均0。交付death2/sugar4/drop9/near16/mid16/far8/recovery3/full4（各16），父点delta -4/+1/-4/0/+1/-1/-1/0，前点-2/+2/-3/0/0/+1/0/-1。full窗口→目标10/10（episode共现）、首方向7/10、持续1460/1977、进度1520/3812、pickup9/16→delivery4/16、death9/self0、恢复21/28；首控28→归因9→pop28/主动22→因果kill9/存活9。drop首方向16/16、持续2843/4703、进度3002/4847、pickup16/16→delivery9/16、death1/self0。原墙death-stock pickup8/8、返程1779ticks/oracle0/delivery0，尚未学习拾包前准备。

唯一固定Review g8：drop3实际左新gain+.341251，旧JS报告首偏航仍非新场负进度；return82实际左-.157497，右+.217503、下+.127499，近似route+.000178906/GAE-.022751309。full stock2286实际wait，上/右各+1.020893，近似route-.001860770/GAE-.045019886。carry_near2实际下+.157500，而右+.267500，仍正进度但非最大；carry_mid98实际上+.202500是新场最佳；carry_far146实际右+.277504低于下+.337502。记录原对手/ability/state hash，未择seed；JS零值GAE近似不是JAX真实优势或Q。

截至g8：control gross19329.5/net16857.75，route+5258.023/-7385.546，capture终局213144。DAgger active3312063/selected181796/native-safe accepted102869/firstquery4117；stock3332/1999882=.1666%，query377，包含bounded选择。override后current/ring/fixed league有效tick占1.887%/4.370%/10.806%，底层条件10.978%/24.402%/64.621%。中间结果不作门禁，r05固定128代继续；下一机制候选将同一prestock selection/revision用于PBRS与联合原生teacher，尚须真实collector/reset/恢复检查后才能预声明安装。

r05 g8固定Review→FIFO只读补证（20261018）：最终该点有效FIFO4117条，其中返程1786条，当前Actor greedy命中1286、target概率均值.705853。固定drop返程82位置[5.94,9.82]附近同terrain/base/origin且L1≤1共12条，标签集合质量下4.5/右7.5、向左0，当前greedy命中11/12、target均值.805724；固定Review输入却仍左概率.927074。完整Actor输入exact匹配0；不能用几何邻近声称监督冲突，亦不能从bounded FIFO缺项推断全轨迹未见。支持进一步分辨威胁/对手/位置细节的分布差异，未支持盲增query预算或删除旧离线数据；评估样本未入训。原始收据r05 `REVIEW_G8_FIFO_COVERAGE.json`，GPU健康固定run继续。

取库存前统一目标候选进度（20261018，未安装）：新增原生导航/开路3项、joint首次偏航配额3项、bounded verifier2项、terminal-only源码接线3项均GREEN；joint FIFO新增运行时hook递归与metadata配对/保存恢复测试，五项GREEN。真实split Actor/Critic网络验证movement/ability/backbone梯度非零、Critic监督梯度为0，真实参数步降低该联合loss。PBRS与teacher消费同一selection/revision，导航必须真实native正进度且持续空手、物理拓扑稳定；开路保留40步真实native witness并与完整joint survivable相交，使用scalar lax.map避免导航行执行长分支。普通movement标签接受所有合法ability，旧FIFO字段/抽样/统计维持bitwise等价；stock导航与opening各一首次偏航quota，仍复用既有槽位。未以独立模块测试替代生产collector/reset/恢复验证；r06连续后备仍权威，新机制只有完整接线核验后才替换预声明。该候选是统一目标机制内PBRS+联合DAgger的整合，不声称分别识别reward与supervision的因果效果。

r05 g16固定点（20261018，COMPLETE后）：Hunter难/中/易2/4/13、Python Tactical8（各64），相对父点+1/+2/+8、-2，前点g8 +2/-1/+2、-4。交付death2/sugar3/drop9/near16/mid15/far9/recovery2/full6（各16），父点-4/0/-4/0/0/0/-2/+2，前点0/-1/0/0/-1/+1/-1/+2。full窗口→目标14/14（episode共现）、首方向4/12、持续1848/2705、进度1850/5493、pickup12/16→delivery6/16、death6/self0、恢复28/36；首次控制36→归因15→pop36/主动31→因果kill15/存活15。drop旧JS首方向8/16、持续3207/4723、进度3184/4848、pickup16/16→delivery9/16、death4/self0、恢复18/18。短点full增加未消除远程、掉包与战斗波动；固定128代继续。

唯一固定Review g16：drop1实际左新gain+.341251，旧JS首偏航仍不等于新field偏航；return80实际左-.157497而右+.267498、下+.215000，近似reward-GAE-.016969048。full stock1885实际右-.705025而左+.755000，近似GAE-.018051221；return1894实际右+.267475但下+.337498，是正进度非最佳集合，近似GAE-.022829979。sugar-stock1077左+.701786为新场最佳；far247左-.157497而右+.162525。原生端点保持记录中的对手与ability，完整state hash/近似reward和优势前后变化保存在固定Review诊断，未新增评估对局。

截至g16训练control gross41122/net33925、route+11005.619/-16530.946、capture终局411432；DAgger active6892198/selected377168/native-safe accepted215923/firstquery8271，stock accepted7751/active4081631=.1899%、query859，包含bounded选择。override后current/ring/fixed league有效tick占1.926%/4.498%/11.380%，底层条件10.842%/24.508%/64.651%，Python/JS/JAX分母分开。r05前g1..16每代固定前16 mismatch（非总体频率）共40条own-recovery全部oracle kind4/PBRS phase2、无旧stock冲突；90条carry均phase3/4，无冻结NONE。证明目标链修复实际可达并进入learner轨迹，尚不构成跨场景能力改善证明；收据`OBJECTIVE_ALIGNMENT_G16_AUDIT.json`。

r05 g16固定Review→FIFO补证（20261018，只读）：返程1755条，greedy命中1225、target均值.678442；固定drop80附近同terrain/base/origin且L1≤1为10条、拟合7条，固定输入仍左概率.950638；full返程1894附近0条，实际右概率.547855、下.448585。两项完整Actor输入exact匹配均0。与g8相比，learner FIFO分布已变化，不能称同分布遗忘或直接证明相同状态监督冲突。行政owner仅重载接续dispatch，首次锁释放延迟已重试恢复；worker marker exact、数值source/plan seal exact，训练和CPU评估未中断，凭据`ADMINISTRATIVE_PRESTOCK_RELOAD.json`。

r06具体配方替换连续后备（20261018，在r05 g50附近预声明，未安装）：ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r06`，父r05最终g128 raw Actor/Critic及完整pool，train2236293928/technical3840110319；8卡固定128代/268435456 fresh ticks，恢复每4代。reset mix固定combat.35/bridge.20/full.35/other native.10；无能力/admission/promotion门禁。唯一新目标机制为“空手取库存前准备真实返程走廊”：PBRS phase5/6与DAgger共用stage/revision，原生导航保持空手/物理拓扑稳定/真实正进度；staging后40步native opening witness与joint safety相交，联合set-mass Actor loss不拆head marginals。普通move标签仍接受所有合法ability，PPO、Actor输入、外系数、控制信用和原生规则不变，复用16 bounded verifier slots。这个机制同时包含PBRS与joint DAgger，是整合目标机制配方，不能分别识别二者因果效果。

具体证据：r04完整曲线原墙拾包后无法交付；r05已COMPLETE g4/8/16/32仍重复该瓶颈，g1..16 capped40/40己方掉包目标已对齐，不能继续归因于已修目标链。候选最终37项focused GREEN、15项relay/连续声明GREEN，真实production-shaped双席collector、终局reset/PBRS零端点、联合payload/FIFO、完整PPO Actor/Critic非零、host保存恢复下一rollout exact；另真实split网络联合DAgger梯度Actor move/ability/backbone非零、Critic为0。收据r05 `PRESTOCK_PRODUCTION_CANDIDATE_TDD.json`，旧后备保存在`OBJECTIVE_CONTINUITY_NEXT_RECIPE_PROPOSAL.json`。运行中数值源码未改；只在r05固定128代、8卡release后安装→最小8卡技术smoke→fresh CPU restore→正式g1 committed nonzero，GPU不等最终CPU/报告。新扩展aux/FIFO schema标记v38-prestock-v1。

异步协议保持g0/4/8/16/32/64/96/128，各Python Tactical64、真实JS Hunter难/中/易各64及八路线各16固定seed；全局唯一CPU/JS owner和COMPLETE/EVALUATION_COMPLETE marker不变。旧JS movement oracle冻结维持纵向比较，新prestock相位/joint ability不能由旧首方向指标直接判错，使用独立native训练首次joint偏航、准备/开路accepted/query、revision和真实奖励/近似优势分解诊断。新增只读joint FIFO分析精确完整Actor输入+mask冲突与各阶段联合集合拟合；不添评估对局、不导入评估样本训练。开路witness是受限有限搜索，不证明主动对抗/全RNG安全。r06开始技术smoke前自动具体预声明同配方r07固定连续后备，新的有证据最小proposal可在终态前替换，最终评估永不成为GPU启动依赖。

r05 g32固定点（20261018，COMPLETE后）：Hunter难/中/易2/4/5、Python Tactical8（各64），父点+1/+2/0、-2，前点g16 0/0/-8、0；四项自伤均0。交付death4/sugar4/drop10/near16/mid16/far13/recovery2/full6（各16），父点-2/+1/-3/0/+1/+4/-2/+2，前点+2/+1/+1/0/+1/+4/0/0。full窗口→目标11/11（episode共现）、首方向4/9、持续1468/1802、进度1513/5573、pickup10/16→delivery6/16、death6/self0、恢复33/41；首控41→归因9→pop41/主动32→因果kill9/存活9。drop首方向8/16、持续2631/3749、进度2594/3794、pickup16/16→delivery10/16、death2/self0、恢复17/17。原墙death-stock pickup8/8、return1945ticks/oracle29/匹配13、delivery0；有部分真实开墙机会，不能说此点恒oracle0，但仍未转成交付。

唯一固定Review g32：drop return80实际左-.157497，右+.267498，近似GAE-.017185249；full stock1539实际上-.396250而左+.533036，近似GAE-.015039383；return1577实际下+.176069低于右+.192501，近似GAE-.025551481。death return979实际右-.092503而其他方向+.022499，sugar return1080实际下-.202499而上+.202503。mid98实际上为新场最佳，旧JS偏航不等同当前teacher；far197左-.202576而右+.337498。全部固定game0/seed与唯一Review链路，不择优，不将近似JS优势当JAX真实Q。

截至g32训练control gross86716/net68258.5、route+19992.324/-33179.006、capture终局868392；DAgger active12597192/selected770227/native-safe accepted461770/firstquery17941，stock28803/6577523=.4379%、query2729，仍含bounded选择。override后current/ring/fixed league有效tick1.583%/6.212%/12.759%，底层条件8.065%/29.258%/62.677%。full与far中点改善、Hunter easy回落和原墙交付缺口并存，r05健康固定128代继续，r06预声明机制与源码hash/dispatch已核验，不等最终评估。

r05 g32固定Review→FIFO（20261018，只读）：返程2119条，greedy1363、target均值.639206；固定drop80附近12条、拟合7，固定输入左概率.945936；full return1577附近0条、固定下概率.998058。两者完整Actor输入exact均0，bounded FIFO拟合与不同Review分布分列，不称oracle冲突或同分布遗忘。r05继续固定预算；r06优先已证据充分的真实取包前走廊目标机制，保留当前掉包目标链和安全/战斗保持。

r05 g32全部eligible末状态候选覆盖（20261018，只读）：核验COMMITTED/state SHA后，固定8192个learner rollout-end状态共379个空手/自由/alive/合法敌死亡或原生受控/真实库存/无优先loose；全部379个按device/env顺序分析，ready77、forward frontier226、home frontier76，名义movement355；staged+clear+长窗口0，实际opening witness查询0、接受0。与r04 g96的相同类快照一致，支持必须先学习走到可开路stage，不能仅给稀缺的placement标签。该末状态分布不是全轨迹机会率，名义355亦非temporal-safe通过率；收据`PRESTOCK_SNAPSHOT_G32_AUDIT.json`。r06已预声明同时教stage导航与有native witness的opening，未增bounded槽位；r05健康固定预算继续。

r05 g32原生偏航优势补证（20261018）：按每代固定前16个accepted-set mismatch、g1..32（有截断与重复episode，不代表总体频率），kind3返程168条中115条真实JAX route reward为负、125条control+route零值GAE近似为负，其中77条剩余task advantage分量为正。kind4 own-recovery63条中48条route负、39条extra近似负、9条extra负且task分量正。kind1 enemy-drop222条中196条route正；正值可含负势能折扣项，必须结合实际方向与field，不能只用即时reward判进度。该分解不识别动作因果Q或梯度方向，不支持直接盲增系数；收据`MISMATCH_ADVANTAGE_G32_AUDIT.json`。g32完整FIFO5138条中精确obs/global/movement mask重复输入4组，接受move集合均有交集、无冲突组；旧FIFO缺ability mask，只审计旧movement监督，不伪称完整joint冲突覆盖。`FIFO_G32_EXACT_CONFLICT.json`。新joint FIFO审计保留实际ability mask及联合关联，后继按固定COMPLETE跟随。

终态schema兼容补齐（20261018，行政CPU工具）：旧restore硬编码v38会使未来prestock终态校验误失败，已TDD RED→GREEN改为与封存plan schema逐项一致，并明确允许既有v38/新v38-prestock-v1；7项schema/installer检查GREEN，不改变运行中的r05数值依赖。后继proposal收据和源码hash同步更新，r05 seal仍exact。

r05 g64固定点（20261018，COMPLETE后）：Hunter难/中/易1/1/6、Python Tactical4（各64），父点0/-1/+1、-6，前点g32 -1/-3/+1、-4；四项自伤均0。交付death3/sugar5/drop9/near16/mid16/far14/recovery2/full3（各16），父点-3/+2/-4/0/+1/+5/-2/-1，前点-1/+1/-1/0/0/+1/0/-3。full窗口→目标15/15（episode共现）、首方向5/10、持续1404/1999、进度1435/6885、pickup9/16→delivery3/16、death6/self0、恢复43/48；首控制49→归因15→pop48/主动40→因果kill14/存活14、另native causal kill1，既有控制fixture与新首控分开，不强造pop/control比例。drop旧JS首方向8/16、持续3231/4784、进度3219/4869、pickup16/16→delivery9/16、death2/self0、恢复17/17。原墙death-stock pickup8/8、return3282ticks/oracle0/delivery0，不能仅从远程子任务改善声称规划普适提升。

唯一固定Review g64：drop1实际左新gain+.341251（旧JS首偏航假阳性），return81实际左-.257496而右+.337502、下+.237499，近似reward-GAE-.023511111。mid334实际左+.022501低于上+.167525，为正进度非最佳集合；far211左-.202499而右+.187475，近似GAE-.007269830。固定full game0只有COMBAT906ticks与STOCK231ticks，无记录首方向偏航不等于成功；有无oracle机会与阶段必须分开。原生state hash、动作、事件、目标、势能、奖励与近似优势前后变化均从唯一production Review导出，不择seed。

截至g64训练control gross172608.75/net131703.75、route+35159.455/-64853.838、capture终局1829352；DAgger active22490480/selected1556821/native-safe accepted983205/firstquery39098；stock91884/9887326=.9293%、query7334，含bounded选择。override后current/ring/fixed league有效tick占2.436%/7.321%/13.785%，底层条件10.251%/30.543%/59.206%。Hunter/Tactical回落、full交付回落和原墙缺口排序优先于near/far子任务单点改善；继续固定128代，不设停训门禁。r06已具体预声明目标机制，奖励/DAgger外系数及mix保持；完整最终曲线将用于准备r07最小因素。

r06联合FIFO审计异步接线（20261018，terminal-only未安装）：由同一个CPU/JS owner在固定快筛COMPLETE后执行，输出`DAGGER_JOINT_COVERAGE.json`（ordinary/准备导航/opening/各kind的joint set概率、greedy拟合和精确Actor输入+合法mask无公共pair冲突）；诊断异常单独marker，不影响已完成固定评估或GPU。7项相关focused GREEN，收据r05`JOINT_EVALUATOR_GREEN.log`；r05当前CPU评估源码未改。

r05 g64固定Review→FIFO（20261018，只读）：返程2142条、当前greedy1403、target均值.645467；drop返程81附近同terrain/base/origin且L1≤1共21条、拟合14，固定Review向左概率.993531。没有完整Actor输入exact匹配，不能声称同状态矛盾；固定full无C_RETURN首偏航，未选择其他seed补造样本。当前完整已COMPLETE曲线图为`reports/je10_r05_curve_20261018.png`（g0..64，无平滑/最佳点筛选，后续固定点完成后覆盖更新），图中远程改善、full与战斗波动并列；不能用JAX训练事件率的增长替代真实JS对局保留。绘图依赖单独放在忽略的run cache，未改训练环境。

r05监督里程碑（20261018）：g96/201326592 fresh ticks已COMMITTED，8副本exact，g1..96全部finite/nonzero；唯一CPU/JS owner已跟随g96，g0/4/8/16/32/64全部COMPLETE，无评估failed。健康run继续至固定g128；r06具体proposal/hash/最小技术收据已准备，终态先核验8卡release→r06 smoke/restore→formal committed g1 nonzero，再综合最终曲线，不等待CPU终点或人工批准。当前未安装prestock数值配方，r05 source/plan seal仍exact。

r05 g96固定点（20261018，COMPLETE后）：Hunter难/中/易0/1/7、Python Tactical9（各64），父点-1/-1/+2、-1，前点g64 -1/0/+1、+5；Hunter self0/0/2、Tactical self2。交付death4/sugar3/drop10/near16/mid16/far10/recovery4/full4（各16），父点-2/0/-3/0/+1/+1/0/0，前点+1/-2/+1/0/0/-4/+2/+1。full窗口→目标9/9（episode共现）、首方向5/7、持续1051/1441、进度1100/4372、pickup7/16→delivery4/16、death5/self0、恢复25/30；首控31→归因8→pop31/主动23→因果kill8/存活8。drop首方向8/16、持续3552/5733、进度3503/5791、pickup16/16→delivery10/16、death2/self0、恢复16/17。原墙death-stock pickup8/8、return3681ticks/oracle0/delivery0；不从某点far14/16择优宣称稳定改善。

唯一固定Review g96：drop1实际左仍新场最佳+.341251；return80实际左-.157497而右+.267498、下+.215000，近似reward-GAE-.018757083。far83实际左-.127499而下+.202503，近似GAE-.001306305。固定full无C_RETURN首偏航，不换seed补造；状态hash、动作、事件/阶段/目标、势能与近似优势保持唯一Review入口。r05继续固定128代，未因自设能力阈值或短点回落停训；self2/64的风险记录进入完整曲线，不混同攻击效能。

截至g96训练control gross256631.25/net193074.75、route+50245.501/-95941.468、capture终局2793600仍主导；DAgger active32270112/selected2343308/native-safe accepted1505314/firstquery58867；stock153668/13199930=1.1642%、query11554，包含bounded选择。override后current/ring/fixed league有效tick占2.272%/8.421%/13.878%，底层条件9.267%/33.684%/57.049%，真实JS/Python与JAX训练协议分列。最终只在r05终态、r06正式真实非零之后综合全曲线；当前r06预声明机制不等待最终CPU。

r05 g96 FIFO补证（20261018）：返程1976条，当前greedy1280、target均值.635821；drop80附近12条、拟合4，固定Review左概率.999472，完整Actor输入exact0。该bounded FIFO与固定JS输入不同，不由几何邻近推断同输入冲突；后继继续独立报告joint set拟合/覆盖/首次偏航。r05截至g118全部健康finite/nonzero，CPU固定点只剩g128；不增补择优seed或额外门禁。

固定run `runs/je10_bun_20261018/control_route_v38_r05` 完成128代/268,435,456 ticks；compute=4396.396s、wall=5250.523s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r06`；本次后继配方选择依据 `Completed r04 curve and r05 fixed g4..32 retain native original-wall pickup-to-return disconnection; carriers cannot place bombs. r05 target alignment is live (40/40 capped own-recovery labels align), yet stock native-safe coverage remains thin and corridor preparation is absent. Use one shared preparation objective before inventory acquisition.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。


r05完整固定终态及r06有效启动（20261018，正式非零后综合）：r05完整128代/268,435,456 fresh ticks，128次均finite/nonzero，最终8副本exact；compute4396.396s、wall5250.523s。8卡compute_clients=0释放后立即交接r06。r06最小技术smoke Actor/Critic每卡L2=5.591779/2.253800，保存与fresh CPU完整state/next RNG恢复通过；正式g1/2,097,152 ticks已COMMITTED，L2=5.489913/2.253784，8副本exact、finite、policy SHA与FORMAL_NONZERO_CONFIRMED亲自核验。r05最终133份保存源码及fresh CPU终态恢复通过，r06数值source/plan seal exact。无能力/admission/promotion门禁；唯一CPU/JS evaluator与GPU并行，后继已有效更新后才综合本段结果。

r05全部固定点（各列分母固定，未择优）：

|代数|JS Hunter难/中/易，各64|Python Tactical64|death/sugar/drop，各16|near/mid/far，各16|recovery/full，各16|
|---:|---|---:|---|---|---|
|0|1/2/5|10|6/3/13|16/15/9|4/4|
|4|4/1/12|11|4/2/12|16/16/7|3/5|
|8|0/5/11|12|2/4/9|16/16/8|3/4|
|16|2/4/13|8|2/3/9|16/15/9|2/6|
|32|2/4/5|8|4/4/10|16/16/13|2/6|
|64|1/1/6|4|3/5/9|16/16/14|2/3|
|96|0/1/7|9|4/3/10|16/16/10|4/4|
|128|2/4/10|8|6/6/10|16/16/13|5/7|

最终相对父点Hunter+1/+2/+5、Tactical-2；相对g96 +2/+3/+3、-1。最终交付death/sugar/drop/near/mid/far/recovery/full相对父点0/+3/-3/0/+1/+4/+1/+3，相对g96 +2/+3/0/0/0/+3/+1/+3。完整曲线仍波动，不能将终点或最佳far点当跨场景稳定提升。Hunter终点self0/0/0、normal death3/64/trade1，Tactical self1/death1；g96曾Hunter easy self2/Tactical self2，安全波动保留。曲线图覆盖所有预声明点：`reports/je10_r05_curve_20261018.png`。

最终full按计数重新归一：窗口→目标12/12（episode共现，非逐window因果转化），首方向9/11，持续2120/3058，进度2223/5608，拾包10/16→交付7/16，death3/self0，恢复切换22/32；相对父点进度率+7.8535百分点、持续一致率-5.2902百分点、恢复率-14.9709百分点。首控33→可归因11→最终pop33（主动30）→因果击杀11/存活11，初始化fixture与自然首控分开。drop窗口目标16/16、旧JS首方向8/16、持续3864/6069、进度3595/6123、拾包16/16→交付10/16、death1/self0、恢复16/16。原墙death/sugar/recovery均拾包8/8、交付0/8，返程3798/3288/3851 ticks且oracle机会0；库存总体改善来自开放墙分层，不冒称原墙解决。原始计数、normalized delta及COMPLETE哈希链位于r05 `CURVE_ANALYSIS.json`；SUMMARY逐局比值之和不用于总体率比较。

最终唯一固定Review（game0/seed2026101800，不补选失败seed）及native一步fork：drop动作前1向左新gain+.341251为最佳，旧JS首偏航是假阳性；返程80仍左-.157497而右+.267498/下+.215000，近似JS reward-GAE-.018970627。far207等待0而下+.157497；full返程595实际右+.200176，低于下+.333218但仍正进度，近似GAE+.014485708；stock1285实际左+.631965为新场最佳，不按旧JS判错。mid138实际上+.182500亦为新场最佳，不能因近似GAE-.112351声称动作Q负。状态hash、原动作/对手/ability、目标/阶段/事件、势能、分量及前后近似优势保留于r05 `REVIEW_G128_PHASE_NATIVE_ENDPOINTS.json` / `REVIEW_G128_PHASE_FIELD_COMPARISON.json`，入口仅production Review Player。JS重构GAE不是JAX真实优势或因果Q。

r05最终FIFO返程1894条、greedy1278、target均值.660931；固定drop80同几何附近10条/拟合7，固定输入左概率.983082；full595附近0条，右概率.931747；两者完整Actor输入exact均0。只支持有限FIFO拟合与固定失败分布分列，不以几何邻近推断同输入oracle冲突，也不将缺项视为全轨迹未见。此前精确旧movement冲突审计无无交集组；r06使用新完整joint输入/mask独立审计。

r05累计控制gross341661.75/net255995.5（entry332449.25、residual9212.5、clawback-85666.25）；route+65403.191/-127563.610/net-62160.419，capture终局3755856仍主导，不以net掩盖正负组成。DAgger active42250470/selected3129871/native-safe-progress accepted2025712/query79009；stock216663/16509711=1.31234%，query15748，包含bounded选择而非独立几何通过率。训练JAX full每百万有效ticks：window1011.555、kill665.123、库存/native steal472.393、归因custody404.611、capture317.976、loose17.999；不合并真实JS/Python分母。override后current/ring/fixed league实际有效tick份额2.1682%/8.6325%/14.3136%，底层league条件8.6801%/33.9431%/57.3768%；full实际有效tick占76.8527%，与reset35%分列。

证据排序已保存r05 `FULL_CURVE_ROOT_CAUSE_REVIEW.json`：①原墙取库存前没有真实返程走廊，原生持包不能放泡；r06已选共享stage/revision PBRS+native-safe joint监督，是整合机制而非分别识别reward/DAgger因果。②返程有限标签拟合及分布覆盖仍弱；r06固定点先检查joint准备导航/开路覆盖、精确冲突与首次偏航，不盲增系数。③full/战斗保持及恢复切换波动，结合最终动作来源曝光诊断。新run首代准备active100346/selected6747/accepted6016/query1352，其中opening accepted/query各1，证明信号进入实际learner collection，不证明开路能力已学会。健康r06固定128代继续，当前源码/plan不热改。

r07连续后备已在r06 smoke前具体预声明：ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r07`，父r06最终g128 raw Actor/Critic/完整pool，train1777901369/technical814965590；同8卡128代/268435456 ticks、mix .35/.20/.35/.10、恢复每4代，初始化fresh Adam/RNG/env/EMA-to-raw/FIFO/counters。固定异步点0/4/8/16/32/64/96/128，唯一全局CPU/JS owner、COMPLETE及EVALUATION_COMPLETE、Python Tactical/真实Hunter各64和八路线各16、固定Review不变。中间完成点仅准备有证据最小proposal/TDD；若尚无充分可核验新因素则按同配方后备连续训练，终态先核验8卡释放→最小技术smoke/fresh restore→正式g1 committed nonzero，再最终综合。无能力门禁，不等待CPU/报告/人工批准，不因短点不佳中断。r06/r07连续前瞻收据分别保存在run本地，run/checkpoint不入Git。


r06固定早期监督（20261018）：g4/8,388,608 fresh ticks已COMMITTED，四代全部finite/nonzero、8副本exact，无技术failure。首两代含编译约173.005/171.494s，g3/g4约38.222/38.255s；八卡持续训练，无评估门禁。唯一CPU/JS owner已完成g0并进入g4，g0 joint FIFO为空（fresh初始化，非缺失标注）。g0 Actor canonical tree与r05 g128逐叶值/shape/dtype/结构exact，Hunter/Tactical/八路线完整结果逐项exact，收据r06 `PARENT_BASELINE_PARITY.json`。四代准备active777691/selected30768/accepted28781/query7564，opening accepted21/query16；限定learner训练数据覆盖，不能当路线能力改善。中间点只诊断，健康run继续固定128代。


r06 g4固定完成点（20261018，COMPLETE后）：Hunter难/中/易1/3/9、Tactical9（各64），相对父点及前点g0 -1/-1/-1、+1；Hunter self0/0/0，Tactical self0。交付death2/sugar0/drop6/near16/mid16/far8/recovery2/full3（各16），相对父点/前点-4/-6/-4/0/0/-5/-3/-4。full窗口目标12/13（episode共现）、首方向4/8、持续1886/3124、进度1976/5445、pickup8/16→delivery3/16、death6/self0、恢复31/37；首控38→归因16→pop38/主动27→因果kill16/存活14。drop首方向16/16（旧冻结JS）、持续3453/5769、进度3463/5869、pickup16/16→delivery6/16，death1/self0、恢复16/16。原墙death pickup8/8→delivery0/8，return2062ticks/oracle0；sugar原墙8/8→0/8、return2837/oracle0，开放墙亦8/8→0/8。早点不佳只用于后继诊断，健康128代继续。

r06 g4新joint FIFO：valid7985，准备导航6229/greedy2616/mean joint mass.419417；opening8/greedy2/mass.277088；ordinary1748/greedy1034/mass.567981，return kind3为741/greedy407/mass.542771。完整Actor obs/global/合法move+ability mask精确重复4组，无无公共joint pair冲突。准备导航占FIFO78.01%，accepted变多不等于已学会；单个早点尚不能识别样本竞争的因果作用。已明确的候选方向是检查准备导航是否大量占用普通返程监督，待g8/16/32同协议完整点与曲线确认后，再选择一轮最小数值因素；不盲加系数或同时改mix。

r06 g4全部8192个实际learner保存末状态（只读，无新对局/原生fork）：prestock active1240、ready82、preparing1158；已达staging14、其中clear7、clear且native控制>40仅1，且place合法1。objective2返己方走廊266/staged8/clear5/long0；objective3通敌库存892/staged6/clear2/long1；全部无pushable。使用实际checkpoint selection/revision，不以重新选择的stage代替训练目标。此为rollout-end快照，不是全轨迹机会率；“满足前提”不是安全witness接受率。收据r06 `ACTUAL_PREPARATION_G4_SNAPSHOT.json`。先学习到达stage与长窗口/清场的联合稀缺均需分开，不从opening样本少直接放宽安全。

r06 g4唯一固定Review game0/seed2026101800：掉包return80依旧实际左gain-.157497，右+.267498/下+.215000，近似JS GAE-.019083478；far85实际右-.337500而下+.162500；full stock1127实际右-.195200而上+.685000，近似GAE-.015761369。full return1455实际左+.022499为新field等优集合，旧JS首偏航不可直接判错；drop64左+.681250低于上+.731250但仍正进度。库存原墙death动作前3实际右-.153751、sugar动作前4右-.341250，对照上为正；此处是直接stock场信号，preparation目标应独立解读。所有状态hash、事件/动作、目标/阶段、势能与近似GAE来自唯一production Review，未择seed、未导入评估数据训练。

g4训练累计control gross7616.75/net7087.75、route+2660.709/-3037.789、capture122088仍主导；DAgger active1456097/selected89934/accepted81710/query9660。override后current/ring/fixed实际有效tick .5016%/5.7709%/12.4638%，底层league条件3.1869%/30.2665%/66.5466%，保持分列。g0/g4已COMPLETE、唯一CPU owner跟随g8；r06 g15全部finite/nonzero，数值配方冻结，无门禁和提前停训。r07已有同配方连续后备，新的证据因素只有TDD与完整proposal后才替换。


r06 g8固定完成点（20261018）：Hunter难/中/易0/3/10、Tactical12（各64），相对父点-2/-1/0、+4，相对g4 -1/0/+1、+3；Hunter self均0，Tactical self2/death2，安全保持分列。交付death0/sugar0/drop4/near16/mid15/far9/recovery0/full3（各16），相对父点-6/-6/-6/0/-1/-4/-5/-4，相对g4 -2/0/-2/0/-1/+1/-2/0。full窗口目标13/13、首方向7/11、持续1757/2775、进度1856/5965、拾包10/16→交付3/16、death5/self0、恢复35/41；首控41→归因13→pop41/主动32→因果kill13/存活12。drop旧首方向8/16、持续3335/5576、进度3361/5681、拾包16/16→交付4/16、death2/self0、恢复16/16。原墙death拾包8/8→0/8、return1983/oracle0；sugar8/8→0/8、return1977/oracle34/match28，动态已有部分开墙但仍无交付。

joint FIFO g8 valid16177，其中准备导航12665/greedy5208/mass.411261（78.29%）；opening20/greedy10/mass.482473；ordinary3492/greedy1913/mass.529235；返程kind3 1625/greedy786/mass.492793，较g4返程fit.542771回落。完整obs/global/move+ability mask精确重复8组，无无公共pair冲突。实际8192保存末状态prestock active1298、ready122、preparing1176，staged10/clear4/clear且long0；objective2 196/staged2，objective3 980/staged8。该分布与g4均显示到stage不足及开路前提联合稀缺；有限FIFO占比/拟合不能单独证明梯度竞争或安全条件过严。

g8唯一固定Review：drop1实际左+.341251为新场最佳，不用旧JS判错；return80实际左-.157497、右+.217503/下+.127499，近似JS GAE-.019065605。mid167实际右-.022500而上+.257500，far143实际左-.187473而下+.162500，full stock645实际下-.386607而左+.636225；full return839右+.284641低于下/左+.337502但为正进度，近似GAE-.022675533非动作Q。持续监督r06全部固定预算；唯一CPU owner已进入g16，不增补评估seed。

r07候选方向（尚未选择/安装）：只改变DAgger回放阶段分配，防止准备导航占满联合监督并保留掉包/返程；保持FIFO新增/容量、当前/历史各半、总回放64每设备、监督/奖励系数、query/native安全、scene mix和PPO不变。先对无组/稀有组/重复保存恢复及当前历史边界写RED，候选不会热改r06。g16/32/64等完成点继续检查实际分布与首次偏航，再决定具体proposal；当前同配方连续后备始终可执行，无门禁或CPU等待。


r06 g16固定完成点（20261018）：Hunter难/中/易0/2/11、Tactical5（各64），父点-2/-2/+1、-3，前点g8 0/-1/+1、-7；Hunter self均0，hard/normal各trade1，Tactical death/self0。交付death1/sugar0/drop4/near16/mid16/far12/recovery1/full4（各16），父点-5/-6/-6/0/0/-1/-4/-3，前点+1/0/0/0/+1/+3/+1/+1。full窗口目标12/12（episode共现）、首方向6/9、持续2851/4389、进度3007/7789、pickup9/16→delivery4/16、death4/self0、恢复35/45；首控45→归因11→pop45/主动39→因果kill11/存活11。drop首方向旧JS16/16、持续3404/6229、进度3405/6286、pickup16/16→delivery4/16，death2/self0、恢复16/16。原墙death拾包8/8→0/8、return4458/oracle0；弱漏斗尚未稳定改善，健康固定run继续。

joint FIFO g16 valid16384，准备导航11848/greedy4560/mass.388084（72.31%）、opening52/greedy16/mass.329502、ordinary4484/greedy2336/mass.500867；返程kind3 2321/greedy1076/mass.464052。普通返程mass从g4 .542771→g8 .492793→g16 .464052，准备导航亦未掌握；精确Actor输入+mask重复3组、无无公共pair冲突，不由无冲突小样本宣称全分布无冲突。全部保存末状态prestock active845/ready94/preparing751，staged15/clear8/long4/place4；objective2 220/staged7/long1，objective3 531/staged8/long3。虽有可用开路前提，末状态到stage仍稀疏，不将样本计数当能力或全轨迹覆盖率。

g16唯一固定Review game0：drop82实际左-.157497而右+.267498/下+.215000，近似JS GAE-.018362705；far87实际右-.337500而下+.162500，GAE-.002337863。drop3左+.341251和mid138上+.182500均为新field最佳，旧JS偏航是假阳性。fixed full game0无记录route首偏航，保留无机会/阶段缺项，不换seed补造。原墙death/sugar stock2实际左-.241249，对照上+.027500；此为冻结direct-stock oracle目标，新prestock stage必须独立解读。

g16训练累计control gross50136.25/net41794.75、route+12821.886/-16850.732、capture终局414984；DAgger active7116679/selected380584/accepted339084/query44806。override后current/ring/fixed实际有效ticks1.2349%/5.2340%/11.3123%，底层league条件7.6944%/29.4206%/62.8850%，不混为曝光比例。g16恢复checkpoint与8副本exact已核验，g32训练checkpoint已COMMITTED，唯一CPU评估g32中。未来最小候选仅回放阶段分配；11项focused GREEN（含终态/8卡释放才能安装）和真实collector/PPO/reset/恢复检查独立准备中，不修改当前worker/数值源码。


r06 g32固定完成点（20261018）：Hunter难/中/易2/4/8、Tactical8（各64），父点0/0/-2、0，前点g16 +2/+2/-3、+3；Hunter self均0、hard trade1/easy death2，Tactical death/self0。交付death2/sugar3/drop8/near16/mid16/far12/recovery1/full5（各16），父点-4/-3/-2/0/0/-1/-4/-2，前点+1/+3/+4/0/0/0/0/+1。full窗口目标10/10（episode共现）、首方向6/9、持续1405/1882、进度1432/4410、pickup7/16→delivery5/16、death7/self0、恢复26/32；首控32→归因11→pop32/主动22→因果kill11/存活11。drop旧首方向8/16、持续3218/5123、进度3214/5196、pickup16/16→delivery8/16、death2/self0、恢复16/16。原墙death拾包8/8→0/8、return3631/oracle0。中段有恢复但尚未稳定超越父点，既不择优宣称改善也不因早期不佳停训。

joint FIFO g32 valid15705：准备导航8506/greedy3501/mass.409304（54.16%），opening42/greedy17/mass.407895，ordinary7157/greedy3765/mass.512858；返程kind3 3860/greedy1850/mass.480193（24.58%）。精确输入+move/ability mask重复8组、无无公共pair冲突。全部learner末状态prestock active387/ready101/preparing286，staged5/clear3/long1/place1；objective2 92/staged3/long0，objective3 194/staged2/long1。占比随learner分布明显改变，不能机械沿用g4的78%作为整段恒定值。

g32唯一固定Review：drop80实际左-.157497，右+.267498/下+.215000，近似JS GAE-.018563406；far306实际左-.162472而右+.157497，GAE-.005798739。drop1左+.341251、mid143上+.157500均为新场最佳，full return1686下+.022499为新field等优，近似GAE+2.562751不是动作Q；stock无固定首偏航不补选seed。记录完整hash/动作/事件/目标/势能/近似优势前后变化于r06 `REVIEW_G32_PHASE_FIELD_COMPARISON.json`。

g32训练累计control gross95094.25/net74943.25、route+22508.838/-33661.665、capture845568仍主导；DAgger active12944606/selected773796/accepted688985/query79896。override后current/ring/fixed实际有效ticks1.7516%/6.6359%/12.3374%，底层league条件8.5296%/31.7062%/59.7642%，不将reset权重或底层抽样当实际对手曝光。唯一CPU g0/4/8/16/32均COMPLETE，无failed；GPU健康继续。

回放分配候选依据g32修正（仍未选入/安装）：最初ordinary三类均分会将返程降至每半约5/32，低于g32已有约25%返程，追加RED捕获并改为每半32中ordinary16（有组时return8/drop4/ready-stock4）、navigation14、opening2；缺组在同一current/history半内再分配，完全无preparation时旧uniform逐位exact。14项focused GREEN；真实collector/PPO/保存恢复首次版本已通过，分配修正后必要复验正在CPU独立进行。没有改FIFO容量/新增随机选择、query/native安全、原生规则、系数、PPO或scene mix；只有返回的监督batch分配改变。

保存g32全部8个device FIFO、固定sampler seeds6107..6122、每设备每seed32条history、总4096条：uniform ordinary1893（drop513/ready313/return1067）、navigation2187、opening16；候选ordinary2048（512/512/1024）、navigation1792、opening256。完整joint合法性逐条通过，候选返程1024并不高于该特定uniform抽样1067，诚实保留此代价；总量不变，开路提高来自重复有效历史样本而非新增状态或更多原生query。该只读分配审计验证机制而非能力，不导入评估样本。TDD与终态installer/relay检查只决定技术可执行性，不构成能力门禁；r07同配方后备继续有效，最终选择结合后续完成曲线。


后继回放候选最终技术收据（20261018）：14项focused及15项relay/连续声明GREEN；修正普通返程份额后真实production-shaped双席collector→joint payload/FIFO→完整PPO Actor/Critic非零、terminal reset/PBRS零、host保存恢复下一rollout exact复验通过。收据r06 `PHASE_REPLAY_PRODUCTION_CANDIDATE_TDD.json`，candidate源码hash与所有初始RED/最终GREEN日志均本地保存。只读g16/g32历史FIFO分配证据独立保留，重复opening标签不是新增独立机会。未安装candidate数值配方；r06继续固定预算。

行政交接支持已仅重载唯一监督owner以加载未来replay installer/封存helpers（20261018，g50附近）；原GPU worker和唯一CPU/JS evaluator均未重启，146份数值source与plan SHA exact，日志无failure。`ADMINISTRATIVE_PHASE_RELOAD.json`为run本地收据；这项行政变动不会在r06执行回放修改。新的installer只在父128代成功终态且8卡release后修改aggregate dispatch，不改变loss、buffer或env规则，支持后继按同配方连续预声明。r07当前仍是原具体连续后备，后续完成点可选择已验证的最小回放因素；CPU最终报告不成为GPU依赖。


r06 g1..32原生capped偏航信号补证（20261018，只读，不新增轨迹）：每代保存前16条accepted joint-set mismatch，共512条；按generation/device/env/control episode只保留已保存最早条后410条。准备导航290条中247条当步route负、148条距离增加，82条虽当步route负但零值extra-GAE正、134条remaining-task-GAE正；route均值-.003164、extra-GAE中位-.003500、task-GAE绝对均值.162439。返程62条中53条route负、12条距离增加，extra-GAE中位-.018215，task-GAE绝对均值.805429；现有信号能在部分偏航处给负分，不支持简单认定奖励完全不可达。drop31条全部当步route正但仍不在joint最佳集合，说明正进度与最高回报集合一致必须分开。opening仅2条，不能用此截断样本推断总体开路率。

收据r06 `CAPPED_NATIVE_DEVIATION_SIGNAL_G32.json`保存源hash与全量/去重分层；这是设备/时间前序截断样本，已保存最早不必是原生episode首偏航，不能当总体事件率或因果梯度证据。extra/task均是零值GAE近似，不是带真实Critic的PPO advantage或动作Q；hash只覆盖Actor observation/global，原生完整state/失败录像仍走唯一production Review。当前数值146份source/plan、评估11份source均exact；GPU/worker/唯一行政owner健康，继续固定128代。


后继候选补充实际replay遥测（20261018，未安装）：新增RED确认原先未记录每半实际阶段使用量，GREEN后逐代按current/history记录valid、ordinary、returning、drop、ready-stock、navigation和opening真实计数（含fallback重复），不以FIFO占比代替监督batch暴露。15项focused、15项relay/continuity与真实collector→joint FIFO→完整PPO Actor/Critic非零/reset/保存恢复复验通过；更新收据与源码hash于r06 `PHASE_REPLAY_PRODUCTION_CANDIDATE_TDD.json`。采样配方不变；遥测未进入r06数值源码。g64/134217728 fresh ticks已COMMITTED，唯一CPU/JS owner同步跟随，GPU继续后半程。


r06 g64固定完成点（20261018，COMPLETE后）：production JS Hunter难/中/易0/3/8、Python Tactical8（各64）；相对父点-2/-1/-2、0，相对g32 -2/-1/0、0；Hunter self全0，Tactical death/self0/0。八路线交付death1/sugar0/drop7/near16/mid16/far11/recovery1/full2（各16），相对父点-5/-6/-3/0/0/-2/-4/-5，相对g32 -1/-3/-1/0/0/-1/0/-3。完整机会归一化parent/previous变化保存r06 `POINT_G64_OWNER_REPORT.md`与`CURVE_ANALYSIS.json`，曲线图包含全部g0..64固定点；不以部分恢复或单点择优替代整段结论。健康run继续128代。

full窗口→目标9/9（episode共现）、首方向5/8、持续819/1203、进度871/4589、pickup8/16→delivery2/16、death6/self0、恢复27/34；首控34→归因6→pop34主动23→因果kill6/存活6。drop旧首方向8/16、持续3627/5841、进度3639/5989、pickup16/16→delivery7/16、death2/self0、恢复16/16。原墙death/sugar/recovery均pickup8/8→delivery0/8，return2213/1667/1682ticks、oracle112/555/376，已有动态开墙机会但闭环交付仍缺失；death原墙death8/8需作为保留风险记录，不能仅看self0。

joint FIFO g64 valid14424，准备导航6723/greedy2672/mass.401872、opening60/greedy11/mass.187832、ordinary7641/greedy3793/mass.491363；返程4293/greedy1887/mass.446698（29.76%），精确Actor输入+mask重复4组，无无公共pair冲突，不证明全分布无冲突。实际8192末状态prestock active363/ready81/preparing282，staged17/clear12/long+place5；这是末状态前提分布而非安全接受率或全轨迹覆盖。g1..64实际prestock active8012480/selected426523/accepted395005/query83401，其中navigation394248、opening757/query452，开路可达但稀少。

g64唯一固定Review game0：drop1实际左+.341251为新场最佳，return80左-.157497而右+.267498/下+.215000，近似JS GAE-.018867591；mid103右-.157499而左+.157501，GAE+.004423688非动作Q；far181左+.182501为新场最佳，不沿旧JS误判。固定full无记录首偏航，不另选seed补造。stock3右-.153751、上+.021250为旧direct-stock场对照，新prestock准备阶段独立解读。g1..64 capped样本1024，保存最早去重796；返程154中141当步route负，准备导航499中432负，重复/前序截断与零值GAE限制继续保留。

训练累计control gross178882.500/net136463.250、route+39094.836/-67230.319、终局capture1763160主导；DAgger active23401682/selected1560538/native-safe accepted1397499/query139253。override后current/ring/fixed有效ticks2.3854%/8.3514%/12.9760%，底层league条件9.8863%/34.7340%/55.3797%；实际对手暴露与reset mix分列。

r07候选分配进一步修正（仍未安装）：g64固定4096 history draw中原uniform返程1141、初候选1024（不同FIFO条目1008/916），opening10/256（不同条目10/59）。初候选仍稀释返程，追加RED后将每半32设为ordinary20（return12/drop4/ready-stock4）、navigation10、opening2；无preparation仍旧uniform逐位exact，缺组仅同半分配。原审计保存`PHASE_REPLAY_G64_ALLOCATION_SUPERSEDED_16_14_2.json`，不覆盖历史机制证据。该修正仍只改变监督batch分配，奖励/外系数/query/native安全/FIFO新增容量/总量/PPO/scene mix不变；最终技术复验独立CPU执行，r06不受影响。


r07具体proposal已在r06 g80附近选定并封存（20261018，无门禁）：`JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r07`，父r06 final g128 raw Actor/Critic与完整pool；train seed1777901369、technical814965590；fresh Adam/RNG/env/EMA-to-raw/FIFO/counters。固定8卡128代/268435456 fresh ticks，reset mix .35 combat/.20 bridge/.35 full/.10 other native routes，g0/4/8/16/32/64/96/128唯一CPU/JS评估及全部seed/局数不变，恢复每4代。终态先8卡release→安装→最小technical smoke/保存恢复→正式g1非零，不等待最终CPU或报告。

唯一新数值因素为joint DAgger回放阶段分配：current/history各32，每半有组时return12/drop4/ready-stock4/navigation10/opening2；缺组同半再分配、无preparation旧uniform逐位exact。奖励及监督外系数、native安全/query预算、FIFO新增容量、Actor输入/network/PPO/retention/mix均不变。具体来源hash、15项focused+15项relay与真实双席collector/PPO/reset/恢复GREEN、证据路径与限制保存r06 `NEXT_RECIPE_PROPOSAL.json`和`PHASE_REPLAY_PRODUCTION_CANDIDATE_TDD.json`；原同配方后备保存`CONTINUOUS_FALLBACK_BEFORE_PHASE_SELECTION.json`。r06数值仍未修改。

选择依据按证据排序：①g64开路mass.187832/greedy11/60而757 accepted/452 query说明可达但稀有，uniform history4096仅10条开路；②返程mass.446698、固定drop80实际负方向且交付/自然full回落；③准备navigation与ordinary分布随learner改变、见证限制和原墙闭环缺口。修正候选history4096为return1536（不同条目1295）/opening256（不同59）/navigation1280，对照uniform1141（不同1008）/10（不同10）/1984。开路重复上采样不等于新增机会，导航份额下降是明确风险；检查逐代实际监督计数与固定点fit/推进/安全，不设阈值门禁。当前父选择为连续训练轨迹最终raw，沿用已记录发布起点，不宣称r06是能力最佳发布点；顺序followup不是同父随机因果实验。r08仍会在r07启动时预声明同配方连续后备，完整最终曲线用于选择其最小改动。


测试计数口径补充（20261018）：最终candidate为12项sampler/installer focused GREEN，另15项relay/continuity GREEN（其中3项为candidate recipe relay，先前“15项focused”将这3项并入且与relay有重叠）；真实collector/PPO/reset/恢复为独立1项GREEN。收据已标明分组，避免误读为30项互不重叠测试。所有最终source/proposal hash与r06数值146份seal亲自核验exact。


后继诊断非数值补充（20261018，r06未修改）：新增2项RED→GREEN，封存installer在终态后为worker偏航行写入`collector_gae_advantage`、`collector_value_target`、`collector_value_estimate`、`behavior_action_log_probability`。字段直接读取已选中device/tick/env的batch scalar，不复制完整observations/batch，不变更GAE、PPO归一化或更新。真实GAE注明Critic-backed rollout且在per-device PPO normalization之前；与已有zero-value extra/task GAE近似分列，不冒称Q或反事实。最终14项sampler/installer/诊断focused与15项relay/continuity GREEN；分配修正后真实collector/PPO/reset/保存恢复已独立通过。proposal/receipt源码hash已同步；r06数值146份seal仍exact。


原生capped诊断字段校正（20261018）：`phase`是DAgger kind，kind4为attributed own-loose recovery，不是CAGE_DELIVERY；PBRS阶段在`trace.route_phase`，delivery子阶段不能仅凭kind3/4倒推。只读汇总已将kind4明确标为`own_recovery`并重生成g32/g64收据，其数值和训练不变。r06 g96/201326592 fresh ticks已COMMITTED、8副本exact，唯一CPU/JS owner异步跟随；GPU继续固定末段。


r06 g96固定完成点（20261018，COMPLETE后）：JS Hunter难/中/易2/7/7、Python Tactical6（各64）；相对父点0/+3/-3、-2，相对g64 +2/+4/-1、-2。Hunter self0/2/0，Tactical death/self0/0。交付death5/sugar1/drop10/near16/mid16/far13/recovery2/full5（各16），相对父点-1/-5/0/0/0/0/-3/-2，相对g64 +4/+1/+3/0/0/+2/+1/+3。中后段有恢复但原墙/糖泡窗口闭环与完整曲线仍不稳定，不改变健康固定预算或后继启动条件。

full窗口→目标10/10（episode共现）、first6/8、持续1215/1722、进度1256/5021、pickup8/16→delivery5/16、death10/self0、恢复28/36；首控36/归因6/pop36主动32/因果kill6存活6。drop旧first8/16、持续3632/6119、进度3665/6322、pickup16/16→delivery10/16、death1/self0、恢复16/17。原墙death/sugar/recovery均pickup8/8→delivery0/8，return1719/1578/997ticks、oracle36/357/31；death原墙仍death8/8。完整parent/previous机会归一化变化在r06 `POINT_G96_OWNER_REPORT.md`，缺分母保留null，不混池不同simulator。

joint FIFO g96 valid14580，navigation6876/greedy2690/mass.390994、opening93/greedy32/mass.349103、ordinary7611/greedy3895/mass.502709；return4388/greedy2051/mass.473376。开路fit较g64 .187832回升仍有限，准备导航与返程均未掌握；不得把不同FIFO分布当固定holdout。实际8192末状态prestock active375/ready96/preparing279、staged20/clear12/long+place2。修正候选history4096抽样return1536/opening256/navigation1280，对照uniform1225/27/1997，仍只验证分配机制，非能力结果。

g96唯一固定Review：drop return80仍左-.157497、右+.267498，近似JS GAE-.020736774；far82左-.202499而右+.222502，GAE+.000850374说明后续回报近似不等于动作Q。drop1左+.341251、mid93上+.202499、full stock1273下+.474998为等优、full return1323右+.202499为新场最佳，旧JS偏航均不能直接判错。stock旧direct场与新prestock目标分列，完整原生hash/动作/事件/目标/势能/奖励和近似优势链保存r06 `REVIEW_G96_PHASE_FIELD_COMPARISON.json`，不择seed。

累计control gross261030/net197298、route+55609.406/-100721.657、终局capture2675064；DAgger active33730964/selected2347368/native-safe accepted2106697/query198617。override后current/ring/fixed有效ticks2.2813%/8.6906%/13.7458%，底层条件9.1943%/34.8224%/55.9832%。g96完成点不成为门禁，r07继续使用已封存phase-balanced replay最小因素；r06 GPU末段健康，CPU仅剩g128。


r06固定终态与立即交接（20261018，正式非零尚待确认）：r06完整128代/268435456 fresh ticks、128次finite/nonzero；compute5224.867s、wall6087.909s。8卡release收据released=true、compute_clients=0已亲自核验，唯一owner立即启动r07最小技术smoke，不等待最终CPU；r06最终CPU/JS g128已并行运行。r07已安装封存phase-balanced replay及selected Critic-backed GAE诊断，148份数值source/plan与评估source exact。r08同配方连续后备已具体预声明：`JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r08`，父r07 final g128 raw Actor/Critic+完整pool，train2176341172、technical2775223227，固定8卡128代/268435456ticks、mix/评估不变；收据在r07 `NEXT_RECIPE_PROPOSAL.json`。r06最终146份保存源码snapshot已验证、fresh CPU完整state与next RNG恢复passed，后继新source不能替代前run保存源码验证。完整最终综合将在r07正式非零后进行，监督继续。

固定run `runs/je10_bun_20261018/control_route_v38_r06` 完成128代/268,435,456 ticks；compute=5224.867s、wall=6087.909s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r07`；本次后继配方选择依据 `Completed r06 g0..64 shows opening mean accepted-pair mass 0.187832 (11/60 greedy), return mass 0.446698, persistent fixed drop return80 wrong direction, full delivery2/16 versus parent7/16. Opening labels are reachable (757 accepted/452 query) but rare beside394248 navigation labels. Increase stage-balanced replay of verified opening and ordinary return while holding numerical coefficients, collector/query and scene mixture fixed. This is diagnostic sequential evidence, not proof of gradient competition.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。


r06完整终态及r07正式有效接续（20261018，后继非零后综合）：r06完整128代/268435456 fresh ticks，128次finite/nonzero，compute5224.867s、wall6087.909s；最终146份保存源码与fresh CPU完整state/next RNG恢复passed。8卡compute_clients=0释放后立即交接r07；最小smoke Actor/Critic每卡L2=3.408885/2.256444、finite/保存fresh CPU恢复passed。r07正式g1/2097152 ticks已COMMITTED，8副本exact、finite、Actor/Critic L2=3.463828/2.273113、policy SHA亲自核验，`FORMAL_NONZERO_CONFIRMED.json`及r06终态successor=true均已确认；当前148份数值source/plan exact。唯一CPU/JS owner已开始r07 g0，监督持续，不在报告后退出。

r06全部预声明固定点（未择优、分母固定）：

|代数|JS Hunter难/中/易，各64|Python Tactical64|death/sugar/drop，各16|near/mid/far，各16|recovery/full，各16|
|---:|---|---:|---|---|---|
|0|2/4/10|8|6/6/10|16/16/13|5/7|
|4|1/3/9|9|2/0/6|16/16/8|2/3|
|8|0/3/10|12|0/0/4|16/15/9|0/3|
|16|0/2/11|5|1/0/4|16/16/12|1/4|
|32|2/4/8|8|2/3/8|16/16/12|1/5|
|64|0/3/8|8|1/0/7|16/16/11|1/2|
|96|2/7/7|6|5/1/10|16/16/13|2/5|
|128|1/3/11|17|1/1/5|16/16/7|0/1|

最终相对父点Hunter-1/-1/+1、Tactical+9，相对g96 -1/-4/+4、+11；Hunter终点self0/0/0、Tactical death/self0/0，但g96普通Hunter self2/64和中间波动保留。八路线交付相对父点-5/-5/-5/0/0/-6/-5/-6，相对g96 -4/0/-5/0/0/-6/-2/-4。原生prestock整合机制没有给出稳定路线提升证据；不得用Tactical终点提升覆盖JS路线与Hunter差异。完整曲线图`reports/je10_r06_curve_20261018.png`已包含全部点并人工view核验，训练事件按natural-full tick归一，与JS/Python不混池。

最终full窗口→目标7/7（episode共现）、首方向3/5、持续758/1145、进度785/2105、pickup4/16→delivery1/16、death6/self0、恢复11/14；首控15→归因3→pop14主动12→因果kill3/存活3。相对父点首方向-21.8182pp、持续-3.1255pp、进度-2.3476pp、拾包/交付各-37.5pp、恢复+9.8214pp、归因-13.3333pp；人口/机会改变与缺分母null独立记录。drop旧JS首方向16/16、持续3510/6157、进度3451/6246、pickup16/16→delivery5/16、death0/self0、恢复16/16。原墙death/sugar/recovery均pickup8/8→delivery0/8，return2688/2087/2323ticks、oracle0/187/0；death分层终点death0却交付0，表明不能只用存活率代替路线能力。

最终joint FIFO valid14671：导航7391/greedy2957/mass.400843、opening56/greedy17/mass.330983、ordinary7224/greedy3801/mass.512364；返程4032/greedy1906/mass.476660。精确Actor input+mask重复5组、无无公共pair冲突，有限FIFO不证明总体无冲突。实际8192末状态prestock active428/ready101/preparing327、staged13/clear8/long3/place2；全run active12811949/selected825732/accepted752391/query140823，其中navigation750467、opening1924/query1032，既有可达标签也有严重阶段稀疏。末状态前提计数非全轨迹成功率。r07新增实际每半replay计数，g1 current/history各valid256、ordinary160/return96/navigation80/opening16；history来自fresh FIFO fallback重复，不能当独立历史机会。

最终唯一Review game0/seed2026101800：drop初次真实偏离变为64步，实际左+.681250但上+.731250更优；return81左-.257496而右+.337502/下+.237499，近似JS reward-GAE-.016523604；far107右-.202499而上/左+.202501，GAE+.007016748非动作Q。mid92上+.172500为新field最佳，旧JS误报分列。固定full无记录route首偏航，不换seed补造。drop81向左概率.995055；返程4032 FIFO中同terrain/base/origin且L1≤1共14条，左target接受0、greedy拟合3、target概率均值.236103，完整Actor输入exact0；这支持执行/分布缺口，不构成同输入矛盾或未曾访问的证明。原生hash、动作、事件、阶段、目标、势能与奖励/近似优势链均保留唯一Review来源。r07偏航收据已实际保存Critic-backed collector GAE/value target/estimate/logp，明确区别于zero-value重建GAE，未改优化。

最终累计control gross343451.750/net258465，route+72225.692/-134363.487，capture终局3606192仍主导；DAgger active44079409/selected3134060/native-safe accepted2811055/query257225。override后current/ring/fixed有效ticks2.4129%/8.5026%/14.2882%，底层条件9.5318%/33.5582%/56.9100%。natural-full JAX事件own_window197189/ownkill132083/pickup4538/steal100820/capture61534，对应每百万full tick窗口981.202、capture306.190，不能用训练bot事件增长替代JS自然完整局能力。

证据排序根因假设及已启动最小改动：①开路joint标签可达但回放稀少、拟合不足，返程又明显退化；r07只改变current/history阶段分配（return12/drop4/ready-stock4/navigation10/opening2每半32），实际量/覆盖/fit可逐点验证。②原墙拾包前未稳定完成通路准备，拾包后仍有负方向执行；有限安全开路见证与准备阶段目标可观测性是后续待检假设，不能凭低fit证明梯度竞争。③真实完整对局、威胁恢复、JS战斗和不同地图覆盖仍不足；按实际曝光和固定失败状态继续诊断，不一次改mix/系数/Actor输入。完整源hash链、曲线、normalized delta、阶段FIFO、actual preparation、capped reward/advantage、固定Review、ranked hypotheses保存r06 `FULL_CURVE_ROOT_CAUSE_REVIEW.json`。原生Bun806/open墙与座位混杂尚未证明跨地图泛化；本实验目标尚未完成，r07固定128代继续，r08具体连续后备有效，后续proposal依据固定完成点准备，无能力/admission/promotion门禁。


r07 g0/g4异步评估（20261018，仅诊断）：g0与父Actor/Python Tactical/JS/固定路线结果exact parity（Actor逐叶value/dtype/shape，序列化SHA可不同）；g4 Hunter难/中/易0/0/14、Python Tactical12（各64），death/sugar/drop/near/mid/far/recovery/full交付1/0/7/16/16/8/0/0（各16）。相对g0尚早期波动，不用于停训门禁。g4 actual replay current/history各256有效：return各96 (37.5%)、navigation88/80、opening8/16，replay来源含current/historical fallback与重复标签，不是独立机会数。唯一CPU/JS evaluator完成g0，继续预声明g8及后续点；r07 GPU健康固定128代继续。


r07 g8 COMPLETE后消费固定漏斗并记录实际回放分布；八路线交付death/sugar/drop/near/mid/far/recovery/full=1/1/5/16/16/11/1/6，Hunter难/中/易0/2/13、Tactical10。未形成稳健跨场景结论，固定128代继续；Review与末状态prep审计异步运行。


r07 g16 COMPLETE（20261018，异步诊断）：Hunter难/中/易1/2/9、Python Tactical13；death/sugar/drop/near/mid/far/recovery/full交付1/2/8/16/16/12/1/6。相较g8掉包5→8、full维持6，Hunter hard0→1但normal2、easy9有波动。actual replay各半valid256、return96，opening current/history 2.05%/5.91%，navigation35.4%/31.6%；FIFO重复/回退口径已明示。g32固定CPU评估开始，GPU健康run继续，早期点不构成能力结论或门禁。


r07 g32/g64固定诊断及r08具体proposal（20261018）：g32 Hunter难/中/易0/2/20、Python Tactical16；death/sugar/drop/near/mid/far/recovery/full交付4/4/11/16/16/10/0/2。g64 Hunter0/5/6、Tactical6；交付3/2/6/16/16/10/3/0。full g64窗口目标10/10（episode共现）、first3/4、持续574/953、进度618/5382、pickup5/16→delivery0/16、death5/self0、恢复39/40，首控43/归因6/pop42主动36/kill6存活6。局部掉包/战斗改善未持续，不因短点回落停健康run；当前g77 finite/nonzero。

联合FIFO：g4/8/16/32/64开路mass.969344/.990422/.993732/.965650/.958427，greedy24/25、32/32、24/24、27/28、22/23；导航mass.320527/.344536/.297872/.352904/.369536；返程.705250/.614773/.598365/.584889/.559542。g64 valid13091、navigation7115/opening23/ordinary5953/return2831；精确输入重复7组无无公共pair冲突，有限FIFO/in-sample分布不同，不能证明梯度竞争或全分布无冲突。实际8192末状态prep active/ready及完整分层在`ACTUAL_PREPARATION_G64_SNAPSHOT.json`，preparing303/staged26/clear16/long+place5；开路集合高fit仍未转化原墙闭环。

actual replay g1..64每半return37.5%，current导航35.5835%/opening1.9165%，history导航31.7993%/opening5.7007%；current/history fallback、重复标签和固定一次selected batch而非optimizer使用次数均明示。capped g1..32已保存最早行导航315中264当步route负，其中157 raw Critic-backed GAE正，raw GAE中位.025693；返程44中34当步route负、19 raw GAE正，中位.045606。raw GAE在PPO per-device normalization之前，不能凭其正号断言policy-gradient正向，zero-value extra/task近似分列。观测hash非完整原生state，capped前序样本不是总体频率。

g64唯一Review fixed game0：drop1左+.341251为新场最佳，return80左-.157497而右+.267498，近似JS GAE-.018462911；far52左-.157499而右+.167501，GAE-.011307598。mid92上+.172500为最佳，旧JS误报不直接判错；fixed full无记录首偏航，不换seed补造。g64累计control gross176797.5/net137480.75、route+36716.284/-67835.959、capture1930056，完整parent/previous机会归一化与实际曝光保存在`POINT_G64_OWNER_REPORT.md`/`CURVE_ANALYSIS.json`。

r08 `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r08`已在r07 g77附近具体预声明：父r07最终g128 raw Actor/Critic+完整pool，train2176341172、technical2775223227，固定8卡128代/268435456 ticks；reset mix .35 combat/.20 bridge/.35 full/.10 other native routes、checkpoint每4代、g0/4/8/16/32/64/96/128唯一CPU/JS评估协议与seed不变。唯一因素为joint replay quota每半32 return12/drop2/ready-stock2/navigation15/opening1（current/history各半），不改奖励/DAgger/retention系数、query/native安全、FIFO新增容量、Actor输入/network/PPO/GAE或mix。8项focused+18项relay/continuity（合并26）GREEN，真实双席collector/PPO Actor/Critic非零/reset/保存恢复独立GREEN，初始RED与源码hash封存r07 `NAVIGATION_REPLAY_TDD.json`/`NEXT_RECIPE_PROPOSAL.json`；原同配方后备保存。

同g64已保存8设备FIFO、固定16 sampler seeds、4096 history draw：当前return1536/navigation1312/opening224；候选return1536/navigation1936/opening112，不同FIFO导航1709、开路28（与g32同计数但snapshot/sourcehash独立）；drop/ready-stock各512→256是明确保留风险，开路重复减少并不增加独立机会。g64仍导航low-fit、开路high-fit，选择此最小分配而不同时更改目标可观测性或系数；顺序followup没有同父随机对照，非因果结论。r07 g96/final只诊断及准备r09，固定run不早停；终态先8卡release→r08最小smoke/恢复→formalg1真实非零，再综合最终CPU，不等报告或阈值。

唯一行政owner已仅重载一次以加载未来navigation installer（r07 g77，20261018）；原GPU worker与唯一CPU/JS evaluator未重启，r07数值148份source/plan exact。run本地`ADMINISTRATIVE_NAVIGATION_RELOAD.json`为收据；数值candidate仍未安装，行政接线非训练restart。r08启动时会自动预声明r09同配方后备，持续迭代不静默空卡。


r07 g64 frozen-reference方向补证（20261018，只读）：每phase最多256条按device/slot均匀固定抽样，navigation current/parent/g0/stage1 mass=.376618/.329793/.109084/.096184（greedy90/83/32/32，各256），return=.580870/.428434/.125878/.148545，opening current.958427、parent.192939、g0.213069/stage1.191110（23条）。这表明现有保留参考在新目标上方向经常不一致，但FIFO不是在线PPO分布、未计算loss梯度或权重贡献，不能由此证明KL导致低fit或直接改系数。收据`REFERENCE_ORACLE_ALIGNMENT_G64.json`保存checkpoint SHA和限制；r08仍只改回放配额，该潜在竞争/目标可观测性留待后继固定证据验证。r07 g84健康finite/nonzero，监督持续。


r07 g96固定完成点（20261018）：production JS Hunter难/中/易0/3/8、Python Tactical16（各64），相对父点-1/0/-3、-1，相对g64 0/-2/+2、+10；Hunter self全0、Tactical death/self0/0。death/sugar/drop/near/mid/far/recovery/full交付1/1/9/16/16/11/2/1（各16），相对父点0/0/+4/0/0/+4/+2/0，相对g64 -2/-1/+3/0/0/+1/-1/+1。完整中间曲线波动仍大，不择最佳子任务宣称跨场景稳定提升，不中断健康run。

full窗口目标10/10（episode共现）、first8/8、持续802/1383、进度865/4865、pickup8/16→delivery1/16、death7/self0、恢复36/39；首控40→归因13→pop40主动36→因果kill13存活13。原墙death/sugar/recovery均pickup8/8、delivery0/8，return3291/1716/2379ticks且oracle均0。joint FIFO valid13562：navigation7177/greedy2768/mass.389691、opening10/greedy10/mass.999517、ordinary6375/greedy3655/mass.558973、return3236/greedy1842/mass.564386。实际末状态preparing331/staged18/clear8/long+place1；开路fit高、到stage和完整路线仍稀少，有限FIFO与末状态不是整体能力证明。

g96唯一fixed Review：drop65实际上+.341250为新场最佳，return80左-.157497而右+.217503，近似JS GAE-.021150073；mid100右+.157500低于上+.232500但为正进度，GAE-.007793910；far108右-.202497而上/左+.162500，GAE-.007254031。原生hash/事件/动作/阶段/目标/势能/近似优势链保存`REVIEW_G96_PHASE_FIELD_COMPARISON.json`，full固定game0无route首偏航，不另选seed。reward/advantage实值与zero-value近似在capped收据分列、PPO normalization之前限定不变。

累计control gross259731.5/net199257.5、route+52925.367/-101912.270、终局capture2917776；DAgger active32659608/selected2347530/native-safe accepted2120684/query181633。override后current/ring/fixed有效ticks1.9449%/10.2995%/12.7115%，底层条件7.6859%/41.1957%/51.1184%。r08封存navigation_replay仍是单一配额因素，源hash/技术收据exact；r07 g106健康，终态先接续r08再综合最终CPU。


r07固定终态及r08正式接续核验（20261018）：r07完成128代/268435456 fresh ticks，128次finite/nonzero，compute5277.189s/wall6153.142s；最终checkpoint8副本exact，8卡release已核验compute clients0，148份封存源码的fresh CPU完整状态/下一RNG恢复passed。首次行政交接因解释器缺少NumPy在r08目录创建前失败，使用项目训练解释器恢复同一预声明配方/seed；保留本地原故障和恢复收据，未更改已完成r07。随后r08最小smoke Actor/Critic每卡L2 3.297072/2.215420，finite/nonzero及保存/fresh CPU恢复通过；正式g1/2097152 ticks已COMMITTED且policy SHA核验、8副本exact、每卡Actor/Critic L2 3.273269/2.239966，正式非零亲自确认。无能力/admission/promotion门禁，正式训练已先于r07最终综合继续。

r08 ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r08`，父r07 final g128 raw Actor/Critic+完整pool；父Actor75叶value/dtype/shape exact。train2176341172、technical2775223227；149份数值source和plan seal exact。唯一新数值因素为每半32 joint replay：return12/drop2/ready-stock2/navigation15/opening1；mix .35 combat/.20 bridge/.35 full/.10 other native routes，8卡128代/268435456 fresh ticks、每4代恢复checkpoint和g0/4/8/16/32/64/96/128唯一CPU/JS固定评估均不变。r09同配方连续后备已预声明，父r08 final g128，train219297504、technical646635502；只在后续固定点证据和TDD支持时替换为单轮最小改动，最终CPU不延迟GPU交接。


r07完整固定曲线综合（20261018，已先确认r08正式非零）：全部g0/4/8/16/32/64/96/128均COMPLETE，唯一production Review保持game0/seed2026101800，不择seed或最佳checkpoint。图 `reports/je10_r07_curve_20261018.png` 已更新全部固定点并目视核验，完整计数/机会归一化父点及前点变化保存在run `CURVE_ANALYSIS.json`、`POINT_G128_OWNER_REPORT.md`、`FULL_CURVE_ROOT_CAUSE_REVIEW.json`。

|g|JS Hunter难/中/易（各64）|Python Tactical（64）|死亡/糖泡/掉包交付（各16）|近/中/远返程交付（各16）|恢复/自然完整局交付（各16）|
|---:|---|---:|---|---|---|
|0|1/3/11|17|1/1/5|16/16/7|0/1|
|4|0/0/14|12|1/0/7|16/16/8|0/0|
|8|0/2/13|10|1/1/5|16/16/11|1/6|
|16|1/2/9|13|1/2/8|16/16/12|1/6|
|32|0/2/20|16|4/4/11|16/16/10|0/2|
|64|0/5/6|6|3/2/6|16/16/10|3/0|
|96|0/3/8|16|1/1/9|16/16/11|2/1|
|128|0/1/6|12|0/1/7|16/15/9|2/2|

最终相对父点Hunter−1/−2/−5、Tactical−5；相对g96为0/−2/−2、−4。Hunter self均0，Tactical death/self1/1须保留报告。路线最终相对父点−1/0/+2/0/−1/+2/+2/+1，前点−1/0/−2/0/−1/−2/0/+1。中期自然局6/16未保持，最终不足以支持稳定跨场景提升。固定run已照常完成，以上指标不作能力门禁。

最终natural full漏斗：窗口→目标9/9（episode共现，非逐事件因果转化）、首oracle一致3/5、持续700/1241、进度768/3961、拾包5/16→交付2/16、恢复威胁切换29/31；death7/self0。首次控制32→施害归因7→最终pop32/主动接触26→因果kill7/存活7；预置糖泡、首次新控、主动pop分开，不能混成pop/control成功率。相对父点持续一致率−9.7947pp、目标ticks进度−17.9031pp，拾包/交付各+6.25pp；相对g96首一致−40pp、持续−1.5838pp、进度+1.6090pp，机会数随轨迹变化不作等暴露因果比较。原墙death/sugar/recovery均拾包8/8→交付0/8，返程2740/3111/1637ticks且oracle机会均0；地形与座位混杂，不能称多地图泛化。

唯一fixed Review终点诊断：drop1实际左+.341251为新连续场最佳（旧JS oracle假阳性）；真正drop返程80左−.157497，对照右+.267498/下+.215000，近似JS GAE−.017917143。far51左−.082497、下+.337502/右+.277504，近似GAE−.006055049。stock3右−.153751对照上+.021250是旧direct-stock目标，当前准备stage需分列。fixed full无记录route首偏航，不换seed补造。完整原生state hash、动作、事件、阶段、目标、势能和近似优势链在 `REVIEW_G128_PHASE_FIELD_COMPARISON.json`；这些一原生步端点不是Q或完整时间安全证书。drop80对应Actor左概率.989764，保存FIFO邻近同地形/基地/携带origin18条，其标签均不接受左、greedy拟合7/18、集合mass.361321，但完整Actor输入exact0，不能直接当相同状态标签冲突或证明从未访问。

joint FIFO最终valid13607；navigation7307/greedy2649/mass.372898，opening17/greedy16/mass.937785，ordinary6283/greedy3605/mass.555079，return3151/greedy1750/mass.550638。精确输入+mask重复6组、无无公共pair冲突，仅限有限FIFO。末状态8192 learner中preparing304/staged23/clear15/long+place2，不能当全轨迹成功率。全run prep active12960953/selected841267/accepted754547/query137487，其中navigation753246/opening1301、opening query366；新增开路机会仍稀少。actual replay current/history返程均37.5%，导航35.7239%/31.7627%，opening1.7761%/5.7373%，是每代selected batch包含fallback重复，不是独立native机会或每optimizer-step独立暴露。

capped真实信号：2048条截断偏航，按generation/device/env/control-episode保留已保存最早后1575条。导航994中868当步route负、457距离增加，其中490 raw Critic-backed GAE正，median .018866；返程279中229当步route负、106 raw GAE正，median−.007895。raw GAE在per-device PPO normalization之前，其符号不能当归一化策略梯度方向；zero-value extra/task近似与实值分列。截断前序、重复episode与observation-only hash限制保留，不能把这些计数当总体偏航率。

训练累计control gross343636.5/net261043.75、route+69411.458/−135858.351/net−66446.893、capture3882360，终局主导；control gross/capture8.8512%，route gross absolute/capture5.2872%。自动段落“总绝对合计”分母是combined reward的绝对合计，不是route自身gross，不能混读。DAgger active42827181/selected3134255/native-safe accepted2833341/query236986，几何独立候选分母未记录，保留null。override后current/ring/fixed有效ticks1.9903%/9.8986%/13.5220%，底层league条件7.7498%/39.0324%/53.2178%，JAX对手曝光不与Python/JS评估池混合。

证据排序与继续方案：①开路有限FIFO拟合高、准备导航及普通返程低，首要验证r08已封存的单一回放配额因素，保留return12而将navigation10→15；acquisition/opening减少为明确保留风险，不承诺因果改善。②原墙准备/返程闭环、有限native opening witness和真实首次偏航覆盖仍弱，r08按固定点检查实际导航推进/见证/DAgger覆盖后再选择r09一个因素。③冻结保留参考在新route目标上的方向常不一致，raw GAE和局部route符号也可不同，但尚无在线loss梯度贡献证明，不据此同时改KL系数/奖励/输入。若r08最终仍不佳，用整曲线和首偏航证据替换r09同配方后备的最小proposal，按RED→GREEN→最小技术smoke→8卡训练继续；健康run不因中间回落停训，终态先接续后综合。目标仍未完成，唯一CX owner持续监督r08。


r07保留竞争的进一步只读补证（20261018，r08继续）：在g128已保存FIFO按device/slot固定均匀取navigation64/return64/opening17，同一组Actor输入，使用实际网络参数梯度分别计算joint DAgger(.2)、假设非combat的g0/stage1 KL(.025/.01)、parent非携带KL(.05)。导航DAgger与合并保留梯度cosine−.459699，norm5.048387/2.298534，dot−5.334294；返程cosine−.571244、norm5.398283/1.523787、dot−4.698960，parent携带mask梯度为0。开路cosine−.387877、norm2.284850/.401676。负dot支持这些固定状态上存在局部参数梯度对抗，但导航反向投影约为DAgger自身norm²的20.9%，不能据此认定保留项压倒训练或是唯一根因。

收据r07 `FIFO_CONDITIONAL_GRADIENT_ALIGNMENT_G128.json`保留固定索引、checkpoint hash、权重和限制：FIFO不是online PPO batch，缺scene/danger上下文而明确使用weak条件；未包含PPO/value/entropy/其他expert、跨状态混合或Adam预条件，不是实际完整更新梯度或因果证明。不新增轨迹/query/评估episode、不用GPU、不导入训练。r08仍只改变replay配额，等待其固定点导航/返程/保留证据后再选择r09一个数值因素，不直接同时改KL。

后继诊断候选（未安装r08）：`scripts/ppo_advantage_diagnostics.py`与`scripts/ppo_advantage_diagnostic_recipe.py`经8项focused RED→GREEN，仅从每个PPO replica原始rollout优势归约mean/std两个scalar，给已选中的capped偏航补归一化GAE（ratio/clipping之前，非Q或完整loss梯度）；常数批/独立卡归约/原始正但归一化负、无输入变更、幂等installer及固定终态+8卡release约束均覆盖。当前worker/数值源码未改，候选hash与日志在r08 `NORMALIZED_ADVANTAGE_CANDIDATE_TDD.json`。这项非数值可观测性改进不替代r09数值因素的证据选择，也不引入能力门禁。


r08 g4固定完成点（20261018，COMPLETE后）：production JS Hunter难/中/易0/2/11、Python Tactical9（各64）；相对父点暨前点g0为0/+1/+5、−3。Hunter self全0，Tactical death/self3/3（父点1/1），如实记录安全回落，单点不当能力门禁。death/sugar/drop/near/mid/far/recovery/full交付2/0/4/16/16/9/4/2（各16），相对父点暨前点+2/−1/−3/0/+1/0/+2/0。机会归一化完整变化保存在r08 `POINT_G4_OWNER_REPORT.md`、`CURVE_ANALYSIS.json`。

full窗口目标10/10（episode共现）、first1/5、持续887/1412、进度929/3860、pickup5/16→delivery2/16、death6/self0、恢复21/24；首控25/归因6/pop25主动19/kill6存活6。joint FIFO valid7716：navigation5678/greedy2030/mass.370459，opening36/greedy30/mass.852460，ordinary2002/greedy1253/mass.606483，return901/greedy606/mass.666444。actual current/history return均37.5%，navigation48.1445%/46.8750%，opening1.8555%/3.1250%，配额机制已实际生效；尚不能由首点认定能力改善。末状态preparing1105/staged59/clear38/long+place5，不当全轨迹成功率。

g4唯一Review fixed game0：drop return80仍左−.157497，右+.267498/下+.215000，近似JS GAE−.018940562；far51左−.082497、下+.337502，GAE−.003701749。drop1左+.341251和mid98上+.202500均为新场最佳，旧JS偏航不能直接判错。full return1188下+.301861低于上/右+.337502但有正进度，近似GAE−.022848218不代表动作Q；full stock1154等direct-stock场与prestock准备目标分列。原生hash/动作/事件/阶段/目标/势能保存在 `REVIEW_G4_PHASE_FIELD_COMPARISON.json`。

训练累计control gross12228.75/net11674.75，route+2652.948/−3287.276，capture126336终局主导；DAgger active1500112/selected91219/native-safe accepted81560/query9210。override后current/ring/fixed有效ticks1.6725%/8.8433%/7.9035%，底层league条件9.3891%/47.3072%/43.3037%，两类比例不混读。

只读条件梯度g4复核（同固定选样规则，但FIFO分布及parent已随run改变，不是同状态跨checkpoint对照）：navigation DAgger/合并保留cosine−.027716、norm6.338884/.954000、dot−.167605，反向投影仅约自身norm² .42%；return cosine−.634990、norm3.018163/1.365825、dot−2.617609，opening cosine−.399144。相比r07最终导航的局部反向更弱，**不支持凭r07单点就选“导航保留冲突主导”或直接减KL**；继续观察g8/16/32的导航拟合、真实首次偏航与机会覆盖。此审计仍是假设weak非combat条件的FIFO局部参数梯度，非实际在线PPO/完整更新，限制与索引保存在 `FIFO_CONDITIONAL_GRADIENT_ALIGNMENT_G4.json`。r08 GPU固定128代继续，r09数值因素尚未改选。


r08 g8固定完成点（20261018，COMPLETE后）：JS Hunter难/中/易1/1/4、Python Tactical14（各64）；相对父点+1/0/−2、+2，相对g4 +1/−1/−7、+5。Hunter self0/1/0、Tactical death/self0/0（g4为3/3），不因早期安全波动设置门禁。交付death1/sugar1/drop10/near16/mid16/far9/recovery0/full1（各16），父点+1/0/+3/0/+1/0/−2/−1，前点−1/+1/+6/0/0/0/−4/−1。full窗口目标11/11（episode共现）、first2/7、持续465/984、进度485/5135、pickup6/16→delivery1/16、death6/self0、恢复38/39；首控41/归因6/pop41主动36/kill6存活6。原墙death/sugar均pickup8/8→0/8，return2680/2762ticks且oracle0；recovery原墙pickup8/8→0/8但return1181/oracle539，区分新增可达机会与真正闭环交付。

joint FIFO valid15908，navigation11514/greedy4170/mass.371940，opening45/greedy43/mass.928437，ordinary4349/greedy2652/mass.586673，return2136/greedy1363/mass.630200。导航在增加配额后首两个固定点仍低fit，掉包和战斗有恢复但不能称稳定跨场景改善。末状态preparing773/staged47/clear32/long+place3，限定snapshot非全轨迹率。累计control gross28246/net24961.5、route+5569.375/−8622.482、capture249480终局主导；DAgger active3436334/selected187599/native-safe accepted168162/query20264。override后current/ring/fixed有效ticks1.5897%/8.2179%/7.9089%，底层league条件9.1600%/45.7554%/45.0845%。

g8唯一fixed Review：drop return80左−.157497、右+.267498，近似JS GAE−.018197180；mid445右−.202499而左+.162500，GAE−.009171229；far51左−.157499而右+.207500/下+.162500，GAE−.003819139。drop1左+.341251仍是新场最佳，旧JS假阳性分列；full固定game0没有route首偏航，不换seed。完整parent/previous机会归一化与原生hash链在r08 `POINT_G8_OWNER_REPORT.md`、`CURVE_ANALYSIS.json`、`REVIEW_G8_PHASE_FIELD_COMPARISON.json`。

保存状态的阶段历史只读审计g4：1105 preparing learner中固定均匀256条，objective均与stateless重规划相同，staging/first-bricks/distance分别251/256相同；微小.05几何势能方向可比256条，仅2条最佳方向集合不相交，均为changed-stage。收据 `STAGE_HISTORY_GEOMETRY_G4.json`；不是全轨迹或精确Actor标签冲突证明，探测方向不是原生合法oracle/时间安全/Q，不新增fork或训练query。该样本暂不支持把目标历史歧义列为主要根因，也不证明Actor已能学会拓扑目标。r08继续固定预算，下一轮数值因素未改选；继续检查阶段奖励与原生事件机会。


r08 g16固定完成点（20261018，COMPLETE后）：JS Hunter难/中/易2/1/10、Python Tactical18（各64）；相对父点+2/0/+4、+6，相对g8 +1/0/+6、+4。Hunter self均0，Tactical death/self4/4，战斗safe win与自伤必须同时报告，未构成当前硬安全中断判定。交付death0/sugar0/drop12/near16/mid16/far11/recovery1/full0（各16），相对父点0/−1/+5/0/+1/+2/−1/−2，相对g8 −1/−1/+2/0/0/+2/+1/−1。full窗口目标9/9（episode共现）、first2/6、持续616/1099、进度651/3190、pickup4/16→delivery0/16、death6/self0、恢复23/24；首控27/归因4/pop27主动21/kill4/存活3。局部掉包/返程及战斗改善未形成原墙/自然全局闭环，不宣称稳定改善，健康固定run继续。

joint FIFO valid16384，navigation11301/greedy3743/mass.341219（g4/8 .370459/.371940），opening23/greedy22/mass.948377，ordinary5060/greedy2880/mass.554752，return2574/greedy1553/mass.596036。累计control gross54108.25/net45993.75、route+10725.555/−18560.942、capture486792终局主导；DAgger active6793118/selected383915/native-safe accepted344021/query39948。override后current/ring/fixed有效ticks1.4381%/7.9017%/9.4232%，底层league条件7.7654%/42.4103%/49.8243%。完整机会归一化与父点/前点变化在r08 `POINT_G16_OWNER_REPORT.md`/`CURVE_ANALYSIS.json`。

g16唯一Review：drop return80左−.157497、右+.267498，近似JS GAE−.018604175；far51左−.157499、右+.207500，GAE−.005253523。full return1824下−.157501而右+.170715，近似JS GAE+1.444194，说明后续回报可与局部进度相反但不是动作Q；完整原生hash/动作/阶段/目标/势能见 `REVIEW_G16_PHASE_FIELD_COMPARISON.json`，stock direct目标仍与prestock准备目标分列。

【20261019就地更正：以下r08污染末状态计数仅保留为历史证据；“当前不可达/未来爆破改变连通”的真实返程根因解释撤回，不证明真实取包后无路。】

当前携带通路只读补证g8：所有保存rollout-end敌origin携带learner2354个，旧weighted-home距离2354个均finite，其中880个在当前wall+brick图不可达；含pushables后计数相同。自然full携带1709个中757个当前砖阻断；scene1为72/72、scene7为51/56。weighted>=64为116/2354，仅scene1/7，不能把距离clip当自然full主要根因。收据 `CARRIER_TOPOLOGY_G8.json`：这是末状态snapshot非全轨迹率，敌人未来爆破可能改变连通，不当永久不可达证明。native carrier不能放炸弹，有限weighted场不是当前可执行返家路线。

【20261019就地更正：以下历史提案中的“开路前激励/提前规划”解释已撤回；真实取包有可逆入站通路，不需要预开返程通道。】

后继最小候选尚未选入/安装：仅重建共享routing的home2/3 channel时将当前砖视为blocked，drop0/1 channel仍原weighted逐位exact，复用原dirty cache/同schema，不改系数、PPO、scene mix、query或Actor输入。共享home场的legacy carry/bridge和PBRS消费者会共同改变；native DAgger已有独立砖阻断距离，保持其oracle/query不变，明确这是同一field机制的整合影响，非奖励单变量。5项field RED→GREEN覆盖双席、开墙/绕行、多尺寸布局、hard断连和砖内goal；cache/installer及真实collector/PPO/reset/恢复复验独立准备中。即使field修正减少不可执行进度，它本身不会补开路前的即时取包奖励激励，也不会保证提前规划成功，需依g32/64完成点再选择r09最小因素；r08原配方继续128代。


r08 g32固定完成点（20261018，COMPLETE后）：JS Hunter难/中/易0/2/5、Python Tactical11（各64）；相对父点0/+1/−1、−1，前点g16 −2/+1/−5、−7。Hunter self全0，Tactical death/self2/0。交付death4/sugar1/drop11/near16/mid16/far10/recovery2/full3（各16），相对父点+4/0/+4/0/+1/+1/0/+1，前点+4/+1/−1/0/0/−1/+1/+3。full窗口目标11/11（episode共现）、first2/7、持续848/1412、进度899/5170、pickup4/16→delivery3/16、death2/self0、恢复39/41；首控41/归因8/pop41主动39/kill8存活8。局部navigation fit较首点提高、自然局有恢复，仍无稳定原墙/跨场景闭环结论，不变更run预算。

joint FIFO valid14582，navigation8020/greedy3507/mass.438389，opening16/greedy13/mass.785326，ordinary6546/greedy3569/mass.532013，return3240/greedy1912/mass.581544。末状态preparing330/staged22/clear14/long+place3，限定snapshot。累计control gross97364.75/net78942.75，route+19402.718/−36566.718，capture976728终局主导；DAgger active12292086/selected777128/native-safe accepted702931/query71408。override后current/ring/fixed有效ticks2.1842%/8.4045%/11.0021%，底层league条件9.7235%/39.4841%/50.7925%。完整机会归一化与父点/前点变化见r08 `POINT_G32_OWNER_REPORT.md`/`CURVE_ANALYSIS.json`。

g32唯一Review fixed game0：drop return80左−.157497对右+.267498，近似JS GAE−.018555288；far108右−.157499对上/左+.162500，GAE−.010007573。full return1793上−.263573而下+.337502，GAE−.021702117；full stock1601左+.274645但下+.523930，GAE+.027124329，direct-stock与prestock准备stage仍分列。drop1左+.341251为新连续场最佳，旧JS偏航假阳性保留。完整原生hash/动作/事件/目标/势能及近似GAE在 `REVIEW_G32_PHASE_FIELD_COMPARISON.json`。

【20261019就地更正：以下计数包含污染合成初态/末状态，不能支持真实取包不可达；旧拓扑根因解释撤回。】

返家reward/DAgger拓扑偏差g16补证：全部保存敌origin携带learner2274，weighted reward home2274均finite，native DAgger物理home857 unreachable；当前wall+brick不可达亦857（含pushables计数相同）。自然full携带1514，其中698不可达、DAgger不可达698。该结果支持先修reward侧当前可执行拓扑对齐；不把保存末状态比例当全轨迹失败率，不假设未来敌爆破不可能开路。

r09具体proposal已在r08 g46附近选定并封存（20261018，无能力门禁）：`JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r09`，父r08最终g128 raw Actor/Critic+完整pool；train219297504、technical646635502；fresh Adam/RNG/env/EMA-to-raw/FIFO/counters。固定8卡128代/268435456 ticks、mix .35 combat/.20 bridge/.35 full/.10 other native routes、恢复每4代、g0/4/8/16/32/64/96/128唯一CPU/JS评估与全部seed/局数不变。原同配方连续后备保存在 `CONTINUOUS_FALLBACK_BEFORE_CARRIER_FIELD_SELECTION.json`；具体收据 `CARRIER_FIELD_PRODUCTION_CANDIDATE_TDD.json` 与 `NEXT_RECIPE_PROPOSAL.json`。

【20261019历史解释撤回：以下“未来敌爆破/取包前开路”表述基于污染合成末状态，不能解释真实取包；成功原生取包已有可逆入站通路，不再保留真实不可达或预开返程为候选。】

唯一新数值因素：共享routing home2/3 channel将当前砖视为blocked，drop0/1保持原weighted逐位exact。仅一次四通道Bellman搜索不同blocked/brick权重，不添cache字段，不改query、DAgger oracle/新增容量、replay配额、所有系数、PPO/GAE、Actor输入或mix。它共同影响legacy carry/bridge及PBRS的同一返家field，native DAgger独立砖阻断oracle不变，不能声称奖励单变量或同父因果实验。风险：不可执行路线上的局部进度信号会减少；原生pushable移动/未来敌爆破仍需检查；即时条件steal .75仍保留，field修正本身不会保证取包前开路。另附非数值normalized-GAE诊断（两scalar/device，仅capped选中行），明确与数值因素分列。

TDD：9项field/cache/installer、18项relay/continuity、8项normalized-diagnostic合并35 GREEN，初始RED日志均保存；独立真实双席production collector→joint FIFO→完整PPO Actor/Critic非零→terminal reset/PBRS零→host保存恢复下一rollout exact GREEN。candidate未安装r08。唯一行政owner已在g46仅重载以支持后继安装并使用正确训练解释器；首次锁尚未释放造成attach失败，锁释放核验后重试active，原failure本地保留；GPU worker/CPU evaluator均未重启，149份数值source/plan exact。`ADMINISTRATIVE_CARRIER_RELOAD.json`记录行政恢复，不是训练restart。r08 g48 healthy，终态先release→r09最小smoke/恢复→正式g1非零再综合final，r10会在r09启动时自动预声明同配方后备。


【20261019就地更正：以下snapshot结果不证明真实取包返程无路；其支持的旧不可达/预开路解释撤回。】

r08 g32 carrier topology只读复核（20261018）：所有保存敌origin携带learner2238个，reward weighted home均finite；DAgger unreachable783，当前wall+brick不可达783（加入pushables同数）。natural full携带1506个，其中636物理及DAgger不可达。与g8/g16独立snapshot一致支持r09候选，但不是全轨迹失败率。r09 proposal十份source hash再次exact，r08未安装carrier/normalized候选。r08 g59/123731968 ticks healthy，唯一CPU队列等待g64，无评估failed。r10仅在r09启动时自动预声明；目前具体封存后继为r09。


r08 g64固定完成点（20261018，COMPLETE后）：JS Hunter难/中/易0/1/6、Python Tactical11（各64），相对父点0/0/0、−1，前点g32 0/−1/+1、0。Hunter self全0、easy trade1，Tactical death/self0/0。交付death4/sugar3/drop7/near16/mid16/far14/recovery3/full3（各16），相对父点+4/+2/0/0/+1/+5/+1/+1，前点0/+2/−4/0/0/+4/+1/0。远返程与部分窗口改善，掉包和战斗波动，不能由中段单点宣称跨场景稳定提升。曲线 `reports/je10_r08_curve_20261018.png` 已包含g0..64全部完成固定点并目视核验。

full窗口目标12/12（episode共现）、first7/8、持续899/1285、进度956/6078、pickup8/16→delivery3/16、death5/self0、恢复42/46；首控48/归因8/pop48主动42/kill8存活8。原墙death/sugar仍pickup8/8→delivery0/8，但return1768/oracle78和return1557/oracle1471，说明动态开墙产生过部分合法返程机会；应区分“无路”和“已有路但未完成”，不全归因为静态断连。joint FIFO valid13472，navigation6165/greedy2912/mass.472899（g32 .438389），opening29/29/mass.992802，ordinary7278/greedy3862/mass.520833，return3737/greedy2029/mass.540372。末状态preparing327/staged40/clear30/long+place6，snapshot非全轨迹率。actual current/history return37.5%，navigation48.8464%/46.9421%，opening1.1536%/3.0579%。

g64唯一Review：drop return80左−.157497、右+.267498，近似JS GAE−.013786283；far54左−.202499、右+.157499，GAE−.008416090。native sugar return689下−.337502，其他三向+.022499，GAE−.001508331；death return1139下+.184994为新连续场最佳、mid94上+.202500亦最佳，旧JS假阳性不能直接算真偏航。near2下+.157500低于右+.267500但有正进度；drop64左+.681250低于上+.731250但正进度；direct-stock与prestock准备目标仍分列。fixed full game0无记录route首偏航，不另选seed；完整原生hash/动作/事件/阶段/目标/势能/近似GAE保存在r08 `REVIEW_G64_PHASE_FIELD_COMPARISON.json`。

累计control gross181491.75/net143075.5、route+34887.912/−74005.084、capture2009904终局主导；DAgger active22325570/selected1563852/native-safe accepted1432331/query125315。override后current/ring/fixed有效ticks2.5564%/8.6878%/13.0163%，底层league条件10.3198%/36.3557%/53.3245%。完整父点/前点机会归一化变化在r08 `POINT_G64_OWNER_REPORT.md`/`CURVE_ANALYSIS.json`。

g64携带拓扑snapshot2247个，reward weighted均finite，physical/DAgger不可达624；natural full1531中478不可达，较g32 snapshot636/1506下降但分布不同，不能当同样本改进率。r09封存shared reward-home field对齐继续有效；原墙已有路未交付、开路/导航/返程fit同时保留为后续弱漏斗，不据局部改善提前结束r08。GPU g76/159383552 fresh ticks healthy，唯一CPU队列完成到g64、等待g96，无failed。


r08 g64固定Review与learner FIFO覆盖补证（20261018，只读既有状态）：固定real_drop return80动作左，checkpoint在该JS Actor输入的左概率.989387；完整Actor输入精确匹配0条，限定“相同wall/brick、pushable、base、携带origin且位置L1<=1”几何匹配17条，其中标签下9/右8、无标签接受左，9/17当前greedy在集合内，目标概率均值.495685。全return FIFO3737条mass.540372。收据 `REVIEW_G64_FIFO_COVERAGE.json`：附近几何监督确实存在，但威胁/时间/其他global等不完全匹配（平均global绝对差.057899），不能当exact标签冲突或全分布覆盖结论，也不能证明KL为根因。该固定状态仍需检查执行泛化、归一化优势和时序安全；r09只修共享reward-home field，现有数值run不改。r08当前g84/176160768 ticks健康，唯一CPU队列等待g96，无failed，149份source/plan与后继10份source hash核验exact。


r08 g96固定完成点（20261018，COMPLETE后消费）：production JS Hunter难/中/易0/0/11、Python Tactical11（各64）；相对父点0/−1/+5、−1，前点g64 0/−1/+5、0。Hunter self0/0/1，Tactical death/self0/0。八路线交付death1/sugar2/drop11/near16/mid16/far10/recovery1/full2（各16），相对父点+1/+1/+4/0/+1/+1/−1/0，前点−3/−1/+4/0/0/−4/−2/−1。full窗口目标8/8（episode共现）、first4/5、持续634/1136、进度709/5381、pickup5/16→delivery2/16、death6/self0、恢复38/40；首控42/归因12/pop42主动35/kill12存活12。原墙death/sugar/recovery均pickup8/8→delivery0/8，return1359/1258/1492ticks且oracle均0，与g64出现部分动态可达机会不同，不能把不同轨迹snapshot混为改善率。

joint FIFO valid13481，navigation5469/greedy2798/mass.508226（g64 .472899），opening52/greedy40/mass.781789（g64 .992802），ordinary7960/greedy4080/mass.505446，return4149/greedy2141/mass.515095（g64 .540372）。精确Actor输入重复12组无无公共pair冲突；FIFO分布改变，非holdout或全分布无冲突证明。末状态preparing285/staged47/clear27/long+place5，限定snapshot。actual replay g1..96 current/history return均37.5%，navigation48.7712%/46.9198%，opening1.2288%/3.0802%；含fallback与重复标签，非独立native机会或optimizer使用次数。

g96唯一production Review固定game0/seed2026101800：drop return81左−.257496，右+.337502/下+.237499，JS近似GAE−.018738240；death stock2左−.241249，向上+.027500，GAE−.002251335；sugar stock3右−.153751、向上+.021250，GAE−.005980061，direct-stock与prestock准备目标分列。mid138上+.182500为新连续场最佳，旧JS偏航假阳性；far52下+.217501低于右+.267502但正进度，GAE−.002704147。drop1左+.341251亦新连续场最佳。fixed full game0无记录route首偏航，不换seed；原生hash/动作/事件/阶段/目标/势能/近似GAE在 `REVIEW_G96_PHASE_FIELD_COMPARISON.json`。

【20261019历史解释撤回：以下“未来敌爆破/取包前开路”表述基于污染合成末状态，不能解释真实取包；成功原生取包已有可逆入站通路，不再保留真实不可达或预开返程为候选。】

累计control gross266705.75/net208612.5、route+50207.498/−113108.515、终局capture3071784仍主导；DAgger active32726743/selected2350609/native-safe accepted2160329/query179277。override后current/ring/fixed有效ticks2.5772%/9.0914%/13.4715%，底层league条件10.1501%/36.4873%/53.3626%，两口径分列。携带末状态2238个中weighted finite2237、physical wall+brick不可达495（含pushables同数）、DAgger不可达496；natural full1373中346物理/DAgger不可达。scene6有1例物理finite而缓存/DAgger非finite，故不笼统声称两field逐例恒等。非全轨迹或永久不可达结论，未来敌爆破仍能改通路。当前曲线包含全部g0..96固定点并已目视核验；导航FIFO fit上升未稳定转化原墙/完整局，不早停健康r08。r09仍只修shared reward-home field并附非数值normalized-GAE诊断；终态先接后继正式非零，再综合final。


r08 g96条件网络梯度补证（20261018，只读固定FIFO、每phase最多64均匀device/slot样本）：navigation DAgger/合并weak保留cosine−.470144，norm4.482342/1.555396，dot−3.277758，反向投影约自身norm²16.31%；return cosine−.641673、norm4.298952/.884913，dot−2.441053（约13.21%）；opening为正cosine+.446850。相对g4导航−.027716/.42%投影表明这种条件局部竞争会随checkpoint/FIFO变化，不是稳定同状态因果对照，也不足以直接改KL。系数假定非combat weak g0 .025/stage1 .01/parent .05，joint DAgger .2；不含在线PPO/value/entropy/其他expert/Adam或实际scene/danger选择。收据 `FIFO_CONDITIONAL_GRADIENT_ALIGNMENT_G96.json`；该证据保留作r09/r10诊断，既定r09唯一数值因素不变。r08 g109/228589568 ticks仍finite/nonzero，CPU固定点g96完成、仅等待g128，无failed。


地图范围入口只读审计（20261018）：`web/assets/maps/levels.json`带bun标记且native_rule3的资产仅1张（806）；`jax_bomb/je07_bun_env.py`的加载器显式选择806，固定JS路线亦使用同一level。现有多尺寸纯field/oracle测试及Bun806原墙/开墙不等于不同原生地图泛化，且stock fixture的seat与墙状态混杂已持续明示。后续真实跨地图/变体协议应独立预声明并验证native目标、出生与cache一致性；当前r08/r09封存数值因素不在健康run中扩张。该能力目标仍未完成，不把资产单一误报为训练技术阻塞。

固定run `runs/je10_bun_20261018/control_route_v38_r08` 完成128代/268,435,456 ticks；compute=5464.397s、wall=6330.240s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r09`；本次后继配方选择依据 `r08 completed g4/8/16/32 navigation joint set mass .370/.372/.341/.438 while original-wall pickup-to-delivery still missing. Saved g8 carrier states2354:880 current wall+brick disconnected though weighted reward home distance finite; g16 states2274:857 disconnected and existing native DAgger distance already unreachable. Full carriers g16:698/1514 disconnected. Align reward-side carrier home topology with current executable brick-blocked routes before changing coefficients or oracle coverage.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。


r08固定终态及r09正式接续亲自核验（20261018）：r08正常完成128代/268435456 fresh ticks，128次finite/nonzero，compute5464.397s、wall6330.240s、atomic_stop=false。终态先核验8卡compute_clients0及released=true，随后立即交给预声明r09，不等待final CPU/JS。r09最小smoke每卡Actor/Critic L2 3.225763/2.283982，完整保存及fresh CPU恢复passed；正式g1每卡Actor/Critic L2 3.247570/2.290320、8副本exact、finite，2,097,152 fresh ticks，COMMITTED policy SHA亲自核验。`OWNER_HANDOFF_VERIFIED.json`记录父raw Actor75叶/Critic75叶/完整pool75叶value/dtype/shape exact，r08保存149份source及r09当前/保存153份source与plan seal exact、seed符合预声明。r08完整终态状态/next-RNG fresh CPU恢复passed。r09正式持续128代，唯一CPU/JS owner已接过全局锁开始g0，无能力/admission/promotion门禁。

r10同配方连续后备已在r09启动时实际预声明：`JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r10`，父r09 final g128 raw Actor/Critic+完整pool；train2553551348、technical926923147；fresh Adam/RNG/env/EMA-to-raw/FIFO/counters；8卡128代/268435456 fresh ticks、mix .35 combat/.20 bridge/.35 natural-full/.10 other native routes、每4代恢复点与g0/4/8/16/32/64/96/128唯一CPU/JS协议和seed不变。后继保持carrier-route-fields及非数值normalized-GAE诊断，无新增数值因素；可在终态前以已测的最小证据因素替换，不等final阈值或人工批准。r08 final综合此后异步执行，报告完成不结束r09监督。


r08完整终态综合（20261018，在r09正式g1非零亲自核验后）：JS Hunter难/中/易1/4/7、Python Tactical8（各64），相对父点+1/+3/+1、−4，相对g96 +1/+4/−4、−3；Hunter self0/0/1、Tactical death/self0/0。death/sugar/drop/near/mid/far/recovery/full交付2/3/8/16/16/9/0/3（各16），相对父点+2/+2/+1/0/+1/0/−2/+1，相对g96 +1/+1/−3/0/0/−1/−1/+1。固定曲线所有8点均保留且无平滑/选优：full2/2/1/0/3/3/2/3，far9/9/9/11/10/14/10/9，Tactical12/9/14/18/11/11/11/8。局部导航fit提高及部分窗口交付不足以证明稳定跨场景规划，恢复威胁与Tactical仍弱，不把训练JAX capture增加当JS完整局能力。完整曲线 `reports/je10_r08_curve_20261018.png` 已目视核验。

full最终window→target13/13（episode共现）、first6/9、持续1368/2432、进度1530/4815、pickup8/16→delivery3/16、death11/self0、恢复32/38；首控39/归因12→pop39主动28→因果kill12/存活12。原墙death/sugar/recovery拾包均8/8、交付均0/8，return1677/402/2536ticks、oracle0/16/0；sugar有短暂动态可达机会但未交付，不能全归于无路。open recovery已有1695合法返程机会仍0/8交付，现有可执行返程与威胁切换同样需要继续优化。

最终joint FIFO valid13811：navigation5115/greedy2513/mass.487682，opening50/greedy44/mass.868106，ordinary8646/4338/mass.490407，return4890/2452/mass.495114。navigation mass g4/8/16/32/64/96/128=.370459/.371940/.341219/.438389/.472899/.508226/.487682；return=.666444/.630200/.596036/.581544/.540372/.515095/.495114。FIFO随run分布改变、非holdout；10组精确Actor输入重复未见无公共pair冲突，不证明全分布无冲突。actual current/history replay return37.5%，navigation48.6877%/46.9208%、opening1.3123%/3.0792%，含回退和重复标签；末状态preparing284/staged36/clear25/long+place5，非全轨迹率。全部prestock active12322692/selected810966/accepted677364/query118232、导航accepted674520/开路2844/query599，独立机会仍受native窗口/见证制约。

最终唯一Review固定game0/seed2026101800：drop return80左−.157497，对右+.267498，JS近似GAE−.017613861；far52左−.207500、右+.162502/下+.122501，GAE−.006273946；native sugar return1087左−.213215、右+.337502，GAE−.018334238。death/sugar stock3右−.153751，对上+.021250，direct-stock与prestock准备目标分列。mid138上+.182500及drop1左+.341251是新连续场最佳，旧JS偏航假阳性保留。fixed full game0无记录首偏航，不换seed。drop return80 checkpoint左概率.966465，完整Actor输入匹配0、限定几何匹配18条却无标签接受左、目标mass.490868；不能把几何匹配当威胁/时序完全相同或exact标签冲突。原生hash/动作/事件/阶段/目标/势能/近似GAE与FIFO覆盖在最终Review收据。

最终累计control gross352037.25/net273350.75、route+65152.882/−153909.318、终局capture4115280仍主导；DAgger active43344035/selected3137393/native-safe accepted2885239/query234865。override后current/ring/fixed有效ticks2.3501%/9.6471%/13.5726%，底层league条件9.1686%/37.8923%/52.9390%，分母不同。natural-full JAX窗口192564/kill129409/pickup4379/steal109079/capture76716，按百万full tick窗口978.520、capture389.835；该bot暴露与JS/Python分列。

证据排序与下一最小改动：①最终携带末状态2345中507当前wall+brick及native DAgger不可达、旧reward weighted2345均finite；natural full1468中351不可达，无weighted>=64，因此先对齐shared reward-home field。已启动r09仅改home2/3通道、drop0/1 exact，legacy carry/bridge+PBRS共同消费者，native DAgger本已brick-blocked不改；不是单奖励因果实验，顺序continuation无同父对照，未来敌爆破仍能改连通。②导航/返程fit、stage/独立opening机会及原墙完整闭环未稳，r09配额/query/系数不变，以fixed首次偏航和新增归一化GAE判别最弱漏斗，r10随后选一个因素。③条件FIFO网络梯度在g4/g96的导航反向投影由约.42%变16.31%，不等于真实在线完整gradient，当前不直接减KL；目标历史几何审计仅2/256方向集合不交，不支持把历史目标歧义列首因。完整固定曲线、机会归一化delta、奖励/优势、DAgger/FIFO/实际曝光、携带拓扑、保留和ranked hypotheses封存r08 `FULL_CURVE_ROOT_CAUSE_REVIEW.json`。目标尚未达成，r09健康长训继续，r10具体连续后备有效，报告完成不结束监督。


r09 g0父点评估parity（20261018，COMPLETE后）：production JS Hunter/Python Tactical/全部固定route/control-funnel结果与r08 final逐项exact，raw Actor/Critic与pool初始化已独立核验。收据 `PARENT_BASELINE_PARITY.json` 与 `OWNER_HANDOFF_VERIFIED.json`。r09正式g4/8388608 fresh ticks finite/nonzero；后续固定点额外只读审计ready routing cache的drop旧weighted exact与home物理brick-blocked exact，不新增对局/query、不设门禁。r10同配方后备已有效生成，当前8卡持续训练。


r09 g4固定完成点（20261018，COMPLETE后）：production JS Hunter难/中/易1/3/15、Python Tactical13（各64），相对父点/前点g0 0/−1/+8、+5；Hunter self0/0/1、Tactical death/self0/0。八路线交付death2/sugar2/drop5/near16/mid16/far10/recovery0/full1（各16），相对父点0/−1/−3/0/0/+1/0/−2。full窗口目标12/12（episode共现）、first3/7、持续1248/2039、进度1295/5133、pickup6/16→delivery1/16、death7/self0、恢复32/35；首控37/归因7/pop36主动27/因果kill7存活7。原墙death/sugar/recovery拾包均8/8、交付均0/8，return1561/2656/1739ticks、oracle均0。open recovery已有1501合法返程机会但仍0/8交付，区分当前不可达与执行失败。

r09真实接线审计：保存g4全部8187 ready routing cache上drop0/1与原weighted结果逐值exact；home2/3与当前wall+brick物理距离逐值exact，saved-home-finite却物理INF的cell为0，收据 `SAVED_HOME_FIELD_PARITY_G4.json`。全部2328携带learner中343当前物理不可达、saved-home-finite却物理不可达0；natural full1511中167不可达、错误finite0。reset未ready行排除；scene6有1例物理finite而保存/DAgger非finite，不假设所有snapshot完全同当前时序。该结果证明信号改动接入，不证明能力改善或时序安全。

joint FIFO valid7355：navigation4723/greedy1817/mass.397030，opening39/greedy34/mass.864190，ordinary2593/greedy1549/mass.578323，return1309/greedy814/mass.615502。末状态preparing689/staged69/clear48/long+place13，限定snapshot。累计control gross10231/net9706.75、route+2464.644/−3347.468、capture132432终局主导；DAgger active1455045/selected91985/native-safe accepted82464/query7800。override后current/ring/fixed有效ticks2.2729%/5.7655%/10.6601%，底层league条件12.2734%/32.0311%/55.6955%，分母分列。

新增实际PPO归一化优势：g1..4 capped前16偏航共64行无缺失，navigation30行route负29，其中22行normalized GAE正，normalized中位.131999；return16行route负7、这7行normalized正0，normalized中位−.225649；drop14行全部normalized正且route非负；own-recovery4行route负/normalized负。raw正转normalized负当前0，不预设未来恒等。导航示例g1 device0/env795/tick0距stage1→2，route/total−.002751、extra-zero-value GAE−.013012、raw Critic GAE+.071475、normalized+.043153。此处当步无其他任务reward，正优势来自后续收益/价值基线与局部信号混合，**不能由该标量归因即时steal奖励或完整policy gradient**。收据 `NORMALIZED_NATIVE_DEVIATION_SIGNAL_G4.json`；前序cap偏差、非总体失败率、ratio/clipping/其他loss/Adam限定持续保留。

g4唯一fixed Review：drop return81左−.257496，对右+.337502，JS近似GAE−.017387804；far53左−.157499、右+.157499，GAE−.010147741。death stock2左−.241249、上+.027500；sugar stock3右−.153751、上+.021250，direct-stock与prestock stage分列。full return985下+.337502与右并列新场最佳，旧JS假阳性，GAE+1.540069仍非Q；drop1左+.341251同为新场最佳。全部原生hash/事件/动作/阶段/目标/势能在 `REVIEW_G4_PHASE_FIELD_COMPARISON.json`，不换seed。曲线 `reports/je10_r09_curve_20261018.png` 已含g0/g4并目视核验。健康r09继续固定128代，当前g20/41943040 ticks finite/nonzero；唯一CPU完成g0/g4、运行g8，无failed。r10同配方后备有效，具体新因素尚未改选。


r09 g8固定完成点（20261018，仅诊断）：Hunter难/中/易4/0/9、Tactical7（各64），路线交付death1/sugar1/drop4/near16/mid16/far10/recovery3/full2（各16），相对g4 +2/−2/−1/0/0/0/+3/+1。早期波动不构成能力结论，r09继续固定预算。g8 saved-home cache parity全ready rows逐值passed，未出现旧weighted有限而当前砖墙INF的错配。


r09唯一CX owner接管及g16完整消费（20261018）：健康run持续固定128代，不设置能力/admission/promotion门禁；g32固定评估已由现有唯一CPU/JS owner接续，不新建评估owner。g16 Hunter难/中/易0/1/10、Python Tactical15（各64），相对父点−1/−3/+3、+7；相对g8−4/+1/+1、+8；Hunter self0/0/0，Tactical death/self1/1。death/sugar/drop/near/mid/far/recovery/full交付2/0/4/16/16/7/0/2（各16），相对父点0/−3/−4/0/0/−2/0/−1，相对g8+1/−1/0/0/0/−3/−3/0。full window→target9/9（episode共现）、first5/7、持续846/1341、进度875/4033、pickup6/16→delivery2/16、恢复29/31、death8/self0；首控31/归因7/pop31主动28/因果kill7存活7。全部八漏斗机会分母及相对父/前点率差已封存POINT_G16_OWNER_REPORT和CURVE_ANALYSIS，不以episode比值之和作总体率。

保存g16全部8180 ready routing cache上drop旧weighted、home当前brick-blocked逐值exact，错误finite物理INF cell为0。joint FIFO valid16300：navigation8814/greedy3768/mass.430629，opening27/23/.827706，ordinary7459/3970/.521469，return4223/2307/.541913。累计control gross46872.75/net40489.5、route+8712.760/−19373.059、终局capture533064仍主导；DAgger query33921/candidate6216519，按阶段selected/accepted的解释见统一曲线，不把query/candidate当安全通过率。override后current/ring/fixed有效ticks1.3712%/6.2892%/11.8758%，底层league条件6.9145%/32.1816%/60.9038%。

g1..16 capped首16偏航样本256行无normalized缺失：navigation105、route负96、其中normalized GAE正65，中位+.087259；return57、route负44，其中normalized正10、中位−.295143；drop41、normalized正37；own recovery43、route负40、其中normalized正8。raw正→normalized负合计6行。该样本仅device/time顺序cap，非总体失败率；标量不是Q、完整梯度或loss冲突因果证明。固定唯一Review seed2026101800：drop返程80左−.157497、右+.267498，JS近似GAE−.017666；far103右−.177502、左+.157497但近似GAE+.008352；mid100右+.157500、上+.232500。原生hash/动作/阶段/目标/势能/事件与单步endpoint记录在REVIEW_G16_PHASE_FIELD_COMPARISON；direct-stock与prestock准备目标分列，不据静态场把全部旧偏航当真实训练目标冲突。r10同配方连续后备及其seed/预算/统一异步协议仍有效；具体因素按后续固定点证据选择，不更改健康r09数值配方。


g16补充条件梯度/覆盖证据（20261018）：同一FIFO固定最多64行的navigation DAgger与合并weak/parent保留梯度dot−.234819，DAgger norm5.736845，反向投影仅约.7135%；return dot+.493540、norm4.156952，未出现合并反向投影。opening27行反向投影约7.16%，但不是在线PPO/全loss/Adam，不能据此将KL列第一根因。固定drop return80在完整Actor输入精确匹配0、近邻几何9条；9条无标签接受录像左动作，geometry greedy5/9、mean target mass.542571。其他全局/威胁/时序并不精确匹配，既不证明历史从未见过该状态，也不证明oracle冲突。收据FIFO_CONDITIONAL_GRADIENT_ALIGNMENT_G16及FIFO_REVIEW_COVERAGE_G16保存run，不入Git。


r09 g32固定完成点（20261018，COMPLETE后）：Hunter难/中/易1/4/8、Python Tactical9（各64），相对父点0/0/+1、+1，相对g16+1/+3/−2、−6；Hunter self0/0/0，Tactical death/self0/0。death/sugar/drop/near/mid/far/recovery/full交付1/0/5/16/16/5/1/6（各16），相对父点−1/−3/−3/0/0/−4/+1/+3，相对g16−1/0/+1/0/0/−2/+1/+4。full window→target11/11（episode共现）、first5/8、持续1581/2173、进度1580/6061、pickup8/16→delivery6/16、恢复38/44、death5/self0；首控45/归因9/pop45主动36/因果kill9存活9。近/中保持16/16，但返程和恢复闭环未稳，不将单点full6视为验收。原墙death/sugar/recovery均拾8/8、交付0/8；death原墙出现277 oracle机会而未交付，sugar/recovery该点为0；开墙sugar返程有2344 oracle机会而0/8交付，开墙drop team1有2425机会而0/8。不同seat和墙体当前配对，不能当独立因果。

g32保存8188 ready cache全字段parity通过，错误home-finite物理INF为0。joint FIFO navigation4708/greedy2469/mass.519585、opening47/44/.926072、ordinary9067/4410/.476479、return5206/2526/.485951。return mass g4/16/32=.615502/.541913/.485951，navigation=.397030/.430629/.519585，FIFO分布变化、非holdout。累计control gross89795/net74137.5、route+15929.718/−39650.956、终局capture1049808仍主导；override后current/ring/fixed有效ticks1.7509%/7.7637%/12.8048%，底层league条件7.7626%/34.4666%/57.7708%。累计return active6419648/native-safe accepted249053/query18158；drop769427/254936/9399、stock4376250/211215/34992；bounded selection包含在accepted/active内，不当全几何安全率。全部normalized机会变化封存POINT_G32_OWNER_REPORT/CURVE_ANALYSIS。

g1..32 capped偏航512行：navigation189、route负167中normalized正107，中位+.068353；return147、route负130中normalized正39、中位−.262663；drop82、normalized正76；own recovery64、route负55中normalized正11；非总体率/非完整梯度。唯一Review固定game0/seed2026101800：drop返程80左−.157497对右+.267498、JS近似GAE−.018715；far51左−.157499对右+.207500、GAE−.003239；mid155右+.157500对上+.232500；drop1左+.341251为新场最佳，保留旧JS偏航假阳性。准备末状态258、staged60/clear32/long+place15，限定snapshot。

后继候选（尚未选用或安装）：只将有效合法kind3非preparation的联合DAgger监督贡献乘2，分母仍全部valid joint labels，其他阶段损失贡献不改；query、FIFO容量/新增/选择、scene mix、奖励、KL、Actor输入和PPO不改。依据返程fit下降、固定远返程/掉包/开墙恢复有合法机会但执行失败，且当前条件梯度不支持优先改KL。候选必须RED→GREEN覆盖梯度、非法/失效标签、set-valued ties、保存恢复及终态释放安装约束；不热改r09，不设训练门禁。r10同配方连续后备始终有效，后续g64/96完整证据决定是否替换；若r10保持后备，则该候选留给完整终态分析后的r11。


r09 g49附近候选准备完成（20261018）：返程DAgger贡献x2候选已完成RED→GREEN，含损失/梯度、valid/illegal、joint ties、恢复一致、终态释放安装、后继继承/封存与连续接续共22项focused通过；收据RETURN_DAGGER_CANDIDATE_TDD。候选未安装、未选入NEXT_RECIPE_PROPOSAL，r10同配方预声明保持不变；当前数值源码/plan未热改。g32限定FIFO条件梯度navigation/return/opening反向投影24.46%/23.13%/16.47%，相对g16波动明显；不是在线总gradient，不能把KL直接当确定根因。按后续固定点重复证据决定一项改动，不同时动系数、query和mix。


g32补充（20261018）：保存携带末状态2324中当前砖墙不可达487，natural-full1583中326不可达；错误saved-home-finite但物理INF均0。固定drop return80 FIFO完整Actor输入match0、近邻几何9、标签接受录像左动作0，近邻mean target probability.196107（g16 .542571，分布变化，不当同态纵向因果）。候选补上真实replay-loss接线/梯度集成后23项focused通过，只增强kind3且非preparation的有效联合返程标签，当前未安装。g52 numerical source seal及plan逐项exact，全部既有代数finite/nonzero。


r09固定半程监督（20261018）：g64/134217728 fresh ticks、64代均finite/nonzero，g64 COMMITTED及policy hash亲自核验，唯一CPU/JS evaluator已跟随固定g64；GPU健康继续后半程至g128，不等待评估/报告。r10原预声明同配方后备有效，返程DAgger x2仅为已测候选，未安装。


r09 g64固定完成点（20261018，COMPLETE后）：Hunter难/中/易1/4/9、Python Tactical11（各64），相对父点0/0/+2、+3，相对g32 0/0/+1、+2；Hunter self0/0/0、Tactical death/self0/0。death/sugar/drop/near/mid/far/recovery/full交付0/1/7/16/16/8/1/2（各16），相对父点−2/−2/−1/0/0/−1/+1/−1，相对g32−1/+1/+2/0/0/+3/0/−4。full window→target10/10（episode共现）、first1/6、持续985/1606、进度993/4008、pickup5/16→delivery2/16、恢复30/32、death10/self0；首控32/归因7/pop32主动19/因果kill7存活5。不同固定点full交付3/1/2/2/6/2，尚无稳定提升结论。

【历史解释已撤回：以下静态快照不能支持真实取包后无路或预开返程通道；仅保留原始统计，见 r12 因果初始化审计。】原墙death/sugar/recovery均pickup8/8、delivery0/8，返程oracle机会1223/1126/1070、匹配388/303/281；开墙death有2012机会仍0/8，sugar1654机会1/8、recovery1485机会1/8，不能把失败全归不可达。g64 ready routing8176行drop/home/physical逐值parity通过、错误finite物理INF0。joint FIFO valid12754，navigation3955/greedy2102/mass.527053、opening54/45/.830599、ordinary8745/4242/.477288、return4877/2405/.491073。实际selected replay current/history返程均37.5%，navigation48.3704%/46.8994%、opening1.6296%/3.1006%，包括回退和重复样本，不是独立query量。准备末状态246/staged54/clear30/long+place7。

累计control gross174702.5/net139584.25、route+30167.919/−80921.734、终局capture2125824主导；override后current/ring/fixed实际有效ticks2.2406%/8.5765%/13.8826%，底层league条件8.9443%/34.5681%/56.4876%。capped g1..64偏航1024行：navigation324、route负287中normalized GAE正165，中位+.044620；return309、route负280中normalized正91、中位−.191803；own recovery154、route负133中normalized正52。非总体失败率，不当action Q/完整梯度或单loss因果。统一八漏斗原始机会与相对父/前点率差见POINT_G64_OWNER_REPORT和CURVE_ANALYSIS。

固定唯一Review game0/seed2026101800：death return603、sugar return663、recovery return611均向下−.337502，其他合法端点上/左/右+.022499，JS近似GAE约−.004709；drop return80左−.157497对右+.217503、GAE−.021235；far51左−.157499对右+.207500、GAE−.004463；mid138上+.182500为新场最佳，保留旧JS偏航假阳性。原生hash/动作/状态/事件/阶段/目标/势能封存REVIEW_G64_PHASE_FIELD_COMPARISON。有效返程执行仍是证据支持缺口，返程DAgger x2候选已TDD待完整曲线选择；r09不提前停止，r10原连续后备不等待CPU。


g64补充根因收据（20261018）：携带末状态2387中当前brick-blocked不可达456，saved-home/native DAgger不可达同456，错误finite0；natural-full和分scene原始计数见CARRIER_TOPOLOGY_G64。限定FIFO64行条件梯度navigation/return/opening合并反向投影16.36%/22.05%/26.34%，非在线完整梯度。固定drop return80完整Actor输入match0、近邻几何13、无标签接受录像左动作、目标mass.578488，跨时点分布变化且录像仍偏离，说明FIFO拟合不能替代完整路线验收。唯一Review所有返程fork替代动作均legal/alive、拓扑stable，但该单步检查不证明未来安全。已在g80再次核验全部numerical seal exact，r09继续固定预算、无能力门禁。


r09四分之三固定预算监督（20261018）：g96/201326592 fresh ticks，96代均finite/nonzero，g96 COMMITTED/8副本exact/policy hash亲自核验，唯一CPU/JS evaluator已跟随g96；GPU继续最后32代。r10同配方预声明维持，无能力门禁，无CPU/报告等待；下一最小因素按完整r09曲线在r10有效接续后最终选择。


r09 g96固定完成点（20261018，COMPLETE后）：Hunter难/中/易1/3/7、Python Tactical7（各64），相对父点0/−1/0、−1，相对g64 0/−1/−2、−4；Hunter self0/0/0，Tactical death/self0/0。death/sugar/drop/near/mid/far/recovery/full交付1/1/7/16/15/9/0/1（各16），相对父点−1/−2/−1/0/−1/0/0/−2，相对g64+1/0/0/0/−1/+1/−1/−1。full window→target10/10（episode共现）、first5/7、持续1106/2128、进度1152/5608、pickup6/16→delivery1/16、恢复41/43、death5/self0；首控43/归因10/pop43主动38/因果kill10存活9。完整固定曲线与机会归一化父/前点delta封存POINT_G96_OWNER_REPORT/CURVE_ANALYSIS，不用中间低分停止健康run。

g96 routing8187 ready行drop/home/physical parity全通过，错误finite物理INF0。joint FIFO valid12909：navigation3384/greedy1936/mass.564387、opening42/39/.889238、ordinary9483/4482/.465618、return5678/2612/.463794；返程fit未随准备导航fit同步改善。累计control gross257368/net203260、route+44131.957/−122614.496、终局capture3169776仍主导；override后current/ring/fixed有效ticks2.4539%/8.8962%/14.0939%，底层league条件9.5076%/34.8640%/55.6283%。capped g1..96共1536行，navigation442、route负370中normalized正217、中位+.048201；return517、route负480中normalized正151、中位−.218155；own recovery199、负170中normalized正74。限定cap样本/非action Q/非完整梯度。

唯一Review固定game0/seed2026101800：drop返程80左−.157497对右+.267498，JS近似GAE−.017814；far89左−.202499对右+.177498，GAE−.003427；mid92上+.172500为新场最佳，保留旧JS偏航假阳性。恢复stock4右−.341250、上+.411250，但direct-stock与prestock准备目标需分列；不是exact训练oracle冲突。r09在g107已224395264 fresh ticks、全部finite/nonzero，继续g128。r10仍按原预声明同配方立即接续；完整终态曲线后再最终选择已测返程DAgger x2或更有证据的一项后继因素，不改动健康r09/r10运行中数值。


g96补充（20261018）：携带末状态2381中当前brick-blocked不可达448，saved-home/native DAgger不可达同448，错误finite0。限定FIFO条件梯度navigation/return/opening反向投影21.28%/11.44%/0，随点波动，未构成优先改KL的充分证据。准备末状态226/staged57/clear29/long+place6。return FIFO mass从g4 .615502到g16 .541913、g32 .485951、g64 .491073、g96 .463794，固定掉包/远返程/恢复闭环仍不稳定；返程DAgger x2候选收据已补全这些证据，仍未安装。r09 g109健康继续128；不因完整局g96低点停训。


r09固定终态及r10技术恢复（20261018）：r09正常完成128代/268435456 fresh ticks，128代finite/nonzero；compute5626.557s、wall6536.050s、atomic_stop=false。8卡释放收据devices0..7/compute_clients0亲自核验后立即启动预声明r10，8卡smoke先于最终评估/报告。r09 final完整state/fresh CPU恢复passed，g128唯一CPU/JS评估独立运行。r10 smoke01在首更新后host诊断封存前因重复collector_gae_advantage kwargs技术失败，formal更新0、无能力结论；旧phase telemetry安装器对已normalized的源码非幂等。新增组合安装RED复现，修复后11项focused通过，仅host字段与安装器组合幂等修复；plan、父点、seed、PPO/reward/DAgger数值均未改。失败smoke01完整保留，TECHNICAL_TELEMETRY_RECOVERY记录三份host/helper源码hash变更，technical_smoke02重新smoke→保存恢复→正式训练，恢复owner已启动。未把失败技术smoke当能力门禁；后继正式g1非零仍需亲自确认。


r09固定终态及有效r10交接（20261018）：r09完整128代/268435456 fresh ticks，128次finite/nonzero，compute5626.557s/wall6536.050s、atomic_stop=false。8卡释放compute_clients0后已立即启动预声明r10。其smoke01因继承phase/normalized telemetry重复kwargs技术失败、formal更新0，旧产物保留；组合安装RED复现，host-only修复后11项focused通过。smoke02 Actor/Critic L2=3.352034/2.242606、finite/nonzero、save/fresh CPU restore passed；formal g1 L2=3.233818/2.290936，8副本exact、finite/nonzero、2,097,152 fresh ticks，COMMITTED/policy hash亲自核验。父final raw Actor/Critic/完整pool分别75/75/75叶value/dtype/shape exact；父153份保存source、r10当前/保存153份source与plan seal exact。r10固定128代健康训练继续，CPU/JS独立跟随，不等待能力门槛。技术恢复未改变数值配方、种子或父点。

r09完整曲线g0/4/8/16/32/64/96/128（每桶16局）：

| 路线交付 | g0 | g4 | g8 | g16 | g32 | g64 | g96 | g128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| death | 2 | 2 | 1 | 2 | 1 | 0 | 1 | 3 |
| sugar | 3 | 2 | 1 | 0 | 0 | 1 | 1 | 1 |
| drop | 8 | 5 | 4 | 4 | 5 | 7 | 7 | 5 |
| near | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| mid | 16 | 16 | 16 | 16 | 16 | 16 | 15 | 16 |
| far | 9 | 10 | 10 | 7 | 5 | 8 | 9 | 10 |
| recovery | 0 | 0 | 3 | 0 | 1 | 1 | 0 | 1 |
| natural full | 3 | 1 | 2 | 2 | 6 | 2 | 1 | 1 |

最终JS Hunter难/中/易0/3/9、Python Tactical11（各64），相对父点−1/−1/+2、+3，相对g96−1/0/+2、+4；Hunter self1/0/0、Tactical death/self1/0。八路线最终相对父点交付+1/−2/−3/0/0/+1/+1/−2，相对g96+2/0/−2/0/+1/+1/+1/0。full window→target9/9（episode共现）、first5/7、持续1046/1864、进度1106/4882、pickup7/16→delivery1/16、恢复31/33、death3/self0；首控35/归因4/pop34主动29/因果kill4/存活4。末点原墙death/sugar/recovery拾包均8/8、交付1/0/1（各8），返程oracle机会395/400/396；开墙death/sugar/recovery返程机会2219/3255/3499、交付2/1/0（各8）。字段修复进入真实训练，未形成跨场景稳定闭环。全部固定机会归一化父/前点变化、stage/seat/墙体分层与实际暴露在CURVE_ANALYSIS/POINT_G128_OWNER_REPORT，曲线图reports/je10_r09_curve_20261018.png已目视核验。

证据排序①可执行返程行动尚未学稳：掉包拾取16/16，交付4-8/16；开墙恢复/库存返程有大量oracle机会仍缺交付。return FIFO目标set mass g4/16/32/64/96/128=.615502/.541913/.485951/.491073/.463794/.410324，末点greedy2738/6820；preparation navigation末点mass.584062。current/history实际return replay均37.5%，navigation48.4436%/46.8994%、opening1.5564%/3.1006%，包含回退/重复，非独立query量。不存在“返程完全没有被抽到”的证据。末点valid joint13879，exact Actor输入重复组11、无公共joint pair冲突0组，但不证明全分布无冲突。

证据排序②优势与目标混合：g1..128 capped首16偏航共2048行，return741中route负694，其中normalized GAE正228，中位−.189766；navigation546中route负451/其中normalized正264，中位+.052441；own recovery272中负237/其中normalized正103。不是总体失败率、Q或完整gradient。条件FIFO梯度navigation/return/opening反向投影9.87%/17.13%/16.77%，随点变化，不足以直接将KL当首因。唯一Review固定game0/seed2026101800：drop return80左−.157497，对右+.267498、JS近似GAE−.018484；far107右−.202499，对上/左+.202501、近似GAE+.000784。drop1左+.341251为新场最佳，保留旧JS偏航假阳性；direct-stock与prestock目标分列。原生hash/动作/状态/事件/阶段/目标/势能在REVIEW_G128_PHASE_FIELD_COMPARISON。drop return80 checkpoint左概率.976304，exact Actor输入匹配0、近邻几何12、无标签接受该左动作、近邻mass.329522；威胁/时序并不精确匹配，不能称精确标签冲突。

【历史解释已撤回：以下静态快照不能支持真实取包后无路或预开返程通道；仅保留原始统计，见 r12 因果初始化审计。】证据排序③残余拓扑/开路：最终携带末状态2513中415当前物理brick-blocked不可达、native DAgger及saved-home不可达417（2行缓存/状态时序差异），natural-full1707中265不可达；错误saved-home finite但物理INF为0。全部8183 ready routing cache的drop/home/physical逐值parity passed。字段错配已修复，仍不能保证携带时未来可通或时间安全。准备末状态262/staged60/clear24/long+place6，限定snapshot，不能当全部轨迹覆盖。

证据排序④信号可达但闭环与战斗保持不足：累计control gross340870.75/net265884，route+58063.822/−165028.060，capture4171800仍主导；native-full事件window175149、kill111820、pickup4780、steal111201、capture79553，按每百万full tick window894.164/capture406.131（JAX bot，只作训练计数，严格分列JS/Python）。DAgger active42847632、selected3139920、native-safe/progress accepted2875266/query219062；return active25924371/accepted1009083/query84583。accepted/active含bounded selection，非全几何安全率。override后current/ring/fixed实际有效ticks2.4953%/8.8073%/14.4991%，底层league条件9.5589%/34.1191%/56.3220%；其他rule/弱课程/assist单列。cross-map仅Bun806，泛化未证明，目标尚未达成。

下一最小可验证改动已选定并预声明r11：JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r11，父r10 final g128 raw Actor/Critic+完整pool；train3961118378、technical1518516916，fresh Adam/RNG/env/FIFO/EMA-to-raw/counters，固定8卡128代/268435456 ticks、mix .35 combat/.20 bridge/.35 full/.10 other native routes。只把valid/accepted/legal/ordinary kind3 carrier-return联合DAgger贡献乘2（每返程行.2→.4），分母仍全部valid joint labels，其他阶段贡献、query、FIFO容量/新增/选择、mix、奖励、PPO/GAE、KL与Actor输入不改。RED→GREEN 23项focused覆盖损失/梯度、无return exact、非法/失效标签、joint ties、恢复、实际replay接线、终态释放安装、封存/继承。r10 NEXT_RECIPE_PROPOSAL/SUCCESSOR_PREDECLARATION已替换，原同配方后备保留。顺序继承无同父对照，不声称单因素能力归因。

r11仍每4代恢复点，唯一全局CPU/JS owner消费g0/4/8/16/32/64/96/128 COMPLETE，统一Tactical及JS Hunter难/中/易各64局、八路线各16局、固定Review/seed及全部漏斗；GPU不等评估、阈值、报告或人工批准。中间结果只诊断/准备后继，健康run完成固定预算，终态先核验8卡释放并启动预声明后继，再作全曲线最终综合。整体实验继续，不标为能力完成。

r10 owner恢复监督（20261018）：唯一CPU/JS评估曾因把已恢复smoke01的历史`TECHNICAL_FAILURE.json`当作当前worker故障而退出，队列未消费。保留原失败marker，改为仅当失败marker晚于`FORMAL_NONZERO_CONFIRMED.json`时阻塞；无恢复收据仍阻塞，恢复后新技术失败也重新阻塞。新增3项focused测试覆盖三种时间序关系并通过。r10 evaluator已作为CPU-only唯一owner重启，队列从首个未完成点g0开始，`OWNER_RESUMED.json`标记恢复且旧marker保留。恢复时formal训练已完成g4，finite/nonzero、8卡持续活跃；Tactical v2、Hunter与路线漏斗随g0异步运行，无训练门禁。继续跟随g0/4/8/16/32/64/96/128及128代固定run终态。


r10评估恢复补充（20261018）：恢复判定进一步验证正式非零收据内容（finite、8副本、g1/fresh ticks、Actor/Critic八卡有限正更新），坏JSON/空收据/非finite/零更新均不能解除故障。新增有效性用例先RED四个失败，再GREEN，5项focused通过，日志`EVALUATOR_RECOVERY_RED.log`与`EVALUATOR_RECOVERY_GREEN.log`保留在run。首轮时间序修复是在测试前实现，未冒称该部分先RED；本次收据有效性遵循RED→GREEN。旧marker未删除。监督时formal g14/29,360,128 fresh ticks，全部finite/nonzero；g0路线前六任务产出，固定COMPLETE前不消费局部结果作结论。r11预声明两份实现/两份测试hash一致，尚未安装，GPU与评估独立。


r10 g0固定完成点（20261018）：唯一CPU/JS owner已写COMPLETE并继续g4；Hunter难/中/易0/3/9、Python Tactical11（各64），八路线death/sugar/drop/near/mid/far/recovery/full交付3/1/5/16/16/10/1/1（各16）。与r09最终点Hunter、Tactical、八路线全部原始结果与控制漏斗逐值一致，行为基线复现；序列化policy文件hash不同，不能据此声称二进制相同。full window→target9/9（episode共现）、first5/7、持续1046/1864、进度1106/4882、pickup7/16→delivery1/16、恢复31/33；首控35/归因4/pop34主动29/因果kill4/存活4；death3/self0。全部机会分母及局限写CURVE_ANALYSIS。r10 formal g17/35,651,584 fresh ticks，finite/nonzero，健康固定128代继续。

r10 g4固定完成点（20261018，诊断不设门禁）：唯一CPU/JS owner已消费g0、g4并转入g8，队列无失败；Hunter难/中/易2/2/6、Python Tactical11（各64），self均0。八路线oracle机会/匹配/拾包/交付：death 3760/1865/15/2，sugar 3540/1663/15/1，drop 7057/3934/16/3，near 208/204/0/16，mid 1074/947/0/16，far 4185/2234/0/6，recovery 4180/1796/14/1，natural full 1750/1061/7/2（各16局）。恢复威胁切换18/11，死亡3/0。相对r10 g0的完整变化、机会分母、阶段目标和固定Review首偏航见`authority_je10/g000004/DIAGNOSIS.json`与`CURVE_ANALYSIS.json`；g4仅诊断。

g4累计训练信号：control entry/residual/clawback=9974.25/283.25/-715.75；route shaped=-939.046，positive/negative=2426.542/-3365.588；transport capture=131544，combined_total=191068.531。DAgger candidate/query=1488766/6682（0.4488%），bounded query覆盖不能直接解释安全通过率；route净为负且终局搬运仍主导，继续观察后续点。r10 formal g28/58720256 fresh ticks、finite/nonzero，健康固定128代继续。


r10 g8固定完成点（20261018）：Hunter难/中/易1/2/8、Python Tactical9（各64），相对g0+1/-1/-1、-2，相对g4-1/0/+2、-2；八路线交付1/1/4/16/16/6/0/1（各16），相对g0-2/0/-1/0/0/-4/-1/0，相对g4-1/0/+1/0/0/0/-1/-1。full window→target14/14（episode共现）、first3/5、持续892/1721、进度956/6961、pickup10/16→delivery1/16、恢复34/38；首控39/归因9/pop39主动30/因果kill9存活9，death5/self0。全部八漏斗机会归一化父/前点delta与分层见POINT_G8_OWNER_REPORT及CURVE_ANALYSIS。CPU队列已继续g16，formal g36/75497472 fresh ticks健康完成，固定128代不改。

g4 learner-state诊断补充：FIFO valid6557，return1690/greedy970/目标joint mass.581909，navigation3511/greedy1523/mass.442262；current/history返程replay都37.5%，含重复回退，不是独立query比例。g1..4 capped首偏航64行：return17中route负12/其中normalized GAE正2、中位-.510740；navigation27中route负26/其中normalized正17、中位+.158915；drop20中route负0、normalized中位+3.249048。限定cap样本，不是总体失败率、Q或完整loss梯度。

r10 g8只读根因补充（20261018）：固定Review game0首返程偏航保持原seed；real_drop tick82连续候选方向[-.227501,+.215000,-.157497,+.267498,0]，JS近似GAE-.018110；carry_far tick51[-.127499,+.337502,-.082497,+.277504,0]，近似GAE-.005098。saved-home字段与当前brick-blocked拓扑8180 ready rows逐值parity通过，mismatch=0、physical INF与saved finite错配=0；这是实现一致性，不是策略能力。g8 capped normalized优势：return44行route负39/其中normalized正9、中位-.075900；navigation41行route负36/其中normalized正22、中位+.138231；own_recovery11行负10/其中正7。g8 FIFO条件梯度诊断（固定最多64行、非在线PPO全梯度）：return DAgger vs weak-retention dot=-2.595868/cos=-.655727，navigation dot=-4.105622/cos=-.628675，opening dot=-.640739/cos=-.091544；仅说明候选方向在固定FIFO上与保留项方向相反，不能直接归因训练loss。返程实际replay current/history仍各37.5%。这些证据继续支持r11只乘valid carrier-return DAgger贡献2，保留战斗/其他阶段不变。

r10 g16固定完成点（20261018）：唯一CPU/JS owner已完成g16并转入g32，无failed；Hunter难/中/易0/2/13、Tactical15（各64），Hunter self0/0/0、Tactical death/self0/0。八路线交付death/sugar/drop/near/mid/far/recovery/full=1/0/3/16/15/5/0/1（各16）；full window→target8/8、first3/5、持续639/1179、进度654/4339、pickup4/16→delivery1/16、恢复31/33；首控34/归因9/pop33主动27/因果kill9存活9。相对g8 Hunter -1/0/+5、Tactical+6，路线drop -1、mid -1、far -1、recovery不变；短点只诊断不停止。

g16 FIFO valid15060，return kind3=6431/greedy2725/target mass.430875，navigation5325/greedy2609/mass.486787；current/history返程selected占比仍37.5%。capped首偏航return94中route负86/normalized正26/中位-.129505，navigation61中route负51/normalized正28/中位+.0529587，own recovery44中负41/normalized正12；固定漏斗和实际对手暴露current/ring/fixed=1.0643%/6.9345%/11.9012%。这些是机会归一化和有限cap诊断，不是能力门禁；r10 formal g46/96468992 fresh ticks时已finite/nonzero，继续固定128代。

r10 g32固定完成点（20261018）：唯一CPU/JS owner已完成g32并继续g64；Hunter难/中/易2/5/6、Tactical10（各64），无新failed。八路线交付death/sugar/drop/near/mid/far/recovery/full=0/1/4/16/15/7/0/2（各16）；natural full window→target9/9、first7/7、持续1072/1586、进度1118/3604、pickup6/16→delivery2/16、恢复14/19；首控20/归因6/pop20主动17/因果kill6存活6、death5/self0。相对g16 Hunter+2/+3/-7、Tactical-5，路线drop+1/far+2/full+1；短点波动不停止健康run。

g32 capped优势诊断：return213中route负204/normalized正72、中位-.167876；navigation117中负97/normalized正61、中位+.092884；own recovery83中负78/normalized正31；drop76中route负4且normalized正4。实际对手暴露override后current/ring/fixed=1.5836%/8.1599%/12.6596%。自然full交付由g16的1/16升至2/16，但仍不足以改变预声明后继；r10 formal g55/115343360 fresh ticks，finite/nonzero。

r10 g64固定完成点（20261018）：唯一CPU/JS owner完成g64并等待g96；Hunter难/中/易1/5/4、Tactical14（各64），Hunter self0/0/0、Tactical death/self0/0。八路线交付death/sugar/drop/near/mid/far/recovery/full=1/1/4/16/16/9/0/1；natural full window→target12/12、first5/10、持续1182/2321、进度1197/5941、pickup3/16→delivery1/16、恢复40/42；首控44/归因12/pop43主动31/因果kill11存活11、native causal1、death4/self0。相对g32 Hunter-1/0/-2、Tactical+4；far+2，其他主要路线持平/波动。

g64 capped优势：return449中route负424/normalized正161、中位-.110931；navigation195中负157/normalized正97、中位+.112332；own recovery169中负155/normalized正63；drop157中route负8。实际对手暴露override后current/ring/fixed=2.0059%/9.1983%/13.4950%。自然full交付仍1/16，继续保留r11返程x2候选；r10 formal g74/155189248 fresh ticks、finite/nonzero。


【历史解释已撤回：以下携包末状态分母含无事件预置训练初态；其静态INF不代表真实取包后无路，亦不支持动态封路或预开返程通道。原始计数仅作污染证据保留。】r10 g64固定Review/拓扑补充（20261018）：原seed game0 drop return80左gain-.157497（右+.267498）、JS近似GAE-.018309；far51左-.082497（下+.337502/右+.277504）、近似GAE-.005185；mid117右+.202500为新连续场最佳，保留旧oracle偏航假阳性，不误算训练错误。全部动作/state hash/事件/目标/phase/势能存REVIEW_G64_PHASE_FIELD_COMPARISON。8184 ready缓存drop/home/physical逐值parity通过；末状态carrier2221中physical不可达397、saved-home/native不可达398（时序差异1），错误saved finite而物理INF为0。静态当前连通不等于未来安全通路。固定FIFO条件gradient反向投影navigation/return/opening为5.156%/2.917%/15.994%，仅最多64行非在线PPO全梯度，不据此将KL视作首因。

r10 g96固定完成点（20261018）：唯一CPU/JS owner完成g96并等待最终g128；Hunter难/中/易1/2/7、Tactical9（各64），Hunter self0/0/1、Tactical death/self0/0。八路线交付death/sugar/drop/near/mid/far/recovery/full=1/0/3/16/16/6/0/3（各16）；natural full window→target8/8、first5/6、持续937/1405、进度972/3672、pickup5/16→delivery3/16、恢复23/26；首控27/归因8/pop27主动19/因果kill8存活7、death6/self0。相对g64 full交付+2、pickup+2，Hunter/Tactical短点波动不设门禁。

g96 capped优势：return645中route负612/normalized正236、中位-.114276；navigation287中负228/normalized正143、中位+.096696；own recovery266中负236/normalized正90；drop238中route负12。实际对手暴露override后current/ring/fixed=2.0452%/9.6348%/13.7955%。固定g128仍照常完成，r11只在8卡释放后安装。


r10安全计数校核（20261018）：前述人工摘要将JS natural-full死亡数误抄入Python Tactical列，已据权威SUMMARY更正g16/g64/g96 Tactical death/self均0/0，g96 Hunter self为0/0/1；不同simulator严格分列。SOURCE_SEAL153份训练源码与plan、r11候选4份源码hash均核对一致。r10 return FIFO mass g4/8/16/32/64/96=.581909/.471964/.430875/.408307/.441119/.459589；navigation .442262/.516114/.486787/.569325/.601287/.575663，返程相对弱。r11预声明仍为单因素贡献x2，当前r10训练不变。


r10固定128代终态（20261018）：128次finite/nonzero逐代核验，268435456 fresh ticks、atomic_stop=false；compute5769.818s、wall6628.497s。GPU_RELEASE_RECEIPT已核验devices0..7、compute_clients0；随后预声明r11 supervisor启动并正在8卡technical_smoke01。r10 g128唯一CPU/JS评估同时独立运行，GPU未等待最终评估或报告。r11正式g1非零尚待确认，不把smoke当能力门禁。

固定run `runs/je10_bun_20261018/control_route_v38_r10` 完成128代/268,435,456 ticks；compute=5769.818s、wall=6628.497s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r11`；本次后继配方选择依据 `r09 complete g0/4/8/16/32/64/96/128: full delivery3/1/2/2/6/2/1/1; drop8/5/4/4/5/7/7/5; near stays16, far9/10/10/7/5/8/9/10. Verified return FIFO target mass .615502/.541913/.485951/.491073/.463794/.410324 versus preparation navigation .397030/.430629/.519585/.527053/.564387/.584062; replay return actual37.5%. Fixed Review drop return80 wrong left has negative route gain, no exact-label conflict established. Strengthen only valid native carrier return DAgger contribution before changing query, mix, rewards, or KL.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。

r10固定终态综合（20261018）：g128完成后唯一CPU/JS评估COMPLETE，r10全曲线g0/4/8/16/32/64/96/128均消费；g128 Hunter难/中/易3/1/12、Python Tactical10（各64），Hunter self0/0/1、Tactical death/self0/0。八路线交付1/1/10/16/16/11/0/0（各16），相对父点death/sugar/drop/near/mid/far/recovery/full=0/0/+5/0/+1/−1/−1/−2，相对g96=0/+1/+7/0/0/+5/0/−3；natural full window→target11/11、first3/4、持续349/530、进度359/5062、pickup2/16→delivery0/16、恢复40/41；首控43/归因9/pop42主动32/因果kill9存活9、death7/self0。g128 Tactical/Hunter短点保持指标不构成门禁，route交付未稳定改善。

终态训练累计control gross337616.25/net260684.5，route positive56502.886/negative−166481.688/net−109978.803，终局capture4060392仍主导；native full每百万window877.526、kill560.006、pickup25.267、capture392.609、transfer69.786。DAgger verifier eligible42918806、bounded selected3139511、native-safe/progress accepted2849515、query209637、return accepted1011942/query95556；accepted含bounded selection，非全几何安全率。raw sampled league current/ring/fixed=8.5520%/36.9554%/54.4927%，override后实际有效ticks由CURVE_ANALYSIS单列。g128 FIFO valid11901，return kind3=5494/greedy2672/target mass.485468，navigation2588/greedy1645/mass.621021，exact duplicate groups9、公共joint pair冲突0。

根因证据排序：①返程闭环仍是首要失败，g128 natural full pickup2/16→delivery0/16，return capped route负777/832且normalized正306、中位−.105922；actual return replay占37.5%，并非零抽样。②返程监督与弱保留方向固定FIFO上反向，但仅为最多64行条件梯度，不是完整PPO因果。③拓扑字段实现已parity通过，g128 carrier末状态与当前brick-blocked不可达仍需按终态诊断读取，不能把静态INF直接当策略失败。④combat保持可接受但Hunter normal下降、easy升，需r11监测防遗忘。最小后继改动保持预声明r11：仅valid/accepted/legal/ordinary kind3 carrier-return联合DAgger贡献x2，其他阶段、query、FIFO、奖励、PPO/GAE、KL、mix和Actor输入不变；顺序继承无同父对照，不宣称单因素能力因果。

r10终态先核验GPU_RELEASE_RECEIPT devices0..7/compute_clients0，再启动r11；r11 smoke、fresh CPU restore和正式g1已通过，8副本exact，Actor/Critic L2=3.312004/2.316223，2,097,152 fresh ticks，policy hash匹配。r11固定128代与唯一CPU/JS评估已运行，g0队列开始消费；无能力/admission/promotion门禁。

r11交接后早期健康（20261018）：正式worker g4/8388608 fresh ticks，全部finite/nonzero；r11唯一CPU/JS owner已启动g0路线任务，当前无FAILED marker；8卡利用率约79–91%。r11固定128代和预声明g0/4/8/16/32/64/96/128评估继续，不因r10终态报告或r11短点能力停训。


### JE10 r11 排除与 r12 因果初始化修复（2026-10-18）

**撤回旧“真实取包后无路／动态封路／需要预开返程通道”根因结论**。实际进入敌笼取包存在其入站可逆通路，这是本任务规则不变量。旧 CARRIER_TOPOLOGY 读取 rollout-end 的 `live.states`，按 `bun_carried==enemy` 聚合，未检查真实取包事件、入站轨迹或 custody provenance；静态 brick-blocked BFS 的 INF 不能用来推断真实取包后的不可达。r10 g128 原分母 2,267 carrier snapshot（其中退休 scene2/3/6=33/14/3），389 INF；scene8 的 1,446/238 也只是课程末状态标签、不是经真实取包链验证的网页自然局。这些计数仅保留为历史实现输出，全部退出“真实取包后无路”的推断分母。旧账本中所有相同推断与“预开返程”建议均由本条明确撤回，不再作为候选根因。

具体缺陷：`je05/je07/je09_bun_scenes.init_batch` 曾在 scene2/3/4/6 直接写 learner `bun_carried`；旧低层 carry_home/carry_return lesson 也直接赋携包；`je10_route_oracle.labels` 又以 scene2..4 替代事件 custody。权重入口仍可采这些 ID。现删除 learner 携包赋值，所有课程及旧强制 course 排除这些 ID（全无效权重回退空手 full），旧 lesson1/11 同样排除；不改名保留携包。返程 oracle 仅认真实 native acquisition custody。每代另审计退休 scene tick、无 custody 的 active return，统计与 DAgger 阶段分类保持分列。

r11 按用户指令在 g18 原子停止（37,748,736 fresh ticks、atomic_stop=true），`EXCLUDED_INVALID.json` 保存；全部 r11 权重与能力曲线 **EXCLUDED/INVALID**，不作能力结论。其实际有效训练 scene2/3/4/6 tick=209560/80773/20438/30490，合计341261；这是污染场景 tick 分母，不是初始化次数或有携包监督行数。checkpoint、events、评估与旧 smoke 证据保留；旧 trainer/evaluator 已停止，重新接管时八卡 compute clients=0。不重启两个旧 r12 namespace 的任何 smoke。

固定 replay 并非全部无效：现有 hash `daf2aac92097d4b0e259441fb9470bd0a5e22ad521b67c83683dce6a5bc518c4` 对应 Stage E `validation/RETURN_REPLAY.json`，short1062/long996，共2058行。生成器 learner 初始空手，native drop/pickup、nonce/credit/chase 对齐门控后采样；保留这份有事件证明的监督。所有固定 replay 训练读取入口均须校验 hash、native proof、无 takeover/future information 和各组行数；缺证明或不匹配直接拒绝，禁止把旧无证明携包监督混入。新 FIFO 从空创建。

新实验 `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r12-clean-causal-init`，namespace `runs/je10_bun_20261018/control_route_v38_r12_causal_clean`。按用户指定从 r10 g128 raw Actor/Critic+full pool 回退；不从排除的 r11 继承。train seed `467682012`，technical seed `194572106`，SHA256(identity:role)前4字节大端。保留合法普通返程联合 DAgger 贡献×2，其他相位贡献、all-valid denominator 不改；叠加本次因果初始化删除，属于整合修复配方，不声称单变量能力因果。fresh Adam、EMA-to-raw、env/RNG、FIFO、计数；全新 run/state/cache/eval，GPU cache 不链接旧目录。预算8×1024×256、128代/268435456 ticks；reset mix .35 combat/.20真实drop bridge/.35自然full/.10合法空手路线，退休ID质量0，其余路线仅1/5/7重新归一化。所有真实 stock、loose pickup、native contact transfer 保留。

无能力/admission/promotion门禁。focused TDD、导入、finite/NaN/OOM、Actor/Critic真实非零及checkpoint fresh CPU恢复之后立即正式长训，不等网页、报告或人工批准。每4代原子恢复点，唯一独立CPU/JS owner消费g0/4/8/16/32/64/96/128，完成marker `authority_je10/gXXXXXX/COMPLETE.json`；全局evaluation.lock串行、GPU不等CPU。固定Tactical v2 64、production JS Hunter难/中/易各64、八路线各16固定seed、唯一Review首失败seed。carry near/mid/far人工预置题明确为诊断，绝不计训练或推断真实取包后不可达；自然完整局与控制fixture分列。每点完整漏斗、父/前点delta、首次偏航/奖励优势/DAgger/FIFO、战斗安全、current/ring/fixed实际暴露继续记录。

预声明后继 r13_causal_clean：父r12最终g128、同固定128代与配方、身份派生新seed、独立状态；终态先核验8卡释放并启动后继，再最终报告。中间评估只诊断和准备最小证据改动，不中止健康run。完整曲线→机会归一化信号→首偏航→oracle覆盖冲突→DAgger/FIFO→阶段势能→保持/对手暴露排序根因；最终不佳继续最小可验证迭代。唯一CX owner继续监督。

r12修复验证补充：focused58项、相关回归123项、接线回归30项全部GREEN（集合有重叠，不累加冒称独立用例）。原RED/中间失败日志保留；旧测试改用明确未安装配方fixture或当前已安装配方预期，非放宽原生取包约束。新8卡technical_smoke01已启动，未复用旧smoke。g0完整快照核验8192 learner空手、退休scene0、FIFO空、Adam0、Actor/Critic逐值等于r10 raw输入；cache独立。`CAUSAL_FRESH_INITIALIZATION.json`与`CAUSAL_TDD_GREEN.json`保留run中。正式训练不等待独立CPU集成/网页/报告。

独立真实collector集成用例已单独GREEN：实际native collect→GAE→联合FIFO→PPO的Actor/Critic均非零，完整aux/FIFO保存恢复通过；仅技术正确性，不作能力门禁。

r12正式首代非零已确认：g1/2097152 fresh ticks，finite，8副本exact，Actor/Critic L2=3.250805855/2.309925318，policy SHA256 `c69cec4dbc1a8dece1545ceb3c0382583c9acde33159d744154c1facdee0901a`。新smoke保存/fresh CPU恢复passed；正式g0 raw Actor/Critic与完整pool逐值等于r10输入、Adam0/FIFO空、8192 learner空手，schema clean-causal-init-v1。g1退休scene tick=0、无真实custody的active return=0。CPU/JS唯一owner正在g0、无failed。正式8卡固定128代继续，r13_causal_clean预算/seed/后继已预声明，任务尚未完成。

CPU诊断口径修正（不改运行中的训练源码/plan/数值配方）：旧自动报告将退休carry场景零采样列为缺陷，且把所有scene事件和除以full tick的比率误标native-full。现采样诊断只检查合法空手scene1/5/7；native-full事件按各opponent role中已由scene==8筛选的八项native_transitions求和，固定carry题明确diagnostic-only且不证明取包后不可达。2项RED→8项GREEN；EVALUATION_SEAL旧版保留、单独记录CPU诊断版本，训练SOURCE_SEAL全部160项仍匹配。旧报告所有使用all-scene事件/full-tick分母的native-full率均撤回，后续曲线按正确分母重算；原始训练计数不改写。

r12 g0固定点已COMPLETE并消费：JS Hunter难/中/易3/1/12、Python Tactical10（各64），安全原始计数与r10 g128逐值一致，父行为基线复现。八路线交付death/sugar/drop/near/mid/far/recovery/full=1/1/10/16/16/11/0/0（各16）；其中near/mid/far均是明确人工携包诊断题，不能作真实取包链或训练样本证据。其余受控库存fixture与真实natural_full分列。固定漏斗、机会分母、seed0首次偏航与父点基线见CURVE_ANALYSIS及权威SUMMARY；g0诊断已用修正后的分类与分母，未推荐恢复退休scene。CPU继续g4，formal已健康g4，训练不等评估。

r12 g4固定点消费：JS Hunter难/中/易0/1/6、Python Tactical6（各64），相对g0为−3/0/−6、−4；Hunter death0/2/0/self0/0/0，Tactical death/self0/0。八路线交付death/sugar/drop/near/mid/far/recovery/full=1/0/7/16/15/12/2/2（各16）。自然full窗口→目标10/10为episode共现、首oracle5/7、持续1098/1882、推进1164/4414、拾包5/16→交付2/16、恢复30/32；首控34/归因10/pop33主动26/因果击杀10存活10，death6/self0。完整机会归一化父/前点delta、分层、首次偏航与实际对手暴露见CURVE_ANALYSIS/POINT_G4_OWNER_REPORT。短点只诊断，不作门禁或稳定提升结论。

正确natural-full训练事件为window5527/kill2159/stock2033/pickup2/capture607/transfer45，使用scene8 role-filtered口径，与所有scene和JS/Python严格分列。累计control gross9151.75/net8631.75，route+1960.116/−2813.832/net−853.716，capture108672仍主导。DAgger return active611118/accepted29659/query1308；accepted含bounded selection，非几何安全率。FIFO valid5063，return1309/greedy735/target-set mass.575787，navigation2492/mass.496644；精确输入重复组7、无公共joint pair冲突0。current/history返程replay各96/256=37.5%，含重复，返程×2实际已接线。override后current/ring/fixed实际有效tick占比.51696%/12.70485%/5.76873%。

前4代固定首16偏航cap按普通return/stock/drop/own recovery与preparation navigation/opening分列奖励及真实normalized GAE，见POINT_G4_OWNER_REPORT；不是总体失败率、Q或完整梯度。唯一Review固定seed2026101800的drop首返程tick80为左、oracle下；完整动作/hash/事件/相位/目标/势能与近似优势保留原生Review和trace。继续跟踪完整128代，不因早期战斗或路线波动停训。

r12 g8固定点消费：JS Hunter难/中/易0/0/6、Python Tactical10（各64）；相对g4 Hunter0/−1/0、Tactical+4，相对父点−3/−1/−6、Tactical0。八路线交付1/1/4/16/16/10/0/3（各16，near/mid/far仅人工携包诊断）。natural full窗口→目标11/11（episode共现）、首oracle6/8、持续1231/2167、推进1410/4773、拾包6/16→交付3/16、恢复27/33；首控33/归因15/pop33主动23/因果击杀15存活15，death8/self0。掉包闭环与Hunter保持回落，尚非稳定提升；固定128代继续。

g8 FIFO valid11460，return3679/greedy2053/target mass.557845（g4 .575787），navigation4802/mass.527299；FIFO分布变化，不能当同分布遗忘。前8代128条cap中return39，route负30/其中真实normalized GAE正10/中位−.125274；navigation43，负36/其中正21。累计control net19233.75、route净−4214.997、capture244248仍主导。override后current/ring/fixed实际有效tick比例1.11349%/11.02317%/6.75721%。完整漏斗、分母、相对父/前点变化、固定Review首偏航/近似优势与DAgger覆盖保存CURVE_ANALYSIS/POINT_G8_OWNER_REPORT；CPU接g16，健康训练不改预算或门禁。

r12 g16固定点消费：Hunter难/中/易2/4/8、Python Tactical12（各64）；相对g8为+2/+4/+2、+2，相对父点−1/+3/−4、+2。八路线交付death/sugar/drop/near/mid/far/recovery/full=0/0/5/16/16/10/2/3（各16，人工carry题仍只诊断）。natural full窗口→目标11/11（episode共现）、首oracle4/6、持续1072/1920、推进1154/4337、拾包7/16→交付3/16、恢复28/29；首控30/归因8/pop30主动24/因果击杀8存活8、death2/self0。完整父/前点机会归一化delta、固定seed首偏航及stage/seat/geometry分层在CURVE_ANALYSIS/POINT_G16_OWNER_REPORT。

FIFO valid11770，return4183/greedy2276/target mass.541820（g4/g8=.575787/.557845），navigation4223/mass.519313，精确无公共joint pair冲突0；变化仍混入learner-state分布变化，非同分布遗忘证据。前16代cap256条中return82，route负67/其中真实normalized GAE正21/中位−.146857；navigation68，负54/其中正34；own recovery56，负51/其中正25。累计control net37444.5、route净−10541.482、capture490704仍主导。实际override后current/ring/fixed有效tick比例1.98106%/8.45770%/9.49899%。库存题与完整闭环仍弱，短点改善不称目标达成；固定128代继续，CPU依次等待g32，不设门禁。


r12 g32固定点消费（20261018）：JS Hunter难/中/易6/2/10、Python Tactical8（各64）；相对g16为+4/−2/+2、−4，相对父点+3/+1/−2、−2。八路线交付death/sugar/drop/near/mid/far/recovery/full=0/1/8/16/16/9/0/3（各16；near/mid/far仅人工携包诊断）。natural full窗口→目标11/11为episode共现、首oracle6/7、持续1068/1450、推进1131/5418、拾包7/16→交付3/16、恢复35/39；首控39/归因13/pop39主动31/因果kill13存活12，death9/self0。库存拾包16/16而交付0/16或1/16，仍需定位实际返程偏航，不能再解释为真实取包后无路。

g32 FIFO valid10729，return4197/greedy2256/target mass.534550；navigation2865/mass.618619；精确输入无公共joint pair冲突0。前32代固定首16偏航cap：return172、route负149/其中normalized GAE正46、中位−.161814；navigation110、负84/其中正53、中位+.133615；own recovery114、负107/其中正45。累计control net70352.75、route net−22813.425125、capture982824仍主导。override后current/ring/fixed实际有效tick比例2.28343%/8.91326%/11.38386%。完整固定漏斗、父/前点机会归一化变化、固定Review与DAgger覆盖见CURVE_ANALYSIS和POINT_G32_OWNER_REPORT。

固定输入拟合校核：在g16保存FIFO的首512条合法return与首512条navigation上，g16→g32 Actor的return target-set mass .547007→.634424、greedy278→325；navigation .532026→.640968、278→350。固定行索引SHA256 d1a984cee39d56262619cbdbe9526546426e96970bb97d93c7d742cae7d635e3。此为历史train-only样本，不是holdout、总体能力或在线梯度因果；但已说明各点变化FIFO均值下降不能直接称同分布遗忘。g16合法return FIFO几何覆盖含near572/mid1477/far1993/边界未判141、距离中位23；不能将返程问题归因为完全缺失长距离监督。训练数值源码不变，固定128代继续；下一固定点g64只用于诊断和准备后继，无门禁。


r12固定库存fixture因果来源审计（20261018，只读，不改训练或固定评估）：`je10_route_funnel.setup()` 先按敌笼距离选 reset cell，直接覆盖 learner `pos`，最后将 `bunSpawnPos` 改成该位置；没有从己笼/自然出生点走入敌笼的过程。偶数seed还保留原生砖。对预声明全部16个seed，death-stock、sugar-stock、recovery三题各有8/16在reset时己笼静态距离已INF，而敌笼距离16/16均finite。因此这三个受控题的16局原始交付分母仍完整保留，但不能整体充当真实入站取包后返程能力分母；尤其这8/16的初始连通属性是fixture设计产物，不是模型取包后造成，也不是动态封路证据。不得择seed删除原分母后伪称改善。

唯一Review链路逐值hash核验g0/4/8/16/32的固定seed2026101800既有库存、掉包、自然完整录像，共20份，无新模型评估/rollout。所有发现的真实pickup快照都能回到其实际reset起点、已走过格点在pickup时阻塞计数均0；g16/g32 stock seed0在tick105/165拾包时，回fixture起点距离2，而home字段为INF，直接复现“真实事件但没有真实从home入站”的语义错配。g8/g16自然full固定seed真实pickup的home/back-to-start距离36/32均finite。审计保存STOCK_FIXTURE_PROVENANCE_THROUGH_G32；源为已有Review动作和状态hash，不将静态INF解释为本任务规则下真实取包后不可达。此次定位进一步支持已撤回旧无路/预开路结论，不热改运行中的数值配方；后继优化应依据自然完整链及合法固定数据拟合，受控题仅独立诊断。


r12固定Review方向口径校核（20261018）：既有g0/4/8/16/32首seed的29条阶段首偏航，在相同状态hash、目标mask、当前wall/brick/pushable图上用封印训练的continuous_route_metric/action_set重算，无新对局或参数更新。其中4条被训练共享场接受：g4 sugar-return709、g8 death-return649、g8 sugar-return709、g16 drop-return80。这些只是网页旧距离近似与训练连续势能不同，不再计为训练方向错误；尤其g16 return80向右gain+.217499应明确保留为诊断oracle差异。g32 drop-return80向左gain−.157501，最佳右+.267498（下+.215000），仍是具体方向偏离；JS近似GAE−.022189不能替代真实collector优势。g32 drop-chase1向左虽有+.157499推进，最佳下+.215002，属于次优而非反向。完整动作/hash/五方向gain/共享oracle在FIXED_REVIEW_SHARED_FIELD_COMPARISON_G32。此比较未重建训练aux与native temporal安全证书，几何一致不能声称完整监督冲突；不改固定协议或健康run。


r12共享势能首偏航全段复核（20261018）：对既有固定seed game0的real_drop与natural_full录像逐动作重放，并在原记录phase/target时间线上扫描3375个目标状态、17种拓扑/目标场；无新模型对局。g16 real_drop原JS首返程80被训练场接受后，实际共享场首偏离应后移到tick90：向左−.172501、最佳右+.157499，state hash 51b144696f9db28dedd172c025f0fdc0323050d8506759fcfa8a1bf2ea7f57a1。g32 real_drop首偏离仍tick80，返程几何oracle一致184/309机会、反向84；父g0为158/405、反向161，g16为56/181、反向77。机会分母来自单一固定录像，不是总体提升结论；g32仍有显著反向动作。

自然full固定seed：g8真实pickup后共享场返程一致356/409，首偏离1256为停步（oracle右）；g16一致178/198，首偏离2178为向左−.157497（oracle右+.182476）；g32该seed无pickup，不能用零次偏航冒称成功。动作/hash/目标/势能/五方向gain/JS近似GAE完整保留FIXED_REVIEW_SHARED_FIRST_DEVIATION_G32。局部fuse/linger安全与共享势能均复用原图，但尚不包含native多步安全证书或collector critic，仍与真实训练cap normalized GAE分列。r12数值配方不变，等待g64/96/128固定点。


r12 g32控制信用量级复核：首次控制奖励累计82493、残余爆破信用1914.25（残余/首次控制=2.3205%），死亡clawback−14054.5，净70352.75；对应首次控制credit计数21629、残余计数10100。终局capture982824仍明显主导，不将这些不同事件机会计数误当统一kill转化率。控制归因与自然完整局机会漏斗继续按固定点评估，奖励量级不是能力门禁。


r12 g32固定偏航×FIFO覆盖补充：唯一Review real_drop seed2026101800/tick80的Actor move概率约[.000598,.003294,.982022,.014022,.000065]，错误向左高度确定；该点几何共享势能向左gain−.157501。g32保存FIFO有15条同wall/brick/pushable、同base/origin、位置L1<=1的返程行，其movement标签合计[0,3,0,12,0]，没有支持向左；该几何子集平均target mass.379854，global特征平均绝对差.052696。完全相同Actor输入匹配0，不能视为精确标签冲突，不能把近邻全部视为与Review相同威胁/安全状态。固定训练历史输入g16→g32拟合改善与此Review高置信错误并存，优先区分状态分布/泛化与局部拟合，不以FIFO快照缺某输入断言全程未访问。产物REVIEW_G32_FIFO_COVERAGE、REVIEW_G32_ACTOR_OBSERVATIONS；只读无新对局，不改当前配方。


r12 g64固定点消费（20261018）：COMPLETE及28份hash逐一核验。JS Hunter难/中/易0/1/12、Python Tactical12（各64）；相对g32为−6/−1/+2、+4，相对父点−3/0/0、+2。Hunter death/self均0/0；Tactical death/self1/1。八路线交付death/sugar/drop/near/mid/far/recovery/full=0/0/4/16/15/9/0/4（各16；受控库存与人工carry诊断范围仍单列）。natural full窗口→目标13/13为episode共现、首oracle7/9、持续1436/1933、推进1456/5371、拾包9/16→交付4/16、恢复34/38；首控39/归因4/pop39主动29/因果kill4存活4，death7/self0。自然完整交付相对父点+4、前点+1，但掉包与战斗指标波动，不能宣称稳定提升；g64的seed0自然局无取包，不以零偏航计成功。

g64 FIFO valid10302，return3988/greedy2080/target mass.528504，navigation2501/mass.582508；current/history返程实际replay仍各37.5%，贡献×2未改变采样配额。前64代固定cap：return330、route负289/其中真实normalized GAE正103/中位−.126569；navigation202、负159/其中正100/中位+.128587；own recovery225、负216/其中正97。累计control gross167861.75/net135546.75、route+26303.208/−74151.375/net−47848.169，终局capture1993392仍主导。verifier eligible20614435、bounded selected1564417、accepted1397354、query87504；普通return active11855212/accepted496537/query32647（含bounded selection，不当纯几何拒绝率）。正确scene8事件window84491/kill52299/stock53232/loose2111/custody42464/capture40518/transfer5874，完整事件机会归一化在CURVE_ANALYSIS。

动态联赛基础抽样current/ring/fixed条件比例9.8976%/38.6289%/51.4734%；override后实际有效tick占比2.47299%/9.47153%/12.89052%（其余固定teacher/规则/弱对手/assist单列），不将底层抽样概率当实战对手暴露。所有父点/前点rate delta、固定Review、阶段覆盖与优势分解在POINT_G64_OWNER_REPORT；训练在评估完成时已健康g74并继续128代，r13不等待最终CPU报告。


r12 g64固定输入×实际偏航复核：同一g16 FIFO固定首512 return输入，g16/g32/g64 target mass=.547007/.634424/.634155，greedy278/325/337；navigation=.532026/.640968/.574778，greedy278/350/299。旧返程train-only状态拟合有保持，不能用变化FIFO的.528504均值断言整体遗忘；也不能将此保持因果归给×2（无同父x1对照）。g64现有唯一Review real_drop固定seed全段扫描，返程共享场一致102/189机会、反向64，首偏离tick80向左−.157497，最佳右+.217499，state hash bdf2d792186f8db08fb804e8dc70545736db3a134324c957dfe4a0924dc87d06、JS近似GAE−.019282。追包首偏离tick1是次优正向+.157499而非反向，最佳下+.215002。g64固定seed自然full无路线目标，不用零错误替代成功。g64因果fixture审计与先前一致；同一reset选择问题不再进入真实入站返程解释。产物FIXED_FIFO_G16_FIT_THROUGH_G64、FIXED_REVIEW_SHARED_FIRST_DEVIATION_G64、STOCK_FIXTURE_PROVENANCE_THROUGH_G64；不新增模型对局、不改训练。


r12固定单状态局部保持证据（20261018）：g16/g64 real_drop seed2026101800/tick80的完整Review状态hash相同（bdf2d792186f8db08fb804e8dc70545736db3a134324c957dfe4a0924dc87d06），共享场最佳向右、向左gain−.157497。固定该同一obs/global/legal输入，对g0/4/8/16/32/64 raw Actor无新对局推理：右动作概率=.009259/.016191/.006266/.900652/.011565/.032274；左=.968974/.975448/.988908/.077697/.985335/.956478。greedy只有g16正确向右，其后回退。该状态局部保持问题与历史g16 FIFO整体return mass维持.634可以同时成立；既不能只看均值称无遗忘，也不能用单点称总体灾难性遗忘。真实collector cap和固定Review/优势继续分列，未将固定评估状态导入训练。FIXED_REVIEW_G64_ACTOR_FIT_THROUGH_G64保存同输入/各policy hash核验结果。后继根因排序应同时检查新状态覆盖、首返程阶段保持与现有loss混合，而非盲目再加返程系数；当前128代不停，最终曲线后决策。


r12 g96固定点消费（20261018）：COMPLETE及28份hash通过。JS Hunter难/中/易0/2/10、Python Tactical10（各64），相对g64为0/+1/−2、−2，相对父点−3/+1/−2、0。Hunter death0/1/2、self0/0/2；Tactical death/self0/0。八路线交付death/sugar/drop/near/mid/far/recovery/full=1/3/6/16/14/8/2/3（各16；受控库存与人工携包仍仅诊断）。natural full窗口→目标12/12为episode共现、首oracle6/10、持续1472/2125、推进1555/5271、拾包9/16→交付3/16、恢复31/37；首控38/归因9/pop38主动31/因果kill9存活9，death12/self0。完整交付保持高于父点0/16但未超过g64，返程、威胁切换和战斗保持仍波动，不作稳定提升或门禁结论。

g96 FIFO valid10503，return4357/greedy2274/target mass.526628，navigation2249/mass.625031，exact-input无公共joint pair冲突0（重复组10）。前96代固定cap：return487、route负435/其中真实normalized GAE正153/中位−.130996；navigation301、负241/其中正155/中位+.135116；own recovery343、负323/其中正136。累计control gross250841.25/net200036.25、route+39050.066/−111909.374/net−72859.311，终局capture2997432仍主导。scene8事件window125099/kill78376/stock80971/loose3426/custody64219/capture61596/transfer9157；严格按native-full分母归一化，见CURVE_ANALYSIS。

动态对手基础抽样current/ring/fixed条件比例9.20321%/39.89393%/50.90285%；override后实际有效tick占比2.35754%/10.13919%/13.10689%，其余来源另列。完整父/前点机会归一化delta、固定Review、cap优势与覆盖在POINT_G96_OWNER_REPORT。评估完成时训练已健康g104，最后24代照常完成；后继r13已预声明，先接续再综合最终结果。


r12 g96共享势能和固定输入补充：同g16 train-only FIFO固定512 return行，g96 target mass.627533/greedy325，navigation .650947/greedy340；相对g64返程平均仅轻微变化。g96真实drop固定seed首返程反向tick81向左−.257500，最佳右+.337502，返程一致150/289机会、反向103；自然full固定seed真实路线首返程反向tick721向左−.157497，最佳右+.182476，一致165/195、反向19。完整state hash/目标/动作/共享势能/近似GAE见FIXED_REVIEW_SHARED_FIRST_DEVIATION_G96。平均历史拟合保持与具体转向失败并存；不把人工携包fixture或静态INF再混入根因，也不将这些单录像机会率当总体成功率。

固定run `runs/je10_bun_20261018/control_route_v38_r12_causal_clean` 完成128代/268,435,456 ticks；compute=5915.514s、wall=6783.659s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r13_causal_clean`；本次后继配方选择依据 `Predeclared same-recipe fixed continuation; a concrete evidence-backed tested next factor may replace it before terminal. No final evaluation wait.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。


### r12 固定128代终态、r13正式接续与下一轮诊断方向（20261018）

r12完成128代/268435456 fresh ticks、atomic_stop=false，128次finite与Actor/Critic真实非零逐代通过；退休scene tick与无真实custody的active return累计均0。compute5915.514s、wall6783.659s。已先核验8卡释放（devices0..7/compute_clients0），立即启动预声明r13_causal_clean，再消费最终结果；未等待CPU、报告或能力阈值。r12最终独立CPU恢复full-state/next-RNG digest及160份封存源码通过。FINAL_CAUSAL_AUDIT、GPU_RELEASE_RECEIPT、TERMINAL_RESTORE_MONITOR、RUN_TERMINAL保存原始证据。

r13最小新smoke已通过finite/nonzero/save/fresh CPU restore，正式g1/2097152 fresh ticks已确认：8副本exact，Actor/Critic L2=3.248070478/2.244113207，policy SHA256 d0ad7c91f0bb797d622ea9c0d22f44b22444e272d9a2983bd0423dd9fb8f7950。formal g0独立核验raw Actor/Critic与完整pool等于r12 g128、retention reference相同、EMA=raw、Adam0/FIFO空/计数0、身份seed379251886的RNG精确一致、8192 learner空手、退休scene0，GPU cache独立、无旧namespace状态或旧smoke重启。固定128代继续，唯一CPU/JS owner跟随同g0/4/8/16/32/64/96/128。r14_causal_clean同预算后备已预声明，健康训练无门禁且终态先接续。

r12全固定曲线（所有点COMPLETE，g128的28份hash通过；均不择优）：

| g | JS Hunter 难/中/易（各64） | Python Tactical（64） | real drop交付/16 | natural full交付/16 | 人工far诊断/16 |
|---|---|---:|---:|---:|---:|
|0|3/1/12|10|10|0|11|
|4|0/1/6|6|7|2|12|
|8|0/0/6|10|4|3|10|
|16|2/4/8|12|5|3|10|
|32|6/2/10|8|8|3|9|
|64|0/1/12|12|4|4|9|
|96|0/2/10|10|6|3|8|
|128|1/0/10|7|13|2|5|

最终八路线交付death/sugar/drop/near/mid/far/recovery/full=1/0/13/16/16/5/0/2；相对父点0/−1/+3/0/0/−6/0/+2，相对g96=0/−3/+7/0/+2/−3/−2/−1。人工near/mid/far及受控库存/恢复fixture保持明确诊断标记；库存reset每题8/16缺真实home入站来源的偏差已定位，不能用来推断取包后无路，也不删seed改分母。最终natural full窗口→目标10/11为episode共现、首oracle5/7、持续966/1689、推进986/3480、拾包6/16→交付2/16、恢复21/25；首控25/归因7/pop25主动16/因果kill7存活5、death6/self0。JS Hunter death2/1/1/self均0，Tactical death2/self0；相对父点Hunter安全击杀−2/−1/−2、Tactical−3。真实drop与自然局有部分收益，但完整路线、长程诊断与战斗保持未稳定提升；不宣布任务能力完成，不作停训/晋级门禁。

终态正确scene8累计事件window165796/kill104560/stock108773/loose4778/custody85505/capture82857/transfer12417；每百万full tick依次892.206/562.674/585.346/25.712/460.132/445.883/66.820。reward首次控制327215.25、残余6852.5（2.0942%）、clawback−69501/net264566.75；route+51725.410/−149372.318/net−97646.911，capture4000176仍主导。verifier active40326187/selected3137565/accepted2809377/query170296；selected前有bounded selection，不能据active差值宣称几何或安全拒绝。current/ring/fixed底层条件比例9.24086%/38.89052%/51.86862%，override后实际有效tick占比2.40688%/10.05185%/13.54995%，其余teacher/规则/弱角色/assist在原始exposure单列。

终态FIFO valid10049，return3886/greedy2158/target mass.552397，navigation2285/mass.656560；exact-input重复组10、无公共joint pair冲突0。前128代固定首16 cap中return651、route负578/其中真实normalized GAE正212/中位−.127863；navigation372、负292/其中正189/中位+.111134。不是总体失败率、Q或完整在线梯度。固定g16 train-only首512 return输入mass .547007→.634424→.634155→.627533→.652177（g16/32/64/96/128），greedy278/325/337/325/333，历史平均拟合改善；但固定相同Review tick80输入正确右概率g16 .900652→g128 .007427、错误左 .991811，存在明确局部回退。最终固定drop首返程80向左gain−.157497，最佳右+.217499，返程一致141/407机会、反向149；自然full首返程偏离972向上gain−.202499，最佳下+.157501，返程一致161/208、反向28。完整动作/hash/目标/阶段/势能及JS近似GAE与collector真实优势分列。曲线见 [r12完整固定点曲线](../reports/je10_r12_curve_20261018.png)。

证据排序的后继假设：①真实返程闭环与局部转向保持是首要薄弱点，不能用提升的平均FIFO拟合掩盖高置信反向；历史合法长返程覆盖已存在，不是零采样。②代码明确每个rollout episode chunk只保留普通return第一次已认证偏航，后续不同位置的已认证纠正会被同一个failed[2]屏蔽；这是否限制路线多转弯监督尚是可验证假设，而非已证明能力根因。③约36.7%的负route返程cap仍有正normalized GAE，loss混合值得后续核验，但多数返程cap为负优势，不能直接认定增奖励系数就是解法。④战斗与长程保持下降，需保留固定监测及现有combat/安全配方，不在同轮同时更改混合与DAgger。

下一最小候选：只将真实custody、已通过原生安全/进度认证的普通return查询，从每chunk首1条扩为首条加1条不同格点、至少间隔8tick的后续偏航；保留首偏航、其他fragment及preparation配额、所有oracle安全判定、FIFO容量/采样、×2贡献、奖励/PPO/网络/场景混合不变。先RED→GREEN验证配额、去重、terminal reset、phase切换与原有preparation行为，再决定安装于已预声明后继；绝不热改健康r13，不将固定Review输入导入训练。此候选不作为训练门禁，未准备好仍按同配方后备接续。唯一CX owner继续监督与迭代。

### r14 后继预声明：合法返程第二不同格查询（20261019）

实验ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r14-clean-causal-init`，命名空间 `runs/je10_bun_20261018/control_route_v38_r14_causal_clean`。父点预声明为健康r13固定g128（最初干净链从用户指定r10 g128，经r12/r13；r11仍EXCLUDED_INVALID）；train seed4241458138、technical seed2007916228。预算8卡×128代/268435456 fresh ticks；全新Adam/EMA-to-raw/RNG/env/FIFO/state/cache/eval，完整raw Actor/Critic及pool继承。场景混合combat .35、bridge .20、natural full .35、其余合法空手scene1/5/7合计.10；scene2/3/4/6严格零采样，固定return replay继续要求原生真实事件证明。

唯一新增数值因素：普通真实custody返程在每rollout episode chunk保留全部原有首偏航query，再允许1条不同Actor观测格点、距前条至少8tick的后续原生认证偏航；真实terminal后重置，phase/revision切换不能补配额。其他fragment及preparation配额、原生安全/进度认证、FIFO容量/采样、合法返程贡献×2、奖励/终局主导、PPO/网络/场景混合均不变。选用理由：r12同一固定Review输入的局部转向高置信回退与历史FIFO平均拟合改善并存；代码首次配额会屏蔽后续不同位置的认证纠正。其能力影响仍是假设，顺序接续无同父对照，不声称单变量因果能力结论。

TDD：查询baseline RED实际4项行为失败→10项GREEN；终态安装/继承RED6项失败→35项相关GREEN；独立CPU真实collector→GAE→FIFO→PPO→finite Actor/Critic非零→保存恢复1项GREEN。后者只在进程内编译候选，未改当前数值源码。原始日志 `runs/je10_bun_20261018/RETURN_QUERY_{DIVERSITY,RECIPE}_{RED,GREEN}.log`、`RETURN_QUERY_COLLECTOR_GREEN.log`。r13数值seal161份逐hash变更0。行政relay识别已运行旧owner保留的nested proposal，候选helper/test逐hash封存；安装器要求前继g128正常终态及8卡释放收据，禁止热改健康r13。更新NEXT_RECIPE_PROPOSAL/SUCCESSOR_PREDECLARATION，保留同配方后备历史证据。

评估不变：唯一CPU/JS owner，g0/4/8/16/32/64/96/128；每点Tactical Python v2、网页JS Hunter难/中/易各64局，固定八路线各16局，原seed顺序不择优。人工carry及受控库存reset仅作明确标注诊断，真实natural full独立；不能推断真实取包后无路。COMPLETE及artifact hash后消费，输出相对父点/前点的固定机会漏斗、首控→爆破→击杀/存活、目标→首次方向→持续推进→拾包→返程→交付、恢复切换、真实collector优势及JS近似优势分列、DAgger覆盖/FIFO拟合及current/ring/fixed实际动作来源占比。训练/CPU独立；恢复点每4代，CPU或报告不得令GPU等待。

r13继续跑满128；终态先核验8卡释放→立即执行r14新技术smoke→fresh正式训练→确认首代finite/Actor/Critic真实非零后主动通知，再综合全曲线报告。中间能力结果只用于诊断；无admission/promotion/能力门禁。最终仍不佳则按机会归一化、共享场首次偏航、奖励/真实优势、查询覆盖及保留指标排序根因，准备后继单轮最小可验证改动；改善则补最弱漏斗且保持战斗。健康run不因短点停止。

### r13 g4 固定点消费（20261019）

COMPLETE的28份产物及policy hash通过；唯一评估已转g8，GPU继续。JS Hunter难/中/易1/1/12（父1/0/10）、Python Tactical11/64（父7/64）；真实drop交付10/16（父13），自然full2/16（父2）。自然full窗口→目标11/11为局级共现、首oracle4/8、持续1476/2648、推进1482/5853、拾包6/16→交付2/16；恢复32/35，首控36/归因12/pop36主动29/因果kill12存活12，death7/self0。人工carry及受控库存另列，不作真实入站结论。

当前FIFO返程1001条、greedy634、target mass.633271；exact-input冲突0。前4代固定first16 cap仅9条返程，其中route负3条的normalized GAE均正、中位1.990435；不是总体优势率。native full累计window5657/kill2246/stock2119/loose3/custody1656/capture725/transfer57；capture114912、control净9202、route+2016.661/−2933.818/net−917.156，终局主导。override后current/ring/fixed有效tick占比2.36429%/8.55287%/8.19938%，其他实际来源见POINT_G4_OWNER_REPORT。

唯一Review固定seed2026101800重放且训练共享势能重算：real_drop返程首偏离tick81，左gain−.257500、最佳右+.337502，机会匹配75/402、反向191；自然full取笼126/331、反向191，返程175/195、反向8。自然full返程首偏离1940仍是正进度但非最高进度方向（下+.138927、右+.323570），不能称反向。JS近似GAE与collector真实normalized GAE分列；JS近似字段中tick65不再直接作为共享场首偏航。该固定局取包前与真实drop返程薄弱并存，不以单局替代总体结论。证据保存在run的POINT_G4_OWNER_REPORT及FIXED_REVIEW_SHARED_FIRST_DEVIATION_G4；r13配方不变，r14预声明覆盖假设不作能力门禁。

### r13 g8 固定点消费（20261019）

COMPLETE的28份产物及policy hash通过；唯一评估已转g16。JS Hunter难/中/易0/6/7、Python Tactical7/64；相对父点−1/+6/−3、0，相对g4−1/+5/−5、−4。八路线交付death/sugar/drop/near/mid/far/recovery/full=3/1/5/16/16/11/0/1；real drop对父−8/对g4−5，自然full对父及g4均−1。人工near/mid/far不混入真实闭环能力结论。自然full窗口→目标10/11（局级共现）、首oracle3/8、持续1879/3690、推进1882/6086、拾包8/16→交付1/16，恢复26/30；首控34/归因14/pop33主动26/因果kill13存活13，death8/self0。

FIFO返程2759条、greedy1557、target mass.567858，当前分布变化不能与g4当作同输入遗忘率。前8代cap返程17条、route负5条均有正normalized GAE、中位+.740850；不是总体优势率。control净20363.25、route+3851.284/−8179.927、capture255072仍主导；override后current/ring/fixed有效tick占比2.41899%/8.53517%/8.34857%。

唯一Review固定seed共享场复核：real_drop追取75/79、无反向；返程109/501、反向201，首偏离tick80记录动作gain−.157497、最佳右。自然full取笼112/193、反向74，返程489/998、反向411；首返程1038动作gain+.022499但非最大进度方向，不能把它与后续反向混为一谈。原始状态hash、动作、目标、势能和JS近似GAE在FIXED_REVIEW_SHARED_FIRST_DEVIATION_G8；collector真实normalized GAE在POINT_G8_OWNER_REPORT独立保存。追包较一致而真实交付差，闭环仍弱；该诊断不提前终止健康run。监督至g24/50331648 ticks，finite/nonzero及因果审计正常。

r13同输入复核：固定g4 FIFO首512返程输入，g4→g8 target mass .652174→.683356、greedy328→353；导航.540496→.612316、279→324。输入/标签/mask固定、索引SHA256 421abcf8fd80dfcb4a3c647b74dbefcc0e8ddb6f1f2578f07b00c2b4c4be549d。平均历史拟合改善与实际路线下降并存；不能将变化分布FIFO的均值或一条失败轨迹直接称整体遗忘。原始FIXED_FIFO_G4_FIT_THROUGH_G8保留；不将诊断样本导入训练。

### r13 g16 固定点消费（20261019）

28份COMPLETE hash及policy通过。Hunter难/中/易0/0/12、Tactical7/64；对父−1/0/+2、0，对g8 0/−6/+5、0。八路线交付2/1/5/16/16/11/3/0。自然full窗口→目标9/9（局级共现）、首oracle2/5、持续890/1334、推进924/3084、拾包5/16→交付0/16、恢复16/20；首控21/归因6/pop21主动14/因果kill6存活6，death5/self0。机会与交付同时记录，不能把较少机会与策略未走对混成一个原因。

FIFO返程3751条、greedy2058、mass.547599；exact-input冲突0。累计cap返程53、route负39/其中GAE正15、中位−.026391；不是总体率。control净39185、route+7318.049/−17949.905、capture502920仍主导。实际override后current/ring/fixed有效tick占比3.44491%/8.08172%/8.80228%。同g4首512固定FIFO返程mass(g4/g8/g16)=.652174/.683356/.665920、greedy328/353/351；导航.540496/.612316/.590394、279/324/312。

固定seed共享场：real_drop追取76/79、无反向；返程140/536、反向229。tick80记录左gain−.157497，右+.217499，状态hash bdf2d792186f8db08fb804e8dc70545736db3a134324c957dfe4a0924dc87d06，与r12固定失败状态一致。该输入已从本轮唯一Review逐值hash验证后导出，用于只读历史Actor概率比较，绝不加入训练。固定natural full只有160个取笼目标tick、共享oracle机会0，且无返程阶段，首偏航/返程率保持缺失而不伪造为0；不据此解释真实取包后不可达。r13固定预算继续，r14只增加后续不同格认证返程query的预声明保持不变。

同一固定Review hash的Actor概率补证：r13 g0/g4/g8/g16正确右概率 .00742738/.00105378/.00107978/.00069516，错误左 .99181056/.99750817/.99738652/.99866831。该错误在r12 g128父点即存在，r13并未新引入；本轮×2仍未修正这一局部状态，不能据单状态称整体灾难性遗忘。原始FIXED_REVIEW_G16_ACTOR_FIT_THROUGH_G16保存Actor policy hash校验、相同obs/global/legal及来源Review hash；无新对局或标签导入。

### r13 g32 固定点消费（20261019）

28份COMPLETE及policy hash通过。Hunter难/中/易2/3/9、Tactical7/64；对父+1/+3/−1、0，对g16 +2/+3/−3、0。八路线交付0/1/6/16/16/10/1/2。自然full窗口→目标15/15（局级共现）、首oracle7/12、持续1805/3005、推进1848/5043、拾包11/16→交付2/16，恢复25/29；首控30/归因14/pop30主动19/因果kill14存活12，death7/self0。拾包机会较父6/16增加，交付仍仅父水平2/16，不能只凭拾包宣称路线闭环改善。

当前FIFO返程3960条、greedy2111、mass.530732；精确输入冲突0。累计cap返程131、route负109/其中真实normalized GAE正44、中位−.071295。control净73187.75、route+13810.919/−36907.969、capture1006272，终局仍主导。native full累计window45258/kill26842/stock25965/loose1044/custody21494/capture20008/transfer2722。override后current/ring/fixed有效tick占比3.20702%/8.24716%/11.26555%。verifier bounded selection与安全/进度认证分母继续分列，不将未选中误称拒绝。

固定seed共享场real_drop追取73/79、反向0；返程123/223、反向80，首偏离仍tick80、左gain−.157497。相同hash固定输入g32右概率.00131819、左.99781185。固定natural full取笼73/122、反向33、无返程阶段，不编造返程分母。同g4首512 FIFO return mass(g4/g8/g16/g32)=.652174/.683356/.665920/.616972、greedy328/353/351/323；navigation=.540496/.612316/.590394/.672599、279/324/312/350。历史返程拟合后段回落而导航上升，局部错误始终未修正；这比仅比较变化分布FIFO均值更能定位返程薄弱，但仍非总体遗忘率或DAgger加权的单变量因果证明。

全部原始POINT_G32_OWNER_REPORT、FIXED_REVIEW_SHARED_FIRST_DEVIATION_G32、FIXED_REVIEW_G16_ACTOR_FIT_THROUGH_G32、FIXED_FIFO_G4_FIT_THROUGH_G32保留。r13配方/预算不变，监督至g44仍finite/nonzero、污染0；唯一evaluator按协议等待g64，未添加非预声明对局。r14只改变第二不同格返程query的预声明继续，技术检查后不等能力结果。

r13固定COMPLETE曲线保存为 [r13固定点曲线](../reports/je10_r13_curve_20261019.png)，当前涵盖g0/4/8/16/32；后续按固定点更新同一图。人工carry/受控库存题显式诊断标注、Python/JS/JAX分列、JAX事件按natural-full有效tick归一化，无择优或平滑。

### r13 g64 固定点消费（20261019）

28份COMPLETE及policy hash通过。Hunter难/中/易0/5/7、Tactical12/64；对父−1/+5/−3、+5，对g32−2/+2/−2、+5。八路线交付1/0/8/16/16/9/0/1。自然full窗口→目标12/12（局级共现）、首oracle3/8、持续853/1648、推进975/4870、拾包8/16→交付1/16，恢复34/36；首控38/归因12/pop36主动31/因果kill11存活11，death8/self0。相对父自然交付−1、drop−5；战斗各协议表现混合，不能挑Tactical改善宣布全部保留成功。

当前FIFO返程3965条、greedy2104、mass.534189；精确输入冲突0。累计cap返程277、route负240/其中真实normalized GAE正84、中位−.166580；cap非总体率或完整梯度。control净139001.75、route+26544.420/−74778.550、capture2015976仍主导。native full累计window86080/kill53343/stock54062/loose2319/custody43316/capture41206/transfer6198。override后current/ring/fixed有效tick占比2.91402%/8.80096%/13.18811%，其余真实动作来源保留在POINT_G64_OWNER_REPORT。

固定seed共享场real_drop追取68/79、无反向；返程145/344、反向116，首偏离tick80、记录左gain−.157501。natural full取笼12/17、返程18/22，均无反向，首非最优方向仍有正进度；小机会分母单列，不能称整个自然局已学会闭环。同一固定g16失败输入，g64右概率.0674249、左.9238704，greedy仍左；局部概率回升未形成可靠纠正。同g4首512 FIFO返程mass(g4/g8/g16/g32/g64)=.652174/.683356/.665920/.616972/.613548，greedy328/353/351/323/321；导航=.540496/.612316/.590394/.672599/.658070、279/324/312/350/351。现象仍是返程和导航分化；不能将变化分布FIFO的平均值当作全部泛化能力。

原始POINT_G64_OWNER_REPORT、FIXED_REVIEW_SHARED_FIRST_DEVIATION_G64、FIXED_REVIEW_G16_ACTOR_FIT_THROUGH_G64、FIXED_FIFO_G4_FIT_THROUGH_G64保留，固定曲线图已更新g64。训练监督至g75仍finite/nonzero、两项污染0，CPU唯一owner按协议等待g96；r13预算不改，r14预声明最小查询覆盖改动继续。

r14接续补充（20261019）：为判断新增信号是否实际进入训练，增加只读host计数：`dagger.return_query_diversity`记录ordinary_return_queries、first_return_queries、additional_return_queries、max_queries_per_episode_chunk、quota_violations。复用已在host的context，按真实terminal重置、排除preparation及delivery独立配额；不重跑oracle、不复制Actor整批输入、不改变标签/loss/奖励或context schema，不作为训练门禁。安装器仅在终态释放后为worker加row日志hook；先完整预检collector与worker patch再写入。RED4项失败→最终39项相关GREEN；日志RETURN_QUERY_AUDIT_RED及RETURN_QUERY_AUDIT_GREEN_FINAL。真实collector→GAE→FIFO→PPO→恢复的既有GREEN保留，数值查询算法未再改。候选6份源码/test hash与接续marker已更新，实际封存旧owner函数兼容检查保存在SUCCESSOR_RELAY_COMPATIBILITY；r13的161份数值源码仍逐hash不变。

r14统计口径补充：additional_return_queries是collector选中监督候选、发生在bounded FIFO选择前，不保证每条追加样本进入loss；既有replay_current_returning/replay_history_returning继续给出实际当前/历史返程采样曝光，和FIFO snapshot拟合分别报告，不跨分母推断。此为只读统计范围澄清，数值配方不变。

### r13 g96 固定点消费与后继冻结（20261019）

28份COMPLETE及policy hash通过。Hunter难/中/易2/0/11、Tactical15/64；对父+1/0/+1、+8，对g64+2/−5/+4、+3。八路线交付0/1/1/16/15/8/0/1。自然full窗口→目标11/11（局级共现）、首oracle5/7、持续1000/1796、推进1113/4143、拾包5/16→交付1/16，恢复23/24；首控25/归因10/pop25主动13/因果kill10存活10，death2/self0。真实drop对父−12、自然交付−1，Tactical改善与路线下降并存；没有将g96作为停训或选优门禁。

FIFO kind3携包/交付类4955条、greedy2408、mass.488815，精确输入无公共pair冲突0。累计cap返程455、route负408/其中真实normalized GAE正155、中位−.135800；不是总体率/Q/完整梯度。control净204116、route+39249.212/−113064.700、capture2995992仍主导；native full累计window127013/kill79980/stock81846/loose3616/custody65222/capture61976/transfer9598。override后current/ring/fixed有效tick占比2.74705%/8.82493%/14.09803%。

固定seed共享场real_drop追取75/79、无反向；返程69/149、反向46，首偏离tick80左gain−.157497。natural full取笼oracle机会0，返程81/89、反向2，首非最优方向1423仍是正进度；小分母与未知值保留。相同g16失败输入的g96右概率.000430765、左.998427，局部仍未纠正。固定g4首512返程输入mass(g4/g8/g16/g32/g64/g96)=.652174/.683356/.665920/.616972/.613548/.582050，greedy328/353/351/323/321/303；导航=.540496/.612316/.590394/.672599/.658070/.689939、279/324/312/350/351/354。共享场只是可观测拓扑/局部危险诊断，不等同原生时序安全认证，禁止自动据此改标签。

实际replay current/history两半截至g96各9216/24576个kind3非preparation样本，占37.5%，并非返程完全缺样本；包括有放回/回退重复，不能当唯一状态数或全部PPO重复调用数。该类与原有×2 loss均可能包含delivery标记行；新host普通返程query统计排除delivery且单列kind3_delivery_queries，不混分母。r14测试的是在相同FIFO/replay额度和×2下增加不同更晚位置的认证纠正，不是增加返程slot/系数。原始REPLAY_EXPOSURE_G96_OWNER及POINT_G96_OWNER_REPORT保留。

r14 host分组统计RED1项KeyError→39项相关GREEN（RETURN_QUERY_DELIVERY_SCOPE_RED/GREEN），没有改数值查询算法；真实collector链GREEN继续有效。候选6份source/test hash、parent/seed/预算/协议、NEXT_RECIPE_PROPOSAL和SUCCESSOR_PREDECLARATION已冻结。实际封存旧owner函数确认保留nested proposal；候选collector/worker不与前继11份评估源码重叠，r13最终CPU可继续，技术/数值源码161份hash不变。后继安装仍必须等r13正常128终态与8卡释放；GPU正式主线不会等待最终评估/报告或能力阈值。

### r13 固定128终态与r14接续中（20261019）

r13正常完成128代/268435456 fresh ticks、atomic_stop=false，compute5993.682s、wall6887.704s。逐代128份事件核对finite及8份Actor/Critic L2均有限且>0；退休scene tick与无真实custody返程监督累计均0。FINAL_CAUSAL_AUDIT、TRAINER_EXIT、g128 COMMITTED均保留。GPU_RELEASE_RECEIPT独立实际查询devices0..7、compute_clients0/released=true；已先启动r14 owner及新8卡技术smoke，未等待最终CPU/报告或能力结果。此时r14正式非零更新尚待确认，不提前宣称正式训练完成。

r14实际plan、ROUND_DECLARATION与167份源码逐hash通过：父点r13 g128，train4241458138/technical2007916228，clean因果初始化、独立cache、新FIFO/Adam/env/RNG，普通返程第二不同格认证query、间隔>=8tick及host计数已安装；原有kind3非preparation×2与其他配方保持。r13 g128唯一CPU评估同时运行；安装文件不与前继评估seal交叉。r15同配方固定128代后备在r14技术smoke前已预声明，继承query diversity，不设能力门禁。唯一CX owner继续监督smoke→fresh正式首非零→固定128；未重启任何已终止旧smoke。

固定run `runs/je10_bun_20261018/control_route_v38_r13_causal_clean` 完成128代/268,435,456 ticks；compute=5993.682s、wall=6887.704s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r14_causal_clean`；本次后继配方选择依据 `r12 same-input fixed Review shows a high-confidence local reverse action despite improving average historical FIFO fit; collector first-only quota suppresses later distinct-state verified corrections. Test this query-coverage hypothesis without changing losses or scene mix.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。

### r13 完整证据、r14正式首非零及r15预声明（20261019）

r13固定128代正常终态、8卡释放→立即r14新smoke→正式训练的顺序已完成，没有等待最终CPU/报告或能力门禁。r13所有128代finite/8副本Actor与Critic非零、两项污染计数0；g128独立CPU full-state/next-RNG及161份封存源码通过。全固定评估COMPLETE，g128的28份artifact/policy hash核验通过。r11仍EXCLUDED_INVALID，不纳入能力曲线；r12最初从用户指定r10 g128干净回退，后续fresh接续链可追溯，不恢复旧stash或旧smoke。

r14正式g1/2097152 fresh ticks：Actor/Critic L2=3.144831657/2.178624392，8副本exact、finite；policy SHA256 ff853cd393ed2ea59b6c9a24638a7f08d58760162cfcebb9f784c325180adb46。FORMAL_NONZERO_CONFIRMED及parent RUN_TERMINAL.successor_formal_nonzero_verified均真。独立FORMAL_FRESH_INITIALIZATION核验raw Actor/Critic/完整pool/retention reference等于r13 g128、EMA=raw、Adam0/FIFO空/计数0、seed4241458138的RNG精确匹配、8192 learner空手/退休scene0、resume=None/独立cache，不复用旧smoke状态。实际smoke Actor/Critic3.299270630/2.275296926、save/fresh CPU restore通过。

新query信号已到达collector：smoke首条230/追加23；正式g1首条225/追加28，普通返程query共253，最大每episode chunk2、quota异常0；delivery查询独立计数0。它们是bounded FIFO选择之前的监督候选，不能直接称全部进入loss；实际current/history replay曝光仍按独立字段汇总。原有kind3非preparation×2定义包括delivery标记，未偷换成更窄普通返程口径。查询算法RED实际行为失败→GREEN、终态/host分类RED→最终39项相关GREEN、真实collector→GAE→FIFO→PPO→保存恢复GREEN；不是能力门禁。

r13完整曲线（各Hunter/Tactical64局，路线各16局，固定seed/greedy/assist-off，无择优）：

|g|JS Hunter难/中/易|Python Tactical v2|真实drop交付|natural full交付|
|---|---|---:|---:|---:|
|0|1/0/10|7|13|2|
|4|1/1/12|11|10|2|
|8|0/6/7|7|5|1|
|16|0/0/12|7|5|0|
|32|2/3/9|7|6|2|
|64|0/5/7|12|8|1|
|96|2/0/11|15|1|1|
|128|0/2/7|5|4|5|

最终八路线death/sugar/drop/near/mid/far/recovery/full交付1/0/4/16/16/10/1/5；对父0/0/−9/0/0/+5/+1/+3，对g96 +1/−1/+3/0/+1/+2/+1/+4。人工carry及受控库存仅作分列诊断，真实闭环不能由人工far改善替代。natural full最终窗口→目标13/14为局级共现、首oracle8/11、持续1472/2141、推进1491/5451、拾包10/16→交付5/16，恢复28/34；首控35/归因10/pop35主动23/因果kill10存活9，death8/self0。Hunter death1/1/0/self均0、Tactical death0/self0，但最终安全击杀0/2/7及5低于部分中点。自然局末点有收益，真实drop与战斗保持不稳定；不宣布两类能力稳定完成。

真实JAX full有效tick185290318，累计window167939/kill106597/stock109336/loose4987/custody86646/capture81893/transfer12971；每百万full tick依次906.356/575.297/590.079/26.915/467.623/441.971/70.004。control entry333887.75、残余7062.75（约2.12%）、clawback−71988.75/net268961.75；route+51743.302/−151465.014/net−99721.710、capture3950280仍主导。verifier active40452448/selected3138758/native-safe-progress accepted2809265/query168344；return active23464721/accepted997979/query65503。未记录各phase独立selected/全部prequota mismatch分母，不能将active差值称几何/安全拒绝或把accepted未query全部归于首条配额。override后current/ring/fixed有效tick占比2.67186%/8.97443%/14.39243%，其他真实动作来源单列；不把底层league采样比例冒充执行曝光。

终态FIFO valid9962、kind3携包/交付3916/greedy1990/mass.506294，navigation2218/mass.665092；exact-input重复组6、无共同joint pair冲突0。固定g4首512返程输入mass=.652174/.683356/.665920/.616972/.613548/.582050/.599883（g4/8/16/32/64/96/128），greedy328/353/351/323/321/303/309；导航=.540496/.612316/.590394/.672599/.658070/.689939/.629034、279/324/312/350/351/354/331。返程历史平均拟合仍有后段回落，不能仅以当前变化分布FIFO均值解释。

唯一Review固定seed2026101800最终real_drop追取75/79/反向0，返程257/522/反向221；真正共享势能首返程偏离tick80，左gain−.157497、最佳右+.217499、距离32.895，hash bdf2d792186f8db08fb804e8dc70545736db3a134324c957dfe4a0924dc87d06。原路物理可逆不变量不变，不提出真实不可达/动态封路/预开路解释。同输入最终右概率.0512358、左.920719；该错误在父点即存在。tick80 JS近似reward-GAE−.017386928，前一动作−.017624838，邻时刻差+.000237909；这是可观测重建/零value近似，不能当Critic优势或动作Q。真实collector固定first16 cap累计return626、route负558/其中normalized GAE正215/中位−.126352；navigation321、负252/其中正153/中位+.127880。cap非总体率，二者与JS近似分列。固定natural full game0没有可用目标阶段组，不伪造返程率。全证据保存于POINT_G128_OWNER_REPORT及FIXED_REVIEW/FIXED_FIFO系列；曲线图已覆盖全部8点。

证据排序与最小下一步：①真实取包后的方向保持/多转弯闭环最弱，同一固定失败输入持续高置信反向；原先每rollout episode chunk仅首1条普通返程偏航是代码事实，后续不同位置认证纠正被配额屏蔽，能力影响仍待验证。r14只增加1条不同格、>=8tick的后续认证查询，保留安全/进度判定、其他fragment/preparation配额、FIFO/replay额度、×2、奖励/PPO/网络/场景混合；g1追加28条确认可达，正式128继续。②实际return replay两半各37.5%的kind3非prep采样且×2已执行，不能再归因返程零曝光；拟合、状态覆盖与混合loss/真实优势的相对作用需继续分开检验，尚无完整在线梯度因果证据。③最终战斗保持回落、对手真实来源与自然路线分布仍需监测，不同轮同时改场景混合或系数。④初始化语义污染已删除且r13审计0，旧无路/预开路解释仍撤回，fixture来源错误不再用于真实游戏结论。未用固定Review输入回灌训练，也未自行择优seed或中断健康run。

r15后备实验ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r15-clean-causal-init` 在r14 smoke前已预声明：父r14 g128，train1955372956/technical1673720716，8卡×128代/268435456 fresh ticks，fresh namespace/state/cache/eval，继承query diversity、合法×2与当前完整配方。场景combat.35/bridge.20/natural full.35/合法空手其他.10，退休scene2/3/4/6为0。固定评估沿用seed20261001、g0/4/8/16/32/64/96/128、Tactical及Hunter难中易各64、八路线各16/固定seed2026101800..15，唯一全局CPU/JS owner及COMPLETE hash后消费。r14跑满128，终态先核验释放→立即后继smoke/正式非零→主动通知，再消费最终曲线/报告；若有更明确且经过TDD的最小下一因子，可在终态前替换后备proposal，绝不热改健康r14。中间结果只诊断，最终不佳继续根因排序与最小改动，不设能力/admission/promotion门禁。


### r14 g4异步消费与返程查询实际曝光（20261019）

唯一CX owner已重新实际核读dirty diff、原CAUSAL_INIT_RED、no_prestock_carry测试和r11 EXCLUDED_INVALID；原修复GREEN与r10 g128→r12→r13→r14干净接续证据保留，不重启旧smoke、不恢复旧stash。r14的167份数值源码hash一致，健康固定128继续，退休scene与无原生custody返程监督计数均0。

g4 COMPLETE的28份artifact及policy hash全部通过；JS Hunter难/中/易0/2/11（相对父0/0/+4），Python Tactical11（+6），各64局。八路线death/sugar/drop/人工near/mid/far/recovery/natural full交付0/0/4/16/16/9/0/1，各16局；自然完整局窗口→目标10/10为局级共现，首oracle5/6、持续1216/2272、推进1284/4342、拾包5/16→交付1/16、恢复切换20/24，death3/self0。对父自然交付−4、真实drop0，不能用人工carry推断真实闭环，更不重提真实取包后无路/预开路。

累计普通返程查询1308=首条1106+追加202（15.443%），每rollout episode chunk最多2、异常0，kind3 delivery候选另计1。实际current/history各384/1024个kind3非prep replay（37.5%）；这些包含delivery且有放回/回退重复，不等于唯一状态或每次PPO复用数。追加候选发生在bounded FIFO选择之前，缺逐候选loss追踪，不声称202条全部进入loss。RETURN_QUERY_EXPOSURE_G4保存完整逐代分解及哈希核验。固定first16 cap返程18例/route负7/其中真实normalized GAE正3，中位−.164281；不是总体冲突率或完整梯度结论。r14本轮只检验不同更晚格点的认证纠正覆盖，不改×2、FIFO或其他配方，不以中点结果停训。


g4补充只读同输入与唯一Review证据：固定本轮g4 FIFO首512返程行，父g0→g4 target-set mass .442762→.617762、greedy225→315；navigation .435269→.504378、219→259。输入与标签固定且hash保留，属于训练历史FIFO拟合，不是holdout能力。唯一固定seed2026101800 real_drop共享场追取74/80、无反向；返程157/496、反向195，首次返程偏离tick81，左gain−.257500、最佳右+.337502、距离33.2075，state hash74134b5c40d0e4a463337ebb74789ea3878b5f591469ee9b1a37aa5caf9b37b7；JS零value近似GAE−.0188093与真实collector GAE分列。共享场只复核可观测拓扑/局部危险，不等于原生时序安全认证。固定natural game0无相应有效目标阶段组，保持缺失分母。暂不能由平均FIFO改善推出多转弯返程闭环已解决；继续消费后续固定点，不热改、不停健康run。FIXED_FIFO_G4、FIXED_REVIEW_SHARED_FIRST_DEVIATION_G4与Review输入已保留，诊断输入绝不导入训练。


### r14 g8异步固定漏斗与同输入拟合（20261019）

g8 COMPLETE的28份artifact及policy hash核验通过。JS Hunter难/中/易0/1/9，对父0/−1/+2、对g4 0/−1/−2；Python Tactical6，对父+1、对g4−5，各64局。八路线death/sugar/drop/人工near/mid/far/recovery/natural full交付0/0/6/16/15/9/0/6，各16局；自然完整交付对父+1/对g4+5，真实drop+2/+2，仅为早期固定点收益，未称稳定完成。自然窗口→目标13/13为局级共现，首oracle8/13、持续2276/3138、推进2274/5771、拾包12/16→交付6/16、恢复28/36；首控38/归因14/pop38主动24/因果kill14存活12，death7/self0。

普通返程候选3463=首2885+追加578（16.691%），最多2/异常0、delivery另1；实际current/history各768/2048个kind3非prep replay（37.5%）。return active1459803/已选择后的native安全进度accepted61439/query3463，独立phase selected和全部prequota mismatch分母仍缺失，不把active差值归为安全拒绝或把accepted差值全归为配额压制。真实full累计window12374/kill6306/stock5821/loose87/custody4819/capture4093/transfer327，与JS机会分母分开。

固定本轮g4 FIFO首512返程同输入mass g0/g4/g8=.442762/.617762/.678915，greedy225/315/352；navigation=.435269/.504378/.601993，219/259/314。平均历史拟合继续改善，但唯一固定seed real_drop追取75/79/无反向，返程140/479/反向196，首返程tick80左gain−.157497、最佳右+.217499，hash bdf2d792186f8db08fb804e8dc70545736db3a134324c957dfe4a0924dc87d06，JS近似GAE−.016257663；natural game0取笼4/4、返程237/357/反向86，首次1483向上gain−.202496、最佳右+.337502。共享场非原生时序oracle、JS近似非Critic优势，不自动回灌标签。固定first16 cap返程28/route负17/其中真实normalized GAE正9，中位−.092096；并非总体率或完整梯度证据。继续健康128、仅准备后继，不把短点改善/回落作停训门禁。


### r14 g16固定异步消费（20261019）

g16 COMPLETE的28份artifact与policy hash通过，167份封存数值源码无变化；GPU健康继续，CPU已追平并等待g32。JS Hunter难/中/易0/1/10、Python Tactical6，各64局；对父0/−1/+3与+1，对g8 0/0/+1与0。八路线death/sugar/drop/人工near/mid/far/recovery/full交付2/0/5/16/16/7/1/6，各16；对父+1/0/+1/0/0/−3/0/+1，对g8+2/0/−1/0/+1/−2/+1/0。natural窗口→目标11/11为局级共现，首oracle5/9、持续1601/2435、推进1643/4591、拾包9/16→交付6/16、恢复21/25；首控26/归因8/pop25主动14/因果kill7存活7，death4/self0。真实库存及恢复后交付仍弱，未判定稳定完成。

累计普通返程查询7339=首6126+追加1213（16.528%），最多2/异常0/delivery另4；实际current/history各1536/4096返程replay（37.5%）。额外查询候选与实际loss曝光仍分列。控制首次44484.5/残余1065（2.39%）/clawback−5985.75，route+7483.812/−18295.372/net−10811.561，capture521016仍主导。override后实际current/ring/fixed占valid tick2.54805%/7.33519%/10.35294%，原底层联赛采样占比不冒充执行来源。

固定g4首512返程输入mass(g0/4/8/16)=.442762/.617762/.678915/.675742，greedy225/315/352/351；navigation=.435269/.504378/.601993/.590890、219/259/314/310。同一固定g4 Review输入g16向右.0028184、左.994122；原错误父点已存在，平均FIFO拟合好转尚未覆盖该转弯状态。唯一seed2026101800本点real_drop追取72/82无反向、返程116/453反向186，首返程tick83左gain−.257500/右+.337502、hash84bf72f44f31bd7ee9f6ad6be982fa8b51c52e1094ccfd10fd2413e29dce2722、JS近似GAE−.0192104。natural game0库存51/57/反向4，返程86/102/反向0，首次非最优tick756仍是正进度；不把所有非oracle动作称反向。真实first16 cap返程56/route负44/其中normalized GAE正21、中位−.114947；该cap不是总体冲突率、JS近似不是Critic优势，尚缺逐次PPO minibatch/完整梯度证据。继续固定128，不热改本轮，不以短点收益或回落设门禁。


### r14 g32固定异步消费与后继准备依据（20261019）

g32 COMPLETE的28份artifact/policy hash通过，GPU已连续推进至g40，无门禁或提前停训。JS Hunter难/中/易0/4/12、Python Tactical13，各64；对父0/+2/+5及+8，对g16 0/+3/+2及+7。八路线death/sugar/drop/人工near/mid/far/recovery/full交付0/0/5/16/15/9/0/3，各16；自然交付对父−2/对前−3，真实drop+1/0。自然窗口→目标8/9为局级共现，首oracle5/7、持续922/1434、推进1005/2847、拾包6/16→交付3/16、恢复15/20；首控20/归因3/pop20主动14/因果kill3存活3，death5/self0。战斗短点改善与路线波动并存，不能宣布稳定提升，也不能据此停健康run。

普通返程查询14746=首12368+追加2378（16.126%），最多2/异常0/delivery另8；实际current/history各3072/8192 kind3非prep replay（37.5%），并非返程零曝光，追加候选逐条进入loss的数量仍未知。真实JAX full累计window45436/kill26705/stock26167/loose874/custody21317/capture20807/transfer2460，事件机会归一化、奖励分量、动态对手实际执行占比均在CURVE_ANALYSIS/POINT_G32_OWNER_REPORT。

固定g4首512返程mass(g0/4/8/16/32)=.442762/.617762/.678915/.675742/.678680，greedy225/315/352/351/355；navigation=.435269/.504378/.601993/.590890/.638819、219/259/314/310/334。固定g4失败输入g32右概率.000320752/左.999285，单状态高置信反向仍未纠正；尚无完整在线梯度证据区分分布覆盖和混合loss的作用。唯一Review seed2026101800共享场real_drop追取71/79/无反向，返程202/656/反向256，首返程tick80左gain−.157501、最佳右+.267498、hash ddd3b486a83ffe2e963860339927d198a8df0565fcef892c4b8aa7221d9b2e17，JS近似GAE−.018672454；natural game0库存66/116/反向49、返程205/239/反向25，首非oracle1591仍有正gain+.189644，不把非最优或安全拒绝一概称反向。共享场非原生时序oracle、JS近似非Critic优势，诊断输入不导入训练。

固定first16 cap返程122/route负107/其中真实normalized GAE正49、中位−.131320；导航114/负93/其中正60、中位+.117646，不作总体冲突率或Q。根因待完整曲线排序：认证查询实际可达、平均FIFO拟合改善，但固定转弯闭环不稳；覆盖/泛化及混合更新影响仍未分离。保留r15同配方后备，不同时加loss、改混合或改奖励；新数值候选须证据支持并TDD，且只在固定终态安装。图je10_r14_curve_20261019.png已含g0/4/8/16/32全部完成点。


g32额外只读learner-state覆盖审计：FIFO valid9954，其中kind3非preparation3630，携敌包3533、携己包97、空手0。代码je10_route_oracle.labels以任何实际carried>=0归为kind3，global_vec[13]表示任意携包、[14]表示敌包；此前97条非敌包位不代表空手或无事件初始化。原×2和replay计数类同时含真实己包回收与delivery，不能冒充纯敌包返程曝光；保留既有配方，只补清统计口径。FIFO_CUSTODY_STRATA_G32绑定原state hash，训练初始化污染仍0。

同一g32固定real_drop首返程偏航附近有9条同wall/brick/pushable、同笼与同携包身份、曼哈顿距离<=1的FIFO行；其方向标签总质量下2/右7/左0，当前greedy仅1/9在集内、平均move target mass .187614。固定JS输入本身左.999483/右.00015933，exact Actor输入匹配0；其他威胁、物品、时序/global特征仍不同，9条近邻不构成精确oracle冲突证据。natural固定偏航附近同几何身份样本0，不等于训练从未访问。REVIEW_G32_FIFO_COVERAGE保留最近样本、round、地形差异和全局差异；此处是有路后的方向拟合/泛化问题，不能回到真实不可达/预开路解释。×2与总体FIFO均值不能单独证明硬状态已拟合；完整曲线后再排序覆盖与混合更新假设，当前健康r14继续。


### r14 g64完整固定点与detour诊断候选（20261019）

g64 COMPLETE的28份artifact/policy hash通过；八卡健康、167份封存源码一致、初始化与无custody监督污染0。JS Hunter难/中/易0/4/2、Python Tactical12，各64局；对父0/+2/−5与+7，对g32 0/0/−10与−1。八路线death/sugar/drop/人工near/mid/far/recovery/full交付0/0/8/16/15/10/1/1，各16；真实drop对父+4/前+3，natural对父−4/前−2，人工carry不替代真实闭环。自然窗口→目标9/10为局级共现，首oracle4/5、持续607/792、推进621/2423、拾包5/16→交付1/16、恢复17/19；首控20/归因4/pop20主动13/因果kill4存活4，death4/self0。未见硬安全恶化证据，能力保持与交付波动只诊断，固定128继续。

普通返程查询30471=首25418+追加5053（16.583%），最多2/异常0/delivery另22；实际current/history各6144/16384 kind3非prep replay（37.5%），含真实己包回收和delivery、含放回/回退重复，不是纯敌包/唯一状态/每次PPO复用分母。g64 FIFO kind3非prep3816=敌包3688+己包128，空手0、exact-input联合标签冲突0。控制首次169615.5/残余3628.5（2.139%）/clawback−32263.75；route+26739.643/−73764.328/net−47024.685，capture2086824仍主导。override后current/ring/fixed实际valid tick占2.11290%/9.27333%/13.47771%，原底层联赛分母独立。

固定g4首512返程mass g0/4/8/16/32/64=.442762/.617762/.678915/.675742/.678680/.643056，greedy225/315/352/351/355/335；navigation=.435269/.504378/.601993/.590890/.638819/.654922，219/259/314/310/334/342。固定g4 Review输入g64右.00998065/左.945488，仍反向。唯一seed real_drop共享场追取68/79无反向、返程92/179反向74，首返程tick80左gain−.157501、右+.267498、hash ddd3b486a83ffe2e963860339927d198a8df0565fcef892c4b8aa7221d9b2e17，JS近似GAE−.01908177；natural库存4/4、返程122/154反向23，首2255向上gain−.202499、最佳下+.157501。仅共享场/局部危险诊断，非原生时序oracle或动作Q。真实first16 cap返程248/route负226/其中normalized GAE正100，中位−.107945，非总体率或完整在线梯度证据。

g64固定掉包转弯附近14条同地形/笼/携包身份一格内原生FIFO样本，标签下5/右9/左0，当前greedy4/14、平均move target mass .325477；exact Actor输入匹配0，威胁/global仍不同，不能据此称精确标签冲突。natural附近该描述性队列0，不等于从未访问。下一只读诊断预声明：量化原生认证动作集全部要求暂时远离己笼的detour状态是否稀少/低拟合，优先区别硬状态曝光与平均拟合。新增独立未安装分类器，RED2项真实行为失败→4项GREEN（RETURN_DETOUR_CLASS_RED/GREEN），仅分类已有合法原生标签，不生成标签、不接入当前训练、不替换r15后备；若分布证据不支持，则不改返程采样。DETOUR_HYPOTHESIS_DRAFT标明诊断假设，数值改动须进一步proposal/TDD并仅在固定终态安装。


g32/g64原生detour分层审计已完成并否定总体稀缺解释：合法kind3非prep FIFO内，所有原生认证可接受动作都需暂时远离己笼的detour分别1664/3630（45.840%）与1824/3816（47.799%），其联合目标mass .562530/.564185，非detour .543190/.506586；敌包detour1652/1811、己包12/13。因而不实施“将detour返程配额提高到50%”这一缺乏区分力的候选。DETOUR_HYPOTHESIS_DRAFT改为诊断完成、总体稀缺假设不支持，r15原预声明未替换。分类器4项GREEN只证明分层语义，不是能力/admission门禁，也未安装至任何训练入口；167份当前数值封存代码不变。仍存在特定转弯近邻低拟合与自然完整闭环不稳，继续固定后续点以区分条件状态覆盖、拟合保持与混合更新，不从单一均值作因果结论。JS受控fixture与JAX课程的时限归一化按各自声明horizon，跨simulator分布仍需明确区分；没有验证其为根因，未改编码或固定评估协议。


### r14唯一Review实际位移与晚取包预算复核（20261019）

只重放既有、经唯一生产Review协议校验的固定seed2026101800 game0，不开新局、不调用模型、不选择其他seed。FIXED_REVIEW_ACTUAL_MOTION_G4/G8/G16/G32/G64区分“动作候选gain”和原生step实际位移；此前共享场反向计数是候选动作负gain，不能一概叫逐帧实际反向率。已核对的真实drop首次返程偏离在g4/8/16/32/64均实际左移.18格、拓扑未变，实际势能gain分别−.257496/−.157497/−.257496/−.157497/−.157497，故该首偏航不是侧滑转向假象；取包剩余1117–1120tick，当前场/速度返程估算183–185tick，时间充足，核心是有路后的首方向选择与保持。

固定natural game0各点g0/4无拾包；g8/16/32/64首次控制1310/424/621/2083，首次库存目标1311/425/622/2084，真实拾包1476/755/1589/2246。g8/16/32取包后剩924/1645/811tick，按当前场/速度估算201/196/200tick；g64剩154tick、距离35.9575、原生有效步长.18、估算200tick、余量−46。g64取包后第一步实际向右正推进+.157505，首返程偏离2255实际向上−.202499，当时剩145/估算190tick。它同时体现晚机会/晚取包预算与随后方向偏离，不能把该game0失败全归为首返程动作。g16与g32首次非oracle动作实际仍是正推进+.022503/+.189644，不把非最优/安全拒绝自动叫反向。

时间估算只用当时场和当前速度，忽略转弯/碰撞/危险等待及未来速度/拓扑变化，既非严格物理最短时间证明，也非无路结论；真实取包后实际入站可逆不变量维持。单个固定game0不是16局总体失败分解，不外推总体晚取包比例。未改奖励、编码、oracle或本轮训练；继续健康128和固定g96/128评估。


### r14持续监督与r15返程误差replay预声明（20261019）

已重新实际读取dirty diff、无预置携包测试、CAUSAL_INIT_RED与r11 EXCLUDED_INVALID；原污染修复已GREEN并由r12从r10 g128干净回退，r13/r14连续执行。r11 g18排除与旧trainer/evaluator/八卡释放receipt保留；固定return replay仅保留hash绑定原生drop/pickup证明的1062+996行，无证明拒绝。旧“真实取包后无路／动态封路／预开返程通道”结论持续撤回；当前r14 167份数值源码逐hash一致、无热改，退休scene及无native custody返程监督累计0。原smoke未重启。

r15正式替换同配方后备为单因素候选：只在现有valid kind3非preparation返程replay槽内，以0.05+1−当前raw Actor原生合法联合目标概率质量抽样。其他阶段逐行原抽样、current/history及阶段槽数、FIFO容量/additions/schema、query/原生标签、返程×2、reward/PPO/网络/输入/场景混合均保留；当前fit只影响抽样，stop_gradient，不把JS固定Review或人工carry诊断输入训练。kind3同时包括真实己包回收/delivery，不冒充纯敌包曝光；额外Actor前向的技术开销由新命名空间smoke核验。所有有效返程保持正支持，相同误差时原随机实现逐行不变。

依据g32/g64固定掉包偏航同地形近邻分别9条/14条、greedy1/9及4/14、move mass .187614/.325477；固定首转弯在g4/8/16/32/64实际反向，而历史固定512平均返程拟合改善，说明均值掩盖具体状态弱拟合。广义detour已有45.84%/47.80%且拟合略高，否定“总体detour稀少”，不做detour配额。误差抽样是对条件状态拟合假设的最小检验，尚不证明根因或能力因果；没有同父控制，完整曲线仍需后续点。

TDD实际行为RED（难样本原占.5而非期望>.85；安装/继承接线3项失败）→扩展47项GREEN，另真实collector→GAE→候选FIFO聚合→PPO Actor/Critic非零→保存恢复1项GREEN。扩展恢复测试先暴露NumPy直调.at错误，已改为与实际JIT恢复入口一致并验证；新评分字段不入FIFO或loss。安装器仅允许正常g128及8卡释放后改未来worker dispatch；当前r14完全不变。NEXT_RECIPE_PROPOSAL与SUCCESSOR_PREDECLARATION同步绑定6份source/test hash。

r15 ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r15-clean-causal-init`；namespace `control_route_v38_r15_causal_clean`，父r14 g128，train seed1955372956、technical1673720716，fresh Adam/EMA=raw/RNG/env/FIFO/state/cache/eval，8卡128代／268435456 fresh ticks。场景combat .35/bridge .20/natural full .35/合法其他 .10。r14终态先核验8卡释放即启动该后继最小技术smoke，save/restore finite/真实非零后立即正式训练，不等待g128 CPU评估、网页/报告或能力阈值。r15 g0/4/8/16/32/64/96/128由唯一CPU/JS owner按COMPLETE消费，Hunter难/中/易各64、Python Tactical64、原固定路线16seed，JAX bot独立；额外关注实际返程抽样mass/error与固定同输入拟合、首次偏航和自然交付。中间结果只准备下一proposal，健康128不提前停。固定终态先后继再报告，后继本身预声明同配方连续接续，保持8卡训练主线。


### r14 g96固定消费、拟合保持与后继证据（20261019）

g96 COMPLETE的28份产物与policy hash核验通过；唯一CPU/JS owner等待最终g128，GPU继续g105以上。JS Hunter难/中/易0/3/14、Python Tactical8，各64；对父0/+1/+7与+3，对g64 0/−1/+12与−4。Hunter self0/0/0、Tactical self1/death1，自然death2/self0，分simulator记录，不以短点改变固定预算。八路线death/sugar/drop/人工near/mid/far/recovery/full交付2/0/8/16/16/9/2/3，各16；drop对父+4/前0、natural对父−2/前+2。自然窗口→目标10/10为局级共现，首oracle3/6、持续869/1405、推进916/3902、拾包5/16→交付3/16、恢复22/24；首控26/归因11/pop25主动19/因果kill11存活11。完整父/前点机会归一化delta封存POINT_G96_OWNER_REPORT/CURVE_ANALYSIS，不把不同机会分母混算。

普通返程查询48852=首40313+追加8539（17.479%），最多2、异常0、delivery另40；实际current/history各9216/24576 kind3非prep replay（37.5%），含真实己包回收和delivery及放回/回退重复，追加候选未逐条追到loss。FIFO kind3非prep4713=敌包4558+己包155、空手0；exact-input no-common-pair冲突0。native full累计window128127/kill80145/stock82839/loose3692/custody65503/capture64477/transfer9033，按full tick机会归一化见曲线。控制entry252822.25/residual5307.25（2.099%）/clawback−50635.5；route+39553.493/−111229.588/net−71676.095，capture3110712仍主导。override后current/ring/fixed实际valid tick占1.94429%/9.52659%/14.13406%，其他规则/弱课程单列，底层联赛条件分母独立。

固定g4首512返程mass g0/4/8/16/32/64/96=.442762/.617762/.678915/.675742/.678680/.643056/.611065，greedy225/315/352/351/355/335/319；同输入navigation .435269/.504378/.601993/.590890/.638819/.654922/.663786、greedy219/259/314/310/334/342/342。返程拟合后段回落与导航继续提高并存；这是固定历史训练FIFO描述，不是全分布遗忘率或完整gradient证据。固定g4 Review输入g96左.995629、右.00044005，仍高置信偏离。当前real_drop首返程偏航近邻13条，同地形/笼/携包身份一格内，标签下4/右9/左0，greedy5/13、mean move mass .398047；exact Actor输入匹配0、威胁/global不同，不称精确标签冲突。natural近邻0不证明从未访问。

唯一固定seed2026101800 Review：real_drop追取共享场65/79且无负gain动作，返程162/454、负候选gain170；首返程tick80左gain−.157501对最佳右+.267498，state hash ddd3b486a83ffe2e963860339927d198a8df0565fcef892c4b8aa7221d9b2e17，JS近似GAE−.018381707。natural game0已真实取包并于1797tick完成交付；返程156/168共享场一致、首1602虽非最佳却gain+.128929，不能叫反向失败，更不能据game0概括16局。真实first16 cap返程387、route负347中normalized GAE正149、中位−.074944；该cap非总体率、JS近似GAE非Critic优势、共享场非原生时序oracle、没有完整在线梯度归因。

g96未推翻已预声明r15最小假设：特定原生返程状态的拟合/保持不足比广义detour稀少更有证据；仅改现有返程replay槽内误差权重，保留×2及其他配方。自然闭环依旧波动，战斗保持与条件泛化待完整128曲线，继续固定终态再接续，不设能力门禁。


g96后继误差抽样只读复核：在同一COMPLETE原生历史FIFO、固定seed、8个device各2048次描述性抽样中，基线与候选返程槽均6144，其mean合法joint mass从.494632变为.217654、mass<.25抽样2405→4243，敌包5938→5958；其他阶段逐行exact、阶段槽数exact。各device返程唯一行基线420/424/428/453/426/432/433/424，候选358/367/359/359/340/359/349/350，说明优先低拟合伴随预期重复增加。此为保存FIFO上的前瞻重抽样，无模型更新、新对局、JS输入、query/label生成，不是实际训练曝光或能力改善证据；current新增子集分布不同，正式run必须记录真实mass/error及交付/保留曲线。正epsilon保持全部合法返程支持，不因诊断指标设门禁。

g96 Review实际位移补充：real_drop首返程确实左移.18、actual gain−.157497、拓扑不变、剩1120tick/当前场稳速估算184tick；natural首取包后1598向右actual gain+.157501，首非oracle动作1602虽命令下但原生steering实际仍右移.18、actual gain+.316071，剩798tick/估算195tick，最终1797交付。动作候选gain+.128929与实际运动gain不同，均不能把该已成功自然game0描述成返程反向失败；只描述固定seed，不外推16局。


自然原生完整对局的独立机会分解（仅COMPLETE、固定16局/点）：

| g |有控制/死亡窗口局|有目标局|真实拾包局|拾包而无交付局|交付局|
|---|---:|---:|---:|---:|---:|
|0|14|13|10|5|5|
|4|10|10|5|4|1|
|8|13|13|12|6|6|
|16|11|11|9|3|6|
|32|9|8|6|3|3|
|64|10|9|5|4|1|
|96|10|10|5|2|3|

窗口/目标是局级共现、不是逐事件因果转化。g96 stock阶段first5/9、持续233/540、target3018tick；return阶段first17/24、持续636/865、target884tick，first计每次目标重入且会跨威胁切换，不能混成局级首动作3/6。JS observable库存目标不含JAX原生preparation/时序安全oracle，低oracle机会覆盖不是“模型有路却无标签”的直接证明。后段窗口及拾包机会变化也影响交付分母，返程动作一致率改善不能代替闭环改善。该表不含人工carrier或传送库存fixture，亦不支持真实取包后无路；r15保留返程最小因素而不同时改战斗/窗口/混合，完整终态后据证据排序继续最弱漏斗。


### r14固定终态与r15实际接续（20261019）

r14正常跑满128代／268435456 fresh ticks，compute=6018.821s、wall=6897.809s；全128代finite、Actor/Critic真实非零，退休scene与无原生custody返程监督0。GPU_RELEASE_RECEIPT核验全部8卡且compute_clients=0后，supervisor立即启动r15并占用8卡执行新命名空间technical_smoke01；未等待g128异步CPU/JS结果或报告，未复用/重启已终止旧smoke。r14最终checkpoint独立CPU恢复通过、167份保存源码snapshot hash通过，完整state/next RNG digest通过。r15正式非零确认尚待smoke与fresh正式首代，不能把smoke当正式能力或更新证据。

r15实际plan已安装误差抽样receipt，172份数值源码封存且当前hash无差异；父r14 g128、train1955372956/technical1673720716、reset混合.35/.20/.35/.10与预声明一致，保留合法返程×2。本轮仅在原有合法返程replay槽内按当前Actor误差抽样；fresh state/cache/eval，技术smoke后立即正式128代，无能力门禁。唯一CPU/JS owner仍完成r14 g128；后继eval在全局锁等待后跟随固定点，不影响GPU。

连续后备r16已在r15启动smoke前具体预声明：ID `JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r16-clean-causal-init`、namespace `control_route_v38_r16_causal_clean`，父r15 g128，train seed1235718504、technical1212388，8卡128代／268435456 fresh ticks，保留r15误差抽样、query diversity与返程×2，fresh全部learner状态/cache/eval，混合与固定评估协议继承；g0/4/8/16/32/64/96/128、唯一CPU/JS owner、COMPLETE消费、Python/JS/JAX分列、先后继再报告。若完整证据支持更小改动，可在r15终态前以实际proposal/TDD更新；无论能力短点如何，健康r15不提前停止，最终CPU结果不控制接续。

固定run `runs/je10_bun_20261018/control_route_v38_r14_causal_clean` 完成128代/268,435,456 ticks；compute=6018.821s、wall=6897.809s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r15_causal_clean`；本次后继配方选择依据 `Native return supervision is reachable and averaged historical fit improves, but specific same-terrain turn neighborhoods remain poorly fitted and fixed real-drop first-turn reversal persists. Broad detour rarity is contradicted by native FIFO evidence. Test current-fit prioritization within existing return replay slots.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。


### r15正式非零确认后：r14完整曲线、根因排序与已执行最小后继（20261019）

r15新命名空间技术smoke finite/Actor-Critic真实非零/保存恢复通过后已立即启动8卡正式128代；正式g1/2097152 fresh ticks原子checkpoint与policy hash核验通过，8副本exact，Actor L2=3.2859482765、Critic L2=2.2746827602。独立formal g0审计：父r14 g128 raw Actor/Critic/full pool及retention reference精确一致，EMA=raw，Adam0/FIFO空/counter0、新RNG seed1955372956、8192 learner空手、退休scene0、resume=null、独立cache；旧smoke未重启。r14终态已先核验8卡compute_clients=0并交给该后继，现在才综合最终报告。r15继续健康固定128，唯一CPU/JS owner按预声明点异步跟随，无能力门禁。

各固定点全部COMPLETE，28份artifact/policy hash逐点验证；JS与Python各64局，路线各16局：

|g|JS Hunter难/中/易安全击杀|Python Tactical安全击杀|真实掉包fixture交付|自然真实拾包局|自然完整交付|
|---|---|---:|---:|---:|---:|
|0|0/2/7|5|4|10|5|
|4|0/2/11|11|4|5|1|
|8|0/1/9|6|6|12|6|
|16|0/1/10|6|5|9|6|
|32|0/4/12|13|5|6|3|
|64|0/4/2|12|8|5|1|
|96|0/3/14|8|8|5|3|
|128|0/3/9|6|6|3|0|

最终JS Hunter对父0/+1/+2、对g96 0/0/−5；Python Tactical对父+1/对前−2。真实掉包交付对父+2/对前−2；自然交付对父−5/对前−3，曲线5/1/6/6/3/1/3/0不能称稳定提升。最终八路线death/sugar/drop/人工near/mid/far/recovery/full交付1/1/6/16/16/11/0/0；前三类库存/掉包/recovery为受控fixture中的原生真实拾取，不能证明从己笼真实入站；near/mid/far人工预携包仅诊断，全部training_eligible=false/proves_post_pickup_unreachability=false。只有natural_full为完整原生因果入站链；不存在真实取包后不可达/动态封路/预开返程通道的解释。跨地图仍只有Bun806及开墙变体，额外地图泛化未证明。

最终自然控制首次34/归因12→pop34（主动25）→因果kill12/存活12；不能把含fixture初始控制的raw pop除以新归因数当转换率。合法窗口→目标11/11为局级共现，首oracle0/5、持续337/559、推进364/4433、拾包3/16→返程阶段351tick/79 oracle机会→交付0/16，恢复31/31；death6/self0。库存阶段target4082tick、oracle265/480、阶段首1/6；返程阶段72/79、阶段首1/1。自然最后的损失同时发生在拾包之前及拾包之后，不能只用返程匹配率解释。唯一固定game0没有真实拾包、只有103库存目标tick且共享场oracle机会0；不另选成功/失败seed伪造返程轨迹。JS observable库存目标没有JAX preparation与时序安全证明，不能把其机会少直接称训练oracle漏标或真实无路。

证据排序①条件返程拟合与保持不足：固定g4首512返程mass g0/4/8/16/32/64/96/128=.442762/.617762/.678915/.675742/.678680/.643056/.611065/.548677，greedy225/315/352/351/355/335/319/279；同输入navigation .435269/.504378/.601993/.590890/.638819/.654922/.663786/.631131、greedy219/259/314/310/334/342/342/331。早期改善未稳定保留，且具体首转弯从父点就有高置信反向；不是“新增query未可达”。最终普通返程query76123=首61143+追加14980（19.679%），最多2/配额异常0、delivery另53；实际current/history各12288/32768 kind3非prep replay（37.5%），含己包回收/delivery、重复与fallback，追加候选不等于逐条进loss。最终FIFO返程7051=敌包6880+己包171/空手0，move/joint目标mass .431039、greedy2973；exact Actor输入无共同joint pair冲突0，不等于不存在隐藏条件分布问题。

最终固定掉包首返程附近17条同地形/笼/携包身份一格内原生FIFO，标签下4/右13/左0，greedy4/17、move mass .288416；exact Actor输入匹配0，威胁/global不同，不称精确标签冲突。固定g4 Review同输入最终左.953973/右.00163410；当前g128首返程输入左.976280/右.00669315。该固定真实掉包轨迹在取包后有路、剩余时间充足，模型仍反向，直接支持优先检验低拟合状态的抽样。广义detour原本45.84%/47.80%、拟合略高，已反证“总体detour太少”，不加该类配额。

证据排序②上游控制窗口与库存进入不稳：自然有窗口局父14/16、g96 10/16、最终11/16，而真实拾包10→5→3，说明交付恶化还涉及取包前选择/推进及机会时序；单个g64 game0很晚取包有当前场稳速预算不足，g96 game0则取包1598、交付1797，不能将其外推为所有局唯一根因。最终game0未拾包。后继只改返程replay，暂不同时改窗口/混合/导航loss；r15完整曲线若返程拟合改善而自然取包仍弱，将以原生learner-state库存/准备首次偏离覆盖优先诊断后继，不预设能力停训条件。

证据排序③奖励与优势混合仅有局部迹象：最终first16 cap累计返程586、route负533中真实normalized GAE正217、中位−.118238；navigation370、负273中正170、中位+.124756；stock167、负82中正31。该cap非总体冲突率、不是动作Q/完整在线梯度，不能仅据此调整奖励或断言KL为首因。累计control entry335150.5/residual6987.25（相对entry2.085%）/clawback−71085，route+52189.966/−150633.747/net−98443.779，终局capture4065816仍主导。首次控制信号保留主信用，残余爆破较小；未改本轮奖励。JAX full累计window169564/kill106944/stock110961/loose5296/custody87656/capture84401/transfer12621，每百万full tick分别913.292/576.013/597.649/28.525/472.126/454.594/67.978，严格独立于JS/Python胜率。

唯一Review seed2026101800最终real_drop追取77/81/负候选gain0，返程225/391/负候选gain146；首返程82左gain−.157501对最佳右+.267498、距离32.988750，hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd。原生实际左移.18、拓扑不变（ACTUAL_MOTION独立核验）。JS近似reward-GAE−.018281603、前−.018502951、变化+.000221348；近似重建route分量+.000176953不是当前训练共享场reward，不能从其符号判断训练方向奖励。真实collector Critic优势与JS零value近似必须分列。

战斗/安全保持：最终Hunter难/中/易death1/0/1、self0/0/0；Tactical death1/self1、natural death6/self0。战斗各固定点波动且hard均0，能力未完成；不以这些短点中断r15。动态联赛override后current/ring/fixed实际valid tick占2.11665%/9.67869%/14.20104%；底层sampled league条件占8.25139%/37.24807%/54.50053%，分母不同，rule/弱课程/assist与固定g0/stage1在原始执行曝光单列。

已执行的最小单轮检验为r15原生普通返程槽内误差抽样：0.05+1−当前raw Actor合法joint mass，stop_gradient、正支持，同误差完全保留原抽样；其他阶段/槽数、FIFO、query、原生标签、×2及reward/PPO/混合不变。47项focused/相关与1项真实collector→GAE→FIFO→PPO→恢复GREEN，smoke及正式g1均finite/Actor-Critic非零，未等待最终CPU分数。没有同父控制，属于顺序后继假设验证，不将相关变化当单变量能力因果。r15固定g0/4/8/16/32/64/96/128独立评估，继续按真实抽样mass/error、固定同输入拟合、首次偏离、自然取包/交付、战斗/安全与实际对手暴露共同诊断；r16同配方后备已具体预声明，健康run不因短点停训，固定终态先后继再报告。


### r15 g0父点复现与固定输入跨run对照（20261019）

r15 g0 COMPLETE的28份产物和policy hash核验通过；原生策略树结构/shape/dtype/叶子字节与r14 g128完全一致，Hunter、Tactical、全部路线episode与控制漏斗逐项exact相同。两个policy文件重序列化后文件字节hash不同，不能据此推断权重不同；已用语义digest和独立fresh审计核实，而非修改任何训练内容。G0_PARENT_FIXED_PROTOCOL_PARITY保留两种口径。基线仍JS Hunter0/3/9、Python Tactical6（各64），natural真实拾包3/16→交付0/16，real_drop交付6/16。

额外只读跨run对照固定沿用r14 g4首512返程/导航FIFO和已验证Review输入；r15 g0复现其父g128的target mass .548677/.631131、greedy279/331，以及同Review左.953973/右.00163410。参考数据hash固定，不随r15 learner分布变化；仅分析，不导入训练或新增模型对局。r15自身snapshot的FIFO fit与该固定历史fit分列，前者不能单独证明遗忘。r15 g4已提交并由唯一CPU/JS owner独立评估，健康8卡继续128，不设能力阈值。


### r15 g4固定异步消费与条件输入审计（20261019）

g4 COMPLETE的28份产物/policy hash通过，唯一CPU/JS owner已转向g8，GPU继续g18以上。JS Hunter难/中/易0/3/2、Python Tactical7，各64；对父/前点0/0/−7与+1。八路线death/sugar/drop/人工near/mid/far/recovery/full交付2/2/1/16/16/8/0/0，各16；真实掉包对父−5、natural0不变。natural窗口→目标10/10（局级共现）、首oracle3/4、持续576/1013、推进657/3760、拾包3/16→交付0/16、恢复27/28；首控31/归因8/pop31主动22/因果kill8存活8，death8/self0。Hunter self0/3/0，Tactical self0。早点未见闭环改善，完整128照常完成；不以该点作为能力门禁或停止依据。

普通返程query1381=首1154+追加227（16.437%），max2/异常0、delivery0；实际current/history各384/1024 kind3非prep replay（37.5%）。新抽样真实pre-update Actor平均选中合法joint mass分别.213835/.363183、error .786165/.636817，history含g1 current回退；低mass是优先难样本的输入分布，非post-update能力或损失结果。current FIFO普通return1381=敌包1379+己包2、空手0，当前joint mass .668462、greedy920/1381，exact-input no-common-pair冲突0。JAX full累计window5704/kill2175/stock2052/loose3/custody1572/capture589/transfer48，机会归一化见CURVE_ANALYSIS。entry9286/residual240.25/clawback−768，route+2018.586/−2865.951，terminal capture114048仍主导。override后current/ring/fixed实际valid tick占3.21525%/2.88422%/13.42175%，其余源独立单列。

固定父r14 g4首512同输入返程mass由r15 g0 .548677→g4 .600665、greedy279→316；父导航.631131→.537425、331→275，说明保留需要同时看。固定r15自身g4队列跨Actor g0→g4返程.414950→.688676、209→350，navigation.451340→.537596、232→280；两个固定数据集不能混成同一分母，snapshot均值与历史同输入分列。固定父Review输入右.001634→.005317、左.953973→.983875，仍反向。当前首返程Review输入左.963460/右.013946，与两条同地形/笼/携包身份一格内原生FIFO的右.999787/.962032、均标签右形成条件泛化差异；它们位置分别(6.039999,9.179996)/(6.379898,9.5)，而Review为(5.89,9.82)，跨当前格且其他危险/时间/respawn不同，不能称exact输入覆盖、标签冲突或单一编码bug。

只读feature sensitivity固定检查这两条（不挑seed、不新开模型对局）：review obs+native全部global仍左.978599/.972470；native全部obs+review global则右.999958/.812137。单换时间平面为第一条native值可把右.013946→.721092，换为第二条值仅→.023884；同时换global0为第一条→.730091，第二条→.024132。实际差异含位置splat、己泡/危险图、时间平面、携包位置及enemy respawn，global其余仅极小speed浮点差；时间敏感存在，但没有“时间唯一根因”的证据。hybrid输入未必对应合法状态，没有oracle/action outcome，不入训练，不当Q或根因因果。JS时间特征取t/maxSteps，JAX取t/LESSON_MAX_STEPS[state.lesson]，课程HORIZONS另行终止；二者的预算语义是否匹配尚需按原生scene/lesson/实际horizon核验，不能由hybrid敏感性直接宣称编码bug。当前172份数值源码不变。继续g8/16/32等固定点核对相同当前格/更细位置与条件覆盖，再决定后继最小改动。

唯一Review固定seed2026101800真实掉包追取70/81共享场一致、负候选gain0，返程107/364、负候选gain184；首返程82左gain−.157501对右+.267498、hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd、JS近似GAE−.018418630。actual motion左.18/−.157497、拓扑不变，剩1118tick/当前场稳速估算184tick；真实有路后偏航，不能回到不可达或预开路解释。natural固定game0未拾包，仅205库存目标tick且共享场oracle0，不选其他seed替代。真实first16 cap返程4/route负4其中normalized GAE正2、中位+.227431；navigation25/负23中正13，中位+.133458，样本cap非常小，非总体冲突率或完整gradient。JS近似与Critic-backed优势分列。r15健康继续128，r16同配方后备维持，尚未用早点修改后继数值配方。


### r15 g8固定消费、原生转弯同输入拟合与时间字段审计（20261019）

g8 COMPLETE全部28份产物/hash通过，CPU唯一owner正在g16，GPU固定128继续。JS Hunter难/中/易1/1/7、Python Tactical10（各64），对父+1/−2/−2、+4，对g4 +1/−2/+5、+3；Hunter self均0，Tactical death1/self1。八路线death/sugar/drop/人工near/mid/far/recovery/full交付0/0/3/16/14/8/1/4（各16），真实掉包对父−3/对前+2、natural交付对父与前均+4，但仅为一个中间固定点，尚无稳定改善结论。natural窗口→目标14/14局级共现、首oracle6/10（前3/4）、持续2006/3214、推进2050/6180、拾包10/16→交付4/16（父与前均3→0）、恢复31/37；首次控制42/归因9→pop40主动32→因果kill9/存活9，death4/self0。fixture raw pop不作归因转换率；人工预携包仅诊断，不能推断真实取包后无路。仅Bun806及开墙变体，未证明额外地图泛化。

真实普通返程query4347=首3480+追加867（19.945%），max2/异常0，delivery另2；current/history各768/2048 kind3非prep replay（37.5%），pre-update所选合法joint mass .200903/.307704，error .799097/.692296，低mass是难样本抽样分布，不是post-update拟合下降。当前FIFO普通return4349=敌包4300+己包49、空手0，joint mass .583850、greedy2585；exact Actor输入无共同pair冲突0。JAX full累计window12050/kill6143/stock5764/loose108/custody4761/capture3821/transfer392；每百万full tick为1117.158/569.519/534.382/10.013/441.393/354.246/36.342，与JS/Python严格分列。控制entry21415.75/residual525.5/clawback−2441.25、route+3815.348/−8250.753、capture248616，终局仍主导。override后current/ring/fixed实际valid tick占2.76797%/4.46368%/12.29090%；sampled league条件14.35098%/23.33521%/62.31381%，分母不同。

父固定首512返程mass g0/g4/g8=.548677/.600665/.596149、greedy279/316/317；父navigation .631131/.537425/.570570、331/275/303，保留仍弱于父。本轮固定g4返程.414950/.688676/.755108、209/350/403，navigation .451340/.537596/.588384、232/280/308。两个队列固定hash、同输入跨Actor，不将动态snapshot均值下降误称遗忘。固定父Review g8仍左.998160/右.000173334；当前g8固定seed2026101800首返程83、pos(5.72,9.82)，左.998091/右.000135342、hash 84bf72f44f31bd7ee9f6ad6be982fa8b51c52e1094ccfd10fd2413e29dce2722。附近6条原生query标签下2/右4/左0、greedy3/6、mass.539457；exact输入0，同当前格2，L1<=.25格0，最近.640004格，不能称该首转弯精确覆盖。

新增固定原生anchor审计冻结g4全部两条近邻原始obs/global/legal joint标签并逐行hash：index2374位置(6.039999,9.179996)的右mass g0/g4/g8=.999810/.999801/.999166；index8539位置(6.379898,9.5)为.040959/.965172/.361777，greedy左→右→左。后一条明确存在相同原生训练输入的拟合回落，前一条保持；不是仅JS/native输入错配，也不能据两条推断总体遗忘率或是哪项梯度负责。此固定两行batch的bf16推理数值与另一审计不同batch略有差异，跨Actor均使用同一入口/shape，不能混为逐bit不一致。只读helper初次遇NumPy/tracer索引错误已修为JAX数组并成功重算，训练源码/状态未动。

时间字段只读核验formal g0：scene0/1/5/7/8/9实际horizon300/600/900/1200/2400/1200，但全部lesson0，make_obs ch6和global0分母均LESSON_MAX_STEPS[0]=2400；JS两字段按t/maxSteps，固定掉包1200、自然完整2400。该具体字段与终止预算差异已定位，但尚未证明其预期契约、是否决定偏航，不能据混合输入敏感性宣称唯一根因。后继r16仍为已预声明同配方后备；不热改r15数值源码。后续继续用原生合法状态及同输入拟合证明覆盖/保持与预算语义，若支持新因素再proposal→RED→GREEN，只于正常128/释放后安装。

唯一Review共享场真实掉包追取65/82（负候选gain0）、返程123/449（负候选gain207），首83左gain−.257500而右+.337502，JS近似reward-GAE−.014027732；真实位移核验与拓扑证据见FIXED_REVIEW_ACTUAL_MOTION_G8。natural固定game0无可用目标阶段，不改选seed。真实collector first16 cap累计返程26/route负25其中normalized GAE正14、中位+.210912，navigation32/负28中正17、中位+.137356；cap非总体率/动作Q/梯度归因。真实入站可逆不变量维持，旧“真实取包无路/动态封路/预开返程”解释已撤回。r15继续128，终态先核验八卡释放并立即r16，正式非零后主动报告；CPU最终不阻塞接续。


### r15 g16完整消费及固定转弯保持（20261019）

g16 COMPLETE/policy/28产物hash通过，g32已交唯一CPU/JS owner，8卡健康继续128，172份数值源seal逐hash无变化。JS Hunter难/中/易2/0/3、Python Tactical17（各64），对父+2/−3/−6、+11，对g8 +1/−1/−4、+7；Hunter death0/1/1、self全0，Tactical death0/self0。八路线交付death/sugar/drop/人工near/mid/far/recovery/full=4/0/6/16/16/10/2/2，各16。真实掉包对父0/对前+3，natural对父+2/对前−2；natural拾包6/16（父3、前10）。窗口→目标12/12局级共现，首5/7、持续1089/1874、推进1222/4796、拾6→交2，恢复30/33；首控36/归因10/pop35主动29/因果kill10存活10，death4/self0。库存oracle677/1350、首10/14，返程412/524、首5/7，不能把分阶段多次首动作与局级首混为同一分母。

普通返程query9138=首7360+追加1778（19.457%），max2/异常0、delivery另4；current/history各1536/4096 kind3非prep replay（37.5%），平均选中pre-update target mass .202394/.285899，仍优先低拟合状态。当前FIFO返程4793、joint mass .615869、greedy3005，exact-input无共同pair冲突0。父固定512返程mass g0/4/8/16=.548677/.600665/.596149/.623175、greedy279/316/317/325；父navigation .631131/.537425/.570570/.586221、331/275/303/310。本轮固定g4返程.414950/.688676/.755108/.773837、209/350/403/408；navigation .451340/.537596/.588384/.573650、232/280/308/296。整体固定返程改善但导航保留未恢复父，不以单点当因果能力结论。

冻结g4的两条原生转弯输入逐行hash不变：index2374 g16右mass .997787、仍正确；index8539 g16右.216729/左.773121，比g4 .965172与g8 .361777继续回落。父固定Review仍左.972664/右.014557；当前g16固定seed2026101800返程首82、pos(5.94,9.82)，左.957669/右.035761，hash 23d8dfc2e86552ef6bc3b0bec118be0d89637944bea31aacac48880d0afcc7f0。近邻原生16条、标签下4/右12/左0、greedy8/16、mass.514971；exact Actor输入0、相同当前格10条、.25格内1条/最近L1 .219898。最近原生index4621在(6.059900,9.919998)，round11、原生标签右，但g16右.336780/左.636660；与Review的敌人alive/hp/respawn、危险/泡/时间不同，不能称精确冲突。

只读输入敏感性表明该近邻native obs+review global右.960128，而review obs+native global右.339966；单替换review敌人位置ch2右.975013、同时替换两个time特征仅.266696。混合特征未必合法state/无动作结果，说明不能把时间或距离单独定为根因；r16仍维持同配方后备，后续用真实输入/梯度方向等证据决定最小变量。不得热改r15。

固定game0 Review：drop追取76/81/负候选0，返程123/210/负候选54，取包首82实际左移.18、共享场距离32.895000→33.052498、拓扑不变，剩1118tick/当前场稳速估183tick；真实有路但偏航。JS近似GAE−.016079066，不当训练Critic优势。natural game0库存769目标tick、191/358一致/负候选156，首1620动上gain+.022500、最佳右+.157500，虽有正推进仍非最优；未拾包，不选其他seed替代。first16 cap累计返程54/route负51中normalized GAE正20、中位−.172726，navigation55/负48中正27、中位+.073328；cap非总体率或Q。

JAX full累计window24302/kill13409/stock12461/loose348/custody10491/capture9335/transfer1122，每百万full tick1084.613/598.452/556.142/15.531/468.220/416.627/50.076。entry43926.75/residual1042/clawback−6440.75、route+7318.153/−18342.976、capture496800主导。override后current/ring/fixed实际valid tick占3.19012%/5.75584%/11.46100%；sampled league条件15.76365%/28.10248%/56.13387%，严格分母分列。累计污染0；旧真实取包无路/动态封路/预开路解释维持撤回。健康run128不因本点波动停止，终态先后继正式非零再综合报告。


### r15补充只读：冻结参考与原生返程输出方向（20261019）

FIXED_NATIVE_REFERENCE_FORCES_G4_THROUGH_G16固定g4首512条普通返程及两条已声明近邻anchor，去重共513条，全部敌包；无新对局、标签、训练导入。当前计划实际DAgger系数.2、合法return倍率2，故逐行×2项系数.4；不能误用legacy未安装默认.05。以正确joint质量增加方向的单位move-logit梯度为方向，计算冻结参考KL方向导数；正值表示该局部方向冲突。g0/4/8/16分别354/424/424/419条为正。固定参考g0/stage1正确joint均值.108722/.164197；dead254/trapped28/free231分层，g16当前质量.768318/.680862/.788764，没有支持“全部复活后状态都更差”的结论。

两条anchor参考正确质量为2374的.001046/.0000453及8539的.0000553/.0000750。8539在g4/g8/g16的弱参考方向导数+.022987/+.105142/+.070735，强系数为其4倍；对应逐行×2 DAgger导数−.021112/−.354269/−.439992，方向相反。2374仍拟合良好，故参考冲突存在并非足以单独解释所有状态退化。parent_noncarry retention对两条carrier严格为0；与g0/stage1的move/bomb参考项不能混称同一项。另一个固定两行推理batch与本513条按64分批的bf16输出略不同，所有本审计跨Actor数据使用同shape入口，不跨审计逐bit比较。

这些是输出logit空间的逐行局部导数，不是参数梯度、实际在线采样率、replay抽样权重或完整PPO更新；FIFO不含scene/danger上下文，弱/强两组均列，未擅自给某行分配强系数。不能直接比较这些逐行数值推断KL在真实总loss占主导，不能据此删战斗保留项。该证据将与g32/64及最终完整曲线、原生首次偏航/优势/覆盖共同排序；当前r15源码不变、128继续，r16具体同配方后备不变。


### r15 g32完整异步消费与库存fixture分母重核（20261019）

g32 COMPLETE全部28份产物及policy hash通过。JS Hunter难/中/易0/2/3、Python Tactical6（各64），对父0/−1/−6、0，对g16 −2/+2/0、−11；Hunter death0/1/0、self全0，Tactical death2/self0。八路线death/sugar/drop/人工near/mid/far/recovery/full交付0/0/11/16/16/10/0/2（各16），real_drop对父+5/对前+5，natural对父+2/对前0；不能以中点drop改善称能力完成。natural窗口→目标11/11局级共现、首4/6、持续1151/1749、推进1206/6268、拾4/16→交2/16、恢复41/46；首控48/归因15/pop48主动31/因果kill15存活14，death7/self0。库存阶段541/1022、首8/12、目标5287tick；返程610/727、首5/9、目标981tick。自然拾包3/3/10/6/4及交付0/0/4/2/2仍波动，库存/完整闭环未稳定改善。

普通返程query17265=首14116+追加3149（18.239%），max2/异常0，delivery另8；current/history各3072/8192 kind3非prep replay（37.5%），所选pre-update target mass .201985/.280082。当前FIFO返程4018、joint mass .628789/greedy2574、exact-input无共同pair冲突0。父固定512返程g0/4/8/16/32=.548677/.600665/.596149/.623175/.646983，最终本点greedy334；父navigation本点.684472/358，高于父g0 .631131/331。本轮固定g4返程本点.755195/397，较g16 .773837/408回落但高于g0 .414950/209；navigation .659951/343，较g16 .573650/296上升。固定近邻anchor2374仍右.997556，8539右.048561/左.948846（g4 .965172→g8 .361777→g16 .216729），存在局部明确保持失败；平均值不能替代首偏航状态。

父固定Review本点仍左.999076/右.000301。当前game0 real_drop首返程80、pos(5.89,9.82)、hash ddd3b486a83ffe2e963860339927d198a8df0565fcef892c4b8aa7221d9b2e17，左.998840/右.000143；drop追取70/79/负候选0，返程113/244/负候选87，首左共享gain−.157501/最佳右+.267498、JS近似GAE−.018925101，实际位移及拓扑不变单列核验。近邻6条标签下4/右2/左0、greedy4/6、mass.619513；exact输入0、相同当前格5、.25格内0、最近L1 .429905。natural固定game0没有控制窗口、目标或拾包，不另择seed伪造首返程证据。

库存固定分母再次只读复现：setup按敌笼component选cell后覆盖pos/bunSpawnPos，没有真实home入站；death/sugar/recovery每题8/16在reset时已无home来源、敌笼可接近。这8局是人工初态语义，不是实际取包后通路消失。r15固定game0 death/sugar分别tick105/165发生原生拾取，取包瞬间回实际fixture起点距离均2、入站遍历格阻塞0；home字段INF不构成真实入站不可逆证据。其余real_drop固定game0回起点32/home34、实际存在返程；natural该seed本点未拾包。STOCK_FIXTURE_PROVENANCE_THROUGH_G32验证g0/4/8/16/32共20份原Review，无新模型游戏。三个库存题原始16局分母保留且不得当完整入站能力分母；本点native-brick8局均原生拾包但无交付，open-brick8局death拾3/sugar拾8/recovery拾4、交付均0，墙与seat/seed混杂，不据分层称墙因果。旧“真实取包无路/动态封路/预开路”维持撤回；fixture偏差不回流训练或改变固定评估。

JAX full累计window45850/kill27110/stock26210/loose1033/custody21618/capture20169/transfer2676，每百万full tick1004.424/593.892/574.176/22.630/473.580/441.837/58.622。entry86930/residual1934.25/clawback−15678、route+13732.408/−37141.168、terminal capture1000560主导。override后current/ring/fixed实际valid tick占2.77914%/8.38928%/11.73097%；sampled league条件12.50485%/35.65348%/51.84166%，不同分母不混。

first16 cap累计return113/route负102其中normalized GAE正34、中位−.232888；navigation107/负84中正51、中位+.110082，非总体率、Q或完整梯度。只读下一步在固定原生历史cohort上分开考察监督与冻结参考的参数梯度方向，不写参数、不做优化或导入数据，以区分标签可达/拟合保持/共享表示竞争；这种人工同分布loss也不能冒充真实在线PPO梯度。r15数值配方仍封存、固定128继续，r16同配方后备维持，g64/96/128按COMPLETE消费后决定下一最小因素，终态先后继再报告。


### r16预声明被动逐行replay来源；r15参数方向审计边界（20261019）

r15固定g4首512普通返程加两条已声明anchor（去重513，前512主要来自device0）做零优化步参数梯度审计，Actor仅用COMPLETE g16/g32。将均匀×2 CE、stop-gradient误差权重(.05+1−mass)×2 CE、弱冻结move KL分开作用于同一固定人工cohort；全局权重按实际权重和归一化，padding不进分母。对anchor8539正确joint log-mass梯度，g16三项loss梯度内积分别+38.914688/+143.825195/+1.675359；g32为+15.554152/+38.727192/−2.756346。正内积表示该人工cohort的SGD一阶方向反而降低该anchor正确质量；本审计error加权未消除此现象。g32冻结项在参数空间对该anchor反而同向，说明先前输出logit空间局部对立不能直接升级为“KL导致遗忘”。2374的内积与全部原始范数/cosine见FIXED_NATIVE_PARAMETER_INTERFERENCE_G4_THROUGH_G32。

这不是实际8卡PPO或抽中replay梯度，不含其他阶段、online分布、专家、entropy、bomb KL、clip/Adam；前512的设备偏差也使它不能当分布代表。bf16图/shape可造成数值差异；未执行任何优化、未修改参数/训练环境、无新对局/监督导入。只支持继续检查共享表示与真实样本使用，尚不能据此选定删除KL、换网络或改变奖励。旧只看snapshot拟合与固定标签存在仍不能证明某条query进过loss。

为补该可观测缺口，r16保留当前数值配方的具体预声明，仅新增被动诊断replay_exposure：每代对已经实际传给DAgger loss的有界current/history各32槽/卡记录输入hash、合法joint标签hash、source query代次/kind/preparation/opening；重复抽样不去重，fallback按实际槽位半区，hash可与native first-deviation及FIFO绑定。不更改labels/query/抽样/loss/optimizer/RNG/课程；不读取整个rollout或环境。不把FIFO存在冒称入loss，不把payload次数当128个optimizer reuse次数。

TDD先有REPLAY_EXPOSURE_RED（模块缺失），再有REPLAY_EXPOSURE_RELAY_RED（具体后继flag未继承），实现与focused/相关接续测试共30项GREEN。真实g4保存FIFO的首64行/卡转换JAX数组后host writer及输入不变校验通过；该schema验证载荷不是实际抽样，不计入实际曝光。安装器严格要求前run正常g128且8卡compute_clients0/released；诊断写入异常只标marker，不中断健康训练。r15当前172份数值source逐hash不变，当前worker未注入；已加载owner的行政源码hash原样保存在ADMINISTRATIVE_SEAL，新行政接续文件hash及理由单列SUCCESSOR_ADMIN_EXTENSION。新helper/test只在后继bootstrap安装并封存，不修改运行中数值worker。

r16两个具体预声明已补replay_exposure及4份候选源hash，仍父r15 g128、train1235718504/technical1212388、8卡128代/268435456 ticks、fresh全部状态/cache/eval，合法return×2、query diversity和error replay保留。最小技术smoke→正式非零后主动通知，不等待CPU最终结果或能力阈值。数值最小优化方向仍需结合g64/96/128完整证据决定，可继续准备更后继；不得把此次可观测性修正虚称能力新变量实验或停止健康r15。


### r15固定原生anchor的FIFO存在与实际曝光边界（20261019）

FIXED_NATIVE_TURN_FIFO_PRESENCE_THROUGH_G32仅复核既有COMPLETE保存FIFO，无新游戏或训练输入。g4冻结的index2374与8539在g4/g8均exact-input各1行、原生标签一致、query generation2；g16/g32两条均已不存在，原槽分别为generation10/26。每卡容量2048、每代新增256，循环覆盖周期8代，round2于g10覆盖。8539在g8仍存在时正确质量已从g4 .965172降至.361777，因此首次回落不能只归因FIFO过期；2374同样过期却保持。FIFO存在也不能证明实际抽入loss，故r16被动逐行replay来源诊断有必要，尚不足以选择扩大容量或删参考项。r15数值配方不变，继续固定128代；后继按预声明接续。


### r15 g64完整消费：真实取包后的首次返程偏航（20261019）

g64 COMPLETE/policy及28产物hash通过。JS Hunter难/中/易2/1/8、Python Tactical15（各64），对父+2/−2/−1、+9，对g32 +2/−1/+5、+9；Hunter death0/1/0、self全0，Tactical death2/self0。八路线death/sugar/drop/人工near/mid/far/recovery/full交付0/0/6/16/16/6/1/3（各16）；drop对父0/对前−5，natural对父+3/对前+1，natural拾6/16（父3、前4）。自然窗口→目标12/12为局级共现、首3/6、持续1365/2150、推进1414/6335、拾6→交3、恢复40/45；首次控制46/归因7/pop46主动34/因果kill7存活6，death4/self0。库存阶段525/1014、首6/12、5178目标tick；return840/1136、首17/39、1157目标tick。阶段重入首动作与局级首动作分母分列。人工carry题仍仅诊断，库存fixture旧分母偏差不作为真实home入站链结论；不同地图泛化尚未证明。

普通return query32140=首26581+追加5559（17.296%），max2/配额异常0，delivery另25；current/history各6144/16384 kind3非prep replay（37.5%），pre-update所选target mass 0.212997/0.283585，误差采样信号确实可达。当前FIFO return3884、joint mass .606587、greedy2413，exact-input无共同pair冲突继续按原joint审计单列。父固定512 return本点.636909/333（g32 .646983/334），navigation .609557/314（g32 .684472/358）；本轮固定g4 return .720104/377（g32 .755195/397），navigation .581047/300（g32 .659951/343）。同输入后段回落，不能把动态snapshot变化单独当遗忘。固定anchor2374正确右.998955；8539右.011954/左.983560，比g32 .048561继续回落。历史FIFO存在与实际入loss边界维持，不能仅归因循环过期。

唯一production Review固定seed2026101800/game0，不新增游戏或另择seed。real_drop追取69/79且负候选0；return101/214、负候选60，首返程tick80、hash bdf2d792186f8db08fb804e8dc70545736db3a134324c957dfe4a0924dc87d06，左.768755/右.073325；实际左移.18、共享距离32.895000→33.052498、拓扑不变，最佳右gain+.217499、所选−.157497、JS近似GAE−.022145037。nearby相同地形/基地/来源12条标签下3/右9/左0，greedy3/12、mean mass .335914，exact输入0、相同当前格9条、.25格内0/最近L1 .439902；较近但地形不同的query不冒充same-input冲突。父固定Review本点左.884412/右.002797，跨run固定父输入口径另列。

natural同一game0首次stock偏航891，右gain−.202500、最佳左+.202500，近似GAE−.048369909；首次原生取包后返程tick899、hash 30fbfcffe4b0771a98e3d0c99fe3d8cfe2993b04550fbe493ef985b2f6b468bd，左.373037/右.301975，左移.18、距离35.571251→35.728748、拓扑不变，最佳右+.200005、JS近似GAE−.053162950。return62/168/负候选83，stock9/12/负候选3；保存FIFO相同地形/基地/来源且一格内0、exact输入0，不能把其他地形最近query作为此态已覆盖证明。这是有路但偏航，绝非“取包后无路/动态封路/需预开路”；旧解释维持撤回。实际位移/事件/原生hash见FIXED_REVIEW_ACTUAL_MOTION_G64，JS近似GAE不冒充JAX Critic优势或动作Q。

真实collector first16 cap累计return229/route负197其中normalized GAE正83、中位−.040446；navigation191/负146中正85、中位+.099812；cap非总体比例或梯度归因。JAX full累计window/kill/stock/loose/custody/capture/transfer=87678.0/54020.0/54563.0/2338.0/43981.0/42379.0/6057.0，每百万full tick=950.543394/585.64696/591.533785/25.346957/476.811161/459.443401/65.665747。control entry171207.25/residual3576.50/clawback-34435.25；route+26565.152/-74391.906，capture2037888终局主导。override后current/ring/fixed实际valid tick百分比=2.755315/8.779411/13.390701；sampled league条件百分比=11.317235/34.809045/53.87372，不混分母。声明scene抽样权重与实际tick占比不同，full累计92239871/130721599有效tick（70.562%），退休scene及无原生custody监督0。

证据优先级：固定局部保持失败与自然返程同地形监督缺口均已观察；reward局部负而GAE正的cap现象不能单独归因系数不足；人工cohort梯度不等同真实8卡PPO。下一步先补r16已声明的实际replay逐行曝光，结合g96/128固定原生anchor及整段曲线再选择最小数值因素，当前没有证据足以删保留KL或改初始化/时间字段。r15健康固定128不中断，终态先核验释放、r16正式非零，再综合终态报告。


### r15 g64自然场景角色曝光分母补证（20261019）

仅消费既有g64 COMPLETE及事件计数，无新游戏。NATIVE_FULL_ROLE_EXPOSURE_G64校验按role累加的8种原生事件与正式native full累计逐项一致。role0/弱courier/弱guard的native full visited ticks分别76650176/12694015/5855803；capture33000/7942/1437，按各自visited百万tick430.527/625.649/245.398。不能把训练交付全部归于弱对手；role0仍是JAX多个动作来源，不能冒充JS Hunter、Python Tactical或特定动态联赛家族，也没有已执行动作来源×full的逐项交叉统计。visited含learner非有效决策tick，不能与前述92239871 full alive-valid tick或全场130721599有效tick混用。

worker的opponent_course eligible/query没有natural mask，故不能用其除以同组visited自然场景tick推算自然覆盖。targeted_course明确natural mask后，none/追掉包/库存/返程/交付课程query分别42392/11384/365/840/1445，visited分别76650176/11859785/1178057/834230/4677746；这些是课程片段目标身份，不等于每tick实际oracle phase，也不证明没有query的tick都被安全拒绝。实际current/ring/fixed仍沿用override后有效tick分母；不据此临时换mix或停健康r15。


### r16未安装候选的函数名碰撞修正（20261019）

监督r15 g93/94期间，代码复核发现新被动日志原先在worker循环内导入名为summarize的host函数，会覆盖同作用域已有的jax.pmap更新范数/finite检查器。第一代能通过、下一代会TypeError，故单代技术smoke不能单独发现此问题。当前r15从未安装此候选，172份数值源码逐hash仍一致，训练未故障/未停止。

先新增使用原始patch hook、真实两次迭代的回归测试，REPLAY_EXPOSURE_SYMBOL_RED明确复现第二代缺digest的TypeError；再仅将新导入改为独立别名summarize_replay_exposure，保持原检查器身份与两代调用。19项focused/相关接续测试GREEN，REPLAY_EXPOSURE_SYMBOL_GREEN保存结果；此前30项历史验证不抹除。两个r16具体proposal的4份候选源hash已同步，SUCCESSOR_ADMIN_EXTENSION追加修正凭据。未改任何数值loss/query/RNG/优化器/课程/标签，仍仅在r15正常128及8卡释放后安装；不重启任何旧smoke。


r16被动日志异常路径补证（20261019）：新增测试先复现“正常诊断及其FAILURE marker均无法写入”时OSError逃逸的RED；修正为优先落FAILURE marker、若marker也写失败则仅输出不含状态/隐私的提示并继续更新。真实数值错误检查保持原位置与行为；仅隔离被动日志I/O故障。两代原更新检查器保持与双写失败用例连同focused/接续共20项GREEN（REPLAY_EXPOSURE_FINAL_GREEN），原RED保存。两个r16具体proposal的4份最终source hash再次同步，行政扩展追加凭据，r15的172份数值源码仍未变、未停训。


### r15 g96完整消费与固定拟合后段回落（20261019）

g96 COMPLETE/policy/28产物hash通过。JS Hunter难/中/易0/1/9、Python Tactical9（各64），对父0/−2/0、+3，对g64 −2/0/+1、−6；Hunter death0/0/1、self全0，Tactical death0/self0。八路线death/sugar/drop/人工near/mid/far/recovery/full交付0/0/7/16/15/9/0/1（各16），真实drop对父+1/对前+1；natural对父+1/对前−2，拾4/16（父3、前6）。窗口→目标11/11为局级共现、首6/9、持续979/1878、推进1108/5395、拾4→交1、恢复36/40；首控41/归因9/pop41主动34/因果kill9存活9，death4/self0。库存677/1489、首11/15、4789目标tick；return302/389、首5/10、606目标tick。首动作阶段重入与局级分母仍分列。库存fixture人工来源与人工carry诊断分母保持标注，不据其推断真实取包后无路。

普通return query54183=首43890+追加10293（18.997%），max2/异常0、delivery另37；current/history各9216/24576 kind3非prep replay（37.5%），pre-update所选mass 0.211633/0.276619。当前FIFO return6655、mass .527356/greedy3565，属动态训练分布。父固定512 return本点.604650/323（g64 .636909/333）、navigation .650731/337；本轮固定g4 return .651028/342（g64 .720104/377），navigation .618619/320。原生固定anchor2374正确右.997371；8539右.007138/左.990965，固定hash不变，继续呈局部保持失败。平均error采样信号已可达，不能据此宣称原生每条高错query都实际抽入过loss；r16逐行来源日志补此缺口。

唯一固定Review seed2026101800/game0 real_drop：追取72/81、负候选0，return122/202、负候选69；原生取包后首返程82、hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd，左.499668/右.077146。共享场所选左gain−.157501、最佳右+.267498、JS近似GAE−.017580024，实际左移.18、距离32.988750→33.146248、拓扑不变、剩1118tick；归因有路但偏航，旧真实不可达/动态封路/预开路解释撤回。nearby原生13条标签下6/右7/左0，greedy5/13、mass .424594，exact输入0、当前同格6、.25格内0/最近L1 .259993；不能把几何近邻当exact输入覆盖或标签冲突。父固定Review本点左.589695/右.034890。natural该固定game0没有可用目标阶段，不另择seed补返程；JS近似GAE不当Critic优势或Q。

真实collector first16 cap累计return396/route负351中normalized GAE正141、中位−.074982；navigation255/负199中正114、中位+.091089，非总体率或梯度归因。JAX full累计window/kill/stock/loose/custody/capture/transfer=128835.0/80644.0/82891.0/3692.0/66292.0/63701.0/9275.0，每百万full tick=929.453368/581.7894/597.999915/26.635168/478.249875/459.557643/66.912562；entry254193.00/residual5210.50/clawback-52943.25、route+39343.820/-113011.443、capture3033048仍主导。override后current/ring/fixed实际valid tick百分比=2.646893/9.214942/13.818039；sampled league条件百分比=10.51643/35.572847/53.910723，不混分母。污染计数0，8卡继续固定128代，不因本点下降停止。

当前排序：固定输入保持/共享样本竞争与真实自然状态覆盖需优先区分；仅凭人工cohort参数梯度或局部route负/GAE正，仍不能断言KL主导或只需加奖励。r16最小已验证修正是被动曝光可观测性及其两代/异常路径技术修复，无新数值因素；保留合法return×2、query diversity和error replay，正常128/释放后立即接续。以r16实际逐行曝光和固定原生同输入拟合验证根因，之后选择最小数值改动；不把继续同配方的后备接续虚称单变量能力改进。


### r16固定点实际replay拟合审计预声明（20261019）

保持唯一CPU/JS能力评估owner及原固定g0/4/8/16/32/64/96/128不变。另准备只读audit_actual_replay_fixed_fit：仅在r16 g4 COMPLETE后消费g1–g4已实际传入loss的被动载荷日志，按device、obs/global hash、合法joint标签hash、query generation及kind/preparation/opening对齐尚未循环覆盖的原生g4 FIFO；逐代来源Actor digest和current/history分组计数必须与events一致。保留replacement/fallback重复并另列distinct输入数，固定同一实际payload cohort比较g0/g4/g8及后续COMPLETE Actor拟合，不能把未抽中的FIFO条目算作曝光。

该本地只读helper已语法检查，尚无r16真实载荷，不宣称实际曝光分析已执行或GREEN。不添加对局、标签、优化步骤或训练导入；不消费g1能力分数。g0→g4变化包括四代所有PPO目标，不能当×2 DAgger单独因果贡献；同输入重复FIFO行身份不可区分时明确记录。该审计不阻塞GPU接续、固定评估或健康run完成。

固定run `runs/je10_bun_20261018/control_route_v38_r15_causal_clean` 完成128代/268,435,456 ticks；compute=6007.744s、wall=6896.559s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r16_causal_clean`；本次后继配方选择依据 `Predeclared same numerical recipe continuation; passive selected replay row hashes distinguish FIFO presence from actual loss payload exposure. Does not wait for ability results.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。


### r15正常终态已接续r16正式非零（20261019）

r15固定128代/268435456 fresh ticks全部完成；128代finite、8副本Actor/Critic非零，两项污染计数0。TRAINER_EXIT passed且atomic_stop=false；compute6007.744s/wall6896.559s。GPU_RELEASE_RECEIPT核验0–7全部释放、compute_clients0；立即启动r16，不等待最终CPU分数或报告。父g128独立CPU全状态/next-RNG恢复及172份原封存snapshot逐hash通过；运行中从未热改r15数值源码。

新实验JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r16-clean-causal-init，namespace control_route_v38_r16_causal_clean，父r15 g128；train1235718504/technical1212388，新run/state/cache/eval，8卡128代/268435456 ticks，原混合与固定g0/4/8/16/32/64/96/128协议不变。合法kind3非prep返程DAgger贡献×2、query diversity、error replay全保留；仅新增被动来源可观测性，无新数值因素，不宣称能力单变量因果实验。r17同数值配方后备已具体预声明，后续按完整证据及实际曝光再选择最小数值变量；无能力/admission/promotion门禁。

新r16 smoke真实更新Actor/Critic L2=2.975518/2.295374，8副本一致finite、保存恢复通过；未重启任何已终止旧smoke。正式首代g1/2097152 fresh ticks，Actor/Critic L2=3.207041/2.271299，8副本exact/finite/真实非零，FORMAL_NONZERO_CONFIRMED已原子落盘。正式g0独立fresh审计：父raw Actor/Critic/完整pool/reference exact、EMA=raw、Adam0/FIFO空/计数器0、新RNG、独立cache、resume=null；8192 learner初态空手，退休scene0、无原生custody返程监督0。176份正式source seal逐hash通过。

修正后的被动hook在真实新smoke及正式g1均落盘512条实际payload；正式current/history各256 valid、96返程、121 navigation、7 opening，与events的pre-update source Actor digest/全部分组计数一致，FAILURE marker0。两代函数名碰撞及双写失败RED→20项focused/接续GREEN已保存，当前正常训练不会因诊断I/O失败停止；不改变labels/query/loss/optimizer/RNG/课程。唯一CPU/JS owner正在独立跟随固定点，继续唯一CX owner监督r16固定128代，不因短点评估停止。


### r15完整终态诊断；r16已先正式非零（20261019）

本报告在r15正常128/8卡释放→r16新smoke保存恢复→正式首代非零之后完成，不作为能力门禁。r15全部8固定点224份artifact及各policy hash重新核验通过，完整曲线如下；r11始终EXCLUDED_INVALID，不进入能力曲线。

|g|JS Hunter难/中/易，各64|Python Tactical/64|真实drop交付/16|natural拾包/16|natural交付/16|
|---|---|---:|---:|---:|---:|
|0|0/3/9|6|6|3|0|
|4|0/3/2|7|1|3|0|
|8|1/1/7|10|3|10|4|
|16|2/0/3|17|6|6|2|
|32|0/2/3|6|11|4|2|
|64|2/1/8|15|6|6|3|
|96|0/1/9|9|7|4|1|
|128|0/2/6|10|7|4|0|

最终Hunter对父0/−1/−3、对g96 0/+1/−3；Tactical对父+4/对前+1。真实drop交付对父+1/对前0，natural交付对父0/对前−1；自然拾包最终4/16，高于父3而没有形成交付。不能称路线稳定改善或能力完成。八路线death/sugar/drop/人工near/mid/far/recovery/full最终交付1/1/7/16/16/6/0/0（各16）；人工carrier仅诊断，stock fixture敌笼component传送及保留砖初态的偏差分母仍显式标注，不用于推断真实取包后无路。自然最终窗口→目标12/12为局级共现，首3/5、持续509/944、推进589/3411、拾4→交0、恢复21/23；首次控制26/归因10→pop26主动18→因果kill10存活10，death8/self0。stock245/423、首3/6、目标2839tick；return264/521、首24/40、572目标tick。阶段重入首动作与局级分母分开。Hunter最终death/self均0，Tactical death/self0；natural death8/16不能被单独战斗题的零死亡掩盖。

普通return query82619=首65612+追加17007（20.585%），max2/配额异常0，delivery另49；current/history各12288/32768 kind3非prep实际replay槽（37.5%），所选pre-update正确joint mass 0.208260/0.269980，说明误差采样信号实际可达，不能把FIFO内每条错误状态都称已被使用。最终动态FIFO return7370、mass .466358、greedy3404，原生exact-input冲突按独立joint审计口径保留。

同输入历史数据明确显示后段保持下降：父固定g4首512 return质量 g0/4/8/16/32/64/96/128=.548677/.600665/.596149/.623175/.646983/.636909/.604650/.566053，最终greedy294；本轮固定g4首512 return=.414950/.688676/.755108/.773837/.755195/.720104/.651028/.592137，greedy209/350/403/408/397/377/342/308。最终均值仍高于各自g0，不夸大为完全遗忘；其导航固定集最终父.665622/344、本轮.638250/330。固定原生anchor2374最终正确右.995304；8539右.003678/左.994913，从g4右.965172一路回落，g8仍在FIFO时已下降，不能仅归因g10循环覆盖。

唯一Review固定seed2026101800/game0：真实drop追取73/79、负候选0，return121/283、负候选127；首返程tick80/hash ddd3b486a83ffe2e963860339927d198a8df0565fcef892c4b8aa7221d9b2e17，当前左.969147/右.015895，所选左gain−.157501、最佳右+.267498、JS近似GAE−.017986445。真实左移.18、距离32.988750→33.146248、拓扑不变、剩1120tick，确认为有路但偏航；旧真实不可达/动态封路/预开返程解释撤回。附近相同地形/基地/来源一格内6条标签下3/右3/左0，greedy3/6、mass .550597，exact输入0、.25格内0、最近L1 .470001。父固定Review最终左.929127/右.008438，属于不同已冻结输入口径，不混概率分母。natural固定game0仅stock42目标tick/安全oracle机会0、无取包和返程，不另择seed补失败。所有近似JS GAE不当训练Critic优势、动作Q或完整梯度。

真实collector first16 cap累计return610、route负553中normalized GAE正216、中位−.098841；navigation304、负230中正137、中位+.113862。此cap不能当总体失败率，也不能直接据“局部route负而GAE正”选定奖励系数。verifier全native routes active41103918/selected3137534/native-safe-progress accepted2800392/query178381；ordinary return active24098740/accepted996849/query82619，没有逐phase独立selected/几何候选分母，不能把差值都称安全拒绝或全部归于首条配额。

JAX full累计window169828/kill107201/stock110511/loose5073/custody88369/capture83476/transfer12807，按185416089 full alive-valid tick每百万依次915.929146/578.164498/596.016239/27.360085/476.598339/450.209043/69.071676。直接从父r14的128条events重算并与其curve逐项核验：full有效185662433、capture84401、capture/百万454.593849；本轮归一化交付−4.384806/百万，没有支持训练事件效率稳定上升。父stock/百万597.649175、本轮596.016239；父loose28.524887、本轮27.360085。训练事件与JS局级结果严格分列，不能混为一个成功率。

control entry336483.00/residual6880.75/clawback-73286.50，残余约2.045% entry；route+51898.771/-152512.287，capture3974952仍主导。override后current/ring/fixed实际valid tick百分比=2.457465/9.201227/14.36523；sampled league条件百分比=9.62648/35.165067/55.208453，不混分母。自然角色分层只读g64已说明大部分交付来自非弱课程角色，不能把差距全部归为弱对手，也不能把该角色当JS Hunter或动态league实际家族；额外地图泛化仍未证明。

证据排序与下一最小验证：第一，原生固定状态曾拟合后回落，直接支持优先查实际replay使用与保持/共享表示竞争；人工cohort参数方向审计并不等于实际PPO梯度，不能直接删KL。第二，固定JS真实取包首步与原生query有分布/几何近邻缺口，exact输入0；仍需用learner-state实际曝光区分“生成query”与“抽入loss”，而非导入固定Review标签。第三，局部推进与Critic-backed GAE有不一致cap，但奖励比例本身不证明调系数是最小正确修方。第四，战斗、自然存活与实际对手暴露共同约束后继，不能只看人工返程题。

r16已执行的最小修正是实际payload来源可观测性及其两代/异常路径技术修复，保留现有数值配方继续有效长训，无新数值因素；不得冒称已证明error replay改善能力。r16按g4 COMPLETE后冻结g1–g4真实selected cohort并跨固定Actor核对拟合与后续曝光，再据此选择r17最小数值候选；当前没有验证足够的理由更改KL/时间字段/容量或导航奖励。r17具体同配方连续后备已预声明，确保r16正常终态先接续、不以任何能力门禁等候。持续proposal→TDD→技术smoke→正式训练→异步评估/唯一Review诊断；r16当前健康继续128，不能以本报告结束唯一owner职责。


### r16 g0父点复现通过；正式主线持续（20261019）

r16 g0 COMPLETE/policy/28产物hash核验通过。raw Hunter、Python Tactical、全部八路线、控制漏斗与控制协议聚合结果均与父r15 g128逐项exact；权重树逐叶exact已由fresh审计独立验证，policy pickle文件hash相同与否不作为权重语义变化依据。JS Hunter难/中/易0/2/6、Python Tactical10（各64），death/sugar/drop/人工near/mid/far/recovery/full交付1/1/7/16/16/6/0/0（各16），natural拾4交0、死亡8/self0；这只是继承基线，不是r16改善。G0_PARENT_REPRODUCTION保存复现凭据。

CPU/JS唯一owner已转g4，GPU正式持续，g3/g4稳定代耗时45.27/43.60s，finite/Actor-Critic非零/零污染，实际payload日志无FAILURE。先前两代编译耗时不作为能力门禁；不因短点停训，不用重复技术smoke。等g4 COMPLETE再执行已预声明实际payload固定拟合审计与唯一Review诊断。


### r16 g4/g8实际监督覆盖与固定返程诊断（20261019）

g4与g8均只消费COMPLETE，policy与每点28产物hash核验通过；CPU/JS唯一owner继续跟随，GPU健康固定128代不因短点下降停止。JS Hunter难/中/易与Python Tactical各64：g4=0/2/9、9，对父0/0/+3、−1；g8=0/0/6、8，对父0/−2/0、−2，对前0/−2/−3、−1。两点Hunter/Tactical death/self均0；自然死亡g4=4、g8=5，自伤均0，不能混安全分母。八路线death/sugar/drop/人工near/mid/far/recovery/full交付分别0/1/4/16/16/7/0/3与1/0/3/16/16/3/0/2（各16）；真实drop对父−3/−4，自然交付对父+3/+2，对前−1，尚无稳定提升。人工carrier只作诊断；库存fixture传送偏差保持明确，不推断真实取包后无路。

自然固定漏斗g4：窗口→目标13/14（局级共现）、首5/9、持续1243/1774、推进1320/6136、拾6→交3、恢复37/40；首次控制45/归因10→pop45主动35→因果kill10存活9。g8：窗口→目标9/9、首2/5、持续1005/1692、推进1065/4577、拾5→交2、恢复30/33；首控35/归因6→pop35主动30→因果kill6存活6。分母与相对变化全量保存在POINT_G4/G8_OWNER_REPORT与CURVE_ANALYSIS，阶段重入、局数和事件机会不能混用。

首次取得真正送入native joint loss的来源证据：g1–g4共2048个有效槽、1386个distinct输入标签，逐代source Actor digest及current/history分组计数与events全一致；replacement/fallback重复保留，FIFO同输入歧义0。返程768槽/520个distinct：槽加权正确joint mass g0→4→8=.290218→.885178→.748856，greedy189→697→597；distinct=.318546→.900474→.767903，greedy149→477→414/520。navigation969槽=.387565→.944328→.851715；opening55槽=.150033→.989560→.917242。冻结selection hash=277772e0134e7c484c7e05e29dbff49f1b04666539eb1f29cc0d53779a424113，后续固定点沿用，不重选有利输入。

同一g4 FIFO已生成原生return1406行、1405个distinct，其中520进入前四代native joint replay、885未进入。全部query mass g0/4/8=.430353/.667989/.740061，未selected组=.496047/.531388/.723702，selected组如上；这些分组有误差采样选择偏置，不作独立因果效应。代码核对PPO中的native joint CE仅通过dagger.replay_loss接收该payload；未入载荷不能说这些状态从未训练，仍可能进PPO/旧expert。g5/6/7/8各192个return槽中，仅18/7/9/9槽来自原520输入，各17/7/9/8个distinct；仍在FIFO不等于反复实际监督。同固定前512 FIFO总体return却由.446399→.680294→.734545上升，显示动态平均提升可掩盖已用局部输入保持下降。父固定512=.592137→.664459→.698660，不能以不同集合混归因。

唯一Review固定seed2026101800/game0 g4真实drop首返程80/hash ddd3b486a83ffe2e963860339927d198a8df0565fcef892c4b8aa7221d9b2e17：左.952726/右.007933，所选左gain−.157501、最佳右+.267498，实际左移.18、距离32.988750→33.146248、拓扑不变，JS近似GAE−.018521084。有路但偏航；旧“真实不可达/动态封路/预开路”解释撤回。其近邻5条标签下1/右4/左0、mass.793532、greedy4/5，exact输入0。自然game0原生拾包865后首步右正推进+.200005，首次oracle不一致884/hash ec74d02225a402e2c1d3b547fe72c472620fadd0f08a46294f0ac2020681b6fc选下仍正推进+.112503（最佳右+.160002），不能称反向；下.628281/右.369293，JS近似GAE−.003426278；相同地形/基地/来源一格内监督0。近似JS GAE不当训练Critic优势或Q。

真实collector first16 cap g4 return16、route负10中normalized GAE正7，中位+.006137；g8 return29、负23中正13，中位−.001903。navigation g4负19中正11、g8负30中正18，两点中位+.127512；cap不是总体率或完整梯度。control entry/residual/clawback g4=9089/238/−623.25，g8=20920.75/521/−2411.25；route正/负g4=2045.313/−2960.022，g8=3814.788/−8158.959；capture114864/247032仍主导。JAX full每百万有效tick交付g4=142.255498、g8=358.245173，是fresh初段累计，不与JS局级率混合。override后current/ring/fixed实际valid tick百分比g4=3.511860/5.349838/10.226899，g8=3.300111/5.083177/10.840419；sampled league条件百分比g4=18.439416/28.431479/53.129105，g8=16.944022/26.620592/56.435386，严格分母分列。

当前最小候选为每卡current/history replay各32→64，仅增加有界监督批量；比例、均值loss、合法return×2、error epsilon、query/FIFO/标签、PPO/奖励/观测/混合不变。先在隔离源码执行真实aggregate，RED已复现64而非128槽；GREEN已验证128槽、phase比例、FIFO逐字段不变、fallback、保存恢复及重复batch均值loss/梯度不翻倍。真实collector/GAE/PPO/恢复与接续测试仍在验证，尚未安装或宣称能力效果；当前r16数值源码未变。候选需完成技术验证及具体预声明后才允许在r16正常128和8卡释放后安装，不设能力门禁。


g8唯一Review实际位移补证：同hash ddd3b486a83ffe2e963860339927d198a8df0565fcef892c4b8aa7221d9b2e17取包后tick80，Actor已改向右.769904（左.127279），真实右移.18、共享距离32.988750→32.721252、gain+.267498，拓扑不变，正确开始返程。首oracle不一致tick81/hash 5bd0ee3d17361b8c59078ae5cb8ae6920aeea930af09bd17f379e6cc30d7552b仍向右，gain+.092503但最佳下+.105003，不能称反向；JS近似GAE−.011880040。全return123/552，负候选237，持续路线仍弱；real_drop追取65/79、负候选0。自然该固定game0无目标阶段，不择优补seed。近邻6条标签下1/右5，greedy4/6、mass.724331，exact输入0。FIXED_REVIEW_ACTUAL_MOTION_G8逐state hash与实际位移验证通过。特征替换仅是描述性敏感度，不据此改时间/观测。


### r16 g16固定点评估及保持证据（20261019）

COMPLETE/policy/28产物hash与8项只读诊断产物已落marker FIXED_EVIDENCE_CONSUMED_G16，无新对局或训练导入。JS Hunter难/中/易0/1/9、Python Tactical11（各64）：对父0/−1/+3、+1，对g8 0/+1/+3、+3；Hunter/Tactical death/self均0。八路线death/sugar/drop/人工near/mid/far/recovery/full交付2/0/5/16/16/5/1/2（各16）。真实drop对父−2/对前+2，自然对父+2/对前0，拾4/16（前5），尚无稳定完整路线增益。自然窗口→目标11/11局级共现、首3/6、持续755/1340、推进776/5726、拾4→交2、恢复40/40；首控41/归因10→pop41主动34→因果kill10存活10，death5/self0。

普通return query8876=首7175+追加1701（19.164%），max2/异常0、delivery另6；实际current/history各1536/4096 return槽（37.5%）。g1–g4冻结实际return768槽mass .669109/greedy524（g4 .885178/697、g8 .748856/597），distinct520 mass .692679/greedy369，已用集合保持继续下降；全部1405 distinct g4原生query mass .723984/greedy1047，未selected885 .742378/678。不能将动态总体拟合与已实际使用固定输入保持混为一个结论。固定前512 return本轮.720516/383（前.734545/396）、父.685011/354（前.698660/367）。更多replay槽的候选针对监督覆盖/保持，不能宣称已经证明因果根因或能力有效。

唯一Review固定seed2026101800/game0：real_drop追取65/80、负候选0；共享场return136/303、负候选102，首拾后81向右真实gain+.337502，首共享oracle不一致82/hash 26fe8e0997d4c7c6351709f1a58dec1244728fde4e7ebef207da47f194b2ced9继续右移+.022499（最佳下+.157501），拓扑不变，距离32.869999→32.847500，JS近似GAE−.014404266。该点是次优正推进而非反向。JS诊断自身首return偏航另在84/hash 467948cf592d69eb441a6c73aabd9ce3798815b8a8491a3b6da3bc85b0369a73，不能把84输入的Actor概率当82输入；84下.433991/左.395050/右.169423，与6个近邻的下2/右4标签、greedy3/6、mass.519081仅描述关联，exact输入0。自然固定game0无目标阶段，不换seed。所有真实取包后无路/动态封路/预开路旧解释撤回。

first16 cap累计return64、route负56中normalized GAE正22、中位−.202324；navigation62、负52中正27、中位−.002416，仅cap不是总体率或梯度归因。control entry42235.75/residual1028.25/clawback−6093.25；route正7207.216/负−17956.058，capture481008仍主导。JAX full按有效tick每百万window1058.782667/kill575.523085/stock538.404049/loose16.334156/custody460.160326/capture399.942034/transfer47.800773。override后current/ring/fixed实际valid tick百分比3.309323/5.956436/10.859758，sampled league条件16.252758/29.657683/54.089559，分母分列。退休scene和无原生custody监督0，r16主线继续健康128，评估不作停训或接续门禁。


### r17最小数值候选正式预声明：有界native replay批量×2（20261019）

已完成proposal→行为RED→GREEN。REPLAY_BATCH_RED复现真实production aggregate仍仅64槽；REPLAY_BATCH_RELAY_RED复现后继缺少新因素继承。REPLAY_BATCH_FINAL_GREEN共40项focused/相关接续通过；REPLAY_BATCH_COLLECTOR_FINAL_GREEN独立真实collector→Critic-backed GAE→128槽有效native监督→PPO Actor/Critic finite非零→完整aux/FIFO/RNG保存恢复逐叶exact通过。此为隔离CPU技术验证，不是能力门禁，也不是重启旧GPU smoke。额外验证current缺失的history fallback、每半64有效槽/ordinary32/return24/navigation30/opening2、FIFO写入逐字段不变、joint标签与观测配对、重复batch均值loss/Actor梯度不翻倍。r16的176份数值源码仍逐hashexact，未安装候选或改当前训练。

新实验ID JE10-CONTROL-ROUTE-INTEGRATED-20261018-v38-r17-clean-causal-init，namespace control_route_v38_r17_causal_clean；父r16最终g128 raw Actor/Critic+完整pool，干净链从r10 g128起，r11仍EXCLUDED。train4031452829、technical1541256312，8卡128代/268435456 fresh ticks，独立run/state/cache/eval；Adam0、EMA=raw、新RNG/空手env/FIFO空/计数器0，不resume旧run。此为序列后继，无same-parent对照，不能把能力变化单因果归于batch。

唯一新增数值因素：每卡current/history各32→64，总native joint replay64→128；原return×2、error epsilon .05、current/history比例、ordinary/return/navigation/opening比例、均值loss分母、原生query/合法标签、安全验证、FIFO2048/additions256、奖励/PPO/网络/观测/保留项与课程混合不变。方案元数据明确slot计数已变但比例未变，覆盖旧error recipe中计数不变的陈述，避免误导。课程combat .35/bridge .20/natural full .35/其他合法原生路线 .10，退休scene2/3/4/6和旧lesson1/11为0；合法历史replay仍须hash-bound native acquisition proof。

选择依据是实际前四代520/1405个distinct返程query被native replay使用、同输入拟合g4→8→16回落及g5–8稀疏实际重用；不能把未selected状态称从未训练，也不能忽视精确Review输入覆盖缺口。预期验证g4实际distinct曝光增加与g8/16/32固定输入保持，再对照自然完整路线交付及战斗安全；不设置能力阈值、不因未改善停健康128。若仍弱，按实际曝光、局部首次偏航、native oracle覆盖/冲突、共享目标拟合与对手暴露排序根因，下一轮只改最小有证据因素；不恢复预携包、改名保留无效场景或预开返程解释。

异步协议完全继承：唯一CPU/JS owner，COMPLETE g0/4/8/16/32/64/96/128，每点Python Tactical v2及真实JS Hunter难/中/易各64，固定八路线各16及唯一固定Review seed2026101800/game0；JAX训练bot事件/机会分母单列，人工carrier与库存fixture偏差始终标注。query/FIFO、实际payload和固定拟合、奖励/优势cap、阶段目标/势能、首次偏航与实际位移、current/ring/fixed实际占比都按统一口径输出。中间结果仅诊断与准备后继，不阻塞GPU。

r16/NEXT_RECIPE_PROPOSAL与SUCCESSOR_PREDECLARATION已原子同步具体候选、seed、预算、协议、RED/GREEN及5份候选source hash；原同配方后备凭据保留，ADMINISTRATIVE_SEAL不改写，SUCCESSOR_ADMIN_EXTENSION单列下一进程行政扩展。当前owner已加载旧代码，nested proposal仍完整传给后继新owner，测试覆盖该兼容链。r16正常128→核验0–7卡释放→仅此时安装→新r17最小技术smoke→立即正式8卡→fresh/finite/首代Actor-Critic非零主动通知，再完成终态综合报告；不等网页评分或人工批准，不重启任何旧smoke。r17启动时将自动具体预声明r18连续后备，避免健康训练静默空卡。


### r16 g32局部掉包改善、自然转移仍弱（20261019）

g32 COMPLETE/policy/28产物hash与8项只读证据marker通过。JS Hunter难/中/易2/1/8、Python Tactical8（各64），对父+2/−1/+2、−2，对g16 +2/0/−1、−3；Hunter death0/1/2、自伤0/0/1，Tactical death/self0，未出现须中断的硬安全恶化。八路线death/sugar/drop/人工near/mid/far/recovery/full交付1/0/13/16/16/7/1/1（各16）；真实drop对父+6/对前+8，natural对父+1/对前−1，拾3/16。掉包题改善不能当自然完整路线稳定提升，也不据短点停止健康run。

自然窗口→目标10/10局级共现、首3/4、持续346/558、推进371/2516、拾3→交1、恢复19/20；首次控制21/归因5→pop20主动17→因果kill4存活3，death8/self0。原生return query17949=首14660+追加3289（18.324%）、max2/异常0，delivery另8；current/history各3072/8192 return槽。g4冻结实际return768槽mass .603765/greedy474、distinct520 .637803/342，持续低于g4 .885178/.900474；全部1405 distinct原生query .699786/1030，未selected885 .736205/688。前512固定本轮return .679977/357、父 .692646/364；不以动态总体FIFO掩盖旧输入保持。两者依然高于各自g0，不夸大成完全遗忘。

唯一Review固定seed2026101800/game0：real_drop追取65/79、负候选0；共享场return158/245、负候选65，较g16的136/303、102负候选好。真实拾后80/hash bdf2d792186f8db08fb804e8dc70545736db3a134324c957dfe4a0924dc87d06选下、实际gain+.127499，最佳右+.217499，拓扑不变，距离32.895000→32.767502，JS近似GAE−.011705648；是次优正推进。JS诊断自己的首return偏航另在83/hash 9f52478119efe8620798720a0762db36ef619c50cdf36c52e8bc32a560109f15，左.905470/右.071441；其11条近邻标签下5/右6、greedy7/11、mass.638832、exact输入0，不能把83的概率或覆盖当80输入。自然固定game0仅stock52目标tick、oracle机会0、无原生取包/返程，不换seed。旧取包后真实无路/动态封路/预开路解释始终撤回。

真实collector first16 cap累计return150、route负131中normalized GAE正53、中位−.100860；navigation108、负80中正47、中位+.110360，仅cap不当总体率或实际完整梯度。control entry83667/residual1909/clawback−14310，route正13691.897/负−36747.080，capture992328终局主导。JAX full每百万有效tickwindow982.536901/kill575.072104/stock561.846277/loose20.767829/custody468.653378/capture431.293148/transfer58.171781。override后current/ring/fixed实际valid tick百分比2.526634/6.940413/13.221093；sampled league条件11.580377/30.423075/57.996548，分母分列。训练finite/Actor-Critic非零，退休scene/无原生custody监督0；r17保持已具体测试与预声明的单一批量因素，不因短点换奖励/观测/混合。


g32实际replay去重补证（20261019）：逐代Actor digest及各half事件计数核验，6144个return槽对应5438个distinct输入标签、重复槽706，current/history各3072；history query年龄0/1/2/3/4/5/6/7/8代分别96/740/441/377/338/310/274/255/241槽，年龄0来自首代合法fallback。全部16384槽/13193 distinct；navigation7797槽/5777 distinct，opening395/188。该计数是实际native joint载荷、保留replacement/fallback重复、优化器重用前，不是生成query或轨迹覆盖分母；未曝光不意味着未进PPO。ACTUAL_REPLAY_UNIQUENESS_G32为r17批量候选提供固定比较口径，不新增游戏或训练导入，不改变候选与启动顺序。


r16中点技术记录（20261019）：g64原子checkpoint保存134217728 fresh ticks、8副本exact、passed；全部前64代finite/Actor-Critic非零、两项污染计数0。唯一CPU/JS owner独立评估g64，GPU继续后半128代预算；该记录不消费partial能力结果或设能力门禁。


### r16 g64完整漏斗、实际转向与局部保持（20261019）

g64 COMPLETE/policy/28产物hash与8项只读证据marker通过。JS Hunter难/中/易0/1/6、Python Tactical12（各64）：对父0/−1/0、+2，对g32 −2/0/−2、+4；Hunter death0/0/1、自伤全0，Tactical death2/self2，如实分列，不以胜局增加掩盖安全回退。八路线death/sugar/drop/人工near/mid/far/recovery/full交付1/2/8/16/16/11/0/4（各16）。真实drop对父+1/对前−5，natural对父+4/对前+3，拾10/16（父4、前3），是该固定点改善，尚不能称跨场景稳定提升或停止训练。

自然窗口→目标15/15局级共现、首5/9、持续1314/1782、推进1392/6273、拾10→交4、恢复31/36；首控39/归因14→pop38主动27→因果kill13存活12，death4/self0。return query37454=首30432+追加7022（18.748%）、max2/异常0、delivery另11；current/history各6144/16384 return槽。实际return12288槽/11078 distinct；全部32768槽/26014 distinct，保留fallback/replacement，不冒充轨迹或生成query覆盖率。冻结g4实际return768槽mass .592103/greedy449、distinct520 .618415/322，低于g4/g8/g16/g32；全部1405原生query .695702/1006，未selected885 .741113/684。前512固定本轮return .696353/364、父 .703762/374。局部保持下降与自然g64单点提高并存，不能选一个平均数代替完整曲线。

唯一Review固定seed2026101800/game0 real_drop：追取69/79、负候选0；共享场return59/170、负候选71。真实取包后80/hash bdf2d792186f8db08fb804e8dc70545736db3a134324c957dfe4a0924dc87d06左.855253/右.123129，实际左移.18，距离32.895000→33.052498、gain−.157497，最佳安全右+.217499，JS近似GAE−.022005964；拓扑不变，有路但偏航。同一完整state hash在g32曾选下正推进，当前局部决策回退明确；不据不同输入的概率作同输入结论。12条近邻标签下5/右7，greedy6/12、mass.479114、exact输入0。

自然固定game0有原生取包：994向右实际gain+.176430、拓扑不变；首“名义方向oracle”不一致1000/hash 172476a1ecfb46048cf47c4a78ceba4f0b3aa32177e317624a12a1b8fbc6fb37请求下.988732，却由web/sim.js原生_steer转为右移.18，实际gain+.064999，距离34.108036→34.043037。movementStatus0，排除香蕉强制方向；静态调用五个原生几何端点证明合法请求下/右[1,3]达到完全相同右向端点，不能将其解释为模型反向或错误路线。该补证只对既有唯一Review求几何端点，不包含伤害/能力安全反事实，也不替代训练native verifier。当前固定JS漏斗与共享名义方向指标保留原协议；它们的动作一致率不是原生端点集合一致率，反向候选数不是实际反向位移数。训练oracle早已用五个真实native fork端点的set-valued安全验证，未据此更改训练。自然同地形/基地/来源一格内query0、exact0，仍有跨模拟器覆盖缺口。旧“取包后真实不可达/动态封路/预开路”结论撤回不变。

first16 cap累计return280、route负248中normalized GAE正97、中位−.143440；navigation192、负148中正91、中位+.097780，不当总体率、Q或完整梯度。control entry166275.75/residual3537.25/clawback−32905；route正26462.151/负−74828.461，capture1995840仍终局主导。JAX full92242829有效tick、capture41408，每百万window934.598396/kill572.369696/stock585.790794/loose25.454553/custody472.947333/capture448.902104/transfer65.360094。override后current/ring/fixed实际valid tick百分比2.176675/7.960607/14.766605；sampled league条件9.130048/31.774530/59.095422，分母分列。GPU继续健康128，r17仍只采用已测试批量候选，不依据本点重选seed或增加变量。


r16 g64同输入数值口径补证（20261019）：固定同一Review输入跨COMPLETE raw Actor、CPU bf16、batch1的real_drop正确右概率g0/4/8/16/32/64=.032520/.025358/.556190/.787488/.408250/.133766，左=.957858/.880901/.172227/.199215/.204891/.845306；g8/16曾拟合、后段再次左偏明确。该reference输入hash bdf2d792186f8db08fb804e8dc70545736db3a134324c957dfe4a0924dc87d06，导出source hash3f5f390097652f87e5689abfec8470d92eb9c7981e708938c442ee6544a7bd49，仅后验固定状态诊断，不当总体率或新增训练标签。ACTOR_FORWARD_NUMERICS_G64隔离比较batch1/64及动态/闭包参数：动态与闭包逐项一致，batch改变bf16概率（g32下.382644→.403808、右.408250→.412303；g64左.845306→.855253、右.133766→.123129），这两条输入未发生CPU greedy切换。网页原生float32实际g32选下与CPU近似平局选右也须分列；不将CPU输出当网页已执行动作，不改网络精度或训练。CPU强制环境重做后g64 state.pkl live raw Actor与policy.pt逐叶exact，排除拿错EMA/参数文件。

一次独立只读参数核对漏设CPU环境，反序列化触发其自身GPU预分配失败；该审计进程已退出，未中断/重启训练、未改参数或源。随后显式CUDA不可见/JAX CPU重做exact核对，并核验仍只有8个正式compute client、显存恢复为原训练占用；r16继续g82且全程finite/Actor-Critic非零、零污染。READONLY_AUDIT_BACKEND_CORRECTION保存中性凭据。该故障属于已纠正的审计环境，不能冒称trainer OOM或据此排除健康run；后续反序列化一律显式CPU环境。


r17批量元数据澄清（20261019）：每半ordinary32/return24/navigation30/opening2仅指相应组全部存在的混合情形；缺失组继续原分配与半边fallback，不承诺每代开墙固定2槽。仅补slot_counts_scope说明、未改patch/采样数值；4项终态安装相关测试通过，REPLAY_BATCH_METADATA_GREEN保存，两个具体proposal及行政扩展hash同步。原40项/真实collector GREEN仍有效，r16数值源码未变。


### r16 g96自然交付上升与返程保持回退并存（20261019）

g96 COMPLETE/policy/28产物hash与8项只读证据marker通过，CPU/JS队列无失败。JS Hunter难/中/易0/1/9、Python Tactical8（各64）：对父0/−1/+3、−2，对g64 0/0/+3、−4；Hunter death0/1/1、自伤0/1/1，Tactical death/self0。八路线death/sugar/drop/人工near/mid/far/recovery/full交付2/1/4/16/15/6/0/6（各16）。真实drop对父−3/对前−4；natural对父+6/对前+2，拾9/16（前10），自然改善未同步保留掉包/远返程/战斗，不称跨场景稳定完成。

自然窗口→目标12/12局级共现、首6/9、持续1693/2287、推进1705/5699、拾9→交6、恢复30/36；首控36/归因7→pop36主动25→因果kill7存活7，death8/self0。原生return query57196=首46215+追加10981（19.199%），max2/异常0、delivery另23。current/history各9216/24576 return槽；实际18432 return槽/16688 distinct，全部49152槽/38714 distinct，未将优化器重用、生成query和实际曝光混为分母。

冻结g4实际return768槽mass .578298/greedy456、distinct520 .587664/310；同输入g4峰值.885178/.900474后持续回落，而全部1405 distinct原生query .671797/969、未selected885 .721230/659。前512固定return本轮.674306/359、父 .684027/361。局部保持下降不是动态FIFO输入分布变化造成的假象，但仍高于g0；不能夸大为完全遗忘，或仅凭这些平均值断言KL/奖励系数是根因。

唯一Review seed2026101800/game0：real_drop追取70/80、负候选0；共享场return153/301、负候选111。首原生拾后返程81/hash 74134b5c40d0e4a463337ebb74789ea3878b5f591469ee9b1a37aa5caf9b37b7，Actor CPU64左.989909/右.003905，实际左移.18、距离33.207500→33.464996、gain−.257496，最佳安全右+.337502，拓扑不变，JS近似GAE−.018633755；这是有路但偏航。同一完整state hash在g16原生拾后曾向右正推进，实际保持回退明确，原生可逆入站规则不受质疑。9个同地形/基地/来源近邻标签下4/右5，greedy7/9、mass.640146、exact输入0；近邻不等于同输入或训练标签冲突。natural固定game0只有stock160目标tick/oracle机会0、没有取包/返程，不择优换seed。JS名义动作和原生steering的限制沿用g64补证，不把候选负gain计数一概当实际反向。

真实collector first16 cap累计return437、route负391中normalized GAE正158、中位−.088680；navigation263、负197中正124、中位+.102727，只是cap而非总体率、动作Q或完整PPO梯度。control entry248317.75/residual5195.50/clawback−50981.25，route正39178.800/负−112900.633，capture3009696仍主导。JAX full138697373有效tick、capture63121，每百万window916.715272/kill570.825520/stock593.565676/loose27.304050/custody475.437988/capture455.098742/transfer67.513896。override后current/ring/fixed实际valid tick百分比2.087500/9.056617/14.468921；sampled league条件8.441413/35.045808/56.512779，分母分列。r16健康继续128；r17已验证的批量候选用于检验覆盖/保持，不因自然单点提高或其他题回落取消预算、改父点或追加变量。


r16 g96后置joint审计补齐（20261019）：固定快筛COMPLETE先于非阻塞FIFO诊断落盘，早期owner汇总的fifo_audit为null，不能解释为零。现读取既有DAGGER_JOINT_COVERAGE并刷新owner report及证据marker哈希：11274有效行、9个exact-input重复组、no-common-pair组/行均0；return5698行，joint mass .578727、greedy3365。该动态FIFO拟合与冻结实际replay保持指标分母不同。未新建评估对局，未改变训练或COMPLETE。


r16终态与r17接续核验（20261019）：r16正常完成128代/268435456 fresh ticks，compute5998.229s、wall6884.534s；全部代finite且Actor/Critic真实非零，退休scene与无原生custody监督均0。终代Actor/Critic L2 3.435786/2.901825。GPU_RELEASE_RECEIPT确认0–7卡释放、compute_clients=0后，原supervisor已启动全新r17 technical_smoke01，未重启任何旧smoke。r17源封存179项全部匹配，train4031452829/technical1541256312，父r16 g128，合法return×2保留，每卡native joint replay64→128为本次唯一新增数值因素；r18同配方连续后备已具体预声明。此段不冒称smoke或正式非零已通过，后续凭真实marker更新。

只读冻结teacher补证：沿用g4实际payload同一selection hash，768 return槽上旧g0/stage1_teacher合法joint mass .120448/.212251、greedy94/162；520 distinct为.126441/.201367、66/103。该证据说明旧策略在这些原生标签上支持较低，不能直接证明实际rollout KL与DAgger完整梯度冲突，亦不据此热改当前run；下一诊断应区分真实actor输入/掩码、具体KL方向、危险状态保留与样本权重。FROZEN_TEACHER_FIT_ON_G4_ACTUAL_REPLAY保存来源及限制，无新游戏、标签或训练导入。

固定run `runs/je10_bun_20261018/control_route_v38_r16_causal_clean` 完成128代/268,435,456 ticks；compute=5998.229s、wall=6884.534s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r17_causal_clean`；本次后继配方选择依据 `Actual g1-g4 return payload covers 520 of 1405 generated distinct native queries; selected fixed fit declines from g4 to g8/g16 while actual history reuse is sparse. Test larger bounded replay coverage without changing query/labels or mean loss.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。


### r17正式非零与fresh恢复确认（20261019）

r16已先核验8卡释放并启动后继，现r17新technical_smoke01通过：Actor/Critic L2 3.387674/2.270397，finite、8副本exact、保存及独立CPU恢复通过。随后正式g1/2097152 fresh ticks：Actor/Critic L2 3.362454/2.284691，8副本exact且finite，FORMAL_NONZERO_CONFIRMED保存。正式g0只读全状态审计通过：父r16 g128 raw Actor/Critic及完整pool逐叶exact，EMA=raw，Adam零/FIFO空/计数零/RNG为train4031452829，全8192 learner空手、退休scene0，旧state未复用、cache独立，未重启旧smoke。

首代真实native joint payload每卡128/全局1024槽，current/history各512；各半return192/navigation252/opening4，均与event统计匹配。并非承诺每半开墙固定16：缺组沿原redistribution/fallback，槽数变化已真实进入loss。正式128代预算持续，无能力门禁；唯一CPU/JS owner已从g0跟随固定点，独立只读证据consumer派生父r16并等待后置joint audit，不启动新对局。r18同配方连续后备已具体预声明（train3098210435/technical3130659681），依据r17完整证据可在正常终态前用测试通过的最小因素替换；不等待最终报告空卡。整体目标未完成，继续监督。


### r16完整终态综合与后继方向（20261019）

仅在r17正式首代非零确认后综合r16终态。全部8个固定点COMPLETE、policy与28产物hash核验，后置joint audit已齐，固定证据marker8项全齐。g0/4/8/16/32/64/96/128的自然交付为0/3/2/2/1/4/6/2（各16），真实掉包交付7/4/3/5/13/8/4/5；自然拾取4/6/5/4/3/10/9/4。完整曲线证明中点局部改善，未稳定保持，不能因g32掉包13或g96自然6择优宣告完成。最终JS Hunter难/中/易3/1/2（各64），对父+3/−1/−4、对g96 +3/0/−7；Python Tactical15/64，对父+5/对g96 +7。Hunter death0/1/0、自伤全0；Tactical death2/self2。战斗/安全分列，未合并Python/JS/JAX bot。

八路线death/sugar/drop/人工near/mid/far/recovery/full交付1/1/5/16/16/11/2/2（各16）。自然窗口→目标10/10局级共现、首2/5、持续1118/1937、推进1131/3614、拾4→交2、恢复22/24；首控25/归因8→pop25主动19→因果kill8存活8，death2/self0。真实drop首16/16、持续3819/6189、推进3797/6277、拾16→交5、恢复19/20，death3/self0。人工carrier只作明确诊断，库存fixture缺home入站路径的偏差延续标注，均不得推断真实取包后无路，也未加入训练。原生拿包必有可逆入站规则与旧“无路/动态封路/预开路”撤回不变。

原生return query78671=首63322+追加15349（19.510%），max2/异常0，delivery另34。实际进入joint loss的return24576槽/22394 distinct，全65536槽/51458 distinct，current/history return各12288。冻结g4实际return768槽mass .537624/greedy409，distinct520 .554647/287；从g4 .885178/.900474经g8 .748856/.767903、g16 .669109/.692679、g32 .603765/.637803、g64 .592103/.618415、g96 .578298/.587664继续回落。全部1405 distinct query终点.642909、未selected885 .694769；输入与选择偏差分母分列，不将未曝光解释为未进PPO。前512固定本轮return .648399/334、父固定.647089/342。g128当前FIFO11224有效行、exact重复6组、no-common-pair组/行0；训练同输入标签冲突并非当前证据支持的主因。

唯一Review seed2026101800/game0：real_drop首原生拾后82/hash23d8dfc2e86552ef6bc3b0bec118be0d89637944bea31aacac48880d0afcc7f0，左.923847/右.037261，实际左移.18、共享距离32.895000→33.052498、gain−.157497，最佳安全右+.217499，拓扑不变；JS近似GAE−.021571002而非训练Critic优势。仍有1118 tick、当前速度静态估算183 tick，不能将失败归因时间预算或不可达。10个近邻下/右各5、greedy5/10、mass.489109、exact0，近邻不等于标签冲突。natural真实拾后1358实际右移、gain+.157501；名义首不一致1364请求下经原生steering实际右移.18、gain+.337502，不能误报反向；同地形/基地/来源近邻0、exact0。保留JS固定名义指标同时报告实际端点，不据此改评估协议或训练oracle。

真实collector first16 cap累计return607、route负552中normalized GAE正216，中位−.123044；navigation340、负244中正160，中位+.132258。只是遇到顺序cap，不当总体率、Q或完整梯度。control首次331938/residual6880/clawback−70599.5；route正51912.285/负−151168.773，capture4013040仍终局主导。JAX full185150123有效tick，每百万window909.607821/kill570.655846/stock596.591556/loose28.225744/custody476.467412/capture456.359405/transfer69.149292。override后current/ring/fixed实际valid tick百分比2.150298/9.532218/14.334214；sampled league条件8.464616/36.355402/55.179983；实际课程与采样联赛分母不混用。

证据排序：①实际监督覆盖稀疏且固定已用输入后段保持下降，r17只增加每卡bounded replay64→128检验覆盖/保持，已完成TDD、真实collector与技术smoke并正式非零；②旧teacher保留与返程标签可能存在张力：同冻结520 distinct输出logit层面，g4非零519条中440条、g128非零520中375条KL/DAgger梯度dot<0，平均cos−.687906/−.425421。此为旧teacher弱KL .025/.01与DAgger .2×2的同输入logit诊断，生产KL使用当代PPO状态而非该replay；未含共享参数Jacobian、PPO/其他教师/优化器，不能冒称完整梯度根因。③跨模拟器完整输入覆盖不足：固定自然失败邻域0、掉包邻域虽有标签但exact0，需用r17同口径覆盖与Review继续区分。当前未证实确切输入标签冲突，亦不能仅以负PBRS仍正GAE就重调奖励。r17固定128照常完成；g4/8/16/32/64/96/128继续核验distinct曝光、冻结保持、完整路线与战斗，若覆盖上升而保持/交付仍回退，再用真实共享参数/rollout证据准备单一最小因素，未经TDD和终态不安装。r18具体同配方后备保证连续，所有中间诊断无门禁。


冻结同输入共享Actor参数梯度补证（20261019）：在同一520 distinct实际native return输入、g128 raw Actor、CPU bf16 batch64、末块padding排除，合法return .2×2 CE与旧g0/stage1弱KL .025/.01的平均参数梯度norm为4.684254/1.048529，dot−3.843222、cos−.782482；trunk blocks cos−.647653、heads−.835431。由此同输入方向张力不只发生在独立logit坐标；但生产KL来自当前PPO rollout，这批冻结replay不是生产KL采样集，且未重建PPO/value/其他teacher/clip/Adam，不能等同实际完整优化器干扰或证明它单独造成保持下降。FROZEN_RETURN_PARAMETER_GRADIENT_ALIGNMENT保存来源/限制；无任何优化器步、标签、对局或训练源更改。r17配方保持，后继因素须再结合r17固定覆盖/保持及当代分布证据。


r17 g0 COMPLETE核验（20261019）：28产物hash通过，Hunter/Tactical/control/route四组聚合逐项等于父r16 g128；独立formal-g0审计参数逐叶exact。pickle文件字节hash不相同，PARENT_EVALUATION_EXACT_G0明确区分serialized bytes与parameter equality，不误报为同文件或参数变更。训练已g3、45.329s/代、finite非零及零污染，179源seal匹配；CPU下一固定点g4。


r16当代实际选中样本拟合补证（20261019）：g4的实际return192槽（current/history各96）在g4更新前mass .215194/.318056；经过当代完整PPO更新后同批输入.996862/.995880，192/192 greedy在合法集合内。后续g128同批仅.585545、110/192，去重172输入为.591493、99/172。G4_ACTUAL_UPDATE_FIXED_FIT_THROUGH_G128保存同输入/标签映射与preupdate event计数。说明这批原生set-valued监督可达且能在当代有效拟合，保持回退发生在后续训练；不是“DAgger没有真正进入loss/完全学不动”。该前后包含完整PPO目标，不当单独DAgger因果收益；固定g4批次不当总体率。r17相同诊断等待固定g4 COMPLETE后执行，当前训练不断。


### r17 g4实际覆盖与固定偏航（20261019）

g4 COMPLETE/policy/28产物及8项只读marker通过，后置joint审计4647有效行、重复5组、no-common-pair组/行0。JS Hunter难/中/易0/0/9、Python Tactical9（各64），对父及前点均−3/−1/+7、−6；Hunter death1/0/0、自伤全0，Tactical death1/self1。八路线death/sugar/drop/人工near/mid/far/recovery/full交付0/0/7/16/16/8/1/1（各16）；真实drop对父+2、natural对父−1。自然窗口→目标8/8局级共现、首4/5、持续620/1169、推进697/3681、拾4→交1、恢复21/22；首控25/归因5→pop24主动17→因果kill4存活4，death5/self0。人工carrier/库存fixture偏差继续单列，不作实际取包后无路推论，未达稳定能力不能据此停健康训练。

扩大batch的实际曝光：return1536槽/762 distinct，current/history各768；全4096槽。1370 distinct原生return query中762实际进入joint loss（55.62%），r16相同口径为520/1405（37.01%）；均为独立seed顺序接续，不是same-parent因果效应。g4实际current/history各192 return槽更新前mass .189689/.460029，完整当代更新后.994334/.993869、384/384 greedy合法；全g1–g4实际return1536槽终点mass .933202/greedy1464。前512固定本轮return g0 .491694/253→g4 .815151/427；父固定return .648399/334→.721309/384，而父navigation .660590/348→.592913/311，需继续观察跨任务保持。

唯一Review固定seed2026101800/game0 real_drop追取76/81、负候选0，共享场return116/196、负候选71。首原生拾后82/hashd83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd，CPU64左.927155/右.037940；实际左移.18、距离32.988750→33.146248、gain−.157497，最佳安全右+.267498，拓扑不变，JS近似GAE−.015972721。8个同地形/基地/来源一格内native近邻标签右7/下1、greedy7/8、mass.897591，但exact0、最近.369996格；高近邻拟合没有覆盖这个完整Actor输入。native/global与Review/global差异包含敌人respawn剩余比例，混合feature敏感性仅诊断、可能非法混合状态，不能当有效原生反事实或将其导入训练。natural固定game0未出现可分析return偏航，保留缺机会、不择优补seed。

query1368=首1155+追加213，delivery另2、max2/异常0。first16 cap累计return15、route负12中normalized GAE正7，中位+.068986；navigation18、负15中正12，中位+.249832，仅cap不当总体率或完整梯度。control首次9950.75/residual249.5/clawback−752.75；route正2048.887/负−3007.008，capture115248终局主导。JAX full每百万有效tickwindow1225.331898/kill490.343479/stock456.628413/loose.842877/custody352.533150/capture163.096628/transfer10.535958；前4代重启距离阶段，不能把此小点比全程当单因子回退。override后current/ring/fixed实际valid tick百分比2.957842/5.950987/10.117632；sampled league条件15.548888/31.267119/53.183993。r17继续固定128，后续g8/g16先检验同冻结输入保持、跨任务与真实交付，暂不追加数值因素。


r17 g4返程威胁/拓扑分层补证（20261019）：G4_NATIVE_RETURN_STATE_STRATA以既有真实learner FIFO和实际payload计数，无新采样。1370 distinct原生query中敌free/dead/trapped=695/592/83；762实际selected distinct中354/376/32。剩余复活比例0<global12<=.2在query中133条、实际distinct中78条（含重复的1536槽中163槽），因此不能把固定Review低respawn剩余失败简单归因“训练完全没见过该时间段”。实际selected distinct含254个terrain/pushable观测pattern、query含541个，base mask各2（两视角）；这不是254/541张地图，多地图泛化仍未证明。固定失败仍需完整输入的空间×威胁×时间联合覆盖与保持补证，避免用单一全局边缘分布代替。


### r17 g8覆盖保持与网页偏航分离（20261019）

g8 COMPLETE/policy/28产物hash与8项只读marker通过。JS Hunter难/中/易3/1/7、Python Tactical13（各64），对父0/0/+5、−2，对g4 +3/+1/−2、+4；Hunter death1/1/0、自伤全0，Tactical death2/self2，分别报告。八路线death/sugar/drop/人工near/mid/far/recovery/full交付3/2/8/16/15/8/1/1（各16）；真实drop对父+3/对前+1，自然对父−1/对前0。自然窗口→目标12/12局级共现、首4/8、持续747/1686、推进826/4765、拾1→交1、恢复33/34；首控38/归因9→pop38主动31→因果kill9存活9，death7/self0。库存fixture偏差和人工carrier始终不作真实入站返程证据，健康128照常。

原生query3001=首2576+追加425，delivery另3，max2/异常0。实际return3072槽/1857 distinct，current/history各1536；全部8192槽/4821 distinct。冻结g4实际return1536槽mass g0 .366417→g4 .933202→g8 .875274，distinct762为.423247→.950816→.906722。存在早期回落，但仍显著高于g0；不能仅凭此断言增加batch无效或终止训练。前512固定本轮return g8 .909898/480、父固定.748865/392；父navigation .667435/346。g8当前FIFO9440有效、重复11组、no-common-pair组/行0，标签冲突无新增证据。

唯一Review固定seed2026101800/game0 real_drop追取79/81、负候选0；共享场return96/131、负候选29。首原生拾后82/hash23d8dfc2e86552ef6bc3b0bec118be0d89637944bea31aacac48880d0afcc7f0（与父g128同完整state），左.973654/右.016417；实际左移.18、距离32.895000→33.052498、gain−.157497，最佳安全右+.217499，拓扑不变，JS近似GAE−.016910144。12条同地形/基地/来源一格内native标签右11/下1、greedy10/12、mass.772955、exact0。局部native拟合与网页该完整输入偏航并存，必须区分分布覆盖与保持；不能将“附近有标签”当“该输入已受监督”。natural固定game0无可分析返程，不替换seed。

first16 cap累计return22、route负17中normalized GAE正12、中位+.079876；navigation30、负21中正18、中位+.231201，cap非总体率/Q。control首次21945.5/residual528.75/clawback−2581，route正3823.869/负−8192.973，capture256608终局主导。JAX full每百万有效tickwindow1149.306025/kill574.746742/stock545.128351/loose10.216470/custody458.241491/capture396.099108/transfer36.929259。override后current/ring/fixed实际valid tick百分比3.256061/5.024632/10.545578；sampled league条件17.185280/26.611815/56.202904，分母分列。继续g16及后段曲线；未安装保留项候选，也未改r18后备。


r18潜在后继候选准备、未选定/未安装（20261019）：基于r16完整同输入保持下降及共享参数梯度张力，声明仅对“原生custody+敌源实际携包+活着未受控+当前非危险/非终态+安全合法joint返程证书+非preparation”的行，去除旧g0/stage1移动KL，炸弹KL/危险战斗/无证书/其他阶段/parent保留与DAgger×2、reward、query、FIFO不变；原PPO半批分母不重归一。行为RED两项失败、终态patch RED两项失败后focused28项GREEN；真实native接触转移→collector→候选PPO→保存恢复仍在独立CPU验证。新增候选源码未被r17导入，179封存匹配，r18同配方连续后备未替换。NATIVE_RETURN_RETENTION_DRAFT保存证据、source hash与限制；是否采用等待固定后续曲线与有效技术验证，绝非能力启停门禁。


未安装候选真实链路GREEN（20261019）：NATIVE_RETURN_RETENTION_COLLECTOR日志1项通过，learner初态空手，经env.step原生contact transfer取得敌源包、真实native_transfer事件建立custody；collector生成安全joint返程证书，候选PPO实际消费scale=0的证书行，Actor/Critic finite且真实非零，params/Adam/EMA/native states/aux/RNG完整保存恢复逐叶exact。未改r17源、未启动GPU smoke；相关因果初始化/归因/oracle/loss回归在独立CPU进行，r18备份仍原配方。


### r17 g16固定漏斗与后继候选技术完备（20261019）

g16 COMPLETE/policy/28产物hash与8项只读marker通过。JS Hunter难/中/易0/1/7、Python Tactical8（各64），对父−3/0/+5、−7，对g8 −3/0/0、−5；Hunter death0/2/1、自伤全0，Tactical death1/self1。八路线death/sugar/drop/人工near/mid/far/recovery/full交付2/2/8/16/16/7/1/1（各16）；drop对父+3/前0，自然对父−1/前0。自然窗口→目标8/8局级共现、首0/3、持续473/704、推进542/3158、拾3→交1、恢复20/21；首控25/归因10→pop25主动16→因果kill10存活9，native即时kill1另列，death6/self0。人工carrier/库存fixture限制不变，训练不据此门禁或提前停止。

原生query5386=首4692+追加694（12.885%），delivery另6，max2/异常0。实际return6144槽/3792 distinct，current/history各3072；全部16384槽/9798 distinct。冻结g4实际return1536槽mass .795948/greedy1264，distinct762 .836921/659，仍高于g0 .366417/.423247，但低于g4 .933202/.950816、g8 .875274/.906722。全部1370原生query .867357，未selected608 .905502。前512固定本轮return .856456/449，父固定.761641/393；父navigation .656236/344。当前FIFO8413有效、重复7组、no-common-pair组/行0；实际曝光增加后仍见保持下降与自然交付偏弱，未据早期单点判定最终失败。

唯一Review固定seed2026101800/game0 real_drop追取75/81、负候选0；共享场return111/168、负候选44。首真实拾后82/hashd83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd（同g4完整state），CPU64左.765107/右.037408，实际左移.18、距离32.988750→33.146248、gain−.157497，最佳安全右+.267498，拓扑不变，JS近似GAE−.017376144。7条同地形/基地/来源近邻下2/右5，greedy5/7、mass.692991、exact0；自然固定game0无返程诊断机会，不择优换seed。继续分别诊断当前native拟合、历史保持与网页完整输入覆盖，旧无路/预开路结论始终撤回。

first16 cap累计return37、route负30中normalized GAE正19，中位+.065672；navigation60、负48中正29，中位+.056003，均非总体率/Q/完整梯度。control首次44545.25/residual1034.5/clawback−6691.75，route正7258.130/负−17772.109，capture513696终局主导。JAX full每百万有效tickwindow1105.473490/kill598.419134/stock561.855299/loose16.623998/custody481.423824/capture448.131019/transfer52.022809。override后current/ring/fixed实际valid tick百分比2.797568/5.518079/11.640624；sampled league条件14.159238/27.512969/58.327793。GPU已g32原子保存67108864 ticks，继续128；唯一CPU/JS评估跟随g32。

未安装候选技术完备：行为RED2失败、patch RED2失败、relay RED缺继承flag后完成GREEN；focused及因果初始化/custody/oracle/loss相关56项、最终focused/relay/既有后继配方55项通过，真实native-transfer collector/PPO/恢复1项通过。未来行政owner支持safe native return movement-KL候选的nested flag、正常终态安装、plan/declaration/source seal及连续后备继承；当前r17已加载旧owner不变，直接用其sealed choose_recipe验证nested proposal完整透传。SUCCESSOR_ADMIN_EXTENSION_NATIVE_RETENTION单列新代码hash，原ADMINISTRATIVE_SEAL和179项numerical seal保留，PPO源码尚未安装候选。NATIVE_RETURN_RETENTION_DRAFT继续标记未选定，r18同配方后备未替换；最终选择结合固定后续曲线，无能力门禁/CPU等待。


### r17 实际返程抽样：同次重复与跨代复习分列（20261019）

只读核验 g1–g16 真正送入 native joint loss 的返程 payload，以完整 Actor 输入与 accepted-label hash 计数；`ACTUAL_REPLAY_REPETITION_SCOPE_G16.json`分别保存在r16/r17。r16 current1536槽中，同代同设备同half重复159（10.3516%），同half跨代匹配0，distinct1377；history1536槽中，同次重复105（6.8359%），跨代匹配82（5.3385%），distinct1354。r17 current3072槽中，同次重复860（27.9948%），跨代匹配0，distinct2212；history3072槽中，同次重复491（15.9831%），跨代匹配473（15.3971%），distinct2147。跨代匹配可为有益复习，与同次replacement重复不是同一含义；g1 current/history fallback也须分列。总体return重复比例r16为15.0391%、r17为38.2813%，不能整体称作浪费。

这说明较大batch下同次replacement重复增加，同时历史复习也增加；两轮为不同seed的序列后继，不能据此作same-parent因果结论。保留项候选仍未安装、r18同配方后备仍有效，不把抽样变化与KL候选同时加入。继续消费g32/64/96/128和冻结实际输入保持曲线，选择单轮最小有证据因素；健康训练不受诊断结果启停。当前直接核验r17 g39/81,788,928 fresh ticks，finite、Actor/Critic非零、退休scene与无证明return监督均0，g32异步处理中。旧修复RED/GREEN、r11排除与旧进程释放凭据再次读取确认，未重启旧smoke。


### r17 g32：保持下降与首次真实位移偏航（20261019）

g32 COMPLETE/policy/28产物hash与8项只读证据marker核验通过。JS Hunter难/中/易0/2/6、Python Tactical12（各64），相对父−3/+1/+4、−3，相对g16 0/+1/−1、+4；Hunter death1/2/0、自伤全0，Tactical death1/self1。八路线death/sugar/drop/人工near/mid/far/recovery/full交付3/2/6/16/16/7/0/1（各16）；真实drop对父+1、对前−2，自然对父−1、对前0。自然窗口→目标7/8局级共现，首1/2，持续427/663，推进441/2512，拾2→交1，恢复18/19；首次控制20/归因4→最终pop20主动17→因果kill4存活4，native即时kill0另列，death6/self0。库存fixture仍缺真实入站证明，人工carrier仅诊断；不据此推断实际取包后不可达，健康128代继续。

原生普通return query10172=首8940+追加1232，delivery另11，max2/异常0。实际return12288槽/7555 distinct，current/history各6144；全部32768槽/19028 distinct。冻结g4实际return1536槽mass .745688/greedy1167，762 distinct .792801/616，持续低于g4 .933202/.950816及g16 .795948/.836921，仍高于g0 .366417/.423247；未selected608 mass .889913。前512固定本轮return .816945/424，父固定return .775104/408、navigation .758203/396。当前FIFO7497有效、重复7组、no-common-pair组/行0，return2392行mass .813896/greedy2023。这些分别是冻结实际曝光、固定FIFO与动态FIFO分母，不能混成总体能力率。

唯一Review seed2026101800/game0：真实drop追取75/81、负候选0；返程89/108、负候选17。首拾后tick82完整hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd（同g4/g16），CPU64左.665491/右.220326；生产实际左移.18，距离32.988750→33.146248，gain−.157497，最佳安全右+.267498，拓扑不变，JS近似GAE−.012860985。13个同地形/基地/来源近邻标签下4/右9，greedy10/13、mass .755336，exact0。原生steering可能改变名义方向，负候选总数不当实际反向总数；此处已核实首次实际位移。natural固定game0无返程诊断机会，不换seed。

first16 cap累计return83、route负70中normalized GAE正34，中位−.009877；navigation123、负94中正59，中位+.065664，cap非总体率/Q/完整优化器梯度。control首次87468/residual1878.25/clawback−16662.75，route正13597.332/负−36262.081，capture1035552仍主导。JAX full每百万有效tick window1017.447719/kill592.376264/stock575.319636/loose25.011981/custody483.381324/capture463.988775/transfer62.783378。override后current/ring/fixed实际valid tick百分比2.455226/7.670419/12.359699；sampled league条件11.358249/33.455173/55.186578，分母分列。曲线图更新至g32，后继候选尚未安装，继续固定后段证据。


移动KL专项补证（20261019，未安装候选）：只读`FROZEN_RETURN_PARAMETER_GRADIENT_ALIGNMENT_PRODUCTION_V2.json`使用原生产`factor_kl`语义：移动五方向、炸弹仅前两ability在softmax前切片，第三use_item有意无炸弹KL梯度；native joint DAgger仍含三ability。r16最终同520个冻结实际return输入，DAgger norm4.684254、旧teacher movement KL norm1.048529、dot−3.843222/cos−.782482；r17 g32同762个冻结实际return输入，对应2.428901/.700014、dot−.956509/cos−.562565。两批上movement-only与原movement+bomb数值一致，说明本冻结携包集的旧炸弹KL项无额外贡献。该专项核验支持候选只处理movement保留项，旧审计first2炸弹切片与生产一致，并非遗漏第三项。仍未重构当前PPO真实采样集、其他loss、clip/Adam；不得外推为完整优化器干扰或单独根因。候选未安装，当前数值源封存不变。


r17同输入旧teacher支持补证（20261019）：沿用g4实际payload同一selection hash，1536 return槽上g0/stage1合法joint mass .140376/.219803、greedy226/336；762 distinct为.134140/.187009、greedy106/139。与当前Actor当代拟合、后续保持与共享参数移动KL反向证据共同支持窄范围保留项候选；不证明全部网页失败由KL造成，尤其固定Review exact-input覆盖仍0。只读CPU、无新游戏/标签/优化/训练导入。


### r17 g64中期固定评估（20261019）

g64 COMPLETE/policy/28产物hash及8项只读证据marker全部通过，8副本原子checkpoint exact、134217728 ticks；训练主线已继续。JS Hunter难/中/易0/2/5、Python Tactical10（各64），对父−3/+1/+3、−5，对g32 0/0/−1、−2；Hunter death0/3/0、自伤全0，Tactical death0/self0。八路线death/sugar/drop/人工near/mid/far/recovery/full交付3/3/11/16/15/6/4/5（各16）。真实drop对父+6、对前+5；自然对父+3、对前+4。自然窗口→目标11/11局级共现、首8/11、持续1880/2638、推进1962/4438、拾10→交5、恢复24/31；首次控制33/归因10→pop32主动25→因果kill9存活8，native即时kill0另列；death9/self0。局部整局交付改善与历史输入保持下降同时存在，不据此停训或宣称稳定泛化。人工carrier及库存fixture限制始终保留。

原生ordinary return query19192=首16941+追加2251，delivery另29，max2/异常0。实际return24576槽/14925 distinct，current/history各12288；全部65536槽/36099 distinct。冻结g4实际return1536槽mass .692717/greedy1061，762 distinct .742259/567，低于g32 .745688/.792801且仍高于g0；未selected608 .879888。前512固定本轮return .799140/418，父固定return .764379/405、navigation .757731/390。动态FIFO7348有效、重复8组、no-common-pair组/行0，return2214行mass .846613/greedy1943。

唯一Review seed2026101800/game0：追取74/82、负候选0；return60/86、负候选15。首真实拾后tick83/hash84bf72f44f31bd7ee9f6ad6be982fa8b51c52e1094ccfd10fd2413e29dce2722，CPU64左.999901/右.000011；实际左移.18，距离33.207500→33.464996，gain−.257496，最佳安全右+.337502，拓扑不变，JS近似GAE−.015628378。16个同地形/基地/来源近邻下6/右10，greedy14/16、mass .847891，exact0；不能用近邻拟合替代完整输入覆盖。natural固定game0无返程诊断机会，不更换seed。

first16 cap累计return168、route负131中normalized GAE正65，中位+.054985；navigation202、负146中正98，中位+.084798，非总体率/Q。control首次171666/residual3472/clawback−35946.5，route正26219.451/负−72717.666，capture2102760终局主导。JAX full每百万有效tick window955.326319/kill582.994794/stock595.505553/loose28.067330/custody487.570239/capture478.367299/transfer67.389976。override后current/ring/fixed实际valid tick百分比2.289639/8.984463/13.552123；sampled league条件9.575956/35.648311/54.775733，分母分列。曲线图更新至g64。

同762个冻结实际return输入的g64共享参数梯度，DAgger norm2.673176、旧teacher movement KL .730939、dot−.739516/cos−.378476；heads cos−.690630，blocks +.161808，说明层间也不同。该同输入诊断仍不等于生产完整PPO/Adam干扰。r16完整曲线、r17 g0–64实际曝光/拟合/保持与固定偏航共同支持后继只测试已有安全native返程证书行的移动保留项，不把抽样去重、reward、scene、query或网络变化同时引入；当前r17不安装、不提前停止。


### r18具体后继已预声明：安全原生返程移动KL（20261019）

r17仍正常运行，未安装候选。基于r16完整曲线和r17固定g0/4/8/16/32/64证据，已将r18连续后备替换为经过TDD/真实collector/PPO/保存恢复验证的最小因素：只在活着未受控、非危险/非终态、真实原生custody、敌源实际携包、当前合法joint安全返程证书、非preparation且非战斗/退休scene行，把旧g0/stage1移动KL scale设0；炸弹KL、危险/战斗/无证书/其他phase、parent保留与原PPO actor-half分母均不变。保留合法返程DAgger×2、current64/history64、query、error ε.05、FIFO、reward、scene、网络及固定协议，不叠加抽样去重。序列后继不是same-parent单因素因果实验。

`NEXT_RECIPE_PROPOSAL.json`与`SUCCESSOR_PREDECLARATION.json`已原子更新，原同配方后备完整保存为`ORIGINAL_SAME_RECIPE_*`，NATIVE_RETURN_RETENTION_SELECTED记录选择与未安装状态。r18新namespace `control_route_v38_r18_causal_clean`、父r17最终g128、train3098210435/technical3130659681，8卡128代/268435456 fresh ticks、全新Adam/EMA-to-raw/RNG/env/FIFO/state/cache/eval；g0/4/8/16/32/64/96/128固定协议与唯一CPU/JS owner/COMPLETE保持。只有正常128代终态且8卡释放后才安装和进行新的最小技术smoke，不等待g128评估/网页/报告。后继正式首代finite/Actor/Critic真实非零后主动通知，继续监督；无能力门禁。余下g96/128用于持续根因分析与后继诊断准备，不据短点中断当前run。


### r17 g96晚期固定证据与r18被动观测补齐（20261019）

g96 COMPLETE/policy/28产物hash及8项只读证据全部通过。JS Hunter难/中/易0/4/9、Python Tactical14（各64），对父−3/+3/+7、−1，对g64 0/+2/+4、+4；Hunter death0/1/0、自伤全0，Tactical death0/self0。八路线death/sugar/drop/人工near/mid/far/recovery/full交付3/1/8/16/16/12/2/1（各16）。真实drop对父+3、对前−3，自然对父−1、对前−4，g64自然5/16未稳定保持。自然窗口→目标8/8局级共现、首1/4、持续751/1259、推进798/2548、拾4→交1、恢复15/17；首次控制17/归因4→pop17主动9→因果kill4存活4，native即时kill0另列；death7/self0。不据此中断固定128，人工carrier及库存fixture解释限制不变。

原生return query27903=首24663+追加3240，delivery另38，max2/异常0；实际return36864槽/22138 distinct、current/history各18432；全部98304槽/52882 distinct。冻结g4实际return1536槽mass .689118/greedy1060，762 distinct .746314/577，相对g64槽略降、去重略升，仍低于g4 .933202/.950816，未selected608 .862633。前512固定本轮return .792260/410，父固定return .767301/405、navigation .707911/373。动态FIFO7292有效、重复6组、no-common-pair组/行0，return2275行mass .829409/greedy1963。

唯一Review seed2026101800/game0：真实drop追取68/81负候选0，return37/90负候选26。首真实拾后tick82/hashd83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd，CPU64左.957250/右.001648，实际左移.18、距离32.988750→33.146248，gain−.157497、拓扑不变，JS近似GAE−.016455696；13同地形/基地/来源近邻右11/下2、greedy10/13、mass .647426，exact0。自然固定同seed首次可分析return tick1337/hash39da9c003e4d1f4163e6748d6f75d46575546a03d6bceb04240f0b639687c76c：左.947946/右.007989，实际左移.18，距离35.103752→35.441246，gain−.337494、最佳安全右+.040001、拓扑不变；JS近似GAE−.009936756。自然完整输入exact0、同地形/基地/来源一格内近邻0；没有择优替换种子。已直接证实有路时模型向反向走，不回退至旧“无路/预开路”结论。

first16 cap累计return248、route负191中normalized GAE正91，中位+.001080；navigation262、负186中正123，中位+.075553，非总体率/Q。control首次255018.5/residual5073.25/clawback−54663，route正39010.713/负−109027.199，capture3205320终局主导。JAX full每百万有效tick window936.786957/kill578.576165/stock606.692623/loose28.784342/custody486.604203/capture486.858289/transfer68.298383。override后current/ring/fixed实际valid tick百分比2.463122/9.026631/14.091734；sampled league条件9.855282/34.967542/55.177176，分母分列。曲线图已更新至g96；r18保留项最小因素预声明继续有效。

r18被动曝光可核验性补充：TDD先RED两项缺少host summary和worker安装，再相关62项及最终focused32项GREEN。仅在后继终态installer中加入worker被动观测，每代按真实PPO Actor rollout（原Actor permutation每行一次）计总行、原分母、证书例外行及各设备分布；数值scale不变，观测不反哺采样/loss，诊断异常仅记录不停止健康run。旧proposal保留，原子刷新proposal/source hash/selected receipt，新增NATIVE_RETURN_RETENTION_EXPOSURE_EXTENSION分列行政扩展。当前r17 worker与PPO未改动、候选未安装，仍只有一个新增数值因素。


r17正常终态与r18接续实证（20261019）：r17完成128代/268435456 fresh ticks、atomic_stop=false，compute6135.183s/wall7007.126s，全程finite且Actor/Critic真实非零，退休scene/无custody return监督计数均0；终代Actor/Critic L2 3.433439/2.792764。最终原子checkpoint后原supervisor核验0–7卡释放、compute_clients=0，再立即启动全新r18 technical_smoke01。父r17 g128，train3098210435/technical3130659681，安全原生返程移动KL唯一新增数值因素及被动实际曝光观测已按预声明安装；r19同配方连续后备继承该因素。r17最终CPU评估独立进行，未等待报告/阈值/批准。此时尚未把smoke或正式非零冒称通过，后续按真实marker确认。

固定run `runs/je10_bun_20261018/control_route_v38_r17_causal_clean` 完成128代/268,435,456 ticks；compute=6135.183s、wall=7007.126s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r18_causal_clean`；本次后继配方选择依据 `Full r16 and fixed r17 g0-g64 show actual return updates fit but frozen actual return mass declines (.933202 g4 to .692717 g64); old teacher movement KL opposes same-input native return gradients (cos -.562565 g32/-.378476 g64). Test only certified safe native return movement retention; full-update causality remains unproven. No evaluation gate or final report wait.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。


### r18正式首代与fresh恢复确认（20261019）

r17正常128代终态后先核验8卡释放再启动r18，现新technical_smoke01通过：Actor/Critic L2 3.176331/2.400146，finite、8副本exact、原子保存与独立CPU恢复通过。smoke实际certificate移动例外7680/2097152行（.366211%）。随即启动8卡正式128代run；正式g1/2097152 fresh ticks已COMMITTED，Actor/Critic L2 3.138917/2.384253，finite且8副本exact，Adam各128步；FORMAL_NONZERO_CONFIRMED及r17 RUN_TERMINAL均确认后继正式非零。首代例外7710/2097152行（.367641%），每设备930/986/986/963/948/967/976/954，原Actor分母不变，退休scene/无原生custody返程监督均0；这证明例外实际触发，不当作梯度贡献大小或独立事件率。

FORMAL_FRESH_INITIALIZATION逐叶确认：r17 g128 raw Actor/Critic、完整pool和retention parent exact；EMA等于raw，fresh Adam/FIFO/计数器全0，train3098210435派生RNG exact，8192 learner全部空手、退休scene0，未复用旧namespace状态、cache独立、旧smoke未重启。新源封存184项全部匹配；唯一CPU/JS owner已运行g0，旧r17固定证据consumer全部8项/检查点完成退出后，接续r18只读consumer跟随g4/8/16/32/64/96/128。继续唯一owner监督，不因首点评估或训练启动而结束。


### r17终态完整证据结论；r18已先正式非零（20261019）

r17全部g0/4/8/16/32/64/96/128固定评估COMPLETE，7个后续检查点各8项只读artifact hash逐一复核，最终独立CPU完整状态及179项封存源码恢复通过。先释放8卡、启动r18并确认正式g1非零后，现汇总最终结果，不把报告置于接续之前。

|g|JS Hunter难/中/易，各64|Python Tactical/64|真实drop交付/16|自然拾包/16|自然交付/16|
|---:|---|---:|---:|---:|---:|
|0|3/1/2|15|5|4|2|
|4|0/0/9|9|7|4|1|
|8|3/1/7|13|8|1|1|
|16|0/1/7|8|8|3|1|
|32|0/2/6|12|6|2|1|
|64|0/2/5|10|11|10|5|
|96|0/4/9|14|8|4|1|
|128|0/2/7|9|6|6|3|

终点Hunter对父−3/+1/+5、对g96 0/−2/−2，Tactical对父−6、对前−5；Hunter death0/0/1、自伤全0，Tactical death2/self2。八路线death/sugar/drop/人工near/mid/far/recovery/full交付3/3/6/16/15/5/4/3（各16）；真实drop对父+1/前−2，自然对父+1/前+2。自然窗口→目标11/11局级共现、首5/6、持续880/1104、推进910/3582、拾6→交3、恢复22/25；首次控制27/归因9→pop27主动20→因果kill9存活9，native即时kill0另列，death6/self0。局部整局交付曾改善但未跨后段稳定保持，难Hunter与Tactical保留不足；不同地图泛化未证明，不宣称任务完成。库存fixture不证明从home入站，人工carrier只诊断，旧“实际取包后无路/需预开返程”解释继续撤回。

普通return query36773=首32474+追加4299，delivery另47，quota异常0。实际return49152槽/29437 distinct，current/history各24576；全部131072槽/69917 distinct。冻结g4实际return1536槽mass g0 .366417→g4 .933202→g8 .875274→g16 .795948→g32 .745688→g64 .692717→g96 .689118→g128 .706813；762 distinct终点 .751154/greedy590，未selected608 .865710。前512本轮return终点 .799391/427，父固定 .773425/402、父navigation .748931/396。动态FIFO7322有效、重复6组、no-common-pair组/行0，return2355行mass .824842/greedy2032。更大实际曝光确实提高初段distinct覆盖/当代拟合，但不足保证后续保持或网页泛化；不同序列parent/seed不构成同父因果控制。

唯一Review seed2026101800/game0真实drop：追取74/81、负候选0；return82/99、负候选16。首原生拾后tick82完整hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd，CPU64左.948965/右.006462，生产实际左移.18、距离32.988750→33.146248、gain−.157497，安全右+.267498，拓扑不变，JS近似GAE−.014656436；17同地形/基地/来源近邻下5/右12、greedy11/17、mass .641807，exact0。自然固定game0终点无返程机会，不换seed；g96已有自然真实拾后偏航证据保留。候选负gain不直接等于实际反向，首处分开核对原生steering与实际位移。

first16 cap全段return328、route负251中normalized GAE正117，中位−.001254；navigation344、负244中正159，中位+.075553，非总体率/Q。control首次341015.25/residual6644/clawback−74927.5，route正51875.977/负−146000.431，capture4299816主导。JAX full每百万有效tick window933.146778/kill581.166163/stock612.544828/loose29.692517/custody490.342633/capture491.131314/transfer69.284353；相对父各+2.5878%/+1.8418%/+2.6741%/+5.1966%/+2.9121%/+7.6194%/+.1953%，这是各run内部机会归一化且仍受seed/parent/对手分布影响。override后current/ring/fixed实际valid tick百分比2.484639/9.899537/13.570215；sampled league条件9.739746/37.794466/52.465788，分母分列。

证据排序：第一，已有实际返程输入能当代拟合却逐步保持下降，旧teacher对同输入合法集合支持低；终点共享参数移动KL与DAgger norm .710633/2.476735、dot−1.080753/cos−.614047，支持已启动的窄范围保留项测试，但不是完整PPO/Adam因果证明。第二，固定网页完整输入exact覆盖0及自然近邻0表明空间×威胁×时间联合覆盖不足；若r18仍弱，下一轮只改一个可验证的learner-state覆盖因素，禁止预置携包或导入无因果证明样本。第三，同次replacement与跨代复习需分开，不能把所有重复当浪费；第四，路线reward与总PPO GAE在cap失败样本上可能不同向，不凭cap直接改reward。r18已先正式非零，现继续监督固定128代，后续评估只诊断不作停训门禁。FINAL_OWNER_EVIDENCE_RECEIPT保存汇总证据哈希与限制。


r18早期技术与曝光补证（20261019）：g4/8388608 fresh ticks已原子保存，首两代含编译后g3/4约44秒，训练健康、finite、Actor/Critic非零、污染计数0。g0固定Hunter/control/Tactical/全路线聚合与父r17 g128逐项exact，PARENT_EVALUATION_EXACT_G0记录，参数逐叶相等见fresh证明，pickle字节不要求相等。被动例外g1 7710/2097152、g2 7064/2097152、g3 7156/2097152；相对各代eligible return（含delivery）198215/91294/99311为3.8897%/7.7376%/7.2056%，分母为角色汇总原生active条件而非完整未筛候选；全Actor行分母与此严格分列。原生verifier安全return accepted7772/7105/7263不含delivery，差额亦含终态/危险/scene等限制，不可直接把差当漏数。新例外作用于有证书的窄范围，未获证书行仍保持旧KL。r19同配方后备已具体预声明，train2164719885/technical3567771022，父r18最终g128、8卡128代及全新状态，继承该因素；不静默空卡、不设能力门禁。


### r18 g4：当代拟合、首动作修正与后续真实反向（20261019）

g4 COMPLETE/policy/28产物hash与8项只读marker通过。JS Hunter难/中/易0/2/6、Python Tactical11（各64），相对父0/0/−1、+2；Hunter death0/3/1、自伤0/1/0，Tactical death1/self1。八路线death/sugar/drop/人工near/mid/far/recovery/full交付6/2/3/16/16/7/4/3（各16），真实drop对父−3、自然交付不变但拾6→4。自然窗口→目标7/7局级共现、首3/5、持续524/824、推进552/4201、拾4→交3、恢复25/28；首次控制28/归因13→pop28主动21→因果kill13存活12，native即时kill0另列，death4/self0。短点只诊断，健康128继续。

普通return query873=首764+追加109，delivery0、quota异常0。前4代实际return1536槽/658 distinct，占生成873个distinct原生return query的75.3723%；全部4096槽/1972 distinct。g4实际current/history各192 return槽，更新前合法joint mass .265841/.609343，经当代完整PPO后 .995466/.994113、384/384 greedy合法；这是完整PPO后拟合，非单独DAgger因果贡献。冻结实际return1536槽g0 .427384→g4 .940414，658 distinct g4 .951493/642；后续继续同输入保持曲线。前512本轮return .906755/478、父固定 .800942/426、父navigation .622454/321。动态FIFO3834有效、重复3组、no-common-pair组/行0，return873行mass .903784/greedy808。

唯一Review seed2026101800/game0真实drop：共享metric追取69/81负候选0，return165/358负候选119。原父点反向状态tick82/hashd83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd，本点实际右移.18、距离32.988750→32.721252、gain+.267498；CPU64右.890630/左.058062，7近邻下2/右5、greedy7/7、mass .906766、exact0。下tick83/hash9d40b95e6cb70fff84c1935fce8e07e0782766e9e0e9d817e70c876b75499591首次oracle不一致仍是实际向右推进+.092503（最优名义下），不误称反向。

进一步沿既有唯一Review逐tick重放、逐before-state hash核验，新增只读FIXED_REVIEW_ACTUAL_SCAN_G4：81个追取tick实际推进81/反向0；374个返程tick同拓扑有限场可比，实际推进198/反向126/停留50。首真正反向tick94/hashbb1a15e1dd76768309c93b87155d4cf2a0e82434e904a36a14c6345a2ba967ad，pos[6.07,11.80]→[6.07,11.62]，实际左移.18、gain−.157501，JS近似GAE−.008142424。该状态exact FIFO覆盖0、同地形/基地/来源一格内近邻5（下3/右2）、近邻greedy4/5、mass .809912；这些为固定game0 tick分母，非总体率或独立失败局。未生成新策略对局、未导入训练。

精度语义补证：该tick94 CPU bf16审计选右（batch1右.568386/左.428235、batch64右.567892/左.429070），与录像实际左动作不同。直接对录制输入调用生产ONNX float32与JS参考float32均选左，左/右 .632391/.364168与.632384/.364176，max logit差.000018727。隔离函数副本只将JAX backbone内部bf16改float32，参数与输入完全相同，batch1/64均选左（.632385/.632387），对ONNX max logit差.000015974。因而明确是该输入的数值精度敏感性，ONNX与JS一致；GPU bf16未测，不假设与CPU一致，更不把它泛化为全部失败根因。REVIEW_ACTUAL_NEGATIVE_G4_*保存输入、生产前向、CPU两精度和来源hash。当前训练源未变，后继潜在方向增加“精度下的方向边界稳健性”审计，不能未经验证改本run。

首4代证书例外29722/8388608全Actor行（.354314%），对角色汇总eligible return605422行占4.9093%；g4例外7792/2097152，原分母保持。first16 cap return14、route负9中normalized GAE正3、中位−.047325；navigation17、负12中正9、中位+.264044，非总体率/Q。control首次9405/residual242.25/clawback−772.75，route正2080.301/负−3068.567，capture116280主导。JAX full每百万有效tick window1273.426400/kill482.550468/stock439.047012/loose1.267091/custody346.338192/capture162.821186/transfer11.615000。override后current/ring/fixed实际valid tick百分比2.484346/6.117475/10.864608；sampled league条件12.969731/31.088750/55.941519，分母分列。r18曲线图已创建，继续固定检查点。

### r18 g8固定证据及整段精度分母（20261019）

g8 COMPLETE/policy/28产物hash与8项只读marker已消费。JS Hunter难/中/易0/3/10、Python Tactical6（各64），对父0/+1/+3、−3，对g4 0/+1/+4、−5；Hunter death0/0/1、自伤全0，Tactical death4/self2。八路线death/sugar/drop/人工near/mid/far/recovery/full交付4/3/5/16/15/5/4/3（各16）。真实drop对父−1、对前+2，自然对父/前交付均不变。自然窗口→目标9/9为局级共现，首6/7、持续1327/1764、实际推进1358/4766、拾6→交3、恢复28/33；首次控制35/归因12→pop34主动24→因果kill11存活11，native即时kill0另列，death7/self0。局部路线变化与战斗保持下降一起观察，不作门禁或提前停止。库存初始化与人工carrier限制继续分列，不推断真实取包后无路。

原生ordinary return query1787=首1600+追加187，delivery另1，quota异常0。实际return3072槽/1429 distinct、current/history各1536；全部8192槽/4196 distinct。冻结g4实际return1536槽mass g0 .427384→g4 .940414→g8 .886865；658 distinct g8 .903444/greedy615；未selected215 mass .947877。进一步固定g4当代current/history各192槽，g4完整PPO后 .995466/.994113，g8 .936510/.902724（greedy184/178），说明窄范围移除旧移动KL后仍出现部分同输入保持下降，不能把唯一根因归于该保留项。前512本轮return .912394/482、父固定return .814954/426、父navigation .644084/337。动态FIFO7651有效、return1788行mass .896341/1651；完整输入exact0、同地形/基地/来源一格内近邻9，近邻greedy7/9、mass .717570，仍不等于完整状态覆盖。

唯一Review seed2026101800/game0，拾后tick82/hashd83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd继续实际右移.18、gain+.267498。tick83首次oracle不一致仍实际推进+.092503。全段既有动作重放：追取81/81正推进；返程126tick中实际推进100、反向22、停留4。首真正反向tick87/hasheec6166b822ce1d6fc6f5078b0ed50ea89ca5be14fc821190164133d9a051d9a，实际左移.18、gain−.192501，拓扑不变、JS近似GAE−.006493786。较g4返程374tick/反向126缩短，但固定game0结束原因与16局交付分开，不能由tick缩短推断成功。natural固定game0无return，不换seed。

对g4同一既有Review的全部374个返程输入做生产ONNX、JS参考float32、CPU bf16与隔离CPU float32对照：CPU bf16/web greedy差异2/374（tick84与94），CPU float32/web差异0、JS参考/web差异0；float32最大logit误差.000099659。126个实际负位移tick中，仅1个同时发生CPU bf16/web方向差异；tick84为实际正推进，tick94为首实际反向。这证明精度会影响首处解释，但该输入集大多数负位移不能归于CPU bf16与web的greedy翻转。未测GPU bf16，不把CPU等同GPU；未运行反事实完整轨迹，不把2/374当作终局影响率。REVIEW_ALL_RETURN_G4_PRECISION_COHORT与REVIEW_PRECISION_MOTION_G4保存逐输入hash、全部前向和分母；仅既有录像重放，无新游戏、无训练导入。纯诊断依赖尚未落盘及字段名错误已在本地修正并保留失败log，均非训练故障，当前训练源码不变。

first16 cap累计return20、route负15中normalized GAE正6，中位−.047325；navigation29、负20中正13，中位+.198026，非总体率/Q。control首次21826.5/residual518.5/clawback−2664.25，route正3901.669/负−8365.769，capture261816仍主导。JAX full每百万有效tick window1207.719525/kill597.519282/stock543.293982/loose13.059497/custody470.047240/capture402.289273/transfer41.071170。override后current/ring/fixed实际valid tick百分比1.830768/5.840330/11.887595；sampled league条件9.519547/29.511018/60.969436，分母分列。曲线更新至g8，r18继续健康128预算，r19已预声明同配方后备，后继最小因素仍待后段证据，不热改当前run。

### r18 g16：保持下降在float32复现；真实自然game0交付（20261019）

g16 COMPLETE/policy/28产物hash及8项只读证据已通过。JS Hunter难/中/易2/5/8、Python Tactical12（各64），对父+2/+3/+1、+3，对g8 +2/+2/−2、+6；Hunter death0/1/2、自伤全0，Tactical death4/self3。八路线death/sugar/drop/人工near/mid/far/recovery/full交付3/3/4/16/16/7/3/2（各16）；真实drop对父−2/前−1，自然对父/前−1。自然窗口→目标9/9仍为局级共现，首3/5、持续827/1438、路线推进801/4129、拾4→交2、恢复25/28；首次控制29/归因7→pop29主动24→因果kill7存活7，native即时kill0另列，death3/self0。战斗单点回升与自然交付回落均只诊断，固定128预算保持。

普通return query3606=首3236+追加370，delivery另3，quota异常0；实际return6144槽/2987 distinct、current/history各3072；全部16384槽/8694 distinct。冻结g4实际return1536槽mass g0 .427384→g4 .940414→g8 .886865→g16 .750031；658 distinct .792410/542，未selected215 .873690。前512本轮return .811189/427、父固定return .810877/425、父navigation .740717/391。动态FIFO7472有效、重复8组、no-common-pair组/行0，return1821行mass .869596/1639。固定drop首拾输入exact0、同地形/基地/来源近邻12，greedy11/12、mass .903745；自然首偏离输入exact0、同几何近邻0。不能由当前FIFO近邻拟合替代完整状态覆盖或推断历史从未访问。

隔离CPU float32函数副本对同一selection hash的实际训练输入复核，源文件/训练不变：1536返程槽mass g0 .429028→g4 .940304→g8 .888664→g16 .749671，与原CPU bf16保持下降一致。g4实际current/history各192槽，float32 g4 .995379/.994078→g8 .936878/.902774→g16 .829092/.769729，g16 greedy163/150。G4_ACTUAL_UPDATE_FLOAT32_FIT_THROUGH_G16证明总体同输入保持下降不是CPU bf16诊断翻转的产物；这仍不是完整PPO/Adam或单独KL因果证明，未测GPU bf16。新诊断初次对源码bf标记数量假设错误，已修正为全部内部bf16变量，失败log保留；不涉及训练故障。

唯一Review seed2026101800/game0逐tick原生重放：真实drop追取81/81实际正推进；返程156tick实际推进82/反向60/停留14。拾后tick82仍实际右移+.267498，tick83局部oracle不一致仍实际推进+.092503；首真正反向tick214/hash22be122dea30dcc6eda094fb5b6560fffe83ff9038cc1bdd031d8afc839c1ef3，左移.18、gain−.232502、同拓扑，JS近似GAE−.022015975。最终learner tick549死亡丢包，敌人取回后再取己方包并于tick984交付，learner本局交付0。

自然同一game0为成功样本，不能因局部偏航标签称整局失败：tick756真实取敌库存包、tick961完整交付、learner全局死亡0。返程197tick实际推进192/反向5，首局部不一致tick761实际被原生steering转成右移、gain+.337502；首真正反向tick762/hashadbc11184a8ef003e77d144a0e1a9f9af0c0bcf4d3f519690415eb7f5b690d7a，上移.18、gain−.202503、同拓扑，JS近似GAE−.034430378。保留固定种子，不把单局成功替代自然2/16交付或跨场景稳定性。

另发现纯Review顶层事件注释错误，具体位置scripts/je10_route_funnel.js的events.push：用info.died.some(Boolean)判死亡，却始终player:team，并用三元式压缩同tick多事件。g16 drop tick1实际victim1被标0；自然tick617/769实际victim1被标0。逐tick diagnostic_ticks.events.death向量、原生Player.info和before/after hash都正确，路线aggregate的m.death使用info.died[team]也正确，因此不改变已报告死亡/交付分母，仅撤销顶层player注释的身份解释。REVIEW_EVENT_IDENTITY_G16在984/961个转换逐一校验hash并生成明确victim/carrier/origin的sidecar；自然两个死亡均属敌人，learner未死亡。当前封存评估代码不热改，所有后续诊断以原生向量为准；route_deviation标签只代表局部oracle不一致，非整局失败。新sidecar无新对局、无策略调用、无训练导入。

首16代移动KL例外120047/33554432全Actor行（.357768%），是原生证书采样支持范围，不当成完整携包状态占比或梯度贡献。固定verifier每fragment每tick4槽，未选中不等于无路或不安全；旧geometrically_reachable统计实际使用selected位，不能以该字段差值称几何不可达。first16 cap return36、route负26中normalized GAE正11，中位+.056972；navigation63、负45中正28，中位+.076971，均非总体率/Q。control首次44408.25/residual1025.75/clawback−6920.75，route正7325.889/负−18072.658，capture527784主导。JAX full每百万有效tick window1139.270971/kill616.552521/stock565.820354/loose20.801551/custody499.328060/capture454.591100/transfer56.227773。override后current/ring/fixed实际valid tick百分比1.901722/6.927227/11.639260；sampled league条件9.319751/33.503354/57.176895，分母分列。曲线更新至g16；g32训练已完成并由唯一evaluator处理，后继仍有同配方后备，不因短点停止。

### r18 g32：完整交付局部上升；早期FIFO支持已全部过期（20261019）

g32 COMPLETE/policy/28产物hash与8项只读marker复核通过。JS Hunter难/中/易0/3/5、Python Tactical8（各64），对父0/+1/−2、−1，对g16−2/−2/−3、−4；Hunter death1/0/1、自伤全0，Tactical death1/self1。八路线death/sugar/drop/人工near/mid/far/recovery/full交付2/3/6/16/15/11/2/4（各16）。真实drop对父不变/前+2，自然对父+1/前+2；人工远程11/16只是诊断。自然窗口→目标12/12局级共现、首4/5、持续887/1255、路线推进942/4517、拾5→交4、恢复27/31；首次控制33/归因4→pop33主动28→因果kill4存活4，native即时kill0另列，death6/self0。交付局部上升与早期输入保持下降同时存在，不宣称跨场景稳定改善、不作能力门禁。

ordinary return query7670=首6879+追加791，delivery另6、quota异常0。实际return12288槽/6358 distinct、current/history各6144；全部32768槽/17151 distinct。冻结g4实际return1536槽mass .665385/greedy1040，658 distinct .727436/490，继续低于g16 .750031/.792410；未selected215 .863866。前512本轮return .755905/393，父固定return .772597/403。动态FIFO7197有效、重复8组、no-common-pair组/行0，return2133行mass .860005/1903；当代动态分布拟合不能替代冻结输入保持。

唯一Review seed2026101800/game0，drop追取82tick实际均正推进；返程282tick实际推进158/反向96/停留28。首真实拾后tick83/hash84bf72f44f31bd7ee9f6ad6be982fa8b51c52e1094ccfd10fd2413e29dce2722立即实际左移.18、gain−.257496、拓扑不变，JS近似GAE−.019526396；CPU bf16左.999582/右.0000965，同地形/基地/来源近邻4全部greedy合法、mass .971530、完整输入exact0。本点又出现拾后首动作反向，不能只以g4/8/16首动作修正作稳定能力结论。natural固定game0无return分析机会，不替换seed。顶层死亡player注释限制继续适用，以原生逐tick死亡向量与身份sidecar为准。

只读真实learner快照审计SNAPSHOT_VERIFIER_COVERAGE_G16：保存的8192个实际状态，ordinary return eligible677，4槽实际选32/证书30，8槽选64/证书60；多出的30个合法返程状态CPU bf16 greedy全部已在oracle集合。增加verifier确可扩大证书覆盖，但此固定截面不证明会增加返程误差监督，故不把单纯加槽作为当前首选。A/B阶段多出的合法状态32/23中greedy误差2/1，delivery无机会；CPU≠GPU，单快照不代表全部轨迹或历史query配额，未导入训练。

更直接的历史支持证据FROZEN_G4_FIFO_RESIDENCY_THROUGH_G32：g4实际使用过的1536返程槽/658不同完整输入标签，g4、g8均全部仍在FIFO；g16、g32完整输入与原query generation记录都为0。真实存储每设备capacity2048、每代写256槽（含invalid padding），固定8代窗口；g16 FIFO有效query generation9–16，g32为25–32，早期记录确定已淘汰。冻结拟合g4 .940414→g8 .886865→g16 .750031→g32 .665385，与过期并行；过期证明本FIFO已无这些exact输入支持，不证明PPO/其他teacher中也不存在、也不单独证明遗忘因果。跨代复习与同次替换重复分列：首16代current返程3072槽中同次unique累计1978（同次重复1094），history3072槽同次unique累计2394（同次重复678），history跨前代payload匹配1460槽；后者是合法复习，不能当作同次浪费。

已准备、尚未安装的最小候选为native joint FIFO容量2048→4096，每代写入仍256、窗口8→16代；所有native joint阶段共同延长历史，而非仅返程，current/history各64、phase比例、error ε.05、返程DAgger×2、query/verifier、reward、网络、课程均不变。8项RED后GREEN覆盖单常量改动、fresh默认buffer、正常终态且全8卡释放后的安装、当前payload逐叶不变、g12旧样本保留、g16仍保留及g17正常过期；候选不是同父因果实验。当前数值封存184项仍全匹配，仅新增未安装recipe/测试及行政后继装配支持；r18不热改。相关测试仍在进行，当前r19同配方后备未替换，后段固定证据用于选择最小因素。

first16 cap累计return76、route负62中normalized GAE正27，中位−.042535；navigation118、负82中正54，中位+.190649，非总体率/Q。control首次87077.5/residual1882.25/clawback−16797.75，route正13742.987/负−36522.913，capture1070520主导。JAX full每百万有效tick window1040.573251/kill606.358570/stock582.985108/loose27.139802/custody495.606022/capture473.030137/transfer65.689399。override后current/ring/fixed实际valid tick百分比2.450954/7.630316/12.688801；sampled league条件10.571229/33.316112/56.112659，分母分列。曲线已更新至g32，r18健康固定128继续。


r19 FIFO候选补证（20261019）：8项focused及72项相关测试GREEN。相关首跑两项失败来自旧测试把已安装的phase dispatcher/normalized collector误当未安装源码；改用明确历史输入检验patch仅变指定语句，并另检验当前normalized worker幂等，未改训练或原installer。REPLAY_HORIZON_CANDIDATE保存RED/GREEN/source hash与尚未选择、尚未安装状态。新鲜初始化审计新增实际FIFO容量与plan对照，防止只改元数据未改变真实buffer；该只读审计不控制训练。r18数值源仍184项全部匹配，当前g48后继续，r19原同配方连续后备未替换。


既有Review轨迹对照图（20261019）：reports/je10_r18_fixed_review_routes_20261019.png，仅绘制固定seed2026101800/game0已验证的C_RETURN转换，背景为该局首次返程时的拓扑；断段代表其他phase未绘制，不代表传送或断路。对照g4/g16真实drop未交付与g16自然完整返程交付，红叉为共享距离场核验的首个实际反向转换。图只展示固定单局，不替代16局分母、全对局录像或跨地图能力；未生成新游戏。

### r18 g64：掉包交付10/16，历史保持继续下降（20261019）

g64 COMPLETE/policy/28产物hash及8项只读证据全部通过。JS Hunter难/中/易0/0/8、Python Tactical9（各64），对父0/−2/+1、0，对g32 0/−3/+3、+1；Hunter death与self全0，Tactical death3/self3。八路线death/sugar/drop/人工near/mid/far/recovery/full交付5/3/10/16/16/9/4/3（各16）。真实drop对父+4/前+4，自然对父不变/前−1；自然拾3→交3，成功率不能忽略拾包机会下降。自然窗口→目标9/9局级共现，首3/3、持续736/881、路线推进760/2401、恢复13/15；首次控制17/归因2→pop17主动12→因果kill2存活2，native即时kill0另列，death5/self0。自然整局交付与强Hunter尚未稳定提高，健康run仍按固定预算继续。

ordinary return query15748=首14144+追加1604，delivery另20、quota异常0；实际return24576槽/13230 distinct、current/history各12288；全部65536槽/33698 distinct。冻结g4实际return1536槽mass .646384/greedy1008，658 distinct .696717/465，低于g32 .665385/.727436，未selected215 .841527；前512本轮return .734784/385、父固定 .801733/421、父navigation .708282/365。当前FIFO6979有效、重复9组、no-common-pair组/行0，return1981行mass .857922/1755。同输入保持问题持续，当前动态FIFO高拟合与完整交付变化分列，不能用动态输入集掩盖旧状态遗忘。

唯一Review seed2026101800/game0，两种真实因果取包路径本点都最终交付且learner死亡0：drop tick616交付；自然tick927交付。drop追取81tick均实际正推进；返程106tick推进84/反向18/停留4，首真实拾后tick82/hash23d8dfc2e86552ef6bc3b0bec118be0d89637944bea31aacac48880d0afcc7f0立即实际左移.18、gain−.157497、同拓扑、JS近似GAE−.016555108；CPU bf16左.995948/右.002476，exact0、同几何近邻5全部greedy合法、mass .773208。即便最终成功仍可包含首动作反向，必须分清局部方向与终局交付。

自然完整返程235tick实际推进210/反向25，首反向tick683/hash61958e037acab74578918c1c55086647b3df46acb6c9ec4d4c8ee36cb07d5566：名义下动作被原生steering转为实际左移.18、gain−.171783、同拓扑、JS近似GAE−.039710703；原名义方向计算gain−.022499不能替代实际运动。CPU bf16名义下.747513/右.252294，完整输入exact0、同几何近邻0。路线状态覆盖、原生转向与旧状态保持仍须共同观察；不据固定game0成功宣称跨场景稳定。REVIEW_EVENT_IDENTITY_G64核验每tick hash及身份，顶层death.player旧错误不污染原生统计。

first16 cap累计return174、route负140中normalized GAE正63，中位−.051868；navigation206、负147中正91，中位+.095225，非总体率/Q。control首次171824.75/residual3469.25/clawback−35830.25，route正26640.050/负−73316.622，capture2165832主导。JAX full每百万有效tick window972.869069/kill592.469280/stock605.109908/loose28.966283/custody497.445877/capture488.719907/transfer67.609862。override后current/ring/fixed实际valid tick百分比2.442641/8.552545/13.937750；sampled league条件9.792850/34.139096/56.068054，分母分列。r18继续固定128，不根据中点得分设门禁；FIFO窗口候选在等隔离真实collector/PPO/保存恢复收尾，未安装到当前run。

### r19具体后继预声明：native FIFO历史窗口8→16代（20261019）

基于r17完整曲线和r18固定g0/4/8/16/32/64证据，r19已从同配方连续后备替换为单个新增数值因素：每设备native joint FIFO容量2048→4096，每代写256槽保持，名义历史窗口8→16代；所有native阶段共享延长历史，非仅返程。current/history各64 loss槽、phase分配、error ε.05、合法返程DAgger×2、verifier/query配额、原生custody证明、reward/PBRS、课程、网络、PPO/保留系数、评估协议均保持。冻结早期实际返程1536槽/658 distinct从g4 .940414降至g64 .646384，且g16/g32已全部失去本FIFO支持，是具体动机；g8尚resident却已下降，明确expiry并非唯一原因，增大FIFO还可能稀释近期历史抽样，不冒称因果修复。

TDD证据：REPLAY_HORIZON_RED_V2先8项RED，focused8项GREEN；相关72项及最终27项GREEN；REPLAY_HORIZON_COLLECTOR_GREEN_V2真实collector→4096容量/256 additions→128 replay→PPO finite且Actor/Critic真实非零→保存恢复通过。集成夹具仅8行，首跑不满足正式top_k256，后在隔离测试中重复同一真实collector输出至256行以检验正式分支，未增加游戏、未向正式run导入测试数据；失败log完整保留。候选installer只在正常128代终态且8卡compute client为0时改capacity常量，当前r18未安装、184项数值封存仍全匹配。行政supervisor新增后继装配和正确actual replay元数据（128，旧模板64不再冒充当前值），不影响运行worker。

新namespace `control_route_v38_r19_causal_clean`，父r18最终g128，train2164719885/technical3567771022，8卡128代/268435456 fresh ticks；fresh Adam/EMA-to-raw/RNG/env/FIFO/state/cache/eval，父raw Actor/Critic及完整pool逐叶exact；初态8192 learner全部空手、退休scene0、真实FIFO4096且plan相符。只使用全新技术smoke，不重启旧smoke；仅技术正确性通过后立即正式训练，g1必须finite、8副本exact、Actor/Critic真实非零。r18终态先8卡释放核验→立即r19 smoke/正式→确认正式非零并主动通知→最终综合报告，不等待g128 CPU评估、网页或批准。

唯一CPU/JS owner消费g0/4/8/16/32/64/96/128 COMPLETE，Hunter各难/中/易64、Tactical v2 Python64及8类固定路线各16局；所有固定seed、唯一Review链路不变，合成carrier仅诊断、库存fixture不能证明从home入站。固定早期实际payload同输入fit/residency跟踪g4/8/16/32/64/96/128：r19 g16预期仍resident，g32正常过期；该预期是机械验证而非能力门禁。fit若仍下降，优先实际历史抽样支持或完整PPO目标干扰；fit保持而网页仍反向则优先联合状态覆盖/原生steering，保持最小有证据改动和无遗忘评估。序列parent/seed后继不构成same-parent因果实验。NEXT_RECIPE_PROPOSAL与SUCCESSOR_PREDECLARATION已原子更新，旧同配方两文件保留为ORIGINAL_SAME_RECIPE_*，REPLAY_HORIZON_SELECTED记录选择及尚未安装状态。健康r18继续固定128；g96/128仅进一步诊断，不作为本后继启动门禁。


### r18 g96：自然交付5/16，首拾偏航与早期保持下降仍在（20261019）

g96 COMPLETE与八项只读消费产物hash核对通过，唯一CPU/JS评估完成至g96；未生成额外策略对局、未导入诊断标签。JS Hunter难/中/易0/1/8、Python Tactical12（各64），对父0/−1/+1、+3，对g64 0/+1/0、+3。Hunter death0/2/1、自伤0/0/1，Tactical death1/self1。八路线death/sugar/drop/人工near/mid/far/recovery/full交付0/2/9/16/16/12/2/5（各16）；真实drop对父+3/前−1，自然对父+2/前+2。自然窗口→目标13/13为局级共现，首8/11、持续1487/1928、推进1545/5725、拾9→交5、恢复31/37；首次控制38/归因8→pop38主动22→因果kill8存活8，native即时kill0另列，death4/self0。中后段自然交付上升，但强Hunter与早期状态保持尚不足；不按单点能力停止健康run。

ordinary return query24114=首21598+追加2516，delivery34、quota异常0。实际return36864槽/20204 distinct；全部98304槽/49993 distinct。冻结g1–4实际return1536槽mass .629641（g4 .940414、g64 .646384），658 distinct .684976/greedy461，未selected215 .843483。前512本轮return .714822/372，父固定return .793102/410、父navigation .715525/371。动态FIFO7040有效、重复5组、no-common-pair组/行0；return2218行mass .853146、greedy1952，含敌源2157/己源61/空手0。动态拟合与冻结同输入保持分列。固定drop首偏航完整输入exact0、同地形/基地/来源近邻15，greedy8/15、mass .585154；自然首偏航exact0、同几何近邻0，仍提示联合状态覆盖不足，不能仅凭局部近邻称oracle标签冲突。

唯一Review固定seed2026101800/game0逐tick原生hash与事件身份复核：drop追取81tick均正推进，返程123tick实际推进103/反向20/停0，tick492交付、learner死亡0。首真实拾后tick82/hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd立即左移.18，距离32.988750→33.146248、gain−.157497、拓扑未变、JS近似GAE−.018519722；CPU bf16左.996208/右.000428。自然tick1127真实取包后同tick首反向，hash0f0984e4485d49e0e7b0fa52f1bc5fdba61c641b7889330b2cc71c7be269c3f5，实际左移.18、gain−.157501、同拓扑、JS近似GAE−.051376321；CPU bf16左.622486/右.269943。首次携包于tick1455被杀丢失，敌人取回后于1591交付；learner1669再真实取包、1883最终交付。自然425返程tick中423同拓扑可比，实际推进368/反向50/停5。该固定局最终成功，但包含一次携包死亡和首次偏航，不把route_deviation标签误称整局失败或忽略过程安全。顶层death.player旧注释错误仍以原生向量和REVIEW_EVENT_IDENTITY_G96为准。

first16 cap累计return270、route负218中normalized GAE正103，中位−.018278；navigation285、负202中正135，中位+.125764，非总体率/Q。control首次256301.25/residual5050/clawback−54318.75；route正39454.141/负−109734.899，capture3267168继续主导。JAX full每百万有效tick window949.004429/kill585.685173/stock612.766991/loose29.116405/custody497.929040/capture496.294104/transfer67.700910；退休四scene实际tick全部0。override后current/ring/fixed实际valid百分比2.364618/8.848485/14.482442，sampled league条件9.234887/34.317545/56.447569，分母分列。曲线更新至g96；r19已预声明的FIFO容量4096因素仍未安装，r18保持原封存源至128终态，后继先启动再等待最终报告。只在map806与不同开墙状态有证据，多地图泛化仍未证明。旧“实际取包后无路/动态封路/需预开返程”解释维持撤回。


r18实际历史抽样补证（20261019）：ACTUAL_HISTORY_AGES_THROUGH_G96逐代校验保存的exposure hash、source Actor digest及事件counts，无新对局/优化。g5–8历史返程768槽平均query age2.949，早期冻结658 distinct中255个被复习348槽；g9–16历史返程1536槽平均age4.030，仅126个早期输入被复习156槽。g17–32历史返程3072槽平均age4.035，早期输入复习0；g33–64和g65–96历史各6144槽平均age4.141/4.040，早期exact输入各仅1个历史槽，current另2/1槽，属于再次遇到相同输入标签而非旧generation记录永久保存。历史采样实际age覆盖1–8、并非只反复抽最新状态；同次重复、跨代复习、FIFO residency、重新访问必须分开。r19应核对实际age拓展到1–16及旧输入真正进入loss的频次，不能只用容量或resident宣称遗忘已解决。槽计数在optimizer reuse前，不是独立样本数或梯度贡献；不涉及PPO/其他teacher支持。绘图使用既有隔离plot依赖，训练环境未安装/变更依赖，曲线已更新至g96。


r18固定终态与r19接续（20261019）：r18正常128代/268435456 fresh ticks，compute6248.497s、wall7228.768s，atomic_stop=false；128代全部finite、Actor/Critic真实非零，退休scene及无原生custody授权计数全0。GPU_RELEASE_RECEIPT核验devices0..7且compute_clients0后，原supervisor立即启动新namespace control_route_v38_r19_causal_clean的technical_smoke01；未重启旧smoke、未等待r18 g128 CPU评估。r19父r18最终g128、种子与预声明相符，plan及declaration实际目标FIFO4096/additions256/replay128/current64-history64，保留返程DAgger×2。此处只记录终态与接续，正式首代非零和fresh实际buffer/父权重核验随后补证，能力结论仍待完整最终证据。

固定run `runs/je10_bun_20261018/control_route_v38_r18_causal_clean` 完成128代/268,435,456 ticks；compute=6248.497s、wall=7228.768s。已先启动预声明后继 `runs/je10_bun_20261018/control_route_v38_r19_causal_clean`；本次后继配方选择依据 `Full r17 curve and fixed r18 g0-g64 show actual native return payload fits at its update but declines later (.940414 g4 to .646384 g64); all658 frozen distinct inputs are resident g8 and absent g16/g32 because capacity2048/additions256 expires at8 generations. Test capacity4096 only, preserving additions, replay128, current/history64, native authority and x2 return loss; expiry is evidenced, its causal contribution remains unproven. No capability gate or final evaluation wait.`，正式训练不等待最终CPU结果。终态原始marker位于run，未入Git。


### r19正式非零确认与r18完整终局分析（20261019）

按终态顺序已先完成8卡释放核验、启动r19全新technical_smoke01、通过finite/Actor-Critic真实非零/保存恢复，立即启动正式128代。r19正式g1/2097152 fresh ticks，8副本exact，Actor/Critic更新L2=3.063514/2.262621、各128次Adam更新；smoke对应3.173400/2.250872。FORMAL_FRESH_INITIALIZATION逐叶证明父r18g128 raw Actor/Critic、完整pool及retention parent exact，EMA=raw、Adam/FIFO/计数全零、新seed RNG exact、8192 learner全部空手、退休scene0、cache独立、resume=None；实际FIFO4096与plan一致。187项新训练源全匹配；相对父184项训练源唯一改变是CAPACITY2048→4096的精确单常量patch。首更确认已主动通知，未等待最终评估、报告、能力阈值或人工批准。

r18已正常128代，所有固定点COMPLETE、224个快筛产物hash、七点各八项消费marker、完整128代事件、终点封存源恢复及后继正式非零全部核验，TERMINAL_FIXED_EVIDENCE_VERIFIED仅代表技术与证据完整性，绝非能力门禁。完整固定曲线如下，未选最佳点、未平滑：

|g|JS Hunter难/中/易，各64|Python Tactical/64|真实drop交付/16|自然拾包/16|自然交付/16|
|---:|---|---:|---:|---:|---:|
|0|0/2/7|9|6|6|3|
|4|0/2/6|11|3|4|3|
|8|0/3/10|6|5|6|3|
|16|2/5/8|12|4|4|2|
|32|0/3/5|8|6|5|4|
|64|0/0/8|9|10|3|3|
|96|0/1/8|12|9|9|5|
|128|0/2/7|5|12|5|2|

终点真实drop12/16对父+6、对g96+3；自然2/16对父−1、对g96−3，难Hunter仍0，Tactical5/64对父−4、对前−7。Hunter终点death0/0/1、自伤全0，Tactical death0/self0；不能把击杀保持下降误称自伤安全崩溃，也不能用较低死亡掩盖进攻不足。八路线death/sugar/drop/人工near/mid/far/recovery/full交付0/3/12/16/15/12/3/2（各16）；自然窗口→目标7/7是局级共现、首2/5、持续583/881、推进647/2318、拾5→交2、恢复14/15；首次控制16/归因3→pop16主动10→因果kill3存活3，native即时kill0另列，death5/self0。掉包局部改善，完整自然路线与战斗没有跨后段稳定提升，整体目标未完成。人工携包/库存reset仍仅诊断；实际取敌包必有可逆入站路径的不变量不变，旧“无路/动态封路/预开返程”结论继续撤回。

普通return query32469=首29028+追加3441，delivery40、quota异常0。实际return49152槽/27143 distinct，current/history各24576；全部131072槽/66136 distinct。冻结g1–4实际return1536槽mass g0 .427384→g4 .940414→g8 .886865→g16 .750031→g32 .665385→g64 .646384→g96 .629641→g128 .625045；658 distinct终点 .676080/greedy448，未selected215 .834648。前512本轮return .690160/355，父固定return .806349/417、父navigation .728716/384。终点动态FIFO6807有效、重复11组、no-common-pair组/行0，return2131行mass .853937/greedy1889，敌源2064/己源67/空手0。动态分布高拟合不能替代同输入保持。g4/g8旧输入全resident、g16/g32为0，实际历史age1–8且过期后仅极少重新相遇；expiry的机械事实已证，遗忘完整因果仍未证。隔离CPU float32复核早段下降同向，避免把总体遗忘全部归咎CPU bf16翻转。

唯一Review seed2026101800/game0：drop tick82真实拾后立即实际左移.18、gain−.157497，hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd，拓扑不变；CPU bf16左.975993/右.000959，完整输入exact0、同几何近邻12中greedy11、mass .815307。返程97tick实际推进66/反向29/停2，最终774交付、learner死亡0。自然559真实取敌包、1132交付、learner死亡0；271返程tick中270同拓扑可比，实际推进175/反向88/停7，首实际反向tick576/hash1b36971b71be6ceccd1d9911e8d27cb8c2cdf68efe5fbd571be7da69f1a36c1b，左移.18/gain−.167500、拓扑不变，CPU bf16左.795888/右.008690，完整输入exact0、同几何近邻0。自然取包后先有COMBAT再恢复C_RETURN，不以局部标签把本局成功称整局失败。

首反向的JS零value奖励GAE：drop−.017872535、前一tick−.018116280，变化+.000243744；natural+2.498277646、前一tick+2.361496945，变化+.136780701。对应JS observable route reconstruction为+.000176953/−.003821094；它使用网页重建，不能与共享连续距离的实际gain、checkpoint训练reward或normalized GAE等同。正近似GAE也不是该反向动作优于替代方向的Q证据。训练first16 cap累计return366、route负298中normalized GAE正131、中位−.039694；navigation365、负264中正182、中位+.137672，均非总体率/Q。逐tick原生event/hash sidecar修正旧顶层death.player身份注释；补充全段扫描首次漏输入导出、已补齐重跑，失败log保留，不影响训练/evaluator。

reward累计control首次338539.75/residual6610.75/clawback−71957.5；route正52215.974/负−145919.667，capture4360464主导。JAX full每百万有效tick window933.470550/kill580.238895/stock615.130064/loose29.326840/custody495.290499/capture497.415160/transfer67.793526。override后current/ring/fixed实际valid百分比2.274911/8.922219/14.842567，sampled league条件8.776427/34.207892/57.015682；实际动作源与抽样族分列。JAX事件机会归一化、Python Tactical及生产JS Hunter仍严格分开，不互当同一bot成绩。

证据排序与下一步：第一，同一实际输入当代拟合后持续下降，且exact FIFO支持按8代过期，是最直接可机械验证的问题；r18窄范围旧移动KL例外没有消除总体下降，不能宣称已证明保留项是唯一根因。已启动r19只扩FIFO窗口8→16代，检查实际residency、进入loss的age与复习频次及冻结fit整曲线；resident本身不等于有效优化支持。第二，固定网页失败完整输入exact0、自然同几何近邻0，以及原生steering/phase切换说明联合状态覆盖不足；若fit保持而网页仍反向，优先该因素。第三，g8尚resident就已下降，须继续拆解实际history抽样与完整PPO/其他teacher共享参数干扰，当前没有完整optimizer反事实证据。第四，路线分量与总GAE可不同向，先保留证据，不凭cap直接调reward。r19是顺序parent/seed后继，不能冒称same-parent因果控制；全阶段共享FIFO延长且可能稀释近期抽样的风险明示。r19健康固定128继续，CPU/JS唯一owner与只读consumer跟随八个固定点，短点只诊断。只在map806与不同开墙状态有证据，多地图泛化未证明；改善后仍须补最弱漏斗和防遗忘。


r19 g0协议复核（20261019）：PARENT_EVALUATION_EXACT_G0证明生产JS Hunter、control_funnel、Python Tactical及八路线完整聚合与r18g128逐字段完全一致；独立fresh父权重逐叶exact已由FORMAL_FRESH_INITIALIZATION证明，不要求pickle序列化字节相同。正式前两代耗时188.33/175.47s，与父run前两代188.84/181.48s相近，不能把首次编译摊销误称FIFO运行性能故障；继续健康128代。已生成r20同配方连续后备（父r19g128，train2392524396/technical3457791706，128代/8卡，FIFO4096、返程DAgger×2不变），用于保持终态无评估等待；后续固定证据允许在终态前以测试通过的最小proposal替换，不构成能力门禁。


### r19 g4：当代拟合可达，完整自然交付未增加（20261019）

g4 COMPLETE及八项只读诊断hash复核完成，无新策略对局或诊断标签导入。JS Hunter难/中/易0/3/9、Python Tactical12（各64），对父/前g0均0/+1/+2、+7；Hunter death0/1/1/self全0，Tactical death5/self5（父0/0），击杀上升伴随安全代价，后段继续跟踪，不能只报击杀改进。八路线death/sugar/drop/人工near/mid/far/recovery/full交付3/3/9/16/16/8/2/2（各16），真实drop对父−3、自然不变。自然窗口→目标6/6为局级共现，首4/6、持续1002/1388、推进1049/2348、拾5→交2、恢复12/17；首次控制18/归因7→pop17主动14→因果kill6存活5，native即时kill0另列，death2/self0。人工carrier和库存reset仍仅诊断；无能力门禁，不按早点评分停止健康128。

return query856=首735+追加121、delivery0、quota异常0。实际return1536槽/610 distinct、current/history各768；全部4096槽/1884 distinct。冻结实际return集合mass g0 .397276→g4 .948284，distinct .472741→.953085/greedy596，未selected246 .645187；说明真实loss监督可拟合，不是奖励/标签完全不可达，但g4尚不能检验16代窗口保持。前512本轮return .887219/465、父固定return .711393/374、父navigation .603290/309。动态FIFO3640有效、重复11组、no-common-pair组/行0，return856行mass .864600/greedy759。当前与冻结输入分母不同，聚合高拟合不替代网页完整状态泛化。

固定seed2026101800/game0逐tickhash与事件身份核验：drop真实拾包tick82后首动作仍左移.18、gain−.157497、拓扑不变，hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd，JS近似GAE−.016532374，CPU bf16左.898392/右.072893；完整输入exact0，同几何近邻12中greedy10、mass .845576。drop追取81tick均正推进，返程148tick推进114/反向21/停13，最终419交付、learner死亡0。自然1141真实取敌库存、1321交付、learner死亡0；171返程tick中170同拓扑可比，推进166/反向3/停1。首局部oracle不一致tick1147 CPU名义下.786420/右.213281，但首真正实际反向直到tick1232：hash c9411dc1f168f7836da4ddc105992f253360395d487ecdcdc7f0657d13927cf0，右移.18、gain−.158928、同拓扑、JS近似GAE−.025223007。自然局部偏航状态exact0/同几何近邻0；不能将该局部标签或候选动作gain直接当实际反向，更不能将最终成功称失败。原生死亡向量和身份sidecar继续纠正旧顶层death.player注释。

first16 cap return20、route负12中normalized GAE正0、中位−.283183；navigation13、负11中正7、中位+.050746，非总体率/Q。control首次9141.75/residual252.75/clawback−869；route正1980.333/负−2904.966，capture115752主导。JAX full每百万有效tick window1196.048673/kill453.284341/stock434.113481/loose.639029/custody344.649466/capture160.183188/transfer11.289507；这是新环境早段累计，不能与父长run终段累积率作无条件因果比较。override后current/ring/fixed实际valid百分比2.340294/6.739233/9.827286，sampled league条件12.455338/35.334642/52.210021，Python/JS/JAX及分母严格分列。r19已健康到g14附近，唯一evaluator正处理g8；后继r20保留预声明连续后备，不以本点改权重或停训。下一关键机械证据是g16仍resident且是否真正被抽入历史loss，再看同输入fit是否保持，随后与完整交付/战斗/安全曲线共同诊断。


### r19 g8：旧输入仍全resident，拟合已下降；交付分层仍分化（20261019）

g8 COMPLETE及八项消费、事件身份、实际运动与FIFO存留只读审计已齐，未生成新对局/优化。JS Hunter难/中/易0/1/7、Python Tactical14（各64），对父0/−1/0、+9，对g4 0/−2/−2、+2；Hunter death0/1/0、自伤全0，Tactical death/self均0（较g4各−5）。八路线death/sugar/drop/人工near/mid/far/recovery/full交付2/7/11/16/16/11/1/2（各16）。真实drop对父−1/前+2，自然对父/前不变；自然窗口→目标8/8为局级共现，首2/5、持续916/1387、推进911/3410、拾6→交2、恢复18/21；首控21/归因3→pop21主动18→因果kill3存活3、native即时kill0另列、death2/self0。自然交付未稳定增加，人工carrier只诊断，无能力停训门禁。

ordinary return query1757=首1519+追加238，delivery0、quota异常0。实际return3072槽/1389 distinct，全部8192槽/4078 distinct。冻结1536槽/610 distinct mass .948284(g4)→.864611(g8)，distinct .953085→.876204/greedy554，未selected246 .900695。真实FIFO容量4096，g4/8早期1536槽/610 distinct全部resident、query generation1–4合法return记录856仍全在；g5–8历史return768槽平均age2.638、early输入复习283槽/215 distinct。此时并未到16代窗口边界，拟合已下降，说明容量过期不是所有下降原因；r19是否改善后续保持须看g16/32完整曲线，不按resident单字段宣称有效。前512本轮return .875038/460，父return .700447/364、父navigation .646079/330。动态FIFO7339有效、重复17组、no-common-pair0，return1757行mass .871910/greedy1571；仍是动态in-sample分布，不能替代冻结输入保持或holdout。

唯一固定Review seed2026101800/game0：drop tick83真实拾后立即左移.18、gain−.257496、同拓扑，hash84bf72f44f31bd7ee9f6ad6be982fa8b51c52e1094ccfd10fd2413e29dce2722、JS近似GAE−.016593271；返程133tick实际推进59/反向59/停15，tick477 learner死亡丢包，敌人取回并在493交付，后928取己方包获胜；learner本局交付0。自然tick1109真实取包、1344交付、learner死亡0，181返程tick推进175/反向6；首反向tick1117/hash ef437c9ee7e727c29dc699b45c1413a1b27b87333d001c085053456406aa1384，上移.18、gain−.173218、同拓扑、JS近似GAE−.034928155。固定失败与成功都保留，不换seed、不将局部oracle不同直接称整局失败，继续使用原生death向量与身份sidecar。

first16 cap return27、route负19中normalized GAE正2、中位−.295306；navigation27、负20中正11、中位−.004923，非总体率/Q。control首次20629.25/residual528.5/clawback−2446.75，route正3745.961/负−7892.873，capture257064主导。JAX full每百万有效tick window1134.796138/kill546.786041/stock526.552216/loose11.629722/custody451.100848/capture388.130157/transfer36.874729；override后current/ring/fixed实际valid百分比2.856495/5.095663/10.812954，sampled league条件15.497432/26.821662/57.680906，分母分列。187项训练源与plan封存匹配；r19继续固定128，唯一evaluator正在g16，后续重点是g16真实resident与被抽入loss支持，再联合完整路线/战斗/安全判断后继最小因素。


r19 g8按实际复习支持分层（20261019）：FROZEN_RETENTION_BY_REHEARSAL_G8对同一610个真实早期输入、固定g4/g8 Actor及实际exposure逐hash关联。395个g5–8未再进入history loss的输入mass .977526→.865449、平均变化−.112077；复习1次170个 .931083→.915372/−.015711；复习2次29个 .870214→.860214/−.010000；复习3–5次16个 .733665→.754549/+ .020884。current未重新遇到这些输入，history总283槽/215 distinct。说明resident不代表有loss支持，且高拟合旧输入可能在误差加权下较少被抽；但组间g4难度显著不同且抽样依赖误差，禁止将此关联冒称复习剂量因果估计。无新增游戏/标签/优化，CPU bf16非GPU。继续g16/32真实residency、历史抽样及fit整曲线，再决定后继最小因素；当前r19不热改、不设停训门禁。


### r19 g16：历史窗口机械成立，拟合下降未消除（20261019）

g16 COMPLETE及八项消费、真实运动、身份、FIFO存留和按复习分层均完成。JS Hunter难/中/易1/1/8、Python Tactical10（各64），对父+1/−1/+1、+5，对g8+1/0/+1、−4；Hunter death0/2/1/self全0，Tactical death/self均0。八路线death/sugar/drop/人工near/mid/far/recovery/full交付2/5/6/16/16/9/3/4（各16）；真实drop对父−6/前−5，自然对父+2/前+2。自然窗口→目标10/10为局级共现，首5/8、持续1094/1675、推进1219/4563、拾7→交4、恢复25/31；首控32/归因8→pop32主动26→因果kill8存活8、native即时kill0另列，death6/self0。自然交付局部上升但掉包回落，仍无跨场景稳定提升结论，不按短点停健康run。

ordinary return query3717=首3258+追加459，delivery3、quota异常0。实际return6144槽/2905 distinct、全部16384槽/8231 distinct。冻结1536槽/610 distinct全部仍在实际4096 FIFO，早期query generation1–4的856条原始return记录也全在；对比父r18同点已0，窗口改动机械验证成立。g9–16实际history return1536槽平均age5.760、最长15，早期输入复习408槽/286 distinct，说明更长历史确实进入loss；g5–16总复习691槽/385 distinct，225个早期输入未再进入history loss且current均未重新遇到。冻结mass .948284→.864611→.820727，distinct终点 .857427/greedy537；父自身同点 .750031，但不同parent/seed/输入集不构成同父因果改善证据。未selected246 .930504；前512本轮return .868080/458，父return .737934/382、父navigation .687899/362。动态FIFO14401有效、重复40组、no-common-pair0，return3720行mass .861308/greedy3292，敌源3629/己源91/空手0。

g16复习分层不支持简单单调剂量解释：0次225输入mass .987509→.928122；1次200输入 .960433→.832135；2次113输入 .938053→.847463；3–5次65输入 .864399→.732293；6次以上7输入 .702795→.630507。各组初始难度和误差驱动选择不同，不能从g8未复习组下降较大便归结唯一根因；resident与进入loss都非完整优化贡献。RETURN_SAMPLING_SUPPORT_G8对同一静态FIFO/固定Actor计算24个return draw/设备的预期：epsilon .05/.2/.5下早期610输入预期被选distinct57.618/61.666/63.022；此前无history的395输入38.845/40.749/41.164，改下限的机械增益有限，暂不把它列为首选修正。此为静态预期非真实未来抽样或效果，未优化/导入训练。

唯一Review seed2026101800/game0：drop tick82真实拾后立即左移.18/gain−.157497、拓扑不变，hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd，JS近似GAE−.014109412；CPU bf16左.750456/右.003394，exact0、同几何近邻22中greedy21、mass .864499。返程21tick实际推进7/反向14，后续因phase切换不在C_RETURN统计不能称传送或断路；tick280携包死亡，敌人取回286交付、再599交己方包，learner交付0。自然同一game0全局无原生取包/交付/死亡，无return机会，不换seed。16局自然4/16交付与固定game0无机会分列，顶层death.player旧注释仍以原生身份sidecar纠正。实际取敌包必有可逆入站通路，旧无路/动态封路/预开返程解释继续撤回。

first16 cap return47、route负36中normalized GAE正7、中位−.194961；navigation53、负40中正25、中位+.136063，非总体率/Q。control首次42720.75/residual1010.25/clawback−6772，route正7057.856/负−17202.564，capture512064主导。JAX full每百万有效tick window1076.060123/kill575.856600/stock545.541235/loose17.774187/custody475.211392/capture437.136737/transfer52.330094；override后current/ring/fixed实际valid百分比2.872560/6.451401/10.470188，sampled league条件14.720040/32.220677/53.059283，分母/模拟器严格分开。当前r19不热改、继续固定128；进一步只读比较实际新旧native监督输入共享参数方向，明确不包含完整PPO/Adam，也不将局部梯度关联当实际总优化因果。后续g32核验旧样本按16代正常过期，再看g64/96/128完整曲线决定最小后继。


r19 g16实际native监督梯度补证（20261019）：NATIVE_REPLAY_PARAMETER_ALIGNMENT_G16只比较固定g16 COMPLETE Actor上的旧g4真实610 distinct return输入与g16已记录实际1024槽payload（current/history各512），不重建新对局/查询或优化。原生loss系数.2、return×2，payload各子集保留全1024分母；旧reference用自身610分母。共享参数方向cosine：all payload对旧return −.124777，current all −.203675，history all +.016262；current return −.136394、history return +.113922、navigation −.087907。支持不仅旧teacher、不同native监督分布也可能在共享参数上竞争的假设；当前history返程方向局部支持旧输入，current则局部相反。全payload范数.043760、旧reference2.556734来自不同条件集合和归一化，不能冒充实际梯度贡献/裁剪后参数更新大小。g16 Actor是更新后参数，实际query-time g15 Actor未保存；不含PPO/entropy/other expert/teacher/clip/Adam，不能据该瞬时关联推断完整因果或直接调大loss。后续固定点复核方向与实际历史支持，再选择最小候选；当前r19健康训练不变。


### r19 g32：自然交付6/16；旧记录按16代过期，保持下降继续（20261019）

g32 COMPLETE、八项消费及固定原生录像/存留/历史抽样审计齐全。JS Hunter难/中/易2/1/7、Python Tactical6（各64），对父+2/−1/0、+1，对g16+1/0/−1、−4；Hunter death0/1/0、自伤全0，Tactical death4/self4。八路线death/sugar/drop/人工near/mid/far/recovery/full交付6/3/11/16/16/12/5/6（各16）。真实drop对父−1/前+5，自然对父+4/前+2；自然窗口→目标13/13为局级共现、首3/9、持续1257/1557、推进1298/4312、拾8→交6、恢复25/30；首控32/归因9→pop31主动19→因果kill9存活9、native即时kill0另列、death4/self0。自然中点改善与早期输入下降并存，战斗保持仍波动，不宣称跨后段稳定、不设门禁。

ordinary return query7376=首6552+追加824、delivery6、quota异常0。实际return12288槽/6042 distinct，全部32768槽/16296 distinct。冻结1536槽mass .948284(g4)→.864611(g8)→.820727(g16)→.725461(g32)，610 distinct .759395/greedy472，未selected246 .835872。前512本轮return .777564/404、父return .732823/378、父navigation .769150/399。当前FIFO13469有效、重复35组、no-common-pair0，return3662行mass .837069/greedy3165。原始query generation1–4记录已全部过期，FIFO有效generation17–32；仅有1个early exact输入（对应冻结加权5槽）因新query再次出现，不能误称原记录继续resident。g17–32实际history return3072槽平均age7.923，覆盖1–16，early exact复习78槽/72 distinct（含过期前支持及再遇见），不是所有重复均无效。窗口机械实现与进入loss已证，但不足维持早期拟合。

隔离CPU float32同一实际输入复核mass g0 .394798→g4 .947115→g8 .862858→g16 .819442→g32 .724626，原CPU bf16同向；g4当代current/history各192槽，从 .995490/.994265降至g32 .784838/.685933。总体下降不能解释为仅CPU bf16 greedy翻转，未测GPU bf16，未重建完整优化器因果。新旧native监督梯度g32补查首次因最老history在本代写FIFO时过期，post-g32存储缺少g16行；现从hash校验的g28原子恢复buffer补实际输入，不重新查询、不将缺失当零、不在g28评估Actor，失败log保留，补查尚在运行。

固定Review seed2026101800/game0：drop tick82真实拾后首动作左移.18、gain−.157497、同拓扑，hash d83552d5b37c10dcf53600b8b5dee2c8fdf3ef3de30bd057839821d5418552cd、JS近似GAE−.017585675；CPU bf16左.990285/右.002346，exact0、同几何近邻16中greedy11、mass .655228。返程133tick推进106/反向27，最终455交付、learner死亡0。自然同game0无实际取包/交付，learner死亡1，2400tick平局；无return机会，不换seed，不把总体6/16交付与此固定局混为一谈。事件身份仍以原生向量和hash sidecar为准，route_deviation只指局部不同。

first16 cap return78、route负61中normalized GAE正23、中位−.052846；navigation94、负72中正46、中位+.134156，非总体率/Q。control首次83516.25/residual1808.75/clawback−15919.25，route正13393.669/负−35125.716，capture1050504主导。JAX full每百万有效tick window993.499933/kill569.663140/stock567.654330/loose24.723813/custody477.478637/capture463.593567/transfer62.780825；override后current/ring/fixed实际valid百分比2.488852/6.530304/13.416833，sampled league条件11.634509/29.332475/59.033016，分母/模拟器严格分列。r19健康继续128，g64/96/128用于后继诊断，不因中点升降停止。

返程威胁proxy只读排查（20261019）：RECORDED_CARRIER_THREAT_GATE_THROUGH_G32对既有game0全部before/after hash核验。敌人free且learner携敌包、Manhattan≤4的drop tick数g4/8/16/32=189/251/168/240，其中当前敌人与learner的wall/brick/pushable四邻格步行距离>5仅3/0/0/0；自然9/54/0/0中为1/5/0/0。没有敌人步行场INF案例；这不是返程可达性审计。少量隔墙代理误差存在，但不能解释g16/g32 drop大段COMBAT或首反向，暂不优先改威胁阈值；长步行距离也不证明安全，远程炸弹/爆炸/连续碰撞尚未包含，JS无training hysteresis，不当作JAX总体错切换率。地图资产只一张带bun_bases的原生806，当前不同出生点和开墙状态不等于多地图泛化；后续仍需独立拓扑泛化验证，整体任务未完成。


r19 g32 native监督梯度复核（20261019）：从hash校验的g28原子恢复buffer补回20条本代写入时已淘汰的实际history输入，仅提取buffer字节、未在g28评估Actor；1024槽payload原始输入全部精确重建。固定g32 Actor对旧610 distinct return的native监督参数cosine：all +.549834、current all+.527669、history all+.514595、current return+.605075、history return+.597239、navigation+.028110。与g16轻度负向不同，不能把瞬时native分布冲突认定持续根因，也不能仅据g16改current/history比例。仍是更新后固定Actor的监督梯度关联，不含真实query-time Actor、完整PPO/teacher/entropy/Adam，未新增优化或游戏。保留证据后继续后段固定曲线，r19数值源不变。


## 用户替代目标：r19安全暂停并集成（20261019）

r19已在g58原子checkpoint/STATE提交后PAUSED，atomic_stop=true，121634816 fresh ticks，剩余70代；不是COMPLETE/FAILED。先阻断supervisor，r20命名空间未创建；trainer/CPU-JS evaluator/consumer均退出，独立核验8卡compute contexts0，active marker stale。恢复必须显式指令。停止/哈希核验在本地USER_PAUSED.json保留；最后完整固定评估g32，g64/96/128未执行。旧不可达/预开返程结论仍撤回。r12–r19共61个完成固定点评估逐shard核验；在基于最新origin/main的独立集成树发布必要代码/测试/总结，原dirty训练树和run保留。训练连续owner职责由此被用户暂停指令替代，不再自动接续。


### 暂停与main发布交付完成（20261019）

r19最终保持g58 PAUSED，r20未创建；再次核验trainer/evaluator/consumer live0、本实验8卡compute contexts0。暂停时8卡已全释放，交付时其他任务已接用，未干预。main提交 `71280717e43fac1c23f7b9221c7a2ab7d5ceb3ae` 已同步，独立集成树和canonical main干净。Pages run39206621979同SHA completed/success；公网build-info同SHA，12个站点资源HTTP200且字节/hash与提交一致，三份Markdown远端逐字节一致；桌面1440×1000与手机390×844真实浏览器61点、r19筛选5点、暂停状态与Review入口均通过，pageerror0。线上结果 https://allwellll.github.io/qqt-RL/training.html 。完整收据在忽略的 `runs/je10_pause_delivery_20261019/DELIVERY_COMPLETE.json`；所有旧run/checkpoint、dirty代码和stash保留；恢复必须显式用户指令。

## RETENTION-20261019-R1：同父、多seed、返程核心库保持筛选（预注册）

2026-10-19，状态 PREPARING / WAITING_RESOURCE。旧r19保持g58 PAUSED，r20禁止；本实验独立命名空间 `runs/retention_20261019/r1`，不恢复旧run。外部分析仅作假说输入，原文SHA256 `9bf64fd21950d22bcbb6fc9b4e28cdd6a2fb326bbd8e8cc84fe99b8b6f9d77fd`。

父点 `runs/je10_bun_20261018/control_route_v38_r19_causal_clean/formal/bun392_bridge_pfsp8/gen000032`，raw policy SHA256 `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`。已验证完整state SHA、full digest、finite、8副本一致及Adam4096步，含配套Critic与完整optimizer。选点因为自然交付6/16、拾包8/16且已完成固定评估；Tactical6/64、自伤4/64的弱点必须独立监测，不能称综合最强。g58无固定能力评估，不作为父点。所有臂都继承同一raw Actor/Critic与父联赛池，统一fresh Adam/RNG/env/EMA=raw/FIFO/core；不冒称完整optimizer续训。父optimizer完整性已审计而主动不用，以避免恢复旧环境携包与seed耦合。

真实代码核验：现行单Critic以实际bootstrap计算GAE(.995,.95)，advantage在每副本整批归一化；已使用合法5×3联合动作集质量和合法返程贡献×2。没有证据证明多Critic必然改善；第一轮不改Critic、GAE、KL、PPO、奖励或loss强度。已有早期610输入在FIFO过期后合法质量回落，支持优先检验持久数据覆盖，但不是已证明根因。撤回旧“取包后无路／动态封路／额外预开返程通道”解释；实际取包必有可逆入站通路。

| 项目 | 预声明 |
|---|---|
| A0 | 当前r19配方：4096 FIFO、每代256 additions/device、actual replay128=current64/history64、error ε=.05、合法返程×2 |
| A1 | 与A0唯一数值区别：历史半部中已有ordinary-return槽的一半（按槽次序交替）改从分层持久核心库抽取；槽数、current半部、其他phase和loss不变 |
| 核心库 | 每device512状态，4层各128：natural scene8/其他因果scene × enemy alive/dead；仅kind3、非preparation、真实custody证明、learner alive且携敌包的合法joint行。稳定双uint32内容指纹去重；每层保留指纹优先级最小128个不同状态（确定性bottom-k内容采样，非FIFO，不声称k-center）。重复不增加配额，时间与来源代不作为内容多样性。库fresh开始，不导入旧run/评估状态；全量checkpoint恢复。无教师KL、无人工携包、无状态重置回放 |
| 配对训练seed | 2026101901 / 2026101902 / 2026101903；每seed先A0再A1，原子创建配对，完全同父，独立run/state/cache/eval |
| 固定预算 | 每臂每seed32代；8×1024×256=2,097,152 fresh ticks/代，128 Adam更新/代；每run67,108,864 ticks，总402,653,184 ticks；不能因短点评估停run |
| 保存/开发评估 | g0/8/16/24/32，recovery每4代；unique CPU/JS owner依次消费，训练不等待；尾部K=3取16/24/32 |
| 固定dev协议 | Python Tactical v2 64局；真实JS Hunter难/中/易各64，seed20261019；原固定spawn64。八路线各32，seed2026102100..1931，greedy assist-off。保留原八任务定义，人工carry仅诊断、库存/recovery为受控fixture，natural_full2400为主要终点；不合并分母 |
| 保留test协议 | 仅筛选全部结束后，各run g32八路线64局，seed2026102900..2963；Hunter三档/Tactical各64、RNG20261029；不用于调参。候选确认再扩5训练seed和更大对局分母，不能把3seed筛选叫稳定显著成功 |
| 主统计 | 每训练seed报告自然交付率尾部均值、最差值和0/8/16/24/32梯形AUC；A1−A0配对差，训练seed为重复单位；列全seed、配对方向、精确宽CI/seed bootstrap，禁止峰值替代保持 |
| 非劣/防规避 | 相对A0尾均值，Tactical safe-win与Hunter各档不降超过5百分点、self-death不增超过2百分点；报告机会归一化kill/control、存活、bombs/game和交战暴露，bombs/game或有效对抗机会下降>20%即标疑似规避，不判安全保持成功。小样本CI不支持则明确不确定 |
| 筛选选择 | 先检查全部fixed run技术有效；自然尾均值改善且至少2/3训练seed方向一致、尾部最差/AUC不反向并满足非劣点估计，方列“待扩大验证候选”；否则保留A0或无候选。CI跨零不得声称显著。test若反向即不选A1，记录最弱漏斗再预注册最小改动 |
| 停止 | 固定32代正常结束；仅技术故障/硬安全恶化/用户停止可提前中断。NaN/OOM/恢复损坏应修复；无能力/admission/promotion门禁。全部筛选结束不自动接128代或r20 |

技术流程 RED→GREEN→相关回归→最小GPU smoke（真实Actor/Critic非零、finite、保存、fresh CPU恢复），随后立即正式训练。GPU当前全部为其他项目任务，禁止停止或抢占；唯一等待器在无compute clients且显存<1GiB、利用率<5%的连续两次检查后再启动，启动每个run前重查；flock保证本实验单控制owner。等待期间只做CPU准备；资源等待不冒称已smoke或已启动。唯一CPU评估owner读取原子COMMITTED、写每点COMPLETE及全队列EVALUATION_COMPLETE；技术失败marker独立记录，评估不能停GPU。权威状态 `runs/retention_20261019/r1/STATUS.json`，不提交原始产物。终态主动通知；不产生或调用任何嵌套代理。

R1正式启动前执行补充（无数值变量）：各run cache仍为独立目录；允许复制同源码seal、同shape、同臂技术smoke已生成的JAX编译产物以减少重复冷编译。复制只涉及编译文件，不复用RNG/env/FIFO/core/optimizer/checkpoint，不使用旧r19缓存；记录cache来源收据。所有臂同样处理，JAX仍按编译key验证适配性。

R1实施记录：两臂技术smoke各g1=2,097,152 ticks，8副本finite，Actor L2=3.1095404625、Critic L2=2.2381048203；fresh CPU读取全量live并验证full digest/next RNG通过。A1首代历史core为空，更新与A0相等，符合只替换历史槽设计；首代核心149条（8设备合计），natural层0，后续需报告可达性与实际采样。所有配对inputs逐文件SHA相同，plan唯一数值开关retention_core。新run g0 Actor/Critic逐字段等于父点、Adam0、FIFO/core valid0；不以pickle字节重序列化差异冒充参数差异。

正式A0 seed2026101901已启动固定32代，唯一CPU评估owner已同时消费g0；dev实际RNG20261019、路线seed2026102100起32局。串行与4任务并行执行真实父模型八任务×两席位结果及Review文件逐字节一致，合并拒绝identity/seed/分母不一致；仅CPU执行调度改变。holdout为独立authority_retention_test命名空间及当日报告，不覆盖dev。中间能力无停训权；正式首非零更新待单独记录。

R1评估执行补充：保留test阶段在全部dev完成后，先对共同父g0按test种子真实评估一次，再评估六个g32；父结果以显式shared-parent receipt复用，不能当6个独立训练seed样本。这样每个test g32有相同协议的真实父分母与delta；不把dev父分数挪作test分数。此项只增加CPU评估，不改训练数值/预算。

R1首次正式非零更新确认：A0 seed2026101901 g1原子COMMITTED，2,097,152 fresh ticks、128 Adam更新/副本；finite，8副本Actor L2=3.0318853855、Critic L2=2.3002736568。`FORMAL_NONZERO_CONFIRMED.json`已生成。唯一训练owner持续固定32代，唯一CPU评估owner同时消费g0，无能力门禁；未恢复r19或启动r20。此处为首次正式启动证据，不是筛选完成或能力结论。

R1 A0 seed2026101901固定终态：完整32代/67,108,864 fresh ticks，atomic_stop=false；8卡无compute app、显存0/利用率0释放验证通过，配对A1按预注册接续，g32评估异步。所有32代finite且Actor/Critic真实非零，无提前能力停训。
# 20261019 RETENTION-R1 用户语义暂停、库存审计及独立假设队列

用户当前暂停覆盖自动接续：`runs/retention_20261019/r1/SEMANTIC_AUDIT_HOLD.json` 存在即阻止 owner/资源等待/direct worker/retry；A1 首代编译已退出，仅 g0 原子点，无 g1 或原子退出记录，标 `EXCLUDED_INVALID`，禁止恢复/能力结论。训练与原评估进程树释放，8 卡无 compute app；不重启旧筛选、r19、r20。

固定自然 JS Hunter normal 32 seed/点逐次取包、原生 `_bunUpdate` 阶段及同货物交付审计完成：共同父 g0 库存 dead/sugar/free=5/7/6，A0 g32=10/4/3；原总交付10/7中，从窗口库存取包后交付8/6，自由存活库存取包后交付2/1。64局所有旧字段一致，两固定 game0 Review 字节一致。9个自由取包真实状态调用生产 phase/prestock/DAgger 均不激活，即使缓存 window/custody=true；未证实库存老师漏门控。游戏规则允许自由偷库存，当步 fresh 奖励0、无专门惩罚，后续真实返程/24交付仍可给 PPO 回报。inactive PBRS 折扣历史项可能非零但不依赖库存方向，不能宣称 route 通道恒零。

最小修复仅为统计：原 raw pickup/delivery 不再解释为合法窗口完整链；新增交互阶段取包/货物 token 资格与对应交付计数，旧点缺测资格=null，保留真实 loose/contact 与返程，未改训练奖励/损失/DAgger×2。真实 JAX 奖励及门控12项通过，相关41项通过，JS新增7案例通过；门控 mutation RED 明确为负控制而非原门控失效，恢复原函数 GREEN。完整根因、事件分母、首偏航/hash、污染范围与限制见 `reports/retention_experiment_20261019.md`；证据与冻结哈希 `runs/retention_20261019/stock_window_audit/BASELINE_FROZEN.json` 不入 Git。旧“取包后无路/预开返程通道”解释继续撤回。

新增两候选全文已读取；顺序冻结为 H1目标相关首次真实拓扑增益 → H2时间展开首次威胁组件/连泡 → 各自有效无投机后才组合。独立单变量配对seed/预算/固定评估/反作弊合同见 `reports/reward_hypotheses_20261019.md`。尚未实现或启动，不混入当前修复，不给任意炸砖、放泡或未爆存量逐tick付费。语义审计参考冻结不等于零自由偷包策略；是否额外抑制原生自由取库存动作仍是待明确的训练目标，自动筛选保持暂停。

## 20261020 用户明确语义及 RETENTION-R1 接续

本条覆盖上一条暂停/待决：用户选择保留环境合法的自由存活自主偷包，不加惩罚/硬禁，老师及库存PBRS目标继续仅在原生死亡/糖浆窗口开放；用户已更新语义 marker 为 audit_complete=true/automatic_training_allowed=true。启动器TDD已RED→GREEN，只有明确授权状态三字段均满足才解除 hold；EXCLUDED目录仍永久拒绝。12项启动器/结果/评估回归通过。

共同父、3配对seed、32代/臂及数值配方不变；替代 A1 `runs/retention_20261019/r1/A1_s2026101901_resume_20261020` 全新 run/state/cache/eval、同 seed2026101901、inputs逐文件等同已完成A0的共同父输入，不恢复被排除旧A1。fresh技术 smoke `smoke_A1_resume_20261020` 后直接正式 A1，再预注册seed1902 A0/A1、seed1903 A0/A1；其余旧未启动namespace无checkpoint，可正常fresh初始化。原 seal保留，新resumption seal仅行政/评估变更，核验训练数值源码无变化。H1/H2不混入。

唯一owner已启动，技术smoke仅finite/真实Actor-Critic非零/保存恢复；固定checkpoint g0/8/16/24/32由独立唯一CPU/JS owner按原64局Hunter/Tactical及32局路线异步消费，训练不等报告。终态核验8卡释放再接续，主动通知正式首非零/终态/故障，不因短点评估停健康run。继续比较原生总自然交付，同时独立保留窗口资格与free-stock解释分量。当天跟进及完整结果写 `reports/retention_experiment_20261020.md`，异步点报告 `reports/retention_evaluation_20261020.md`。

20261020 新A1正式g1已COMMITTED并核验policy/state hash、8副本exact、Adam各128步：Actor L2=3.0318853855，Critic L2=2.3002736568，finite/nonzero，2,097,152 fresh ticks。g0父参数精确、fresh Adam/FIFO/core/custody；g1与配对A0 policy字节相同（旧core为空，首代替换0，新增133条），不作能力归因。新smoke有限非零和fresh完整CPU恢复已通过，g0异步固定评估COMPLETE。唯一owner继续该A1固定32代及余下四run，不因短点评估停止。

RETENTION-20261019-R1：替代A1 seed2026101901于20261020固定32代正常终态，67,108,864 fresh ticks，finite、Actor/Critic非零，atomic_stop=false；原子终态与两次空闲资源核验收据保存在该新namespace。旧排除A1未恢复，r19未恢复、r20未启动。唯一owner按预注册队列接续A0 seed2026101902；独立CPU/JS继续g32评估，不以中点能力决定是否接续。g8/16/24自然交付7/7/6（各32），配对A0为5/6/5；最终点与其余seed尚未完成，不作候选成功结论。当天证据 `reports/retention_experiment_20261020.md`。

RETENTION-20261019-R1接续：A0 seed2026101902正式g1非零确认，2,097,152 fresh ticks，Actor/Critic L2=3.0249445438/2.2277004719，8副本finite；fresh初始化独立审计通过，固定32代继续。

RETENTION-20261019-R1：A0 seed2026101902固定32代正常终态（67,108,864 fresh ticks、finite/nonzero、atomic_stop=false），两次8卡空闲释放收据已保存；唯一owner接续同seed A1，g32评估异步继续，不等待能力阈值。

RETENTION-20261019-R1：A1 seed2026101902正式g1原子非零确认，2,097,152 fresh ticks，8副本finite，Actor/Critic L2=3.0249445438/2.2277004719，Adam各128步；g1 policy与配对A0字节相同（首代core空、替换0，新增151条）。fresh共同父/Adam/FIFO/core/custody审计通过，继续固定32代，无能力门禁。

RETENTION-20261019-R1：A1 seed2026101902固定32代正常终态，67,108,864 fresh ticks、finite/nonzero、atomic_stop=false；8卡释放两次核验收据完整。唯一owner接续A0 seed2026101903，不等待第二组g32能力结果；CPU/JS评估独立继续。

RETENTION-20261019-R1：A0 seed2026101903正式g1非零确认，2,097,152 fresh ticks，8副本finite，Actor/Critic L2=3.1064996719/2.2711923122，fresh共同父/零Adam/FIFO/custody审计通过。继续固定32代，CPU/JS异步点独立。

### RETENTION-20261020-R2：core历史返程曝光单变量后继预注册

20261020预注册，尚未启动，独立于R1与H1/H2。证据：R1前两seed同一训练状态core联合拟合g8约.91、g32约.75，驻留样本仍回落，自然尾部配对收益方向相反；稀疏历史曝光只是待反证假设。共同父仍r19已评估g32 raw Actor/matching Critic/pool（policy SHA a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95），各arm全新Adam/env/RNG/FIFO/core/cache/eval；不从R1末点继承。父r19保持暂停，不启动r20。

臂C0（执行名A0）=R1 half-core（既有history ordinary-return槽一半，8卡合计实际96/代、每副本12），C1（执行名A1）=同core全部既有history ordinary-return槽（8卡合计实际192/代、每副本24），唯一变量是这部分数据来源。保持分层库容量/去重、当前批次/其他历史阶段数据、总replay128行/副本（current/history各64，8卡合计1024）、PPO/老师系数、合法返程×2、课程/对手配置不变；无新oracle查询/标签/loss/奖励；不导入评估数据。禁止无事件预置携包，合法自由存活自主取包仍允许，库存老师/PBRS当前窗口门控不变。

新namespace `runs/retention_20261020/r2/`，三配对seed2026102001/2002/2003，各臂32代，8×1024×256=2,097,152 fresh ticks/代，合计402,653,184；checkpoint g0/8/16/24/32，恢复点每4代。执行顺序C0→C1按seed配对。训练数值不安装到R1，待R1完整证据冻结后独立启动；不是能力/admission/promotion阈值门禁。任何正常固定run不因中点效果停止；仅NaN/OOM/损坏等技术故障恢复或用户显式停止。技术smoke只查focused RED→GREEN、finite、Actor/Critic真实非零及原子保存/fresh恢复。

CPU/JS唯一owner、完成marker与原协议一致：Tactical v2 64、真实JS Hunter难/中/易各64，八路线dev各32，RNG20261019、路线2026102100–2131、greedy/assist-off。dev复用须注明相关性。新一次保留集RNG20261031、路线2026110100–2026110163各64，在R2全部dev完成后统一消费；R1保留集不作为R2独立test。人工携包仅诊断，实际库存资格token分量与合法自然总拾交并列。完整曲线、尾g16/24/32均值/最差、AUC和配对训练seed bootstrap；同时报告Hunter/Tactical、机会归一化、实际对手暴露、固定首次偏航/优势和core驻留/实际采样。

反证规则：若C1同状态拟合提高而自然尾部不提高，则曝光不足不能作为充分解释；若拟合仍回落，则优先查优化项竞争/样本分布，而非继续盲增回放。候选选择沿R1探索性标准（自然配对均值/一致性、尾最差/AUC/test及战斗/安全/规避代价）报告，不控制健康run。三seed仅筛选，不宣称统计非劣。H1目标拓扑与H2时间威胁仍后续独立，按H1→H2，未组合。

接续边界澄清：为遵守用户持续占用GPU的明确要求，R2在R1六个固定训练原子终态及释放核验完成后即可技术smoke/正式启动；R1开发与保留集CPU评估允许继续异步，R2评估owner只在全局唯一CPU锁释放后执行。R1最终CPU成绩不作为开训门禁。R2独立同父单变量已预注册，不根据中途成绩改配方。

RETENTION-20261019-R1：A0 seed2026101903固定32代正常终态，67,108,864 fresh ticks，finite/nonzero，atomic_stop=false，两次8卡空闲释放核验通过。唯一owner已先接续A1 seed2026101903，同共同父点fresh，g32评估独立继续，无能力门禁。

RETENTION-20261019-R1：A1 seed2026101903正式g1原子非零确认，2,097,152 fresh ticks、8副本finite，Actor/Critic L2=3.1064996719/2.2711923122，Adam各128步；policy与配对A0g1字节一致，core新增154、首代替换0，fresh共同父审计通过。继续固定32代。

用户最新指令覆盖连续接续策略（20261020）：当前`A1_s2026101903`仅完成既定32代并原子保存，随后暂停全部训练。R2、H1、H2、下一seed及任何后继均禁止，必须等待用户明确继续。现已先保存USER_PAUSE_AFTER_FIXED_RUN决策，并以R2既有授权检查实际验证launch blocked；R2无owner/worker、尚未smoke。R1是队列最后一个固定run，无剩余训练领取项，当前worker不发停止信号、不改预算。g32正常终态及8卡释放收据完成后退出训练行政owner，CPU唯一评估可继续消费已提交checkpoint。该暂停不是语义审计失败、能力门禁或技术故障，不撤销合法自由存活自主取包决定。

用户暂停已完成（20261020）：`A1_s2026101903`正常固定g32终态，67,108,864 fresh ticks，`atomic_stop=false`，32代全部finite/Actor-Critic真实非零。g32 Actor/Critic L2=3.3332078457/2.9448683262；8副本Adam各4096步。最终`state.pkl`与`policy.pt`SHA256均与COMMITTED精确一致，policy SHA `4d4f2f78709d6c534290a4f8343a7f9b8f1c289cc5e9935fe66d7bcf1da8c706`。未打断任何一代/当前run。worker正常退出；在训练owner完成两次8卡释放核验后，仅退出其等待CPU的行政进程，独立CPU/JS评估保留。追加独立核验8卡全部0MiB/0%、compute app为0。暂停证据保存于 `runs/retention_20261019/r1/USER_PAUSED.json`。

R1/R2既有launch授权均实际验证BLOCKED；R2无smoke/正式run/owner启动，H1/H2未实现载入，不领取下一seed或任何后继。仅已提交点CPU评估可继续，不触发训练。须用户明确“继续”才能恢复授权。此用户暂停覆盖所有此前自动接续、健康run不停与保持GPU占用要求；当前固定预算已经完整兑现。聊天已即时通知终态与释放，不等待评估报告。

RETENTION-20261019-R1终态：6run×32代，402,653,184 fresh ticks；候选 `A0_or_no_improving_candidate`；配对自然尾均值差[0.04166666666666666, -0.04166666666666666, -0.03125]，seed-bootstrap95%[-0.039409722222222214, 0.0258680555555555]。完整逐seed/点报告 `reports/retention_experiment_20261019.md`。未恢复r19、未启动r20，无能力门禁。

## 用户撤销暂停并恢复R2（20261020）

已全文读取`runs/retention_20261020/r2/USER_RESUME.json`，明确授权覆盖此前暂停。R1六run训练和全部dev/一次holdout评估已完成，原训练不恢复，旧EXCLUDED命名空间保持隔离；已将既有完整评估归档汇总。R1三seed自然尾差+4.17/-4.17/-3.13个百分点，holdout差+1.56/-4.69/-3.13个百分点，不支持稳定自然保持收益；R2检验增大core曝光是否足够，仍是可证伪假设。

R2原预注册不变：同父raw Actor/matching Critic/pool、fresh optimizer/env/RNG/FIFO/core；seed2026102001/2002/2003各A0/A1固定32代。唯一因素history普通返程core比例1/2→全部，总replay128行/副本不变，合法返程DAgger×2不变，库存老师/PBRS窗口门控不变、自由存活自主取包合法，无合成携包。两臂技术smoke仅finite/真实非零/保存fresh恢复，不作能力门禁，完成即正式开训。H1/H2无导入或组合。

已核验GPU八卡0MiB/0%、无compute app；原训练/评估均无残存进程。R2历史暂停证据保留，新授权显式覆盖；R2授权检查通过，不解除R1或EXCLUDED入口。focused与技术smoke/正式收据在R2独立namespace保存。中点仅用于诊断，固定run不因短时成绩停止；每个终态/停止报告最后代、原子点、资源与后继，完成筛选后以全曲线、固定漏斗与未消费保留集决定下一独立单变量。

R2恢复技术smoke A0首更新：g1、2,097,152 ticks，finite/nonzero，Actor/Critic L2=3.1095404625/2.2381048203，8副本真实更新；独立g0审计共同父参数精确、零Adam、空FIFO/core/custody，resume=null。原子保存/新CPU进程恢复尚待收据，非正式能力结果。

R2两臂技术smoke已完整通过：各1代2,097,152 ticks，8副本finite/Actor-Critic真实非零，L2=3.1095404625/2.2381048203；原子保存与新CPU进程完整状态恢复通过，g0共同父精确/Adam零矩步数/空FIFO-core-custody分别独立审计。首代core空替换0，两臂相同是预期，不是比例效应。正式A0_s2026102001已启动，固定32代，无评估门禁。

正式命名空间未使用smoke learner state，仅按arm复用6个编译cache文件，逐文件SHA与覆盖前hash保留在`SMOKE_COMPILE_CACHE_REUSE.json`；各run的cache目录仍独立。不改变模型、optimizer、env/RNG/FIFO/core、源码封存或采样配方。原R1预热cache中同key不同内容由本次smoke编译文件原子替换，仅编译缓存。正式fresh与g1另有收据。

R2首个正式非零更新确认：A0_s2026102001 g1原子COMMITTED与FORMAL_NONZERO_CONFIRMED均通过，2,097,152 fresh ticks，8副本finite，Actor/Critic L2=3.0889728069/2.3120789528，Adam各128步、replicas_exact=true。正式fresh审计共同父精确/零Adam/空FIFO-core-custody，OWNER_STATE restore_checkpoint=null；不是smoke续训。g0固定评估完成，八路线交付10/10/24/32/31/28/6/10，Tactical6/64、JS Hunter难中易2/2/8各64，与共同父一致。唯一GPU训练owner和独立CPU/JS owner正常，当前固定32代与后继队列继续，无能力门禁。

R2 A0 seed2026102001固定g32正常终态：67,108,864 fresh ticks，atomic_stop=false，终态finite/nonzero，state/policy原子SHA精确。g32 CPU/JS异步评估正在运行；唯一owner正核验8卡释放后接续fresh A1同seed，不等待能力成绩。聊天已主动报告终态。

R2 A0 seed2026102001释放接续：g32 state/policy原子SHA验证通过，worker正常退出，owner连续两次8卡0MiB/0%、compute app0，GPU_RELEASE_RECEIPT完整；已先启动fresh A1_s2026102001，restore_checkpoint=null，独立g0共同父/零Adam/空FIFO-core-custody/8副本审计通过。唯一训练owner持续，唯一CPU/JS评估独立，未等待能力报告。

A0 g32开发：八路线交付5/5/16/32/31/24/5/5，Tactical11/64、JS Hunter难中易1/6/7各64；自然拾10局/11事件、交5局，死亡12/self0。自然完整曲线10/5/3/6/5，尾均值14/96=14.58%，仍低于父10/32；待配对A1及其余seed，不择峰。首控60/归因13、最终爆破59/归因击杀13/存活13；窗口库存拾7交5，自由拾4交0。库存first8/14、oracle669/1184、推进685/6603；返程first15/29、oracle852/1087、推进884/1185。所有分项/首偏航/近似优势原计数SHA与COMPLETE绑定，R2新保留集未消费。

R2 A1 seed2026102001正式g1原子非零更新确认：2,097,152 fresh ticks，8副本finite，Actor/Critic L2=3.0889728069/2.3120789528，Adam各128步，fresh审计已通过。g1 policy与配对A0字节SHA相同，历史core空/替换0，符合唯一比例变量尚未生效；不是能力结果。继续固定32代，g2起核验full替换192/代。

R2正式比例生效证据：配对seed2026102001 g2 A0/A1实际core替换96/192，8卡总replay1024行不变。g2 rollout奖励分量、事件漏斗、场景计数、最终对手动作来源和因果初始化审计逐字段相同；current/history各512、返程各192及drop/stock/navigation/opening槽数均相同。首代policy已字节一致，因此g2是相同初始行为分布下仅历史返程core来源比例不同的正式证据；后续状态分布分叉是处理后结果，不再要求逐状态相同。PAIRED_G2_SINGLE_FACTOR_AUDIT保留，仍不代表能力改善。

R2 A1 seed2026102001固定g32正常终态：67,108,864 fresh ticks，atomic_stop=false，32代finite/nonzero，最终state/policy SHA与COMMITTED精确，worker正常退出；两次8卡0MiB/0%/compute app0释放核验收据完整。唯一owner先接续A0_s2026102002 fresh同父，A1g32 CPU/JS评估异步继续，无能力门禁。聊天已即时报告终态和后继。

## 第一配对完整开发终态与第二组接续

seed2026102001 A0/A1自然曲线10/5/3/6/5对10/7/8/11/9，尾g16/24/32均值14/96=14.58%对28/96=29.17%，差+14.58个百分点；尾最差3/32对8/32。单训练seed，仅开发筛选，R2保留集未消费，不能称稳定改善或非劣。A1g32八路线7/6/20/32/32/27/6/9，Tactical12/64、JS Hunter1/2/4各64，自然拾12局/13事件、交9，死亡11/self0。首控47/归因15、最终爆破47/归因击杀15/存活15；窗口拾7交6、自由拾6交3；库存first15/18、oracle633/1079、推进660/4552；返程first57/82、oracle1602/2029、推进1627/2040。需全三seed战斗/安全与新holdout，不能仅凭此方向选候选。

同一769早期训练core A0g8/16/24/32 .7733/.7712/.7346/.7413；A1 .8640/.8401/.8312/.7891。末点natural dead/alive .8287/.7698仍高A0 .7280/.6220，其他alive .8166略低A0 .8186。增曝光拟合收益仍有，但随代回落，不是彻底解决保持。

第二组A0_s2026102002正式g1已原子提交且FORMAL_NONZERO_CONFIRMED通过：2,097,152 ticks，8副本finite，Actor/Critic L2=3.1127281189/2.3808686733、Adam各128步；fresh同父参数精确、零Adam、空FIFO-core-custody、restore_checkpoint=null。继续固定32代，先接续后报告，不等待第一配对能力结论。

20261021跨日续做RETENTION-20261020-R2，当前第二seed A0g9健康、g8异步评估中；唯一owner持续，第一配对完成后已正常接续，H1/H2未载入，R2新保留集未消费。当天关键结果续记 `reports/retention_exposure_experiment_20261021.md`，原预注册/namespace/数值封存不变。

R2 A0 seed2026102002固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false，32代finite/Actor-Critic非零；g32 L2=3.2869260311/2.7883985043，8副本Adam各4096步，state/policy SHA与原子COMMITTED一致。worker正常退出，唯一owner两次核验8卡全部0MiB/0%、compute app0并保存GPU_RELEASE_RECEIPT，已先启动同seed fresh A1（attempt0、restore_checkpoint=null），CPU/JS独立继续A0g32；不等待能力结果，无能力门禁。后继g1非零及fresh审计另行确认。

R2 A1 seed2026102002正式g1原子非零确认：2,097,152 fresh ticks，8副本finite，Actor/Critic L2=3.1127281189/2.3808686733、Adam各128步；state/policy SHA与COMMITTED一致，g1 policy与配对A0字节相同，首代core空替换0、加入151条。fresh审计共同父精确、零Adam/空FIFO-core-custody、restore_checkpoint=null，无旧排除状态；继续固定32代。后续g2核验full替换192/代，g0异步评估继续。

第二配对seed2026102002 g2单变量实际审计通过：配对rollout奖励分量、事件漏斗、场景/最终对手来源及因果初始化审计完全一致；current/history各512、返程各192等槽数相同，仅core来源实际替换96→192。后续分布差是处理后变化，不能要求逐状态相同。

R2 A1 seed2026102002固定32代正常终态：67,108,864 fresh ticks、atomic_stop=false、全代finite/nonzero，g32 Actor/Critic L2=3.1667368412/2.8653600216，Adam各4096步、8副本exact；最终state/policy SHA与COMMITTED一致。worker正常退出，两次8卡全部0MiB/0%/compute app0收据完整；唯一owner已先接续A0_s2026102003 fresh共同父，CPU/JS独立继续A1g32，无能力门禁。后继正式g1及fresh审计另记。

第二配对完整开发自然曲线A0 10/7/7/9/3、A1 10/3/12/9/4，各点32局，尾19/96=19.79%对25/96=26.04%，差+6.25个百分点；尾最差3/32对4/32。前两seed尾差+14.58/+6.25个百分点，仅2个训练seed且R2新保留集未消费，不能宣称稳定改善。g32 A1八路线9/8/21/32/32/26/5/4，Tactical10/64、真实JS Hunter2/1/8各64，死亡12/self0，终态曲线仍明显回落，需保留战斗及完整漏斗取舍。完整计数TWO_PAIR_DEV_INTERIM_20261021.json；第三seed已先接续，不据能力门禁停训。

R2第三seed A0_s2026102003正式g1非零原子确认：2,097,152 fresh ticks、8副本finite，Actor/Critic L2=3.0218946934/2.2894899845，Adam各128步，最终policy/state SHA精确。fresh共同父、零Adam/空FIFO-core-custody审计通过，restore_checkpoint=null；继续固定32代，无能力门禁。

R2 A0 seed2026102003固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false，32代finite/Actor-Critic非零；g32 L2=3.2815392017/2.8508181572、Adam各4096步、8副本exact，最终state/policy SHA与COMMITTED精确。worker正常退出，两次8卡0MiB/0%/compute app0释放核验完整；唯一owner已先接续最后预注册臂A1_s2026102003 fresh、attempt0/restore_checkpoint=null，CPU/JS独立继续A0g32。后继fresh及首代非零另记，无能力门禁。

R2 A1 seed2026102003正式g1非零原子确认：2,097,152 fresh ticks，Actor/Critic L2=3.0218946934/2.2894899845，8副本finite/exact、Adam各128步，state/policy SHA精确；g1 policy与配对A0字节一致，core新增178、首代替换0。fresh共同父/零Adam/空FIFO-core-custody已通过，restore_checkpoint=null；最后预注册run继续固定32代，无能力门禁。
第三seed A0完整自然曲线10/7/6/9/8，尾23/96=23.96%、最差6/32；g32八路线10/5/25/32/31/21/3/8，Tactical12/64、JS Hunter0/1/6各64，自然死亡20/self0。配对A1及R2新保留集待完成，不择峰。

R2全部正式固定预算完成：六run各32代，合计402,653,184 fresh ticks，均正常atomic_stop=false、finite/Actor-Critic真实非零，最终原子点SHA核验通过。最后A1 seed2026102003 g32 L2=3.3903024197/2.8458507061、Adam各4096步/8副本exact，worker正常退出，8卡两次释放收据及追加实时0MiB/0%核验完整。唯一行政owner保持TRAINING_COMPLETE_EVALUATING，唯一CPU/JS仍消费最后dev，随后一次R2新保留集；当前无训练worker、GPU已释放，非技术故障或中点能力停训。后继尚未启动，依用户最新要求以完整冻结协议/保留集决定下一单变量，继续监督与实现准备；H1/H2未载入，r19/EXCLUDED未恢复、r20未启动。

## RETENTION-20261021-R3 预注册候选：认证返程 PPO 行贡献（20261021）

状态：R2六run固定32代及30开发点已齐，R2一次保留集尚余最后末点；本节先冻结后继设计，尚未建立R3 namespace、未GPU smoke/训练。按用户授权，以完整R2冻结结果后正式选择是否采用；不以能力阈值阻止健康训练，不恢复r19/EXCLUDED、不启动r20。H1目标拓扑和H2连泡威胁独立保留，均未载入。

父点：`runs/je10_bun_20261018/control_route_v38_r19_causal_clean/formal/bun392_bridge_pfsp8/gen000032`，原父policy SHA256 `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`。该父点有真实自然运包固定评估、matching raw Critic和完整pool，R2新保留集自然19/64；不得以暂停g58或R2末点替代。复用其raw Actor/matching Critic/pool，每run fresh零Adam/env/RNG/空FIFO-core-custody，独立run/state/cache/eval；只复制可执行JAX编译缓存并保留SHA，不复制smoke或历史learner state。

唯一因素：A0认证安全原生普通返程Actor clipped-surrogate行权重1；A1=0.5，保留原Actor半批分母、不重归一化。处理Gate复用已有真实custody、当前合法joint证书、自由存活、无danger/terminal、phase3、非preparation/退役scene资格；不按正优势或错误动作筛选、不读评估标签。其余奖励/PBRS、GAE/bootstrap、adv标准化、Critic、entropy、KL、合法返程DAgger×2、查询/replay行数及选择、课程/对手均不改。两臂共同core比例按R2完整冻结候选一次确定：若full-core满足既有探索性候选规则则均1，否则均0.5；之后不随中点评估改变。原生真实取包/掉包/接触转移有效，自由存活自主偷包合法，库存老师/PBRS仅当前死亡/原生糖浆窗口。

seed2026102101/2026102102/2026102103，每seed A0→A1顺序、8卡各32代，每代2,097,152 ticks、每run67,108,864，正式总402,653,184 fresh ticks。课程/对手沿R2原plan和原时间调度（combat_fraction=.35、bridge_fraction=.2，实际混合/弱对手及current/ring/fixed曝光另行统计），每副本replay128行/current-history各64、原×2返程监督不变。run序列固定，不因中点高低分替换seed或提前停止。技术故障从最近原子点恢复，健康固定预算完整兑现；技术smoke只finite/8副本Actor-Critic真实非零、原子保存/新CPU进程恢复，不是能力门禁。

固定dev g0/8/16/24/32：Python Tactical v2各64、真实JS Hunter hard/normal/easy各64、八路线各32；RNG20261019、路线2026102100–2131，greedy/assist-off，明确跨实验开发seed复用。独立新保留集RNG20261102、路线2026110300–3063，各64，所有dev后唯一owner一次消费，共同父只评一次；人工carry近/中/远仅诊断、不能推断原生取包后无路。Tactical/Hunter/路线/训练JAX bot分开报告。

固定漏斗：归因首次控制→爆破→击杀/存活；死亡/原生糖浆窗口→目标→首合法动作→持续推进→真实拾包→返程→交付；另含恢复威胁切换、自然完整局、战斗自伤、实际对手曝光、奖励分解、首次偏航Review状态hash/事件/目标/势能、实际collector优势与近似分解分列、learner-state DAgger/FIFO/core覆盖。后继增加两臂相同的纯旁路全量gate/偏离/标准化正优势交集，AST检查移除诊断语句后原worker完全相同；优化前计数不能冒充clip后/完整梯度。旧FIRST16缺完整gate，不能推定所有phase3偏航都被新处理覆盖。

统计/决策：训练seed为独立单位；尾g16/24/32均值/最差、全曲线AUC、新保留集、机会原分母、父点绝对下降和战斗/自伤/放泡一起判断，三seed仅探索筛选。沿R1冻结的候选描述（自然均值及至少2seed方向、尾最差/AUC/holdout不降、战斗点估计差不低于−5个百分点、安全差不高于+2个百分点、放泡减少代理）选择待确认配方，不构成训练admission/promotion/停训门禁。若不改善，不能继续盲降PPO，优先检验learner-state错误覆盖/查询、库存上游和课程漂移；若改善，扩大独立seed及评估分母再确认且补最弱漏斗。H1/H2单变量各自认证后再测，不默认组合。

owner/marker：唯一训练行政owner、唯一全局锁CPU/JS owner跟随已COMMITTED点异步消费，评估不阻塞当前固定run和矩阵内接续。DECLARATION/SOURCE_SEAL、TECHNICAL_COMPLETE、FORMAL_NONZERO_CONFIRMED、FIXED_RUN_TERMINAL、GPU_RELEASE_RECEIPT、EVALUATION_COMPLETE、SCREEN_COMPLETE分开保存；每run终态先核验worker退出和8卡两次释放后接续预声明队列并主动通知。全部训练固定预算结束后消费完整评估选下一单变量，继续本线程监督。proposal及技术证据：`reports/retention_ppo_proposal_20261021.md`；已完成RED/GREEN、真实PPO恒等/处理生效、74项focused及35项因果语义回归。正式生效选择依据另行回填。

## R2终态与R3正式生效选择（20261021）

R2六run各32代、402,653,184 fresh ticks及全部30dev/一次新保留集完成，所有worker和两owner正常退出，8卡0MiB/0%、compute app0。R2候选为`A0_or_no_improving_candidate`。自然尾均值A1−A0为+14.5833/+6.25/−4.1667个百分点，seed-bootstrap95%[−1.9097,+12.7778]个百分点；保留集自然A0=8/9/16、A1=15/7/13，各64，差+7/−2/−3局，平均+1.0417个百分点但方向仅1/3。共同父19/64，所有末点仍下降。战斗/安全冻结点估计无越界标记，第三seed Tactical放泡从4.849降至3.516/局触发代理标记；该标记不是已证明规避交战，但不支持无代价保持。完整报告`reports/retention_exposure_final_20261021.md`，不得宣称full-core已解决保持。

R3正式采用上述预注册唯一变量、两臂共同half-core（0.5）。选择依据按证据强弱：增加曝光三seed提高同状态拟合但自然收益不一致；固定同一末点仍驻留样本三seed均回落，反对单纯淘汰充分解释；真实collector偏航有正标准化优势，支持检验优化竞争，但缺完整历史gate/梯度因果，因此只做单变量反证。当前最小可验证改动是认证安全原生普通返程PPO行权重1→0.5，原分母不变，所有其他配方不变。若失败，不盲降PPO，回到库存上游/learner-state覆盖与课程漂移；H1/H2独立且未载入。

正式实验ID `RETENTION-20261021-R3`，新命名空间`runs/retention_20261021/r3`，seed2026102101/02/03各A0/A1固定32代；技术74项focused、35项因果语义回归通过，运行时别名RED/GREEN、真实PPO系数1精确baseline、减半处理生效及纯旁路计数AST隔离通过。先两臂1代技术smoke/原子保存/新CPU进程恢复，通过即正式开训；不等CPU网页/报告阈值或批准。R2所有源码seal458项仍精确；旧排除和r19保持只读、r20不启动。

R3实际启动：全新六run命名空间已prepare、DECLARATION/SOURCE_SEAL冻结，唯一owner启动A0技术smoke，独立唯一CPU/JS owner等待正式原子点。A0 g0原子提交及独立CPU fresh审计通过：共同父参数精确/8副本一致/Adam零矩零步/空FIFO-core-custody/resume=null，运行时worker调用点系数1安装收据完整。当前首代编译/执行中，尚未声明真实非零更新或技术smoke完成；后续需g1原子保存与新CPU进程完整恢复。两臂smoke通过即正式固定32代矩阵，不等待评估能力门槛。

两臂技术smoke现均完整通过：A1 g1 finite/Actor-Critic真实非零，L2=3.1049253941/2.2441422939，8副本exact、Adam各128步，state/policy原子SHA精确，新CPU进程完整恢复通过；fresh共同父/空Adam-FIFO-core-custody独立审计通过。A0系数1 g1 policy字节SHA与R2同seed基线精确相同，loss/奖励/漏斗/实际对手曝光逐字段一致；A0/A1同seed rollout奖励/漏斗/曝光/资格计数相同而更新policy不同，处理确实生效。Critic参数轨迹可因原全局gradient clipping耦合而变化，未修改Critic loss/targets，不能要求参数逐步一致。PAIRED_SMOKE_TREATMENT_20261021.json保存技术证据，不纳入能力结论。

owner技术通过后直接启动正式A0_s2026102101固定32代，attempt0、restore_checkpoint=null、g0已原子提交，未等待评估/网页/报告/批准。正式cache目录仅复制同臂smoke编译key，有逐文件SHA收据，无learner state；fresh审计及首个正式非零更新另行核验，CPU/JS唯一owner已独立消费g0。

R3首个正式非零更新已确认：A0_s2026102101 g1、2,097,152 fresh ticks，8副本finite/exact，Actor/Critic L2=3.0192410946/2.3436717987，Adam各128步，state/policy SHA与COMMITTED精确，FORMAL_NONZERO_CONFIRMED完整。正式独立fresh共同父/零Adam/空FIFO-core-custody审计通过，OWNER_STATE attempt0、restore_checkpoint=null，未使用smoke learner state。实际gate7566/2,097,152，joint偏离141、偏离且正标准化优势49、偏离且即时route负/正优势11；纯旁路诊断，非能力结果。唯一训练owner/8卡worker/CPU-JS owner正常，固定g0评估在独立CPU进行；正式run继续32代并按预注册队列接续，健康run无能力门禁。

R3 A0_s2026102101固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false，32代finite/Actor-Critic真实非零，g32 L2=3.2733540535/2.9868640900，8副本exact、Adam各4096步，state/policy SHA与COMMITTED精确（policy `ca5c411898cea3861ff549c61a2c89231f17320b537c0687d2bee1dfc97ad0b8`）。worker正常退出，owner连续两次8卡0MiB/0%、compute app0释放核验完成，GPU_RELEASE_RECEIPT完整；先接续fresh A1_s2026102101 attempt0/restore_checkpoint=null，worker真实调用点系数0.5已安装。唯一owner持续、独立CPU/JS消费A0g32，未等能力报告。A1 fresh与首代非零另行核验；聊天已主动通知固定终态与后继启动。

R3 A1_s2026102101正式g1非零确认：2,097,152 fresh ticks、8副本finite/exact，Actor/Critic L2=3.0346841812/2.3387567997，Adam各128步，原子state/policy SHA精确；fresh共同父/零Adam/空FIFO-core-custody已独立通过，restore=null。正式配对g1 rollout奖励分量、事件漏斗、实际对手曝光、因果初始化及全量gate/优势计数完全相同，更新policy不同；PAIRED_G1_SINGLE_FACTOR_AUDIT_s2026102101保存唯一系数实际生效证据。健康A1继续固定32代。

第一A0完整开发自然曲线10/5/5/8/7，各32，尾20/96=20.8333%，低于父31.25%；g32八路线7/7/17/32/31/24/10/7，Tactical13/64、真实JS Hunter0/2/6各64，自然拾7事件交7事件、死亡7/self0、归因首控6/击杀6、窗口拾6交6、自由拾1交1。末点返程转化好但取包较少，单点不能替代尾曲线与配对。CPU/JS已开始A1g0，正式GPU主线连续运行，无能力门禁。

R3 A1_s2026102101固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false、全代finite/Actor-Critic真实非零；g32 L2=3.2248437405/2.8970186710，原子state/policy SHA精确、8副本exact、Adam各4096步，policy `0f978ca6fbb5a529c31aa599ba4da7cd44cb99421a14ff9e11c0f65b3ffd5a92`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据完整；唯一owner先接续第二组A0_s2026102102 fresh attempt0/restore=null，CPU/JS独立继续A1g32。第一组两臂共134,217,728 fresh ticks，不据短点评估停止；后继fresh与正式g1非零另报。

R3第二组A0_s2026102102正式g1原子非零确认：2,097,152 fresh ticks，8副本finite/exact，Actor/Critic L2=3.0245463848/2.2556405067，Adam各128步，state/policy原子SHA精确；fresh同父/零Adam/空FIFO-core-custody独立通过，attempt0/restore=null。唯一训练owner/8卡worker持续、唯一CPU/JS消费g0，固定32代及剩余配对队列继续。第一组不佳未触发提前停训或临时改配方。

R3 A0_s2026102102固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false、全代finite/Actor-Critic真实非零；g32 L2=3.4430961609/2.9183962345，原子SHA精确、8副本exact、Adam各4096步，policy `28419b0db46be37d89d81a2d4b450e3db04e2665794e589dca2dd2dff429ac32`。worker正常退出，owner两次核验8卡0MiB/0%、compute app0释放收据完整；先接续fresh A1_s2026102102 attempt0/restore=null，CPU/JS独立继续A0g32。唯一owner持续，后继fresh/正式g1另报；聊天已主动通知。

R3第二组A1_s2026102102正式g1原子非零确认：2,097,152 ticks，8副本finite/exact、Adam各128步，Actor/Critic L2=3.0204660892/2.2478129864，state/policy原子SHA通过；fresh共同父/零Adam/空FIFO-core-custody、restore=null。配对g1 rollout奖励、漏斗、实际对手及gate证据完全一致，PAIRED_G1_SINGLE_FACTOR_AUDIT_s2026102102保存。继续固定32代，无中点能力门禁。对应A0完整自然开发10/9/6/5/3、尾14/96，g32 Tactical6/64、JS Hunter0/1/7各64；未消费R3新保留集。

R3 A1_s2026102102固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false、全代finite/Actor-Critic真实非零；g32 L2=3.3272516727/2.8860781193，state/policy SHA精确、8副本exact、Adam各4096步，policy `12a62aa03d19cec80dadfcd9eb9a1054683dde347f018859911e62a991aa9f51`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据完整；唯一owner已先接续第三组A0_s2026102103 fresh attempt0/restore=null，CPU/JS独立继续第二组A1g32。已完成四run正式268,435,456 ticks，剩余第三组两臂不因前组表现提前终止；fresh/正式g1另报。

第三组A0_s2026102103 fresh/g1独立复核通过：共同raw父参数精确、Adam零矩零步、空FIFO/core/custody，attempt0/restore=null；g1 2,097,152 fresh ticks，8副本finite/exact，Actor/Critic真实非零L2=3.1779475212/2.2785968781，Adam各128步，state/policy原子SHA精确。第三组继续固定32代；唯一CPU/JS已独立开始g8，不阻塞GPU。

第二组固定A1g4真实core752条的同状态拟合已齐：A0 g8/16/24/32 joint mass=.70843/.69044/.71063/.70082；A1=.89504/.87061/.80329/.77283。处理改善这批早期learner状态拟合，但A1自身从g8到g32回落，且自然尾部只比A0多1/96；训练core样本并非独立能力保留集，不能据此宣布保持成功。两组实际已选core目标质量约.79–.88，而current返程约.19–.22，提示层内均匀core可能覆盖较易状态；只是后继可证伪候选，尚未修改R3或选择下一配方。证据REPLAY_SOURCE_FIT_BLOCKS_20261021.json及matched_core_s2026102102；H1/H2仍独立未载入。

R3 A0_s2026102103固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false，全代finite/Actor-Critic真实非零；g32 L2=3.2988967896/2.9063594341，state/policy原子SHA精确、8副本exact、Adam各4096步，policy `27bbe9d43d85fee16f98ecf591aab0be5d0c423c30f003b63dc1556839cf090c`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据完整；唯一owner已先接续最后一臂A1_s2026102103 fresh attempt0/restore=null，唯一CPU/JS独立继续A0g32。五run已完成335,544,320 fresh ticks；A1 fresh/首代非零另报，健康run固定32代继续。

R3最后一臂A1_s2026102103正式g1原子非零确认：2,097,152 fresh ticks，8副本finite/exact、Adam各128步，Actor/Critic L2=3.1916861534/2.2593202591，state/policy原子SHA精确；fresh共同父/零Adam/空FIFO-core-custody、restore=null已独立通过。第三配对g1奖励分量、事件漏斗、实际对手曝光、因果初始化及全量gate优势交集完全一致，policy更新不同；PAIRED_G1_SINGLE_FACTOR_AUDIT_s2026102103保存唯一系数实际生效证据。唯一owner/8卡worker/CPU-JS继续固定32代，无中点能力门禁。第三组A0完整自然开发10/6/7/1/5、尾13/96，g32 Tactical9/64、JS Hunter2/4/5各64。

R3最后一臂A1_s2026102103固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false、全代finite/Actor-Critic真实非零；g32 L2=3.3333764076/2.8273777962，state/policy原子SHA精确、8副本exact、Adam各4096步，policy `9d2192c270e417f5d482dc3c4aefe51e846ef5b9eba53e597eb24cfd8e90f5f6`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据完整。六run全部固定预算402,653,184 fresh ticks兑现，无健康中点停训。唯一行政owner转TRAINING_COMPLETE_EVALUATING，唯一CPU/JS正消费最后dev、随后预声明一次新保留集；当前无GPU训练worker、8卡已释放。后继尚未启动，须完整曲线/保留集后选择；core层内采样候选已独立TDD准备但未建run，无H1/H2混入。聊天已主动通知训练终态、资源及后继状态，继续本线程监督。

## RETENTION-20261021-R4 正式预注册：因果core层内误差采样（20261021）

R3六run固定32代/30开发点齐备，自然尾配对−11/+1/+6局（各96），均差−1.3889pp，冻结候选必要条件“正均值”已失败；新保留集仍由原唯一CPU/JS owner完整消费、不改协议。此前“所有R3保留集齐再决定启动”的实施安排修订为：必要条件已逻辑确定失败时保留原PPO系数1，不等待CPU保留集阻塞后继GPU；不提前推广任何处理，也不按中点停止健康run。R4正式决策依据COMPLETE_DEV_COMPARISON_20261021.json的SHA冻结。R3最终完整报告仍须补齐保留集。

父点仍为干净raw共同父`runs/je10_bun_20261018/control_route_v38_r19_causal_clean/formal/bun392_bridge_pfsp8/gen000032`（policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`），matching Critic/pool。R4新run/state/cache/eval，零Adam、空FIFO/core/custody、fresh env/RNG，不恢复r19暂停状态或任何EXCLUDED，不启动r20。编译缓存仅复制SHA核验的可执行key，不复制learner。

唯一变量A0现有四因果层内均匀core采样，对比A1层内按`.05+1-current_legal_joint_mass`误差加权。原四层总概率保持均衡、全部合法行正支持、只替换原一半history普通返程槽；current和其他phase槽位、core资格/去重/保留quota不变，不新增标签/查询、不按未来成功或评估筛选，质量停止梯度。共同PPO系数1、core曝光.5、合法返程DAgger×2、每副本replay128/current-history64、奖励/PBRS/GAE/Critic/entropy/KL/课程/对手不变；自由存活自主偷包合法，库存老师/PBRS仅当前死亡/原生糖浆窗口。H1/H2独立未载入。

证据与反证：当前返程质量约.19–.22、selected core约.79–.88，core均匀替换覆盖FIFO误差采样部分槽；只读两A0末点中困难core行概率17–19%，层内加权可提高到57–60%，ESS约509降至204–213，有集中/过拟合风险。父在A1g4固定core质量.55–.63，后续训练fit更高但自然保持仍差，故更好core拟合不能替代能力终点。若仅fit提高而完整自然曲线/新保留集不改善，拒绝其充分解释，不叠加奖励或继续盲降PPO。

seed2026102104/2105/2106，固定每seed A0→A1，各8卡32代，每代2,097,152 ticks、每run67,108,864、总402,653,184。固定dev g0/8/16/24/32：Python Tactical v2各64、真实JS Hunter难中易各64，八路线各32，greedy/assist-off，RNG20261019/路线2026102100–2131。独立新保留RNG20261104/路线2026110500–0563，各64，R4所有dev后唯一owner一次消费，共同父只评一次。重复开发seed不增加独立训练重复；人工carry近中远仅诊断。

冻结候选/诊断沿现有尾均值/最差、AUC、至少2seed方向、新保留集、战斗/自伤/放泡点估计与完整固定漏斗；三seed仅筛选，不证明非劣。阶段首动作first_actions按目标段重入解释，不能当独立真实拾包分母；以pickup/delivery局与事件原整数补充分母。Review唯一固定首seed、状态hash/事件/奖励势能与近似优势对齐；实际collector GAE和近似JS分解分列，DAgger/FIFO/core和实际current/ring/fixed覆盖完整保留。

owner/接续：R3原CPU/JS继续保留集；R4唯一训练行政owner独立，CPU/JS评估owner等待仓库全局锁后接续，任意时刻只一条实际CPU/JS权威评估链。技术smoke只finite/Actor-Critic真实非零/原子SHA/新CPU进程恢复；通过即8卡正式，不等待能力阈值/网页/报告。fresh g1空core两臂应精确相同，另在旧core非空后验证真实处理差异。无NaN/OOM/损坏或用户停训指令不截断健康32代；固定终态先核验8卡两次释放再矩阵内接续并主动通知。全部run后据冻结完整证据迭代最小单因素。

实现与技术：采样器RED两项预期失败→GREEN，实际collector/core/FIFO/真实PPO相关51项通过；编排单因素/fresh/cache/调用点/逻辑确定拒绝不等CPU与未决不得提前推广回归16项。prepare/owner尚待执行，不能把测试通过称正式更新。方案详见`reports/retention_core_priority_proposal_20261021.md`。

R4已实际prepare并启动唯一训练owner，实际六plan除ID/schema外唯一差异core_error_priority，ACTUAL_MATRIX_INITIALIZATION_AUDIT保存；原PPO共同1，R3保留集继续同一权威owner，R4 CPU/JS owner等待全局锁不阻塞GPU。A0技术smoke g1、2,097,152 fresh ticks，8副本finite/Actor-Critic非零L2=3.1095404625/2.2381048203，fresh同父/零Adam/空FIFO-core-custody及原子/新CPU完整恢复通过；policy/EMA字节SHA与原基线smoke精确相同。owner自动接续A1，fresh同父/零状态审计通过，实际聚合调用点core_error_priority收据完整。A1尚在首代执行，不提前宣称两臂技术通过或正式训练更新。

R4两臂技术smoke完整通过：A1 g1 finite/8副本Actor-Critic非零L2=3.1095404625/2.2381048203，原子保存/新CPU完整恢复通过，fresh同父/零状态独立审计通过；空core两臂首代policy/EMA字节SHA精确相同，符合处理可达性预期，另待正式core非空后核验差异。owner未等待网页/能力评估，已实际启动正式A0_s2026102104固定32代，attempt0/restore=null，实际聚合调用点core_uniform收据完整；首代真实非零与fresh独立审计另报。R3完整保留集由原CPU/JS继续，R4评估owner在全局锁等待，GPU主线不阻塞。

R4首个正式非零更新已完整确认：A0_s2026102104 g1，2,097,152 fresh ticks，8副本finite/exact，Actor/Critic L2=2.9907333851/2.3123641014，Adam各128步，state/policy原子SHA精确、policy `77115bb15e1ee46f214421236f4f952240f60a3da018a1e2235a9342df559622`。fresh共同父/零Adam/空FIFO-core-custody独立通过，attempt0/restore=null；六实际run raw父输入逐文件SHA完全相同，计划唯一差异采样，运行时真实调用点core_uniform、共同PPO1正确。唯一训练owner/8卡worker健康继续32代；唯一实际CPU/JS链仍在R3保留集，R4评估等待全局锁并跟随已原子点，不阻塞GPU。聊天已主动报告，不以短点评估门禁或静默退出。

## R3完整根因分析与后继落实（20261021）

六run/30开发点/六末点新保留集及一次共同父评估全部完成，原训练和评估owner均正常退出；R4已先完成两臂技术smoke并正式g1有限/Actor-Critic非零，GPU训练不等报告。冻结完整候选仍为A0_or_no_improving_candidate。自然尾差−11/+1/+6局（各96），均−1.3889pp，seed bootstrap95%[−8.75,+5.1215]pp；保留自然A0=23/10/13、A1=9/15/17（各64），均差−2.6042pp。共同父18/64，两臂末点均值均低于父，后继父点保留原raw干净g32，不挑单seed高末点继承。战斗/自伤/放泡冻结点估计无标记，不能证明三seed非劣。

根因假设按证据排序（均非已证完整梯度因果）：

1. 自然状态分布与训练保留样本的有效覆盖错配。三组同一批真实早期core上，父joint质量.627/.550/.572，A0末点约.734/.701/.700，A1末点.817/.773/.762；fit升高但自然尾/新保留均不稳，说明训练core拟合不是能力终点。当前返程被选样本质量约.19–.22，core被选质量约.79–.88，层内均匀core覆盖了FIFO一部分误差采样槽，有“较易记录占据困难历史槽”的机械证据，但没有证明它是充分根因。
2. 统一减半同时削弱有益PPO返程更新。全六run实际覆盖约0.354–0.358%原Actor行，覆盖行中偏离且标准化优势正仅约1.50–1.64%，多数正优势认证行未偏离。正优势冲突确实可达，但一次统一缩放不能只归为抑制坏动作；全局clipping耦合存在，优化前计数不能当完整梯度/反事实Q。R3负均值不支持继续盲降系数。
3. 自然取包机会和交付转化共同变化，不能只给返程loss归因。第一配对保留拾36/22、交23/9、死亡13/21；第二拾18/26、交10/15、死亡17/18；第三拾25/24、交13/17、死亡17/22，各64，自伤均0。控制归因和恢复交战也改变，窗口事件可重复，pickup/window仅曝光计数，非唯一因果转化。
4. 固定first-seed Review证实有路偏航，不能回到旧“取包后无路/预开返程通道”解释。保留首seed2026110300中，A1第一组返程tick973 action1/oracle3、距离34.57714；第三组tick1011 action1/oracle3、距离35.24857，tick+1/beforeSHA/精确oracle对齐通过，JS近似route奖励/GAE均负。实际collector GAE与JS近似分列；缺完整资格的first16行不能全部假定被PPO处理覆盖。其他run首seed无偏航时保留空，不择优换录像。

较弱或已排除解释：六run全代无退役训练scene/无真实custody返程污染；同seed首rollout完全一致，后续实际对手曝光配对接近，不能把调度暴露差异当充分根因。奖励为带符号净和，route负净和不能解释成正奖励占比。JS阶段first_actions会因目标段重入重复计数，已更正解释，不当作独立取包分母；生产JAX安全/开路老师与JS步行oracle口径分开。

单轮最小可验证后继已落实R4：共同PPO1和core曝光.5不变，只把四层core内部均匀采样改为现有FIFO同式误差加权；不新增奖励/标签/查询/行数，不改课程或对手，不混H1/H2。若仅fit增加而自然/保留无改善则拒绝充分解释；ESS下降与过拟合是预声明风险。67项采样相关/编排测试覆盖，实际两臂技术smoke通过、fresh恢复/原子/非零证据齐，首正式A0_s2026102104 g1已确认并持续健康训练。R4评估已在R3释放全局锁后接续g0，唯一实际CPU/JS链正常。

证据：`reports/retention_ppo_final_20261021.md`、FULL_COMPARISON_20261021.json、FINAL_MECHANISM_COUNTS_20261021.json、HOLDOUT_FUNNEL_COUNTS_20261021.json、TAIL_PHASE_COUNTS_20261021.json及各matched_core/OWNER_CONSUMED/固定Review。所有run/checkpoint/本地编排证据不入Git；H1/H2保持各自独立、当前未载入。

R4 A0_s2026102104固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false、全代finite/Actor-Critic非零；g32 L2=3.3449900150/2.8837053776，state/policy原子SHA精确、8副本exact、Adam各4096步，policy `9d476173229d47be8f9f35ed3b3fc4f366da56d4498ae79052f2163fb74d2fea`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据完整。唯一owner已先接续A1_s2026102104 fresh attempt0/restore=null，实际聚合调用点core_error_priority收据完整，独立CPU/JS继续A0g32；未等能力报告。A1 fresh/首正式非零另报，旧core非空后另核验实际采样差异，空core g1预期相同。聊天已主动通知。

R4 A1_s2026102104正式g1原子非零确认：2,097,152 fresh ticks，8副本finite/exact、Adam各128步，Actor/Critic L2=2.9907333851/2.3123641014；fresh共同父/零Adam/空FIFO-core-custody/restore=null独立审计通过。实际聚合调用点core_error_priority收据正确；空旧core时g1 policy/EMA字节SHA与同seed A0精确相同，PAIRED_G1_EMPTY_CORE_IDENTITY保存。该恒等符合预注册预期，不代表处理未安装，后续g2及非空core另查实际差异。CPU/JS独立评A1g0，正式32代主线继续。A0完整自然10/7/6/8/4、尾18/96，末点Tactical13/64、JS Hunter4/3/9，尚不作配对能力结论。

R4第一配对正式g2处理可达性审计通过：g1空core参数精确恒等后，g2两臂rollout奖励/漏斗/实际对手/因果初始化/全量资格优势计数精确相同，替换history返程均96槽；被选core joint质量A0=.9097366333、A1=.6568677823，实际Actor L2=3.1620743275/3.2235689163、Critic=2.4702453613/2.5043766499。相同输入下只采样处理已实际改变监督难度与更新，不是仅安装未调用。PAIRED_G2_SAMPLING_TREATMENT保存；这不是能力结果或门禁，健康A1继续固定32代。

R4 A1_s2026102104固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false，全代finite/Actor-Critic非零；g32 L2=3.4422564507/2.9078400135，state/policy原子SHA精确、8副本exact、Adam各4096步，policy `c0dbfc036a75418368c8326c9f0fa733edd764b35c5b591cb1a5fd90cff0227c`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据完整；唯一owner先接续第二组A0_s2026102105 fresh attempt0/restore=null，CPU/JS独立继续第一组A1g32。首配对共134,217,728 fresh ticks完成，不以短点无改善停止后续配对；fresh/正式g1另报。聊天已主动通知。

R4第二组A0_s2026102105正式g1原子非零确认：2,097,152 fresh ticks，8副本finite/exact、Adam各128步，Actor/Critic L2=3.0778558254/2.3090999126，state/policy原子SHA精确；fresh共同父/零Adam/空FIFO-core-custody、attempt0/restore=null独立通过。唯一训练owner/8卡worker/CPU-JS健康继续固定32代。第一组完整自然A0=10/7/6/8/4、A1=10/6/5/6/9，尾18/20（各96），差+2.0833pp，仅一训练seed不称稳定改善；两臂尾拾包事件均37、交18/20，死亡28/32/self0。core fit末点.72327/.81426，仍不能替代能力终点。全部剩余配对按声明继续，新保留集未消费。

R4 A0_s2026102105固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false、全代finite/Actor-Critic非零；g32 L2=3.3996174335/2.8454191685，state/policy原子SHA精确、8副本exact、Adam各4096步，policy `e003f4c17ddbf0e4295852cec923a6a4088f9fab4cc077ba3268b80c0bf25c83`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据完整。唯一owner先接续A1_s2026102105 fresh attempt0/restore=null，CPU/JS独立继续A0g32，未等能力结果。正式已完成三run201,326,592 fresh ticks，剩余三run继续固定预算，fresh/首代非零另报；聊天已主动通知。

R4第二组A1_s2026102105正式g1原子非零确认：2,097,152 fresh ticks，8副本finite/exact、Adam各128步，Actor/Critic L2=3.0778558254/2.3090999126；state/policy原子SHA精确，fresh共同父/零Adam/空FIFO-core-custody、restore=null独立通过，运行时core_error_priority正确。空core g1 policy/EMA与同seed A0精确相同，PAIRED_G1_EMPTY_CORE_IDENTITY保存；非空core g2另核验处理。A0第二组完整自然10/6/12/6/7、尾25/96，末点Tactical13/64、JS Hunter0/6/6。健康A1继续32代，唯一CPU/JS独立消费g0。

R4 A1_s2026102105固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false、全代finite/Actor-Critic非零；g32 L2=3.4318649769/2.9303400517，原子state/policy SHA精确、8副本exact、Adam各4096步，policy `bb5c299be0d52010e070c99940b211fe184c3be4ea17c01cec4436525fa5b47b`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据完整。唯一owner先接续第三组A0_s2026102106 fresh attempt0/restore=null，唯一CPU/JS独立继续第二组A1g32。正式四run共268,435,456 fresh ticks完成；剩余配对继续固定预算，fresh/首代非零另报。固定A1g4 core889条末点质量A0=.7162489631、A1=.8093909584，仍仅训练样本拟合诊断，不替代自然能力终点。聊天已主动通知。

R4第三组A0_s2026102106正式g1原子非零确认：2,097,152 fresh ticks，8副本finite/exact、Adam各128步，Actor/Critic L2=3.0214450359/2.3194615841；state/policy原子SHA精确，fresh共同父/零Adam/空FIFO-core-custody、restore=null独立通过，实际core均匀对照安装正确。第二组完整自然A0=10/6/12/6/7、A1=10/6/5/6/8，尾25/19（各96），差-6.25pp，与第一组+2.0833pp方向不一致；第二组尾真实拾包事件37/40、拾包局37/38、交付25/19、死亡39/39、自伤0。末点Tactical13/6、JS Hunter[0,6,6]/[0,4,5]各64。固定训练core拟合改善未在该seed转化自然保持，不提前停剩余固定预算，新保留集未消费。B/C阶段首动作按目标段重入分母解释，不能当独立pickup；JS步行oracle覆盖不能等同生产JAX安全老师覆盖。聊天已主动报告。

用户已授权R5显式阶段条件化为R4完整结束后的下一项串行实验。合同全文已读；R4六run/全部dev/新保留集均须完整终态后正式结论，不择峰，当前不改462份封存源码和R4运行语义。隔离只读审计/CPU TDD准备见`reports/retention_phase_conditioning_proposal_20261021.md`。R5优先纯五阶段one-hot同时进入Actor/Critic，不混奖励/采样/H1/H2；新namespace与共同父/fresh三seed固定32代在完整结论后冻结。

R5隔离准备：纯五阶段当前可见one-hot模块RED→GREEN；5项语义/梯度/零迁移、3项真实JS/Torch/ONNX与生产net_forward/split_forward、5项编排边界回归通过。实际392维raw父Actor/Critic在16个既有learner状态扩展零行后输出精确不变。新增模块未安装R4；462份R4原封存源码SHA仍完全一致。拟seed2026102107–2109，新保留RNG20261106/路线2026110700–0763；仅待R4全部dev/保留集完整终态结论后冻结共同采样并创建新R5命名空间，不择峰、不载入H1/H2、无旧cache复用。

R4第三组A0_s2026102106固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false，全代finite/Actor-Critic非零；g32 L2=3.1905343533/2.9340424538，state/policy原子SHA精确、8副本exact、Adam各4096步，policy `e949d77880c30e351f465c6c82ffd6d889c2a2219f9e34353714565c136b54d8`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据完整，唯一owner按预注册接续最后A1_s2026102106；CPU/JS独立消费A0g32，fresh/正式g1另报。五run累计335,544,320 fresh ticks完成。R5合同要求R4六run、30dev与一次新保留集完整结束并正式结论后才创建新R5；隔离准备不影响R4。聊天已主动通知。

R4最后A1_s2026102106正式g1原子非零确认：2,097,152 fresh ticks，8副本finite/exact、Adam各128步，Actor/Critic L2=3.0214450359/2.3194615841；原子state/policy SHA与fresh共同父/零Adam/空FIFO-core-custody/无恢复独立通过。空core g1两臂policy/EMA逐文件SHA精确相同，PAIRED_G1_EMPTY_CORE_IDENTITY保存；非空core g2另核验处理。不以短点停run。第三组A0完整自然10/3/1/2/8，尾11/96；末点Tactical11、JS Hunter1/2/7各64，不择末点峰值作为父点。25/30 dev已完成，唯一CPU/JS消费最后A1g0；8卡主线正常。聊天已主动报告。

R4第三组处理可达性g2核验通过：相同rollout的reward_components、funnel_by_lesson_family、executed_opponent_exposure、causal_init_audit及certified_return_ppo_evidence精确相同；两臂均96个history返程替换槽，selected core joint质量A0=.7976431052、A1=.5321807464，Actor L2=3.2911128998/3.2512605190、Critic=2.5506443977/2.5079450607。PAIRED_G2_SAMPLING_TREATMENT保存，三seed均证明采样真实生效，不据此宣称能力改善。A1健康g3继续32代；R5仍仅隔离准备。

R4最后A1_s2026102106固定32代正常终态：67,108,864 fresh ticks，atomic_stop=false，全代finite/Actor-Critic非零；g32 L2=3.3874962330/2.8540384769，原子state/policy SHA精确、8副本exact、Adam各4096步，policy `f119653da715c35b32e202539b2b51cb6da5af51e7d6ab9d4165e0ebf1b7bf25`。worker正常退出，8卡连续两次0MiB/0%、compute app0释放收据与独立当前核验通过。六run全部固定预算402,653,184 fresh ticks完成；唯一owner转TRAINING_COMPLETE_EVALUATING，唯一CPU/JS消费最后g32后执行一次全新保留集。用户新增R5合同明确要求R4六run/全部dev/新保留集完整终态并正式结论后才能启动R5，因此此边界不提前prepare/smoke/R5训练，不是能力门禁。H1/H2未载入，R5隔离技术准备不改462份R4封存源码。聊天已主动报告终态及后继。


用户全局 GPU 暂停已执行（20261021）。`runs/retention_20261021/USER_GPU_PAUSE.json` 覆盖此前连续训练及 R5 授权；必须等用户明确通知恢复。接管时无在途训练代：R4 六个正式 run 全部正常完成 g32，累计 402,653,184 fresh ticks，最后为 A1_s2026102106。逐个重新计算 g32 state.pkl/policy.pt SHA，与 COMMITTED.json 全部相符，原子保存证据完整，未打断任何代或提交。

等待 CPU 评估的训练 controller 已终止；残留 zombie 仅等待父进程回收，无可执行命令或 GPU 上下文。训练 worker 为零；独立 CPU/JS evalowner 和 evalchild 保留，CUDA_VISIBLE_DEVICES 为空，只处理已提交 checkpoint，不触发训练。8 卡逐卡显存 0 MiB、利用率 0%，compute app 为零。R5 namespace 未创建，禁止 R5 prepare/smoke/formal、下一 run 和全部后继。

全局持久暂停检查加入 `scripts/retention_screen.py`，覆盖 owner、worker、资源等待和 smoke；R5 prepare 在创建命名空间前检查同一暂停。暂停文件存在即阻止启动，旧局部恢复许可、空/损坏 marker 或模糊 resume 字段均不能绕过。原 R4 训练 seal 保持原样；源码变动明确属于训练结束后的暂停编排修复，不改写原 run 证据。暂停回归 6 项先 RED 后 GREEN，连同相关编排回归共 49 项通过；日志 `.hermes/gpu_pause_regression_20261021.log` 仅留本地。

未发现活动训练/等待启动器或匹配的 cron/systemd 服务配置。环境无 crontab CLI，用户 systemd bus 不可用，故不宣称完成不可访问调度接口的穷尽审计；当前进程、可读配置及共享启动入口已核验。持久收据：`runs/retention_20261021/GPU_PAUSE_RECEIPT_20261021.json`，R4 同名收据及 STATUS 同步标记用户暂停、CPU 评估允许。此次无提交、发布或 GPU 重启，后续训练一律等待明确恢复指令。


20261021 用户显式恢复GPU实验：已全文读取 .hermes/r5_authorized_resume_plan.md，旧USER_GPU_PAUSE与释放收据保留，新USER_GPU_RESUME_20261021.json覆盖。R4六run/30dev/新保留全完成，配对保留自然A0=12/9/9、A1=17/7/10各64，冻结结论不继承priority；详见 reports/retention_core_priority_final_20261021.md。新RETENTION-20261021-R5预注册于 runs/retention_20261021/r5/DECLARATION.json，共同干净raw父、fresh两臂、三配对seed2026102107–2109各32代；纯五阶段one-hot同时进入Actor/Critic为唯一变量，无奖励/课程/采样/H1/H2变化。全新state/cache/eval；dev g0/8/16/24/32唯一CPU异步、COMPLETE marker，新保留RNG20261106；尾均值/最差/AUC/完整漏斗/反事实与战斗安全决策，健康run不因短点停止。focused21通过后立即两臂技术smoke与8卡正式串行六run；首非零及每终态主动通知。

R5异步消费补充：唯一训练owner/唯一CPU-JS新对局评估owner之外，只读consumer持久监听已提交COMPLETE+PHASE_COUNTERFACTUAL，校验shard SHA、Review逐tickhash，保存OWNER_CONSUMED及心跳；无新episode、无GPU控制。逐点机会归一化漏斗/父与前点差、自由库存与窗口/掉包分项、真实collector优势与奖励/FIFO-core/对手实际占比、固定首seed倒退/反向/ABA与威胁死亡写当天报告。父自然首seed全部持包home距离有限；合成fixture字段缺失不作不可达结论。

R5恢复执行已实际落地：A0/A1技术smoke均真实8卡g1 finite、Actor/Critic非零、原子保存与fresh CPU完整恢复通过。首正式A0_s2026102107 g1=2,097,152 fresh ticks，Actor/Critic L2=3.2692329884/2.2828981876，8副本exact、Adam各128步、原子state/policy SHA精确，policy a9b469e5c69a41a7c7944c26cc7fb937c6700172752bb8b3707219196524cff2。独立fresh审计原父参数精确、零Adam、空FIFO/core/custody，restore=null；新进程首代审计finite通过。
正式8卡继续固定32代，唯一训练owner/独立唯一CPU-JS/只读诊断consumer持续；CPU已完成并消费g0（自然10/32、Tactical6/64、JS Hunter难中易2/2/8各64），等待g8。A0同状态阶段反事实KL/value精确0。每run终态先核验释放再接续预声明run并主动通知；未设能力门禁、不因短点停止。当前仍在训练，未宣称R5完成或能力改善。执行收据 runs/retention_20261021/r5/RESUME_FORMAL_EXECUTION_RECEIPT_20261021.json。

20261022 R5完整终态与R6已授权接续：现场确认R5六run固定g32正常结束，重新验证6个state/policy原子SHA和COMMITTED收据；30个dev点、共同父/末点各6个64局holdout、6份g32诊断均COMPLETE并已消费。冻结文件`runs/retention_20261021/r5/FULL_COMPARISON_20261022.json`与`FREEZE_RECEIPT_20261022.json`，本地终态接续通知`TERMINAL_NOTICE_20261022.json`。配对尾差A1−A0为−6.25/−8.33/−1.04个百分点，AUC均差−4.43点；holdout自然交付A0=23/14/9、A1=11/13/19各64局，总计46/192与43/192，方向混合，不挑seed。三seed A1阶段反事实均改变Actor/Critic，A0同状态变化精确为零；A1真实取包返程首动作命中1,237/2,321=53.3%，86/192任务有C_RETURN首偏航，交付48/189次拾包，目标tick持续正进度24,018/62,649；这些路线固定夹具从控制窗口开始，不代表自然战斗入站分母。基于阶段输入可达但返程动作/持续推进仍不足，R6只增真实custody、C_RETURN、年龄0–119 tick的set-valued DAgger倍率×2→×4；共同父回raw R5父点，不继承phase-conditioning。完整依据见`reports/retention_custody_dagger_experiment_20261022.md`。

R6已预注册：`RETENTION-20261022-R6-CUSTODY-RETURN-DAGGER`，namespace `runs/retention_20261022/r6`，同父三配对seed2026102210/11/12，A0×2与A1×4，各固定32代、每代2,097,152 fresh ticks、总计402,653,184；checkpoint g0/8/16/24/32，CPU/JS评估唯一owner异步消费。奖励、PBRS、phase输入、PPO/GAE、场景/对手、查询预算、FIFO和return expert不变；H1/H2不进入。owner写append-only notifications与CONTINUATION marker；每run g32原子终态后核8卡释放，先接续预声明后继再等待报告，健康run无能力门禁。技术smoke只核finite、Actor/Critic非零、原子保存及fresh进程恢复。R5/R6运行产物保留在runs，不纳入Git。

R6实际启动回执（20261022）：R5完整冻结和R6六个fresh namespace/source seal已完成；两臂GPU一代smoke均通过finite、Actor/Critic非零、原子checkpoint与CPU fresh-process restore。A0_s2026102210正式g1 attempt0/restore=null，2,097,152 fresh ticks、finite、8副本exact；Actor/Critic L2=3.1724526882/2.2676110268，Adam各128步，state/policy SHA与COMMITTED精确一致（state `1b256ec8b5328d7358885ffffa6ad07dd3bd206eb1eac8e547ae0e07692007a9`，policy `2b9e1f6c99c97aa913ae37193478ada6fae0d7b5973e29395c2e5d67e753371a`）。唯一CPU/JS evaluator已完成A0g0并排队g8；8卡正式A0继续固定32代。细节见`reports/retention_custody_dagger_experiment_20261022.md`及本地marker `FORMAL_NONZERO_CONFIRMED.json`。

R6 A0 g8异步消费（仅诊断）：固定评估COMPLETE，CPU/JS owner已转等g16，训练继续。相对同seed共同父g0，自然交付10/32→8/32，Tactical 6/64→11/64，Hunter hard 2/64→0/64；库存返程首oracle命中率/目标tick推进率/拾包后交付率分别为死亡窗口66.0%/39.0%/31.3%→49.7%/26.8%/9.1%，原生糖浆窗口68.4%/43.8%/30.3%→49.1%/28.1%/21.2%。real-drop首动作命中63.1%→49.8%，交付/拾包70.6%→58.8%；natural-full oracle命中率79.6%→78.7%，正推进/目标tick37.4%→23.0%。g8 FIFO有效6,903条，其中真实custody return kind 1,547条、年龄0–119 tick合格990条（14.3% valid FIFO）；这解释A0×2实际可加权的行数，不把kind3总数冒充合格数。首偏航16条是封顶诊断样本，其中10条PPO标准化GAE为正，不代表总体错误率或动作Q。评估差值及学习器覆盖收据分别见`DELTA_FROM_PARENT_AND_PREVIOUS.json`与`LEARNER_STATE_DIAGNOSTIC_000008.json`。

R6 A0 g16异步消费（仅诊断）：自然完整对局5/32（g8为8/32、g0为10/32），Tactical v2为9/64（g8为11/64），JS Hunter难/中/易为3/64、2/64、3/64（g8为0/64、1/64、3/64）；自然自伤仍0/32，Tactical自伤1/64。受控死亡库存夹具16/32次落在死亡窗口（其中死亡后14、原生糖浆后2），窗口库存交付3/16；C_RETURN首动作oracle一致157/312=50.3%，目标tick正推进1,802/7,881=22.9%。原生糖浆库存夹具窗口拾取16/32、窗口交付3/16，首动作一致208/388=53.6%、正推进2,852/9,558=29.8%。真实掉包拾取33次、交付16次，返程首动作一致492/844=58.3%、正推进4,257/8,699=48.9%。自然完整路线首控21、归因击杀4、拾包/返程启动7、交付5；C_RETURN首动作一致26/31=83.9%、正推进849/996=85.2%。自然与受控夹具分母分开。

g16学习器快照为14,142条有效FIFO，其中3,508条实际custody返程、2,063条处于0–119 tick有效窗口（占有效FIFO14.6%）；当前与历史有界payload在优化器复用前各含192条返程kind标签。16条封顶首次偏航样本中12条标准化PPO优势为正、10条近似extra奖励分量为正，不外推总体率或Q值。奖励分解仍是诊断口径：控制入口/残余/回扣分别占绝对组合奖励量+6.81%/+0.16%/−1.14%，路线净塑形−1.57%（正+1.13%、负−2.69%）。执行覆盖override后的current/ring/fixed league为有效learner决策的3.37%/7.28%/9.31%。此单seed中点不构成停止、准入或晋升条件；g24由唯一CPU/JS评估owner排队，R6六run固定预算和无能力门禁续接机制继续有效。

R6 A0 g24异步消费后，训练到g29并继续固定预算：自然完整路线交付3/32（g16 5/32）；Tactical安全胜12/64、自伤0（g16为9/64、自伤1），Hunter难/中/易安全胜1/4/4（g16为3/2/3），难度hard死亡1、中2、易0，单seed战斗/安全方向混合。死亡窗口库存14次窗口拾取、9次窗口交付（9/14）；C_RETURN首动作一致88/155=56.8%，正推进1,836/4,806=38.2%，较g16的3/16、157/312=50.3%、1,802/7,881=22.9%提高。原生糖浆窗口库存15次拾取、6次交付（40%），首动作一致201/335=60%、正推进2,592/6,323=41%，较g16的3/16、208/388=53.6%、2,852/9,558=29.8%提高。真实掉包18/33交付（54.5%）、首动作一致330/531=62.1%、正推进3,525/6,032=58.4%，较g16的16/33、492/844=58.3%、4,257/8,699=48.9%提高。自然完整路线仍由g16的7拾包/7返程/5交付降至7/5/3；首动作一致10/15=66.7%（g16 26/31=83.9%），正推进683/827=82.6%（g16 849/996=85.2%）。路线夹具局部提升不能替代自然终局指标。

g24实际A0×2训练payload含current/history各192条return-kind行；训练事件return专家eligible fraction 1.800%（g16 1.833%），CE 0.0920（g16 0.1527）。FIFO kind-3 set-fit为3,957行上0.786，对比g16 3,508行上0.821；数据队列不同，不构成配对效应。g24每行receipt未保存custody age，无法从该payload独立重建年龄资格；g16全FIFO的最后明确年龄审计为2,063/14,142合格。单臂不宣称×2因果效果。g24首次偏航样本首行为action[2,0]、标签集合不含该移动、返程势能距离6前后不变，PPO标准化优势+0.305，近似extra/task为+0.018/+0.316；每代16条上限样本，不外推群体率/Q。评估已COMPLETE，CPU owner等g32，健康run和预声明A1自动后继不受中间结果影响。

R6 A0_s2026102210固定32代正常终态：atomic_stop=false，67,108,864 fresh ticks，末代finite、Actor/Critic非零，8副本exact且Adam各4096步；g32 Actor/Critic更新L2=3.380306/2.880634，state/policy SHA与COMMITTED精确匹配（state `d994f2dbf9f3e915ac9b022d75800a645f200f1794898121a201b9af19bb231a`，policy `56b759184a6a7e55c2ce64357efce11aa3c1956c11b5b460fc9b777d470dc182`）。owner记录连续两次八卡全0MiB/0%、无compute client后先接续配对A1_s2026102210。CPU/JS g32已完成；自然路线16拾包/15返程/6交付，C_RETURN首动作一致74/85=87.1%、正推进1,581/1,833=86.2%。Tactical安全胜12/64、自伤0；Hunter hard/normal/easy安全胜0/2/4，自伤均0，hard死亡2、normal死亡1。受控死亡库存15次窗口拾取/8交付，首动作一致186/327=56.9%、推进2,252/6,216=36.2%；原生糖浆17/4，269/495=54.3%、2,938/7,948=37.0%；真实掉包33拾/16交，295/472=62.5%、3,242/6,035=53.7%。这些均为开发点，非holdout。

g32 return专家CE=.09309、eligible fraction=1.722%，当前/历史有界payload各192条返程kind；kind-3 set-fit为3,981条上.7718，无矛盾重复joint标签；不同FIFO队列禁止配对解释。实际override后current/ring/fixed league为3.43%/8.65%/10.53%。首次封顶偏航示例action[2,0]不在标签集合，但势能距离3→2，标准化PPO优势−.128、近似extra/task +.019/−.136；不能只凭oracle不一致标记失败。A1_s2026102210已以fresh attempt0、restore=null、倍率×4接续，g1原子首更新2,097,152 ticks、finite、8副本Actor/Critic L2=3.179339/2.305243、Adam各128步；state/policy SHA精确匹配，CPU/JS评估独立。唯一owner继续六run固定预算，无能力门禁。

一次只读checkpoint反序列化意外初始化默认GPU JAX backend并打印分配失败；该诊断进程已退出，非训练/evaluator，未改checkpoint。立即核验训练owner/worker与8卡仍健康，g16以后state审计强制CPU backend。固定run未停，未将此观察当能力或准入门禁。

20261022 接管复核：R5 `FREEZE_RECEIPT_20261022.json` 显示6个g32原子checkpoint、30个dev点、父点/末点各6个holdout及6份诊断均已完成消费，比较SHA256与freeze收据一致；自然holdout A0=23/14/9、A1=11/13/19，各64局，不择seed。阶段反事实在A1三seed均改变Actor/Critic而A0保持不变，但返程路线仍有首次偏航和交付缺口，按合同维持R6单变量真实custody短窗set-valued DAgger倍率×2/×4配对筛选。

R6 A1_s2026102210的g8/g16异步评估均已消费，CPU队列当前等待g24；训练已到g20且finite/nonzero，8卡仍由正式run占用。g16自然路线12拾包/12返程/8交付，C_RETURN首动作39/51=76.5%、正推进1,563/1,825=85.6%；库存死亡窗口12拾/7交、首动作143/228=62.7%、推进2,084/8,470=24.6%；原生糖浆16拾/10交、200/343=58.3%、2,896/9,056=32.0%；掉包33拾/16交、318/715=44.5%、4,441/9,112=48.7%。Tactical安全胜13/64且无自伤；Hunter难/中/易4/1/12，普通与简单各有1次死亡，战斗/安全信号混合。

g16合法联合标签FIFO审计14,148行，kind-3为3,518行、greedy-in-set 3,009、平均目标概率0.8300，32个重复精确输入组中0个无公共合法动作；current/history有界payload各192条返程标签。return expert CE=.12697、eligible=1.865%；缺逐行custody age审计，不将队列拟合当作×4因果收益。16条封顶偏航样本首条行动距离3→4，实际GAE +.0483、标准化优势+.0079，近似extra/task −.0165/+.0648。累计诊断控制entry/residual/clawback占绝对奖励+6.85%/+.16%/−1.05%，路线净−1.56%；override后current/ring/fixed对手占有效决策3.44%/7.38%/9.45%。g16评估COMPLETE已消费、CPU队列等待g24；训练已至g20、Actor/Critic更新L2=3.3713/2.8862且finite/nonzero，g16为最新周期原子checkpoint，8卡持续用于固定run。中点只作诊断，不改预算或停训。持久复核收据与g8/g16完整漏斗见`runs/retention_20261022/r6/OWNER_REASSUMPTION_20261022.json`和`reports/retention_custody_dagger_experiment_20261022.md`；owner继续固定32代及预声明后继，无能力门禁。

g24自然路线14拾包/13返程/8交付，C_RETURN首动作62/91=68.1%、正推进1,568/2,003=78.3%，交付持平但较g16的76.5%/85.6%下降。库存死亡窗口14拾/3交、首动作428/623=68.7%、推进2,055/9,950=20.7%；原生糖浆15拾/7交、159/340=46.8%、2,793/9,060=30.8%；掉包33拾/17交、406/718=56.5%、4,327/8,221=52.6%。Tactical安全胜9/64、2次死亡/1次自伤；Hunter难/中/易安全胜1/4/11，普通与简单各1次死亡。joint FIFO 14,098行，kind-3 3,834行、greedy-in-set 3,207、平均目标概率0.8024；21个重复精确输入组中0个无公共动作。current/history payload各192条return标签；CE=.11271、eligible=1.823%，仍缺逐行custody age审计。16条封顶首偏航样本首条距离30→30，实际GAE +3.6316、标准化优势+3.3026，近似extra/task −.0154/+3.6470。24代累计奖励占比entry/residual/clawback +7.04%/+.16%/−1.22%，route净−1.71%；override后对手current/ring/fixed 3.66%/8.30%/9.55%。g24 COMPLETE已消费、评估队列等待g32；训练至g28、Actor/Critic L2=3.3325/2.7791、finite/nonzero，g24为最新周期原子点，固定预算继续。
g24自然路线14拾包/13返程/8交付，C_RETURN首动作62/91=68.1%、正推进1,568/2,003=78.3%，交付持平但较g16的76.5%/85.6%下降。库存死亡窗口14拾/3交、首动作428/623=68.7%、推进2,055/9,950=20.7%；原生糖浆15拾/7交、159/340=46.8%、2,793/9,060=30.8%；掉包33拾/17交、406/718=56.5%、4,327/8,221=52.6%。Tactical安全胜9/64、2次死亡/1次自伤；Hunter难/中/易安全胜1/4/11，普通与简单各1次死亡。joint FIFO 14,098行，kind-3 3,834行、greedy-in-set 3,207、平均目标概率0.8024；21个重复精确输入组中0个无公共动作。current/history payload各192条return标签；CE=.11271、eligible=1.823%，仍缺逐行custody age审计。16条封顶首偏航样本首条距离30→30，实际GAE +3.6316、标准化优势+3.3026，近似extra/task −.0154/+3.6470。24代累计奖励占比entry/residual/clawback +7.04%/+.16%/−1.22%，route净−1.71%；override后对手current/ring/fixed 3.66%/8.30%/9.55%。g24 COMPLETE已消费、评估队列等待g32；训练至g28、Actor/Critic L2=3.3325/2.7791、finite/nonzero，g24为最新周期原子点，固定预算继续。

R6 A1_s2026102210 fixed run在g32正常终态：atomic_stop=false、67,108,864 fresh ticks、finite/nonzero、8副本exact、Adam各4096步，末代Actor/Critic L2=3.364644/2.900525；原子state/policy SHA分别为`1f76bd03a67306d6b9b31f5296902343ad0247c04eab6b4eafbd4ad01d891cd6`与`b30d84977ed5928defbd10f31f71e2338b47c269062fbf54fefd727e6c96da3d`。8卡两次空闲后owner写释放收据并已启动预声明A0_s2026102211；A1 g32 CPU/JS评估继续独立消费。固定32代无能力门禁，owner将继续余下四个配对run。

A1 g32异步dev最终消费：自然9拾/8返程/2交付，C_RETURN首动作17/23=73.9%、正推进1,036/1,322=78.4%；Tactical安全胜7/64，Hunter难/中/易0/0/7，普通6次死亡、自伤0。死亡窗口库存10拾/4交、首动作169/290=58.3%、推进1,666/7,510=22.2%；糖浆窗口10拾/4交、273/421=64.8%、1,994/7,948=25.1%；掉包36拾/10交、424/835=50.8%、4,201/8,701=48.3%。kind-3 FIFO 3,949行、greedy-in-set 3,213、均值目标概率0.7830，25个重复输入组0个标签冲突；current/history payload各192条返程行。执行对手current/ring/fixed 3.46%/8.67%/10.53%。CE与逐行custody age审计不可用。该seed的自然结果较弱，保留配对比较，未择seed、不停健康后继。A0_s2026102211已启动fresh首代编译/执行，尚无正式g1非零原子收据；A1 g32评估完成后CPU队列跟随新run，训练预算照常。
A1 g32异步dev最终消费：自然9拾/8返程/2交付，C_RETURN首动作17/23=73.9%、正推进1,036/1,322=78.4%；Tactical安全胜7/64，Hunter难/中/易0/0/7，普通6次死亡、自伤0。死亡窗口库存10拾/4交、首动作169/290=58.3%、推进1,666/7,510=22.2%；糖浆窗口10拾/4交、273/421=64.8%、1,994/7,948=25.1%；掉包36拾/10交、424/835=50.8%、4,201/8,701=48.3%。kind-3 FIFO 3,949行、greedy-in-set 3,213、均值目标概率0.7830，25个重复输入组0个标签冲突；current/history payload各192条返程行。执行对手current/ring/fixed 3.46%/8.67%/10.53%。CE与逐行custody age审计不可用。该seed的自然结果较弱，保留配对比较，未择seed、不停健康后继。

A0_s2026102211正式首个非零更新确认：g1、2,097,152 fresh ticks、finite、8副本exact、Adam各128步，Actor/Critic L2=3.032625/2.259217；原子state/policy SHA为`58367a969ef371c9c4c4393a4f34dfde576a5986b6f5ab3f243d72d7c1ae5fe6`与`8fdff8620958ff28eb8d0f3f7b6295330bee4a88463ebfa4626a9825a07d2e7e`，正式marker核验通过。attempt0、restore=null，输入hash与同父一致，未复制learner state，optimizer/RNG/env/FIFO/core全fresh；A0×2实际只对真实custody且年龄<120tick行生效，奖励、回放选择与查询预算未变。唯一CPU/JS评估已完成A0 g0，等待g8；8卡固定32代继续，owner及预声明队列有效。
A0_s2026102211正式首个非零更新确认：g1、2,097,152 fresh ticks、finite、8副本exact、Adam各128步，Actor/Critic L2=3.032625/2.259217；原子state/policy SHA为`58367a969ef371c9c4c4393a4f34dfde576a5986b6f5ab3f243d72d7c1ae5fe6`与`8fdff8620958ff28eb8d0f3f7b6295330bee4a88463ebfa4626a9825a07d2e7e`，正式marker核验通过。attempt0、restore=null，输入hash与同父一致，未复制learner state，optimizer/RNG/env/FIFO/core全fresh；A0×2实际只对真实custody且年龄<120tick行生效，奖励、回放选择与查询预算未变。唯一CPU/JS评估已完成A0 g0，等待g8；8卡固定32代继续，owner及预声明队列有效。

A0_s2026102211 g8异步评估已COMPLETE并消费，当前队列等待g16，训练同时到g12。自然10拾包/10返程/7交付，C_RETURN首动作18/21=85.7%、正推进1,403/1,573=89.2%；相对本run g0少3次自然交付。死亡窗口库存15拾/7交、307/490=62.7%、2,591/8,300=31.2%；原生糖浆16拾/12交、198/365=54.2%、2,922/7,705=37.9%；真实掉包33拾/20交、294/518=56.8%、3,286/5,605=58.6%。Tactical安全胜6/64、死亡4、自伤3；Hunter难/中/易安全胜0/0/8，普通与简单各1次死亡。战斗保持与路线信号混合。

kind-3 FIFO 1,643行、greedy-in-set 1,494、平均目标概率0.8812，17个重复精确输入组均存在共同合法动作；当前/历史有界payload各192条return标签。return expert CE=.06002、eligible=1.865%，缺逐行custody age审计，不作倍率因果解释。16条封顶首次偏航样本首条行动距离30→30，标准化优势−.0525，近似extra/task −.0203/+.0138。8代累计奖励entry/residual/clawback +6.26%/+.15%/−.88%，路线净−1.18%；override后current/ring/fixed对手1.53%/7.49%/10.25%。训练g12仍finite/nonzero、25,165,824 ticks，g8 CPU/JS评估不阻塞固定32代。
kind-3 FIFO 1,643行、greedy-in-set 1,494、平均目标概率0.8812，17个重复精确输入组均存在共同合法动作；当前/历史有界payload各192条return标签。return expert CE=.06002、eligible=1.865%，缺逐行custody age审计，不作倍率因果解释。16条封顶首次偏航样本首条行动距离30→30，标准化优势−.0525，近似extra/task −.0203/+.0138。8代累计奖励entry/residual/clawback +6.26%/+.15%/−.88%，路线净−1.18%；override后current/ring/fixed对手1.53%/7.49%/10.25%。g8 CPU/JS评估不阻塞固定32代；最新检查A0_s2026102211在g14 finite/nonzero、29,360,128 fresh ticks、Actor/Critic L2=3.3101/2.9015，g8为最新周期原子点，CPU队列等待g16。持续状态收据`runs/retention_20261022/r6/OWNER_REASSUMPTION_20261022.json`已刷新。
2026-10-22重新接管R6：只读复验R5六个g32原子checkpoint、30个dev点、父点及末点共12个holdout点和六份终诊断全部通过；`FULL_COMPARISON_20261022.json` SHA256=`4c1145c91f3e6dfb2a3bc259229e9212829ef5f8822ac37504f9780f1ad073ec`，自然holdout A0=23/14/9、A1=11/13/19（各64局），未择seed。三配对dev tail/AUC均值分别偏向A0 5.21/4.43个百分点，故后继保持预注册单变量R6真实custody、0–119 tick、set-valued C_RETURN DAgger倍率A0×2/A1×4，不混phase、奖励或连泡。R6 `A0_s2026102211`正式首代已确认finite/nonzero、2,097,152 fresh ticks、8副本exact、Actor/Critic L2=3.032625/2.259217及原子SHA通过；当前g18、37,748,736 fresh ticks、L2=3.103671/2.925461，g16为最新原子点，g16 CPU/JS漏斗评估进行中。唯一训练owner、GPU worker、CPU/JS评估owner存活，8卡由当前run占用。固定run无能力门禁；g32原子终态后须核验8卡连续空闲、持久通知，再启动预声明配对后继，不等待评估/报告。详见`reports/retention_custody_dagger_experiment_20261022.md`与`runs/retention_20261022/r6/OWNER_REASSUMPTION_20261022.json`。
R6 A0_s2026102211 g16固定CPU/JS评估已COMPLETE并消费，队列等待g24；自然32局6拾/5返/3交、死亡13/32、自伤0，合法自由存活过窗库存取包3次与合格窗口取包3次分列，后者2次交付。death/sugar/drop/recovery固定路线窗口拾取14/17/17及交付8/9/11，真实掉包33拾23交；Tactical安全胜11/64、死亡2、自伤2，Hunter H/N/E安全胜0/4/7、死亡1/3/0。联合标签FIFO 14,081行中kind-3有3,485、greedy-in-set 2,977、平均目标概率.8296；30个重复精确输入组无矛盾。16条封顶首偏航样本的首条phase id1、route id200001，action[1,0]距离22→22，标准化优势−.1576、近似extra/task +.0423/−.1956，仅作诊断。控制entry/residual/clawback占绝对奖励+6.93%/+.16%/−1.13%，路线净−1.58%；override后current/ring/fixed对手1.75%/6.27%/12.14%。训练设置不变。最新观测g23 finite/nonzero、48,234,496 fresh ticks、Actor/Critic L2=3.205316/2.694485；g20原子state/policy SHA=`9514f52ee883f59b6fe300f125389132cdbb05b4d53391d92204c27679141da7`/`52afbf693bc76d6f4faa1b55eaa4fead9d89053346cf488e6c3445ed346b6b0c`。8卡、owner和异步评估持续运行，固定g32无能力门禁；终态通知后自动启动配对A1_s2026102211。

2026-10-22用户要求固定run结束后暂停。实际边界`A0_s2026102212`已完成g32并验证原子state/policy SHA；接续竞态启动的`A1_s2026102212`仅完成g1，finite、Actor/Critic更新L2分别为2.9620/2.3639，原子保存后`atomic_stop=true`退出，不能视作完成的32代配对run。全局实际授权闸门`runs/retention_20261021/USER_GPU_PAUSE.json`已恢复为拒绝训练，须用户明确恢复才可解除。训练owner/worker已停止，CPU evaluator保留并完成A1 g0、等待g8；8卡显存/利用率均为0且无compute app。唯一未执行训练队列是A1 seed 2026102212的g2-g32。完整SHA、竞态与暂停收据见`runs/retention_20261022/r6/USER_PAUSE_AFTER_CURRENT_RUN.json`及`RACE_ATOMIC_STOP_RECEIPT.json`；此状态覆盖此前“终态后自动启动后继”的旧记录。

2026-10-22用户明确恢复R6训练：保留历史暂停收据并将恢复前全局marker存为`runs/retention_20261022/r6/USER_GPU_PAUSE_BEFORE_RESUME_20261022.json`；新的`USER_GPU_RESUME_20261022.json`被真实启动授权检查接受。复验五个已完成g32与A1 g1的state/policy SHA、SOURCE_SEAL及固定32代预算；focused暂停/R6控制测试10项通过。按`CONTINUATION.json`原命令启动唯一owner，worker明确`--resume .../A1_s2026102212/formal/bun392_bridge_pfsp8/gen000001`，不重跑完成run，复用既有smoke收据。空等g8的CPU evaluator安全移交、已完成点保留；新评估owner独立跟随固定点。唯一变量仍为真实custody短窗口set-valued返程DAgger A0×2/A1×4，未改奖励/采样/查询预算。owner启动时复报g1旧证据不等于新更新成功，需单独核验g2或之后finite/nonzero与后续周期原子SHA；健康run固定完成至g32并主动终态通知。

恢复首个新更新已确认：A1 seed 2026102212 g2 finite/nonzero，八副本Actor/Critic L2=3.272128/2.583599、累计4,194,304 fresh ticks；g3也finite/nonzero。独立恢复非零收据`POST_RESUME_NONZERO_CONFIRMED_20261022.json`与通知已持久保存。下一周期原子点g4，固定g8/16/24/32 CPU评估接续。恢复期间消费A0 seed2212 g32既有开发证据：自然交付曲线10/10/2/2/2（各32局），终点3次合格窗口取包→3返程→2交付、10死亡/0自伤；B_STOCK首动作5/11、推进546/4362，C_RETURN首动作6/7、推进430/479，主要弱点需分别归因。Tactical安全胜11/64，JS Hunter难/中/易0/0/6；返程FIFO4525行、greedy-in-set3622、目标概率.7610、28重复输入组无矛盾。current/ring/fixed实际暴露2.48%/8.81%/11.29%。不以此停止最后A1，不择seed，待完整配对终点与holdout。

恢复后g4原子提交已验收：state SHA=`6eebb3f19e86350a6926b3fcab97671447559dc038f34803c24995ea9f926057`、policy SHA=`fd443e291b40cdb17f350003dac6589d1abb7380d2fbacca7443dc28b55bef04`，COMMITTED SHA=`92c04fbfabe8bec70236373d11e36537da18d877478503267e763f7f40edac56`；八副本exact、Adam各512步，事件g1-g4连续且finite/nonzero。独立`POST_RESUME_ATOMIC_VERIFIED_20261022.json`及主动通知已保存；随后g5仍finite/nonzero。唯一owner/worker继续固定g32，唯一CPU evaluator独立等待g8；旧暂停授权已被显式恢复覆盖，历史暂停收据未改写。

2026-10-22恢复后A1 seed2212固定g8评估COMPLETE/SHA一致并消费：自然11拾/10返/6交（32局），相对父点少4交、死亡少3；12归因首次控制→12因果击杀→12存活，7次合格窗口库存拾取→4交付。B_STOCK首动作11/14、推进500/5122；C_RETURN首动作34/42、推进1455/1668。真实掉包32拾20交；死亡/糖浆库存合格窗口14拾9交、16拾7交。Tactical安全胜7/64、0死亡/0自伤；真实JS Hunter难/中/易2/3/8（各64）、死亡1/1/0、自伤0。FIFO返程1629行、greedy-in-set1432、目标概率.8551，13重复输入组无矛盾。query/candidate=7002/2819980=.2483%，非返程eligible覆盖率。奖励entry/residual/clawback +6.32%/+.16%/−.81%，路线净−1.20%；执行后current/ring/fixed暴露2.53%/8.38%/8.01%。结果仅诊断，固定训练继续，不改配方、不择点。

2026-10-22恢复后A1 seed2212 g16评估COMPLETE且checkpoint SHA一致、已消费：自然12拾/13返/7交（32局，重复返程事件与拾包局数分开），7归因控制→7因果击杀→7存活，8死亡/0自伤；相对g8交付+1、相对父点−3。B_STOCK首动作10/13、推进466/3758；C_RETURN首动作20/25、推进1450/1797。Tactical安全胜11/64，真实JS Hunter H/N/E=3/2/10（各64）。训练同时g20 finite/nonzero，原子g20已发布；下一固定评估g24，维持既定g32，不以中点结果停训或改配方。完整joint fit、奖励分解、当代首偏航及实际暴露见`runs/retention_20261022/r6/POST_RESUME_EVAL_g000016_CONSUMED_20261022.json`。

2026-10-22恢复后A1 seed2212 g24固定评估COMPLETE/SHA一致并消费：自然10拾/9返/6交（32局）、11死亡/0自伤，11归因控制→11因果击杀→11存活；交付相对g16−1、父点−4。B_STOCK首动作7/11、推进351/6874；C_RETURN首动作57/78、推进1004/1664。Tactical安全胜11/64，真实JS Hunter H/N/E=1/1/7（各64）。训练同时g28 finite/nonzero并原子提交；继续固定g32，无能力门禁。完整当代偏航、fit、奖励、暴露见`runs/retention_20261022/r6/POST_RESUME_EVAL_g000024_CONSUMED_20261022.json`。

2026-10-22主动终态：恢复的A1_s2026102212正常完成固定g32，67,108,864 fresh ticks、atomic_stop=false；事件代数严格1至32且全部finite/nonzero，最后Actor/Critic L2=3.369849/2.894280、八副本exact且Adam各4096步。state SHA=`5cd56bfa2f175c860a6d8dd776351f28ad96d9f83e9738b7ab4b9021f4988a96`，policy SHA=`75334d02fc7ebe7fb32b8870541bac625f1bcacc03c8f9d52c3eb54a27fcfc69`，均独立验证；未重跑五个完成run、未重跑smoke、未改R6变量。worker已退出，owner核验连续两次8卡空闲并写GPU_RELEASE_RECEIPT，已发送持久终态通知。STATUS/CONTINUATION为TRAINING_COMPLETE_EVALUATING；训练owner仍监督CPU evaluator，g32 dev正在执行，之后依冻结协议完成holdout。六个预注册run训练预算全部完成，无第七个预注册run；不是能力判停或恢复旧暂停。复验收据`runs/retention_20261022/r6/USER_RESUME_FIXED_RUN_TERMINAL_20261022.json`。

2026-10-22用户要求恢复持续有效训练，预注册独立O0观察run：`runs/retention_20261022/continuous/O0_s2026102220`，共同raw父r19 g32、fresh optimizer/env/RNG/FIFO/core、seed2026102220、原R6 A0×2配方，固定128代，每代2097152 fresh ticks，g0/8/16/24/32/64/96/128异步固定评估；不把O0作为R7配对样本或R6胜出证据。R7后继已预声明三配对seed2026102221/22/23×32代，唯一变量普通已接受路线回放return/drop/stock分配75/12.5/12.5%对50/25/25%，同raw父、同×2及总样本/历史当前/场景/奖励/PPO；以阶段遗忘和取笼推进薄弱为待证假设，不宣称已证根因。持久队列`runs/retention_20261022/continuous/TRAIN_QUEUE.json`预留O1原配置128代技术开发fallback。训练owner不等评估/文档，终态立即接续预声明下一固定run。

O0真实首更新g1 finite/Actor-Critic L2=3.107821/2.266770，原子state/policy SHA通过；随后g3有效43,533 fresh ticks/s、计算期八卡77%-90%，g4完整状态及next RNG经独立CPU新进程恢复通过。最新g12 finite/nonzero、25,165,824 fresh ticks且g12原子保存通过，八卡79%-89%，不是仅占显存。实际唯一训练owner/worker/CPU evaluator各一个，g8评估异步完成，当前长训继续。持久owner已有“终态后不等待evaluator领取下一run”回归，准备储备维护器保持两个预声明128代原配置观察run在前；队列O0→单次候选技术smoke→R7六run→O1，后续按同规则补固定观察储备。RED→GREEN及相关30项测试通过；CPU辅助入口故障已修复且未影响GPU训练。R6 holdout A0=22/19/8、A1=7/22/10（64局/seed），不宣称×4稳定胜出、不择seed、不加倍率。详细预注册、运行证据和技术说明见`reports/retention_continuous_experiment_20261022.md`及`runs/retention_20261022/continuous/LIVE_HANDOFF_20261022.json`；不得将这条记录视作固定128代已经完成。

## CAUSAL-CARRIER-MECHANISM-20261023-v1（预注册）

用户独立授权真实持包机制诊断 A/B/C；旧 continuous O5g95 与储备队列继续暂停。独立 namespace `runs/carrier_mechanism_20261023/v1`，不得自动后继或发布。冻结协议为该目录 `PROTOCOL.json`，checkpoint共同父R19g32、R6首seed A0/A1g32、O5g95，均绑定自己的Actor/Critic/完整状态及相同reward/gamma=.995。

A：96真实库存事件后完整状态，4来源各24、每来源每split8；96独立来源局，每状态8次随机重复。train/native、dev/open、holdout/部分清砖初始布局整局隔离。仅地图806可用，明确不宣称跨地图泛化；所有取包必须真实入站，禁止携包注入。验证完整next-step恢复parity。全程MC与256/512/1024截断+自举敏感性分列，tested V不作为独立校准目标。B：同状态进/退/等、固定其他动作因子及未来RNG，只称subset Q contrasts，另记录真实偏航Bank-P及死亡/丢包，不择幸存。CI按来源局/布局聚类。

C仅在A/B实测后：同父同已保存Adam，PPO-only/DAgger-only/联合，5配对seed2026102301..05，另无更新control；16更新，u0/1/4/8/16保存，32env×128tick/update，minibatch256、1epoch；PPO on-policy，老师为集合安全合法方向，训练Bank与dev/holdout隔离。仅冻结train batch选数值尺度，holdout不选参，最终战斗/自然保留单列。完成诊断释放GPU，不恢复旧长训。结果填入 `reports/carrier_mechanism_diagnostic_20261023.md`。

### 20261023 实施与技术更正

独立诊断四checkpoint完整state/policy/COMMITTED哈希校验完成；focused RED→GREEN通过。启动前修正独立驱动遗漏的生产bridge参数（window120/scale2），未消费错配奖励数据；首次编译发现规则bot tier需批量数组，修复并保留错误日志。自然来源局正在8卡收集。128tick scan的2560最大padding槽与2528诊断截断分列，有效步按真实未终止曝光统计。唯一持久owner跟随已有collector，不重复启动；GPU阶段与完整科学评估采用不同终态marker。详细实测持续写入 `reports/carrier_mechanism_diagnostic_20261023.md`。旧O5/储备暂停未解除。

A来源收集固定预算结束：1536独立自然来源局，735局真实合法窗口库存取包、801局无事件排除，有效环境步3,015,163；按预声明每来源/布局顺序各8冻结96独立状态。相关61项测试重跑全过；唯一旧失败为历史RED测试错误绑定已修复生产oracle，现显式固定历史公式，训练源码未改。A/B实际GAE使用各checkpoint保存plan的256tick rollout长度、实际奖励/V/terminal及原train.compute_gae，明确为从Bank起重建而非历史optimizer样本。C原生Bank评估资源放置在结果前冻结，不改数据/预算/训练；CPU/JS战斗保持独立异步。

### CAUSAL-CARRIER-MECHANISM-20261023-v1 A/B阶段终态

96真实取包状态三种parity全过；四checkpoint×四分支各768/768真终局，无截断混入MC。全部固定Q/V结果见当天报告。Bank-P时间边界RED→GREEN，raw倒退与native-safe首偏航分开；自然JS旧求和比例不用作汇总，重算事件/机会分母。C仍按原协议五seed三臂固定预算，不以A/B/holdout改参，不恢复旧continuous或储备。

C技术终态：256真实on-policy环境步/1优化器步，Actor ΔL2=0.049894745、Critic=0.065477185，finite；fresh-process next-step和Adam精确恢复。正式首run u0独立核验原父params/Adam digest均一致，smoke权重不用于正式实验。正式u0原子state SHA `5b74382e5dbf828fa0c8163a433b3d7e9a4fac7fad9d35cea9cfd6f795e02ae3`。随后执行原固定五seed三臂预算，无能力门禁。

### CAUSAL-CARRIER-MECHANISM-20261023-v1 完整终态

A/B实测、C三臂五seed固定更新、全部保存点dev/最终holdout与CPU/JS战斗保持已消费。固定结果和限制见 `reports/carrier_mechanism_diagnostic_20261023.md` 与独立namespace FINAL_COMPARISON；无模型发布、无自动后继，旧O5/储备暂停保留。

### CAUSAL-CARRIER-MECHANISM-20261023-v1 机制结论与停止

15/15固定run、75原子保存点、61 dev唯一点/16最终holdout、全部16模型CPU/JS固定评估完成。联合真实持包holdout直接交付85.94%→95.31%，训练seed×来源局CI +4.30～+15.00 pp，条件死亡12.11%→4.38%。自然交付五seed均值PPO-only13.2、DAgger-only11.4、joint15.2，父点17/64；含自然对局不确定性后变化CI均跨零，不择峰、不宣称自然总体提升。证据更支持取包前链/跨阶段曝光保持缺口，不能据负EV或单阶段成功证明PPO噪声/破坏性干扰。详见当天报告和MECHANISM_CONCLUSION。GPU预算已终态，8卡释放，旧O5和储备暂停不变，无自动后继、无发布。技术修复均保留证据，未变更配方或holdout选参。

## COMBAT-ITEM-20261023-v1：自然链保持与独立道具结算筛选（已预注册）

四臂共享干净 r19 g32、matching Actor/Critic 与保存 Adam；父 policy SHA256 `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`，完整 state SHA256 `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`。选择依据：前一真实持包诊断自然完整父点17/64，三类微调均未证实总体改善，不能择联合某个seed或盲继承O5。Actor/Critic独立。

namespace `runs/combat_item_20261023/v1`，不可恢复旧continuous/O5/储备。每臂seed 2026102311/12/13，共12 run，各固定u32、64env×128step/u、262144真on-policy步、1024 optimizer步；总3145728环境步/12288优化器步。保存u0/1/8/16/24/32。γ=.995、λ=.95、Adam lr7.5e-5、clip .1、entropy .005、value .5、gradient clip .5，1epoch/global minibatch256；全部同父/同Adam/配对RNG。

A0共同对照：原奖励与集合路线老师.2；携包保持返程目标，native joint安全过滤；空手库存老师/PBRS仍仅死亡或原生糖浆窗口，自由偷包不惩罚，慢慢胶不冒充糖浆。75%真实自然出生+25%经完整真实取包/入站parity认证的train Bank。自然出生库存none/banana/glue等概率，绝不写携包；仅初始化补道具覆盖。Bank仅map806、P0、全far，不能声称跨图/近中距离保持。当前/ring/固定对手配置0/0/100%，固定JAX rule tier3；老师tier4，二者都不是JS Hunter或Python Tactical。

三个独立单变量对照：C只增空手自由敌人的安全combat集合监督.05；B只替换旧香蕉结算为完整owned自吃滑行净返程距离（减普通步行.18cell/tick预算），min(.5,.2gain)、cap2并失包死亡追回；G只替换旧glue命中分为完整真实减速追击延缓且自己向家推进后的min(.5,.25delay)、cap2和追回。两道具臂从旧总项扣掉其对应实际已付/追回，再加新结算，不重复计酬；其余家族不改。不会混入H1炸砖或H2连泡。共同语义修正不宣称与原父点的单变量因果比较；三个处理效应仅相对同配方A0。

技术smoke只查focused TDD、finite、Actor/Critic非零、原子保存/fresh env/aux/RNG/Adam恢复；不作能力门禁。健康固定预算不因中点能力而停止。CPU唯一owner异步消费已提交点，不触发GPU、不阻训练；u16自然JS64，中间native Bank dev，全部最终Python Tactical v2 64、生产JS Hunter难中易各64、自然JS dev64+分离seed holdout64及Bank/Bank-P。既有Bank已被前轮使用，是冻结保留集而非新独立验证；自然holdout seed2026102700..2763在最终一次消费，不用于调参。dev自然2026102380..2443，Hunter seed20261023；全部模型同协议，不混JAX/Python/JS结果。

按配对seed最终/尾部曲线、最差seed/点、机会分母、安全、道具净收益和固定首偏航判断；不按塑形或择峰晋升。小规模系数不扫描、不看holdout调参，只冻结train batch审计量级与可达性。固定12run结束接续有界Bank评估，随后停止本实验GPU；CPU队列清空、报告完整证据即结束，无自动扩展、不发布。日报告 `reports/combat_item_experiment_20261023.md`；完整冻结合同/owner/原子marker见上述namespace。

### v1.1：道具处理启用前的防刷技术修正

第一个A0/control_s2026102311已u32原子完成；combat_s2026102311完成u1原子保存后，通过实际authorize消费的本实验停止标记安全退出，8卡全0MiB。无banana/glue处理臂已启动。新增行为RED实测：换新道具、先退回20再沿旧路走到18，第二次仍获+.328，说明仅稳定事件ID与cap不能防走回头路刷不同道具。

修正仅给每家族加episode历史最小返程距离，以开始前历史最小值约束完整事件起点；不因死亡/失包/重取重置，只随整局reset，防新ID刷旧走廊。所有参数/Adam/native env/RNG不变，恢复combat u1仅扩展两个被动aux字段。A0/C的道具替换系数为0，实际奖励/老师不变，已完成对照有效；不重跑A0、不丢C u1、不增加预算。B/G尚未实际消费，使用修正后的冻结结算。此前C u1的反事实新道具统计缺少历史frontier，不把它当正式B/G处理证据。保留 `TDD_FRONTIER_RED.log`、安全停止及修正收据，原协议不覆盖，另记 `PROTOCOL_AMENDMENT_FRONTIER.json`。该暂停只修真实反作弊缺陷，不是能力门禁。

### COMBAT-ITEM-20261023-v1 跨日验收（20261024）

12个run全部u32原子保存，正式预算已完成；正在完成冻结Bank/Bank-P、CPU/JS保留集和train-only重放，结束后停止本实验GPU，无后继训练。当天验收文档为 `reports/combat_item_final_20261024.md`，启动日完整设计/比较保持原文件。

COMBAT-ITEM-20261023-v1固定12run、CPU/JS与Bank/Bank-P均完成；原子验收及完整配对结果见当天报告和FINAL_COMPARISON。诊断GPU停止，不恢复旧队列；无发布/晋升。

COMBAT-ITEM-20261023-v1最终验收：12/12固定u32，3145728正式环境步、12288优化器步，全部384次Actor/Critic非零finite；73原子点含共同父经SHA/Adam/导出核验。CPU/JS全协议、Bank61dev/13holdout、Bank-P及30固定train-only wiring全部完成。自然holdout交付A0=0/8/0、combat=7/7/7、banana=0/8/0、glue=7/2/1；combat死亡代价+25.52pp [14.06,36.98]，无可靠整体改善；香蕉逐seed最终SHA与A0一致，未形成有效处理差异。阶段安全补记只重放640既有对局、0新独立样本，原逐局指标/固定完整trace精确一致。结论与根因排序见 `reports/combat_item_final_20261024.md`，完整表/图见启动日报告。

最终owner/worker/CPU/monitor均0，8卡0MiB/0%/无compute app，pending eval0；预算封闭USER_STOP实际阻止本namespace重启，无后继。旧O5/continuous/储备全局暂停SHA未变；不push、不发布、不晋升。

## DAGGER-RL-20261024-R1（20261024，已预注册）

旧COMBAT-ITEM12run全部完成，香蕉奖励差0且与control逐seed同权重，自然持包有库存却greedy零释放；tier4联合战斗老师不输出道具，会抑制ability2。旧O5/continuous/储备暂停保持，不提交或发布。干净共同父r19g32 policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`，保存Adam匹配；不能用退化control代替未训练父。

新namespace `runs/dagger_rl_20261024/r1`，协议 `PROTOCOL.json` SHA `ebcd8146eee8e19ad38471da2ada07525cbfe1eeb98a7864220862c27cb73607`。三配对seed2026102411/12/13，sequential（16 DAgger-only→16 pure PPO）、joint（32 PPO+DAgger）、ppo（32 pure PPO），每run64env×128tick×32，262144环境步/1024优化步，总2359296/9216；固定lr7.5e-5及前轮control奖励、对手、课程、网络，唯一比较为监督/RL时间安排，不叠combat/glue新处理。DAgger-only冻结Critic和mu/nu，共享Adam时钟推进并披露；切换不reset。保存0/1/8/16/17/24/32；每更新完整原子恢复点，不把离线监督作PPO。

先在真实train取包Bank恢复后的学生on-policy路径审计：当前公开状态规划“释放→离格arm→折返自吃→完整滑行撞停”，拒绝危险、无净收益、不完整滑行；老师不执行动作。安全能力有限，不保证未来敌人不放泡。bounded FIFO8192聚合学生标签，当前/历史监督各半、道具/路线分族均衡；实际机会和消费预算分臂记录，不宣称learner分布相同。训练禁止人工携包，既有Bank仅单图806/P0/far，已用保留集局限明示。

CPU唯一owner异步：自然dev16/32各64，最终共同父+9模型自然dev/独立holdout64、真实JS Hunter难中易各64、Python Tactical64；Bank/Bank-P分开32来源×8尾。固定dev seed2026102480、holdout2026110100，未来seed冻结不用holdout选参，报告全seed/尾部/最差点及未训练父和匹配control。新干预零须修覆盖并版本化，不重复等价臂。九run健康固定预算不设能力门禁；之后消费完整曲线决定五seed确认或最小单变量修复，需新冻结预算，不无关占卡。持久owner `scripts.dagger_rl_run`，CPU owner `scripts.dagger_rl_cpu`，每阶段/run终态写原子通知并CLI送达。详见当日报告。

### R1正式启动前技术修订v2（20261024）

逐步重查老师的CPU微测试发现：原固定32条中1条源 `O5_g95-train-004` t817有6tick复活倒计时，静态预测漏掉已知复活位置，导致放置后老师因威胁弃权。不是无路，也不是库存/mask缺失。RED真实快照复现；最小修复只将公开复活/原生控制剩余时间及出生位置纳入完整滑行安全预测，不预测敌人未来随机动作、不改奖励/网络/课程。6项focused回归通过。

原 `r1` 无任何正式run，技术smoke512步/2优化步已原子保存后停止，完整state/policy SHA及未完成fresh restore事实保留于 `PREFORMAL_STOP_RECEIPT.json`；旧scope持久停用不恢复。8卡释放已核验。新namespace `runs/dagger_rl_20261024/r1v2`，协议ID `DAGGER-RL-20261024-R1-v2`、SHA `e75da134f25809215e6413379faefedf0b85cd2c4586a96089a5ea4406e31a59`；三臂/父点/配对seed/正式总预算完全沿用预注册，未消费holdout或正式能力结果。v2直接重查原固定32个真实学生状态，按修正后的公开信息资格披露所有弃权；若接受集合不变则复用已验原生物理分支，避免无效重采。干预分支仍不训练。CPU闭环检查与新GPU技术smoke并行，随后正式固定预算；不恢复已终止旧smoke。

R1-v2技术smoke、正式首有效更新及首run u32均已核验，首sequential固定预算完整结束后立即接续joint。实测ability-head非零梯度/更新，8卡利用率峰71–81%，稳态约1166真实步/秒；详细原子SHA/阶段量/源分母见当日报告和 `FIRST_FORMAL_ACCEPTANCE.json`、`FIRST_RUN_TERMINAL_REVIEW.json`。CPU spawn seed复制错误已按原预注册seed修复并隔离旧Tactical，无GPU暂停。CPU新增被动字段不改变动作/状态，保留技术parity证据；全固定筛选仍在继续，不宣称能力提升或整体完成。

DAGGER-RL-20261024-R1-v2接管复核：首seed三臂均u32，第二seed已接续；每run262144环境步/1024优化步。CPU独立评估无故障marker，8卡现场有效利用，旧全局暂停SHA未变。新增只读FIFO链路诊断，不改变训练配方；初步释放拟合强而contact覆盖/拟合弱，须跨seed与完整自然/保留结果核验，禁止据首seed或低死亡宣称改进。证据继续归档当日报告。

## DAGGER-RL-20261024-R2-PARENT-ANCHOR（20261024，后继预注册）

在R1第三seed仍训练、未消费R1 holdout时，依据TRAIN同输入遗忘/闭环及前两seed自然dev共同退化预注册。新namespace `runs/dagger_rl_20261024/r2_parent_anchor`。共同干净r19g32、policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`、state SHA `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`与matching保存Adam；不继承退化R1控制点。

control与anchor均为R1全程PPO+DAgger joint32，DAgger.2/当前历史各半/route-item均衡/FIFO8192完全不变；唯一处理为anchor臂在公开空手自由存活、敌自由存活且无敌源掉包目标时增加.05父策略合法joint KL。持包（包括敌复活）、原生死亡/糖浆窗口、掉包追取、自身死亡/受控均无该原始辅助梯度；parent/Critic原始辅助梯度为0，仍保留原global clip，不能声称Critic实际更新尺度不受间接影响。无新奖励、战斗规则老师、课程/对手/网络/香蕉目标改动，不叠胶或H1/H2。

配对seed2026102421/22/23，每臂64env×128tick×32、262144环境步/1024优化步；六run总1572864环境步/6144优化步，hyper沿R1。量级只取冻结TRAIN joint首seed u16的256步小batch，网格.02/.05/.1/.2/.5/1按辅助Actor范数相对共同目标最接近.15选.05（实际.1206768），零采用更新；不读dev/holdout选系数。

保存0/1/8/16/17/24/32，CPU独立唯一owner异步；自然dev固定2026102480复用以同协议比较，R2新最终自然holdout2026110200..2063；Hunter20261024难中易各64、Python Tactical64，Bank/Bank-P仍单图P0全far/已使用保留集。三seed全最终/尾部/最差点对同seedjoint-control与未训练父，不择峰。来源provenance仅增等价被动日志，须focused原生初始化/transition和零系数loss梯度等价通过。R2新smoke要求完整64环境fresh next-step/Adam/FIFO hash；通过立即接续正式有界六run，禁止恢复旧smoke/旧队列。

R1九run及预声明Bank预算继续完成；R2持久dispatcher只等待准确R1 GPU终态/设备释放与本轮技术就绪，不以能力阈值、CPU报告或网页阻塞。每run终态主动通知。后继依完整自然/安全及链路证据：可信改善再新冻结五seed确认；否则最早零链/保持单变量定位，不盲目扩展或无关占卡。当前预注册不等于已完成smoke/正式首更新。

R2首GPU前技术补充：新增collector live-row元数据，排除原生终局/aux deferred终局结算行的父KL，避免把待结算状态当战斗；该bit不进Actor/Critic输入，不改PPO/control/reward或任何预算/seed/系数。RED→GREEN按原生等价及独立目标梯度复核。旧协议/seal/READY保留于technical_history/terminal_validity；当前协议SHA `75f6a3e97da63d686ccf07cef639dc0fdd7d7e74b18401bbc75bfcd7036061e5`。这不是按短点/holdout改变能力配方。

R2首个GPU步骤前补充只读CPU诊断登记：同R1固定26个TRAIN真实取包机会、8来源局，仅首seed2026102421两臂u16/17/32各64tick greedy学生闭环，最多9984逻辑环境步、0采用优化步；老师只记录不接管。各配对seed固定control u16的同64学生状态比较父策略漂移，两臂固定点精确重建FIFO/replay拟合。诊断不进入训练或holdout选参，不变更协议、奖励、课程、对手或预算；唯一CPU诊断owner与CPU/JS能力评估独立。原始SHA与范围见R2 `ADDITIONAL_CPU_DIAGNOSTIC_PROTOCOL.json`。R1自然局owned自放后自吃仍全零，window阶段释放高于carry，不能把TRAIN局部拟合称部署迁移。

### 中点自然评估seed修复（20261024）

只读审计定位了确定的调用错误：`dagger_rl_cpu.owner → natural(cp)`未显式传`JE10_ROUTE_SEED_START`，u16采用JS脚本fallback2026101800，而冻结dev应为2026102480。九个u16各64局共576个错误协议episode不参与尾部/切换前后比较；原始文件保存在R1 `technical_history/midpoint_seed_default`，不删除、不择优。最终dev由evaluate显式传seed，holdout由独立入口传seed，二者协议正确且不变。父点被动字段回填也正确使用预声明seed。

唯一CPU回填owner按相同9个u16权重、原dev2026102480完整重算，并逐shard断言seed数组后才恢复R1完整比较消费。旧report consumer在等待状态安全移交；GPU Bank及R2接续不以回填/报告为门禁。R2在首GPU前补RED→GREEN：natural默认显式用冻结dev，holdout明确参数，拒绝继承进程环境的旧seed；不变训练数学、奖励、课程、seed或GPU预算。旧seal保留于R2技术历史，更新seal后才重写技术READY。修复开销是纠正评估协议，不计正式训练步，不新增独立holdout、不据结果调参。

R2首GPU smoke attempt01在collector tracing阶段检出`BunState.done`不存在，0有效环境/优化步、0checkpoint，GPU worker安全退出。根因是终局live-bit新增时使用了非该引擎字段；原生终局/horizon及残留结算已有持久化`aux['terminal']`。RED覆盖实际collector调用形状并复现AttributeError，GREEN验证live行与deferred终局行；最小修复仅`live=~aux['terminal']`，未改变冻结科学处理。失败日志/seal/owner收据保留于`technical_history/collector_live_api_smoke_attempt01`。首GPU前原12项机制检查、新2项seed检查和1项collector API检查均GREEN；相关7项另行全通过。

技术恢复立即重封seal并启动attempt02；独立CPU evaluator通过实际锁复用，禁止重复owner。未训练formal从共同干净父与原Adam开始，绝不继承失败smoke。旧r1排除smoke和旧O5/continuous/储备均未恢复。CPU诊断launcher的venv符号链接误解析也已修正，失败仅导入且0轨迹/更新，历史保留，不计正式技术smoke。

R1-v2完整终态（含纠正中点）已验收：九run/u32、288次更新、2359296环境步/9216优化步；63声明checkpoint全部state/policy SHA及Adam计数/配对初态通过，完整比较SHA `53127b60ba1b683037b13f7f8fe18814cbbc63b83b0f5e0471e3640ca9f015e5`。自然holdout sequential/joint均[0,1,0]/64，PPO[10,2,4]/64，父19/64；前二者相对父平均−29.17pp（seed×来源CI分别[-40.625,-18.229]与[-40.625,-18.750]pp），相对匹配PPO−7.8125pp。条件Bank较好不代表自然提升；全部owned自然自吃仍0，无候选晋升。R1 GPU预算正常结束后立即接续已冻结R2，CPU修复不阻塞GPU；旧暂停SHA不变。

R2修复后attempt02技术smoke通过：512真实环境步/2优化步；Actor ΔL2=.127344695、Critic=.069213195，ability-head更新.002210671/梯度7.549710；真实raw KL ability梯度.178899601、raw Critic0。完整64环境fresh next-step/Adam/FIFO hash exact，state SHA `160d456f35ed7fab146c96d29c51bb294c9ef49bec5ab788f2c3a0701f74218a`。正式control_s2026102421 u0已验证params/Adam与父输入精确相同、smoke不继承，GPU正式collector编译/执行中；尚未用u0或编译冒充正式非零更新。

R2首正式control_s2026102421/u1已独立SHA/finite/nonzero验收：8192环境步/32优化步，Actor2.424929659、Critic.460117037、ability-head .086561956，state `bef7c6cd993a799ac7c93b9e74ded29a65439ca964f298168168376641f23687`。158道具行均来自真实取包Bank，自然6400tick本批尚无持包/道具标签，不作覆盖充分断言。被动5秒GPU采样器只记录设备利用率、包括编译/保存零值，不造GPU工作；新增已保存学生状态的全图hazard弃权分来源审计0环境/优化步。正式健康固定预算继续，评估独立。

R2 run1/6 control_s2026102421正常u32终态：262144真实步/1024优化步、32更新finite/nonzero，state `0115f4fd2c4c3ab06b5232af1e8f9101b892f93ef4a78663b108d588c4622bc9`，policy `354cc881c2d66a02d3bf2f559e952209444b47e818794f9b45f5354305a794ae`；配对anchor的u0 params/Adam/state/aux/RNG逐项exact，自动接续，未等待CPU评估。实际源曝光 {"natural": {"ticks": 243616, "item_release_labels": 6, "item_contact_labels": 9}, "real_pickup_bank": {"ticks": 18528, "item_release_labels": 195, "item_contact_labels": 106}}；重置配置75/25不代表全程tick占比，snapshot门控不是整段分母。当前8卡同一worker编译anchor新损失，CPU已开始固定u16/最终评估。

R2处理实际首更新验证：anchor_s2026102421 u1 KL=.003181872、Actor2.421935174/Critic.454058753，配对control/anchor的首collector末state/aux/RNG/history exact；更新后Actor差.413245690、Critic差.117730045（raw辅助Critic0，共同global clip可改变Critic步长），不是零差等价臂。run2/6 anchor_s2026102421已u32，state `c938a6bb955a35861b241748ae9f15f01aa8111eaa111941375bc0a74fc8d08c`、262144环境步/1024优化步、32更新finite/nonzero；立即接续control_s2026102422。自然来源247471tick仅34释放/94接触老师行，实际覆盖分母保留。首seed anchor u16自然dev12交/12死亡（64局），control u16为2/7；中点不是最终提升/晋升证据，未停固定预算。

R2 run3/6 control_s2026102422正常u32，262144环境步/1024优化步，32次finite/nonzero；state `9016e3375c2d4d8da6276c85da72ce0b2e99ca79fa1a84ca1b6913fbd1b6f85b`/policy `b10302e92a393739ae8764fea19792e2aadf9f9529b34d07a41d244c795b73cc`独立验收。自动接续anchor同seed，不等CPU。第一seed anchor最终dev虽保持12/64自然交付，Python Tactical仅2/64（父10、control6）；JS Hunter H/N/E为0/3/6（父1/2/6），owned自放后自吃仍0，不能宣称整体安全战斗提升。完整三seed与独立保留集待消费。

R2 run4/6 anchor_s2026102422 u32完整终态，262144环境步/1024优化步，state `9619280e385d82d00c3a5f4dd35b4cf3ff712ccd3dc58b10aba7e28bf035adf6`/policy `5ae4bd579aadaca51a15139e41176fb373508ee1a96b598104d57ce39bbc5399`独立验收，第三seed control立即接续。首seed同43 TRAIN combat状态移动parentKL control .390953→anchor .037796，固定TRAIN 26机会完整链13→17（正净收益11→17、各0死亡/丢包）；均不是独立部署能力，natural owned链仍0，Tactical风险保留，不择峰、不扩确认。

预注册条件确认 `DAGGER-RL-20261024-R3-CONFIRMATION`：共同r19g32及保存Adam，policy `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`/state `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`；五配对seed2026102431–2435，control/anchor同R2唯一区别+.05空手战斗parentKL，全部目标/奖励/课程/对手/结构/Adam与R2不变。十run各u32、64×128、262144环境步/1024优化步，总2621440/10240；冻结0/1/8/16/17/24/32，dev2026102480、新自然holdout2026110300各64，Hunter20261024难中易各64/Python Tactical64，Bank固定旧集单图P0全far。协议SHA `1783a255118d7a7500a5350e23b10e412c27f090ec175b63351fa48ac6f4422a`。仅在R2全seed/尾部/最终/父点/保留结果支持后，写入可消费SCIENTIFIC_SELECTION才启动；不是当前健康run能力门禁、不调系数、不择峰，不支持时不机械确认。五新训练seed/新自然holdout独立，Bank及战斗保持集重复使用，不能称跨图新确认。RED→GREEN及数学AST一致证明已通过，R3尚0GPU步；新namespace/run/state/cache/eval输入准备完毕，旧队列暂停保留。

R2六正式run/u32全终态已独立验收：192次finite/Actor-Critic非零，1572864环境步/6144优化步，六对终点SHA记录FORMAL_TERMINAL_ACCEPTANCE；最后anchor_s2026102423 state `f1f37fd2d72a4ada9127b2671e98725e4d0eba7d2f5333fc351dc6c12eb9d2ee`。owner已转预声明Bank/Bank-P GPU评估，不充当训练更新；CPU/JS独立消费最终点/新保留集。R2全程joint，无u16→17阶段切换，只有中段相邻点；R1 sequential才有真实切换。R3条件dispatcher已持久等待R2确切GPU释放+技术READY+完整证据后的SCIENTIFIC_SELECTION；无选择不启动，有明确不支持则退出，不恢复旧队列。R3技术检查18项为原12项冻结机制复用+6项新命名空间/选择边界/seed/collector检查，必须另做新GPU smoke，当前GPU训练0步。

### R2自然固定终点与历史标签审计（20261024）

六正式run均u32，1572864正式环境步/6144优化步，192次finite、Actor/Critic非零；当前唯一GPU owner继续预声明Bank/Bank-P，CPU/JS全部7最终+6中点及新自然holdout已完成。评估GPU步不计训练更新。

| 模型 | 自然dev交付/64 | 新自然holdout交付/64 | holdout死亡对局/64 | dev中点/最终尾部均值 | Python Tactical安全胜/64 |
|---|---|---|---|---|---|
| 未训练共同父 | 12 | 14 | 25 | — | 10 |
| joint-control | [0,0,9] | [1,2,7] | [13,21,11] | [1,4.5,6] | [6,6,2] |
| joint+.05战斗parentKL | [12,11,2] | [12,18,7] | [17,14,6] | [12,10.5,5] | [2,9,10] |

anchor相对匹配control交付平均+14.0625pp，来源局×训练seed描述性CI[0,29.167]pp；相对未训练父−2.6042pp、CI[−16.667,12.5]pp。control相对父−16.667pp、CI[−28.125,−4.688]pp。不能从严重退化control宣称整体提升或安全等价；第三seed的dev回落和第一seedTactical回落均保留。全部模型自然owned自放后自吃仍0，其他来源香蕉接触不能冒充该链。

只读固定输入审计：以每seed control-u1前≤64个release为固定参考，u32 control一致率[8/64,28/57,10/32]，anchor[11/64,0/57,3/32]；相同标签在最终FIFO的逐行精确成员数分别[0,0,2]、[2,0,0]。样本含重复，unique分别54/50/32，不当独立环境状态。与当前FIFO近乎全拟合同时存在，支持“当前拟合掩盖旧机会遗忘”，尚不单独证明其导致自然0链。原始证据FIXED_ITEM_RETENTION_COMPLETE及FIXED_LABEL_EVICTION；0新环境/优化步。

六点快照共2304行中，自然来源2134行，205行真实持包且空闲，其中23行有香蕉库存或近处自有香蕉；仅10行通过全图清洁及敌距先决条件。hazard拒绝重叠，非全程分母，不能唯一归因为全图fuse过滤。全程自然来源每run约24万tick，仅6–70条release标签；实际source事件/标签按updates逐项保存，不把75/25重置配置当实际tick混合。后继选择待固定Bank/Bank-P全证据；R3若选择，只确认不改配方的保持作用，绝不宣称自然香蕉已学会。旧暂停收据SHA保持不变。

### R2完整验收及R3科学选择（20261024）

42个声明点SHA/Adam/Actor导出/配对u0全部通过；14份自然dev/holdout逐局seed由独立consumer复核。全部Bank/Bank-P、Hunter/Tactical、固定game0偏航hash与JS近似GAE、原生重建GAE、来源曝光和旧标签保持已消费。完整比较SHA `9bbec6435eb8932eedc396952f0de5e3bb5b46e1d454c32ff3d3b548fbc9195f`；独立证据SHA `ae08a9ec9799a198b956bc749bd18db8963710ba4c8e4e2954650677e737034b`。8卡0MiB/无compute app、训练/评估GPU worker及owner退出已独立核验，COMPLETE_TERMINAL_NOTICE持久保存。

R3选择执行此前已冻结五seed复验，科学对象限于“+.05战斗父KL的部分保持作用能否复制”，绝不是已经优于未训练父、自然香蕉链已学会或晋升。支持：holdout相对配对control交付+11/+16/0（各64），相同combat输入三距离层parentKL均减小，条件Bank直接交付95%左右且Bank-P保持。反证：相对父交付−2/+4/−7，第三seeddev2/64、第一seedTactical2/64、Hunter hard全0与自然owned自吃全0，均进入选择文件而非择优删除。希望有限、仍有风险，五新seed复验用于估计保持效应和尾部，不盲调量级，不扩展退化模型；同干净父/Adam重新开始，不继承R2权重。

SCIENTIFIC_SELECTION真实绑定协议及完整R2比较SHA；原持久dispatcher负责空卡后的新64环境fresh恢复smoke→十run各u32，CPU/JS异步。R3额外只读TRAIN诊断预先登记同26真实机会/8来源×首seed两臂16/17/32×64，最多9984逻辑诊断环境步、0采用更新；全五seed固定旧标签/FIFO成员/64状态parent漂移和实际source曝光辅助解释，均不改变训练配方。后继最小问题优先检验共享FIFO逐出旧机会导致的保持不足，必须另预注册为单一采样变量，不混进本轮。

### 下一变量的独立TDD准备（不进入R3）

根据R2明确的“旧标签被逐出且旧输入拟合遗忘”，单独登记R4 reservoir候选：共享8192条DAgger记忆只改淘汰策略（FIFO→uniform reservoir），同R3 joint+.05保持配方/同r19父Adam、三新配对seed2026102441/42/43、六run各u32，若最终选择总1572864环境步/6144优化步。loss、old/new与item/route抽样比例、老师、奖励、课程、对手、网络均不变，reservoir独立采样RNG不消费环境RNG；下轮新自然holdout2026110400。当前仅问题/预算预注册，必须在完整R3证据后另写选择、封存和真实smoke，未启动GPU或采用更新，不能借此绕过科学判断。

纯采样器RED→GREEN四项通过：未满容量与FIFO相同且输入不变、checkpoint恢复/行字段严格对齐、500个固定采样seed无系统性抛弃早期标签、非法流计数拒绝。该模块不被活动trainer引用，R3全部封存源码逐SHA复核未变。它不生成状态/标签或携包，也不改变R3数学，不是R4已技术就绪或训练已启动。

### R3新技术smoke及首正式非零（20261024）

新namespace完整64环境smoke通过：512逻辑环境步/2优化步、Actor Δ.127344695/Critic .069213195/ability-head .002210671，fresh进程next-step/Adam/FIFO完全一致。首正式control_s2026102431/u1已独立比对u0共同父params/Adam，未继承smoke；8192真实学生环境步/32优化步、Actor2.535293225/Critic.714964272/ability-head .099319283，state SHA `32d7e26cc172d442ea033485bac71003af04597077aff8aa4d31a4b430f61a06`。首批61个道具标签全部真实取包Bank，自然6528tick暂无持包/道具老师行，不称自然覆盖充分。

8卡首次有效峰值约64–75%，被动5秒采样包含编译/保存低值，不能称全程高利用率。为了减少新namespace重复编译，只复制同环境完整JAX cache key校验的编译产物到独立新cache，没有共享writer、覆盖既有entry或继承R2 checkpoint/Adam/RNG；COMPILED_CACHE_REUSE保留逐entry SHA，所有fresh恢复与真实更新仍重新执行。独立只读终态monitor已上线逐run验收SHA/finite/非零与通知，不控制训练。observer一度把ability头(weight,bias)元组当矩阵算norm，旧错误日志保留，改为tree-leaf L2并重启此只读observer，首更新复核通过；GPU trainer无中断/额外更新。

### R4候选整合准备完成，尚未执行

R4协议SHA `536c5a6d211e2bdf19c8f5830798a0392639c2c3b47997eacf1080ddde9acae9`。已物化独立输入、源码seal、CPU/eval/cache命名空间；两臂均保留同一+.05 combat parentKL，唯一差别为DAgger FIFO/reservoir淘汰。23项focused GREEN，包含40批跨容量FIFO控制与原R3的history/seeded replay逐数组精确相同、核心collector/objective/updater/commit AST完全相同、选未获准时无法启动、空机会不伪造标签及历史恢复。新增next_update_probe的RED复现旧单步检查不足后，准备了真实fresh64 on-policy collector→memory/replay→Adam下一整次更新的reference/新进程逐digest校验；尚未执行GPU，不冒充smoke已通过。

若完整R3科学选择该候选，新技术smoke明确计量：保存分支512环境步/2优化步，reference continuation与fresh-process replay各512/2，总执行1536环境步/6优化步（含重复技术分支），正式采用0；fresh所有env/aux/RNG/模型/Adam/stream seen/arrival元数据和回放抽样必须完全一致。正式训练不继承任何smoke分支。该技术检查不使用能力阈值。持久接续脚本只等待完整R3选择与准确GPU释放，不能自行选臂、择峰、恢复旧队列或把候选准备当执行。

### R3前一配对seed异步诊断（非最终结论）

首seed2026102431完整dev（各64）：父12交付/22死亡，control2/15，anchor13/16；Python Tactical父10、两臂均7；Hunter难/中/易父1/2/6、control0/1/3、anchor0/1/4。全部自然owned自放后自吃0，不以首seed或dev择优。固定TRAIN26真实机会下control释放26/接触完成14/正净14，anchor释放26/接触完成23/正净13，均0死亡丢包；完成更多不等于正净更多，不能把合法释放/接触当有益利用。两套生产greedy自然对局和native greedy TRAIN诊断仍分开。

前三个run全部u32独立终态，第四run健康继续；CPU评估不阻塞GPU。R4淘汰干预另预声明首实际生效审计：容量未满前两臂必须同seed/同参数/同回放（仅审计metadata可不同），超过8192后才检查实际抽样与ability参数差异；u1通用非零不冒充淘汰策略生效。若最终干预一直未激活，记inactive并先修覆盖，不重跑等价臂；健康run不会因此短点停训。

### R3前四run原子终态（20261024）

| run | 最后更新 | 正式环境步/优化步 | state SHA |
|---|---|---|---|
| anchor_s2026102431 | u32 | 262144/1024 | `1c535b63c20d0f97a2f2b14093a463b909f73d1346acc3b478181ba5d8e0d1d7` |
| anchor_s2026102432 | u32 | 262144/1024 | `97aeab220a36b95e1c93f77b89adb91121a31b5b03ab25346b5067baaa70cc0a` |
| control_s2026102431 | u32 | 262144/1024 | `9d663486fe65a7ac2ce7627a027953b8e6aa1c953f446d36dd75dc0a0e4f8d0a` |
| control_s2026102432 | u32 | 262144/1024 | `a7eca01a34b5d6614c3344d03ddc37564770add0ea66dc7a90d8133b198ea1b7` |

均经独立SHA、Actor导出、32次finite/Actor-Critic非零复核。累计1048576正式环境步/4096优化步；第三seed control已自动接续，CPU评估异步，不等待报告。R3仍需完成剩余六run及全部固定评估，不宣称实验完成。

### 接管与候选接续修复（20261024）

已独立复核R3前七run全部u32原子state/policy SHA、32次finite和Actor/Critic非零；累计1835008正式环境步/7168优化步。第四seed anchor继续固定预算，唯一CPU/JS owner异步消费，不重复训练owner。339个R3封存源码SHA保持精确不变；旧全局暂停SHA保持不变。

R4候选尚无科学选择、0GPU。发现等待dispatcher的CPU evaluator和报告命令错误引用R3，RED测试明确复现；修为各自R4入口，12项相关GREEN。仅终止并重启未派发GPU的等待dispatcher，未动R3 owner/worker。收据`runs/dagger_rl_20261024/r4_reservoir_candidate/DISPATCH_NAMESPACE_REPAIR.json`，原协议/预算未改。后继仍须消费完整R3与独立新技术smoke。


### DAGGER-RL JS归因统计更正（20261024）

> 20261024测量更正：旧JS `dagger_rl_metrics`把省略的requestedCell登记为undefined，导致自放后接触被漏记。此前R1/R2/R3“owned自吃/完整链全部为0”的断言撤回；这些v1专属归因计数不可作为能力失败或后继选择依据。真实释放、总体接触、交付、死亡、动作/状态、训练标签及JAX Bank不受影响。R3全11模型dev/holdout按原seed精确重放补记，见`METRIC_V2_PROTOCOL.json`，尚未完成前不填入新数值。

只读版本化修复，不改活动R3封存源码/数学/奖励或老师；候选R4若执行，必须使用同一修正版度量，两臂共同不构成处理变量。原始v1结果保留，补记重复局不增加独立样本量。

### DAGGER-RL-20261024-R4-RESERVOIR 执行选择（20261024）

R3完整70原子点/22自然seed清单/固定偏航及奖励-GAE链接/五seed曲线/Bank来源CI已验收，有效证据`VALID_PREDECESSOR_EVIDENCE.json` SHA `e8b80919e72795958d4eb61f6c24f9d19dbe412fba04a099d4381efc0aaab099`。R3 anchor交付相对control +8.4375pp、相对共同未训练父−16.25pp，明确严重共同退化，不晋升；选择本轮只检验已确认的固定旧输入遗忘和FIFO逐出，不把自然owned=0作为依据，也不延长R3权重。

共同r19干净父/保存Adam、三配对seed2441/2442/2443、control/reservoir各u32，新namespace `runs/dagger_rl_20261024/r4_reservoir_candidate`。两臂joint+.05 combat父KL、DAgger.2、新旧.5/.5、item/route各4096不变；唯一变量8192容量FIFO→Algorithm R。各262144环境步/1024优化步，共1572864/6144，保存0/1/8/16/17/24/32；新自然holdout2026110400。当前/ring/固定实际预声明0/0/100% JAX tier3，Python Tactical/真实JS Hunter独立。旧Bank单图P0全far重复保持，不当独立跨图确认。

已30focused及JS fallback归因/60tick全状态parity GREEN；全新smoke执行1536逻辑技术环境步/6执行优化步（含reference和fresh重复），0正式采用；必须fresh64 collector→memory/replay→Adam exact和finite/Actor-Critic/ability非零。首正式通用非零不冒充淘汰干预生效，首次溢出另验实际回放和参数差。CPU修正版v2两臂共同启用，R3补记继续异步；v1专属计数撤回，不据其作选择或假称已全部补记。健康固定run不中断，六run后按冻结条件审视保持/完整链和战斗，不机械重复或盲加倍率；无有理后继或继续严重退化时明确报告。旧O5/continuous/储备始终暂停，不发布。

### R3完整验收及接续上下文（20261024）

最终比较SHA `c704432cf7e56b847ea1ece99f20f99ff9e6fc064683c5db7764c900c393a3e9`；完整独立证据SHA `e45659488a3743136c2520e484b62b4f034d393e57c46e3e5fd8119e0e7cc57a`；归因补记SHA `16a25532a3bb5acfd0f2a5c96dec0e0bda937474c885184213ce9b2b6c86009a`。全部70声明点、22自然seed清单、固定首偏航/近似GAE、源曝光、五seed曲线/尾部/最差点、Bank来源局聚类CI及原子模型/Adam通过。1408重复局新增独立样本0，所有非受影响字段和固定录像trace精确不变。R3 GPU owner已退出且接续前8卡0MiB/0%/无compute app；最终通知中8个compute app属于随后已授权的R4 smoke，不是R3未释放，见`FINAL_HANDOFF_CONTEXT.json`。

更正后R3 holdout所有阶段自有滑行有效持包正净proxy：父2，control[0,0,0,2,3]、anchor[1,2,0,2,1]；持包阶段放置的对应正净proxy父2，control[0,0,0,0,1]、anchor[1,0,0,0,1]。不再声称自然owned链全零，也没有稳定胜过父点的证据。接触按接触阶段、滑行完成按放置阶段归类，跨阶段计数不可直接作cohort转化率。legacy `owned_slide_harm`是未保持有效携包或净位移非正的余类，包含空手/原生受控/掉包和可能在滑行中合法交付，不能等同实际自伤/危险；实际死亡/掉包/交付指标另计。正净proxy不是全释放决策Q，不据它单独晋升。

R4已接续新smoke，正式更新尚未发生；旧队列保持暂停。

### R4新smoke与首正式更新（20261024）

新smoke fresh64下一整次collector→memory/replay→Adam逐digest精确通过，512保存技术步/2更新+1024参考与fresh重复技术步/4更新，总1536/6，正式采用0；ActorΔ.127344695/CriticΔ.069213195/abilityΔ.002210671。共同父/Adam未重置。正式`control_s2026102441/u001`独立验收8192环境步/32优化步，finite、ActorΔ2.355366647/CriticΔ.652384350/abilityΔ.084143826，state SHA `b6bdeaf4fc236c311e8266663cadadc43b0c90fae4d2b53ca67c91d2868793f4`；原子policy/params导出精确，未继承smoke。8卡计算段采样峰71–79%，包括编译/保存的平均值和原始5秒采样另存，不冒充全程高利用率。首批135道具老师行，不把它当自然覆盖充分或淘汰差异生效。已开始三seed六run固定正式预算，CPU/JS独立；须持续监督各run终态和首次实际淘汰效应，不能因启动结束接管。

### R4接管核验及第二seed干预（20261024）

已完成并独立验收control2441、reservoir2441、control2442各u32；每run262144环境步/1024优化步，32次finite及Actor/Critic非零。第三run原子state SHA `dfaa72fc022a61003b5b588a026e21413d368cafabdb2a63588086bb598e2b57`。reservoir2442继续固定预算，CPU/JS唯一owner独立消费，没有新增或重复owner。349封存源SHA逐项不变，旧全局暂停SHA精确保持。

第二seed u0–u6两臂params/Adam/env/aux/RNG及监督数据精确一致；u7 accepted_seen8233首次超过8192，同批新学生状态下重建memory/replay精确、回放分化，Actor差.565868123/Critic差.515760255/ability差.005553620，干预实际生效。该差异不冒充能力改善。首seed对应u9，均见固定eviction_effect收据。

来源覆盖已单列：control2441自然242973tick仅38释放/149接触标签，真实取包Bank19171tick产生196/213；control2442自然241085tick为53/141、Bank21059tick为131/129。来源样本不同，不称数据机会预算实际完全相同；更不可据Bank单图P0保持推断自然多座位泛化。原始全量计数随run冻结。旧“owned链全零”已撤回，R4协议中该早期候选措辞只作历史预注册，实际后继选择以修正版归因、全链及未训练父为准。

### R4六run正式更新终态（20261024）

三配对seed两臂均u32，192次finite且Actor/Critic非零，1572864正式环境步/6144优化步。各run原子state SHA如下；GPU随后只执行预声明Bank/Bank-P，CPU/JS独立完成最后点及自然holdout。此时尚未冻结完整能力结论，不择峰晋升。

| run | 原子state SHA256 |
|---|---|
| control_s2026102441 u32 | `2cef96c6c5ed6100bce78db6e66300a2f69a99bb117710edfdcf382005aba408` |
| reservoir_s2026102441 u32 | `ff17f3260f41f9b649c85e1932accc8ea8d7d09058db8cccf52c481bad925ed3` |
| control_s2026102442 u32 | `dfaa72fc022a61003b5b588a026e21413d368cafabdb2a63588086bb598e2b57` |
| reservoir_s2026102442 u32 | `eab6926621df906e112531511937c48532e7d6052ae832e96be8756b5be4b828` |
| control_s2026102443 u32 | `d55fc6af34d00aed70ac9c0e6b57a9b76d80a5e6505c32e5a25d5efabac215aa` |
| reservoir_s2026102443 u32 | `1d7929f5d7f0ae1c6c15ecec3eca84ef88bd15d6c650d313a3f7e9401772d910` |

三seed首次实际淘汰分别u9/u7/u9；各自淘汰前完整状态/监督精确一致，溢出后同批输入的回放与实际Actor/ability更新不同。真实监督干预已成立，但不代替自然能力证据。新双视角审计只读同64个实际棋盘；P1视角来自规则对手占据状态，不冒充P1学生on-policy或独立对局。

### DAGGER-RL-20261024-R5-SOURCE 冻结合同与接续（20261024）

R4三seed完整结论：自然holdout父21/64，FIFO[15,9,2]、reservoir[9,18,1]；后者相对FIFO +1.0417pp，来源局×seed CI[-12.5,15.625]pp，相对父−18.2292pp、CI[-35.9375,0]pp，所有三个最终点都低于父。Tactical安全胜父10、FIFO[8,19,7]、reservoir[6,10,5]，不能以低死亡（部分伴随机会锐减）声称安全保持。固定早期release输入u32拟合[27,15,36]/64→[57,61,56]/64，记忆机制成立但自然全链不成立。Bank直接交付reservoir相对父+9.7656pp（来源×seed CI[4.0365,16.0156]pp）仍仅旧map806/P0/allfar，不反证自然退化。不晋升/不五seed扩大等价确认。完整比较SHA `6a00aca1259d0b51923e77110981097115dd1fb23423a641da6fd10aec470d33`、独立证据SHA `6df95f44df1b157717e9a142332c9df5965599b20f279df35a9a1c2525d798f6`；42原子点通过，GPU/CPU owner均终态，8卡0MiB/无compute app。

新namespace `runs/dagger_rl_20261024/r5_source_candidate`，协议SHA `5c48335f08b444a9d1316d16cab0be797ba9776d42452ce258b2724309642d99`。两臂control/natural_balance，三配对seed2026102451/2452/2453。共同r19干净父policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`、state SHA `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`，原保存Adam相同；不使用任何R4训练终点。共同joint PPO+DAgger .2、combat父KL .05、uniform reservoir8192、新/旧loss .5/.5、item/route回放4096/4096、学习率7.5e-5、gamma.995、lambda.95、PPO clip.1、entropy.005、value.5、global clip.5、单epoch/全局minibatch256。

唯一变量：旧item回放按来源不分层→自然/真实取包Bank各50%（两者均存在时）；路线抽样及槽位逐行保持，缺来源时精确回退control。共同记忆改用reservoir是保留R4已证实的TRAIN输入保持机制，绝不宣称它整体胜父。来源元数据随到达ID保存，不是模型输入，缺历史证据不推断；PPO始终当前on-policy，旧标签只进监督。香蕉老师/胶目标、奖励、课程、对手、网络均不改。仅真实取包Bank允许携包；无人工预携包。此假设检验来源支持，非已证明唯一根因；来源分层可能改变release/contact比例，必须拆报。

每run64env×128tick×32更新=262144环境步/1024优化步，六run1572864/6144；每更新原子保存模型/Adam/env/aux/RNG/source-memory，声明消费0/1/8/16/17/24/32。新smoke仅技术1536执行环境步/6优化步（含reference/fresh重复），正式采用0，验finite/Actor-Critic-ability真实非零梯度与更新/fresh64下一collector→来源回放→Adam精确恢复。38 focused及生产JS归因parity GREEN，源已封存。

固定评估：自然dev seed2026102480，u16/u32各64；共同父与六最终点自然holdout新seed2026110500各64，真实JS Hunter难/中/易各64与Python-host Tactical v2/JAX环境分别报告；旧Bank/Bank-P仍单图P0/allfar重复保持。唯一CPU/JS owner消费已保存点，不触发GPU。实际对手current/ring/fixed=0/0/100% JAX native tier3，部署greedy老师关闭。

复现入口`.venv/bin/python -m scripts.dagger_rl_r5_run`，CPU入口`CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu NODE_OPTIONS=--require=./scripts/dagger_rl_metrics_v2_preload.js .venv/bin/python scripts/dagger_rl_r5_cpu.py`，终态消费`.venv/bin/python -m scripts.dagger_rl_r5_run complete`；现有marker跳过，不重跑已完成。持久owner串行完成六固定run，再Bank；CPU异步，健康run不中断；每run终态notify及本线程主动消费。`USER_STOP.json`在下一更新边界安全保存退出，不恢复旧O5/continuous/储备。后继依据完整自然/安全/来源证据：有希望才预注册五新seed独立确认；真实干预零先修覆盖，持续共同退化则根因排序后选择新的单变量，不机械重复或盲加倍率。

### R5技术smoke终态与正式派发（20261024）

新namespace全新smoke通过，真实技术1536执行环境步/6优化步（含reference/fresh重复），正式采用0。ActorΔ0.127344694653、CriticΔ0.069213194917、abilityΔ0.002210670616、ability梯度7.549710274。fresh64下一完整collector→source-memory/replay→Adam精确恢复；全参数/Adam finite、原子SHA通过。首512行无自然item标签，恰应精确回退；其params/Adam/env/aux/RNG与旧参考同seed技术批完全相同，来源元数据不影响训练输入。此时尚未验实来源处理的实际效果，不用通用首更新代替干预证明。

唯一训练owner已自动进入正式阶段，配对首seed control的u0共同父/Adam/env/RNG已原子建立；CPU独立评估共同未训练父，未等待网页/报告或能力阈值。来源首次实际干预由独立只读source_effect owner消费，必须同批学生状态与相同memory、路线回放逐行不变、item回放及实际Actor/ability分化。持久terminal/resource/diagnostic/full-completion消费者全部只读，无新增环境或优化预算，不重复CPU/JS evaluator。旧暂停收据SHA保持。

### R5首正式更新独立验收（20261024）

`control_s2026102451/u001`已完成8192真实环境步/32优化步，finite，ActorΔ2.199616237220、CriticΔ0.825062420334、abilityΔ0.083243705332、ability梯度3.914235353。共同父/保存Adam精确、smoke未继承，原子state SHA `57195c48a3e1fcfed6b120c0e36864747a13de9cc8022d6182066189a7bd424e`。全部8个保存on-policy state/key shard均实际变化（各8env×128tick），见EIGHT_SHARD_FIRST_UPDATE；实际设备利用率5秒原样采样，编译/保存零值保留，不夸为持续满载。首更新含启动编译吞吐不能当稳定吞吐。正式固定六run继续，CPU独立，必须继续监督终态而非因已启动退出。

### R5首对来源干预验实（20261024）

control2451已u32正式终态、262144环境步/1024优化步；natural_balance2451继续固定u32。首个实际来源干预u7：u0–u6完整params/Adam/env/aux/RNG/history精确一致；u7两臂同批学生状态和相同memory，路线回放行及位置逐项相同，道具回放由自然353/Bank3743改为2048/2048。Actor差.916511671458、Critic差.609646177640、ability差.007924739945，已实际进入optimizer，不以通用非零更新冒充处理效应。Critic间接差异可来自全局clip，不称共享主干。

当时记忆仅自然7行（release2/contact5）、Bank69行（release52/contact17）；这是相关学生时间点，不是7/69个独立机会或对局，重复回放不创造新环境样本。来源分层同时改变release/contact组成，不可单独归因为几何来源或某个动作子类型，须结合完整链和独立保留集。原始source_effect与FIRST_SOURCE_SUPPORT_SCOPE保留。8卡实际计算段现场采样70–80%，启动期零值仍保留；编译后控制run约1109环境步/秒。

### R5六run正式终态与只读消费修复（20261024）

六run均u32，192次finite/Actor-Critic真实非零，1572864正式环境步/6144优化步；8卡随后执行预声明Bank/Bank-P，CPU/JS独立完成最后点和新holdout，尚不宣称完整科学终态。三seed来源首次实际干预分别u7/u6/u6，均同批状态/记忆且路线回放精确相同，item抽样及Actor/ability真实分化。

| run | 原子state SHA256 |
|---|---|
| control_s2026102451 u32 | `edb565181747b373aa444a520815254a5aa7151348297eb474643f4209680f23` |
| natural_balance_s2026102451 u32 | `4f588a6b5653a929f0a4c3cbf7e3d0736e9b84fbab51951ad593a5d4027ce75e` |
| control_s2026102452 u32 | `14fd1a65a53c10094f25409c2794e1bc418de80743554954f6b19385ca59204f` |
| natural_balance_s2026102452 u32 | `6d0019ca946ad2bdfcd19f76b34493748586220dce24e377eaa0c75965aaa4cd` |
| control_s2026102453 u32 | `3ba5515ceac26f9b3e8b3994d773cdf853ceb7c3435fba7d72f784aa5275cf6f` |
| natural_balance_s2026102453 u32 | `38ecfeaf619025d63f385604694358fbd5dae44fc1e3a71230d2664b599277f3` |

只读旧标签成员审计遗漏R4硬编码路径导致FileNotFound，未修改任何模型、标签、奖励、原始轨迹或对局结果。已RED→GREEN将目录绑定R5 owner并补回唯一失败的成员审计，6模型SHA均对应R5终点；其他通过诊断不重复执行。旧失败日志/退出收据完整保留，见CONSUMER_NAMESPACE_REPAIR。所有封存训练源SHA精确不变；修复新增环境/优化/独立评估步均0。

### DAGGER-RL-20261024-R6-OPPONENT 冻结合同（20261024）

R5六run/42声明原子点、CPU/JS、Bank/Bank-P和全部只读诊断已完整验收；比较SHA `1bd1ca3290ad80e560353d2d4e73a002f00a42a27dde7e21da805d7ebb2b900c`，完整证据SHA `119c567ddea7f88fb738526a8f11a7fcf83e13f0e22e58cc26882d0d98283b5c`。父点自然holdout交付16/64，control[4,2,11]、来源平衡[6,11,2]；三处理均低于父。来源处理实际生效，但不构成整体改善；不晋升、不五seed扩展失败配方。GPU与原owner已释放，旧暂停SHA保持。

新namespace `runs/dagger_rl_20261024/r6_opponent_candidate`，协议SHA `33aea6c80abf5f95eb267b4394709753a2b9fd2d593909add5b6b763804df412`。三个配对seed2026102461/2462/2463、control/parent_mix各u32。共同干净r19 g32，policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`，原state SHA `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`，提取父SHA `61829b5a4a94ff1c468cb2a8ab055b6412415262f89ade97a75376dc32315b8e`；保存Adam不重置，不继承R5终点。

唯一变量为每局对手：rule3固定100% vs reset时rule3/冻结共同父策略各50%。冻结父使用P1自己的公开观测、mask和categorical采样，非老师接管、非未来oracle、非JS Hunter。分配用独立fold-in、不消费learner/env随机流，跨step持久、只在reset重抽。初始env/key/params/Adam/候选分配均配对，仅arm启用标记不同；实际current/ring/fixed仍0/0/100%，固定池内部rule3/历史父的tick/live/近中远战斗曝光逐更新统计，不冒充名义50%实际成立。

两臂共同joint PPO+DAgger .2、combat父KL .05、uniform reservoir8192、旧item不分来源、item/route回放4096/4096、新/旧loss各.5。学习率7.5e-5、gamma.995、lambda.95、PPO clip.1、entropy.005、value.5、global clip.5、单epoch/global minibatch256。奖励、banana oracle、胶目标、75%自然出生/25%真实取包Bank课程、输入/网络均不改。所有PPO仅当前on-policy；学生状态标签只进DAgger，parent对手不输出老师标签。禁止人工预携包。携包持续安全返程，自主自由存活偷包合法，但库存PBRS/老师严格窗口门控。

每run64env×128tick×32=262144环境步/1024优化步；六run总1572864/6144。每更新原子保存完整模型/Adam/env/aux/RNG/对手分配/来源history；固定消费u0/1/8/16/17/24/32。技术smoke仅1536执行环境步/6优化步（含fresh重复），正式采用0，查finite、真实Actor/Critic/ability非零、同状态control/learner输出及RNG精确、对手实际动作差异、fresh64下一完整collector→回放→Adam精确，不设能力门禁。31focused GREEN，RED原记录保留。

CPU/JS唯一owner异步：自然dev2026102480/u16/u32各64，共同父+六最终holdout新seed2026110600各64；生产JS Hunter难/中/易各64，Python-host Tactical v2/JAX环境64分开。原Bank/Bank-P仍map806/P0/allfar重复保持，来源局×训练seed CI，不声称新跨图确认。固定首seedReview首偏航与近似reward/GAE、native Bank重建GAE、TRAIN固定标签保持及来源/动作机会拆分；评估不触发GPU训练。

运行入口`.venv/bin/python -m scripts.dagger_rl_r6_run`；CPU `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu NODE_OPTIONS=--require=./scripts/dagger_rl_metrics_v2_preload.js .venv/bin/python scripts/dagger_rl_r6_cpu.py`；终态消费`.venv/bin/python -m scripts.dagger_rl_r6_run complete`。owner串行完成六run不中断、CPU不阻塞；只允许技术故障或用户暂停安全原子停止。每run通知并主动消费，六run后完成声明Bank后释放GPU；完整结果选择新问题，有整体候选才五新seed确认，若继续严重共同退化则明确报告并转匹配状态优化机制诊断，不盲加倍率或延续退化权重。旧O5/continuous/储备永不恢复，不push/发布。

R5完整结论与只读回放计数勘误已写reports/dagger_rl_iteration_20261024.md；不支持晋升。R6唯一对手暴露新实验已dispatch新smoke，尚未把编译或smoke当正式更新。

### R6技术smoke验收与正式初始化（20261024）

真实smoke ActorΔ.127064515074、CriticΔ.069217592685、abilityΔ.002214133041，ability梯度7.569490433。64环境fresh下一完整collector/source-memory/replay/Adam逐digest精确，含对手候选及启用状态；1536执行技术环境步/6优化步、正式采用0。原子state SHA `685dfe30b5aa14caf26fefe6ee5db89d4dafc0d205e5ad112efd7985db242d15`。初始真实同状态control所有输出精确，替换对手保留learner动作/logp/Value/RNG，选中的5敌方行中1行实际动作差异，非能力阈值。正式control2461 u0参数/Adam与干净父精确，当前正式128tick batch首次编译中，不把smoke或显存占用冒充正式更新。CPU独立父点自然dev12/64，Hunter难中易安全胜1/2/6，未阻塞GPU。

### R6首正式更新（20261024）

control_s2026102461/u001实际8192环境步/32优化步，finite、ActorΔ2.537963072188/CriticΔ.665618944302/abilityΔ.091291472764；223条新道具老师行。共同父/保存Adam精确，smoke未继承。独立8个pmap shard（各8env×128tick）的完整state与RNG全部变化。原子state SHA `e09647b1e2506efa5bf261928f2480ae7de4e771f84766291864470b3131beed`，policy SHA `905da149530086a5bb43fd833e3914bd6981eac687953556e068cfdbda2e70aa`。实际采样8卡计算峰[69,71,70,69,79,78,78,77]%；当前u2额外输入签名编译可为0，不报持续高利用率。首control非零是技术有效更新，不代替parent_mix实际处理效果；后者由配对u0、真实对手tick/动作、后续参数分化单列验收。六固定run继续，CPU/JS独立。

R6固定run终态 `control_s2026102461/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `0c276956c07e092c6c2eca9738180f1f2c41c31c65baf98569b8851453869e79`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R6固定run终态 `parent_mix_s2026102461/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `0be3048aee85720a6451dc2212c70049f8d8c2f06eb52604789aa6ee06ec9b27`。实际固定历史父111085/262144tick（42.3756%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R6固定run终态 `control_s2026102462/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `965cc1f4cfc0bf17a2dfb564060e179b2bba74a90f02623602f5320cfa5473cd`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R6固定run终态 `parent_mix_s2026102462/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `798b19d29480c2d8458cdd44c059a20d83028b628ff6dd9d43ba2aed7f8241fd`。实际固定历史父117843/262144tick（44.9535%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

### R7后继机制诊断预声明（20261024，仅准备，未启动GPU）

R6仍完成原六固定run与Bank，不因短点回落中断。若R6完整证据仍无相对未训练父的全链/战斗安全候选，按既有合同先做匹配状态更新机制识别，不再直接扫回放权重/倍率。新namespace `runs/dagger_rl_20261024/r7_update_mechanism`。固定三seed2461/2462/2463 control-u0原始完整状态和同seed u16全部64个TRAIN学生状态作探针，不按结果/标签筛选，不读取dev/holdout调参。

每原始u0精确重放8192当前on-policy行一次，固定同一聚合回放和原保存Adam，比较joint+.05anchor、去DAgger的PPO+.05anchor、去PPO/value/entropy的DAgger+.05anchor（Critic及moments冻结），另共享无更新父及按joint Actor/Critic各module位移范数配平的固定随机参数反事实。三批共24576重复逻辑环境步、288实际诊断优化步、0采用更新/0新独立样本/0额外评估rollout。先要求joint对真实u1参数/Adam、env/aux/RNG及回放逐digest精确重现；否则只报技术故障，不解释反事实。

测量原始分量梯度、真实更新后的同状态phase/距离分层KL、greedy移动/ability变化、固定标签集合质量；原始cosine和局部漂移不等于长期有害干扰，不以logits/reward/value冒充Q/优势。本诊断仅识别下一项最小单变量，不能自称能力已改善；结果模糊则主动报告决策需要，不机械扩训。GPU串行、不恢复旧O5/continuous/储备、不发布。完整R6证据若支持候选，则改走预声明五新配对seed确认，R7保持未执行。

R6固定run终态 `control_s2026102463/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `ec3e4848f8104adaf06beb846f16c10bc5d68e761b2daa3c7922528750c03100`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R6固定run终态 `parent_mix_s2026102463/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `e62d8a5245450c350bb987fff0109d0728e90720f2960c1e2700bd6a27a65c3c`。实际固定历史父121572/262144tick（46.3760%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R7准备补记（20261024）：实际隔离诊断执行器与固定消费器已实现，新增UPDATER_RED→GREEN覆盖继承非零Critic momentum时的参数/mu/nu冻结及Adam时钟推进；13项focused回归通过。SOURCE_SEAL与READY_TECHNICAL已保存，尚无科学选择、GPU执行或采用更新。真实原u1逐digest重现仍是执行中的技术有效性检查，不以toy测试冒充已通过。复现入口 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 JAX_PLATFORMS=cuda .venv/bin/python -m scripts.dagger_rl_r7_diagnostic`；仅完整R6选择及资源释放后执行。

### R6完整结论与R7实际接续（20261024）

R6六run各u32，1572864真实环境步/6144优化步；42声明原子点、完整Adam计数/Actor导出、dev/独立holdout、JS Hunter难中易各64、Python-host Tactical/JAX64、37 Bank dev曲线/7终点holdout及全部固定Bank-P验收通过。CPU/JS owner与GPU owner均终态；接续前8卡0MiB/0%/无compute app。

自然holdout共享父取包26/64、交付23/64、死亡19/64；control三seed交付[4,5,10]、死亡[8,13,17]；parent_mix交付[4,2,3]、死亡[17,8,11]。处理相对control −5.2083pp，来源局×训练seed CI[−13.5417,3.125]pp；相对共享父−31.25pp，CI[−43.75,−18.75]pp。共同基线严重退化，不晋升/不择峰/不五seed扩大这套配方。低死亡同时伴随机会减少，不等于安全能力保持。

窗口机会父108、处理[41,50,73]；窗口库存取包父25、处理[7,6,8]；同原包直接窗口交付父21、处理[2,2,2]，对应84%对[28.57%,33.33%,25%]。首次方向一致率父49/66，处理18/23、19/24、18/23；条件首次方向较好并未补回完整推进，且这些条件分母不同，不是同状态Q因果对照。14条固定game0 dev/holdout失败录像及各自近似reward-GAE/原生Bank128step重建均消费；无首偏航常因没进入相应目标阶段，不能当成功。

Bank最终直接交付处理[95.7031%,96.875%,93.3594%]，相对共享父+9.375pp、来源×seed CI[3.125,15.8854]pp；相对control仅+1.0417pp、CI跨零。Bank-P失败/死亡/丢包均保留。它仍是已用map806/P0/全far的条件保持证据，不是自然跨图/双座位泛化。当前历史父实际tick比例[42.3756%,44.9535%,46.3760%]，current/ring/fixed=0/0/100%；对手动作及后续Actor处理均真实非零，不能再用干预为零解释结果。

证据排序：①训练自然来源仍有67/71/80次交付，来源/label/聚合监督及实际参数处理存在，不能笼统称无信号；②对手单变量没有修复取包前机会下降和取包后交付下降，固定JAX训练分布仍不同于生产JS，尚不能把剩余差异唯一归于分布；③旧输入保持/条件Bank改善不保证自然全链，先做匹配更新机制识别，避免盲扫倍率；④Actor/Critic独立，需区分原始梯度、global clipping/Adam后的实际漂移，不能预设共享主干冲突或低EV=PPO噪声。

完整比较SHA `a7a60320a223661c752df54dd966d147d6d7517915df16b76532362049a0bee5`；综合证据SHA `ccf3a97de7999ae01abfd0649aaf62d305d19f28429ff9c6c7428ef11667282e`。R7条件分支已科学选择，SCIENTIFIC_SELECTION绑定上述完整证据及预注册协议SHA；PREDECESSOR_RELEASE验证原owner退出/8卡无compute app。唯一有界dispatcher已实际启动 `scripts.dagger_rl_r7_diagnostic`，首批GPU重放/编译中，尚未声称有效更新。原u1重现兼技术smoke，不额外增加声明环境或优化预算；fresh完整state字节精确，CPU/GPU probe按明确容差验证，分别报告。旧O5/continuous/储备暂停SHA不变，无发布。

图见 reports/dagger_rl_r6_20261024.png。

R6道具链补记（20261024）：独立holdout持包香蕉库存tick处理[1627,972,1500]、首槽合法tick[286,0,173]、合法释放[8,0,14]；自有接触[4,0,7]、完整自有滑行[0,0,6]，有益[0,0,1]、危险[0,0,5]。共享父该保留集持包库存7966tick/首槽合法5123tick、释放1、无自有接触；不能据处理释放数更多宣称净收益改善。第三seed的1有益/5危险说明适时释放与完整利用仍需区分；第二seed有库存但首槽合法为0，不能把任意slot库存当可释放机会。原始FINAL_COMPARISON各phase/items完整保存。统计按事件当时phase，跨phase释放→接触不是同放置cohort；净距离proxy不是动作Q。

R7技术恢复更正（20261024）：首批原u1 params/Adam/env/aux/RNG/history及来源统计已精确重现，joint真实ActorΔ2.537963072188、CriticΔ.665618944302、abilityΔ.091291472316，32诊断优化步已原子保存，尚无参数采用。初次fresh CPU推理与GPU参考不满足容差；完整state digest精确，同GPU fresh复测两head逐元素exact/maxdiff0，CPU移动/ability logprob最大差.272919/.762028，greedy差1/64与0/64。具体原因是恢复验收错误地要求不同后端推理等价；不能据此称state损坏，也尚未证明底层精度来源或自然退化原因。此前“CPU/GPU容差验收通过”的潜在表述不成立，现明确采用同GPU fresh精确验收，跨后端差异单列、不放宽阈值。

失败源码/seal/log/退出与故障收据完整保留technical_history/cross_backend_restore；恢复幂等加载COMMITTED的SHA/params/Adam，拒绝损坏/非原u1，新增RED→GREEN且14focused通过。已完成joint不重跑。首批batch未持久保存，需额外8192同源技术重生环境步，0额外优化/独立样本；单列TECHNICAL_REPAIR，科学预算仍24576逻辑步/288诊断优化，实际执行环境步将32768。新batch/回放逐SHA落盘，恢复后继续未完成分支，保持旧暂停不变。

### R7固定匹配更新机制诊断（20261024）

原R6 control三seed的真实u1参数、Adam、完整env/aux/RNG及source-memory/replay精确重现；每seed8192行只重放一次。
以下为相同TRAIN输入上的静态变化，不是能力、Q或长期干扰因果结论。DAgger-only冻结Critic及moments；共享Adam时钟推进。

| 诊断分支 | Actor L2（三批） | Critic L2 | combat ability KL（三组TRAIN状态） |
|---|---|---|---|
| joint_anchor | [2.537963, 2.276907, 2.447535] | [0.665619, 0.615578, 0.757506] | [0.001929, 0.002039, 0.002779] |
| ppo_anchor | [0.710347, 0.832657, 0.896611] | [1.354361, 0.94311, 1.378376] | [0.000231, 0.000876, 0.000782] |
| dagger_anchor | [2.833934, 2.665311, 2.896903] | [0.0, 0.0, 0.0] | [0.003005, 0.011722, 0.008767] |
| no_update | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] |
| module_norm_matched_random | [2.537963, 2.276907, 2.447535] | [0.665619, 0.615578, 0.757506] | [7.5e-05, 0.000101, 0.000148] |

完整分量梯度/原始cosine、phase/距离/来源/标签分母、greedy/集合质量、固定随机位移与fresh恢复收据见 `runs/dagger_rl_20261024/r7_update_mechanism/FINAL_MECHANISM_EVIDENCE.json`（SHA `35a1f27e434fab185ac6b92d8a993dbada60a1b4cdeecfd5313a9c5e2aa70f52`）。

声明24576重复逻辑环境步，另有8192技术恢复重生步，实际执行环境步32768；288诊断优化步、0新独立样本、0采用更新、0额外评估rollout。首批已完成joint按SHA复用，没有重复优化。同GPU fresh输出精确，CPU/GPU输出差异单列为后端测量，不以放宽容差掩盖。未以holdout选量级；不从低EV推断PPO噪声、不从静态漂移证明长期有害干扰。旧O5/continuous/储备暂停不变。

### DAGGER-RL-20261024-R8-DAGGER-SCALE 冻结合同（20261024）

R7完整终态：三原始TRAIN batch，288诊断优化步、0采用更新/新独立样本；另8192技术重生步，实际重放环境步32768。三批真实joint-u1 params/Adam/env/aux/RNG/history精确，同GPU fresh全state及动作输出精确。主证据SHA `35a1f27e434fab185ac6b92d8a993dbada60a1b4cdeecfd5313a9c5e2aa70f52`，实际参数方向/Adam对照SHA `d2d068cb629c842e268071ce9475ee8ea1d3c1768617e4a2785fa8512e5853e7`。联合Actor位移与DAgger-only方向cos[.9661,.9326,.9434]，与PPO-only[.0453,.1831,.1509]；weighted DAgger Actor首minibatch梯度[11.3183,10.5127,11.0148]，PPO[2.3800,1.5838,1.9407]。DAgger-only Critic及moments精确冻结但combat静态分布仍改变，不能称Actor/Critic共享主干冲突；原始cosine/实际漂移不等于已证明长期有害干扰。CPU/GPU bf16网络后端差异单列，生产真JS比较未被替换。

R8新namespace `runs/dagger_rl_20261024/r8_dagger_scale`。只检验DAgger总系数.2→.05，其余PPO/anchor(.05)/学习率(7.5e-5)/保存Adam/奖励/对手/课程/网络/道具老师/记忆及比例完全相同。两臂control/sup_quarter，三新配对seed2026102481/2482/2483，各固定u32，六run1572864真实环境步/6144优化步。共同r19干净父与保存Adam，不继承退化终点；共同父policy SHA a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95，parent输入SHA61829b5a4a94ff1c468cb2a8ab055b6412415262f89ade97a75376dc32315b8e。

四分之一量级只看冻结TRAIN梯度/真实位移而定，不扫描holdout；可能欠拟合是可证伪风险，不预称最优。global clipping及Adam使实际Actor/Critic贡献非简单按四分之一缩放，必须报告真实分量/参数更新。自由存活自主偷包合法，库存老师/PBRS仍限原生窗口；真实携包Bank、不人工预携包。胶和香蕉目标均不新增。

固定点0/1/8/16/17/24/32每次原子保存；唯一CPU/JS跟随，dev seed2026102480，中点与终点各64；新holdout2026110800起64局，共同父及六终点各一次，JS Hunter难中易各64、Python-host Tactical/JAX64分开，既有单图P0/allfar Bank/Bank-P保持及固定失败链/真实训练GAE分列。声明唯一CPU owner及完成marker，不因短点评估停健康run。终态六run后固定Bank，完整证据支持才做五新配对seed确认；若仍退化，不等价扩训，优先检查实际更新与部署后端差异，并在无合理单变量时明确报告决策需要。旧O5/continuous/储备永不恢复，不push/发布。

R8预注册协议SHA `b203e5f623050fe4a0a6a6ce31769183ef7d490740490ce200ba3a813f6dd812`；此时尚未启动GPU，仅进入TDD/实现。

R8启动前冻结更正（20261024）：两臂对手描述规范化为同一字符串；训练seed改为2026102471/2472/2473，以避免初稿2481/2482/2483与固定dev整数seed区间重叠。未执行过初稿seed，未看新holdout，也未改变实验变量/预算。当前权威PROTOCOL SHA `c8120ebe89abd513b2ef2b5013878306eec9f134e6ba436c0843f814fa565b15`，旧b203e5f6版本及contract失败日志保留technical_history/contract_metadata；替代上方初稿SHA/seed。18focused关键断言现全部GREEN（原并行批17项通过+唯一描述字符串断言修复，contract两项复验通过），源码/输入封存，开始新smoke后直接六固定run。

JS诊断字段语义勘误（20261024）：`scripts/dagger_rl_route.js`的`approximate_reward_gae`严格为 z_t=(control_entry+route_observable_reconstruction)_t+.995*.95*z_(t+1)，没有Critic Value/TD residual。因此它只是零Value基线的折扣塑形和，不能称训练GAE、近似优势或据它解释优势变化；撤回此前报告中把这些具体JS数值当优势变化的任何表述。原始trace/Review/状态动作/事件能力数据完整保留，新增JS_DIAGNOSTIC_FIELD_ERRATUM记录SHA与实际公式，0环境/优化步。R8不改已封存动作评估脚本，消费层将其明确重命名discounted_shaping_sum_zero_value；在补足真实Critic残差前禁止用它作优势证据。原生Bank采用各政策独立Critic及真实奖励、128step边界重建的GAE，训练collector实际GAE/optimizer，均是另外两条已验证路径，不受本命名勘误影响。R7/R8科学选择基于TRAIN真实梯度/Adam与事件漏斗，没有依据该JS字段。

数值静态审计（20261024）：同三个R7固定TRAIN64帧（非新能力对局）GPU bf16与实际导出ONNX fp32移动greedy差[1,1,3]/64，ability均0；平均joint KL约[.0001822,.0001598,.0002495]。Torch fp32与ONNX greedy均exact，logprob最大差不超过9.73e-5。训练Transformer的token/block显式bf16，生产参考fp32是具体代码事实，但尚不能将其等同自然退化原因。完整ONNX/输入/权重hash与原始静态输出见DEPLOYMENT_NUMERIC_AUDIT，0新增环境/优化步；不改变R8配方，不作为能力门禁。

R8技术smoke与异步消费核验（20261024）：真实两臂同批512行，各2优化步，control ActorΔ.127344694653/CriticΔ.069213194917/abilityΔ.002210670616；sup_quarter ActorΔ.124213931455/CriticΔ.070997905123/abilityΔ.002144568600。ability梯度分别7.549710274/2.046652317。fresh同GPU全部64环境的下一完整collector/source-memory/replay/Adam恢复精确；总1536执行技术环境步/8优化步，正式采用0。state SHA `df678253ee9a3ea9f1cc0d6251eb5c5c4b97f62d820d542ebedc0324732b7438`。正式worker已从共同父与保存Adam独立初始化，首批编译中，未把smoke/显存占用计为正式更新。唯一CPU/JS已完成共同父dev/Hunter/Tactical，独立于GPU。

只读诊断owner发现并修复旧R6字段 `execute_opponent_test` 引用，改为核验R8 `execute_supervision_scale` 与协议SHA；故障发生在子消费者启动前，0环境/优化步，无训练中断或封存源码改动。原失败log保留，`CPU_DIAGNOSTIC_OWNER_REPAIR.json`记录范围；唯一锁下恢复四个只读消费者。旧全局暂停SHA不变。

R8首批技术故障与原子恢复（20261024）：正式首批已完成8192环境步/32优化步及finite检查，但因遗漏`import os`，在updates日志fsync处NameError退出，u1未COMMITTED，最近完整原子点仍u0。8卡确认释放；原源码/seal、完整失败log/退出收据、孤立u1指标保留`technical_history/formal_missing_os`。修复仅补该导入，删除新增import后的完整AST与旧版逐项一致；真实运行时os.fsync绑定通过，9项focused回归GREEN。SOURCE_SEAL保留历史并只更新该单一文件hash，实验PROTOCOL/唯一变量不变。原owner已从u0恢复，不重跑smoke；唯一CPU/JS与四个只读诊断未重复启动。额外已执行8192环境步/32优化步、0新独立样本、不采用未提交更新，正式仍六run1572864环境步/6144优化步。后续用保留孤立指标与重放u1对照；没有旧u1参数文件，不能声称旧参数字节精确复现。

R8首正式有效更新及恢复验收（20261024）：`control_s2026102471/u001`已原子保存，8192正式环境步/32优化步；finite，ActorΔ2.545267967923、CriticΔ.616459634943、abilityΔ.086309879658，ability原始梯度范数2.509868145。父params/保存Adam精确，smoke未继承。state SHA `b26420e5ba5d34fb765ced71f793199a2381f52eb1aed713879a2fd76a8c1c90`，policy SHA `12688fcc6397be81cbaaf44d28141e3a40a97a9088c0723a8b820fe00bf3cdf2`。8卡实际采样峰[77,77,77,80,68,69,68,67]%，首更新含恢复后加载/编译的吞吐99.17步/s，不称稳态吞吐。保留孤立u1与恢复u1除两个walltime字段外全部科学指标精确一致，`TECHNICAL_REPLAY_ACCEPTANCE.json`记录；旧u1无参数文件，不伪称旧参数字节比对。六run继续固定预算，CPU/JS独立，无能力门禁。

R8固定run终态 `control_s2026102471/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `b4e67b152b3e412f50fb7029b04fab676e7a5bdfa2a5ead34b2ae45ff53f3460`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R8首配对系数干预验实（20261024）：s2026102471两臂u0完整params/Adam/env/aux/RNG/history精确相同，首批collect及老师记忆精确；.2/.05首更新后Actor参数差L2=.876268657420、Critic=.603453356098、ability=.018364043868。处理自身ActorΔ2.207258974688、CriticΔ.842590315152、abilityΔ.073432542384、ability原始梯度.811270356；对照ActorΔ2.545267967923、CriticΔ.616459634943。系数减四倍不等于global clip/Adam后实际位移减四倍，不能据此称处理为零，也不能称独立Actor/Critic共享主干。首批同状态因果范围在`supervision_effect_s2026102471.json`，之后学生状态可分化。此时control最终dev交付5/64、死亡16/64，未训练父12/64、死亡22/64；仅单seed开发结果，不作终态/安全改善或停训依据。

R8固定run终态 `sup_quarter_s2026102471/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `48bdd58bb0d9cb2d2616e0b9a8acd110925dcfac8733d2866dc5f5129d0a9336`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R8固定run终态 `control_s2026102472/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `cea480c050fbaf94ae8b9d8b4fba33bd0888859970eb08f19922499ed3fc583a`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R9条件后继预注册与CPU准备（20261024，尚未启动GPU）：`DAGGER-RL-20261024-R9-STEP-SCALE`，namespace `runs/dagger_rl_20261024/r9_step_scale`，协议SHA `46f0132536328f22cbf8eb31a1116ed71394d54676fa5380bb74b0d675525520`。科学问题限于减小实际Adam步长是否缓解完整自然链的共同退化。共同保持原DAgger .2（不把R8 .05当胜出配方）、joint PPO/combat父KL .05、相同r19干净父/保存Adam和全部奖励/老师/课程/对手/网络/记忆抽样。唯一变量lr7.5e-5→1.875e-5；配对seed2026102571/2572/2573，control/lr_quarter各u32，共1572864环境步/6144优化步；同0/1/8/16/17/24/32保存与dev/Hunter/Tactical/旧Bank协议，新自然holdout2026110900起64局。量级只依据R7/R8首批TRAIN实际更新证据固定，不扫描holdout，欠拟合风险明示。

先写RED（实际arm模块缺失）后GREEN：3项focused覆盖真实arm优化器入口、已有非零Adam时钟/moments保留、同梯度下仅delta按.25缩放、collector/aggregate/objective复用与配对预算。只做CPU物化/封存，无R9 GPU任务，缺少SCIENTIFIC_SELECTION时authorize拒绝启动。只有R8六run/Bank/CPU/全部诊断完整后且仍无可信相对未训练父的候选，才以完整证据新写选择并技术smoke→固定训练；若R8整体可信，则优先五新配对seed确认，本候选不执行。R9若失败不得串行盲扫lr；检查实际phase PPO/训练-部署路径，无法形成合理最小变量则主动报告科学决策点。R8活动源码/运行语义未改；旧暂停不变。

R8固定run终态 `sup_quarter_s2026102472/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `7b91d83185d2a4a397be2b656a705b8da3dc80004834d5cb51f4e7645d5b64ac`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R8固定run终态 `control_s2026102473/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `04d3e4f9a787bf17c363775c1f37b1a4791138714d5340bce57f5a98f16a37b3`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

训练接触遥测口径勘误（20261024）：R1–R8继承的`events.self_contact`及source同名字段实际为`info["trap_hit"][:,0]`，原生实现表示learner触发任意香蕉/胶field，并未验证香蕉类型、自有owner、持包、完整滑行或正净收益。因此只能称`learner_any_item_contact`，不得用它证明自放自吃/完整利用。原始日志/源码/hash保留，R8消费层明确改名，奖励/GAE/老师/优化器及生产JS v2 owned归因不受影响；独立真实因果微测试与JS owned净收益仍是另外的证据。`TRAIN_CONTACT_FIELD_ERRATUM.json`保存具体源码SHA及语义，0新增环境/优化步。后续R9只修公共遥测名称，不改变训练数学。

R8固定run终态 `sup_quarter_s2026102473/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `a837c9c7239d68e83c303584acd81021d0589e9f5437db1241637656423c4439`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R8三组匹配TRAIN首更新实际步长（20261024）：监督系数.2→.05后，Actor整体位移相对control仍为[.867201,.852109,.812126]，ability位移[.850801,.826523,.784908]，独立Critic位移[1.366822,1.632608,1.463971]。完整同批/同Adam证明与每点COMMITTED SHA绑定在`FIRST_UPDATE_STEP_RATIOS.json`；32个mini-update内非线性演化/global clip/Adam下不能按原始系数等比解释位移，更不能从此直接证明长期退化由步长造成。数据全部TRAIN，无新环境/优化步或holdout选参；作为R9条件单变量问题的依据，不是能力晋升证据。

滑行安全用词再限定（20261024）：`owned_slide_harm`实际是“未全程存活/自由/携带同敌包，或扣步行成本后距离净收益≤0”的余类，合法在滑行中交付也可能进入该类。它不等于实际危险/自伤/死亡；前述R6把5个余类结算直接称“危险”的用词撤回，原计数不改。能力/安全选择只看单列死亡、受控、掉包和交付，不用这个proxy定安全结论。R8 holdout的持包owned完成均0，尚无这类结算可解释；自放自吃链未形成与安全余类含义是两件事。

R8全证据终态与R9实际接续（20261024）：六u32/42原子点、1572864环境步/6144优化步和全部固定CPU/JS/Bank/Bank-P/诊断验收。父holdout14/64，control[3,5,4]，监督.05[9,13,7]；处理对control+8.8542pp/CI[0,18.2292]pp，对父−6.7708pp/CI[−18.75,5.7292]pp。仅部分恢复退化对照，不晋升/不五seed扩展。综合SHA daf43d01f2239714cfea89205a416db59517787a75ff88ff927bf066c2c66692；比较SHA5bb460763112ac0cd02b48777d60a455eff57baccb247584fcf08a054f9c0a95。细分机会/安全/标签/道具/实际GAE见当日报告R8完整结论。R9按此前冻结的单变量lr .25科学选择，原DAgger.2共同基线/干净父/保存Adam，持久dispatcher核验8卡0MiB后实际接续；唯一owner当前新技术smoke。初次缺技术就绪收据在GPU前退出，补真实测试/SHA收据后修复，0重复训练步；不重复旧smoke，不恢复旧队列。

R8原生评估步数补记（20261024）：72个固定结果集，共18432条相关随机尾迹，全部终局（censored=0），实际valid环境步5335676、0优化步。它们重复同32来源/不同政策，不是18432独立来源；不计正式训练步。原日志未完整保存设备padding/终局后无效执行的精确物理步数，仅可声明上界52428800，不得伪造精确值。逐结果hash与计数见EVALUATION_STEP_ACCOUNTING。

R9首正式非零更新接管复核（20261024）：`control_s2026102571/u001`实际8192环境步/32优化步，finite，Actor参数L2 2.547090837339、Critic .757139519468、ability .082682982447；父params/保存Adam精确，smoke未继承。state SHA `31432223ee749e0ae5738a3bbeccf0464e89a03621aa8c6abca95feb3772e043`，policy SHA `b65bb0f40aa407a84f35b5bd85b8c0fe6ef8d6d8d282b648a9b6baceeb3dc421` 已重新计算。现场8卡67–79%实际计算，唯一GPU owner/worker及CPU/JS owner均在运行，当前u4；旧全局暂停SHA不变。两臂唯一学习率变量7.5e-5/1.875e-5，共同DAgger .2；按三配对seed各u32继续，固定CPU评估异步，不以短点评估停训。

R9异步只读补证（20261024）：冻结`phase_train_replay/PROTOCOL.json` SHA `652da94a8986cb82822bf0dcff4b2c217f7160e9e3a40aa5a617ccb8405f70fe`，仅消费R7三原始TRAIN batch（3×8192行）及已经执行的joint/PPO/DAgger/no-update分支。测量存储的真实collector GAE、按冻结公式重建的归一化GAE、同CPU后端执行动作logprob变化、老师集合质量与来源/阶段分母；0新增环境步/优化步/采用更新，不改R9封存配方，不访问dev/holdout。CPU与GPU原logprob差另列；拟合批GAE×Δlogprob不是Q、独立能力或长期有害干扰的证明。新增分母/相位互斥/空集统计3项TDD已RED→GREEN。

R9固定run终态 `control_s2026102571/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `72752823213162d5b359d9d6785ea71e7c2d12a1037ef634d3ea18a9f43ebb09`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

### R9异步TRAIN分阶段实际更新补证（20261024）

全部数据来自R7已冻结的三批原始TRAIN学生轨迹，3×8192行、0新增环境步/优化步/采用更新。既存真实collector GAE与各分支已执行32个mini-update后的同CPU后端logprob变化配对；不是Q或独立能力评估。

| TRAIN seed | combat/carry/窗口/非活动行 | 释放/接触老师行 | joint combat A×Δlogp | PPO-only combat A×Δlogp | DAgger-only combat A×Δlogp |
|---|---|---|---|---|---|
| 2026102461 | 6016/2070/0/106 | 223/0 | 0.002575 | 0.002275 | -0.000881 |
| 2026102462 | 6144/2048/0/0 | 166/0 | 0.001650 | 0.004994 | -0.000565 |
| 2026102463 | 5863/2293/5/31 | 146/0 | 0.005809 | 0.007268 | -0.002604 |

DAgger-only在无直接老师标签的combat行产生负的拟合批GAE×Δlogp均值，而joint与PPO-only在这三批combat均为正；不能由此将joint长期退化简单归为PPO噪声或证明DAgger长期有害。joint carry对应均值为[1.111656,1.769763,-.100389]，PPO-only为[.279268,.217010,.126236]；差异是后续可证伪问题的线索，不是能力结果。库存窗口0/0/5行及接触标签全0说明首批未覆盖完整机会→释放→自吃链；不能用高释放拟合代替整链学习。

归一化优势依据原全局mean/variance公式重建，原始GAE直接存储；CPU bf16与历史GPU采样logprob的绝对差p99为[.057510,.056857,.058810]、max为[.458413,.521535,.510650]，因此所有更新方向比较采用同CPU前后，未声称与历史GPU逐bit一致。所有暴露分母/来源/零更新control/老师集合变化及原始hash见`runs/dagger_rl_20261024/r9_step_scale/phase_train_replay/COMPLETE.json`。本补证只读，不改变活动R9配方或唯一变量。

R9固定run终态 `lr_quarter_s2026102571/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `b1a9b497580cd4eba0fa6723eede176e64359b358969e74e10fe0991912c0cf9`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R9固定run终态 `control_s2026102572/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `d23d8d2a964b2cf9bfe7cc6c8a5b12485c7c26ea1e29c49ed038eaa2298a94a1`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R9生产实际轨迹数值补证预注册（20261024）：`trajectory_precision/PROTOCOL.json` SHA `60c44d6d0d41650e8115a04de2bf9c8027ce19be643992202f6b696aa0c81d6d`。只取共同父与六个固定终点既有dev首局（seed2026102480/P0），每16个pre-action tick及固定首偏航±8、最后一帧的并集；逐步重放原动作，逐帧state SHA、原生产greedy动作和终态必须精确。上限16800重执行物理步，0新独立样本/优化步/采用更新；不能把重放称0物理执行步。CPU只读帧消费者独立于唯一正式CPU/JS评估，不重复模型对局。GPU仅在R9固定训练及Bank全部结束、原owner退出和无compute app后执行同帧CUDA bf16推理，比较原生产fp32 ONNX，不阻塞正式训练或抢占其他任务；这是静态沿轨迹输出比较，不是反事实闭环或JAX/JS物理parity。选择/完整二进制行/掩码位置/错误形状RED→GREEN测试通过。已有前三模型重放全部state及动作exact。新命名空间不恢复旧O5/continuous/储备。

R9固定run终态 `lr_quarter_s2026102572/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `5108c2fcad52c28f8a0942c26a9f342ecec0aed924ab1ec7c9a386f172b4e0b4`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R9固定run终态 `control_s2026102573/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `91ae0c266838ff87c76f466a5f2502ebaaea35d3b60dc4a6ca3dc4ff3bb7e2d8`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R9固定run终态 `lr_quarter_s2026102573/u032`：262144真实环境步/1024优化步，32次finite、Actor/Critic非零；原子state SHA `2af5ec8b2f89a60ed8fe9b978985ead466ccb323831d6f6ddb400af139738aa0`。实际固定历史父0/262144tick（0.0000%），余为JAX rule3；current/ring/fixed=0/0/100%。唯一owner按冻结六run队列接续，CPU/JS独立；该技术终态不冒充能力改善或GPU已释放。

R9路线ability只读审计（20261024）：固定18×64=1152个真实TRAIN learner快照，未修改训练。自然空手原生窗口76行/香蕉库存17行，其中2行路线集合接受未获item oracle认证的banana释放；Bank空手窗口4行/库存1行，其中1行同类接受。自然自由存活空手851行/香蕉323行，没有库存路线标签误激活；真实持包自然80行/香蕉4行中2行、Bank127行/香蕉28行中22行也有未认证释放的路线接受。分母仅固定快照，不是全rollout机会或独立来源局。三个空手窗口例子的mask均为同一移动×全部3种合法ability，数学上set loss只监督移动，ability梯度为0，约1e-8浮点残差不能称正监督；原negative_rows阈值1e-8字段已在`NUMERICAL_INTERPRETATION.json`标明数值口径，原始结果保留。具体结论是路线老师未约束这类释放，不是它主动奖励/监督浪费，也未证明每次释放均无益。自然dev窗口中已观测香蕉释放、owned接触为0，因此可考虑后继单变量“只有验证有益计划才将banana释放列入路线老师联合集合”，但须与共同父/control比较，不能作为既定根因或改活动R9。

R9完整终态（20261024）：6×u32、1572864正式环境步/6144优化步，42点/192更新通过；独立holdout父10/64、control[8,12,12]、lr_quarter[7,6,8]，低LR对control−5.7292pp、对父−4.6875pp，3seed均下降，不晋升/不五seed确认/不盲扫倍率。完整漏斗/Hunter/Tactical/保持/曝光/首偏航/真实GAE及1037生产同帧数值补证见`reports/dagger_rl_iteration_20261024.md`当日完整结论与原始hash；8卡释放、所有原owner退出已现场核验。仅有界CPU oracle composition微测试接续（最多6656物理步，0优化）；已完成固定原计划26/26有益滑行，保存类型故障复用trace修复，待重新规划分支后据机制选择最小训练干预。旧O5/continuous/储备暂停不变。

## DAGGER-RL-20261024-R10-RELEASE-QUALIFICATION — 预注册（20261024）

R9完整终态低LR三个最终点自然交付均低于配对control/父，不晋升、不选五seed、不盲扫。26个既有真实TRAIN机会：固定计划与逐步item→route老师均26/26完成有益滑行、死亡/丢包/截断0，合计6656重执行物理步、606有效步、0新独立样本/优化步。老师序列在该enemy-wait微场景可组合，不代表Hunter能力。

唯一变量：路线监督中的香蕉释放资格。control保留原set；qualified_route仅在route中删除没有现有item有益计划认证的banana ability2，item release/contact整集合保留，删空则abstain。部署合法mask/PPO自主行为/胶/奖励/课程/网络/对手不变。未认证不是必然无益，需实测库存保持和整链代价。

共同父r19 g32（policy a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95；state2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5），原保存Adam，lr7.5e-5、DAgger.2、combat KL.05。seed2026102581/82/83，各臂固定32更新，64env×128步/更新，262144环境步/1024优化步每run，合计1572864/6144。新run/state/cache/eval命名空间`runs/dagger_rl_20261024/r10_release_qualification`，不继承R9端点。

所有当前学生on-policy PPO；75/25 reset自然/真实因果Bank，实际暴露另报；reservoir8192和item/route4096/4096、old/new.5/.5不变。训练current/ring/fixed=0/0/100%，fixed是JAX rule3，历史父0%；不混Python或JS bot。旧Bank仍map806/P0/allfar，不能称跨图独立确认。

保存每更新，声明消费u0/1/8/16/17/24/32；自然dev64中点/最终、自然holdout2026111000起64（新独立保留集，仅固定最终）、真实JS Hunter难中易各64、Python Tactical64、Bank/Bank-P32×8，父与六final同协议。记录自然控制→库存→原包交付、所有阶段库存/机会/概率/释放/owned接触/完整净滑行/死亡、训练老师改变行/源与相位、固定首偏航/奖励和真实GAE或明确离线近似，不以塑形回报替能力。

先RED→GREEN/实际TRAIN标签及ability梯度/零开关等价，再最小技术smoke1536执行环境步/6优化步、fresh64完整env/aux/RNG/Adam/memory/replay恢复；不设能力门禁、不采用smoke参数。正式六run由唯一持久owner串行接续，CPU/JS独立跟随，不等报告。每终态主动通知并hash验收；故障安全原子恢复；用户停止优先。完整证据若支持才冻结五新seed确认，否则依据首次损失/机会/序列证据选最小后继，证据不足明确科学决策点，不用无关训练占卡、不扩无界预算。旧暂停SHA保持，不恢复O5/continuous/储备，不push不发布。

冻结协议SHA `9c3761c5caf76bdfac7a362fc16bd8d6aef9dbd0e33254d5bdb5819bf7d7a881`；完整原始配置见该namespace的PROTOCOL/SCIENTIFIC_SELECTION。

R10机制检查补记（20261024）：固定R9三seed×两臂×u1/16/32，共1152个真实TRAIN快照，control旧集合逐项精确；处理改变27行（自然carry2、自然空手原生窗口2、Bank carry22、Bank空手窗口1），0删空。全部item集合、非香蕉行、ability0/1列精确。27行新的释放logit局部set-loss梯度均为正，原梯度约0；这是实际标签干预证据，不是全optimizer或能力结果，不改变物理mask/PPO已执行动作。新focused collector测试证明PPO动作（含ability2）/reward/GAE/合法mask保持，只有teacher和被动计数改变。行为RED三失败→GREEN五通过；完整相关回归进行中。

R9老师序列微测试已终态：两种执行策略均26/26完整有益、owned接触26、死亡/丢包/截断0，各303有效物理步；总6656执行物理步/606有效步，0新独立样本/优化/采用更新。无重跑已保存首臂。重新规划成功仅支持该26状态enemy-wait微场景序列可执行，不证明对手干扰下安全或学生掌握。

R10已执行启动收据（20261024）：59项相关Python回归通过；两组JS指标/二进制轨迹选择测试通过，行为RED与GREEN日志、真实1152行标签审计hash均已冻结。唯一持久dispatcher已创建独立GPU owner、CPU/JS owner、终态复核与只读诊断消费者；启动前8卡0MiB/无compute app、R9 owner退出，随后进入新smoke编译（不冒称正式更新）。六个固定run在同owner内串行接续，CPU报告不会阻塞；完整证据后写SCIENTIFIC_DECISION_REQUIRED并主动通知，不隐式恢复旧队列。

复现入口（先确认无同namespace活跃owner；不要重复执行已完成stage）：
```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_dagger_rl_release_qualification.py tests/test_dagger_rl_r10_contract.py
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.dagger_rl_r10_prepare
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python .hermes/dagger_rl_r10_dispatch.py
```

prepare执行前冻结协议及选择收据；dispatcher拒绝重复DISPATCH_STARTED，训练owner持有独占lock、逐代读取scoped授权/USER_STOP、每代原子保存。用户停止仅当前安全边界，旧全局暂停SHA不变。恢复已有正式点应核验退出状态后直接启动唯一`-m scripts.dagger_rl_r10_run`，完整checkpoint包含env/aux/RNG/Adam/history；不能重复dispatcher或重置完成run。运行产物和编排文件不入Git。

R10正式首更新独立验收（20261024）：`control_s2026102581/u001`，8192环境步/32优化步，finite，Actor/Critic参数L2 2.337445615550/0.600267450177、ability head 0.091488689937。state SHA `564851438fcc4916cebee8c8fde9ff886eb68c944a98eaf7199d824408f2546d`，policy SHA `a3ace738a4c75b3309fc2d45068e487608612751d2e4ea8794ece9cd610e75da`已重算；共同父params/Adam精确，不继承smoke。新技术smoke执行1536环境步/6优化步、fresh64完整collector/env/aux/RNG/Adam/history/replay下一更新hash全精确。CPU/JS父点固定评估已完成，训练未等评估阈值。首提交前GPU被动采样主要覆盖编译，最近60秒每卡峰值仅0–2%，不据参数非零伪称高利用率；持续检查后续真实吞吐/采样。共同lr7.5e-5、DAgger.2、anchor.05按六固定run执行；标签实际干预须等待qualified_route臂，control候选563行/删除0。

参数级TRAIN补证：固定原R9 control首seed u1的64真实状态，.2原始set-loss处理差分ability head梯度L2=2.9469258785、Actor=3.4044399261、Critic直接梯度=0；不等同完整loss family归一化/global clip/Adam后效应，不作Q或能力结果。原始`REAL_TRAIN_HEAD_GRADIENT.json`保留。

R10固定run终态 `control_s2026102581/u032`：262144真实环境步/1024优化步，32次finite/Actor/Critic真实非零，原子state SHA `8344b11d3a607a72ba3310c1c5b9eac79a441ca4aaf5c437eff944045808486d`；标签候选/实际删除/删空=3890/0/0。fixed JAX rule3暴露262144/262144tick，current/ring/fixed=0/0/100%；历史父0%。唯一owner按冻结六run队列接续，CPU/JS独立；终态是技术预算完成，不是能力改善或GPU已释放。

R10首配对实际干预验收：首批8192真实学生步，完整末状态/aux/RNG及事件/奖励分解/GAE均值方差/来源/对手逐项exact；聚合history观察/合法mask/item标识精确，仅目标集合改变。处理实际删除563行、删空0，item老师70行保留。两臂优化后Actor/Critic/ability参数L2差为{'actor': 0.6576569616774186, 'critic': 0.29931146846110734, 'ability_head': 0.027039574945925966}，非等价臂。处理state SHA `142bf15ac025917dbc3dd54f79948fd03d58e6f06600a2a3e96998d357af0537`，finite、Actor/Critic非零。Actor/Critic独立，Critic差可由共同global clip/Adam间接产生，不称共享主干或长期伤害因果证明。FIRST_PAIR_EXECUTED_INTERVENTION保存全部hash/原始统计。

R10固定run终态 `qualified_route_s2026102581/u032`：262144真实环境步/1024优化步，32次finite/Actor/Critic真实非零，原子state SHA `c796d6865282afc9b13f060c685351db7deb1a8757dcf5c587c7cf7681962598`；标签候选/实际删除/删空=4549/4549/0。fixed JAX rule3暴露262144/262144tick，current/ring/fixed=0/0/100%；历史父0%。唯一owner按冻结六run队列接续，CPU/JS独立；终态是技术预算完成，不是能力改善或GPU已释放。

R10固定run终态 `control_s2026102582/u032`：262144真实环境步/1024优化步，32次finite/Actor/Critic真实非零，原子state SHA `889e89822f783d58d258721c1c9a4e033cc898ef3ff2bd834d933bd4281e7b13`；标签候选/实际删除/删空=2787/0/0。fixed JAX rule3暴露262144/262144tick，current/ring/fixed=0/0/100%；历史父0%。唯一owner按冻结六run队列接续，CPU/JS独立；终态是技术预算完成，不是能力改善或GPU已释放。

R10固定run终态 `qualified_route_s2026102582/u032`：262144真实环境步/1024优化步，32次finite/Actor/Critic真实非零，原子state SHA `3ec2b268a3222cc6ebb98d14fb025a48f9d1d7f12b48a4eebf417866d4baebf1`；标签候选/实际删除/删空=5111/5111/0。fixed JAX rule3暴露262144/262144tick，current/ring/fixed=0/0/100%；历史父0%。唯一owner按冻结六run队列接续，CPU/JS独立；终态是技术预算完成，不是能力改善或GPU已释放。

R10条件后继问题补充冻结（不改活动PROTOCOL）：完整三seed/父/尾部/战斗安全证据若支持，下一项为相同配方五新配对seed2026102591–95、每臂固定u32、2621440环境步/10240优化步、独立自然holdout2026111100起64确认；不按单seed或峰值选。若未形成可信整链保持，下一项先做有界生产真实学生持包香蕉动作链机制重放，取父及首配对seed固定dev、每模型最多4释放来源局+4库存来源局、共≤24状态、按局/tick顺序不按结果选；完整JS状态/RNG恢复parity后做commit/replan/student同状态128步分支（common enemy-wait、实际恢复计时），owned/完整滑行/步行成本/死亡丢包/截断/首偏离分列，不称Q或自然能力。最多494592重执行物理步、0新独立来源/优化/采用更新。仅在当前唯一CPU/JS正式owner退出后由唯一机制owner执行，避免重复评估owner；不恢复旧GPU队列。完整定义与SHA见NEXT_SCIENTIFIC_QUESTIONS，活动六run继续固定预算；证据不足则及时报告科学决策点，不盲扫或无关占卡。

R10固定run终态 `control_s2026102583/u032`：262144真实环境步/1024优化步，32次finite/Actor/Critic真实非零，原子state SHA `f28db7c74608426b817bc334da8462570ad5bd1ce66e18b5dd02c4f028cce813`；标签候选/实际删除/删空=1982/0/0。fixed JAX rule3暴露262144/262144tick，current/ring/fixed=0/0/100%；历史父0%。唯一owner按冻结六run队列接续，CPU/JS独立；终态是技术预算完成，不是能力改善或GPU已释放。

R10固定run终态 `qualified_route_s2026102583/u032`：262144真实环境步/1024优化步，32次finite/Actor/Critic真实非零，原子state SHA `ac4df68ed91ecad48433be9781568728350dbe0041573eab577ef5dfcc85d690`；标签候选/实际删除/删空=4714/4714/0。fixed JAX rule3暴露262144/262144tick，current/ring/fixed=0/0/100%；历史父0%。唯一owner按冻结六run队列接续，CPU/JS独立；终态是技术预算完成，不是能力改善或GPU已释放。

**R10被动归因勘误/结论暂停（20261024）**：现场进程environ确认CPU启动遗漏声明的v2 NODE_OPTIONS preload，实际自然评估使用旧fallback-cell归因。此前R10 owned接触/完整自有滑行/有益自有滑行全0的解释暂撤回，待同种子v2被动修复；既有自然交付/死亡/动作/状态、训练奖励/DAgger/参数真实干预不受该只读计数错误影响，不排除六run。原始评估保留。已只停止两个纯读结论消费者，未停止GPU Bank/原正式CPU owner；旧CPU完成后唯一修复owner重放固定原案例、逐项核验非归因字段及固定动作/state trace后补正，禁止以旧计数选择后继。源协议本已要求v2，冻结训练源码不改。EVIDENCE_MEASUREMENT_HOLD记录环境接线和修复要求，不是能力门禁。

R10被动归因修复接续：原CPU唯一owner已完成7最终/父点、6中点和独立holdout并退出；独立CPU-only修复owner受同一全局evaluation.lock串行约束，20×64既有案例、相同导出/种子/生产JS重执行，Hunter/Tactical不重跑。新检查器限定只允许item_metrics下6个owned归因字段变化，其余字段及固定动作/state/event/reward/Review全hash精确；RED→GREEN五测试通过。首版检查器把JSON整数0到实数净距离误判为类型错误，已保留故障日志并排队唯一修复进程复用已执行缓存，不重复物理局；训练/封存协议未改。首批v2父dev携包7释放→1自有接触→1完整滑行，步行扣减净收益−1.3198格；control首seed携包12释放→0自有接触，不能继续概称全部自吃为0。完整补算收据仍待验收。

后继JS完整Bank技术准备补充：已用RED→GREEN处理已知被动ItemMetrics包装器；保存/恢复外部owned Map、serial、ownedSlide、base slide/glue及所有计数，未知函数仍拒绝。不以纯physics快照丢失归因aux。新增32对下一步技术验证，不代表真实携包parity；本准备未接入R10训练或生产。

## R10完整修复验收与R11机制后继预注册（20261024）

R10六run均u32、42声明原子点和192真实更新全部验收，正式1572864环境步/6144优化步；全部dev/holdout、37点Bank曲线、最终Bank-P、Hunter难中易各64与Python Tactical各64、固定偏航/训练奖励及GAE、FIFO/原标签/固定拟合/父漂移/实际对手源曝光全部消费。处理holdout交付5/7/9对匹配control11/8/14，共享未训练父22/64；处理相对control均差−6.25pp，训练seed×来源局95%区间[−16.15,+3.65]pp；相对父−23.44pp，[−36.98,−9.38]pp，三seed仅筛选。处理死亡对局4/18/18，对照8/12/9，父22；少死不能忽略窗口及活动变化。处理Python Tactical6/5/13对control11/8/11、父10；真实JS Hunter难/中/易处理[0,1,7]/[0,2,10]/[0,1,12]，control[0,2,10]/[0,0,3]/[0,0,2]，父[1,2,6]。不支持自然整链稳定改善，不晋升、不扩五seed，不从退化control宣称成功。

被动归因已20组×64原局修复：755字段更正；全部其他逐局字段及固定动作/状态/事件/奖励/Review hash精确。重执行源局物理步2773949，固定Review验证另41813，合计2815762；0新独立样本/优化步。原v1/错误解析日志保留，TECHNICAL_FAILURE_RESOLVED及修复收据明确解决，不排除有效训练。旧零owned解释撤回，修正后dev三control持包owned接触0/7/10、有益完整1次所在seed为后两者；处理1/0/1接触，均0完整有益。不能用稀疏道具proxy替代完整能力。

按R10已冻结条件后继，建立R11独立CPU机制namespace`runs/dagger_rl_20261024/r11_native_item_mechanism`。共同父及首配对两臂固定dev64，按game/tick取各模型最先至多4个真实持包释放来源和4个持包香蕉库存来源；重叠标注，不按成功选。每状态保存完整生产JS env/RNG、外部owned Map/滑行aux、实际原生取包因果证据、贪心无记忆推理协议，逐状态下一步parity。三分支：承诺原计划、逐步item老师（原JAX forecast/eligibility+真实JS合法mask）、学生闭环；共同enemy-wait且原生恢复计时，不叫自然能力或Q。无item集合时的路线fallback仅生产JS可观察方向，不冒称JAX完整时序安全等价。最多24状态×3×128=9216分支步；预声明总上限494592物理步、0新独立来源/优化/采用更新、不用holdout调参。发现具体缺陷才最小TDD单变量后继；无可证改动明确科学决策点，不盲扫、不用无关训练占卡。R10训练/Bank owner均退出，8卡0MiB/0%，旧暂停SHA保持。

R11技术兼容修复（非科学变量）：源game0的ReviewPlayer.countRandom会重新赋值rng；诊断StatefulSim setter原先错误重置已推进随机流，使Review@200校验失败。RED已精确复现，GREEN保留包装前流与snapshot/restore访问器。该错误仅在R11诊断子类，R10/生产不变。原PROTOCOL保留，TECHNICAL_SOURCE_AMENDMENT绑定前后源码hash、相同选样/预算；两个失败源局与200步Review尝试保留并计成本，已完成来源不重跑。失败前/后game0三个已保存真实Bank二进制hash精确相同，支持原源码源局不受observer赋值影响；后续完整源局仍逐项对R10固定结果校验。

## R11实测终态与R12冻结合同（20261024）

R11完成17既有来源局的22个真实持包状态；死亡库存14、原生糖浆库存6、自由存活合法库存2。全部完整env/RNG/归因aux下一步parity通过。父/control/qualified起点获认证2/1/0个，承诺计划3/3安全有益；逐步老师完整有益滑行4/3/3，学生1/0/0，学生老师一致0/71、0/121、56/160机会tick。相关小样本和enemy-wait分支只支持覆盖假设，不是自然能力、Q或JS/JAX物理parity。实际物理成本上界44651，0新独立来源/优化步。

实验`DAGGER-RL-20261024-R12-GREEDY-DAGGER-STATES`，namespace`runs/dagger_rl_20261024/r12_greedy_dagger_states`：只比较独立DAgger辅助collector当前学生categorical（sampled_aux）与masked greedy（greedy_aux）。两臂PPO始终fresh categorical；辅助GAE丢弃，只取老师监督memory。相对R10两臂共同新增辅助stream，所以只新配对两臂构成单变量比较，不与旧R10做单变量归因。共同父r19 g32 policy `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`，state `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`，输入`61829b5a4a94ff1c468cb2a8ab055b6412415262f89ade97a75376dc32315b8e`，同保存Adam不重置。

三配对seed2026102601/02/03，每臂固定32更新，PPO64×128与辅助64×128各262144步，总每run524288环境步/1024优化步；六run3145728/6144。joint PPO+DAgger.2+combat KL.05全程，lr7.5e-5，奖励/网络/对手/课程/老师不变，R10失败资格处理两臂关闭。75%自然/25%真实因果取包train Bank，禁人工预携包；Bank仅map806/P0/allfar且已使用。current/ring/fixed=0/0/100%，fixed为JAX rule3，非Python/JS。每更新原子保存双collector env/aux/RNG、params/Adam/history；声明u0/1/8/16/17/24/32。

23 focused回归和三组JS通过。新smoke仅3072执行环境步/6优化步，验证Actor/Critic/ability非零finite及fresh双stream下一完整更新精确，0采用，不是能力门禁。CPU/JS唯一owner带显式v2 preload，独立全局evaluation.lock；父+6final各自然dev/holdout64、Tactical64、真实JS Hunter难中易各64，中点自然dev64，所有声明点Bank曲线、最终Bank-P；fresh自然holdout2026111200起64，dev2026102480，不用holdout调参。marker为TECHNICAL_SMOKE/TRAINING_COMPLETE/NATIVE_BANK_COMPLETE/CPU_EVALUATION_COMPLETE/COMPLETE_FULL_TERMINAL_NOTICE。训练owner逐run通知且立即接续六run，CPU不阻塞；技术故障保留原子点并通知，USER_STOP在安全边界消费。六run完整证据支持才冻结五新seed确认，否则检查首个辅助标签偏离/机会覆盖与native链，选一项有证据最小改动或报告科学决策点，不盲扫、不择峰、不恢复旧队列。

复现入口（禁止重复活跃owner）：`<PYTHON> -m scripts.dagger_rl_r12_run`；CPU为`<PYTHON> scripts/dagger_rl_r12_cpu.py`；独立完成消费者`<PYTHON> -m scripts.dagger_rl_r12_run complete`。运行产物不入Git，不push不发布。

R12封存协议最终SHA `d1a229ce7547f75bfd404939ed127e6ae919ca140baf87aa3592c1503b518a4a`，源码seal `abac4cadd8d835898f635b2c67539255d513725b3cee43c8eeb7d2504bed0e7c`（398文件）。新技术smoke首保存点Actor/Critic/ability实际L2=0.1239377041/0.0743157295/0.0022605367，finite，双stream完整状态已保存；下一完整更新fresh parity仍待完成，不是正式训练。仅复制12个缺失且SHA精确的历史编译cache到独立目录，不继承历史checkpoint/RNG/Adam；编译与有效训练计时分列。

R12首正式更新前补冻结零交互/零更新CPU诊断：声明点同输入reservoir精确replay拟合、固定首≤64道具旧标签保持/成员、同u16 sampled臂64状态父漂移及战斗距离层；PPO和辅助各64保存状态的机会拒绝按来源分列。唯一只读诊断owner，全部TRAIN快照，不新增rollout、不消费holdout，不作为训练门禁。定义/脚本SHA见ADDITIONAL_CPU_DIAGNOSTIC_PROTOCOL。

R12新技术smoke完整通过：双stream env/aux/RNG、Adam/history新进程下一完整更新hash精确 `c0411450a9177bca9e517ca5f20fa500713ee4aff6ee374e597c9fdb183a8b00`；ability梯度范数7.8475427628、Actor/Critic/ability更新finite非零，3072执行环境步/6优化步、0采用。owner已直接转formal，不等待CPU或能力阈值。

R12正式首更新独立验收：`sampled_aux_s2026102601/u001`，16384真实环境步（PPO8192+辅助8192）/32优化步，Actor/Critic/ability实际L2=2.6727974520/0.6610393389/0.0915749433，finite。state SHA `4cc5e08b1cd8a5319cb8bb64b9f90ec5053eca47d9acef74a420c72a769942ae`，policy SHA `9924dbb5d43a15d557931a284e9300c7dfa0ff070ac2835e6b8994f39ace536d`；父params/Adam精确，未继承smoke，201新道具监督行。首批近120秒每卡GPU被动采样峰值71–79%，当前含编译均吞吐99.67步/秒，尚不代表稳态。父自然dev同64局全部逐局字段、v2owned统计与R10修复结果exact，0新增独立统计支持；Hunter安全胜1/2/6，与生产协议一致。唯一owner继续六固定run，CPU/JS独立。

R12固定run终态 `sampled_aux_s2026102601/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `f625b7de27641121efce273a8b930cfee9429e658393c9650b76a3305707feb8`。辅助实际自然ticks/释放/接触标签=245576/43/124，真实Bank=16568/213/89。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R12首配对实际干预验收：同seed u0参数/Adam及PPO/辅助初态exact；u1 PPO完整末state/aux/RNG、事件、奖励分量和GAE均exact，只有辅助学生行为与监督memory先发生变化。sampled/greedy辅助release老师201/158行、contact均0（首批自然均0，Bank来源），实际Actor/Critic/ability两臂参数差L2=1.0497909875/0.4068669201/0.0248877028。greedy自身更新Actor/Critic/ability=2.5751710396/0.6799791548/0.0912770778，finite；state SHA `5d79571adf9b163498f859d44acc1c90f6d972fd8bdd4d6085140d7a9ff78781`。非等价臂，但不是能力结论，Critic差允许global clip/Adam间接耦合，非共享主干证明。

R12固定run终态 `greedy_aux_s2026102601/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `71092d0ec1c03f9b4f1736fbbff13b4306912f305600f8b2b85042302e66a6ee`。辅助实际自然ticks/释放/接触标签=248446/6/18，真实Bank=13698/168/97。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R12完整holdout消费前补充条件后继冻结：若整链、尾部、最差点与父点战斗/安全支持，保持完全同配方做五新配对seed2026102611–15、十run各u32，5242880环境步/10240优化步，独立holdout2026111300起64，新技术smoke3072/6且0采用。若实际干预/固定拟合存在而整链仍不保持，下一问题先只读核验已有22真实生产状态及26 TRAIN参考的obs/global/mask语义，区分同输入JAX/ONNX数值一致与跨引擎输入一致；0新环境/优化步，不把dev样本纳入训练或把输入审计称物理parity。只有完整固定证据后选择，不盲扫、不让报告门禁阻塞已有训练，也不用无关工作占卡。定义见NEXT_SCIENTIFIC_QUESTIONS。

R12固定run终态 `sampled_aux_s2026102602/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `105144c4f2e457e87aa44b0fd56af9a4bf13ebc1fb353e4c27fb5094b0038646`。辅助实际自然ticks/释放/接触标签=247849/13/45，真实Bank=14295/349/91。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。


R12固定dev首局动作核验（seed2026102480，不按结果选择）：

| 模型 | 战斗决策ticks | 放泡请求 | idle移动请求 | 窗口/取包/交付 |
|---|---:|---:|---:|---|
| no_update | 1944 | 61 | 84 | 3/1/0 |
| sampled_aux_s2026102601 | 2400 | 9 | 61 | 0/0/0 |
| greedy_aux_s2026102601 | 2400 | 4 | 241 | 0/0/0 |
| sampled_aux_s2026102602 | 2400 | 0 | 197 | 0/0/0 |

动作是输出请求，不冒充实际放置/位移或全64局因果结论；三训练点同局均未进入目标阶段，首偏航null不是返程成功。原始统计见EARLY_FIXED_ACTION_ACTIVITY。Python-host Tactical使用native JAX 300tick协议，真实JS Hunter独立；JS零Value折扣塑形和不是训练优势或Q。

R12固定run终态 `greedy_aux_s2026102602/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `7f7e10757ecee6f1bedd3d44fd0246da628e9b9f421390fc413a2227b7962d49`。辅助实际自然ticks/释放/接触标签=248120/5/32，真实Bank=14024/348/46。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R12固定run终态 `sampled_aux_s2026102603/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `9307e52242f62c2b61513e4c3d16a4250205e5e2fecc091f813600f7a2f30359`。辅助实际自然ticks/释放/接触标签=246707/21/31，真实Bank=15437/171/82。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R12固定run终态 `greedy_aux_s2026102603/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `a3a86a290547ccef2a4b443cfbacb6fb224581744988ae16caa6f8a45bd8a9f8`。辅助实际自然ticks/释放/接触标签=248847/14/31，真实Bank=13297/176/30。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R12六run已完成u32（3145728正式环境步/6144优化步），最后greedy_s2026102603 state SHA a3a86a290547ccef2a4b443cfbacb6fb224581744988ae16caa6f8a45bd8a9f8。GPU接续已声明Bank，CPU异步最终/中点/holdout；此时尚未作能力结论。按已冻结条件后继补登记R13只读输入语义审计：22真实生产取包状态+26已有TRAIN参考、三固定模型、双座位，0新增环境/优化步；分开观测投影、危险图算法、movement/ability mask、同输入JAX bf16/ONNX fp32。native不支持的道具类显式统计，不当跨引擎物理parity，不采用dev训练。仅完整R12证据满足失败分支后执行；当前未启动R13训练，旧暂停不变。定义见runs/dagger_rl_20261024/r13_input_semantics/PROTOCOL.json。

R13执行顺序技术更正（执行前v1保留）：零新增环境/优化步的输入审计可在R12六训练预算完成后与Bank/固定CPU评估重叠，不设置等待完整报告的自设串行门禁；任何确认或后继训练的科学选择仍在完整R12固定证据后。R12源码/预算/评估不变，无holdout选参。

## R12完整终态与R13语义审计（20261024）

R12六run u32/192次更新，正式3145728环境步/6144优化步；42声明原子点、双stream状态/Adam与全部dev/holdout/Bank-P/固定偏航/奖励-GAE/历史标签及真实暴露已验收。自然holdout sampled交付7/7/14、死亡12/16/18，greedy交付2/10/12、死亡10/12/14，共享父交付18、死亡15（各64局）。处理对control平均交付−2.08pp，训练seed×来源局95%区间[−12.5,+8.85]pp；尾部6/6/4对6.5/8.5/7.5、最差2对7，不支持整体改善，不择峰、不扩五seed。JS Hunter难/中/易处理[0,4,9]/[0,2,5]/[0,3,5]，control[1,1,10]/[0,6,14]/[0,0,8]；Python-host native JAX Tactical处理7/9/4、control12/1/9，父10。少死不抵消机会及交付下降。两流fixed JAX rule3=100%，current/ring=0；CPU与GPU owner均完成，现场8卡0MiB/无compute app，旧pause保留。完整证据SHA d8ee7ec449e3b2711ee1f058cd989d998323647f8005e4bc081bf708f240b090。

R13为0新增环境/优化步的只读审计，不是新能力样本。22真实生产状态19个、26 TRAIN参考22个，确认唯一非危险图观测差异ch7：生产JS包含可拾取香蕉/胶，JAX core.make_obs遗漏tactical_crate。所有其他23通道/global24在1e−5内一致，已有帧危险图一致；不是跨引擎物理parity。生产活跃learner20状态中14个移动mask不同，TRAIN26中20个；只读用已有实际移动速度替代核心探针成长速度后分别剩1/0。剩余生产墙边界一例未解，单纯量化pos为fp32未消除，不虚称全parity。三个固定模型的same-input JAX/ONNX数值差异、同后端ch7变化及mask支持变化分源/座位保存。mask KL跨支持集不定义，改报总变差与移除概率。首版工具将问号crate7误作普通道具的结果已保留为无效，正确投影后其余通道断言通过。

选择下一轮仅训练ch7对齐；不混mask修正/奖励/倍率/网络/对手。缺陷语义已证明，但对自然退化的因果贡献仍待配对实验，不能称根因已全部解释。RED两失败→GREEN5测试通过；disabled逐位legacy、enabled只合并拾取物而非放置field、无改其他平面。详见R13 COMPLETE及原始审计。

## DAGGER-RL-20261024-R14-ITEM-OBSERVATION 冻结合同

共同干净r19 g32 policy SHA a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95，恢复同保存Adam，不继承退化control/smoke。两臂legacy_obs/aligned_obs唯一变量是训练ch7是否合并可拾取tactical_crate；双collector都categorical，fresh PPO不读历史，DAgger.2/保持KL.05/lr7.5e−5、奖励/固定JAX rule3/75:25 reset/网络/标签及reservoir均不变。seed2026102621/22/23，六run各32更新、PPO262144+aux262144环境步/1024优化步；合计3145728/6144，保存0/1/8/16/17/24/32并每更新原子保存完整双状态/aux/RNG/Adam/history。单run独立进程防止Python/JAX缓存跨臂静态编码开关污染，不重跑已完成run。

smoke仅3072执行环境步/6优化步、0采用，检查finite/真实Actor-Critic-ability非零及fresh完整下一更新exact，非能力门禁。独立CPU/JS owner显式v2 preload、唯一全局evaluation.lock，parent+6最终自然dev/新holdout各64、JS Hunter难中易各64、Python-host原JAX Tactical64、中点自然dev64。holdout2026111400起；dev/Hunter与原协议固定。Bank/Bank-P原单图P0全far、已使用来源保留，R14所有模型统一以生产对齐ch7输入评估（与R12的Bank输入版本不同）；Tactical保留原legacy输入协议，单独标注，真实JS不变。不得跨此协议版本混比较。

无新道具目标、mask/奖励/对手/课程/网络改动，不用holdout选量级，不择峰。R12负对照与未训练父保留；R13只读缺陷证明不冒充自然根因。完整结果支持才五新seed确认，否则下一独立问题是实际速度的mask语义，须新的单变量协议/技术正确性，不自动合并、不盲扫倍率、不用无关训练占卡。旧O5/continuous/储备永久暂停，新owner只消费本六run与Bank；USER_STOP原子边界检查。每run终态通知并立即接续冻结队列，CPU不阻塞GPU。复现：<PYTHON> -m scripts.dagger_rl_r14_run；CPU：<PYTHON> scripts/dagger_rl_r14_cpu.py（禁止重复活跃owner）。原协议及下一问题见R14 PROTOCOL/NEXT_SCIENTIFIC_QUESTIONS。

R14新smoke已完整通过：3072执行环境步/6优化步、0采用；Actor/Critic/ability L2=0.1256475590/0.0731416781/0.0023117267，ability真实梯度范数7.9179439545，全部finite。完整双stream env/aux/RNG、Adam/history下一完整更新fresh hash exact 6235dac95d33a2b63e6c85ad290db6ac58263948c5296dc9b75fd122fc668e2e；原子state SHA 3760d2c61915e9e2f6e5ee55873e9d3fa3c17edcac53bdf8c4074b6d27c84dc1。唯一owner已串行进入首正式legacy_obs_s2026102621，不等待CPU/JS或能力阈值；正式首更新尚待独立复核，不把smoke/编译计作正式训练。

R14监督补证：只读诊断owner因复制遗留selector字段启动即KeyError，已修正为真实execute_item_observation_alignment并核验input_audit_sha256；仅重启此只读消费者，既有不可变快照已补算，无重跑训练。405项封存源码hash无差异，旧global pause SHA保持。首臂u16固定自然dev交付对局9/64，父12/64；原生窗口事件29对84，库存窗口取包6对21，窗口原包交付5对12，首方向20/23对52/59；尚无配对处理终点，不据此停训/晋升。候选carry-only P0实际速度mask的RED→GREEN四测试通过，未接入R14；同状态/动作/RNG完整reward-transition CPU证明独立进行，执行步数和独立来源另计。

R14固定run终态 `legacy_obs_s2026102621/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `b0fd8237196a491a1244a8ca57df80f3a91683413a7890aad4396e0120a77496`。辅助实际自然ticks/释放/接触标签=245148/32/47，真实Bank=16996/168/172。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

独立下一问题技术证明完成：carry-only P0实际速度mask在26既有真实TRAIN来源状态、每状态记录动作+15联合动作分支下，旧/候选共832技术transition重放。next_state、完整aux/reward ledger、reward/done/info/全部reward_components digest完全相同（6c484357e0022f060640b48337e3a1593f1eea35c6a55bcb03009c5b6954d8ed）；20/26来源状态mask改变（320重复分支行），ability及P1 movement exact。0采用环境/优化步，不是新独立能力样本或跨引擎物理parity；候选尚未接入R14。后继若采用应仅包装train.legal_mask，保持env.legal_mask/step原样，进一步隔离奖励语义。R14首配对初态ch7激活覆盖：PPO10/64、aux13/64、各23格；非全rollout机会分母。

R14处理臂aligned_obs_s2026102621首正式u1已复核：16384环境步/32优化步、finite，Actor/Critic/ability L2=2.3973677371/0.9260191363/0.0873150304；state SHA b2489181e98846a472a93c793f34cbf4ed09c480879cb5cccb6e47a9ea18f7ba。首配对相同父/Adam/双流初态，实际跨臂首更新Actor/Critic/ability L2=1.2198270100/0.7916434872/0.0188971162，说明处理实际激活，不证明能力提升；PPO动作/reward/GAE可立即不同，未声称首PPO batch exact。下一候选仅train.legal_mask真实接线证明：26来源中20改变、37新增方向项、无移除，obs/global/ability exact，env mask/step对象完全不动，0新增环境/优化步。

R14首control动作链快照（单seed dev，非处理效应）：aux自然245148tick有32释放/47接触标签，真实Bank16996tick有168/172；u32保存库42释放标签全部greedy匹配、集合质量均值0.97655，50接触也全匹配。自然dev持包香蕉61输出请求/32合法释放/7自放后自吃，按释放时阶段归属的完整自放滑行只有1结算且受益、净home推进0.48；窗口释放来源另有10完整滑行4benefit/6harm。阶段统计不是同一放置cohort，不将这些跨阶段数字相除；release概率/局内普通contact不冒充完整净收益或交付因果。此暴露缺口与拟合并存，待三配对完整证据再择下一问题。

R14固定run终态 `aligned_obs_s2026102621/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `3405dd7e73b00cd99a471e845f883002bf57b852cb2b5ed9ab1fba2d6c6ed202`。辅助实际自然ticks/释放/接触标签=240948/31/98，真实Bank=21196/141/67。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

下一mask候选collector-only安装TDD：新增test先RED（missing install_learner_mask）→GREEN5测试；disabled保持原train.legal_mask对象，enabled只替换collector binding，env.legal_mask/step对象不变。NEXT_MASK_PREPARATION_ACCEPTANCE固定候选/测试/RED-GREEN及26来源输入证明、832同动作技术重放hash；尚未授权进入具体后继命名空间或混入R14。首配对最终dev：control交付26/64、aligned6/64、父12/64；Hunter hard/normal/easy分别0/2/6对0/6/8，Tactical10对8。单seed差异不用于提前停止/择峰或宣称整体处理效果；其余固定配对照常继续。

R14固定run终态 `legacy_obs_s2026102622/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `e1fbee65a21b76e2e24fcead9c9958b928c13242dcf05d0c92164fa05d7d3a04`。辅助实际自然ticks/释放/接触标签=240600/56/116，真实Bank=21544/112/68。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R14固定run终态 `aligned_obs_s2026102622/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `9c8b5ae7b29dd296cc62f937d706c5ba7528ec3ce435e938cd2d5f678211c3da`。辅助实际自然ticks/释放/接触标签=242904/4/27，真实Bank=19240/116/132。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R14固定run终态 `legacy_obs_s2026102623/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `21ea97d487e761d36ee7e8c0782f2d87ec90db26044cb60504dbaf63f902ccb9`。辅助实际自然ticks/释放/接触标签=245065/12/25，真实Bank=17079/88/24。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R14固定run终态 `aligned_obs_s2026102623/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `cc47350e067fd627482dc69d43787034f39c1965a646c31bda48dd62a85bdf95`。辅助实际自然ticks/释放/接触标签=243312/93/158，真实Bank=18832/113/104。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

## DAGGER-RL-20261025-R15-CARRY-MASK 冻结合同

R14完整证据SHA f8bc16f56461618d071d14a53250107e3129d5926cde9183dd77d6b16cc451d1已消费：自然holdout control22/7/14、aligned8/14/1、父13（各64局），处理对control均差−10.42pp、seed×来源局95%描述区间[−28.13,+9.90]pp，尾部6.5/8.5/1对17.5/6/4.5、最差1对7；不扩ch7五seed确认、不晋升。Bank原包交付也低于control，仍为旧map806 P0全far条件证据；不能以Bank改善替代自然整链。

R15协议SHA 683478247d85055849caa2118b5117823d0ea8b26cf2cfb54ab82a9e07c0d5b2。共同干净r19 g32 policy SHA a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95、state SHA 2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5，同保存Adam，不继承R14/smoke端点。两臂legacy_mask/actual_mask唯一变量：train.legal_mask的P0 alive/free/full lesson实际持包移动探针由成长速度改为已有物理实际速度；env.legal_mask/step/reward、P1和ability不变。两臂common legacy ch7、categorical双collector、联合PPO+DAgger.2+combat KL.05/lr7.5e−5、固定JAX rule3、75:25自然/真实train Bank reset、网络/道具目标/标签/均匀reservoir8192均不变。无人工携包，不混奖励/采样/对手/倍率。

配对seed2026102641/42/43，六run各固定32更新、524288环境步（fresh PPO/aux各262144）/1024优化步，合计3145728/6144；每更新双流env/aux/RNG/Adam/history原子保存，固定消费0/1/8/16/17/24/32。单arm/seed独立进程，先3072技术smoke（6优化步、0采用），只查finite/Actor-Critic-ability非零与fresh完整下一更新exact，不设能力门禁。

CPU/JS唯一owner：dev64 seed2026102480起，中点16/最终32；新自然holdout64 seed2026111600起，父+六最终；真实JS Hunter难中易各64、Python-host native JAX Tactical64独立。Bank/Bank-P同旧32来源×8重复，R15全部模型统一production ch7+carry-only实际速度P0输入mask，版本不同于R14 Bank；Tactical共同保留原legacy输入。生产JS不改、老师关闭，不用holdout调参，不择峰，实际current/ring/fixed与近中远暴露另报。

已完成RED/GREEN及7 focused检查，26真实TRAIN来源20 mask变化/37新增方向；832同动作/RNG技术重放完整next-state/aux/reward/info exact，不是新独立能力样本。只读历史标签的跨臂精确行相同率受输入编码/访问状态差异影响，零重合不能当作遗忘或淘汰；同臂补证异步。

终态接续：可信整链与父战斗安全保持才五新配对seed2026102651–55确认（10×32、5242880环境步/10240优化步、新holdout2026111700）；否则先0新增步的真实自然机会/标签冲突与共同baseline保持审计，选新的有界科学问题，禁止无关占卡或倍率/奖励盲扫。旧O5/continuous/储备暂停SHA保持不变；仅本namespace有GPU授权，USER_STOP按原子边界消费。训练owner、独立CPU/JS、每run终态/完整消费通知均持久；报告不阻塞健康训练。

复现（已有owner时勿重复）：<PYTHON> -m scripts.dagger_rl_r15_run；CPU <PYTHON> scripts/dagger_rl_r15_cpu.py。run/state/cache/eval全新，编译cache逐hash复制且无params/Adam/RNG继承；不push不发布。

R14完整能力终态：42声明原子点/全部CPU-JS dev与独立holdout/Hunter三难度各64/Tactical64/Bank及固定Bank-P/全部只读诊断已消费，COMPLETE_EVIDENCE_SYNTHESIS SHA f8bc16f56461618d071d14a53250107e3129d5926cde9183dd77d6b16cc451d1。control自然holdout22/7/14、处理8/14/1、父13；paired交付−10.42pp，95%描述CI[−28.13,+9.90]pp，无法证成精确效应或安全等价。Tactical control10/7/9、处理8/5/8；JS Hunter各难度与尾部/最差点完整保存，不选第二seed宣称改善。Bank原包交付处理低于control均差−2.865pp，来源局×seed CI[−6.641,+0.391]pp；对父改善是已取包条件证据，不抵消自然整链退化。

R14额外0步同臂早期标签审计已完成，原COMPLETE/hash不改：固定每臂自己u1首64释放标签全部Bank，末点各自精确行保留与功能拟合见SAME_ARM_ITEM_RETENTION_COMPLETE。原跨臂legacy输入行对aligned的零重合不能当遗忘/淘汰；输入与访问状态已不同。同臂早期拟合下降可记录，不能据此单独证明自然下降由遗忘造成。72组Bank结果累计5078934有效评估环境步（不含padding重复模型与terminal无效tick），与3145728正式训练步/6144优化步、3072技术smoke步/6优化步分别记账。actual current/ring/fixed=0/0/100%，固定JAX rule3，不是生产JS Hunter。

证据排序：最差自然点先丢失机会创造与进入目标阶段，不能把1/1条件交付率当成功；真实自然道具标签稀疏且固定早期前缀全来自Bank，高库内拟合不等于自然技能；已证明的持包mask速度错配仍未测试其因果贡献；同臂功能保持变化并存，但没有共享Actor/Critic主干或有害梯度因果证明。下一轮只测mask输入语义，保留这些未解上游问题，不混奖励/倍率。

技术证明分母命名更正：26参考状态实际来自8个已使用TRAIN来源局（每局3–4个连续帧），并非26独立来源局。旧NEXT_MASK_SAME_ACTION_PARITY的independent_source_states字段为命名错误，原hash保留，REFERENCE_STATE_PROVENANCE_CORRECTION明确更正。20/26为帧计数，832为重复技术transition验证，0新独立数据，不由此估计能力CI；不影响逐位技术parity或既定单变量选择。档案含旧O5来源轨迹，仅作已冻结真实因果TRAIN诊断，不恢复O5训练。

R15首正式更新独立验收：`legacy_mask_s2026102641/u001`，16384环境步（PPO/辅助各8192）/32优化步，finite，Actor ΔL2=2.45981683088、Critic=0.68522030003、ability=0.08319584268；父参数/Adam精确且未继承smoke。state SHA `a60b90540da9121027c84d133f7b1cfba0850670e4949da82304860b0dcbced1`，policy SHA `dd14b00e3d73f2dffdf158a854a52f4e1d9ff131aae5292e1fdb052d6a74b1e4`。fresh-process双流完整下一更新hash精确，技术预算3072环境步/6优化步、0采用。正式首批8卡均采到有效计算，采样峰值69–78%，不能用保存时瞬时0%或显存替代有效训练量。辅助道具老师219行全Bank、自然0，PPO实际释放23次（其中持包10次），只是首批来源/覆盖，不是自然能力结论。414封存源hash未变、旧暂停hash未变，唯一owner持续固定六run及异步CPU/JS。

R15固定run终态 `legacy_mask_s2026102641/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `ebaee4a3ee889835979d30ef80f6d538cafd8bbd6cc402f41b0172569e0a8366`。辅助实际自然ticks/释放/接触标签=239234/38/98，真实Bank=22910/240/85。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R15固定run终态 `actual_mask_s2026102641/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `8bc4ea554c56f76d8fcaf60d0346e2b6da120ec5fd9f4f6c325114955f13538f`。辅助实际自然ticks/释放/接触标签=242947/3/15，真实Bank=19197/243/94。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R15固定run终态 `legacy_mask_s2026102642/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `764ed99e3f1b29b768fdb072ad33124ae02aaaa4f491c03f4da45fc8f303b137`。辅助实际自然ticks/释放/接触标签=247484/76/128，真实Bank=14660/103/22。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R15续做：已独立复核三固定run原子SHA和全部真实非零更新；第四actual_mask_s2026102642持续正式预算，CPU/JS异步。封存源与旧暂停未变，补充诊断使用独立只读消费marker，不阻塞训练；备选顺序/联合/PPO-only共同KL模块仅13测试及旧TRAIN零步检查，尚未科学选择/预注册/排队GPU，完整证据后按NEXT_SCIENTIFIC_QUESTIONS选择。详见reports/dagger_rl_iteration_20261025.md。

R15固定run终态 `actual_mask_s2026102642/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `d8a4059ebdce01f93255a44006f517e1d126222e45507a66f9672669a836d3b6`。辅助实际自然ticks/释放/接触标签=242383/33/119，真实Bank=19761/120/104。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R15固定run终态 `legacy_mask_s2026102643/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `f9d807d1caaf0a054f93025cc658e97daf188aed36cdd7612bb3282d6fdfc71d`。辅助实际自然ticks/释放/接触标签=244757/14/60，真实Bank=17387/140/67。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

R15固定run终态 `actual_mask_s2026102643/u032`：524288真实环境步（PPO/辅助各262144）/1024优化步，32次finite/Actor/Critic非零；原子state SHA `c6fc521945f2e95db3e9c03e3a092fcda22998bcec3de8dcac3b9f3d5448edd2`。辅助实际自然ticks/释放/接触标签=244221/6/20，真实Bank=17923/149/73。两stream合计fixed JAX rule3=524288tick，current/ring/fixed=0/0/100%，历史父0；不是JS/Python对手。唯一owner串行接续冻结六run，CPU独立；此为预算终态，不是能力改善或GPU已释放。

## DAGGER-RL-20261025-R16-RETENTION-SCHEDULE（正式前预注册）

R15完整证据与0步审计已冻结后选择；协议SHA `1d65cab8b0d0f55fcb4ec777e5d8bebddbdbf72cac752ea5e5ae66df76880cd7`，共同干净r19 g32 policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`、保存state SHA `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`、parent input SHA `61829b5a4a94ff1c468cb2a8ab055b6412415262f89ade97a75376dc32315b8e`。不用退化trained control作父；旧O5/continuous/储备暂停不解除。模式为单变量因果三臂，非整合配方：PPO+共同combat-parent KL .05；u1–16 DAgger .2+同KL、u17–32 PPO+同KL且DAgger严格0；全程联合PPO+DAgger .2+同KL。Actor/Critic独立，Dagger阶段冻结Critic参数及Adam mu/nu，global时钟推进；不称共享主干干扰。

配对seed2026102661/62/63，队列每seed PPO控制→顺序→联合；九run各固定32更新、524288真实环境步（PPO/辅助各262144）/1024优化步，总4718592/9216。保持64env×128step、lr7.5e-5、gamma.995/λ.95、clip.1/value.5/entropy.005、globalclip.5、1epoch、minibatch256；相同自然75%/真实因果TRAIN Bank25% reset、uniform reservoir8192、item/route4096/4096、old/new .5/.5、固定JAX rule3（current/ring/fixed=0/0/100%，实际消费另报）。共同legacy ch7/移动mask，不组合未证成R14/R15输入处理；奖励/老师安全集合/道具目标/对手/课程/网络不变，香蕉与胶不混新干预。无人工携包初始化，库存老师仅原生死亡/糖浆窗口、自由存活偷包合法但无库存老师奖励，持包复活后仍安全返程。

同总采集/查询/优化预算；纯PPO仍查询老师作匹配机会统计但不采用监督损失，顺序RL段不使用历史DAgger。真实每阶段PPO采用环境步、监督优化步、Critic更新步逐条记账；PPO仅fresh categorical学生流，辅助advantages丢弃。量级与16/17切换固定不调参，不用holdout选参。技术smoke仅finite、真实Actor/ability梯度及更新、PPO Critic非零/DAgger Critic与moments冻结、完整双stream env/aux/RNG/Adam/history新进程三分支下一更新hash精确；3072技术环境步/18技术优化步、0正式采用，不含能力门禁。19项focused GREEN，父SHA/未知臂/预算漂移3个RED后修复；原始旧TRAIN真实网络loss梯度检查已通过。

保存0/1/8/16/17/24/32及每更新安全原子点；CPU唯一owner跟随u16/u32，dev64 seed2026102480起，全部父+九最终独立holdout64 seed2026111800起；真实JS Hunter难中易各64、Python-host native JAX Tactical64分报，Bank/P仍单图P0全far且已使用，统一production ch7+actual携包mask输入，不作为跨图独立证明。自己u1/u16 learner memory按自然/Bank×route/release/contact首<=64稳定arrival，比较父/u1/u16/u17/u32固定输入功能拟合；不是独立能力或遗忘因果证明。被动机会统计区分库存/合法any-item/老师前置资格/认证标签/实际释放，按来源阶段分ticks，不混事件cohort。固定首次偏航Review、reward/近似量/Bank重建GAE保持定义，严禁奖励/Value冒充优势或Q。

终态：每个固定run完整原子保存后唯一owner先核验compute apps释放，再串行接续预声明队列；CPU不阻塞健康run。九run+声明Bank/P完成按三seed完整曲线/尾部/最差点和未训练父判断，可信顺序改善才另冻结五新seed确认；否则0步首次漂移/覆盖/切换拟合审计选一个有界问题。共同基线退化必须明确报告，不无限扩预算/盲倍率/新奖励/无关占卡。USER_STOP在最近安全原子点优先，禁止领取下一更新/run/smoke；旧队列永不自动恢复，不push、不发布。复现：`<PYTHON> -m scripts.dagger_rl_r16_run`；CPU `<PYTHON> scripts/dagger_rl_r16_cpu.py`；拟合 `<PYTHON> -m scripts.dagger_rl_r16_diagnostics`；完整消费者 `<PYTHON> -m scripts.dagger_rl_r16_run complete`。已有owner勿重复。

R16估计量澄清（无训练改动）：匹配父点/Adam、总环境/优化步与老师查询预算，不匹配各损失分量的实际采用量——它们本来就是预声明schedule处理的一部分。每run PPO+KL控制采用262144 PPO环境步、0监督优化步；顺序采用131072 PPO环境步、512监督优化步；联合采用262144 PPO环境步、1024监督优化步。故本轮比较这些完整损失时序配方，不能宣称在相同累计分量梯度剂量下单独识别纯顺序效应；不临时追加半权重/倍率归一化。逐阶段原始采用量和总预算分别报告。

R16接管只读核验：首正式`ppo_anchor_s2026102661/u001` finite，Actor/Critic/ability实际参数Δ=0.951460/0.985958/0.012318；16384环境步/32优化步，父params及Adam exact、未继承smoke，state SHA `dbf881c0c27eaaf9fcb0636098ee529f9ca6ecfe7cf1335eb7abb0eeb1b96626`。Bank文件逐SHA复核，train/dev/holdout各32状态确实来自各32独立来源局，每局8次续跑的32×8来源聚类成立；不能误用26参考帧来自8 TRAIN局的分母，也不能把256续跑当256独立来源。Bank仍是既有单图P0全far，0新增环境/优化步；见BANK_SOURCE_CLUSTER_AUDIT。

R16固定run终态 `ppo_anchor_s2026102661/u032`：原子state SHA `d70128a1ca7da3ca8632140b0913196821d80b9a7a3de4dd2a1a20a0fa5ea997`，524288环境步/1024优化步，实际采用PPO环境步262144、监督优化步0、Critic更新步1024。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

R16首顺序臂真实阶段切换核验：u16仍为DAgger.2、Critic参数及Adam mu/nu与共同父exact；u17监督系数严格0，fresh PPO采用8192环境步，监督采用0优化步；Actor/Critic/ability实际Δ=0.884542/1.063227/0.012286。起点双流完整env/aux/RNG与同seed PPO对照exact；不使用历史行为数据冒充PPO，未改冻结16/17边界。

R16固定run终态 `sequential_anchor_s2026102661/u032`：原子state SHA `ec02eb4f0cc6b1219686f6e1bfe5b6ab1cff489596cfa97ea26d4865dfc3fbee`，524288环境步/1024优化步，实际采用PPO环境步131072、监督优化步512、Critic更新步512。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

R16固定run终态 `joint_anchor_s2026102661/u032`：原子state SHA `ec3873c8b64b0a5ab2686a0abc3619a03982e2e42224e0629f1ac61fcd58ae01`，524288环境步/1024优化步，实际采用PPO环境步262144、监督优化步1024、Critic更新步1024。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

R16只读诊断口径补充：冻结全部19份dev固定game0录像（父+九run的u16/u32），运行`node scripts/dagger_rl_direction_trace_audit.js runs/dagger_rl_20261025/r16_retention_schedule`逐Review验证，并在首个几何分歧处固定原始RNG与其他动作做5个原生一步方向分支。已保存v1/v2收据，v2精确计重放调用；不改R16封存训练/评估源，不用holdout调参。首顺序臂u32 seed2026102480 tick1667状态hash `dd78f6c1f57d961d36675b1dc25378b2bebf0de4ed42721adf5c152667776070`：几何标注oracle=[2]、模型动作3，真实一步动作3距离15.94→15.09163，动作2反至15.96591；事实next hash exact。JS oracle用固定.18虚拟坐标位移，未调用原生碰撞/实际速度移动，因此该“首偏航”不等于首个真实错误动作。此为一步距离排序诊断，不是经验Q、GAE或长程安全证明；训练老师另有三角连续势能及安全过滤，不能由JS例直接推断训练标签错误，须在冻结TRAIN learner状态另证。后继分析保留交付/死亡等原生结果，方向一致率仅视为近似proxy。当前6/19份，诊断分支15、前缀重放14059次原生step调用，正式采用环境/优化步0；后续按同声明补齐。

R16独立CPU TRAIN投影审计完成（协议 `a0c4b05165abfce1113a23dfeb67c5e9bb270e976c5405c11c0e718688b826da`）：首配对3臂×六固定保存点×双学生流，按env index在natural/真实TRAIN Bank×empty-window/carry每组首≤4，未按拟合/结果挑样。322行端点状态、189行实际可采用的ability0路线老师集合；11行相对同一原有时序安全候选的原生一步距离有正regret，按完整state hash去重后9个，而非11独立来源。包含两条目标格wait标签，此为任务语义与连续中心代价差，不能当移动投影错误；其余也仅证明一步排序不一致，不能单独证明长程Q或主导自然退化。189行中完整state hash唯一173个，不计算假独立能力CI。固定原敌动作/环境RNG、learner ability0，item override明确排除；有效1610、含padding实际2880个原生分支step，优化/正式采用0。原始state/aux hash、候选集与五方向next距离保存 `TRAIN_PROJECTION_AUDIT.json`，SHA `b11edf210d7bacd2955ae7b170e4c204d23b49c63cc32af5b6b8f1424f669bcc`；未改R16封存源/预算/队列。新脚本 `scripts/dagger_rl_route_projection_audit.py` 未被collector导入。

R16固定run终态 `ppo_anchor_s2026102662/u032`：原子state SHA `5c14ee5a3fa6cc71c3c5f87dd3e3e70ab3be15a8187c6693296fb1d878e35ea3`，524288环境步/1024优化步，实际采用PPO环境步262144、监督优化步0、Critic更新步1024。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

R16 TRAIN审计继续定位具体资格遗漏：`scripts/combat_item_engine.py:route_teacher`仅消费labels前5方向列、忽略第5列active；`scripts/je10_route_oracle.py:labels`在kind=0时active必False但仍可有非零方向/goal-wait，因此allow原生窗口不足以阻止无有效目标的路线标签。样本189条route-adoptable端点中7条kind0（7个完整state hash）；这不是全训练污染率，且端点老师输出不冒充已进入优化器的次数。相关两条goal-wait案例kind实际为0，不能据此责怪合法目标格wait规则。已有R16老师不修改；新独立未绑定guard仅消费现有active位，不改移动投影/奖励/道具目标。真实RED为2失败3通过（前两次fixture setup错误另保存，非机制RED），候选及相关focused GREEN通过；receipt INACTIVE_GOAL_GUARD_PREPARATION逐hash证明427封存源不变。候选未激活/未排队GPU，最终科学选择需完整九run证据；若采用则新namespace单独注册，不与投影修正混入一个处理。

R16固定run终态 `sequential_anchor_s2026102662/u032`：原子state SHA `f23fc19fa255486fba977d2b74e7884de1207da6142a58f073741da1f735c321`，524288环境步/1024优化步，实际采用PPO环境步131072、监督优化步512、Critic更新步512。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

无目标监督真实采用witness：`sequential_anchor_s2026102661/u008` auxiliary env12/source28的kind0端点，对应u009新入库arrival7145，obs/global/move-mask/ability-mask逐值差全部0；保存history中的路线标签move2与端点老师一致。重建原固定numpy replay RNG，u9抽中该row两次；当时DAgger系数.2、一个epoch全部8192个replay位置均参与，因此它确实进入监督损失，而非仅查询可达。GOAL0_REPLAY_WITNESS保存state/history SHA与复现方法；仍不把单例当总体污染率或逐样本Adam效果。DAgger目标是fresh PPO学生流的监督项与辅助学生history监督项各.5，PPO的优势始终来自fresh stream，辅助优势丢弃；不能误说监督仅用辅助流。R16继续作为原有共同老师配方下的固定预算比较，不以其证明老师资格正确时的顺序效果或作晋升。新namespace的资格guard是已实证候选，优先于未证主导性的移动投影改动，不把两者合并。

资格根因与候选更正（优先于前述active门控建议）：完整旧active位不仅含任务资格，还要求action_set当前格danger==0，原route_teacher以时序joint安全允许危险格逃离。真实正例u1 auxiliary env18为kind3/custody true/scene8/current-danger0.01777777，raw active5=0而eligibility7=1、时序安全move0合法；直接补active会错杀，已用REAL_GOAL_GUARD_REJECTED保留并否决，不能推广该候选。真正已证遗漏是有效目标kind>0：kind0仍落到fallback home channel并得到方向，原适配器未排除它。新候选**仅kind>0**，不叠加旧active/current-danger规则，不改移动投影。更新RED2失败9通过、候选及相关GREEN18通过；REAL_GOAL_GUARD_CHECK证明kind0所有joint标签清除且合法真实持包逃生标签逐位保留。GOAL0_REPLAY_WITNESS_v2还验证完整joint标签匹配与u9原采样2次。新GOAL_KIND_GUARD_PREPARATION明确supersede旧未激活准备收据；R16封存老师/奖励完全未变。

R16固定run终态 `joint_anchor_s2026102662/u032`：原子state SHA `f7c420e03b9f6a81ed9102a3529a20c486423249bf09122e7ca32159f9dad37d`，524288环境步/1024优化步，实际采用PPO环境步262144、监督优化步1024、Critic更新步1024。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

R16固定run终态 `ppo_anchor_s2026102663/u032`：原子state SHA `83093d993e27f886c767a08332a5d8dc825d701b84aeaaf925adc5bc8bd2c857`，524288环境步/1024优化步，实际采用PPO环境步262144、监督优化步0、Critic更新步1024。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

## DAGGER-RL-20261025-R17-GOAL-KIND（正式前冻结）

协议SHA `ca84cac861424abdeda932f3866f9ba6de5a586a54980b109b6af1a5944c8f3d`。R16继续固定九run且封存源码不改；后继仅追加路线老师`kind>0`资格，保留原时序joint安全与真实持包危险逃生，香蕉老师优先级、奖励、输入/mask、采样/课程、对手、网络、优化器均不变。整列legacy active门控已被真实持包逃生正例否决，禁止采用。

真实TRAIN审计322端点行、189 route-adoptable行中7行/7独立state hash为kind0，不代表全训练污染率；u8 auxiliary env12/source28的完整输入/标签在u9 history7145逐值一致，原replay RNG抽中2次并进入DAgger loss。只证明具体错误监督被采用，不证明它是全部退化的主因。固定JS首几何分歧需原生一步对照；一步距离不是Q/GAE，JS零Value塑形和不是训练优势。真实正例kind3/current-danger非零仍完整保留原joint逃生集合。

共同干净r19 g32 policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`、state SHA `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`，parent input SHA `61829b5a4a94ff1c468cb2a8ab055b6412415262f89ade97a75376dc32315b8e`，恢复原Adam，不继承退化control或smoke。三个新处理run配对seed2026102661/62/63，复用R16全部同seed顺序控制且冻结终点SHA，u0参数/Adam/双流env/aux/RNG必须逐值exact；属于配对修复筛选，不是三新独立seed确认。共同PPO+KL控制、未训练父同时保留，不择峰/不晋升。

每run固定32更新、524288环境步（PPO/辅助各262144）、1024优化步，共1572864/3072。u1..16 DAgger.2+combat KL.05，Critic参数与mu/nu冻结；u17..32 fresh on-policy PPO+KL，DAgger严格0。DAgger fresh学生与history各半，不用离线数据冒充PPO。每run实际采用PPO环境步131072、DAgger优化步512、PPO优化步512；不声称等累计分量剂量。其他同R16：64env×128step、lr7.5e−5、γ.995/λ.95、clip.1/value.5/entropy.005/globalclip.5、1epoch/minibatch256、自然75%/真实TRAIN Bank25% reset、fixed JAX rule3；current/ring/fixed=0/0/100%。无人工预携包，真实库存/掉包/接触来源分列，胶不开放原生窗口。

全新run/state/cache/eval namespace `runs/dagger_rl_20261025/r17_goal_kind`；每更新完整原子保存，声明0/1/8/16/17/24/32。dev64仍seed2026102480，u16/u32；holdout64仍2026111800，明确已在R16使用，只作固定配对比较，不作独立新保留或调参。真实JS Hunter难/中/易各64、Python-host native JAX Tactical64独立；Bank/P各32真实来源局×8续跑，仍旧map806/P0全far，报告局限。消费完整自然漏斗、真实道具链、固定game0首次几何分歧及原生一步排序、实际训练奖励/GAE、同输入拟合、来源局聚类Bank、尾部/最差seed和战斗安全。新增被动门控计数按双流×来源×阶段拆分；query tick不等于历史replay采用量或独立有益机会。

29 focused检查通过；原kind RED保留。GPU技术smoke仅两损失分支finite/真实Actor-ability非零、DAgger Critic/moments冻结、PPO Critic非零、完整双流fresh下一更新hash exact；3072执行环境步/12优化步、0正式采用，不设能力门禁。持久只读handoff等R16固定预算/Bank、CPU及补充证据、19固定录像消费、owner退出及8卡释放后选择并串行启动唯一R17 owner，CPU/JS、拟合和终态通知独立。没有报告/网页阻塞健康run。

后继规则：只有相对未训练父和匹配控制可信的完整自然整链及战斗保持，才另预注册五新配对seed确认；否则按固定真实首偏航/动作链/机会覆盖定位一个有界最小问题。不得盲加倍率、混奖励/胶目标、无关占卡或扩大预算。需证据才能决定时明确报告边界，不静默退出。USER_STOP优先且原子保存；旧O5/continuous/储备暂停hash不变；不push不发布。

复现（已有owner时勿重复）：`<PYTHON> -m scripts.dagger_rl_r17_handoff`；实际唯一训练入口`<PYTHON> -m scripts.dagger_rl_r17_run`；独立CPU/JS ` <PYTHON> -m scripts.dagger_rl_r17_cpu`。预注册/授权/匹配控制/科学选择/封存源及完成marker均为持久本地证据。

R16固定run终态 `sequential_anchor_s2026102663/u032`：原子state SHA `a5651c70c26e8f15825de94c4e947d2bd0a396bb0bcb9897f9741f87976068f9`，524288环境步/1024优化步，实际采用PPO环境步131072、监督优化步512、Critic更新步512。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

R16固定run终态 `joint_anchor_s2026102663/u032`：原子state SHA `04f8e50b4df05ef31f83f3ee22134f368841ef096748b482f8c4403619b6f454`，524288环境步/1024优化步，实际采用PPO环境步262144、监督优化步1024、Critic更新步1024。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

只读原生移动机制补证（不改R16/R17）：`bun_safety`实际导入旧`bun_env/jax_env`，运行env为`je07_bun_env/je05_core_env`；`_candidate_position`调用`_steer`仅传movement scale，而原生step传`core.spd_g * movement_scale`。322条已冻结TRAIN端点（非322独立局）的旧安全候选距离213条不一致，最大误差0.80740格、最优集合311/322一致。改用当前原生env的公开状态速度/slide/碰撞公式只读投影，同322端点×5方向与原完整env.step的距离逐值相同（max error0、322/322 best-set exact），无需敌人sample动作、未来RNG或模型输出。两版原始结果/协议均保留，不能覆盖旧差异或宣称整个旧安全老师正确。

该核验0新增完整环境transition/优化/采用步；每版1610公开位置投影，重复端点与既有原生分支不算新独立数据。它证明候选几何可用原生公式重现；未验证ability1/2、全时序safe标签是否变化，也不是Q/未来安全或自然提升证据。R17仅kind资格修复仍严格不混移动投影或安全重绑定；后续若要检验投影，须另TDD/预注册单变量并保留匹配控制。旧安全预测“原生准确”的泛称撤回，现称已有时序安全启发式。

R16固定CPU/生产JS终态已齐（GPU Bank/Bank-P仍在执行）：自然holdout各64局，PPO交付17/17/12、死亡22/22/16；顺序交付10/16/18、死亡17/19/11；联合交付14/13/2、死亡19/7/7；父点交付15、死亡25。顺序对PPO平均−1.0417pp，三seed×来源局描述95%CI[−15.625,+14.0625]pp；联合对PPO−8.8542pp，CI[−19.7917,+3.125]pp。顺序对父均差−0.5208pp、PPO对父+0.5208pp；均非可信整体改善证据，不宣布顺序胜出、不做五seed确认或晋升。PPO未显示此前共同control那样的严重自然均值退化，但不能由此证明安全等价。完整Hunter难中易各64、Python-host native JAX Tactical64及阶段/来源分母已封存到`FROZEN_CPU_COMPARISON_20261025.json`；Bank未完成前不标完整科学终态。

现有生产JS道具统计v2另有完整性限制：`ownedSlide`从接触帧计时，不扣放置/离开/重接触准备；滑行中再次接触自己香蕉时对象覆盖，未单独结算上一条链；没有显式终局未完成flush/censor计数。contact按接触阶段、完成按放置阶段归属，不能把某阶段contact/complete直接作同cohort转换率。`ITEM_METRIC_COMPLETENESS_CAVEAT.json`保存只读源码证据；0新增环境/优化/采用步，不改当前冻结统计或训练。此后仅称已结算接触至滑行片段proxy，不能证明完整释放链净收益/Q；complete=0也不等于所有contact均有害。自然交付/死亡/窗口库存主指标不受这条统计缺失直接改变；候选不由该proxy晋升。

R16完整科学终态：63声明原子点、288正式更新及Adam/Actor导出逐点验收，Critic仅在顺序臂前16更新冻结（合计48 outer更新），其余真实非零；正式4718592环境步/9216优化步，实际采用fresh PPO环境步1966080、监督优化步4608，不能称等累计损失剂量。全部CPU/JS、105组固定Bank/Bank-P、19条dev原生Review及来源局聚类补证已消费。Bank合计7313320有效评估环境步，含不同模型与固定重复，不计作训练或新独立来源局。

FINAL_COMPARISON SHA `65ab2a8810d2dc6ba45277fcfdcc13f1ef4414009588ce7a434c5d0f318a9c21`；SUPPLEMENTAL SHA `9a6d46ae2998602369d640cc0826e4b377dc9aa223b41cd99b0d87176b7c9d5d`。自然holdout PPO17/17/12、顺序10/16/18、联合14/13/2、父15（各64）；无顺序优势证据、不做五seed确认、不晋升。Tactical64安全胜PPO12/13/21、顺序8/12/7、联合4/8/3、父10。JS Hunter难/中/易各64：PPO[0,2,4]/[2,1,6]/[0,3,6]；顺序[2,1,15]/[0,0,6]/[0,3,7]；联合[0,1,9]/[0,2,2]/[0,3,17]；父[1,2,6]。安全胜、死亡与主动行为分开，低死亡不自动等于安全能力改善。

旧单图map806/P0/allfar Bank原包直接交付PPO[.980469,.921875,.964844]、顺序[.929688,.929688,.949219]、联合[.910156,.960938,.949219]、父.824219。顺序对PPO−1.9531pp，32来源局×seed描述95%CI[−7.4219,+2.9948]pp；原条件返程较父提高不等于自然整链改善。Bank-P与完整曲线、真实训练reward/GAE、固定录像零Value塑形和原生一步排序均分列，不互相冒充。原老师kind0监督确有采用，因此R16只解释为原共同老师配方下的比较，不外推为资格正确时的顺序因果结论。

R16训练/评估owner均正常结束后，现场捕获8卡显存0MiB、利用率0%、无compute app，`PREDECESSOR_GPU_RELEASE_OBSERVED.json`已存入新namespace。完整证据后的科学选择只激活此前冻结R17 kind资格修复；匹配控制三个u32及u0 SHA均冻结，441封存源未变。R17唯一owner已进入技术smoke，CPU/JS/拟合/终态owner独立持久；报告未阻塞健康训练，旧暂停SHA原样保留，无push/发布。

R17技术smoke终态：3072实际环境步/12优化步、0正式采用；DAgger Actor ΔL2=.1289656、Critic0且mu/nu精确冻结、ability Δ=.002102325、ability原始梯度norm7.182209；PPO Actor .1017501、Critic .07971073、ability .001242140、ability原始梯度norm1.155881，均finite。双流完整状态/奖励aux/RNG/history/params/Adam在独立新进程的两分支下一更新hash exact：Dagger `ae8064c651538fa5061c6d4c72531bbe37486975fb65a2d52d964adecbb84ef7`、PPO `e17321d0187e4605f328f3c4f0e3fbeaf71eebbadd285507afb2ec04bb4c2bb0`。smoke保存state SHA `e01d2cdd6b6a28cdaf949509e67e6228929a07bb9afb601a329760c0c374e9ae`已另行独立复核；此为技术有效性，不是能力门禁。正式恢复仍从原共同干净父，不继承smoke。

R17技术接续记录：首个正式尝试在u1原子保存后发现动态exec先复制旧commit/collector函数，再运行bind；u1训练记录无`goal_guard_telemetry`和matched-initial收据，故不能确认两学生流/持久计数满足冻结语义。u2前授权hold安全停止；u1 SHA `e32c62ad4209a47e51e52aef2956f3f9015225e0e157b742994823b29817cf93`标记EXCLUDED_INVALID，禁止恢复/用于能力结论，其他atom与log保留。R17技术smoke自身通过但不属于formal能力数据。attempt02协议在prelaunch测试准备阶段建立后即废弃，无smoke/formal步。

修复为formal进入动态retention loop之前执行bind，使scope捕获已经包装的PPO collector、auxiliary collector及commit；inner bind保持幂等。新增双流/telemetry与绑定先后回归，focused32项GREEN。attempt03独立namespace/state/cache/eval，完全同R17共同父/变量/seed/预算；R16完整证据、匹配控制SHA和旧暂停hash不变，source seal重新生成，USER_STOP最高优先且无旧队列恢复。attempt03新owner已启动自己的技术smoke，健康正式run仅在fresh独立restore和matched初态通过后开始。

R17固定run终态 `sequential_anchor_s2026102661/u032`：原子state SHA `2a30f40541f0f49190c8712a9eb4585937dcf424d3b1b98657f5be5750c2eb29`，524288环境步/1024优化步，实际采用PPO环境步131072、监督优化步512、Critic更新步512。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

R17固定run终态 `sequential_anchor_s2026102662/u032`：原子state SHA `c05e364c81c3b6d703f05a28b4319130328b161bb083db98039693ee1c7b266e`，524288环境步/1024优化步，实际采用PPO环境步131072、监督优化步512、Critic更新步512。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

R17固定run终态 `sequential_anchor_s2026102663/u032`：原子state SHA `86472af688cd5ba6c10537163028ef8b0b466eb76a45ed7043db17dc007809c5`、policy SHA `5c5893177de8940a8ad09fd60edeba3f72439640cec549d707761ab87d6cfc79`，524288环境步/1024优化步，32/32 finite；前16更新为DAgger（Critic与Adam moments冻结，ability梯度/更新非零），后16为新鲜on-policy PPO（DAgger=0，Actor/Critic更新非零）。attempt03三配对run均已完成，合计1572864环境步/3072优化步；最后模型CPU/JS与native Bank/P仍在独立消费，尚未作能力晋升结论；旧暂停队列未恢复。

R17固定run终态 `sequential_anchor_s2026102663/u032`：原子state SHA `86472af688cd5ba6c10537163028ef8b0b466eb76a45ed7043db17dc007809c5`，524288环境步/1024优化步，实际采用PPO环境步131072、监督优化步512、Critic更新步512。32更新finite/Actor非零，Critic冻结和DAgger0按阶段验证；原owner接续冻结队列，CPU异步，旧暂停保留。

R17固定三配对终态（有效目标kind门控，控制复用R16，非独立新seed确认）：
| seed | 新处理holdout交付/64 | 原顺序对照/64 | 新处理死亡对局 | dev尾部 |
|---|---|---|---|---|
| 2026102661 | 16 | 10 | 30 | 19.5 |
| 2026102662 | 15 | 16 | 27 | 15.0 |
| 2026102663 | 9 | 18 | 20 | 10.0 |

父点holdout交付15/64；保留、完整曲线、真实道具链与query门控计数见FINAL_COMPARISON，不能对退化control择峰宣称整体改善。旧暂停保留，无发布。

## DAGGER-RL-20261025-R18-ITEM-CHAIN（预注册/执行）

- **目的与单变量**：在同一干净父点、奖励、对手、课程、网络、动作 mask、optimizer、学习率、环境/优化步和 Bank 协议下，修正路线老师语义：没有认证的完整“释放→离开→接触→结算”item plan 时，从 route set 只移除 ability=2 释放；认证 item plan 仍由 item set 完整接管。奖励、香蕉/胶目标、战斗处理与采样语义不变；不使用人工携包。
- **父点**：`runs/dagger_rl_20261025/r16_retention_schedule/formal/no_update/u000`，policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`，state/input SHA `61829b5a4a94ff1c468cb2a8ab055b6412415262f89ade97a75376dc32315b8e`。R17 `attempt03` 首更新已标记 `EXCLUDED_INVALID`，不恢复、不纳入能力结论。
- **配对设计**：`ppo_anchor`（PPO-only control）、`sequential_anchor`（u1–16 DAgger，u17–32 fresh on-policy PPO）、`joint_anchor`（PPO+DAgger）；seed `2026102661/62/63`，每臂32 updates，每run 524288环境步/1024 optimizer步，九run总计4718592/9216。保留同父不更新control、原父点和每个固定点，不择峰。
- **阶段/评估**：固定点 `u0/1/8/16/17/24/32`；CPU/JS唯一owner异步消费自然dev/holdout、Bank/Bank-P、Python Tactical v2、真实 JS Hunter 难/中/易各64，拆分来源/阶段/库存/合法释放/认证链/采样/greedy/接触/结算/死亡/掉包/交付与战斗安全。current/ring/fixed 对手占比按实际记录。
- **TDD与动作链审计**：旧 `dagger_rl_oracle.combine` RED 复现为未认证行仍保留2个释放标签；adapter focused GREEN 20项相关测试通过，含 R18 直接/辅助 teacher wiring、零计划/认证计划优先级。只读 Bank 审计记录：train Bank 32状态中旧释放行29、认证item行7，修正后释放行7、移除22标签；R17排除状态另列，capability evidence excluded。`ITEM_CHAIN_AUDIT.json`、`TDD_RED.log`、`TDD_GREEN.log`与 source seal 保存。
- **技术smoke/资源**：仅检查导入、focused TDD、finite、Actor/Critic/ability真实非零更新、DAgger Critic/moments冻结、完整双流state/aux/RNG/Adam/history fresh restore/原子hash；不设置能力门禁。旧O5/continuous/储备保持暂停，不push、不发布。
- **接续规则**：健康run固定预算完整执行；每个checkpoint异步评估不阻塞GPU。u32原子保存后按预声明顺序接续下一臂/下一seed；所有九run和Bank/P完成后再按完整曲线、首偏航和动作链证据决定一个新的有界后继，不因中点或单seed停训。


### PPO-SCALE-LONGRUN-20261025-R1（20261025）

用户优先级切换后登记的独立 PPO 长程与 batch 扩容实验。R18 item-chain namespace 在正式阶段启动前仅有技术 smoke（formal environment/optimizer steps=0），已写入 `runs/dagger_rl_20261025/r18_item_chain/PPO_SCALE_HANDOFF.json` 和持久 `USER_STOP.json`；R18 smoke、失败恢复收据与旧 O5/continuous/储备暂停均保留，正式队列不恢复。

共同起点选择干净 r19 g32 父点（policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`，state SHA `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`，原 Adam/input SHA 由协议封存）。R16 已完成 PPO anchor 虽有配对观察但未形成可靠整体提升，故不择峰继承单个退化/DAgger 候选；本轮以共同干净父做匹配起点，保留 R16 奖励、输入、动作 mask、网络、对手和 Adam 语义。

唯一变量是 PPO fresh categorical on-policy rollout/global minibatch 规模；DAgger=0，老师仅记录机会分母，不进入 loss，离线行为数据不冒充 PPO。`ppo_base_batch` 为 8 env/device、128 rollout、8192 global rollout、256 global minibatch（local32）；`ppo_scaled_batch` 为 16 env/device、128 rollout、16384 global rollout、512 global minibatch（local64）。两臂每 run 1,048,576 环境步，base 128 updates、scaled 64 updates，三配对 seed `2026102751/52/53`，共六 run、6,291,456 环境步、18,432 optimizer steps；checkpoint 按环境步保存 0/1/16/32/64/96/128（base）及 0/1/8/16/32/48/64（scaled）。不线性放大学习率或 rollout 时长。

准入只核技术/恢复、真实动作采样支持、数值稳定性和无明显退化；短点评估是观察指标，不是停训门禁。审计先用真实学生 categorical 轨迹确认释放/库存/接触机会，部署 greedy 单列；正式 smoke 只验 finite、Actor/Critic/ability 非零、fresh 恢复和原子保存。CPU/JS evaluator 独立跟随声明 checkpoint，不阻塞 GPU；自然完整、Tactical、Hunter 难中易、Bank/P、机会分母、尾部/最差点和对手实际暴露分开报告，不择峰、不发布。

协议与源码封存：`runs/ppo_scale_20261025/r1_longrun/PROTOCOL.json`、`SOURCE_SEAL.json`、`PREREGISTRATION.json`。

PPO R1 执行补充（20261025）：首seed base u8/scaled u4各65,536环境步作为有界吞吐prefix，直接计入各自1,048,576步长预算，随后自动恢复完成六run；不设置能力门槛。lr=7.5e-5、epochs1、gamma=.995/lambda=.95、clip=.1、value=.5、entropy=.005、gradclip=.5，combat KL=0。base u8已原子完成，稳态约1484–1543环境步/秒；扩容正在编译，其吞吐另测，不把编译或显存当有效训练。

首formal接线早于fresh比较，已在base u1原子边界暂停修复并保留，未重跑u1。最终两独立新进程的两batch下一步env/aux/RNG/batch/history/loss/params/Adam全部exact，通过后从u1继续至u8；当前验收为FRESH_RESTORE与TECHNICAL_ACCEPTANCE，旧load-only precheck不算恢复验收。u2 state SHA `3bd0bfaf2c4c656e72dd40ec51796f36b08505f4df759a4bfe0fffc2c733c603`，finite、Actor/Critic/ability真实非零。

动作审计实际65,536步（原声明8192已透明勘误），释放972、机会427、contact label0，仅证明可采样而非完整有益链；glue_inventory=0未采集，不作零观测。v3被动测量已修准备/重叠/终局计时，另发现交付后cargo消失误作丢包，v4隔离修正并通过真实拾包全链/重叠/终局/physics-RNG parity测试；旧v3收益字段不作结论。核心CPU/JS唯一owner已消费父/u1/u8，v4同dev64 categorical与greedy审计并行，不改训练、不增老师。详细设计、技术历史和复现入口见 `reports/ppo_scale_longrun_20261025.md`。

PPO R1等步数pilot终态：base u8/scaled u4各65,536环境步，全部finite且Actor/Critic/ability非零；已自动接续base u9–128，无评估门禁。迭代内steady中位1521.6/2622.0步/秒，含原子保存wall-clock中位1146.2/2067.0步/秒；scaled前两更新含编译274.1/251.8秒，采样窗口小，不宣称整段加速保证。共同父v4同dev64 categorical自然持包中，香蕉36释放/7自有接触/1完整正净值，胶23释放/14归因敌接触/1减速且向家推进；支持稀少，不等同Q或稳定技能，greedy另列。正式长预算及三seed对照不变。smoke元数据计数勘误为base512/scaled1024真实环境步、各2optimizer步，不重复smoke；训练EV未单独采集，value loss/GAE不冒充EV。唯一owner/worker/CPU evaluator核验为1/1/1，只读monitor不启动训练；每日报告持续更新。

PPO R1测量归因撤回：此前v4计1条完整正净值香蕉链，原始事件显示新放置在旧滑行中，故撤回独立完整有益链结论。v5只读隔离修复保留旧活跃链、排除新事件的既有本人移动效果污染；不改奖励/动作/老师/训练配方。胶减速推进仅观察proxy，可能重叠旧敌方减速，不作为Q或独立收益。真实释放、自有接触支持仍成立，道具不足不阻塞本次已授权PPO。固定父点64 categorical/greedy补审计与CPU固定评估独立进行。

v5隔离补审计终态：同64固定seed categorical/greedy合计128局的非道具指标与v4 exact，两个固定首seed完整trace exact（其余126局无完整trace声明）；保留旧活跃链后，categorical真实持包香蕉36释放/7接触/2条完整正距离净值链（seed2026102480/2502，+1.5601/+5.1202格，含准备成本、放置时无既有本人移动效果）。greedy7释放/1接触/0正净值；胶仍仅观察proxy。v4独立链结论不恢复，不混加不同版本。PPO配置与长预算不变。

PPO R1首完整长训终态：base/s2026102751 u128，1,048,576环境步/4096优化步，128更新finite/Actor-Critic-ability非零；state SHA `06d47ce4f3e28c9fcf3421ce6df9bba4bba95f292c0caac54f1c1883467b0b61`、policy SHA `f7380f576632ccc51790835d7ecf1a8939cfd5fcb66e5d7318c12160e1399037`已独立核验并主动通知。无需CPU门禁，scaled从u4恢复至u5，finite/非零；队列继续scaled u64和seed52/53四run。u128 CPU评估异步，未宣称能力晋升；完整曲线、固定尾部/最差点均在自动比较报告中消费。

PPO R1首完整base最终dev下降已主动披露：自然交付2/64对父12/64，尾部13/7/2，不择u32峰值；categorical15/64对父26，死亡降低不冒充安全提高。窗口84→47，库存拾取/窗口21/84→5/47，同链交付/拾取12/21→1/5；技术全状态finite且Adam真实计数+4096。只有一个base seed完成，固定scaled和其余seed继续，不因该中间单seed结论停健康run。证据排序假设和PPO最小学习率后继问题写FIRST_LONGRUN_DEV_DIAGNOSIS，不改当前配方、不追加DAgger、无晋升。

### PPO-SCALE-LONGRUN-20261025-R2（20261025，长程续训）

用户已明确要求以长期观察替代百万步短点判断。本实验从 R1 三个已完成 `ppo_scaled_batch/u064` 原子状态逐 seed 完整续训，不重跑 R1、不继承峰值、不恢复 O5/continuous/储备队列。合同与授权收据：`runs/ppo_scale_20261025/r2_longrun/PROTOCOL.json`、`PREREGISTRATION.json`、`SCOPED_GPU_AUTHORIZATION.json`、`SOURCE_MANIFEST.json`、`SOURCE_SEAL.json`。

共同配方保持 R1 scaled：8 GPU、16 env/device、128 rollout、全局 rollout batch 16384、全局 minibatch 512（local 64）、每更新 32 个 optimizer steps、Adam 学习率 7.5e-5、epochs 1、gamma=.995、GAE lambda=.95、clip=.1、value=.5、entropy=.005、gradient clip=.5；DAgger=0、combat KL=0，奖励、网络、输入、mask、固定 JAX rule3 对手不变。三个配对 seed 为 2026102751/52/53；各自源点是 R1 u064，源状态包含 params、完整 Adam、双流 states/aux/keys、history，源 state/policy SHA 逐一保存在 `SOURCE_MANIFEST.json`。

真正长程预算固定为每 seed 2048 个新 update = 33,554,432 个新增 on-policy 环境步、65,536 个新增 optimizer steps，三 seed 合计 100,663,296 新环境步。按 R1 已测约 2622 env-step/s，估算每 seed约3.55小时、总计约10.66小时；该估算不替代实际吞吐，禁止默默降回百万步。每 64 新 update（1,048,576 新环境步）排队一次16局轻量 production-JS natural dev，每512 update做完整 natural64、Python Tactical v2 64、production JS Hunter 难/中/易各64，最终点另做独立自然 holdout64；CPU/JS evaluator 仅消费已提交 checkpoint，不控制或阻塞 GPU，holdout 不调参。checkpoint 队列和完成 marker 在同一 namespace 内持久保存。

技术检查仅验证导入/focused TDD、完整源状态恢复、finite、Actor/Critic/ability真实非零更新、原子保存与新进程恢复；smoke 不进入 formal 计数，不设能力门禁。健康长训不因短期能力波动、seed差异或评估未齐停止；仅技术故障、用户暂停或可靠严重安全异常可安全保存处理。每个 seed 固定终态主动通知，旧队列保持暂停。

执行入口：`<PYTHON> -m scripts.ppo_scale_longrun_continue_run`；独立评估：`<PYTHON> -m scripts.ppo_scale_longrun_continue_cpu`。当前长程 namespace 不含 run/checkpoint 提交文件，R1 封存结果保持只读。

### EXPERT-VALIDATION-REWARD-SIMPLIFICATION-20261025（专家指导的对齐诊断与最小奖励包候选）

独立协议 `runs/expert_validation_20261025/PROTOCOL.json`，运行时勘误见 `EXECUTION_AMENDMENTS.json`，总结见 `reports/expert_validation_reward_simplification_20261025.md`。独立目的为先测当前奖励与自然能力关系、验证 JS 与 JAX 动力学可比范围，再等资源边界可用后执行已预注册 D 两臂奖励包对照；不混入正在执行的 R2 PPO 配方，不恢复 O5/储备、不push不发布。

A 有效执行采用注册8局×64 tick、固定配对 seed 2026102860、训练原生JAX transition、teacher关闭，共512个转移/模型。父R1 u064与R2 u256在这短窗均没有控制/击杀/取包/死亡/道具结算；父点分项total 3.5，R2 u256 total 4.0，回报主要为固定任务项，路线与道具净值皆0。这不足以证明奖励错配，也不具备自然整局能力判断力。早先4×2400、seed2026102480输出与协议不匹配，已保留为 `A_reward_alignment_legacy_mismatched_protocol.json` 并排除；正确结果是 `A_reward_alignment.json`。

B 是parent、candidate（R1 scaled seed2751/u064）、线上default × categorical/greedy × 50局，生产JS natural full2400 + normal Hunter，老师关闭，actor/opponent RNG分离、种子配对。经过生产 `Player.verify()` 的6条独立固定录像已写入 `runs/current_best_model_review_20261025/`；小矩阵里线上seed2481成功取敌笼库存包并交付，另5条固定局无交付，样本不足不晋升candidate。当前专家验证 B 唯一CPU owner已完成3/6行、150/300局：parent categorical 13/50交付、27/50局发生死亡；parent greedy 8/50交付、16/50局死亡；candidate categorical 25/50交付、33/50局死亡。单行对照不得脱离配对seed与死亡/战斗保持宣称candidate胜出。其余3行继续异步运行。早期缺ONNX绑定尝试不纳入矩阵；修正副本与注册模型的policy tensors逐项一致，ONNX由生产loader校验，修正数据使用 `B_js/verified_v2/`。

C 固定地图806初态和动作的JAX `je07_bun_env`/生产JS逐tick状态、mask、observation witness正在CPU执行；当前覆盖移动路径，不含泡、道具或事件随机分支。只有完成后才能报告可比字段/容差和首处差异；不把 teacher movement predictor 差异替换成物理规则失配结论。

D 两臂仍仅预注册未启动：同共同父R1 u064、旧奖励对最小奖励包、配对seed 2026102851/52/53、每臂每seed 5,242,880真实on-policy步，DAgger=0。共同网络/输入/mask/对手/课程/优化器/gamma与terminal/phase offset固定；塑形最多保留一项严格PBRS，去除请求型胶负成本和非势能多层塑形。因R2长程仍占用8卡，不启动D，也不停止/减速R2；A/B/C为CPU异步诊断。

**D预formal实现复核更正（20261025）**：formal updates仍为0。collector ABI 审计发现 item 分项必须严格采用 `combat_item_events.METRICS`，并确认冻结输出 route PBRS 的系数为 `je10_ops.ROUTE_COEFFICIENT=0.5`（gamma=.995，终局next potential为0）；原预注册说明中的0.05已由带旧/新hash的 `PROTOCOL_AMENDMENT_20261025.json` 更正。适配器只读取 emitted `route_shaped_reward`，不二次缩放；无部分可观测策略不变性声明。另修正短组件tensor宽度校验以及持久GPU/CPU owner锁句柄。focused测试9项、相关3个测试文件17项、合计26项GREEN，零系数回退精确；计数收据为D namespace的`TEST_VERIFICATION_20261025.json`。唯一D owner启动条件仍是R2三seed固定预算完成和8卡释放；不影响现行R2或旧暂停队列。

**A分项schema更正（20261025）**：首个 A 输出将90列 collector tensor误配83个标签，不能用于分项归因；旧 JSON 原样归档为 `runs/expert_validation_20261025/A_reward_alignment_COMPONENT_SCHEMA_MISMATCH.json`，中间只修列名的文件 `A_reward_alignment_schema90_without_per_episode.json` 也排除。按同一注册seed/初态/RNG重跑，最终 `A_reward_alignment.json` 已升级 schema v2，90列完全对应 `je10_ops.COMPONENTS[:-1] + combat_item_events.METRICS + combined_total`，并附每局账本/阶段动作与 artificial truncation 标志；父/R2-u256短窗总奖励仍分别3.5/4.0，均无控制、取包、死亡或道具事件。最终SHA `ccf629c5c7e7174255748e4789638a21d34adeed63c013b20f57ec1bd26af028`、执行更正和测试证据已写入报告与 `runs/expert_validation_20261025/EXECUTION_AMENDMENTS.json`，不控制GPU、不改变D协议。

当前监督快照：R2 seed2751 new_update=1209/2048，19,808,256/33,554,432新增环境步，finite；最近原子点u1152的Actor/Critic/ability update L2约0.734/1.451/0.0125，是真实非零更新。R2 owner/worker/CPU evaluator各1，八卡仍在用；D owner持独占锁等待，未申请GPU、未启动smoke。

后续现场复核：R2 seed2751 new_update=1276/2048，20,905,984/33,554,432新增环境步，finite，最近原子点u1216；R2唯一worker与CPU evaluator仍在运行，D唯一owner持锁等待，未申请GPU或启动formal。

现场R2快照（20261025续）：seed2751 new_update=976 / 2048，新增15,990,784 / 33,554,432环境步，finite，最近原子点u960；唯一owner/worker/evaluator各1，八卡持续占用。D仍无GPU run，尚未作能力比较。详见 `reports/expert_validation_reward_simplification_20261025.md`。


奖励简化验证执行更正（20261025，覆盖本条早先“B执行中/C执行中”的状态描述，不改协议或旧原始文件）：

- B 已按六行完整完成：三模型（R1共同父、R1 seed2751/u064候选、线上默认）× categorical/greedy，各50局，共300局；固定配对seed 2026102860–2026102909、production JS natural full2400 + normal Hunter、teacher关闭。汇总 `runs/expert_validation_20261025/B_js/verified_v2/SUMMARY.json` SHA256 `52fa5c47ced9873283afd4214f0b26591825c6e8faa03c9f63e6985ee40da33d`。候选交付 categorical 25/50、greedy 14/50；共同父13/50、8/50；线上9/50、5/50。候选相对父的配对交付差为 categorical +0.24（95%区间[+0.04,+0.44]）、greedy +0.12（[-0.04,+0.28]）；相对线上为+0.32（[+0.14,+0.48]）与+0.18（[+0.04,+0.32]）。区间仅反映固定game-seed不确定性，candidate只有一个训练seed，不代表跨训练seed稳定提升，网页默认保持不变。候选窗口库存拾取35次，其中敌死亡33、原生糖浆2；另有11次自由存活/过期窗口拾取。道具链结果是机会与已观测proxy，不作为动作价值或因果收益。
- B汇总器RED/GREEN：RED发现不应要求重复source SHA字段、旧配对统计将双输误记为同赢且缺候选对共同父区间；实现与测试 `scripts/expert_validation_summarize.py`、`tests/test_expert_validation_summarize.py` GREEN，最终固定输出在上列verified_v2 summary。
- C 已限定范围完成：map806、配对seed 2026102860、48步纯移动、49状态点；JAX与生产JS在测量的位置、环境状态、墙/砖/箱/泡/道具hash、mask及observation无字段差异；最大位置误差1.1658e-6、位置差分proxy误差3.4333e-7、observation误差1.1921e-6，容差2e-5。证据 `runs/expert_validation_20261025/C_dynamics_parity.json` SHA256 `6e5e5ebad3d1d5dce25addf819b350ecceea5b80c75b17d92539f076a38196b0`。未覆盖放泡、道具、死亡、随机事件/RNG分支；不泛化为整个动力学一致。
- 当次监督快照：R2 owner/worker/CPU evaluator各1，B evaluator已释放；R2 seed2751达到new_update 754、12,353,536新增环境步、finite，最近原子提交u704，Actor/Critic/ability更新非零。8卡仍有训练进程及显存占用；瞬时利用率采样会落在更新间隙，结合连续更新/吞吐而非单个util读数判断。R2另外两个seed与固定预算未完成前D仍不启动。

**最终复核更正（20261025；覆盖本条前述过程快照）**：A 固定同 key/env RNG 将 8×64 tick 延至 8×128 tick，父R1 u064与R2-u256各有1,024有效转移；两边均0原生终局、8/8人工截断、0事件且无V bootstrap。P0有限窗未折扣均值分别0.5625与0.6875，仍不能证明V准确或整局回报。逐局证据、截断敏感性和边界解释见 `runs/expert_validation_20261025/A_horizon_parent_128.json`、`A_horizon_r2_u256_128.json`、`A_horizon_sensitivity.json` 与 `A_HORIZON_SENSITIVITY_RECEIPT.json`；汇总SHA为`850b530bd3b76f6f3424d7e734ad1e2408c05f53ac00be9ada8f73e444b3c03b`。诊断/summary focused 5项、D/道具/Bank/PPO相关26项通过；Python compile与diff-check通过。复验命令需用`.venv` Python加`PYTHONPATH=.`。

R2唯一GPU owner当前 seed2751/new_update=1646/2048，新增26,968,064/33,554,432环境步且finite；最新原子u1600 state SHA `dfb973861db29c3a212d9e3f23d9ea333187ba595631e821fd76bc4a34d35dd0`、policy SHA `509e3c31f92111f490a356cfa8a167d84c6bf87b6b434f642f5b532cf87e25b7`，该提交Actor/Critic/ability update L2=`0.7072/1.5060/0.00600`，新增51200 optimizer steps，采样147次item ability及258个item释放机会。固定 batch 16384/minibatch512、lr 7.5e-5、DAgger/KL=0、JAX rule3不变。CPU owner唯一且异步：已消费u1600 light与u1536 full；该单seed中点自然64局0交付、1局自由存活取库存包、11次死亡、返程56/119 oracle匹配；Tactical安全胜2/64；Hunter hard/normal/easy安全胜0/0/2、死亡局4/2/1。只作中点诊断，不作为因果或停止门槛。D唯一 owner持锁等待且GPU worker/formal updates仍为0，必须等R2三seed自然完成、完整原子核验和8卡显存/compute app释放后才可smoke/启动；旧O5/continuous/储备保持暂停。


### R2 用户提前终止与 D1 奖励对照接续（20261025）

用户覆盖 R2 三 seed 长预算继续要求后，现场唯一 R2 owner/worker 已退出；没有恢复剩余 seed/chunk。R2 标记为 `USER_ENDED_EARLY`，固定预算完成为 false。seed `2026102751` 观测到 new_update=1706、27,951,104 新环境步，其中最近可验证原子提交为 new_update=1664、27,262,976 新环境步；其余 42 update/688,128 步没有原子提交，不能纳入保存结论。该 seed 的后续未执行预算为 342 update/5,603,328 步；seed `2026102752`、`2026102753` 各自 2,048 update/33,554,432 步全部未执行。u1664 state SHA `7ec08b98955c13ae3ece44bab72162bbc89dfd45b8b77366658021c54d18830e`、policy SHA `542cb650c0494eab0f8572076b60afd37db34e804faf674c9143ed2e3fcb5466`，state 已核验包含 params/Adam/states/aux/keys/history/seed/update/new_update；续训路径仅在用户再次授权后从该原子点恢复。R2 最终因 GPU rendezvous fatal 退出，但用户 early-stop 收据保留该事实；无 `TRAINING_COMPLETE.json`，controller 不得自动接续。CPU/JS 已提交点评估队列保留为非阻塞、未完成预算明确标注。

D1 `runs/expert_reward_20261025/D_ablation_v1` 采用新的 predecessor early-terminal handoff amendment：接受 R2 `USER_ENDED_EARLY.json` 仅作为安全终态，不把它解释为固定预算完成；要求 R2 owner exit、最近原子 checkpoint state/policy hash、完整 Adam/env/aux/RNG 字段和 8 卡释放。旧 D owner 的暂停收据归档保留，hash-bound `D_RESUME_AUTHORIZATION.json` 只允许该 namespace 的唯一 owner 接管；O5/continuous/reservoir 队列仍暂停。D 的 protocol/source seal/authorization/preregistration 已更新并记录 amendment hash，未修改父点、三配对 seed、两臂、预算或任何其他变量。

D smoke 与 fresh restore smoke 均通过，formal 计数保持 0。D formal 首个更新已真实落盘：`legacy/seed2026102851`, new_update=1，16,384 fresh on-policy 环境步、32 optimizer steps、finite；Actor Δ=0.6518849463、Critic Δ=1.2034321899、ability Δ=0.0126362424，首更新含编译吞吐 66.394 env/s，后续按固定 320 update/5,242,880 环境步合同继续。CPU/JS 评估 owner 独立跟随，不阻塞 GPU，不择峰、不因短点能力停止。

复现与验收：`PYTHONPATH=. .venv/bin/pytest -q tests/test_expert_reward_ablation.py tests/test_ppo_scale_longrun_continue.py tests/test_combat_item_experiment.py tests/test_combat_item_bank_alignment.py` 通过（新增 D handoff/zombie GPU entry 测试后 focused 11 项；相关总集仍通过），`.venv` py_compile 与 `git diff --check` 通过。

D1现场接续收据（20261025）：legacy/seed2026102851 已由唯一 owner 从固定 R1 u064 正式启动，首个 formal update finite 且 Actor/Critic/ability L2=0.651884946/1.203432190/0.012636242，16,384 fresh on-policy 环境步、32 optimizer steps；首更新记录吞吐 66.394013 env/s（含编译）。随后现场已推进至 new_update=18、294912 环境步，最近吞吐 1907.907924 env/s，仍 finite；固定预算与六 run 顺序未改变。

D1授权恢复续接（20261026）：本次用户授权由 `CONTINUATION_AUTHORIZATION_20261026.json` 消费，唯一 owner 从已提交 `legacy_s2026102851/u000064` 恢复；state/policy SHA 为 `5b22e0afeb633c5cfa2e4954184c6d6e0e06ccd72bb7c3410f5671ce3fa4136c` / `5ee2bae1a8ae38a7dfd41e1551c80b43fbae1b6d89d9144337ee602aa051d23c`。旧 owner 的未提交 new_update 65–110 全部跳过，不作为训练步或能力证据；smoke 与 fresh-restore 已通过且不重跑。恢复后的首个正式新更新为 new_update=1（16,384 fresh on-policy 环境步、32 optimizer 步，finite；Actor/Critic/ability L2=0.651884946/1.203432190/0.012636242），收据见 `CONTINUATION_RECEIPT_20261026.json`。现场观测已到 new_update=74、1,212,416 新环境步，最近吞吐约1,933.62 env步/s，仍 finite，最近原子提交仍为 new_update=64；CPU/JS evaluator 独立消费已提交点。R2 用户结束与 O5/continuous/reservoir 暂停保留，D 六 run 固定预算、两臂三配对 seed、DAgger/KL=0 及不择峰规则不变。

### 专家规划遗漏复核（20261025，只读，不改变 D）

读取 `.hermes/expert_plan_review_followups.md` 后完成源码、协议、真实运行证据核验。D 当前仍由唯一 GPU owner 运行，CPU/JS evaluator 独立跟随；本次没有修改 D、重启/重跑诊断或申请 GPU。详细报告：`reports/expert_plan_review_followups_20261025.md`。

- **奖励系数结论（P0）**：`scripts/je05_bun_transport.py` 的 native transport 路径固定 `CAPTURE_REWARD=24.0`。真实 native pickup→返程→capture witness 的实际结算为 `transport_capture=24`、总 reward=24，且 shaping、clawback、carry-home net 为零；capture 先进入 `pending_capture`，只有 `settled && !death/drop/trade` 才支付，token/captured ledger 去重，失败会清 pending 并按旧账本 clawback。D minimal 的 `scripts/expert_reward_ablation.py` 则直接计算 `info["capture"] * DELIVERY_REWARD`，其中 `DELIVERY_REWARD=2.0`，同时保留 kill +16、death −12、control entry/residual 与已缩放 strict PBRS。代码/协议只写“reuse existing retention event scales”，没有找到把 live +24 推导成 +2 的独立标定依据；因此 D 是完整 reward-package 对照（量级、结算时序、pending/clawback 语义均改变），不能描述为“仅删除塑形”。当前 D 协议冻结，不作追溯修改；若系数需修正，另建前瞻版本。
- **A 缺口（P0）**：`A_horizon_*_128.json` 仍是每模型 8×128 tick、1,024 转移，8/8 artificial truncation、无原生终局/事件/V bootstrap，不能回答自然整局奖励对齐。后继必须固定完整局、terminal/timeout/truncation 分母，逐局输出折扣/不折扣回报、exclusive terms、pending/clawback/settlement，并将训练奖励与 JS 近似回算分开。
- **B 样本口径（P0）**：B 的 300 局是 3 models × 2 modes × 50 局，即每条件 50 局；`SUMMARY.json` 的 `games_per_row=50` 与 paired CI 已确认。当前区间只反映 game-seed 不确定性，不反映 training-seed 方差；扩到约 300/condition 必须新 manifest，不回写旧结果。
- **D 双模式（P0）**：协议宣称 categorical/greedy 分开，但 light 明确只排 greedy，CPU protocol 没有把 mode 固定成每 checkpoint 的任务维度。后续需同 checkpoint、同 game seed、独立 actor/opponent RNG 的 categorical/greedy 双行矩阵；缺行即 incomplete，不择峰补齐。

**优先级与验收**：P0 先封存系数 witness/版本边界、登记自然完整 reward-alignment manifest、登记真实评估局数与 mode/engine/dev-holdout、补齐 D 同 checkpoint 双模式；验收要求终局/timeout/人工截断分列、pending/clawback 可复核、不能用总 reward 跨尺度排序。P1 再做阶段 entropy/KL、critic EV/return bias、terminal GAE、事件动力学 replay witness 与隐藏 reward state 审计；P2 才做自然 Bank 分层、item full-chain/no-item 因果对照和独立 batch 实验。当前 D、R2 early-stop、O5/continuous/reservoir 暂停均保持不变，任何系数修正必须新 namespace、显式版本和独立配对。

### 被动整局指标验收（20261026）

用户授权读取 `.hermes/detailed_metrics_acceptance.md` 后，新增 `scripts/passive_evaluation_metrics.py` 与对应回归测试；`scripts/expert_reward_alignment_fullmatch.py` 在 native transition 后附加纯读取 trace，不修改动作、RNG、物理、reward 或 state。报告与覆盖矩阵：`reports/detailed_metrics_acceptance_20261026.md`；执行 manifest、源码 SHA、技术预检和实时状态：`runs/expert_reward_20261025/D_ablation_v1/passive_metrics_acceptance_20261026/`。

技术预检用共同父 `R1 seed2751/u064` 与 `minimal seed2851/u064`，固定 categorical seed2026102680，各真实自然终局1局；有效转移1728/1046，实际 reward 分解最大误差分别 `2.79e-8` 与 `3.42e-8`，reward-to-total误差均为0。它仅验收数据通路，不进入正式样本。正式首组已用独立单 CPU invocation 启动：父/候选×categorical/greedy，每条件32个配对游戏seed、共128局，native JAX JE07固定 JAX rule3、自然出生、teacher关闭；不是 JS Hunter，也不估计训练seed方差。正式评估不控制 GPU；D唯一训练owner/已授权队列保持运行，R2/O5不恢复，D奖励协议不改。

focused 指标/整局测试原先12项通过，py_compile与diff-check通过。被动指标分为 implemented、recomputable、missing-data、unavailable，零机会分母为null，窗口/持包episode分别去重，未结算carry单列censor；详细单项与边界见报告。源评估没有动态拓扑/route oracle逐tick证据时，曼哈顿距离和移动长度只称proxy；item hit只称实际碰撞/减速proxy，不声称完整因果收益。

#### Attempt 1 失败定位与 Attempt 2 接续（20261026）

Attempt 1 实际只完成16/128局，之后评估进程退出；没有完整模式行落盘，因此16局不复用、不进入比较。原始manifest、进度和 `ATTEMPT01_FAILURE_RECEIPT.json` 保留，执行状态已从运行更正为 `FAILED_INCOMPLETE`。

同批复现定位到父点 categorical seed2026102696：第921个有效transition，`combined_total` 对 exclusive 清单误差0.2499997，`combined_total` 对实际reward误差0。原因是 potential ledger 实际支付的聚合 `potential_clawback` 未列入相加；`drop_approach_net`/`carry_home_net`只含净势能delta，而`drop_approach_clawback`和`carry_home_clawback`是该聚合项的分阶段诊断镜像。评估只读加总清单遗漏了真实reward分项，不是奖励实现或训练系数错误。TDD旧清单RED误差0.25；加入聚合`potential_clawback`、不重复计入phase镜像后GREEN。focused测试含`tests/test_passive_evaluation_metrics.py`、`tests/test_expert_reward_alignment_fullmatch.py`及`tests/test_expert_reward_ablation.py`共24 passed；py_compile、diff-check通过。奖励/训练源、D冻结系数和协议均未修改。

新评估唯一CPU owner已在独立 namespace `runs/expert_reward_20261025/D_ablation_v1/passive_metrics_acceptance_20261026/attempt_02` 启动：共同父 `parent_r1_s2751_u064` 对 D 候选 `minimal_s2851_u064`，32配对seed、categorical/greedy各32局，共128局；native JAX JE07固定JAX rule3、自然出生、teacher关闭。manifest SHA为`d1fe4cb1f1ff989475a3f0e7a53cd46584a016a4d4ab8851ab65334a44448432`。状态/进度写入新namespace；此CPU owner不控制GPU。D唯一训练owner和worker保持正常运行，R2/O5仍停止/暂停，无GPU训练更改。报告与指标覆盖矩阵见`reports/detailed_metrics_acceptance_20261026.md`。

#### 被动指标v2最终验收与首组配对整局（20261026）

Attempt 2完整结束但仅保留summary；之后RED定位到恢复自由tick窗口关闭顺序、接触转移取得包被当作同tick失包、进度行数误当局数、结算尾段winner覆盖原生首terminal。它们仅属被动评估/持久化错误；旧summary保留，受影响窗口/转移/winner不进入最终解释，训练reward/动作/物理/RNG、D冻结系数均未更改。

新隔离 `passive_metrics_acceptance_20261026/attempt_03` 已由唯一CPU诊断owner完成父 `parent_r1_s2751_u064` 对 `minimal_s2851_u064` 的32配对游戏seed×2动作模式、共128自然整局；0人工截断，耗时922.08秒。native JAX JE07 + 固定JAX rule3、layout0/P0、teacher off、diagnostic dev，不是JS Hunter/Tactical/独立holdout/跨训练seed结果。每批NPZ原子保存逐tick状态摘要/动作/概率/事件/实际reward/分项，逐局包含trace与state/policy SHA及地图指纹。128局从NPZ逐字段精确重算通过；逐transition分项对实际reward最大误差legacy=1.345761e-6、minimal=4.575122e-7，combined_total对实际reward均0。37项focused回归、py_compile和diff-check通过；只读reader用不可写数值输入验证无状态/动作修改、输出确定性和Python/NumPy RNG不变，源码另校验不调用JAX RNG/env.step。

categorical交付父/候选31/32、31/32，死亡事件各5（死亡局5/32与4/32）；greedy交付17/32、0/32，候选32局全部timeout且无取包/持包。候选−父配对交付率差categorical=0、游戏级95%区间[-.09375,+.09375]；greedy=-.53125、区间[-.6875,-.34375]。不能将候选greedy低死亡称为安全改善。categorical死亡窗取包21/45→20/73、糖浆窗9/45→6/73；闭合持包交付31/33→31/31，另各1链censor。greedy候选持包分母0，其交付链率=null。配对bootstrap以完整来源局重采样，窗口/持包每次合计分子分母；含零分母replicate的CI保持null，不选择性删样本。所有细表及scope/尾段审计见 `reports/detailed_metrics_acceptance_20261026.md`。

覆盖明确分四类：implemented为自然完整局/窗口/包链/阶段动作与道具机会proxy/native真实奖励分解，当前128局全可重算；recomputable为既有训练批次KL/clip/loss/更新步/采样/吞吐/对手实际暴露；missing-data包括动态拓扑oracle、共同state KL与V/EV、危险/自伤/同归、道具完整ID/净收益链和完整中间pending账本；unavailable为本组没有执行的生产JS Hunter/Python Tactical、道具反事实因果收益与训练seed泛化方差。缺失=null+原因，未以0填充，不跨奖励尺度排序，不晋升网页模型。

任务消费与终态：根 `TASK_CONSUMPTION.json`，attempt_03 的 `EXECUTION_STATUS.json`、`COMPLETE.json`、`ACCEPTANCE.json`、`ANALYSIS_PROTOCOL.json` 与最终验收收据。D原唯一训练owner/worker及CPU/JS评估保持独立并按原队列接续；本任务不控制GPU、不改变D协议、不恢复R2/O5、不push不发布。

最终manifest SHA `0945d8c6693477dcd072c2c2db8d3211dc918d977c800c17c5b89c9f8a118f61`；summary SHA `d073f3be6436767968ead52d15a0f3688b0f97015a23d98676c1c77d17f1d1b9`；ACCEPTANCE SHA `dd9c8a59006aa2a0a3a03f2f9f223a6a37cf2ece5238be18c07e53dd8da6853a`。

最终现场监督：原D owner已完成 `legacy_s2026102852/u000320`（5,242,880新增步，10,240优化步），state SHA `6c3d5cda063fdef868ec8ad7dc6e8bbd61c810dc0b5c99315cd00d5ece1ebdc5`、policy SHA `80e7709fb270618ed8bb3c8081a27a581704ace97cb7edd933acf379318e7a14` 均独立核验。原唯一owner已实际接续 `minimal_s2026102852` worker；没有新建训练owner或恢复旧队列。边界观察收据为 `attempt_03/TRAINING_BOUNDARY_OBSERVATION.json`。

### GPU-EFFICIENCY-20261026-P1（只读诊断与D后独立性能probe）

消费`.hermes/gpu_efficiency_followup.md`；原D协议/奖励/batch/owner保持不变，R2/O5/储备不恢复、不push/publish。`runs/gpu_efficiency_20261026/D_observation_v1` 360秒/1Hz连续证据覆盖legacy seed2853 u35–u69和u64原子保存，8卡均映射原worker；完整周期约1550 env/s、GPU mean util33.5–36.7%，现有iteration计时外boundary约19.2%，不能将全部差归因拷贝/保存。精确compile/transfer/kernel/eval内部阶段缺timer均null+原因。

独立manifest `runs/gpu_efficiency_20261026/performance_probe_v1/MANIFEST.json`：共同R1 s2751/u064完整state、D legacy、8×16env×128tick/16384样本、global minibatch512/local64、32optim/iter、lr7.5e-5不变；baseline/只hoist updater/只resident params-Adam分别比较；每arm cold8+cachewarm新进程8，合计786432诊断env步/1536optim，不采纳策略。执行器只在D全部六run固定预算终态、原owner/worker退出且8卡释放后允许独立probe；未加入D队列、未启动新GPU owner。CPU缩小形状fixed-seed parity已完成：父点空手轨迹cold/cachewarm新进程8个比较、真实取包Bank4个比较全部bitwise一致；Bank 32/32真实持包且reward非零、32/32有效老师集合，Actor/Critic和Adam真实非零更新；无新取包/死亡/交付/道具释放事件，单图/P0/allfar限制保留。19项focused回归通过。GPU吞吐收益仍unknown、D后GPU probe未启动；优先测单独hoist updater，不能把缓存串行差或全部boundary当加速。详见`reports/gpu_efficiency_20261026.md`及最终验收收据。


## AUTHORIZED-CAMPAIGN-20261026：GPU语义保持与任务结算保留奖励包

持久唯一owner：`scripts.authorized_experiment_campaign`，状态`runs/authorized_campaign_20261026/CAMPAIGN.json`。D六终点/holdout已验收，不重跑；R2/O5暂停保留。GPU v1六进程与input-placement v2独立probe已完成，逐轮轨迹及最终完整state/Adam parity通过；复用updater与共同初态放置用于下一科学对照，诊断策略不作父点。

奖励V2：既定R1 scaled三seed2751/2752/2753 u064完整state/Adam/env/aux/RNG映射新seed2026102951/2952/2953；legacy与task_semantics三配对、每臂320更新/5242880环境步，合计31457280步。8卡×16env×128tick，global batch16384/minibatch512/local64，epoch1，lr7.5e-5，gamma.995/lambda.95，DAgger/KL0，fixed JAX rule3。唯一科学变量为删辅助奖励包；保留交付+24、kill+16、原安全/控制及对应回收和单PBRS的真实结算。具体源hash、系数、删除清单、检查点1/64/128/192/256/320、技术修订在`protocols/reward_task_semantics_v2_technical03.json`。CPU评估异步，短点分数不设门禁，暂停优先。已完成smoke不重跑；正式终态需真实atomic marker，未执行项不得标完成。


### JAX-JS-EVENT-PARITY：副CX首轮已完成（20261027回填）

状态：`DELEGATED_SIDE_CX → COMPLETE`，指本轮诊断执行完成，不代表两引擎一致或原协议预算无保留验收。结果原样接收自`runs/authorized_campaign_20261026/side_cx_results/parity/COMPLETE.json`；MAIN输入222条全部SHA一致，额外10项原始结果/比较器/执行代码哈希一致。验收与owner消费收据在`side_cx_round1_acceptance/parity.json`、`parity_artifacts.json`及`requests_done/accept_side_round1_parity_20261027.json`（均相对campaign根）。不重跑，不合并分支、不提交。

配置：父R1 s2026102751/u064、D legacy/minimal s2026102851/u320；categorical/greedy分列，native JAX JE07记录的相同双方动作从局首输入生产JS，每事件按seed/tick/模型/模式顺序取dev前16、holdout前8，比较至目标事件后64tick或首分歧/终局。原协议`protocols/jax_js_event_parity_v1.json`；执行前amendment为`protocols/jax_js_event_parity_v1_amendment01.json`，SHA `eadda8f8d871fe8bb557f3a0cc3f121ac7d7b66e7c4f223d162e189f53c69143`（协议路径均相对campaign根）。测试限定RNG结果注入，不改生产源码；未对齐随机项分列。

规模/seed/墙钟：筛查来源dev816、holdout480，共1296个模型×模式来源局；dev seed2026122600–2026122735、holdout2026132600–2026132679，跨模型/模式复用seed。312个事件fixture（dev208/holdout104），实际参与比较的不同来源局分别52/26；fixture相关，不当作独立312局。墙钟3178.05秒，含dev中断/恢复间隔，CPU≤16核。dev一致/物理差异/RNG未对齐为23/180/5，holdout19/61/24；实际到达目标tick仅39/208和30/104。

结论：主要首分歧为速度成长JAX固定0.15、JS约0.157143；dev180例、holdout48例首差字段为spd_g。holdout另见control/trap（seed2026132601/t252，true/false）、status_ticks（2026132603/t411，100/99）、invuln（2026132600/t162，30/0）。不能把前缀早期速度差导致的后续缺测解释为每类晚期事件自身错误，不能外推部署收益。29个fixture保留随机分类审计；原`bank_maximum_natural_episodes=768`未作为聚合来源上限执行，dev816超出，`acceptance_complete=false`保留，COMPLETE不覆盖该限制。完整表与证据见该目录`event_parity_audit_20261026.md`、`eventful_fixture_execution.json`。

### ITEM-CAUSAL-BRANCHES：副CX首轮已完成与香蕉口径更正（20261027回填）

状态：`DELEGATED_SIDE_CX → COMPLETE`，四模型首轮分支执行完成。结果`runs/authorized_campaign_20261026/side_cx_results/itemcausal/COMPLETE.json`，36条MAIN输入及615条引用的副CX证据SHA全通过，验收`side_cx_round1_acceptance/itemcausal.json`、owner消费`requests_done/accept_side_round1_itemcausal_20261027.json`。配置：父R1 s2026102751/u064、D legacy/minimal s2026102851/u320、已结束R2 s2751/u1664；双动作模式×香蕉/慢慢胶，同一自然真实状态首tick使用/不使用，其他动作因子相同、随后学生和rule3正常响应、共享后续随机条件。native JAX JE07/layout0/P0；真实状态恢复，无人工携包，不等于JS Hunter。

执行前协议`protocols/item_causal_branches_v1.json`及`protocols/item_causal_branches_v1_amendment01.json`，amendment SHA `9a938df3eab6c234a26c196311829cbe6e7617ade6bee9375a2850ab62172662`。dev每道具×模型×模式96对，冻结后holdout每条件48对；256tick短窗与原生终局（总局长上限2560）分列。dev主要三模型1152对、R2另384对；holdout主要三模型576对、R2另192对，合计2304机会对/4608分支、4,803,991真实物理转移。审查来源56局、其中29局提供机会，不能称2304个独立来源。dev实际seed集合2026122600/2601/2602/2603；holdout实际2026132600/2601/2602/2603/2604/2618，具体模型/模式/阶段映射见COMPLETE。正式runner6785.30秒；含无效前置runner的正式运行总墙钟10886.91秒，初始化以来15157.45秒，不能混作纯有效计算时间。batch16初始440对与漏更早机会96对均排除，证据保留。

结果：独立holdout16条件的整局交付use−no-use均未见正均值（多数0，legacy greedy胶−0.0625、minimal categorical香蕉−0.0208）；dev有局部正值（父greedy胶+0.083、R2 greedy香蕉+0.031），不能称所有条件严格无收益或用它晋升。机会点bootstrap CI不能替代来源局聚类或训练seed泛化；胶自困/控制时长增加的归因待第二轮细分，胶减速不构成原生糖浆取包窗口。首轮香蕉按“敌人滑倒→控制/死亡”评估，明确标为**评估口径不符用户语义，仅作次要参考**，其无正交付结果不否定持包自踩加速用途。

用户语义：香蕉主用于持包时放置后自己踩上加速回笼，以机会到交付tick缩短、128/256tick交付、交付前死亡/丢包、真实自踩滑行与向家最短路推进为主；不奖励香蕉杀敌链。独立第二轮协议`protocols/item_causal_branches_v1_amendment03_banana_self_accel.json`已预声明，SHA见20261027验收报告，由副CX继续CPU执行；主线不领取、不把首轮补改成该口径。完整首轮表见该目录`item_causal_branches_20261027.md`及`native_matched_branch_results.json`。

### COMMON-STATE-KL-V-GAE：副CX首轮已完成（20261027回填）

状态：`DELEGATED_SIDE_CX → COMPLETE`；194条MAIN输入和104条副CX引用证据SHA全通过，结果`runs/authorized_campaign_20261026/side_cx_results/statekl/COMPLETE.json`，验收`side_cx_round1_acceptance/statekl.json`及owner消费`requests_done/accept_side_round1_statekl_20261027.json`。父R1 s2026102751/u064、D legacy/minimal s2026102851/u320、R2 s2751/u1664及V2两臂s2026102951/52/53 u320，共10模型。共同状态来自父自然局，按对拼/原生取包窗/持包返程分阶段；JAX JE07/layout0/P0/rule3，源dev/holdout各重放304局，seed分别2026122600–2026122903、2026132600–2026132903。模型run seed、逐状态清单及fold_in分支RNG规则保留在COMPLETE。

协议`protocols/common_state_kl_v_gae_v1_technical02.json`、执行前`protocols/common_state_kl_v_gae_v1_amendment01.json`（SHA `b9126df8ee963dfc54945b12a7acabb62224a8625c8a8494f54306a580cfcb88`）；算力修订`protocols/common_state_kl_v_gae_v1_amendment02.json`（SHA `80a0e813beddbd7b7358b8e0f4b146b7ebe34b8542cd061de3b322467b739ba7`）将MC改为冻结前缀dev每阶段128状态×4 categorical终局rollout，holdout每阶段64×4，greedy仅动作一致率。KL/熵用完整冻结清单dev每阶段512、holdout256。主要MC dev15360条、holdout2304条，全部原生终局、不用待测V自举；补充已存rollout单列。holdout仅父与预冻结dev KL排序两端V2 legacy s52/s53。CPU≤16核，墙钟23377.69秒（约6.49小时），batch_seconds不冒充并行墙钟。

结果：未见阶段均值熵塌缩；R2 dev熵/父为1.114/1.853/1.306，holdout熵1.076/0.886/0.284，父为0.941/0.459/0.213。holdout对拼EV父0.048、V2 legacy s52/s53为0.493/0.076，明显低于部分dev，不能以训练loss代替。持包返程dev V−MC偏差：D legacy−2.304，V2 task s52/s53为+1.959/+2.520；返程EV对应0.274/0.282。holdout返程V−MC父−0.525、legacy s52−0.517、s53−1.326。不同奖励尺度不跨臂直接比V；四rollout均值有MC噪声，EV不检出常数偏移。

限制：父访问分布、单图/P0，不证明部署泛化或漂移因果；隐藏奖励账本未输入Critic只提供假设，不能单独证明根因。时间线更正：dev KL确已在amendment02前计算并查看，不能照协议reason宣称未看任何KL；EV/偏差在修订清单冻结后计算，未用holdout调参。完整结果`conclusion_20261027.md`、`phase_KL_and_independent_MC.json`。第二轮返程Critic分析由副CX执行，本轮完成不触发重复采样或GPU干预。



### AUTHORIZED-CAMPAIGN-20261026：低util实证接续与异步评估

输入放置独立GPU干预8轮exact，第二collector77.214→5.987秒，正式两臂共用且不改D。V2正式run持续真实非零增长，u64完整原子hash已复核。新增独立4核CPU评估lane，冻结45条件dev/holdout双模式矩阵，实际消费u1；formal stage/checkpoint闸门排除smoke旧请求。评估不阻塞GPU，未恢复R2/O5/储备。详细数据、短窗GPU均值及限制见`reports/authorized_experiment_campaign_20261026.md`；任务消费收据`runs/authorized_campaign_20261026/LOW_UTILIZATION_TASK_CONSUMPTION.json`。全campaign未完成。


### AUTHORIZED-CAMPAIGN-20261026：GPU空闲事故安全恢复与防复发

owner被动进度读取FileNotFoundError逃逸已定位；具体缺失文件/底层消失机制未知。task2951/u320完整finite/hash复核后COMPLETE；唯一watchdog已自动启动legacy2952，u1真实非零更新及连续8卡样本已验收，详见`GPU_RESUME_ACCEPTANCE.json`。owner-only退出73自动恢复演练保持全部GPU/CPU/JS原进程，见`WATCHDOG_RESTART_ACCEPTANCE.json`。V2后实际GPU接续SCALE env/minibatch/LR各3seed两臂5242880步，多卡固定global batch131072步，技术恢复先验且不更改D/V2。剩余各科学项已冻结协议/证据依赖，尚缺数值producer的数据不当作已完成；原生/生产JS/补漏斗CPU异步。报告`reports/authorized_experiment_campaign_20261026.md`及任务状态`GPU_IDLE_RECOVERY_EXECUTION_STATUS.json`。


### CPU-EVAL-PARALLEL-20261026-REV2

调度独立修订，冻结科学协议不变。主评估27核+副CX48核+GPU保留12核≤88核配额；Native3/checkpoint4/FUNNEL2/JS两批次，运行caps按同run暖更新吞吐下降>5%自动缩容，20更新恢复≥98%后渐进恢复。20261026首审时三个副CX任务为DELEGATED_SIDE_CX、仅有0局阻塞审计；20261027已按上方三项原位回填首轮数值结果并由唯一owner消费为COMPLETE，旧审计作为历史保留，第二轮仍由副CX执行。30项GREEN与131个零游戏merge验证任务已落盘；全部评估/训练仍在执行。详见reports/authorized_experiment_campaign_20261026.md和runs/authorized_campaign_20261026/EVAL_PARALLEL_EXECUTION_RECEIPT.json。旧R2/O5暂停，无push/publish。


### CAMPAIGN-REMAINING-CPU-20261027：输出自等待修复与规模验收

根因实证：NATURAL-BANK-COVERAGE（无依赖）、FUNNEL-COMPLETE（上游已COMPLETE）、BANK-RATIO-AB原入口均为`campaign_evidence_acceptance`验收器，`ready_files`却是自身最终输出，没有producer。唯一owner/资源健康，调度依赖集合本身未错误；形成输出自等待。通过实际requests/requests_done替换pending job为`scripts.campaign_remaining_cpu`生产者，readiness改为冻结协议与真实输入COMPLETE。owner安全exec/adopt既有CPU worker，无重复owner、无信号中断，无新GPU。

FUNNEL-COMPLETE已完成对四模型×双模式×304局原生JAX（2432局）及相同模型/模式生产JS（2432局）的分引擎漏斗汇总，另列ABI2父/legacy/minimal共1824局，不重复计数。死亡/原生糖浆窗口→取包→同custody交付已链接；死亡/丢包/censor/未知保留，own_alive/actionable与终局前/尾段仅ABI2可用，R2缺。香蕉完整自踩加速到交付链、胶完整追击净收益尚不可用，COMPLETE只代表可用轨迹汇总完成，scientific_acceptance=false；不以命中proxy冒充完整链。证据`runs/authorized_campaign_20261026/funnel_complete_v1/COMPLETE.json`。

NATURAL-BANK-COVERAGE现按冻结上限768个自然来源局执行真实动作/RNG前缀重放：四模型×categorical/greedy×固定首96局，选择在看结果前冻结，保留无取包局。真实死亡/原生糖浆库存取包才保存完整before/after/env/aux/RNG、真实动作/来源SHA及next-step parity；不人工初始化携包，不把既有Bank审计当采集。相同初态seed在全部模型/模式共用source-game split。当前冻结来源本身仅map806/layout0/P0，跨地图与P1覆盖明确缺失；重放完成前不得称采集完成。进度与可恢复分批证据`natural_bank_coverage_v1/PROGRESS.json`、`batches/*.json`。

BANK-RATIO-AB保留依赖并由同owner自动接续CPU验收。原冻结正文是.25/.125、三seed六run、每run5242880步GPU A/B，但队列实际为analysis验收器；paired_seeds与sources键还不一致，实际fresh/provenance均硬编码.25，缺已验证.125实现。最新用户明确不新增GPU，故依赖满足后将落BLOCKED_PROTOCOL/原始证据，不伪造训练或静默扩展合同。旧R2/O5暂停保留。

规模实验22个run/checkpoint原子state/policy SHA及更新日志已核验，三SCALE组各3seed两臂等环境步；54个dev条件、18个holdout条件、双动作模式共144行独立评估。全量报告`reports/scale_summary_20261027.md`与机器可重算`SCALE_RECOMPUTABLE_SUMMARY_20261027.json`。env32配对暖吞吐+67.6%，但holdout categorical死亡事件三seed均增加、greedy第二seed交付回落，未证明整体非劣；不把它作为能力无损推荐。mb256吞吐+0.6%且能力下降不推荐；lr5e-5吞吐−3.8%，能力候选不等于性能优化。多卡单seed8更新probe没有独立能力评估，1卡约2634对8卡2654 env/s，不能当作已证实并发组泛化。保守保持env16/batch16384/mb512/lr7.5e-5，env32只标优先性能候选；不混合未测配置。

RED/GREEN：新增输出不能作为自身readiness、来源split、同custody链接与missing-data测试，连同owner/CPU调度28项focused通过。GPU登记预算全部完成且没有GPU后继，8卡空闲属预期，owner新增`NO_REGISTERED_GPU_SUCCESSOR`持久原因与去重通知；不会为了占卡造新实验。未commit/push/publish。

首批启动验收：已完整重放40个固定自然来源局，真实保存35个因果取包状态，35个next-step parity通过，分批pickle SHA已复核；Bank继续运行，不宣称最终覆盖完成。owner=1、CPU worker=1、亲和2核，GPU8卡0MiB/0%，没有新增GPU。持久收据`runs/authorized_campaign_20261026/REMAINING_CPU_RESUME_ACCEPTANCE_20261027.json`。

### RULES-V2-CRITIC-AB-20261027-V1：G1阻塞期间的CPU实现

冻结协议`runs/authorized_campaign_20261026/protocols/rules_v2_critic_ab_v1.json`（SHA `875bbafd682eb1dd52a8d38f81c913494c0432b143a4ffe2cfa10a7ab88896f7`），实现细节`rules_v2_critic_ab_v1_implementation01.json`。按编排方最新判定，G1=BLOCKED：第四轮fresh两类规则顺序残差未消除，第五轮原副线负责、fresh2 seed从2026152600起。主线不写G1 amendment、不修parity、不启动GPU。G2/G4尚未开始；旧规则/历史checkpoint不改。

共同父R1 scaled s2026102751/u064，state SHA `7c029e11eb7669d6404ab2eaf07e4f4895b535f13eb09c6c04796d0953f5e781`，policy SHA `fe34de1cf1cb51256723e44385a28b17326b84c8d31fd73162f738cbdca6359f`。未来三臂A0原Critic、A1仅Critic账本15维、A2=A1+同custody交付辅助BCE（系数.1冻结），三配对seed2026110151/52/53，每run5242880真实on-policy步/u320。新增投影/头零初始化；旧Adam/count保留，新叶moment零。Actor参数/输入不追加账本，辅助项不直接进入Actor梯度。reward按原协议既定晋升结论，未晋升用legacy，所有臂一致；本阶段没有奖励/规则干预。不得把未执行9run标完成。

G3按`reports/scale_summary_20261027.md`回退默认：env32非劣证据不足；8卡×16env、rollout128、batch16384、global mb512、epoch1、lr7.5e-5、gamma.995、lambda.95、DAgger/KL=0。不采用混合未测配置。新规则Bank与完整状态ABI需G2后真实因果取包验收；旧中局不能补零后伪称无损迁移。

CPU实现`scripts/rules_v2_critic.py`，18项focused回归通过（raw24/phase29、非零输入原forward对照、Actor梯度隔离、同custody标签/未知尾部mask、Adam迁移与loss/optimizer接线）。实际共同父+7个旧规则真实取包状态检查：三臂Actor权重/logits/greedy动作在eager/JIT分别精确一致，初始V差0，原Actor/Critic Adam叶保持。首失败源于检查器将JIT与eager混比，未变父本身移动logits差0.18494；改为同执行方式比较，未放宽容差，原RED保留。实证`runs/rules_v2_critic_20261027/v1/REAL_PARENT_CPU_ACCEPTANCE.json`；不能外推新规则/GPU parity或能力。

新规则探针manifest与终局MC评分入口已准备，核SHA/规则seal、gamma和分层ABI，拒绝人工截断和待测V自举；真实采样及GPU collector尚未执行/接线。按父自然来源独立dev/holdout、双动作模式、各阶段首128状态×4MC，不足披露，最终JS Hunter三难度各64局/模式。唯一既有campaign owner已消费被动`RULES-V2-CRITIC-ROUND5-WATCH`并RUNNING，校验第五轮到达只写通知、不自行放行。G1/G3/空闲>10分钟通知持久化。详见`reports/rules_v2_critic_preparation_20261027.md`。本阶段CPU准备完成、等待G1；8卡空闲符合用户禁止GPU要求，旧R2/O5/储备仍暂停，未commit/push。

### BANK-RATIO-AB-20261027-AMENDMENT01：旧规则真实Bank比例六run授权接续

用户“都做”授权解除原BLOCKED_PROTOCOL，原`bank_ratio_ab_v1/BLOCKED.json`保留，唯一owner已消费`requests_done/20261027_320_rebind_bank_ratio_block.json`。冻结`protocols/bank_ratio_ab_v1_amendment01.json`（SHA `94da70a237be43c1aacef604ecf35c9137c8726a3f1667a35f08427ddfcd976d`），只改真实认证Bank重置概率.25/.125。paired2026103151/52/53按序映射source2026102951/52/53，完整R1 s2751/52/53 u064 params/Adam/env/aux起点；训练RNG按paired seed生成并明确不是原RNG连续续训。使用原配方train Bank，不混新446状态；旧规则结果与rules_v2分开。

每run5242880步/u320，legacy reward、global batch16384、rollout128、mb512、epoch1、lr7.5e-5、DAgger/KL0。执行amendment02说明8×16环境重分片1×128保证global batch不变，原父env/aux不丢；旧两份冻结协议不改。默认fresh/provenance逐位等价与相关focused测试通过，原fresh快照SHA匹配历史seal；.125真实GPU重置频率将在smoke记录，实际done-reset分母与占用tick分开。首GPU频率样本.25为2014/8192=.24585、二项p=.39268；JSON numpy.bool_技术故障在PPO前发生，u0保存、原owner从u0恢复，见TECHNICAL_REPAIR_01。未伪称正式更新。

固定u320 dev64+holdout304/动作模式，三父点与六候选18条件独立CPU异步，无老师、不择峰；manifest `protocols/bank_ratio_eval_v1.json`。首run baseline进行中，正式六run及最终结论未完成；详见`reports/bank_ratio_gpu_slots_20261027.md`。

### GPU-SLOT-PARALLEL-20261027-V1：技术吞吐并发验收

用户冻结`protocols/gpu_slot_parallel_v1.json`：每卡一run，保持global batch和优化配置，devices=1不作跨设备逐位保证。唯一campaign owner管理`BANK-RATIO-GPU-SLOTS-V1`子控制器及六worker，无重复owner。单独control s3151的u3–10八暖更新中位B；再六run各八暖更新≥.85B方可正式接续，失败6→3→8卡串行。完整配置同seed的更新计预算；吞吐阈值只做技术并发判定，不以短期能力判定停训。

核段冻结：parity0–15；GPU0–5分别16–18/19–21/22–24/25–27/28–30/31–33；主线CPU/eval/analysis/JS40–66，互不重叠共61核，GPU6/7不造任务占用。默认cpu_group offset12保持，当前配置40、OMP/BLAS1。GPU_SAMPLES记录功率/util，真实更新记录env/s/collector/cycle；20更新中位低于.85B暂停新入队并通知。smoke判定/run终态/空闲10分钟写持久notifications。当前B和六并发结果尚待采样，未提前判PASS。rules_v2仍等编排方G1/G2/G4，新组需自己基线，不偷用本组B。未commit/push，旧R2/O5/储备暂停。

BANK-RATIO/SLOT首更新实证回填：六run均u1原子hash复核、finite、Actor/Critic非零；首control s3151 Δ=.6809928203/1.057992344，state SHA `8b9f52ba7ec8fb6e4b761e2ece0d1da385ad7768e77f3bda9521658af1e0b110`，合计98304真实步，证据`ALL_SIX_FIRST_UPDATE_ACCEPTANCE.json`。三对u0 params/Adam与各自源完全一致、env/aux仅轴重分片全部字段exact、两臂新RNG一致，`PAIRED_SOURCE_ACCEPTANCE.json`。GPU fresh每臂8192样本，control2014/2046/2079、reduced998/1047/1008，全部二项检验与.25/.125相容，分母不与occupancy混同。默认完整provenance逐位测试2项、调度/owner最终26项通过。

单卡暖B=2632.8441env/s（u3–10），吞吐线2237.9175env/s；u10独立新进程完整env/aux/RNG next-step和Adam exact。同步并发计时歧义执行前登记`gpu_slot_parallel_v1_implementation02_synchronized_window.json`：等全部六worker完成冷编译和原子NEXT_STEP后，再共同释放八暖更新；首批有效更新仍计固定u320，不能把先暖槽位与冷编译错位计时判PASS。仅CPU槽位控制器安全替换，GPU worker未被信号中断、原owner实际adopt，无重复owner。六并发同步结果尚待完成，未伪称正式放行。报告`reports/bank_ratio_gpu_slots_20261027.md`。

GPU-SLOT正式放行回填：同步六并发八暖更新PASS，B2632.8441、阈值2237.9175env/s；control/reduced s3151为2592.52/2541.61，s3152为2800.57/2552.76，s3153为2718.33/2439.59env/s（最低.927B），所有finite/Actor/Critic非零、原子提交。未减半/未回退；原六worker直接续作各自u320，已产生smoke边界后更新，未重跑。实际唯一campaign owner/slot controller/GPU worker数1/1/6，现场GPU0–5 util77–82%，GPU6/7无任务；证据`bank_ratio_ab_v1/GPU_SLOT_PARALLEL_ACCEPTANCE.json`、`SLOT_PROGRESS.json`、`GPU_SAMPLES.jsonl`。smoke/run终态与空闲通知机制有效，CPU异步评估不阻塞，rules_v2仍等G1/G2/G4。本阶段启动与技术验收完成；六run训练终态/能力评估尚未完成，不把吞吐通过称能力提升。

### RULES-V2-CRITIC-AB：G1放行、G2版本化合入与阻塞（20261027）

用户/编排方第五轮G1判PASS，证据`side_cx_results/parity/round5/G1_GATE.json`；实际校验57个交付产物SHA及累计patch SHA `7634ab60c8cb12fd0d44900e67281f20737ea94121be9fd087b5e3e269fcf7a9`。fresh2一致199/206，余7个浮点/格线边界，规则型及未解释残差均0；dev203/208、holdout100/104、旧fresh199/208无回退。限制：转移类覆盖14/16，缺2窗口未补抽；只覆盖已比较Sim.step字段，不代表真人frameStep一致。香蕉默认Sim.step维持268px/s，真人入口480px/s不改。

新增冻结`protocols/gate_auto_advance_v1.json`已消费：机器判定PASS、无未解释/规则残差、无回退/新失败/歧义，且后继是原合同下一步时立即推进并通知；否则BLOCKED等待，不擅改协议或绕过失败。G1已自动推进G2。为保留旧规则CPU评估及旧checkpoint可复现，累计patch在隔离临时目录应用并逐一核对原候选三文件SHA，原始patched源码留`rules_v2_critic_20261027/v1/candidate_source/`；新模块`je05_core_env_v2.py`、`je07_bun_env_v2.py`、`je07_native_items.py`，仅JE07的core导入改为版本化入口。默认旧环境文件未修改。新训练必须在任何collector导入前显式安装`rules_v2_runtime`；checkpoint NamedTuple保留v2模块身份，不把旧中局补零伪装迁移。

共同父R1 s2026102751/u064的state/policy SHA已验；隔离旧读取器导出params/完整Adam及source_update64，digest完全一致，`PARENT_EXPORT.json`，导出SHA `1a4d5ee0f734b547b45b3941bba44a0255927ad3c2fc0fcc2f864db9e5356625`。旧env/aux/RNG明确不跨版本复用，后续是自然局边界全部重建，非旧rollout逐位续训。

G2当前BLOCKED：全量旧规则回归在collection阶段发现缺失`tests.test_je07_native_integration`，影响`test_je08_boundaries.py`与`test_je08_integration.py`；测试namespace冲突修正后仍确实缺文件，未跳过后冒称全套通过。定向首轮另有round3 JS comparator未复制导致1个工具路径失败，已原样复制所需artifact并重查，不修改规则补丁。关钩子snapshotReplay已实际通过2700tick/3seed，两组生产JS道具回归通过；新规则集成与恢复检查在途，不能提前标PASS。`rules_v2_critic_20261027/v1/G2_GATE.json`及两处notifications明确阻塞。按最新自动推进规则无G4/无GPU启动；九run尚未执行，不冒称首更新。旧BANK评估继续异步，R2/O5/储备暂停、不commit/push/发布。详细结果写`reports/rules_v2_critic_preparation_20261027.md`。

G2补充：parity重查31/31通过，首轮唯一失败是缺round3比较器artifact，现已原样补齐，未改规则补丁。新ABI实际初始choose/transition及reward/账本finite已运行，但直接auto_reset检查在scene int32（E.initial）/int8（env._fresh）分支不兼容处失败；实际collector为auto_reset=False+select_tree，故不将此局部API失败泛称实际训练collector故障或新physics残差。fresh-process恢复尚未执行；原生状态已保存并封SHA，仅作为排障证据。G2维持BLOCKED（全量历史helper缺失仍未解决），G4与9run未启动。新自动闸门规则已消费、两处BLOCKED通知持久化，jdme终端通知已成功发送；BANK旧规则CPU评估继续。

G2最终阻塞收据（本次不越闸）：给定集成回归实际23通过、1失败后按新闸门规则安全SIGINT终止当前CPU检查，余项明确未跑。失败`test_bun_spawn_buckets.py::test_reproducible_and_ratio_matches_weights`：native bucket样本((3,12),(10,14))不在声明native_pairs集合。该测试导入历史bun_env，归因未定，不冒称新规则physics退化；未改测试/规则去抹平。证据`G2_NATIVE_INTEGRATION_RESULT.json`与原日志SHA，原子`G2_GATE.json=BLOCKED`、`EXECUTION_STATUS.json=BLOCKED_G2_WAITING_REVIEW`；本次CPU检查已退出。G1 PASS、定向31/31、snapshot2700tick均保留；全套缺helper、直接auto-reset dtype及此断言分别列范围。最新现场唯一campaign owner1、watchdog1，BANK评估现场存活1路；8卡0MiB/0%，新GPU启动0，G4和9run均NOT_STARTED。两处最终BLOCKED通知及GPU空闲>10分钟原因已落盘；等待规则要求的阻塞复核，不冒充已入槽或首更新。未commit/push/发布。

### 常设结果指标口径（20261027起）

所有本campaign后续结果（BANK-RATIO、Rules Critic、规模/温度、parity后续）按`protocols/funnel_complete_v1.json`及`passive_evaluation_metrics.py`分环节报告：对拼窗口数与施害归因/本方死亡/超时；死亡、原生糖浆窗分别按闭合窗口算窗内库存取包率及起点到拾取tick；自由存活取包单列；持包按同custody ID报告交付/死亡/掉包/censor和返程时长；道具按空手战斗/受控窗口/持包分请求、真实放置、自吃和敌接触，香蕉与慢慢胶分开；有数据才报返程Critic EV及偏差。采样/argmax、dev/holdout、JAX/生产JS严格分列；比例从pooled分子分母计算，配对seed差与来源局bootstrap区间同时报告；缺失值写null并说明原因，不由命中proxy、逐tick库存或存活时长代替事件证据。香蕉首要语义为持包自踩加速及交付tick缩短，敌方滑倒只作次要观察。冻结协议没有的评估不新开GPU任务。BANK六run与三父点18个固定条件已完成此分环节重算，报告`reports/bank_ratio_ab_funnel_20261027.md`、机器可重算汇总`runs/authorized_campaign_20261026/bank_ratio_ab_v1/FUNNEL_RESULT_20261027.json`；共6624自然局、36个split/mode cohort，未重跑对局或新增GPU。原trace缺Critic V/MC、typed item consumption IDs、完整香蕉自加速链、JS Hunter/Python Tactical、own-alive/actionable和首terminal/settlement item边界的项均为null并附原因；配对来源局CI不冒充训练seed不确定性。

对拼过程细项（放泡质量、威胁、逃生、死亡原因、接敌、分类型道具）由副CX `cx-campaign-statekl` 在独立worktree执行，状态记为`DELEGATED_SIDE_CX`，产物预期位于`runs/campaign_side_combat_metrics/`；主线不重复实现或启动同一评估，收到后仅验收输入哈希、样本数与指标定义。

GPU空闲复核后，依用户`.hermes/gpu_utilization_now.md`授权建立`GPU-BATCH-SWEEP-V1-20261027`，冻结父R1 s2751/u064（state SHA `7c029e11eb7669d6404ab2eaf07e4f4895b535f13eb09c6c04796d0953f5e781`）、env16/64/128/256/512/1024及env256/1024 minibatch4096技术扫描，目标每臂6更新，能力评估/择峰均禁止。attempt01暴露训练入口覆盖`CUDA_VISIBLE_DEVICES`使8个进程映射至同卡，只有独立fresh u0、零优化更新，全部定向SIGINT退出并保留状态，结果标INVALID_DEVICE_MAPPING。修复单卡物理设备覆写并通过4项focused测试后，attempt02在GPU0–7各绑定一个worker；8个worker已各到u1、finite且Actor/Critic更新非零，真实步数2048–131072，首更新周期194–234秒，其中collector 182–217秒。GPU功率115–121W、显存3221–17995MiB、时钟满频，但`nvidia-smi`和`dmon`均连续报告SM util 0%；按限制不声称测得非零util，保留原始采样待查遥测语义。attempt02继续完成每臂6更新；首更新证据`attempt02/FIRST_UPDATE_ACCEPTANCE.json`，原始GPU样本`attempt02/GPU_SAMPLES.jsonl`。G2全量CPU回归继续后台；结果日志`runs/rules_v2_critic_20261027/v1/G2_FULL_NEW_REVIEW.log`。


### SCALE-ENV1024-PILOT-V1 / GPU-BATCH-SWEEP attempt04（2026-10-27；已冻结、待启动）

A协议：`runs/authorized_campaign_20261026/scale_env1024_pilot_v1/PROTOCOL.json`（SHA256 `5a2d4ac45b48868cf005c277bde113ac0cafcf0b2982ced6d101b37795308b2a`）；异步评估：`runs/authorized_campaign_20261026/scale_env1024_pilot_v1/eval/PROTOCOL.json`（SHA256 `4115618c6376d06d200938079cb31ce08c5501ec3103c0d710447eb5ab2dc575`）。父点为R1 s2751/52/53 u064，各自与seed 2026103051/52/53配对，state/policy SHA在`GPU_NEXT_AFTER_SWEEP_PRESTART.json`逐项核验。每run单卡GPU5/6/7、env1024、rollout128、global minibatch4096、32优化步/更新、lr7.5e-5、legacy奖励、JAX rule3、Bank.25、DAgger/KL=0，固定320更新/41,943,040步；仅每更新采样批放大8倍，更新数与其他算法设置冻结。对照既有同seed env16×8：报u320等更新与u040约等环境步两种，不择峰。dev在u1/u160/u320各64局/模式，final holdout304局/模式，categorical与greedy分报，CPU独立消费。

B协议：`runs/authorized_campaign_20261026/gpu_batch_sweep_v1/attempt04/PROTOCOL.json`（SHA256 `fadb5803f55c1eec2dee10798eda6d7af1a0d904753bfcc7d7618e4ef94c2878`），新namespace attempt04，attempt03失败零更新证据保留。GPU0-4各执行env2048/4096/8192/16384及env1024/mb4096复测，rollout128、global minibatch分别8192/16384/32768/65536/4096，恒32优化步/更新，每档6次更新；OOM作为饱和边界，不重试降档。只做性能测量，不作能力结论。CPU核心段在冻结协议内错开G2（20–31）与副线评估（56–71），本次GPU0–4使用32–55/72–87，GPU5–7使用88–111，互不重叠。旧R2/O5队列未恢复。

G2完整新规则回归由原CPU进程继续；本次A/B为用户单独授权的legacy计算，互不依赖G2、不进入Critic九run。启动/更新/终态另回填本节及当日报告；不commit/push。


A/B实证回填（2026-10-27）：B attempt04唯一owner已消费五档并终态；env2048/4096/1024复测均6/6 finite、Actor/Critic非零、state/policy hash正确，暖env/s中位分别约19.8k/25.9k/13.4k；env8192/16384按冻结规则作为首阶段OOM容量边界。B解释收据`runs/authorized_campaign_20261026/gpu_batch_sweep_v1/attempt04/CAPACITY_INTERPRETATION.json`，原始`SWEEP_RESULT.json`未改。A三seed正式u1均为131,072步finite/nonzero且checkpoint双hash验真，冷首更约224–231秒；u2以后单卡暖吞吐约12–13.5k env/s。A固定320更新继续，CPU evaluator已异步消费u1/等待后续点；GPU5–7继续训练。当前8卡样本见启动通知和实时`nvidia-smi`，G2全量CPU回归继续。


A/B实时续记（2026-10-27 18:40 CST）：B attempt04固定扫描已终态；env1024复测、env2048、env4096均6次finite且Actor/Critic非零，env8192/16384在更新前OOM并按容量边界收尾。保留原`SWEEP_STATUS.json=FAILED`，语义解释收据=`COMPLETE_WITH_CAPACITY_BOUNDARIES`；attempt04协议实际SHA为`fadb5803f55c1eec2dee10798eda6d7af1a0d904753bfcc7d7618e4ef94c2878`。A三seed仍正式训练，最近分别u29/3,801,088步、u25/3,276,800步、u25/3,276,800步，finite且Actor/Critic更新非零，最近周期9.9–10.9秒；GPU5/6/7分别80%/74%/73%，显存各18,031MiB。CPU评估异步跟随u1。完整样本与限制见`reports/gpu_next_after_sweep_20261027.md`及`CAPACITY_INTERPRETATION.json`。

### SCALE-ENV2048-4096-PILOT-V1（2026-10-27）

用户授权消费`.hermes/gpu_fill_0to4_1905.md`，在GPU0–4使用已验收的`campaign_scale_train`入口，以env1024 pilot相同R1 s2751/52/53 u064父点、seed 2026103051/52/53、完整Adam映射、legacy规则/奖励、固定JAX rule3、lr 7.5e-5、Bank .25、DAgger/KL=0、rollout128及每更新32个优化步，做同环境步三档对比。env4096/minibatch16384每seed80更新；env2048/minibatch8192每seed160更新；两组均41,943,040真实环境步。协议`runs/authorized_campaign_20261026/scale_env_batch_pilot_v1/PROTOCOL.json` SHA256 `f4234369b6cdd4d293e5a53f67b6239e69966d59cd2d35e0f1fbb57b85a14858e`。唯一owner状态分别在`OWNER_PID.json`及`STATUS.json`；GPU0–2启动env4096三seed，GPU3/4启动env2048 s51/s52，s53由owner在首个槽释放后自动接续。独立CPU evaluator已启动，固定dev检查点u1/中点/终点各64局/模式、终点holdout304局/模式，categorical与greedy分报且不门控训练。19:08启动采样处于XLA初始化，尚无首个优化更新，不记作训练验收。首更新finite、Actor/Critic非零与原子SHA待后续验收；全程固定预算、不择峰。旧R2/O5保持暂停。

首个正式更新验收续记（2026-10-27 19:15 CST）：env4096三seed与env2048前两seed的u1均已独立复核finite、32优化步及Actor/Critic L2非零，state/policy哈希与各自`COMMITTED.json`完全一致；值和SHA见`runs/authorized_campaign_20261026/scale_env_batch_pilot_v1/FIRST_UPDATE_ACCEPTANCE.json`。env4096每次524,288环境步，u1冷周期约225–227秒；env2048每次262,144步，冷周期约211秒。暖速约25.3–25.8k和19.8k env/s。19:15:37 nvidia-smi为GPU0–4 util 87/89/87/82/85%，显存67,475/67,475/67,475/34,073/34,073MiB；GPU5–7的env1024 pilot仍运行。新试点固定预算继续，CPU评估异步。首更仅为技术有效性，不作能力结论。

19:18现场续记：env4096三seed推进到u12（各6,291,456步，recent finite），env2048 seed51/52到u21（各5,505,024步，recent finite），seed53仍由唯一owner排队待首槽释放；最近周期分别20.8–21.0秒与13.1–13.3秒。env1024既有三seed同期u255/243/243、finite，继续到u320。新异步CPU评估4个worker在跑、20条件等后续原子checkpoint，未形成终点评估。nvidia-smi 19:18:49 GPU0–4 util86/89/87/86/82%、显存67475/67475/67475/34073/34073MiB；GPU5–7 util80/81/78%、显存各18031MiB。报告`reports/gpu_next_after_sweep_20261027.md`与实时owner状态可复核。

### GPU-FILL-2155-ABC-V1（2026-10-27）

按`.hermes/gpu_fill_2155.md`授权在确认8卡显存全空后立即启动新命名空间`runs/authorized_campaign_20261026/scale_followup_2155_v1/`，协议SHA `fa061b7fbb97f1f869c7dc1fdad78dd32f55f48e9605e2d5508a3250ff0d1908`。父终态分别为env2048三seed u160、env4096三seed u80、env1024三seed u320；9个被续训起点的state/policy COMMITTED哈希均启动前验证并复制为独立namespace的恢复checkpoint，保留完整params/Adam/env/aux/RNG/history。A env2048 u160→640三seed（GPU0–2），B env4096 u80→320三seed（GPU3–5），C先在GPU6/7同R1 u064父做env1024/4096各5,242,880真实步匹配seed51，再按seed52/53接续；C对照完成后GPU6/7轮转执行env1024 s51/52/53 u320→u960全状态续训。固定legacy reward、JAX rule3、lr、Bank概率、32 optimizer steps/update与其他算法项不变。CPU dev/holdout categorical+greedy及终点生产JS Hunter hard/normal/easy评估异步，非训练门槛。唯一owner和queue已启动；每槽释放自动接队列，旧R2/O5/储备未恢复。启动通知及运行状态见`notifications/20261027_2159_GPU_FILL_2155_START.json`、`scale_followup_2155_v1/STATUS.json`。21:58启动样本处于编译/分配，八卡有计算进程但首个优化更新尚未验收；不把launch当训练更新。详见`reports/gpu_next_after_sweep_20261027.md`。

### GPU-FILL-2155-ABC-V1-20261027：有效更新与接续修复（2026-10-27）

协议runs/authorized_campaign_20261026/scale_followup_2155_v1/PROTOCOL.json SHA256 fa061b7fbb97f1f869c7dc1fdad78dd32f55f48e9605e2d5508a3250ff0d1908保持不变。八卡首更finite、Actor/Critic L2>0，验收FIRST_UPDATE_ACCEPTANCE.json；两条C s51固定终点已完成并原子SHA验收，自动领取s52。只修编排：队列task_id解析、active进度、合法JSON、worker采用与watchdog；未改训练/奖励/优化配置，未中断健康worker。见OWNER_RECOVERY_ACCEPTANCE.json。

CPU评估补六个C终点holdout至75条件，四路128–135核；JS Hunter CPU消费者112–115核，与旧评估120–127核及GPU worker核段隔离。缺失四分点明确MISSING_CHECKPOINT，原因是复用入口未消费声明保存点参数，不补造或重跑。初次JS零局前seed元数据校验失败已修复，历史失败保留。指标、限制及实时收据见reports/gpu_next_after_sweep_20261027.md与namespace/eval。当前为技术与执行验收，非能力胜出结论。旧R2/O5暂停，不commit/push。


## COMBAT-MIX-V1-20261027：对拼四组阶梯

冻结协议 runs/combat_mix_v1/PROTOCOL.json，澄清 protocols/amendment01.json。用户授权网页对比→近似Hunter审计→训练接线→12run。旧暂停队列不恢复、不发布。原A/B/C冻结源码不改，独立collector/worker。

## 配方与预算

三seed2026103051/52/53，各自env2048 pilot u160完整state（参数、Adam、env、aux、RNG）为共同父。四组control、opp_mix、opp_mix_curriculum、opp_mix_curriculum_shaping，各新增160更新/41,943,040环境步/5,120Adam步，固定u320不择峰，总503,316,480新增步。env2048、rollout128、mb8192、epochs1、lr7.5e-5，DAgger/KL=0。每组两更新技术smoke可纳入预算，新进程full-state与next-step验收后接续。

池权重20/25/20/15/10/10%，原rule3/Hunter/hard/current/ring/固定父。原rule3本身是Hunter index3，前两槽策略相同，不当独立bot。记录实际tick/live/new-episode分母。current每更新动态传入；ring四槽每16更新替换，逐更新轮换，完整checkpoint保留。

后两组25%固定行下次真实reset进入map806清砖/hp1/300tick对拼，官方64spawn。保留native trap和原奖励及capture终止，不声称与JS快速协议完全一致。课程行替代该行Bank抽样，整体Bank曝光改变，属于课程包；不传送已携包状态。第四组系数.1、单局cap.5：旧泡所有者/格/出生tick、真实时间先后、无遮挡blast射线、首次新增敌方格覆盖和安全放置，危险完整结算后一次付费。死亡/受控取消pending，同旧事件/敌方格去重，不按存量tick付费。原threat/pressure/safe-resolution塑形保留，存在事件重叠。

## 实测

CPU网页12模型已启动，runs/combat_mix_v1/js_web_compare_v1/OWNER.json。每模型生产JS Hunter难中易各64，自然完整categorical/greedy各64，共3840局。父点已完成，其他按冻结顺序继续。线上release HTTP503，本地selfplay3072/transport1536 checkpoint SHA与本地manifest精确一致，尚未验证线上默认及发布raw/ONNX字节相同，不冒称已完成。

96固定状态（含32明示近接诊断、无人工携包），非法JAX动作0；easy/normal/hard移动一致58/69/34，放泡一致80/92/96。PASS_APPROXIMATE_TRAINING_OPPONENT，仅作近似对手，非JS策略/物理等价。父点JAX各64局kill/game=.547/.250/.141，death=.281/.375/.344，bombs/game=10.03/9.66/9.95。生产JS快速各64局safe_win=3/2/0，death_episode=0/4/0。初态/规则/终止不全相同，不作单因素引擎因果解释。

连泡RED日志 tests/SHAPING_RED.log；初轮focused15通过。真实动作/RNG/logprob/value不变、池划分、遮挡/所有者归因和相关worker回归见 tests/INTEGRATION_GREEN.log。三条control已正式u161+，finite、Actor/Critic非零；暖周期约13秒/262144步，约2万env/s/卡。完整SHA各u*/COMMITTED.json。独立唯一owner/watchdog已adopt，状态STATUS.json；旧C完成前GPU6/7保留。

## settled_kill=0解释

je04_bun_ops.retention_reward按16×surviving_kill累pending，危险settled且未死亡才付。je10_control_credit.parent_reward_events有意排除敌方受控死亡，防首次控制4/残余.25再叠旧16。因此controlled-pop的settled_kill=0是明确去重设计，不能据此断言击杀丢失；直接未受控击杀仍可进入旧pending。本轮不改奖励。此前305/304是事件数/局，可超过1，不能写成305独立胜局。

## 异步验收

u240/u320提交CPU请求：JS Hunter难中易64、自然完整categorical/greedy、JAX holdout304；dev/holdout、JAX/JS、采样/argmax分列。窗口归因→取包→同源交付；死亡/掉包/censor/未知拆分，香蕉持包自吃和胶单列。缺失报null+原因，评估不阻塞GPU。当前尚无本轮能力结论。每run完成验收原子SHA并京ME通知，技术故障保留证据；12run完结汇总，不擅扩预算。

## COMBAT-MIX-V1 旧设置复原补充（20261028）

用户追加legacy_mix三个seed，总15run、629,145,600新增环境步。直接复用JE10 ops.collect、课程40/25/35、11成员PFSP、current/四ring更新、历史g0/stage1固定对手及原弱对手/assist逻辑。匹配各seed env2048 u160完整Adam，预算与batch不变；不引入DAgger/KL梯度。详见 runs/combat_mix_v1/protocols/amendment02_legacy.json，包含固定历史输入SHA、原有两层对手比例、复原包边界与P0席位限制。已启动四组不回改；旧collector的RNG/GAE/课程重置作为复原包显式记录，不声称只改对手。实状态回归已通过，后续同样先两更新smoke及新进程恢复。

## Control 三seed固定终态

三条control均u320，完整state/policy SHA重新核验；每条新增41,943,040步、累计83,886,080步。能力结果尚待异步CPU/JS，不将训练结束当改善。

- control_s2026103051：state SHA eef84e7c83103fe7447f09587b5305a6cfa7d3cbf788ab11b7844474413aeb05；policy SHA 68a49b4d51d2ba17e9b00d449c15cb71ae850d784742e43b03a740d9937a4cd4。
- control_s2026103052：state SHA 58a427b2bb470be79744c73064ea17e04c77ed03aae1402c843355cde0e2bcb0；policy SHA 453e17d23f6f957050c71617ecf3b41468ea0fdf2f16da9116a0d81ed7007d13。
- control_s2026103053：state SHA ce6da2bc18551f241fadf53d4e4697fb35684a85466622365441a84e04f71ba3；policy SHA b26b37a6181ab9f0faa88fb2a6bc599dd9857d9f25c8fe8ab6f18e40fe205751。

GPU3–5接续课程/塑形技术smoke，其余混合组及旧C保持健康训练。完整收据 runs/combat_mix_v1/CONTROL_TERMINAL_ACCEPTANCE.json。

## 续接验收与线上身份更正（20261028）

legacy_mix seed51已完成u161真实PPO更新，finite，Actor/Critic L2分别1.008144/1.836599，新增262,144环境步、32 Adam步；state/policy原子SHA已重新验证（runs/combat_mix_v1/LEGACY_FIRST_UPDATE_ACCEPTANCE.json）。首冷周期381.91秒，不能用作暖吞吐结论；等待u162与fresh恢复，未冒称正式smoke通过。课程/塑形seed51/52已完成smoke和fresh恢复并进入正式u163+。

真实线上 models.json 与app/index已读取：默认选项是bun.coop_hunter@hard；模型不自动下载，模型目录首项bun392-je10-v38-r01-g8。此前12模型比较键online_web_default实际为selfplay3072，现明确更正，不改历史结果。已有online raw SHA19b98f2192a237e994ffd29801de7934dfe6168de3507530b19b3c7dd4d5ca45、ONNX SHAb1da0c13915e1093d18b07d1f059f579b68a7f47c8469c3b49e378c2c117dc16，与在线manifest完全相同；来源checkpoint SHAd392bf8caeb0ce149a91ce5af1c69806c556cadd69d89bef3c00758d2d3a89fc。补充单模型同协议320局比较已在CPU152–155启动，协议runs/combat_mix_v1/js_online_actual_v1/PROTOCOL.json；原12模型不重跑。

## 技术测量更正：独占资源复测（20261028）

legacy u162吞吐4752.88env/s（control的23.46%）时，旧C seed53实际也在GPU6运行：两owner在CUDA显存分配前竞态领取同卡。该吞吐样本不得当独占性能结论；先前“适配器造成瓶颈”仅假设，撤回确定归因。已完成两次finite/nonzero更新及原子checkpoint仍保留。原始SMOKE归档为SHARED_GPU_INVALID，不重复从父跑。按amendment04在独占卡从u162全状态继续u163/u164，仍计入固定u320预算，reward/league/course/batch不改；满足原85%技术吞吐与fresh恢复后才释放其他legacy seed。跨owner设备预占检查及7项队列回归通过；其他健康训练不受影响。独占probe尝试发现GPU6占用后未启动，未抢卡。

## opp_mix 固定终态首批（20261028）

- opp_mix_s2026103051：u320，新增41,943,040环境步，state SHA 50adcf7c754e9869f8da7124365fc27caa1eb84b982096dc29f30790be0bc7f8，policy SHA 35ca1d256991feb69058abec566618baf7439dc92390c7b7ec80e5133e0e057a；finite、Actor/Critic非零，dev/holdout与JS请求已排队。
- opp_mix_s2026103052：u320，新增41,943,040环境步，state SHA d658f3b17bdc3db0f5193bb0a35ed3c4b406d3b6bd9a131f3dfd81258f0443c8，policy SHA 1eb38bed4aec5a15616cd3b88eda77b840d47457b52b433c738a24b0d63b8a69；finite、Actor/Critic非零，dev/holdout与JS请求已排队。

GPU独占复测从legacy seed51 u162续接；未重跑prefix。单个seed终态不作能力胜出结论，三seed及独立评估继续。

## opp_mix 三seed固定终态验收（20261028）

三条opp_mix已全部u320、各新增41,943,040真实环境步，state/policy双SHA复核通过；未因中点能力结果提前停止。收据runs/combat_mix_v1/OPP_MIX_TERMINAL_ACCEPTANCE.json；GPU释放后已自动接课程/塑形seed53。能力固定终点评估仍异步消费，不能据训练完成宣称获益。legacy per-env包装的batch轴退化另记amendment05，旧试跑不进入有效三seed比较；直接整批原JE10适配器在CPU做实状态下一步等价性，核验后启用独立namespace。

## legacy 原生整批复原与验收（20261028）

确定根因：旧适配器vmap(n=1)把原collect的rows%2、2+rows%3及安全角色全部折叠到第0行，违反复原意图；原u161/u162数值更新真实但已EXCLUDED_ADAPTER_SEMANTIC_MISMATCH，保存于原formal目录。停止时双hash与无临时写入验收见SEMANTIC_STOP_ACCEPTANCE.json。独占复测尚在编译，未增加有效更新；不宣称已测得独占吞吐。

新适配器直接调用完整2048环境的原JE10 collect，恢复交替席位、rule2/3/4、每device rollout共享league成员及原device RNG；无重写课程/league/reward。amendment05明确此ABI和独立namespace runs/combat_mix_v1/formal_legacy_native_v2，仍为三seed同父u160→u320、每条新增41,943,040步，legacy奖励及40/25/35课程不变。新state从共同父逐byte复制，完整Adam/env/aux/RNG保留，设备RNG取原第一行后按原collect返回值续接；与本轮四臂的RNG/席位本就不同，按旧设置整体包解释。

实状态下一步等价性2项PASS（LEGACY_BATCHED_REAL_GREEN.log）：6真实状态、1tick，覆盖两席位/三rule分级，全部env/aux/RNG/action/logprob/GAE/mask/stats exact；这是有限定向回归，不代表所有轨迹证明。另10项命名空间/队列/曝光测试通过。新进程full-batch next-step恢复和独占两更新/85%匹配control暖吞吐仍需GPU验收；已在GPU2启动seed51 smoke，后两seed持久hold自动等待。CPU+JS评估的9个legacy条件已重绑新目录，不消费旧排除样本。


## ITER-PBT-V1 — 2026-10-28 用户授权的持续种群优化

ITER-PBT-V1 D 网页发布（2026-10-28，准备时间 2026-10-28T15:27:36.079332+00:00）：用户已授权发布Release及push main；默认可对战模型由 `bun392-je10-v38-r01-g8` 改为 `bun392-pbt-slot05-u720`（Bun392 PBT u720（D）），旧9条模型原样保留，Review历史默认不变。tag `web-bun392-pbt-slot05-u720-20261028`；JSON SHA256 `ed52b79d7a1cc138a1458e275364dff0369e6510c928c09461d4e46b6fe82088`，ONNX SHA256 `2380ddc1c44afc3ad75831bbfdc6bd201a7257db18687d107cdeb2ee591cffa7`；来源policy SHA256 `ed5a2b1860e8256fa34ac5a5830ff9b5c81e67a0b33d5c8a73101b3149796663`。JS Hunter保留快筛困难/普通/简单safe_win 18/27/32、死亡0/0/2（各64局，seed20261001）；JAX采样/argmax交付304/302（各304局）。这是重复自适应快筛，不作为显著性或综合晋升结论；fp32网页与bf16训练有动作差异。发布状态、最终发布时间与线上验收见 `reports/publish_best_d_web_model_20261028.md`。

- 合同：`runs/iter_pbt_v1/PROTOCOL.json`、`protocols/amendment01_async_selection.json`、`protocols/amendment02_dtype_recovery.json`；详细报告 `reports/iter_pbt_experiment_20261028.md`。
- 用户取消未启动配对补seed，不再按因果消融设计；已运行候选保留u320。旧R2/O5/储备暂停，不push/发布。
- 8独立slot；env2048/rollout128/mb8192/epochs1，DAgger=KL=0。共同可用起点为COMBAT完成u320全state，初始来源、双SHA、原生update与谱系见每slot协议；新dev排序未齐前明确临时初始化，不宣称最佳。slot种子2026110100–2026110107只标识训练随机流，不要求配对。
- 每chunk固定80更新/20,971,520新增环境步/2560 Adam步；40/80保存并发CPU评估。多次chunk无总终点，用户暂停优先；模型/Adam/env/aux/RNG/ring/fixed actor完整保存，donor累计计数与本slot新增工作量分开。
- dev20261101：真实生产JS Hunter各64+原生JAX自然categorical64，严格分引擎。保留20261001：JS各64+JAX304每模式。选择分、90%交付donor资格、top2/bottom2及差6、探索范围按合同；异步迟到cohort后续边界消费，不等待CPU空卡。
- 新最佳仅保留确认后写BEST并京ME；重复自适应保留确认的统计限制明示。生产网页模型不改变。
- 技术：focused10项PASS；首smoke int8/int32 scan ABI错误发生于更新前，证据保留，新chunk修复后验真实非零更新/有限/新进程恢复；终态通知重复unix的旧owner编排bug已修，不损坏worker。持久唯一owner看门狗、每chunk终态/技术阻塞/空卡>5分钟/确认最佳均落盘通知。


ITER-PBT-V1 首更新回填（2026-10-28）：`GPU_RESUME_ACCEPTANCE.json` 已验4个正式slot（02/04/05/07）从u320续训，Actor/Critic有限非零，完整Adam/env/aux/RNG/ring及next-step新进程恢复PASS；暖态20,082–20,605环境步/秒，连续GPU样本和原子双SHA在收据，京ME返回0。相关focused25项PASS。COMBAT候选legacy暖态比例0.69578按新50%标准接受，NumPy league pool恢复错误RED→GREEN、完整固定父对手持久化修复后u163真实提交（COMBAT amendment06）；无smoke重跑。PBT amendments03–08仅编排：实际GPU跨阶段保留、CPU四路、采用进程与监控、同slot缓存复用、真实工作量、技术尝试与有效generation拆分；均保留原错误证据且不改变已跑配方。dev父分21，control三候选23/−1/16，自然64/64接近饱和，尚无保留确认最佳，不据此晋升或改网页。


ITER-PBT-V1 接续回填（2026-10-28）：首批slot02/04/05/07的u400双SHA及每条80更新/20,971,520环境步/2560 Adam步已验，合计83,886,080新增环境步；`AUTO_CONTINUATION_ACCEPTANCE.json` 包含后继chunk已真实u401/u402非零更新的证明。首轮含冷编译、fresh恢复及保存的端到端约12,346–12,963环境步/秒，与暖态约2万分列，不以显存占用冒充计算。CPU selector在amendment09按预声明+40/+80筛点，20个技术u+1/u+2排队请求在执行前明确排除，未消费它们的能力数据；原请求SHA保留，RED→GREEN及相关26项通过。新COMBAT候选完成后自动补新dev比较，不等CPU接续GPU。持久GPU owner/CPU owner/monitor各唯一；当前证据及后续进度见EXECUTION_STATUS、STATUS、WORK_TOTALS与当日报告。


ITER-PBT-V1 08:55评估滞后修复（2026-10-28）：amendment10已实时部署；每slot仅领取最新保存点，过时未启动请求保留跳过收据，最佳holdout优先。JS全局28路、每job4分片，原生JAX新任务2路，20个CPU评估job上限，核段120–143/160–183避开训练。指定slot05_c000005_u000720 holdout seed20261001已真实产出分片；实际SUMMARY分124，未复现口述171，不先称最佳。淘汰仅完整cohort且每slot有效generation差≤2，过期继续训；报告新增保存/评分/滞后chunk。32项RED→GREEN及相关回归通过，未向GPU worker发送停机信号；两条新阶段入口hash竞态失败后双SHA验收u2002/u722并接续，训练/调度hash现分开记录；slot03_c18恢复PASS并formal非零。完整证据/吞吐局限/分片SHA见 `runs/iter_pbt_v1/EVAL_FIX_ACCEPTANCE.json` 与当日报告；未改训练配方、奖励、网络，未推送发布。


ITER-PBT-V1 09:20原生GPU评估迁移（2026-10-28，执行中）：amendment11冻结每GPU最多1评估、64/304整批、预分配关闭与MEM_FRACTION=.03、6GiB保护、训练暖态下降>5%自动降并发。CPU JS28路保持；仅chunk末评估，中点仍保存。默认GPU精度首8局交付/死亡一致8/0，5局放泡不一致，首tick概率已有差异；原始BLOCKED证据保留，最高精度复核中，未先切生产评估队列。训练保持运行，不改奖励/配方。35项测试PASS。详见当日报告及 `runs/iter_pbt_v1/native_gpu`。

ITER-PBT-V1 共卡迁移上线回填（2026-10-28）：完整CPU/GPU首8局同点交付8/死亡0/终局8一致，5局放泡不等；同obs/state/mask/keys探针及bf16 logit差异，8/8首次动作分岔由同Gumbel精确重建，按用户“可解释差异”通过，严格BLOCKED原件保留，不称逐轨迹等价。GPU owner已启动304/64整批评估；CPU已完成分片保留，新摘要显式后端。训练未停；GPU0暖吞吐下降>5%实测触发8→4并发，评估协作让GPU，收据与京ME告警保存。37测试PASS。`native_gpu/REPORT.json`持续记录每点完整耗时和lag；尚未达到lag≤2，不先称追齐。协议amendment11、PARITY_GATE及当日报告可复算。

ITER-PBT-V1 GPU评估正式首点回填（2026-10-28）：slot05_c000023_u002160已完成JS192+原生GPU64，双marker/SUMMARY SHA验收，原生64交付/1死亡、选择29，不称能力改善；slot05 lag0，整体滞后未全达≤2。原生冷启动528.4秒（两冷scan合计450.6秒、9暖scan约30.3秒）；不以暖scan冒称整点提速，GPU冷点未快于此前CPU约7分钟估计。共卡并发保护现4，显存约0.7–1.1GiB；每卡前后暖吞吐在native_gpu/ACCEPTANCE.json，8slot训练继续，holdout异步。
