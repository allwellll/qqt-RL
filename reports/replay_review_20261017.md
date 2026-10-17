# 低质量对局回放诊断验收（20261017）

## tick177 返程振荡与优势链路追加审计

本轮基线为 `54f8b7e067741c713af4fdaa276b56051bd96bd8`，同一网页工作树继续；只读冻结训练 artifact，未改训练 checkout、进程或使用 GPU。实验是在原任务奖励之外增加路线 PBRS，试图学会掉包后拾取、持包沿可达路线返己笼和交付；目标标签出现不代表动作已学会。以下保留真实失败，不改动作或录像字节，不把诊断场景冒充自然完整局成功。

### 人话结论与坐标、路径

- `tick177 / 00:17.700` 的抖动是真实策略上下往返，并非播放跳帧。生产动作 `0=向上、1=向下、2=向左、3=向右、4=停留`；位置数组是 `[行 y, 列 x]`。本 tick 从 `[1.80,9.64]` 向下到 `[1.98,9.64]`，变的是 y，x 恒为9.64。
- 己方3×3包笼左上角 `[1,4]`，当前格 `[1,9]`。中央墙阻挡直接向左，唯一最短路首步是向下，沿右侧下到第9行后从底部向左，再沿左侧回笼。独立重复松弛最短路与路线字段一致，未发现己/敌笼、x/y或目标颠倒。
- 网格距离采用 `floor(y), floor(x)` 的整格加权距离；1.80与1.98属于同格，距离均38，Phi均 `.421875`。格内前进未获得连续路线梯度，每 tick 路线项只剩折扣惩罚：`.05*(.995*.421875-.421875)=-.00010546875`，公开量化值为 `-.000105`。此数是诊断路线奖励，绝不是训练总 Reward 或 A1。
- tick177向下的真实合法策略概率为 **70.2199204%**；tick178在 `[1.98,9.64]` 却以 **55.9185597%** 选择向上，向下仅 **11.7418620%**。两步均无碰撞校正。冻结生产状态、对手与随机流的见证显示：若连续向下，tick178到 `[2.16,9.64]`，距离立即降为37。因此除了整格信号平台，还存在面对可立即进展动作却折返的策略失败；概率和路径见证不能证明训练因果。
- tick177已连续 **61 tick 距离未下降**，包括早前偏航导致35升至38，不是61tick距离始终为38。位置周期2从tick134起；保守检测须同目标返程、持敌包、存活、至少20tick无进展，且最近10个位置严格周期2或静止。确认振荡区间143–590；停滞599–1000、1010–1160、1170–1199。仅整格距离不变但人物仍连续前进，不会判成振荡。
- 终态更正：此前本报告写“平局”有误；生产Sim原始终态 `winner=0`，笼内存包为模型 `[1,0]`、对手 `[0,0]`，按原生存包超时判模型胜；模型仍携敌包、交付为0。这不是自然返程交付成功。公开录像本身及其正确终态未改。

固定用户定位窗口 **tick147–207（前后各30tick，61行）** 的逐行审计见 `web/review-evidence/route-g32-return.json`：包含每步动作、前后坐标、完整网格路径及首步、move/ability合法mask、实际Actor概率、事件、距离、244个合法联合一步分支的坐标/距离/hash。没有完整状态、全量Q表或模型张量。CPU ONNX重新推理整局1200动作全部一致；窗口独立生产JS forward概率最大差 `9.595731851330669e-7`；taken反事实分支与真实状态hash完全相同，随机流通过draw count恢复，固定对手动作。当前窗口无关键事件。

### A1尚未补齐：有Critic，缺严格训练ledger适配

所需定义保持 `Q1=r+gamma*V(next)`、`A1=Q_taken−sum(pi_legal*Q1)`，合法联合 `move×ability`（ability含放泡/道具）概率重归一，对手冻结；终局value为0。两种Critic若未来可算，将使用同一v35 Actor概率，以免把策略变化混进Critic差。跨Critic差 `A1_shaped−A1_original` 与时间增量 `A1(t)−A1(t−1)` 分开，首项/缺口时间增量为null。当前窗口及本局原始/塑形Q、策略均Q、A1、两种差值均null，不以reward/value/logit代替。

| 角色 | checkpoint SHA-256 | Critic证据 SHA-256 |
| --- | --- | --- |
| H8 v34 g32 原始任务 | `b491758761b32872555555f08cf4e42f9cab120215501509a2bdbf905e64ce11` | 参数规范摘要 `7002b1b4f40d87021462d69a4bd3594964698ce2608592c5c4e17db6d1d5cf85` |
| H8 v35 g32 塑形任务 | `e141311268c6eaf419f1f651252bb27d46c106aeb407230c20eed4c8897a34e2` | NPZ文件 `3174b62f39ea9d56f206c917d2107a8f5e31bda2fe693e88630f72715c0604d8` |

追踪训练结果提交 `6358fa192c847016bf1745247efae2027804f549` 的不可变 Git 对象并按 `SOURCE_SEAL` 核验实际训练路径（不依赖训练工作树后续编辑）：PBRS collect → H8/H6/H5c collect → H2 observed_reward → H/F combined_reward → PBRS.apply_reward → PPO/GAE。gamma=.995、lambda=.05来自封存配置。具体缺口如下，不能把“Critic存在”误当“Q已测量”：

1. 本诊断仅在外部Tracker预置trapCause，原生Sim tick1 causalDamageSource/creditedKill/causalKill均false；训练scene9却在native state预置trapped_source/trapped_causal。Tracker credit与native info不是同一因果ledger。
2. 完整训练reward依赖scene、terminal/settlement、drop nonce/credit/cycle_paid、transport/bridge、carry_items、potential/routing等隐藏aux。源诊断没有初始化或采集这些训练aux，也没有声明JS→训练精确适配版本。训练COMMITTED批状态不属于本seed诊断episode。
3. 训练PBRS有阶段边界、topology/safety、terminal Phi归零；诊断使用转移后目标offset且末Phi未归零，不能把公开路线reward加到旧env adapter后称实际train reward。
4. 已用旧冻结authority CPU JAX适配器对tick176跑一步分支。合法动作0/1/2/4的位置与JS误差约3e-7以内，但blast causal/trigger linger、kill_window、milestones、danger/tactical标记、trapped_source/causal等未完整映射。位置parity并不证明reward/critic观测/隐藏状态parity；非法3本来不参与比较。

下一步所需是版本化的完整JS-native event到冻结训练state/info桥、声明该诊断初态aux并逐tick推进ledger、对全部合法分支克隆ledger/RNG、两代同源Critic输出和独立reward/Q/A复算。此处缺的是可验证工程链路，不宣称数学上不可能；本轮不任意补隐藏状态、不启动训练。用户要求的真实A1数值是明确未完成项。

### 页面与验证

- 新增返程振荡/停滞中文警示与可点击失败区间轨，不把“目标已识别”表述为有效靠近。暂停详情展示真实候选概率、合法动作、一步位置/距离、两代Critic来源和具体缺失链；单步178能看到合法向下38→37与实际向上的差别。
- 原始/塑形优势、两种增量、Reward、DAgger、授权、标注、深链、蓝色对手、中文Reward、多场景及浮点平滑保留。画布→播放控制/进度条→多轨时间轴DOM顺序保持。旁证单独按局懒加载并核hash；原录像SHA `207680e66de70c44652586fdc7f3aab1919bf5268a8f3e8ef88dcd95fdce69f2`、162526 raw / 7746 gzip bytes未改，旧标注继续绑定同hash。
- 审计旁证SHA `3e9446f928f30a80273d8eaad979067b8a42e4233edfdeca144844477c37ef4e`、**106126 raw / 16175 gzip(level9) bytes**。生成脚本 `scripts/audit_route_return.js`，只读固定artifact；重复生成字节一致。source、Actor、Critic、simulator及反事实摘要hash全部在旁证内。
- 新测试先RED再GREEN：真实tick177判定、整格平台不误判、动作坐标/路径/概率、手算策略加权A1与非法排除、Critic不可混用、首项/缺测delta null、hash绑定与伪造Q/A拒绝。`npm test`、`npm run test:review`及桌面/手机旧授权/标注/DAgger/多场景/深链回归通过。
- 完整1200tick，4x桌面/手机、平滑开/关四组共7203显示帧；约60FPS，安全连续最大帧位移模型/对手：桌面12.39/12.31px、手机6.46/6.43px（4x）。异常跳变、alpha重置、浮点覆盖丢失、console/HTTP错误均0；合法碰撞/携包等snap保留。实际双方renderer锚点严格单调探针、pause/seek/step alpha=0验证通过。
- 真实桌面1600×1100与手机390×844深链177验收：61tick警示、失败轨点击177、单步178、候选表、null解释、授权刷新/新tab、无横向溢出和单局懒加载通过；pageerror/console/HTTP均0。公开截图及小检查结果位于 `web/review-evidence/return-177/`，未使用网络mock或fixture。

这例说明路线目标和奖励已经存在也不保证学到稳定返程，整格势函数可能形成局部平台；但单段反事实不能把全部训练退化归因于离散距离。结合fixed32/full交付未改善及攻击指标回落，维持“不建议lambda=.05原样续训”的结论，不自动改变训练。

发布按同一head等待Actions/Pages成功并核公开字节/hash/尺寸及真实浏览器；最终SHA验收写入忽略目录，不为记录自身SHA再次触发部署。


## 真实路线 g32 接入与播放跳变追加审计

下文既有审计按历史保留；此前真实 artifact 阻塞已解除。本轮从 `087e48ee073db885bc08e4038a27cb0fd63c7d38` 继续，远端 main 为 `167be513889569bd4835184e06cf66df1aa995fa`。训练结果来源 `6358fa192c847016bf1745247efae2027804f549`；采用无提交 cherry-pick 集成内容，保留 main 总账并只加入本实验条目，纠正该条 Hunter normal 与终态报告的不一致。发布提交使用项目维护者身份，不引入来源提交中的个人作者元数据。没有创建 workspace、启动/控制训练、读取可变 checkpoint 或使用 GPU。

- 只读核验真实包 Actor JSON、ONNX、policy、同代 Critic NPZ、配置、trajectory、recalculation_proof 的全部字节哈希。另从封存 g32 `COMMITTED.json` 及 `state.pkl` 独立核验 policy/state 哈希，CPU 读取 `live.params.critic`，75 个参数叶与 NPZ 全部精确相等且 finite。Critic 是本实验塑形 Critic，不是原始任务 Critic；参数相等不等于反事实 Q/A 已测量。
- `scripts/import_route_pbrs_g32.js` CPU ONNX 逐 tick 重推 1200 个动作全部一致；与原版本生产 Sim 的每个前后快照全部一致。独立重复松弛最短路算法重新计算目标、路程、连续 offset/Phi、delta 与 reward，1200 tick 最大残差 0。原始 verifier 的状态链/公式也通过。完整再次生成字节相同，seed 固定 `2026101732`，不看结果筛选、不改动作。
- 原生产 Sim SHA 为 `9aa8385489affdfdb61c1deff61cee3d8401e744cf80f2b474feb931103884e3`，与当前生产版本不同。按哈希加载字节不变的 `web/review-sim/route-v35.js`，旧录像继续使用原来的当前 Sim；共用生产 renderer，不创建近似 renderer。周期 state hash、seek RNG 重建、终态和 identity 校验通过。
- 明显标注“诊断场景，不代表自然完整对局成功”。这是预声明 drop-return 初态，敌方持己包、trap=1、归因预置，assist-off；不是模型自然击杀。源 manifest 对手文字误称 stationary，真实动作是 `[0,0,0]`（固定向上、不放泡/道具），页面按动作真实标注。终局 1200 tick 平局，仍持敌包，未交付。
- 路线导出器不是完整训练 PBRS ledger：按转移后目标分类，末 tick Phi `.421875` 未归零。扩展语义 `diagnostic-post-target-uncorrected/v1` 保留原值，明确 terminal correction=false，不称训练总 Reward；原协议对其他录像仍要求 terminal Phi=0。前后距离按转移后同一目标独立复算；topology/safety/invalidated 等未采集字段保持 null。
- 原环境 Reward、训练总 Reward、原始任务 Critic A1/ΔA1、塑形 Critic A1/ΔA1 均未记录。该轨迹没有训练 reward ledger 与严格联合动作反事实结果，因此不从路线 reward 推断、不把塑形 Critic 冒充原始 Critic。已发布六局原始 A1/Q/ΔA、Reward、三种真实场景及十二局历史文件字节不变。
- 新增真实录像 1 局，raw **162526** / gzip(level9) **7746** bytes，SHA `207680e66de70c44652586fdc7f3aab1919bf5268a8f3e8ef88dcd95fdce69f2`。每局独立懒加载，最小双方动作、量化路线列、目标字典和周期 hash，没有逐 tick 完整 state、模型张量或全动作 Q 表入 Git。来源/复算小型证据为 `web/review-evidence/route-v35-g32.json` 与 `route-v35-critic.json`。
- 页面结论使用终态报告：32代 / 67,108,864步；g32 诊断 pickup16/16、25%7/16、50%6/16、home6/16、capture2/16；fixed32 主评/holdout capture 均0/16；full2400 pickup/capture 均0/16；Tactical causal kill12→9/64，Hunter hard9→1/64，不建议当前 lambda=.05 原样续训。不得从本录像宣称自然完整链成功。

### 跳变 RED 与根因

真实 g32 前240 tick，桌面/手机、平滑开关分别逐帧采样实际 production `drawImage` 坐标。桌面平滑开时约60.06 FPS，最大连续安全帧位移约3.10px，但放泡 tick3/21/37/61 出现约18.55–20.53px位移；手机相应约9.68–10.67px。相邻 next state 始终可用，alpha 在播放中没有重置，CSS 比例恒定，浮点 override 在所有采样帧均完整到达 renderer。因此没有证据支持缺少预读 tick、整数取整或 Canvas/CSS 缩放为本次原因。

根因是旧门控把 `bomb` 一律当作人物不连续事件，并把自身起始格新泡泡也一律当作路径拓扑变化，导致真实安全直线移动整个 tick 静止、边界瞬跳。`web/test_review_motion_regression.js` 在真实 tick2 先失败 RED。修复只对已记录自身放泡、真实新泡泡归属和生产 `_tryMove` 全路径扫掠均验证通过的直线移动允许插值；泡泡及所有其他权威字段仍在整数 tick 切换。死亡/复活、携包变化/拾取/掉落/交付、控制、碰撞对齐/侧移/截停、速度阈值、异常位移和终局仍 snap。显式输出每方 `snapReasons` 及 source/next tick，便于把合法 snap 与视觉错误区分。

原播放器已预读独立 next Sim，修复不额外延迟动作或改证据 tick。暂停/seek/单步清零 alpha，吸附当前整数 tick。开关显示“平滑播放”，保留实际 FPS / 10 tick/s / alpha。真实头顶多行信息与靠近的对手冲突时，将文字组移到空侧并画引线，保留蓝色对手和文字身份，不遮住人物。

`npm test` 全量生产回归与 `npm run test:review`（19局、488项原协议检查、真实来源/紧凑 schema/数学/单调移动/合法 snap/旧功能）通过；原训练来源同版本的10个聚焦 CPU 测试通过。修复后再次通过真实 g32 页面与三旧场景、授权/标注/DAgger/深链的桌面手机浏览器回归，pageerror/console/HTTP 均为0。

### 完整播放验收

- 无 mock 的真实 Chromium 桌面1600×1100和手机390×844：4x 播放完整1200tick，平滑开/关四组各1801帧，实测约60.03FPS；另以正常1x播放相同前240tick，四组各1444帧，约60.05–60.06FPS。完整录屏和逐帧原始采样保存在忽略目录 `runs/replay_review_20261017/motion-route-after-full/` 与 `motion-route-after-1x/`。不是通过减速、模糊或隐藏人物获得结果；场景原生屋檐遮挡仍沿用 production z-order。
- 两方五组真实安全片段（模型放泡前tick2/20/36/60、对手tick101），逐一实际调用 production renderer，8个 `drawImage` 锚点严格单调且有小数；Sim state不变。正常1x采样分别观察到模型112个、对手6个具有多个严格单调中间绘制点的tick组。4x每tick通常只分到1–2帧，因此另用确定性真实片段证明中间位置，不拿帧数代替坐标验证。
- 正常1x，放泡tick3/21/37/61的模型最大连续帧位移：桌面修复前18.55/20.53/20.49/20.48px，修复后3.08/3.10/3.08/3.08px；手机修复前9.68/10.67/10.66/10.65px，修复后1.61/1.62/1.61/1.61px。正常1x全部安全连续帧最大位移：桌面模型/对手3.10/3.08px，手机1.62/1.62px。4x对应12.39/12.31px和6.46/6.43px，按真实倍速比例增加。
- 仍保留合法snap。4x桌面最大单帧21.42px在碰撞侧移边界，手机最大11.61px在拾包边界；完整列表包含tick10/18/23/59/64/116/117/118的碰撞侧移和tick80的携包状态变化，未将其强行平滑。死亡/复活通过权威事件单独记录；角色不可见期间不伪造精灵位置。末tick终局也保持整数边界。
- 相邻显示帧扫描未发现超出安全速度上界的位移、缺少相邻预读状态、浮点覆盖丢失或播放中alpha重置；八组 `unexpected_jumps=[]`、scheduler gaps=0、console/HTTP错误=0。暂停/seek/双向单步在tick79/80/104/115/116/500逐项验证：alpha=0、visualPos完全等于权威pos，实际绘图锚点也一致。开关关闭时保留明确的离散10tick/s画面，作为对照。
- 公开小型报告 `web/review-evidence/route-g32-motion.json`（47161bytes）包含测量定义、每组最大位移、合法snap的tick/分类、双方严格单调锚点、原始录屏采样摘要SHA和缺测范围，不发布巨量逐帧state。真实页面截图/检查结果位于 `web/review-evidence/route-g32-browser/`，包含桌面/手机整页及画布，未使用fixture或网络替换。

本地跳变修复与原合同验证完成后才恢复提交/发布步骤。发布后按同一head SHA等待Actions/Pages终态、逐文件HTTP/hash/尺寸与线上桌面手机验收；最终同SHA记录保存到忽略目录，避免为记录自身SHA再触发一次部署。

## H8 新批次预声明与平滑根因审计

- 新任务起点 `41b16967f33fc2c6538cc5b717ef6263b0cd67f0`，远端与当前分支一致且干净；保留此前审计。仅只读已完成 H8 v34 g32 的固定 COMMITTED、policy 与生产 Web 导出，不读取 LATEST 或训练中 checkpoint，不使用 GPU。
- 在运行新批次前冻结 24 个 seed：2026101740–2026101751（前 6 原图 Hunter normal，后 6 清砖图 Hunter normal）；2026101760–2026101771（前 6 原图 Hunter hard，后 6 原图 Tactical v2）。上限 2400 tick，真实 greedy 联合动作，禁止修改动作。沿用原先六类失败规则，每类最早 seed 优先，再按 seed 补足最多 6 局；缺少的失败类型如实报告，不追加看结果挑 seed。
- H8 g32 policy SHA-256 `b491758761b32872555555f08cf4e42f9cab120215501509a2bdbf905e64ce11`；CAMPAIGN_COMPLETE completed/actual_generation=32，COMMITTED passed/generation=32；生产导出 ONNX SHA-256 `0c4df18e853a833476afa8e18fce660645ed4075974025e051eae3f2f59972fa`。
- 先线上采样真实生产画布，963 帧，CSS 比例恒定 0.7；双方实际 drawImage 坐标全部整数，旧测试只验证 fraction 变化。前 200 tick 模型移动 185 次中 58 次、对手移动 80 次中 15 次被吸附，含正常贴墙或离开泡泡。无相位重复重置证据。原始坐标证据保留 `runs/replay_review_20261017/motion-baseline/`。
- critic 审计：冻结 worker 保存 policy.pt=host.params.actor，训练 critic 为独立参数树；Web exporter 仅导出该 actor。actor 内 head_wv 不证明是训练 critic。当前生产导出缺少可验证的 trained critic，因此本批 A1 / taken Q / policy mean Q / advantage delta 一律 null，绝不由 reward/logprob 估算。生产 Sim 的随机性均来自 seeded rng，可从起始随机流与 draw count 恢复；reward 另有跨 tick ledger，未经同状态 ledger 分支验证也不可宣称反事实等价。本批不伪造 rollout GAE。
- 新生成器移除重复 trap timer 注入，直接采用生产 `_getLaggedInputs` 的 bun24-trap-timer-v2 编码；历史 H7 文件保持原样，历史动作重推证据仍对应其当时生成器版本。

### H8 与浮点渲染验收

- 原因已从线上实际 drawImage 坐标复现：人物/阴影坐标取整，旧 AABB 门控把相邻实体和脚下泡泡当作整段不可插值；并非只在 tick 刷新或 CSS 比例变化。修复向生产 renderer 传入独立浮点 `visualPos`，不修改 Sim。沿用时间戳累积器与 rAF，前瞻状态只用于安全人物位置；生产 `_tryMove` 校验直线全部中间点，真实碰撞截停、侧向校正、死亡/复活/携包/交付/控制/放泡及变化格仍吸附。真实策略停顿、反向及状态断点仍会可见，不人为平滑动作或事件。
- 旧线上 SHA 的生产坐标断言真实触发 RED（整数精灵丢弃小数位置）；新页面双方均在同一安全 tick 画出 8 个严格单调的中间锚点，包含非整数位置，权威状态不变。桌面 H7 同录像前后对照各采 963 帧，CSS 比例均为 0.7；新 H8 手机 390×844 采 483 帧，比率恒定 0.411111，双方都出现真实中间绘制坐标。相位未在同 tick 重置。`motion-red/`、`motion-baseline/`、`motion-h7-green/`、`motion-h8-mobile/` 保存详细证据。
- 页面显示实际 FPS / 模拟 10 tick/s / 插值 alpha；暂停、seek、单步清零 alpha，保持整数 tick、精确局内时间及 URL 定位。蓝色对手、中文 Reward、授权记忆、标注与深链均保留。关键事件轨按 tick 合并死亡/控制/包子/失误标记，每个显示 R 数值；悬浮与详情列非零中文分量。同 tick 多事件共享该 tick 证据，不声称因果分摊，不重复求和。放泡仍是权威状态事件，沿用此前关键事件轨的筛选范围。
- 优势定义为联合 move×ability 的策略加权一步优势：`Q1=r+gamma*V(next)`，`A1=Q1(taken)-Σ pi_legal Q1`，合法候选策略概率重归一；终局 next value 为零。`gamma=0.995` 来自冻结 H8 plan。metadata 记录计算 schema、checkpoint / policy export / simulator hash、critic=null、合法动作口径及缺测原因。本批 `advantages` 为版本化稀疏量化列式空记录，读取任何 tick 的 Q/A/delta 均为 null；没有导出全 Q 表、logits 或伪造轨迹优势。未来有效记录必须校验 A1 公式、量化误差、相邻 delta、首项/断档 delta=null 及反事实证据 hash。
- 预声明 24 局全部 CPU/JS 真实重跑；按原规则选出 6 局，覆盖 enemy_drop_unpicked / window_closed_empty / return_deviation / self_bomb。未观察到 cage_no_delivery / stagnation；没有为覆盖失败类型改动作。6 局再次真实推理，双方逐 tick 动作、状态 hash、终局完全一致，ONNX/JS logits 首次最大误差 `0.000005449342688645942`。独立 Python CPU 进程复算 6 局的生产 Reward ledger 完全一致；仍明确它是冻结 ledger 在真实 Web 轨迹上的重建，不是训练 episode telemetry。
- 全量重新生成后 catalog 与 6 局文件字节完全一致。旧 6 局文件字节不变，catalog 默认 H8，可切换历史 H7，旧 replay+tick 深链和标注仍绑定原始文件 hash。每局独立懒加载，静态 metadata 一次，最小动作、稀疏 Reward、空优势列和周期 hash，无逐 tick 完整 state 或全动作 Q 表。

| 新 Replay seed / 地图 / 对手 | ticks | raw bytes | gzip bytes |
| --- | ---: | ---: | ---: |
| 2026101760 / 原图 / hard | 2400 | 43881 | 8635 |
| 2026101740 / 原图 / normal | 1783 | 41637 | 9037 |
| 2026101742 / 原图 / normal | 2400 | 53502 | 10135 |
| 2026101741 / 原图 / normal | 2400 | 43788 | 8313 |
| 2026101745 / 原图 / normal | 2094 | 48562 | 9612 |
| 2026101750 / 清砖 / normal | 2344 | 46618 | 9511 |
| 合计 | | 277988 | 55243 |

- catalog（含历史摘要）11358 / gzip 1765 bytes。筛查明细见 `reports/replay_screen_h8_20261017.json`，逐局 provenance/hash 见 catalog 与录像 metadata；尺寸按实际 UTF-8 与 gzip level 9 测量。
- `npm test` 生产全量回归通过；`npm run test:review` 覆盖新旧 12 局共 312 项原协议校验，另含浮点严格单调/断点吸附、19 项 Reward 翻译、A1 手算/策略加权/非法动作/终局/首项与断档增量/稀疏 schema/hash、DAgger 与授权回归。真实 Chromium 桌面与手机通过播放/暂停/FPS/alpha、4x、seek/逐步、事件 Reward、优势缺测、标注持久化/编辑/删除/导入/导出、深链及授权记忆；pageerror / console error / HTTP error 为零。证据在 `browser-h8-local/`，发布后同 SHA 证据另存 `release-h8.json`、`browser-h8-public/` 及 `release_h8_acceptance_20261017.md`，不覆盖既有审计。
- 提交范围 `git ls-files`（含本次已暂存文件）隐私扫描 0 项，`git diff --check` 通过；未修改生产模拟器或训练目录，未调用 GPU。

## 冻结协议

- 独立分支 `feat/je09-replay-review`，初始 HEAD `6e5f69698889187e1ed07e89ce05bd4d0a475128`，初始工作区干净。
- 仅 CPU 离线推理；不读取 H8 checkpoint，不修改训练目录，不停止训练。
- 最新完成 H7 v33 g32，checkpoint SHA-256 `779fa8a20587f2df7e6e34242ce23cfba16183108fe57c4618e22b4b247ccc63`；COMMITTED 与 CAMPAIGN_COMPLETE 均通过。
- 预声明 seeds：2026101700–2026101711，共 12 局。前 6 局原始 806 图，后 6 局明确声明的清砖 806 图；对手均为真实 Hunter normal；最长 2400 ticks，原始动作不修改，跑到模拟器终局。
- 预声明失败规则：真实 enemy drop 后 120 tick 未拾；运输窗口关闭仍在空手运输阶段；持敌包返程距离连续倒退；到己笼仍不交付；自身炸弹死亡；连续 100 tick 未移动。每类优先最早 episode，最多 6 局，其他已运行 episode 摘要保留。
- 本批为定性诊断，不作为能力提升或统计显著性证据；不会人为制造掉包/持包/死亡。
- 首批 12 局仅观察到“窗口关闭仍空手”，其他类别不伪造。补充运行前声明固定 seeds 2026101720–2026101731：前 6 局真实 Hunter hard、后 6 局真实 Tactical v2，均原始 806、相同 H7 g32、上限 2400 tick；相同失败规则与最多 6 局筛选上限。连续返程倒退按连续改变的距离样本计数，等距 tick 不重置方向。
- 高阶设置是静态页面 UI 门控，不是服务端鉴权，无法防御可读源码的攻击者。只保存版本化授权标记与不可逆校验摘要，密码不写入存储；清存储或授权规则升级后重新验证。

## 验证结果

### 真实录像与边界

- 24 局均为冻结 H7 g32 的 CPU greedy policy 与真实 Hunter / Tactical 对手，运行至模拟器终局。原始 episode ID、seed、模型/对手/模拟器/地图/checkpoint SHA 保存在协议与筛查报告中。没有修改动作，未访问训练中的 H8 checkpoint。
- 最终按预声明规则选出 6 局：观察到 `window_closed_empty` 和 `return_deviation`；未观察到 `enemy_drop_unpicked`、`cage_no_delivery`、`self_bomb`、`stagnation`。窗口标签表示对手死亡/控制窗口结束时仍空手，不声称测得模型内部 FSM。胜局也可能有具体失误，不把全部入选局称作败局。
- 6 局重新真实推理，双方逐 tick 动作、周期状态 hash 和终态完全一致；ONNX 与生产 JS forward 的首次 logits 最大误差为 `0.000002533833360462978`。
- Reward 为冻结 H7 原始 ledger 作用于真实 Web 观测轨迹的重建值，沿用 CPU safety detectors 与原生事件，桥接参数 `window=120, scale=2`。19 个 component 每 tick 求和与原函数 reward 一致；新进程复算稀疏 reward 一致。它不是训练 episode telemetry，也不声称训练/Web 轨迹完全相同。教师 eligible/query/accept/reject 未采样，页面明确显示“未测量”。
- 复用生产 `QQT.Sim` 与 `QQTVisual` 画布。协议每 tick 仅双方四个最小动作，reward 仅非零 component delta，附关键事件；每 200 tick 和终态存状态 hash。Seek 的完整状态只在浏览器内存由生产模拟器重建，不写入录像。目录轻量，每局独立懒加载，未打进 JS bundle。

| Replay（seed） | ticks | 原始 bytes | gzip bytes |
| --- | ---: | ---: | ---: |
| native-2026101700 | 1380 | 26254 | 4377 |
| native-2026101705 | 2359 | 41066 | 5317 |
| native-2026101701 | 2400 | 37912 | 5056 |
| native-2026101702 | 1426 | 26898 | 4346 |
| native-2026101704 | 2400 | 42638 | 5449 |
| native-hard-2026101720 | 2400 | 37411 | 4824 |
| 合计 | | 212179 | 29369 |

尺寸使用实际 UTF-8 文件与 gzip level 9 测量。逐局公开 SHA/大小见 `web/review-catalog.json`；全部 24 局筛查与重新推理证据见 `reports/replay_screen_20261017.json`。

### 本地验收

- `npm test` 全量生产回归通过；最终 `npm run test:review`：6 局、156 项 schema RED/GREEN、非法动作/长度/reward 和/hash/seek/标注检查通过；设备授权单元测试通过，HTTP smoke 与 `git diff --check` 通过。
- Chromium 无网络 mock：1440×1050 与 390×844，生产画布采样 1660 种颜色、Lucide 图标 134 个非透明采样像素，移动端无横向溢出、控制重叠；诊断页 pageerror / console error / HTTP error 均为零。
- 播放/暂停时间保持、seek、双向单步、4x、缩放、reward 与曲线、标注持久化/编辑/删除/导出/导入、错误 hash 拒绝、复制定位与 URL hash 深链均通过；精确定位验证为 `tick 123 · 00:12.300`。仅选中录像被请求，切局懒加载通过。
- 高阶设置真实浏览器覆盖：首次提示 → 错误密码无授权 → 正确验证 → 刷新免输 → 新 tab 免输 → 主动忘记后重新提示 → 授权版本与验证规则变化失效 → 新浏览器 context 重新提示。原密码明文从当前源码移除；localStorage 只存版本、验证规则摘要与标记。存储不可用时只保留当前页面内存授权。
- 本地详细证据保存在忽略目录 `runs/replay_review_20261017/`：`inference-verification.json`、`browser/checks.json`、`browser/desktop.png`、`browser/mobile.png`。权重、逐帧原始 trace、浏览器标注和本机配置均不提交。
- 额外运行的旧 `verify_round2_browser.js` 在结算关闭后点击画布重开的断言失败；原始脚本配合原始 `ui_panels.js` 复核也在同一断言失败，属于已有验收缺口。本次设备授权流程已通过独立真实浏览器验证；不将旧脚本声称为全量通过，不扩大修改结算流程。
- 提交前 tracked + 待提交文件隐私扫描为 0 项；当前受追踪内容中原密码明文为 0 处。3 个既有浏览器验证脚本改从 `QQT_SETTINGS_PASSWORD` 环境读取，旧文档移除密码明文并保留原验收语义。未改写历史提交。

公开目标：<https://allwellll.github.io/qqt-RL/review.html>。提交后按实际发布 SHA 等待 Actions/Pages 成功，再校验公开文件字节/hash/尺寸并运行线上真实浏览器；最终发布记录保存在 `runs/replay_review_20261017/release_acceptance_20261017.md`，避免为记录自身 SHA 再次部署。

## 视觉与可读性追加验收

- 基于已发布 `5d2c0f1` 同步分支，保留以上录像、筛查与真实性审计。6 局录像和 catalog 字节不变，未读取新 checkpoint，未使用 GPU 或触碰训练工作区。
- 仍复用生产 `QQTVisual` renderer。核对仓库原生角色资源后，没有可直接使用的蓝色毛毛 skin；通过 renderer 可选 palette 对现有毛毛帧的红色区域缓存换蓝，保留肤色、白色和透明度。未复制外部仓库资产。模型 P0 保持原生毛毛；对手 P1 为蓝色毛毛，画布头顶、标题与侧栏同时标注“模型 / 对手”，不依赖颜色辨认。默认游戏不传 palette / 标签时沿用原行为。
- 时间轴四行常驻“阶段轨 / 关键事件轨 / Reward 轨 / 我的标注轨”，每行有简短含义和悬浮说明，行首独立于横向滚动区，缩放和横向滚动不移走标签。
- 默认目标 60 fps，可选 30 / 60 / 120 fps，显示实际刷新采样与 10 tick/s 逻辑频率。rAF 只在相邻整数 tick 间插值安全人物位置；权威模拟器、动作、reward、事件和物件仍读取当前整数 tick。暂停、seek、单步清零插值。死亡、复活、控制、携包、道具、HP、移动状态、保护阈值、异常位移、终局、碰撞盒接触实体或变化格均保守吸附，防止穿墙或虚构动作；可随时关闭平滑插值。
- 新增显式 19 项 reward 中文名称及释义，覆盖全部 6 局双方 metadata 与非零记录；右栏、曲线选项和时间轴 tooltip 使用中文，英文原名可选或悬浮查看。未知项显示“未翻译（英文键）”并醒目标记，映射覆盖测试失败，不静默回退英文。
- `npm test` 全量生产回归通过；`npm run test:review` 的 156 项协议检查、19 项翻译完整性、palette、连续/不连续状态边界、终态 hash 和设备授权回归通过。真实第一局 2760 个双方转换中，1064 个允许插值、1696 个保守吸附，终态 hash 不变。
- 无网络 mock 的 Chromium 桌面 1440×1050、手机 390×844 验收通过：默认 60 fps、rAF 多帧插值、同 tick 权威状态不变、控制断点不插值、4x 平滑播放与整数 tick 暂停、精确 seek / 双向单步、时间轴固定行首、中文 reward、红色模型及蓝色对手真实画布像素检查。授权记忆/撤销/版本失效、标注持久化/编辑/删除/导出/导入、错误 hash 拒绝、定位复制和深链全部通过；无横向溢出，诊断页 pageerror / console error / HTTP error 均为零。
- 本轮本地截图、帧数和像素证据在 `runs/replay_review_20261017/browser-visual-local/`；同 SHA 发布证据与线上浏览器结果分别保存为 `release-visual.json`、`browser-visual-public/` 和 `release_visual_acceptance_20261017.md`，不覆盖原始审计。显示帧率为目标值，实际刷新取决于设备和浏览器。

## DAgger 追加验收

- 从干净且与远端一致的 `aae0565` 继续；该 SHA 的 Actions/Pages 成功，19 个公开资源字节/hash/尺寸一致。保留原生模型、蓝色对手、60 fps 插值、中文 reward、授权记忆、标注和深链。本次不改变模拟器、renderer、录像或录像 catalog；训练目录仅只读质量 JSON、已完成 COMMITTED 元数据及 smoke 日志，未读取 H8 checkpoint、访问 GPU 或操作训练进程。
- 六局本身没有逐 tick DAgger telemetry。新增本局摘要、右栏及固定中文“DAgger 轨 / 教师与梯度采集”，均明确“本局逐tick DAgger：未记录”，不会把 reward、事件、阶段或 run 汇总转换成采集记录。H7 汇总通过 COMMITTED 的 policy SHA 与六局 checkpoint 精确关联；H8 smoke 和 H6 历史参考明确与本局/本 checkpoint 无关。
- `web/review-dagger/` 独立小 catalog 与三份懒加载汇总，另有按需下载的原始质量快照和白名单数值日志投影。所有 source run/path 为项目相对路径，保留来源 schema、原始字节 SHA-256、来源生成时间与转换时间。H8 funnel 本身没有 schema/生成时间，保留 null；投影标明自身 schema 与原始日志 hash，并使用实际末代日志时间，不伪造原 artifact 时间。

| 来源 | 范围 | 候选 | proposed | strict accepted | Actor 梯度查询 | 接受率 | 最终监督覆盖 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H7 v33 g1–g32 | 同模型 run，formal + smoke 快照 | 2342320 | 695980 | 389036 | 2970 | 55.90% | 0.13% |
| H8 v34 attempt06 g1–g4 | smoke；严格 A 链验收失败 | 196554 | 139554 | 53261 | 296 | 38.17% | 0.15% |
| H6 v28 g1–g32 | 历史参考；accepted 为 query 下界 | 2035270 | 1102974 | 995127 | 995127 | 90.22% | 48.89% |

- H8 无完整 DAGGER_QUALITY，smoke artifact 与四代实际日志逐项核验：stock theft 50、enemy-carried drop 44、enemy-origin pickup 0、enemy-path query 0、own-recovery pickup 2、A 课程 Actor query 26。26 与全 smoke 梯度查询 296 分开显示。原 funnel 的 `passed=true` 与 `strict_declared_acceptance=false` 原样保留；页面以严格条件明确显示 FAILED，缺少 native loose pickup / enemy-origin pickup / enemy-path query。己包回收不能满足敌源要求。`a_loose` 分段沿用来源未区分的聚合，不声称 A 真实敌源追拾成功。
- H6 分段 accepted 为 A 71549 / B 371152 / 返程 540012 / 交付 12414，污染 0；全部从 JSON 重读。H7/H8/H6 的明确定位和拒标细因、教师短反事实存活/进展、专家一致率保留 null，显示“未记录”；合法率仅显示 artifact 或实际日志计数支持的数据。固定语义 fixture 不冒充模型识别精度。
- 来源快照 SHA-256：H6 quality `aab926efa2cb1ca461b88d8123190af6de54fb08e4dc39b46aa4dfd419e164a0`；H7 quality `65d5f16f6bf850aaefd039ce054fb200e3b3cc6d7b9beab36dbbcb94d613b184`；H7 COMMITTED `d283f52d0ce8be4d1b472d62ede1d55008213f43732d9a0844a8c8a6c1f52aeb`；H8 funnel `25875950b5e2efd7b1e05c7bb2fe9ab884ebd3719db96fa45c2ad0ce0dd1a513`；H8 原日志 `2e947f3bbd06012ba18595e03291ef077c9341ef4b3af0428dfd16e22356b76e`。公开投影保留原日志 hash，完整日志不进入网页 bundle。
- DAgger catalog 为 1086 / gzip 557 bytes；H7 汇总 3055 / 1264、H8 汇总 3306 / 1344、H6 汇总 2677 / 1095 bytes。6 局录像仍为 212179 / gzip 29369 bytes，逐局大小/hash不变。
- 回放 v1 增加可选 `dagger_events`：按 tick/player 唯一有序的点观测，仅允许 tick/player、片段枚举、nullable boolean 的 candidate / explicit_localization / teacher_proposed / strict_accepted / actor_query / actor_gradient、拒标原因机器键及最小 `[move, ability]` 动作集合。禁止非法长度、动作、重复、未知字段、不一致接受/梯度标志。缺字段与非记录 tick 均保持未知，不延续上一记录。存在该字段时纳入 replay identity hash；旧录像身份兼容。A 来源未区分与 `A_ENEMY_DROP_RECOVERY` 明确区分，不凭泛 A 标签认定真实敌源掉包。
- `npm test` 全量生产回归通过；`npm run test:review` 包含真实来源快照/hash、转换数值、缺字段 null、own/enemy 分离、不推断、稀疏 schema RED/GREEN 与 DAgger identity 篡改拒绝，另用只读原始目录复核来源字节 hash。原有 156 项协议及 19 项 reward 映射回归通过。
- 无网络 mock 的 Chromium 桌面/手机通过真实质量数值、未记录状态、默认 60 fps / 4x 插值、断点吸附、整数 tick 暂停/seek/单步、蓝色像素与双方文字、授权、标注和深链回归。稀疏轨点击使用隔离的合成协议测试记录，真实录像未注入或持久化测试数据；点击生产轨道按钮确实跳到 tick 321。手机表格横向滚动保留类别列，页面无横向溢出；诊断页 pageerror / console error / HTTP error 为零。
- 新证据保存在忽略目录 `runs/replay_review_20261017/browser-dagger-local/`、`browser-dagger-public/`、`release-dagger.json` 与 `release_dagger_acceptance_20261017.md`。发布必须等待同一最终 SHA 的 Actions/Pages，再验证页面、录像/catalog、DAgger catalog/汇总/来源文件 HTTP 200 与本地字节/hash/尺寸一致，不覆盖前述审计证据。

## 真实 Critic A1 与多场景追加验收

从干净且同步远端的 `6d1e7a7184aed66f8e00e9279b49f899f9ef50bc` 继续，保留前述 H7、H8、视觉、DAgger 和授权审计。全程 CPU，读取已完成 H8 v34 formal g32 的不可变文件与封存代码；没有训练新模型、使用其他代 Critic、读取可变 checkpoint 或操作训练进程。

### Critic 与优势真实性

- 校验 COMMITTED、policy.pt、ema_policy.pt、state.pkl 与 plan hash。选择 raw policy.pt，确认它与 state.live.params.actor replica 0 逐元素一致；EMA 同架构、不同权重，未参与本批次。Critic 从同一 state.live.params.critic 提取，75 个参数叶、8 副本完全一致；没有从 Actor 的未使用 value head 取值。
- 原始 checkpoint policy SHA 为 `b491758761b32872555555f08cf4e42f9cab120215501509a2bdbf905e64ce11`；state SHA 为 `f2fcf9c44401b924776e7e7e990bceebe1660fb1397de30d1476416766716ec7`；Critic 规范参数 SHA 为 `7002b1b4f40d87021462d69a4bd3594964698ce2608592c5c4e17db6d1d5cf85`。公开安全提取证据在 `web/review-evidence/h8-g32-critic.json`，保留 EMA、网络和 plan hash。
- 运行真实训练 Transformer BF16 forward 与 128 bins 的 HL-Gauss value decode（范围 -20 到 20）。输入为生产 encodeObsJAX 的 24×13×15 与 bun24-trap-timer-v2 state24，使用原有固定通道缩放，没有另加 running normalization。3 个固定生产观测在原 checkpoint 与提取模型上的 value 完全相同：5.153899669647217、5.152127742767334、5.185464382171631，最大误差 0。
- 定义严格为 `Q1=r+0.995*V(next)`、`A1=Q1(taken)-sum(pi_legal*Q1)`，terminal next value=0。gamma=0.995 从真实 plan 及其 receipt hash 重读。动作口径为合法 move×ability 联合动作，ability=2 已包含模型道具决策；每个合法联合分支固定同一对手动作、完整生产 Sim 状态、RNG draw 位置与有状态 reward ledger。取实际生产 ONNX FP32 actor 的 masked move/ability softmax 乘积，再对合法联合动作重归一；执行动作为同 checkpoint 的 greedy policy。该比较分布是生产导出策略分布，不声称 ONNX FP32 logits 与训练 BF16 logits 逐位相同。
- tick t 的数值对应 s[t-1]→s[t] 的动作与 Reward。ΔA 使用相邻有效决策之差；首项与缺口后为 null。开局 tick 0 不存在已执行决策，显示未记录；对手没有本 checkpoint 的优势，绝不套用模型数值。新批次 5,806 个决策均测得 A1，缺测 0。
- 所有 tick 的 taken 反事实分支完整 snapshot/info 与原录像一致，全部非零 Reward 分量一致。全量第二次计算的量化列和反事实摘要 hash 完全一致，双方原 reward ledger 又分别整局重算通过。从原始 state.pkl（不用提取 pkl）重载 Critic，以 seed xor 0xA100 的固定 PCG64 抽取每局 8 个随机 tick，另含开局、终局及所需前驱，共复算 112 个 tick 的全部合法分支；Q、均值、A 与 ΔA 最大误差为 4.997314446697487e-07，小于量化半单位 0.0000005。证据为 `reports/replay_a1_verification_20261017.json` 与公开副本。

### 冻结矩阵与文件大小

原生关卡资源审计只确认一张完整可复现的 Bun 底图 806；其他项目的 242 个真实关卡涉及不同玩法，不把它们套成未经验证的 Bun 地图。本批次采用三种真实规则/出生场景，共享底图，页面明确显示规则、qqt_id、map/scene hash 与区分标识：原生抢包；生产 eval 已支持的开放清砖；封存 je05_web_tasks.setup(carry_home) 的清砖、1HP、持敌包返程任务。最后一项初态与原始封存函数的生产 snapshot 完全相同。场景协议及来源代码 hash 单独保留，不伪称三张不同地图。

`reports/replay_a1_matrix_20261017.json` 在生成前冻结 seeds 2026101900–2026101905、场景和真实 Hunter normal 对手。全部预声明局保留，无按结果手选、无修改动作。观察到原生局窗口未切换，开放局两局对手胜，返程两局交付获胜；没有给未观察到的失败类型造标签。原有历史录像继续提供其他已观测失败证据。新批次为默认目录，历史 12 局按需显示，仍按单局懒加载。

| 场景 / seed | ticks | 终局 | raw bytes | gzip bytes |
| --- | ---: | --- | ---: | ---: |
| 原生抢包对战 / 2026101900 | 2400 | 平局 | 124965 | 44278 |
| 原生抢包对战 / 2026101901 | 2400 | 平局 | 131448 | 45577 |
| 开放抢包对战 / 2026101902 | 291 | 对手胜 | 22682 | 8918 |
| 开放抢包对战 / 2026101903 | 512 | 对手胜 | 33414 | 13037 |
| 持包返程任务 / 2026101904 | 100 | 模型胜 | 12688 | 5225 |
| 持包返程任务 / 2026101905 | 103 | 模型胜 | 12639 | 5261 |
| 合计 | 5806 | | 337836 | 122296 |

大小按实际 UTF-8 与 gzip level 9 测量。公开协议只增加量化列式 Q、策略均值、A、ΔA、合法动作数（scale=1e-6）及 provenance；没有逐 tick 完整 state、候选 Q 表或 logits。私有原 trace、Critic 参数和反事实分支只保存在忽略目录。六局再次从真实 ONNX 策略与 Hunter 重新生成，双方动作、状态 hash 和终态完全一致，Reward 和与优势复算通过。

### 页面与本地验收

- 画面后第一块为精确时间、定位、播放/暂停、单步、速度和主进度条，紧接完整七轨时间轴；DAgger 卡片移到时间轴之后。桌面和手机 DOM 与实际坐标均验证无间隔卡片，画布底部与播放区顶部相等。行名常驻，优势密集点在缩放后保持轨道紧凑，数值标签及 tooltip 可点击定位。
- 模型头顶默认显示三行 A、带正负号的“优势增量 ΔA”、模型 P0；对手保留蓝色与“对手 P1”。文字跟随插值浮点位置，边界夹紧并避免双方标签相叠；手机按 Canvas 实际 CSS 尺寸放大，ΔA 实际字号至少 10px。可关闭优势叠加。暂停、seek、单步与深链按整数 tick 同步头顶、右栏与时间轴数值。
- `npm test` 全量生产回归通过；追加 renderer 改动后 visual parity、death overlay、airdrop/death 与 `npm run test:review` 通过。后者含 18 局当前/历史共 462 项协议检查，19 分量中文映射、连续/不连续状态、手算联合优势/非法动作排除/概率归一/terminal/首项与缺口 delta、合法数/来源证据 schema RED/GREEN、三个场景 production Sim/hash、来源函数初态、设备授权与 DAgger 来源真实性回归。
- 无网络 mock 的 Chromium 桌面 1600×1100、手机 390×844：三个场景逐一播放，固定 tick 50 的头顶、右栏与时间轴 Q/A/ΔA 和文件一致，真实 ΔA 随动作变化，优势轨点击定位、开关、深链刷新、授权免输通过。每次 1.4 秒播放约 87 帧；确定性生产 renderer 的优势标签 6 个中间屏幕坐标严格单调，权威状态不改。手机无横向溢出，诊断页 console/pageerror/HTTP 错误为 0。
- 完整浏览器回归另覆盖首次/错误密码无授权/正确/刷新/新 tab/撤销/版本与校验规则变化，蓝色实际像素与中文身份，Reward/事件证据去重，60fps、4x、断点吸附、整数 tick 暂停/seek/单步，标注持久化/编辑/删除/导入导出/错误 hash 拒绝，复制定位及深链，DAgger 未记录与真实历史汇总。证据位于忽略目录 `runs/replay_review_20261017/browser-a1-local/`、`browser-a1-regression-local/`、`motion-a1-local/`；截图包含 desktop/mobile 各场景及画布局部。公开裁切截图为 `web/review-evidence/screenshots/desktop.png` 与 `mobile.png`，保留真实页面像素，不加入标注或模拟内容。

发布前扫描 git ls-files 与全部待提交文件，公开 evidence 不包含机器身份、凭据、绝对路径、内网地址或 PID。提交后按同一 head SHA 等待 Actions/Pages 成功，再核对公开 catalog、录像、DAgger、Critic/A1 evidence HTTP200、字节/hash/尺寸及线上真实浏览器。发布验收记录保存在忽略目录 `release-a1.json`、`browser-a1-public/`、`browser-a1-regression-public/`、`motion-a1-public/` 与 `release_a1_acceptance_20261017.md`，不为记录自身 SHA 追加部署。

## 路线势能协议、网页与导出接口（等待真实实验 checkpoint）

本轮从干净且与远端 main 同步的 `167be513889569bd4835184e06cf66df1aa995fa` 继续，保留前述全部审计。只读训练侧设计报告 `route_potential_advantage_analysis.md`、路线训练合同及训练总账；没有写训练目录、读可变 checkpoint、操作训练进程或使用 GPU。训练报告关于“未发现网页 A1”的旧描述已过时；本网页已发布的同代 Critic 严格 A1 仍按上节证据保留。

### 已完成工程

- 可选扩展 `qqt.route-potential/v1`：稀疏 tick、目标字典与 episode域 SHA-256 脱敏身份、真实来源/阶段、Phi前后、`gamma*Phi_after-Phi_before`、`lambda*delta`、网格加权路长（含显式 `INF`）、拓扑区域/增益/前后地图及安全证据hash、阶段切换、失效、terminal边界修正及安全拒绝原因。严格校验列长、时序、null、有限数、公式、相邻Phi连续性、终态归零、来源绑定、开墙证据和缺测计数。目标A/B不接受错误 provenance，own-recovery独立。route新增列纳入identity；未加字段的18局文件及其identity保持不变。
- 塑形 Critic 的 Q/策略均值/A1/ΔA/合法数独立列和来源；没有经过塑形训练的Critic时全部null，不把人工路线reward加到原A/Q上。原始 `advantages`/`rewards` 不改名、不覆盖。塑形A1仍为全部合法move×ability的策略概率重归一一步反事实比较，首项和缺口后ΔA为null。
- 画布后仍直接是播放控制+进度条，再紧接完整时间轴。新增中文“路线势能变化”“路线奖励”“塑形 Critic 优势”“塑形 Critic 增量”四轨，行名/解释常驻、正负数值和阶段直接可读、tooltip关联同tick原Reward证据且不重复求和、点击定位。稀疏点排成互不遮挡的行，密集点按实际标签宽度抽样显示文字。右栏原始/塑形Critic并排；路线详情展示目标/来源/脱敏ID/路长/Phi/ΔPhi/lambda/路线reward/拓扑/阶段边界/拒绝原因，明确人工辅助值与模型价值估计的区别。
- 模型头顶默认保留原始A/ΔA并加紧凑路线行；值只取当前整数tick、不延用缺口。所有文字跟随生产renderer的浮点视觉坐标，按实际文字宽度夹紧边界；对手仍蓝色+文字身份且不套模型优势。可分别开关路线/优势叠加；只有存在经过验证的塑形Critic证据时才启用塑形口径切换。旧录像新增字段明确未记录。
- `scripts/export_diagnostic_route.js` 和 `docs/replay_route_export_v1.md` 提供版本化训练侧交接：稀疏事件、独立采集/重算artifact、可选原始/塑形Critic反事实输入，转成量化列式数据、来源hash和raw/gzip报告。原始结果已有时只校验不覆盖；尚未记录时可按原协议导出。白名单字段、数学与量化一致性检查、固定同episode/动作/版本绑定，拒绝全量state/Q表输出，CLI不覆盖文件、不修改catalog、禁止fixture写进web。正式CLI额外校验生产Sim整局statehash/终态，并拒绝同一文件充当采集与独立复算两份输入。
- 此适配器不自行证明上游训练/RNG/路由/反事实真实性；它校验输入证据与复算一致性。真实发布仍须审计独立CPU推理、全部合法分支、RNG/ledger克隆、训练Critic parity和不可变receipt。两份相同对象/fixture不构成真实训练效果证据。
- 旧六局 `scene_protocol_sha256` 仍保留原始值。协议文件新增诊断扩展后，场景回归显式接受已审计旧版本，另锁定 `setup.toString()` 的原始hash并完整重放终态，未改旧录像的来源hash来假装重新生成。

### 测试与本地浏览器证据

先创建fixture及测试并确认缺模块RED，再实现转GREEN。`npm test`全量生产回归、`npm run test:review`、renderer/death/airdrop针对回归通过。新增测试覆盖旧identity、null/INF/负值、阶段切换、terminal、Phi连续性、同tick原Reward关联、来源脱敏、公式/量化/重算篡改拒绝、原始与塑形Critic独立、手算加权平均/非法项排除/terminal、首项/缺口ΔA、确定性重生成、CLI拒绝覆盖和fixture发布。

真实Chromium桌面1600×1100及手机390×844：仅在隔离浏览器网络响应中使用带“合成测试fixture”标题的测试文档，磁盘/公开catalog及录像均未注入fixture。五个固定点检查正/负/缺测/阶段/INF/终态，原始与塑形面板/切换、轨道实际点击、同tickReward tooltip、暂停/seek/单步/深链、授权免输、只请求选中单局、画布后DOM顺序和零横向溢出均通过。生产renderer六个严格单调中间屏幕坐标证明路线文字跟随浮点移动，而数值固定在同一已测整数tick。两端console/pageerror/HTTP错误均为0。开发中发现并修复了路线点相互拦截点击、右栏长机器字段溢出，未用强制点击掩盖问题。

另对未改动的真实六局跑完整浏览器回归：三种真实场景均可播放，tick50的原始Q/A/ΔA、头顶、时间轴与录像一致；1.4秒各87–88帧，浮点标签单调，蓝色对手/中文Reward、DAgger来源与缺测、授权首次/错误/刷新/新tab/撤销/版本失效、标注持久化/编辑/删除/导入导出、复制定位与深链全部通过。证据在忽略目录 `runs/replay_review_20261017/browser-route-local/`、`browser-route-regression/`、`browser-route-a1/`，截图明确为UI fixture或原始历史实录，不发布fixture截图为训练证据。

合成五点路线列实测1522 raw / 556 gzip(level9) bytes；带原六局之一的完整测试文档为133936 / 46551 bytes。此数字仅说明接口紧凑性，不能冒充正式新录像大小。公开默认6局及历史12局的文件/catalog字节保持原样。最终真实新批次须重新逐局测量。

### 当前发布阻塞和所需artifact

训练侧总账的实验 `JE09-STAGE-H8-ROUTE-PBRS-20261017-v35` 尚为“预注册/实施中”；本轮收尾只读检查时，声明的路线实验namespace尚不存在，没有可用不可变正式路线checkpoint。旧H8 v34 g32没有本次路线塑形，不能改标签充当新实验录像。因此本轮提交协议/UI/接口/测试，不将fixture加入公开catalog，不提前推送main/Pages，不声称新的线上SHA或真实路线录像已验收。

继续真实导出需要：(1)已COMMITTED的路线实验Actor/同代塑形Critic、state及hash/receipt、训练gamma与最终lambda、原始任务Critic的独立来源；(2)路线ledger/状态机/加权路由/安全核验源码版本和训练reward→GAE→PPO证据；(3)CPU Python与导出模型parity、生产JSSim/真实对手/ledger/RNG克隆一致性；(4)预声明seed×场景×对手矩阵，至少路线专项及完整局；(5)本接口的真实稀疏事件与独立重算结果、拓扑/目标来源证据及缺测说明。满足后再生成、核验、隐私扫描、安全集成main并等待同head Actions/Pages与线上HTTP/hash/桌面手机验收。此等待不要求网页任务启动或控制训练。

本轮收尾隐私检查覆盖所有tracked及待提交文件：本机标识/绝对路径/内网IP/PID扫描0项，历史验证口令在当前文件中明文匹配0项；既有日期报告保留。只读公开 `build-info.json` 仍为 `167be513889569bd4835184e06cf66df1aa995fa`，没有发布本轮fixture或网页变更。路线数值显示保留最多六位小数，避免把非零微小路线reward显示成0。
