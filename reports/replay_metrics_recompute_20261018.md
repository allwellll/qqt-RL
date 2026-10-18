# 回放训练指标复算与发布修复（20261018）

## 生成前冻结

基线 `8b705eb16e441c8700235aedc97a2e80ce8159c2`。综合发布最佳仍以已发布选择报告及 catalog 的37个 COMPLETE 固定协议候选为准：JE10 v38 r01 attempt04 g8 raw，policy `d392bf8caeb0ce149a91ce5af1c69806c556cadd69d89bef3c00758d2d3a89fc`，state `fbdc05d14ab682553a4040198ed05d3993531dabfc8d862e802a157f1ba5f0fc`。仅只读固定 gen000008；不读 LATEST 或正在训练的 r12，不启动训练或 GPU。

优先复算已公开的固定 seed2026101800 四条 JE10 原轨迹（real_drop / native_death_stock / native_sugar_stock / natural_full2400），动作、初态、随机流与终态不变。新增完整自然对局预声明 seed **2026101810、2026101811**，生产原生806场景、2400 tick上限、相同已发布JE10 Actor与生产真实对手。两个seed全部保留，不按结果挑选。四条原轨迹若封存源/初态ledger无法可靠复算，不作为合格入口；新增两局作为当前发布最佳替代旧缺测项。禁止用fixture冒充自然成功。

采用同代真实 total-reward Critic，在生产Sim同一状态、同一RNG、固定实际对手动作枚举所有合法move×ability（道具包含ability=2）。Q=r+gamma V(next)，终局V=0；masked双头概率乘积在合法联合空间重归一。原始继承任务reward与JE10 total reward分别进入Q；**raw侧沿用同一total-reward Critic，是 offline_approximate_one_step，不声称独立原始任务Critic或历史训练优势**。ΔA=shaped A1−raw A1，另存三项相邻tick增量。gamma从封存plan读取；每个候选用克隆的Reward ledger；缺测不能当0。发布须先验证封存Reward依赖与Web事件转换的适用边界。

## 原始指定录像证据

原文件保持逐字节不变：

- `h8-v36-g32-carry_home-2026101785.json` SHA256 `22f0359884838fff2a5d57c8fcbd12952b2547d44d0042e80f6f5499e1309114`。
- `h8-v36-g32-open-2026101782.json` SHA256 `04a572ad739b70acff36cba744817bc185b854f1617f571b30b439e2a17ec5cb`。

carry tick107，模型携敌源包，从(y=4.10,x=5.42)向上到(y=3.92,x=5.42)，交付终局。己笼为左上(1,4)的3×3区域；动作前尚在笼外。原离线源把己方半场HOME误作D_DELIVERY，同时continuous_distance只对phase3选home通道、phase4误选无drop目标通道，距离不可达使base=-1，初始offset=+1，Phi整个107tick恒0；原JSON并非UI舍入0。原距离1→0、Phi0→0、路线reward0；20项总Reward也为0，其交付奖励需危险结束后结算的问题见下文。原始/塑形A1全部null。

open tick436，模型空手、对手携模型源包，双方存活未受控。模型(y=6.5,x=6.2499)向上到(6.11,6.2499)。原route phase=NONE、无target、Phi0→0、路线reward0，却导出distance4→4：训练函数NONE分支默认返回home距离，导出器/页面未屏蔽。不存在激活的己笼返程目标；真实PBRS在此刻未满足敌死亡/受控库存窗口，也无归因敌源散包，不能把辅助敌库存目标冒充已激活ledger目标。模型总reward0；优势null。对手C_RETURN的delta=0.0016710758209228516、路线reward=0.00008355379395652562与模型侧严格分开。

旧v36来源是冷启动aux的 offline_recomputed_current_reward，非封存训练rollout。新JE10计算不能回填为这些旧tick的历史训练真值。

## RED

`web/test_review_admission.js` 在现行32条入口缺少逐决策完整Reward/Critic证据时失败。`web/test_review_route_targets.js` 在错误接受笼外D_DELIVERY时失败，且约束NONE不得泄漏home距离。RED日志保留本地runs，fixture不进入公开catalog。

carry tick107 的环境事件与奖励还有第二条独立原因：终态仍有一颗fuse=23的活泡，transport ledger把真实交付24分放入pending_capture，需hazards全部结束且存活才settle；Web已立即终局，录像不再推进结算，因此当tick实际已付分量为0。这里既不是“未交付”，也不是UI丢分；不能把未结算的24手工写作已付。新详情需区分事件、已付Reward和终态未结算负债。open tick436仍有fuse8/28；与目标不存在是独立事实。

同代Critic已CPU提取，参数副本/Actor一致，固定输入提取前后误差0。critic pickle SHA256 `0d2ad5345e4a38f63dbba5379c1a582dd47a7b34985d1073aede9b8ae878c278`；gamma=.995。原带预置trap的fixture缺初始化因果ledger，不能按“原训练真值”补齐；本轮优先复算自然原轨迹seed2026101800，另两局固定seed原生完整局替代。两局全部生成后保留：2026101810在1493tick真实交付胜，2026101811在2400tick平局/未交付。成功与失败均保留，不择优。

## 多场景扩展（额外生成前冻结）

为保留多场景支持，新增固定矩阵：开放真实806布局 seed2026101812/2026101813（生产规则、清砖变体、2400tick）；持包返程 seed2026101814/2026101815（现有生产carry_home重置、1HP、预置持敌包、清砖、300tick）。全部保留，不按结果筛选。与两局原生自然新对局及一条原自然轨迹复算合为7局。场景差异是已审计生产初始化规则，均属806资源，不能声称跨地图泛化。返程预置不是自然取包成功。此扩展在这四局任何运行或读取结果之前声明。

## 最终准入与全量复算结果

32条旧缺测诊断入口退出可选catalog，历史资产字节及兼容测试保留；两个指定旧深链明确不可用，不暗中切换录像。新catalog有7局：原轨迹复算1局、同checkpoint新seed类别0局、当前综合发布最佳替代重跑6局。六个新seed均预声明、全部保留，包含未交付/窗口关闭、敌掉包未拾、对手获胜及模型交付；不按胜负择优。原生/开放/持包返程是同一真实806资源的三种生产初始化规则，不声称三张地图。预置持包成功不等于自然取包成功。

| seed | 场景 | tick | raw bytes | gzip bytes | 终局 |
|---|---|---:|---:|---:|---|
|2026101800|原生原轨迹复算|1773|441057|94178|模型胜|
|2026101810|原生新局|1493|349359|68662|模型胜|
|2026101811|原生新局|2400|604400|122718|平局/未交付|
|2026101812|开放布局|2400|710845|112100|对手胜/敌掉包未拾|
|2026101813|开放布局|560|137703|27645|对手交付胜|
|2026101814|预置持包返程|85|39890|10168|模型交付（非自然取包）|
|2026101815|预置持包返程|91|40761|10665|模型交付（非自然取包）|

全8802个决策tick、86362个合法联合动作分支，使用封存Actor/Reward和同代真实Critic、相同状态/RNG及固定实际对手动作。重新CPU提取Critic后再全量复算，两次未量化结果JSON完全一致（包括Reward、Q、A、增量及反事实digest）。六个新seed独立生成，动作、事件、检查点和终态完全一致。量化步长1e-6、单值误差至多0.5e-6。逐tick不公开完整state、全部Q或logits；catalog与每局懒加载。完整hash与尺寸收据：`web/review-evidence/metrics-20261018/verification.json`。

定义为明确标注的 `offline_approximate_one_step`：raw Q使用继承H8账本reward，shaped Q使用JE10 total reward，均加gamma=.995乘同checkpoint total-reward Critic的next value；terminal next value=0。raw并非独立原始任务Critic，继承账本本身也含既有运输塑形。ΔA为同Critic两种奖励口径的A1之差，各自相邻tick增量另列。Reward在Web真实事件上调用同版本封存函数，但课程场景时限和训练终局结算存在差异，非历史训练rollout真值，不声称严格跨Critic比较。当前新数据不回填两个旧v36 tick。

## 验证与视觉证据

RED先于修复：缺完整指标目录失败、NONE距离泄漏/笼外D阶段失败；GREEN覆盖准入非空、全部seed、真实Critic/checkpoint/gamma一致、手算公式/合法概率重归一/terminal/首项null/finite/Reward与route副本对齐、指标绑定replay identity、旧32资产兼容。`npm test`完整JS/Python测试通过。两个原文件hash仍精确等于任务书值。

浏览器采样实际送入生产renderer的双方浮点坐标；固定0.15格阈值会把50ms内合法连续0.195格误判，现改以权威路径分段长度及实际tick+alpha预算核验，而非放宽无条件阈值。合法死亡/复活、碰撞校正/侧移、保护状态、持包/道具、地图拓扑等断点继续snap；暂停/seek/单步整数tick。截图与净化收据保存在 `web/review-evidence/metrics-20261018/`，线上验收运行同一脚本并核对同head Pages及所有公开字节。

最终本地桌面/手机真实浏览器：7局、3种场景全部加载；每局5个tick数值与量化源相同；双方文字/蓝队、时间轴、进度DOM顺序、深链、标注持久化/导入导出、授权错误不记忆/正确后刷新新tab免输/忘记/版本失效均通过。平滑开关各采样4.5秒，双方单调中间坐标通过；实测FPS随浏览器调度变化，页面同时显示10 tick/s及alpha。大位移保留明确断点分类，无未解释跳变；console及HTTP错误0。切换奖励玩家时，模型指标卡仍绑定P0，事件tooltip不把P0优势套给P1。
