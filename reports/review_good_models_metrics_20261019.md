# 多检查点完整训练指标回放（20261019）

## 生成前声明

基线 e5377f779ac2229a353994782732415f3153ca5f，Review工作树干净且与origin/main一致。只读已完成不可变checkpoint和COMPLETE固定评估，不读r18状态/可变checkpoint，不修改训练目录、不启动训练、不用GPU。

核验候选：r01 attempt04 g8（既有综合发布基线）、r01 g128（完整运包）、r13 causal-clean g128（干净因果链完整运包参考）、r16 g128（战斗专项，存在自伤）、r17 g128（最新已完成/库存恢复参考，不宣称全面平衡最优）。五者均有同SHA的固定评估协议；需要逐项校验COMPLETE、shards、Actor、state、Critic、plan、SOURCE_SEAL后才准入，不能用r01 Critic替代。

在任何新局生成前冻结相同矩阵，每模型3局、共15局：

| seed | 场景 | 时限 | 来源/限制 |
|---|---|---:|---|
|2026101900|原生自然完整局|2400 tick|生产806原始布局，双方空手/1HP/初始无无敌、原生糖泡与道具、真实Hunter normal、assist-off|
|2026101901|原生自然完整局|2400 tick|同规则不同seed，完整终局，胜败全部保留|
|2026101902|库存窗口→受威胁恢复|1200 tick|复用固定评估recovery_threat生产reset，模型空手，敌方初始8tick糖泡窗口；之后真实取库存、真实Hunter恢复追逐，不预置模型携包|

库存窗口初始控制是明确的诊断reset，无模型控制归因、无历史奖励余额，冷启动aux从零，不把初始糖泡虚记为模型自然击杀。使用从零声明的未归因初始控制，所有后续取包/死亡/掉包/交付由生产Sim真实事件产生。不保证各模型一定取包或完成返程，失败也保留；不为“好看”改seed、动作或初始化。本矩阵不等于自然因果击杀成功评估，不以单局替代固定16/64局指标。

全部局使用各自Actor、同代total-reward Critic及各自SOURCE_SEAL Reward，以同状态/RNG、冻结真实对手动作枚举合法move×ability联合动作。raw/shaped A1明确是同Critic两种Reward口径的offline_approximate_one_step，非独立原始任务Critic/历史训练优势；terminal V=0；ΔA=shaped−raw，三种相邻tick增量首项null。量化与合法动作概率/掩码列式存储，所有tick finite，不发布全部候选Q/完整state。

既有7局合格资产与入口保留，默认仍r01 g8，不因本次某条胜局改综合最佳。r18未完成且由其他owner管理，明确不入选，也不读取。

## 固定同协议选择审计

共同PROTOCOL SHA256：`f2fcf50022fbb61e9630855ca3d08ccef4b1d390d89a783a5bdf998b9205a936`。每点COMPLETE所有shard、checkpoint/state、plan及SOURCE_SEAL逐hash通过。不是按新录像输赢选模型。r18未读；r11历史无真实custody特权导致的无效能力结论排除。

|点/标签|Tactical安全击杀/自伤（64局）|Hunter难/中/易安全击杀（各64）|自然拾包→交付（16）|真实掉包交付|死亡/糖泡库存交付|人工近/中/远交付|受威胁恢复交付|
|---|---|---|---|---|---|---|---|
|r01-g8 综合发布基线|12/0|0/2/22|9→6|8|2/3|16/16/14|3|
|r01-g128 完整运包|5/0|1/5/15|12→8|11|3/4|16/15/11|4|
|r13-g128 干净因果链·完整运包参考|5/0|0/2/7|10→5|4|1/0|16/16/10|1|
|r16-g128 战斗专项·注意自伤|15/2|3/1/2|4→2|5|1/1|16/16/11|2|
|r17-g128 最新完成·库存恢复参考|9/2|0/2/7|6→3|6|3/3|16/15/5|4|

Tactical为Python原生协议，Hunter/路线为JS，分列不能混为相同模拟器总体胜率。库存/人工返程/受威胁reset都是诊断题，不代表自然因果击杀与完整运包。r16/r17各有Tactical自伤2/64；最新完成参考不等于最新综合最佳，默认r01 g8保持。

## Critic提取

- r01-g8：Actor `d392bf8caeb0ce149a91ce5af1c69806c556cadd69d89bef3c00758d2d3a89fc`；Critic `0d2ad5345e4a38f63dbba5379c1a582dd47a7b34985d1073aede9b8ae878c278`；gamma=0.995，提取前后固定输入最大误差0.0，8参数副本exact。
- r01-g128：Actor `7f8d835bdb70075ec29121d3ffcc6c12b9f0c183dcb3770142c1fe58d6efd744`；Critic `c2285c025dbe99fec035fcd83a92be38fc4e26770d1f4ef552a79b38498394d5`；gamma=0.995，提取前后固定输入最大误差0.0，8参数副本exact。
- r13-g128：Actor `e25dc7c7c010548c1238e0a949ebeecba526740e8178e15bdf40df9b4874566a`；Critic `8c1ac2231f0474d3ec06615cb3957a0570092800519f89fbee0b3d95e1742e58`；gamma=0.995，提取前后固定输入最大误差0.0，8参数副本exact。
- r16-g128：Actor `317de3e25dc13928701b7e59bbbc66df6dbab22f8b1dd70a799475c89aafee28`；Critic `e41508221fff3b3dd7b79cd59b56bd7b8d2e7996d84efcadca003b0b793e2f35`；gamma=0.995，提取前后固定输入最大误差0.0，8参数副本exact。
- r17-g128：Actor `ed06ecbef3b0814526828b5a973b346fd064faa60a494ef711eb8723e0bdc3f3`；Critic `9d8e873e13b265dcf5bc128b587e30f488aabb4581fe3f027e2c69fbe7a4ec0f`；gamma=0.995，提取前后固定输入最大误差0.0，8参数副本exact。

## 数值依赖边界

逐模型绑定全部实际导入的训练数值模块：r01两点为45个（38个来自各自SOURCE_SEAL），r13为54个（增加因果初始化、库存开路、连续路线等真实依赖）；其余通用模块绑定不可变生产基线e5377f7。唯一差异是通用`jax_bomb/bun_env.py`中不执行的`_fresh`重置：顶层常量及其他全部定义AST相同，安全模块不调用该重置或init_batch。转换模板来自封存je07环境，状态演化仅来自生产Web Sim。依赖凭证逐模型保存，不能将这一离线Web口径宣称为原生训练rollout完全等价。

RED阶段：多模型准入测试在catalog尚无`model_comparison`时按预期失败；Actor概率fixture覆盖概率归一、非法动作零概率、掩码、长度、版本和量化，禁止完整logits/Q表混入紧凑协议。既有A1手算、terminal和三种首项增量null测试保留。

Web Sim、Hunter、固定场景reset及Web评估适配器不在训练run的Python SOURCE_SEAL内；本次对照分别绑定不可变生产Git基线e5377f7，并核验当前文件逐字节一致，写入每模型依赖凭证。不要将两种来源混称为同一训练run封存。五个Actor导出各75个float32张量亦已与各自policy.pt逐字节相同。

因果清理模型封存账本的phase 5/6分别是取库存前开通己笼/敌笼走廊，目标为frontier准备点，非携包返程。新增RED/GREEN覆盖与中文释义；仅转换语义，原动作、Phi、Reward、Q/A不变。

浏览器RED/GREEN补充：真实r13恢复局tick176的小Reward精确小数字符串与累计值并排导致桌面右栏超界52px。保持原数值/精度，给数值flex行增加自然换行；同一深链从scrollWidth1652修复到1600，超界元素0；随后重跑全部桌面/手机验收。

## 本地全量结果

新增15局、25,034个决策tick、239,725个合法联合动作反事实；各点第二次独立Critic提取/全量复算及真实Actor/Hunter重生成完全一致。既有7局原字节/hash不变，总catalog22局。首项三种时间增量null，其余全部finite。

全部模型共用seed 2026101900、2026101901、2026101902，前两局自然完整，后一局明确未归因初始库存窗口诊断。所有预声明局全部保留。

|模型|新局模型胜/负/平|自然局交付成功/2|恢复诊断交付成功/1|敌源/己包拾取事件|
|---|---|---|---|---|
|r01-g8|1/2/0|0|0|1/0|
|r01-g128|2/1/0|1|0|2/0|
|r13-g128|3/0/0|1|0|3/0|
|r16-g128|1/0/2|0|0|1/0|
|r17-g128|3/0/0|1|0|3/1|

终局胜负与运包交付分开；尤其score为0的超时判胜不能算自然交付。统计是本批固定小样本，不替代固定同协议16/64局评估，也不按此批输赢重选默认模型。

|模型/任务/seed|tick|raw bytes|gzip bytes|Replay SHA-256|
|---|---:|---:|---:|---|
|r01-g8 / natural_full2400 / 2026101900|1202|371145|82751|`f5ddf493551c06d74d530b0d4c0f12c29fa28f513df9b2300225ab03a784ded2`|
|r01-g8 / natural_full2400 / 2026101901|2400|709184|152432|`e0d55567a6497c2a9cec59d547a0475bf4e40946f8ffd91b6d54868994b819d6`|
|r01-g8 / recovery_threat / 2026101902|1200|408984|78325|`1b17f12cb73279c2c847240ccdeb06435deca5c7648eef75028159079b326688`|
|r01-g128 / natural_full2400 / 2026101900|1446|408624|92491|`90c3d998731dc4416140b98036ffe30f2436ead089745d63b78e1a9b5cb38ce7`|
|r01-g128 / natural_full2400 / 2026101901|1746|542129|127306|`b29f399f5502f13d7e29eedb8715fbaa89461f1ba47ae214d09bc58f240e1227`|
|r01-g128 / recovery_threat / 2026101902|1200|406891|76731|`a9c233c44968456a017e97d11d0f85947201edb4853c8c76200cd37e662c0799`|
|r13-g128 / natural_full2400 / 2026101900|1394|408825|87642|`ab579dc28d42c05db195a3973a72e12e78084fe84babf766c856cf11814be845`|
|r13-g128 / natural_full2400 / 2026101901|2400|722211|156608|`e56d6bd6659892e92d6cb27b92710a94445e6252691251c1823cd55c3a7e9153`|
|r13-g128 / recovery_threat / 2026101902|1200|396060|75537|`7d40c9ab02855ff17ccf8bb37f5fdaad9f22e8b904c8232564cae93232ae49f8`|
|r16-g128 / natural_full2400 / 2026101900|2400|588887|141032|`f811211c99ec0fb181458612ced7139bd4dd24c76106363eecb4e8d1011c70a3`|
|r16-g128 / natural_full2400 / 2026101901|2400|718464|158652|`4e5c3e8645723e4e4319ceb6a13eb99162442bafd7bae90799eb8c93c3b05c27`|
|r16-g128 / recovery_threat / 2026101902|1200|390498|68631|`dd4cdbebe606842fd47f75fe4760e61a3004f66c850fdeb1f0c35cee5de75e5c`|
|r17-g128 / natural_full2400 / 2026101900|1246|340400|76795|`3adf5d1cdaa739e704414cd2b55b4cf66653dcea9a2e738581a2454a05ab1d17`|
|r17-g128 / natural_full2400 / 2026101901|2400|779955|158565|`311d30d74d37584c7ebe4fdb2c23b54e5791e538d1f03a2597c90b163d8fd64f`|
|r17-g128 / recovery_threat / 2026101902|1200|388572|67979|`3fd263e6655862bf4e7429cbd29c4795b5384fe6b0732b1dfc21b7c14806d03a`|

## 验证与证据

`npm test`与`npm run test:review`全部通过；A1手算、非法动作排除/重归一、terminal、首项null、三种增量、量化概率/掩码、跨代Actor/Critic/Reward配套拒绝、phase5/6开路目标、22局catalog及原7局完整性通过。第二次全量复算比较未量化数值JSON完全一致。

真实桌面1600×1100和手机390×844浏览器逐局懒加载/深链/seek/整数tick/Reward与全部6种优势指标可见、真实策略概率、五模型切换、布局无溢出、授权首次/错误/记忆/清除/版本失效、标注持久化/导入导出均通过；console与HTTP错误均0。生产renderer双方角色安全移动中间坐标严格单调，平滑开/关与合法snap、FPS/tick/alpha通过。详情见`web/review-evidence/good-models-20261019/browser-checks.json`，截图同目录。

- `models.json`：SHA-256 `26a4bead569944cd2aac3ce48ba2bac9e1070d901775bb1eb9177423d07eaa8f`，69138 bytes。
- `verification.json`：SHA-256 `c4d4e6be9d063007558c35de665994191e2d4e75b4d4e673f2191a947ec569ca`，35816 bytes。
- `browser-checks.json`：SHA-256 `527a0664491ef0ca1752ad401116df450e516b810fab4413959a8f7a53ca3a68`，163719 bytes。

上述为提交前本地验收；公开发布仅在同head Actions/Pages成功、build-info SHA一致、线上逐资源200/hash/尺寸与真实浏览器验收后报告。大模型/训练state/全量Q表均不入Git，训练目录只读，r18与GPU不操作。
