# R3恢复与u4002发布（20261031）

授权和消费：`runs/r3_resume_publish_2110_20261031/AUTHORIZATION.json`、`TASK_CONSUMED.json`。当前任务优先于旧录像剩余部分，旧录像已完整验收，不重录。

## 训练接续

- SP主seed从u3316、第二seed从u3409完整Actor/Critic/Adam/env/aux/RNG/ring恢复；原640更新终点u3680，分别剩364/271更新，固定55%，不升70%。两次fresh-process next-step、Adam、env/aux/RNG、ring验收均PASS。
- 历史真实bot回退仍存在，并不支持“历史只有passive_increase”说法；本次明确恢复授权覆盖恢复前点的旧停止指令，后补旧评估保留为不利证据。恢复后新点仍受bot与资源护栏约束。被动局增多仅禁止升比例。
- GPU2给SP主seed；GPU3当前R2-a smoke/恢复自然退出后已给SP第二seed。R2-a已赚取的u5282/u5283保留，仅较新分配覆盖原GPU3正式启动要求。
- 现场GPU0/5旧BT线已在训练，和21:10空卡快照不同；不向健康worker发信号，已安装完整chunk退出后专卡交接。R3-a/b均从u4002各续640更新，GPU0/5的持久唯一R3 owner等待领取，不与旧worker重叠。详见`R3_ALLOCATION_QUEUE.json`和旧BT各slot_CONSUMED收据。
- slot01旧chunk至u15040后复制slot07的授权继续，由原owner处理；其旧R2-a验收期待已显式标为被21:10资源授权覆盖，不冒称正式更新完成。

## 选型及发布

2304既有完整自然局逐分片SHA、模型身份、规则、seed、出生/复活本侧、终局、模式和每席128分母全部复核。三模型×三档，每档256。以下为greedy，不混categorical；单地图、16重复几何，不宣称稳定或统计显著。

|模型|P1 hard/normal/easy|P1合计|P0合计|自伤|护栏|
|---|---|---:|---:|---:|---|
|网页u3040|91/110/119|320|345|0|基线|
|R3-a u4002|99/116/123|338|353|0|通过|
|R3-b u4002|81/111/124|316|338|0|hard未过且总分未超|

候选固定R3-a u4002，policy SHA `f12ab745ca746ae364bcb172fb425399cbf47038f60f7129617036a4051d6a30`。协作猎手hard基线256局复用，候选新增256局压力列CPU异步执行，不参与选型门槛。发布namespace `runs/r3_resume_publish_2110_20261031/selection/publication`，规则bun-native-v2保持不变，仅更新默认模型、目录验收测试；健康训练不等待发布。

录像已实际播放验收：`runs/new_model_replays_20261031/RECEIPT.json`，五模型60入口/53独立完整局；其中R3-a u4002固定前8局及披露补选失败/平局。本轮复用，不构造或择优重录。网页发布须继续完成逐张量/JS/ONNX/fp32、回归、浏览器完整真人模式局、隐私、普通tag/非force快进、同head Pages、线上SHA，尚未完成不称上线。

## 实时证据

恢复首更新以`RESUME_2110_FIRST_UPDATE.json`为准，冷编译与暖态分开；原子点另验state/policy SHA。完整控制链及GPU样本见`FIRST_RESUME_STATUS.json`。恢复与发布仍在执行，未完成项未标为完成。

复现：`.venv/bin/python -m scripts.r3_a4002_publication_20261031`（已接受分片不重跑）；训练只由原SP/R3 owner消费版本化恢复合同，不另建训练owner。

## 协作猎手hard压力列（已完成）

同冻结seed2026321000起，每模型256局、各席128，argmax/老师关闭、本侧出生。完整2816局证据已完成（新跑256，复用2560）。

|模型/席位|W/D/L|WL|因果击杀|死亡次数|零泡|自伤|
|---|---|---:|---:|---:|---:|---:|
|u3040/P0|118/5/5|113|62|20|0|0|
|u4002/P0|119/1/8|111|57|35|0|0|
|u3040/P1|103/13/12|91|47|23|0|0|
|u4002/P1|110/7/11|99|41|64|0|0|

**压力短板**：P1胜负差提高8，但死亡次数+41、击杀−6；P0死亡+15、击杀−5，不能把自然三档胜负差提升表述为全面战斗增强。这些死亡是复活制事件次数，不等于负局数。按明确授权只单列，不事后增加发布门槛。

恢复技术验收23项通过，具体命令及源码SHA见`runs/r3_resume_publish_2110_20261031/FOCUSED_ACCEPTANCE.json`。两seed恢复首正式分别u3317/u3410，每次262144真实步、32 Adam步、Actor/Critic均非零、finite，首冷周期与暖态分开。
