# A3SR u1280 网页默认与新录像发布 — 20261030

状态：发布生产提交已上线，导出/准入/回归/Release/tag/main快进/同SHA Pages、完整资产hash与线上浏览器P1整局验收通过。浏览器使用已核验Pages模型字节缓存；未缓存冷加载曾超时，性能限制如实另列。训练与异步评估不停。

来源 `runs/bomb_threat_v1/formal/bt_slot05_c000028/u001280`，严格对应slot05 c29起点，与fresh256固定a3sr_u1280身份一致。policy SHA `a552a3094182db7d49d498d8ebe37cdae34fc8ba058396661b7e58c2e812699a`。

fresh256 192/192分片已完整验收，两个候选1536新增局与u800复用768局；本次发布固定u1280，不按latest或峰值选择。生产JS猎手bun.hunter，stock-window-v1，argmax，teacher off；每档256局，P0/P1各128。网页真人固定P0、模型固定P1。P1 hard/normal/easy胜负差+13/+26/+52，合计+91；u800为0/+6/+11，合计+17。满足批准的sum>u800、每档≥u800−5标准，不宣称逐项全面不差、显著性或真人泛化。

| 档位 | A3SR胜/平/负 | u800胜/平/负 | 样本数/模型 |
|---|---|---|---:|
|hard|53/196/7|42/207/7|256|
|normal|68/185/3|68/180/8|256|
|easy|111/141/4|87/154/15|256|

已知短板：仍有零泡局、左右振荡、持包被砖困；P1击杀等并非逐项全面改善。64出生几何循环复用，不当作256独立地图。

## 导出和验证

75个权重张量逐个精确。JSON元数据便携化，flat/tensors保持评估版本不变；ONNX复用同policy原生导出并核验SHA。8个双座位fixture的Python fp32/JS/ONNX/Chromium最大误差<8e-6，masked argmax相同。训练bf16数值漂移另列，不能称部署与训练逐位一致。
真实本地对战页worker-WASM完成2400tick终局，模型P1、greedy=true、nativeItems/nativeTrap=true；2400次动作finite，推理16.8–55.7ms（此主机短程测量），console/pageerror0。口令未提供，仅程序化选择目录第一模型触发既有change事件，没有改生产鉴权；默认模型是模型目录首条，不声称默认规则对手自动改为模型。所有数据库写入已屏蔽。
对战页sim SHA `2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45` 保持不变。native规则3840对照帧通过。旧11条模型对象完整保留；旧Review/training/排行榜未改。

- `bun392-a3sr-u1280-736bc0c2d458682d771879be0cca80e25b90779336440beb40cbd800de0b7fed.json`：40294780 bytes，SHA256 `736bc0c2d458682d771879be0cca80e25b90779336440beb40cbd800de0b7fed`
- `bun392-a3sr-u1280-f3ab40a46f935796b50185a63f5b43ed28ee66abd67c57d1d23a9b6af592abe1.onnx`：30242279 bytes，SHA256 `f3ab40a46f935796b50185a63f5b43ed28ee66abd67c57d1d23a9b6af592abe1`

## 新录像固定清单

新建 `20261030-a3sr-u1280-review.html`；旧u800页面未覆盖。21入口/18独立完整局，先固定首8局（两座位各4），再各座位最早3胜/1负、最早2个P1单方零泡，以及u800 BRICK_OBSERVATION_MANIFEST前3个同seed。重复入口如实披露，不搜索额外对局。每局与fresh256原始胜负/泡/因果击杀/死亡/交付/终局逐项一致；Review完整hash/播放/终局seek均通过，奖励显示未记录。

| 入口 | index | seed | 座位 | 胜负 | 因果击杀 | 泡 | 交付事件 | 选局规则 |
|---:|---:|---:|---|---|---:|---:|---:|---|
|1|0|2026142900|P0|平|0|37|0|顺序前4局P0；与旧u800页01–08同seed|
|2|1|2026142901|P1|平|0|1|0|顺序前4局P1；与旧u800页01–08同seed|
|3|2|2026142902|P0|胜|1|23|1|顺序前4局P0；与旧u800页01–08同seed|
|4|3|2026142903|P1|平|0|4|0|顺序前4局P1；与旧u800页01–08同seed|
|5|4|2026142904|P0|胜|1|45|0|顺序前4局P0；与旧u800页01–08同seed|
|6|5|2026142905|P1|平|2|120|0|顺序前4局P1；与旧u800页01–08同seed|
|7|6|2026142906|P0|平|0|12|0|顺序前4局P0；与旧u800页01–08同seed|
|8|7|2026142907|P1|平|0|92|0|顺序前4局P1；与旧u800页01–08同seed|
|9|2|2026142902|P0|胜|1|23|1|P0按index最先3个胜局|
|10|4|2026142904|P0|胜|1|45|0|P0按index最先3个胜局|
|11|28|2026142928|P0|胜|1|32|1|P0按index最先3个胜局|
|12|34|2026142934|P0|负|0|20|0|P0按index最先1个负局|
|13|27|2026142927|P1|胜|0|18|1|P1按index最先3个胜局|
|14|45|2026142945|P1|胜|1|31|1|P1按index最先3个胜局|
|15|53|2026142953|P1|胜|0|21|1|P1按index最先3个胜局|
|16|59|2026142959|P1|负|0|27|0|P1按index最先1个负局|
|17|37|2026142937|P1|平|0|0|0|P1按index最先2个单方零泡局|
|18|39|2026142939|P1|平|0|0|0|P1按index最先2个单方零泡局|
|19|3|2026142903|P1|平|0|4|0|u800砖隔开清单前3个同seed|
|20|11|2026142911|P1|平|0|2|0|u800砖隔开清单前3个同seed|
|21|19|2026142919|P1|平|0|3|0|u800砖隔开清单前3个同seed|

自包含HTML 1639742 bytes；SHA256 `747a9f846463c57ec8e36041da4f25d2dd7dc7321268355f2666de8423f24895`。OSS上传HTTP200回读bytes及SHA一致；浏览器远端21入口完整hash、终局seek、每入口至少10tick播放通过；短程约59fps，不当作持续性能基准。外网地址由本地RECEIPT交付，不向追踪文件写入私人对象前缀。

## 生产发布

生产提交 `bec569eabbb9ec1d61b16603c4d3b4d8b862d4d8`；tag `web-bun392-a3sr-u1280-20261030-r1`。
[Release](https://github.com/allwellll/qqt-RL/releases/tag/web-bun392-a3sr-u1280-20261030-r1) 四份资产已上传并核验digest。
[生产Actions #40997063896](https://github.com/allwellll/qqt-RL/actions/runs/40997063896) 在同一生产SHA最终completed/success；构建包含npm全量及Review测试、12个目录模型完整资产校验和Pages部署。

本地 `npm test`、`npm run test:review` 均PASS；Review历史可选外部模型检查按原合同SKIP，本候选由独立真实加载完整验收覆盖。`git diff --check`、隐私扫描0 findings；Git最大文件7,958,608 bytes，大模型仅在Release。181个既有Review/规则/训练相关追踪文件逐字节不变。

网络运输异常如实保留：原生Git tag push受代理认证/CONNECT失败；并发Release调用在推送确认前自动创建原始tag，指向旧main。原tag未移动/删除，旧Release已注明非生产；新增r1不可变tag绑定本生产SHA。后续通过GitHub官方Git Data API上传与本地SHA完全一致的blob/tree/commit/annotated-tag，以祖先校验及force=false快进main；没有冒称原生Git push成功，没有force/reset或历史改写。原main `4f5bf548e31c82a6ca97a9a638ede1a2350bc428` 保留为祖先。

线上 `https://allwellll.github.io/qqt-RL/` 的build-info精确返回生产SHA；models.json全文SHA `ae34dc32873408d686b784af612bd0cce0201b4c3ff941951da002a3a3e74012`，首条u1280、旧11条保留。Pages同源JSON及ONNX按严格HTTP Range下载完整bytes，SHA均与Release/本地相同；实际对战页加载stock-window-v1，sim SHA仍为前述2767cb4e完整hash。

线上Chromium真实P1对局PASS：2400tick、2400次动作、完整终局，worker-wasm、greedy=true、nativeItems/nativeTrap=true；8组fixture masked argmax一致，console/pageerror0（已知CPU vendor警告另列）。推理本机短程 17.3–53.7ms，不代表其他用户设备或网络性能。数据库写入全部屏蔽，设置口令未测，只通过既有选择事件选目录首模型。

网络边界：首次未缓存冷加载在模型Range请求的120秒超时重试后失败（The user aborted a request），独立原样重试单列收据，不能冒称冷加载性能验收成功。完整P1实战采用已从Pages下载并完整SHA核验的ONNX字节缓存，页面/脚本/目录/规则/ORT runtime仍真实线上加载；没有替换模型计算或伪造动作。缓存验收与冷网络情况严格分开。本次未修改下载器或超时设置。

本报告及ledger随后作为纯文档快进提交；最终head的Actions/build-info验收单列在完成收据，模型release tag保持本生产commit不移动。

## 收据与复现

- 导出/跨端/本地浏览器/测试与发布：`runs/bomb_threat_v1/publish_a3sr_u1280_20261030/`。
- 录像固定manifest、逐局raw、完整Review、上传回读与本地/远端Chromium验收：`runs/bomb_threat_v1/replay_a3sr_u1280/`。外网链接在其`RECEIPT.json`，本报告使用中性占位`<REPLAY_URL>`。
- 线上完整字节：`<PYTHON> -m scripts.a3sr_verify_online_assets`；线上真实浏览器：`WEB_URL=https://allwellll.github.io/qqt-RL/ EXPECTED_HEAD=<PAGES_HEAD> node scripts/verify_a3sr_publication_browser.js`（需本地浏览器依赖）。生产JS argmax、teacher off，所有数据库写请求被本地拦截。
- 新旧JSON资产SHA：u800 `455d40c702ff444e3c8cd5c01275c67c7dbe31817ec690986a67d576bafca928` → u1280 `736bc0c2d458682d771879be0cca80e25b90779336440beb40cbd800de0b7fed`；旧条目与资产不改。

