# v3-items 专项候选可选上架（20261107）

消费训练 owner 的真实 READY：DUEL u019902 τ0.3 与 SP u023302 τ0.3，两者均为可选项，默认 G_frozen τ0.3 不变。以已发布 v4-facing main 为基础，未修改训练源码或运行进程。没有重跑原856局。

## 原始证据与当前规则

所有原始成绩标注为 **bun-native-v3-items**（引擎 `24c5bcc0f12088a11f24bc0f581d4b3cfd25094148a144afa573bf7af88a7c40`），不是当前 v4 成绩。原交付包含216局筛查、320局独立确认、320局网页快筛；每候选τ0.3说明对应人工短局dev/holdout各16、完整局holdout8、五网页对手各8，共80局。仅P1、有选点与有限样本限制。

|候选|v3 长处依据|v3 短板|
|---|---|---|
|DUEL u019902 τ0.3|人工短局dev/holdout因果击杀8/7，同批G为1/0；网页40胜/0负、交付40|网页自死9次，G为4；完整局可行动静止23.94%，G为18.83%，非全面更流畅|
|SP u023302 τ0.3|网页39胜/0负/1平、自死1次，G为37胜/3负、自死4；困难猎手8局因果击杀7、交付10，有连续喷泉信号|完整局可行动静止34.08%；近身击杀未建立显著优势；喷泉串弱于同点argmax|

公开原始候选指标、比较及配对区间：`web/specialists/items-20261107.json`。公开件只选取明确字段，未复制私有READY中的本机绝对路径。

当前 v4-facing 追加 **24局**针对性兼容快筛：候选各8、G参照8；每模型4个人工近身holdout、困难猎手2、协作猎手2，沿用原v3相同seed/任务子集、τ0.3、P1；所有新局完整跑到其原任务终止条件并独立重放检查。

|模型|v4 近身4局 W/L/D · 因果杀|v4 两类猎手共4局 W/L/D · 交付 · 因果杀 · 自死|
|---|---|---|
|DUEL|3/0/1 · 3|4/0/0 · 7 · 6 · 4|
|SP|2/1/1 · 1|4/0/0 · 5 · 4 · 0|
|G_frozen|0/0/4 · 0|3/1/0 · 3 · 0 · 0|

新v4与原v3相同seed子集的胜负、击杀、交付、自死和喷泉指标一致，未观察到描述的明显反转；小样本不能证明全面优势。独立列出两规则对照及逐局SHA：`web/specialists/v4-screen-20261107.json`。网页说明卡同时显示原v3证据和当前v4快筛，未把v3数字冒充v4。

## 导出、身份与资产

原checkpoint与COMMITTED的SHA均核验。以本地CPU独立重导出，38维→37维只移除末尾固定mode0输入行，8组投影误差均0；保留权重flat/tensors逐项完全相同；重新导出的ONNX与READY SHA完全一致。正式发布保留生产方原始JSON/ONNX字节，避免元数据差异改变原证据身份。

|模型|完整policy SHA256|发布JSON SHA256|发布ONNX SHA256|
|---|---|---|---|
|DUEL|c2058a0f2dbf0bb2ecd4ea9b17be975a2bd51caf4e47c1713d17c9c6947f0a70|b90889c1136841206f6497f23f0771f1292e10dc8b5b1872e7f8cb28594fa109|6ce7ee768c9f6a80cef455a23e7eb9305a3a748682de776765b6d16036a899d3|
|SP|39b0dea7327221891bdd4cfba93ab917c897aa86fd78d83401241ad945a982cf|6d175a9dba370d45ac6218c066b9b1eb154c2a329c58ed242c9bed91e6853543|c22e3ea0ab8c3c86389f374c693df6948b50471d39910a920a722d7140ca7037|

v3及v4正常图/对拼双方12状态+零输入，每模型每规则13个样本；JS/CPU ONNX最大误差均小于0.002、合法masked argmax完全一致。[独立导出核验收据](cx_web_specialists_items_v3_20261107/candidate-export-verification.json)。此前四种37/38/52/53维导出对接夹具仅用于接口测试，不是候选能力证据。

四个SHA命名资产仅上传 `web-models/`，对象public-read、immutable，bucket保持private、未扩大CORS（仅GitHub Pages和本地预览）；匿名重下校验大小与SHA成功。[DUEL](cx_web_specialists_items_v3_20261107/oss-duel-upload.json)、[SP](cx_web_specialists_items_v3_20261107/oss-sp-upload.json)。凭据仅本机环境加载，未进入仓库、前端、报告或日志。

[GitHub Release回退](https://github.com/allwellll/qqt-RL/releases/tag/web-items-specialists-20261107)已发布，四个服务端SHA256 digest与本地/OSS一致；CI静态站构建将从Release实际下载并再次SHA核验。网页仍使用OSS→Release→同源镜像的现有回退链。

## 测试与浏览器

- `npm test` 全套成功，覆盖v4、v3物品、旧Review、模型目录、控制和音效；新增说明范围校验测试通过。
- 真实Chromium 1280桌面与390手机触摸视口：分别选择两候选，真实worker-wasm推理至少3次、当前物理v4、τ0.3、长处/短板/两规则证据说明可见；每候选实际OSS请求状态verified、SHA通过，无页错误/横向溢出；回切G默认仍τ0.3。[本地结果](cx_web_specialists_items_v3_20261107/specialists-local-browser.json)。属于桌面Chromium手机触摸环境，未冒称真iOS设备。
- 隐私边界为tracked与本次明确交付文件/资产；未扫描历史本地run/checkpoint。公开JSON/ONNX不含私有本机路径、标识或凭据，tracked隐私扫描0命中。

## 发布

待正常push与同SHA Actions/Pages、线上桌面/手机选择和OSS加载回读后回填。
