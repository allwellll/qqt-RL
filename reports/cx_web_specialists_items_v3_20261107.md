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

正常快进push main：`cd9cddf0af110f7825a31181e519e08ee1b8921f`。

[同SHA Actions/Pages](https://github.com/allwellll/qqt-RL/actions/runs/42346134048)：test-and-build、deploy及整个workflow最终均success。构建从Release实际下载并SHA校验全部模型资产，已证明新候选回退资产可用；本机代理对Release匿名链接下载失败不冒充成功，OSS匿名回读及浏览器均成功。

[线上关键文件逐字节回读](cx_web_specialists_items_v3_20261107/specialists-release-public-readback.json)的build-info.commit与上述SHA一致，默认物理仍v4。

[线上桌面/手机浏览器结果](cx_web_specialists_items_v3_20261107/specialists-public-browser.json)：1280和390各选两候选，worker-wasm实际推理、每候选OSS下载verified、SHA通过，v3原评测和v4快筛说明同时可见；默认G回切、τ0.3保持，零页错误、无横向溢出。截图已查看，本任务最终线上证据回填本地报告，不为回填触发额外部署。

## 最新授权与 READY 复核（20261107）

用户最新明确：立即停止训练，撤销全部后继，覆盖此前等待active段完成的安排；未来重新授权前不得续训、启动新训练或后继。训练owner已交付worker归零、八卡释放与COMMITTED保存回执；持久停止闸门由训练 CX 管理；网页交付侧不写训练工作树、不更改或终止其进程，不请求任何新训练。后续选模只允许读取既有 checkpoint/READY 并评估。

本次只读复核 READY 仍为 DUEL u019902 τ0.3、SP u023302 τ0.3，两候选 JSON/ONNX SHA 与已发布 manifest 完全相同，没有新的待消费候选，不重复导出上传或重跑能力评测。证据：`cx_web_specialists_items_v3_20261107/latest-web-ready-check.json`。现有长短板标注、默认 G_frozen τ0.3、手机修复均已实际部署，不等待训练或全面胜出。

当前 main 为 `c2ed1cb26b4c9546c7e8554f10dceae2eb451c74`；同 SHA Actions/Pages 最终 success，部署后桌面/手机候选 OSS 校验与真实推理均 PASS。此次复核只回填本地报告，不为无代码变更重复发布。未将旧 READY 中历史性的 training_continues 字段解释为当前续训授权。

发布回执 `.hermes/WEB_CANDIDATE_PUBLISHED.json` 已生成，并按用户明确要求回写训练侧同名交付文件；包含两候选资产身份、OSS、同 SHA Actions/Pages、桌面/手机真实加载与默认保持说明。训练侧仅写该交付回执，未触碰训练源码、停止状态或进程。
