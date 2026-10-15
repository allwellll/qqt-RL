# JE09 Stage C g8模型发布回执（20261015）

发布对象为Stage C g8 **raw Actor**（来源提交`5777835`），网页名“Bun392 JE09运包桥接g8”。Stage D没有综合晋升，本次不发布D g16。保留main原有7模型，新增独立条目，共8项。

Hunter hard/normal/easy安全击杀2/3/29（各64局）、Tactical17/64；carry near/mid/far capture16/16/2（各16局）。同局受控真实掉包返程主评/holdout首拾均16/16、25%均16/16、50%为11/16和9/16、capture为15/16和13/16。旧fixed32首拾0/16且自炸2/16，full2400归因drop/首拾/capture仍均0/16。课程返程改善不代表完整自然链解决；样本仅探索，不宣称统计显著或各维度支配。

Release：https://github.com/allwellll/qqt-RL/releases/tag/web-bun392-je09-stage-c-g8-v1

网页：https://allwellll.github.io/qqt-RL/

tag：`web-bun392-je09-stage-c-g8-v1`，只创建一次，不覆盖历史tag/Release。

## 发布前验证

复用main既有`export_bun_release.py`导出，固定时间`2026-10-15T00:00:00Z`；JSON/ONNX/manifest两次字节一致，packed张量逐项等于哈希核验的raw Actor。原生24×13×15/global24、embed392/depth4/attention4、5移动及3能力动作、value输出。既有8状态Python/JS/ONNX parity覆盖双seat、糖泡、死亡复活和持包；另含zero参考与原生1v1状态，finite且在既有容差内。训练BF16误差单独诊断，不冒充FP32一致性。

catalog测试先RED，新增条目后GREEN；29项Python导出/下载/隐私测试、main预览完整npm测试（含HTTP smoke）、git diff检查通过。新增文件、全部Release资产与待发布main树隐私扫描零发现；实验分支历史已追踪遗留只审计、不改写、不进入main白名单。未扫描或修改ignored旧run。

只上传raw JSON、ONNX、manifest、evaluation摘要和SHA256SUMS。原始pickle、Critic/optimizer/state及训练日志不上传。来源与每项资产size/SHA见同名JSON。

## 真实发布与公网验收

不可变tag指向发布提交`2784af81a1ae594a8514b08a6297f4e97f93cb62`。Release非draft、非prerelease，五资产均已上传，GitHub服务端size/digest全部一致。main从`5ba6ccd9a54659fe5198f90f9992edd616b6d548`非强制快进：Git分支/tag推送成功，main直推CONNECT503后通过GitHub API更新到完全相同的Git对象；未覆盖tag、未改写历史，实验分支继续保留全部训练历史。

[精确发布提交Actions](https://github.com/allwellll/qqt-RL/actions/runs/38749341434)的test-and-build和deploy均completed/success，Pages deployment `7095006037`最终success。公网build-info精确返回该SHA；models.json及7个关键网页文件与待发布树逐字节一致，全部旧7模型条目原样保留。

五个Release资产经无凭据公开代理分段HTTP回读后，与本地逐字节、size及SHA256全部相同。当前环境直连GitHub下载超时，代理只作为公开传输，未使用凭据；公网浏览器直接访问Pages并下载同源模型，无网络mock。

真实Chromium选择“Bun392 JE09运包桥接g8”，下载通过生产loader大小/SHA校验，使用worker-wasm后端，原生Bun双玩家1v1执行16tick、16次finite模型推理；另完成Hunter三档300tick及残留危险结算技术smoke，不能把技术smoke当新权威能力分数。pageerror、意外console error、模型HTTP错误、意外写请求均0；旧站根favicon404及ORT宿主CPU识别警告单独记录，不冒充零警告。验收采用真实HTTP/1.1，保持生产loader和模型不变；当前环境HTTP/2分段吞吐慢，4个256KiB分段HTTP/1.1实测3.45秒。请求仅被动观测，不拦截、不mock。截图及逐请求/逐推理证据在同名JSON和`runs/je09_stage_c_release_20261015_v1`下。

本回执后续以JSON/Markdown及浏览器验收器整合到main；web树不变，验收器仅增加被动请求观测、失败诊断与可选HTTP/1.1。模型tag仍固定上述发布提交，不移动。回执提交自身的精确SHA无法写入自身，最终回执提交的Actions/Pages/build-info与Chromium证据在相对验证namespace继续保存，并在最终交付中给出精确SHA。

| 模型资产 | bytes | SHA256 |
| --- | ---: | --- |
| raw JSON | 40294827 | `b80373c1a34cceb6eaeb549074c4fa154313b7b31272b42698f4430e2c7c50d2` |
| ONNX | 30242279 | `b606294fc8bc1e137c55db7600e4a1eb0284b03a1721279869978ba3bf82764d` |
