# Bun392战斗运输1536发布（20261013）

发布JE05 `bun392_combat_transport_pfsp8` g1536 **raw** Actor。来源`runs/je05_bun_20261013/combat_transport_pfsp8_v1/formal/bun392_combat_transport_pfsp8/gen001536`，完整checkpoint及对应Critic恢复状态已校验。EMA哈希已校验但未同协议评估，本次只发布实际获得下述成绩的raw，不把raw成绩归给EMA。正在运行的JE06不参与本次候选选择，未停止、热改或写入其目录。

## 同协议候选与局限

| raw候选 | Hunter hard/normal/easy /64 | Hunter死亡/自炸/换命合计 | Tactical安全击杀 /64 | 原生carry capture /16 | 原生near steal /16 |
| --- | --- | --- | --- | --- | --- |
| JE04 g768 | 0/9/24 | 1/0/0 | 19 | 0 | 13 |
| JE04 g2048 | 0/5/29 | 6/1/2 | 26 | 0 | 13 |
| JE05 g1536 | 3/1/32 | 3/0/1 | 22 | 15 | 15 |
| JE05 g3072 | 2/6/28 | 5/3/0 | 17 | 16 | 16 |
| JE05 g4096 | 1/2/27 | 3/0/2 | 18 | 15 | 14 |

JE04两点使用已补齐的JE05真实网页原生任务结果，不能沿用旧JAX idle代理分数。g1536 Hunter合计36/192、无自炸，Tactical死亡/自炸/换命均0；near死亡1、无条件抢包1。相邻g1024 Hunter总36，g2048回落至23且安全退化，因此保留中间点；不默认终点最好。g1536是战斗、任务与安全的Pareto折中，不支配所有维度：normal仅1/64，弱于g3072和JE04 g768；carry/steal也没有满分。所有候选300tick route_break/full capture均0；600/900tick诊断亦未capture，完整抢包运输仍未突破。每档64局、任务16局仅探索快筛，不是统计显著性或最终晋级证据。

## 策略与导出

- raw SHA256：`16ba817d53ef1f3ed4cf6039ff58f33f1f0de4adc69b3578047c5c3cdfbda719`。
- EMA SHA256（未发布）：`96355fe4a9826bc6d9a63107210b5dede02ac1bb59100038531c880bcdeee48d`。
- COMMITTED SHA256：`34cede065d72d2c3674eaec42cf8f3e57292a2fc0bc40fb47a10c2d2c1b67e44`。
- 完整state SHA256：`867d38c103a58faaec510de7fa14528df8c0a05cb0c9c1982b8ede0bbf6f5bab`，state/优化器不上传。
- JSON：`bun392-transport1536-raw-2ec354adfd72c65f0b5b601b40a13d7f879729350572897c660cbc2866710dc0.json`，40,294,842 bytes。
- ONNX：`bun392-transport1536-raw-d5c8eb84936c18610d50f7044a13ec7d2ad480d5024b96ee75f82e1441b7b60a.onnx`，30,242,279 bytes。

两次独立导出JSON、ONNX及manifest逐字节一致；固定时间`2026-10-13T00:00:00Z`。packed fp32张量逐项对应原raw。保留原生24空间平面/global24、392×4、完整15联合动作，无非Bun adapter。checkpoint声明`bun24-trap-timer-v2`：仅global9/10在alive基值上加自身/对手糖泡剩余ticks/60；维度不变、慢慢胶不等同糖泡。网页模型按元数据使用该原生版本，旧模型默认编码完全保留。

8个合法状态含双出生席位、糖泡、死亡复活窗口、携敌包；Python FP32、JS和ONNX最大误差约4.77e-6，合法masked greedy动作一致。训练JAX BF16相对FP32误差最高约0.113，单列诊断，不声称BF16逐值等同网页FP32；本次成绩本来来自FP32网页权威路径。

## 发布与验证流程

沿用`scripts/export_bun_release.py`与现有模型目录/worker协议，新增可配置资产前缀、内容寻址ONNX和显式原生编码。27项Python发布/下载测试通过，完整npm网页测试通过；本地Chromium worker实际加载、1v1开局及Hunter三档技术smoke由`scripts/verify_bun_release_browser.js`验证。完整静态站必须逐资产校验大小/SHA；Git追踪文件、待提交文件与Release说明/元数据独立隐私扫描，根级AGENTS保持忽略。凭据只在本地内存读取。

新tag/Release：`web-bun392-transport1536-v1`，[Release](https://github.com/allwellll/qqt-RL/releases/tag/web-bun392-transport1536-v1)。JSON+ONNX、manifest和SHA256SUMS发布，不含state、Critic参数、优化器、训练日志、run、通知、锁或编排文件。网页目录首项为“Bun392战斗运输1536”，全部旧模型保留。Release远端回读、精确main head的Actions/Pages及公网页面验收记录以最终验证补记为准，尚未完成的步骤不得记成功。

```sh
CUDA_VISIBLE_DEVICES='' <PYTHON> scripts/export_bun_release.py --checkpoint <EXTERNAL_CHECKPOINT> --output <LOCAL_EXPORT> --source-ref runs/je05_bun_20261013/combat_transport_pfsp8_v1/formal/bun392_combat_transport_pfsp8/gen001536 --strategy raw --generated-at 2026-10-13T00:00:00Z --name Bun392战斗运输1536 --asset-prefix bun392-transport1536 --onnx
CUDA_VISIBLE_DEVICES='' <PYTHON> -m pytest tests/test_bun_release.py tests/test_fetch_web_models.py -q
npm test
```
