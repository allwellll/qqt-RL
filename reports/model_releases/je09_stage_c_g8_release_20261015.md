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

## 发布进度

发布前验证完成；本段将在真实Release上传、精确head Actions/Pages和Chromium公网验收后更新为最终回执。
