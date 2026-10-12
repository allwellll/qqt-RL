# Bun392动态自对弈3072 模型发布（20261012）

来源为 `runs/je03_bun_stage2_20261012/bun392_selfplay12_v2/formal/bun392_selfplay_dynamic/gen003072`。
COMMITTED、plan、完整恢复状态、raw/EMA及权威评估全部分片均重新校验；没有修改历史checkpoint或启动训练。
完整来源、导出和评估hash见 `reports/je03_bun_stage2_20261012/bun392_model_release_20261012.json`。

## 策略选择与限制

| g3072策略 | Hunter hard/normal/easy安全击杀（各64） | 总计 | Tactical（64） | Hunter死亡/自炸/换命（按档） | full任务死亡/自炸（16） |
|---|---|---|---|---|---|
| 原始policy，网页发布 | 0/4/23 | 27/192 | 24/64 | 1/0/0、0/0/0、3/0/1 | 2/2 |
| EMA，独立同冻结协议追评 | 1/5/16 | 22/192 | 18/64 | 0/0/0、2/0/0、1/0/0 | 0/0 |

优先核查EMA后，其综合击杀快筛低于原始策略，故发布已评估的探索最佳raw，而非把raw成绩贴到EMA上。
EMA同协议CPU追评137.02秒，保留于新的 `publication_ema3072_v1`，不覆盖原权威结果。
两份策略carry_home均14/16、near_steal均16/16；route_break/full的capture/steal/return均0。
64局/档仅探索快筛，没有独立seed或扩大样本确认，不能描述为显著性晋级。困难档raw为0，改善主要来自easy；
完整任务及复杂路线未突破，raw的full死亡/自炸风险仍在。4096回落，未误选终点。
Tactical是原生JAX300tick诊断；Bun任务是idle对手诊断，与真实网页Hunter分开解释。

## 发布格式与复现

沿用 `qqt.web-models/v1` 和现有base64 fp32 JSON/Transformer浏览器协议。
原生Bun：24×13×15平面、global24、embed392/depth4/heads4/patch3、5移动与3能力；无非Bun输入adapter。
只发布Actor及现有导出格式的value head；不发布独立Critic、优化器、训练状态、原始pickle、run或日志。

- Release/tag：`web-bun392-selfplay3072-v1`。
- Release：<https://github.com/allwellll/qqt-RL/releases/tag/web-bun392-selfplay3072-v1>。
- JSON：`bun392-selfplay3072-raw-c100cc13aa01315b9263b300448c18a5d9a534af07438ae6b86e07ca7a02e94a.json`，40,294,795字节。
- JSON SHA256：`c100cc13aa01315b9263b300448c18a5d9a534af07438ae6b86e07ca7a02e94a`。
- ONNX SHA256：`7618ec5f6c23b2e2f0c9f45d086a6ca1429e73808aab3276c8b5cdbf29af5639`。
- policy SHA256：`dc3919a5e7a2f415f951e23f20069d14c3c1df332acf5dfa7b1bb21314ca4f70`。
- EMA SHA256：`c2215cc50d06360f26f66ef1da5bf3ca102306e63b1ff87287a2b2c112bddead`（仅比较，未发布EMA权重）。
- COMMITTED SHA256：`a6b4149329b42f203e58d05d607866dd357c1d3d657a9c40474e42aedf658c8b`。

复现命令（外部位置仅为占位符；时间参数固定以保证JSON字节可复现）：

```sh
CUDA_VISIBLE_DEVICES='' <PYTHON> scripts/export_bun_release.py \
  --checkpoint <CHECKPOINT_DIR> --output <EXPORT_DIR> --strategy raw \
  --source-ref runs/je03_bun_stage2_20261012/bun392_selfplay12_v2/formal/bun392_selfplay_dynamic/gen003072 \
  --generated-at 2026-10-12T00:00:00Z --onnx
```

两次独立JSON导出逐字节一致，全部packed权重与原权威评估model一致。
8组合法状态/双席位浏览器JS和ONNX最大误差5.72e-6。
当前main真实网页Hunter三档各8局smoke为0/0/3，300tick加残余危险均完成，耗时10.14/8.64/8.68秒。
当前main Hunter源码已前进，这些smoke不混入冻结训练曲线。

## 网页、测试与隐私

目录新增“Bun392动态自对弈3072”，旧5模型不变；Release URL为资产来源，Pages按目录下载、校验并同源提供，避免浏览器Release跨域问题。
`fetch_web_models.py`支持逐模型独立Release/tag，拒绝非Release URL、路径穿越及损坏资产。
`verify_bun_release_browser.js`执行真实目录选择、下载、大小/SHA校验和原生Bun1v1开局；本地16tick/16次有限推理通过，不提交远端赛果。
12项Python测试、完整npm回归、语法检查及diff检查通过。公网与精确head的Actions/Pages最终状态以发布后回执为准；成功前不宣称部署完成。

隐私扫描仅覆盖 `git ls-files -z` 和明确待发布/待提交文件；旧已追踪文档中的用户标识路径、内网地址、机器PID改为相对引用/中性占位符，不重写已发布历史。
测试网络样例改为文档保留地址。模型flat是已核验的数值base64，偶然出现PID样式不当作机器元数据；所有元数据仍扫描，用户标识扫描覆盖整个文件。
根级本地 `AGENTS.md`明确后续代理路径/隐私规则，根级 `/AGENTS.md`忽略，不纳入Git。

## Release下载瞬态故障修复

首次推送head `7265ee5` 的Actions `38265081242` 最终failure：npm成功，但Assemble首次下载Release资产收到HTTP500，旧下载器没有重试，deploy未执行。保留此失败记录，不重跑旧代码冒充修复。

下载器现默认最多5次请求，仅对HTTP429/500/502/503/504及网络超时/断连重试，指数退避1/2/4/8秒；Retry-After支持秒数及HTTP日期，取其与退避的较大值。单次等待预算60秒，服务器要求更久时明确失败，不提前请求。其他HTTP、未知URL/TLS证书、完整性错误立即失败；失败不跳过模型。所有6模型仍须完整大小与SHA256校验后原子写入。

确定性测试覆盖首次500恢复、持续500耗尽5次失败且不产出模型、hash损坏保留原文件、Retry-After两种格式与超预算、非瞬态立即失败和断连恢复。相关Python测试26项通过，HTTP/目录及语法检查通过。本地完整静态站使用已公开且重新核验的资产缓存，经相同生产目录/大小/hash/原子写入路径装配；新模型来自实际Release完整下载，旧5模型来自此前Pages公开缓存，不跳过任何模型。

完整装配站真实浏览器测试通过：目录选择后实际加载新模型，原生Bun 1v1完成16tick及16次finite推理，无网络mock。
