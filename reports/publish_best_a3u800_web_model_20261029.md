# A3 u800 网页发布与规则切换 — 20261029/20261030

用户明确选择方案1：对战页切换 stock-window-v1 并发布 A3 u800；A3SR 暂停在 GPU 启动前，其他七路训练与 GPU2 评估继续。发布完成后继续 A3SR。

采用最小影响方式：仅 `web/index.html` 从 `sim.js` 改为 `sim-stock-window-v1.js`。旧 `sim.js`、Review、training、历史回放及其素材逐字节不变；145 个相关追踪文件比对清单保存在本地发布验收目录，实际数量以清单为准。旧模型目录10条对象完整保留，新增 A3 u800 为目录第一条；不改 argmax 解码。

## 模型与能力边界

来源 `runs/bomb_threat_v1/formal/bt_slot01_c000015/u000800`，policy SHA256 `8153dafb2b15b7e4be17225bc95cc482417ce0afd8455081234f85f2406b16b6`，与既有 fresh256/录像完全一致。网页真人固定P0、模型P1。fresh256 hard argmax 256局为42胜207平7负，P0胜负差+35、P1胜负差0；normal 256局为68胜180平8负，胜负差+60。已知P1偏弱、零泡、左右振荡、持包被砖困住。原fresh256推广门槛FAIL，用户据这些限制明确选择发布，不宣称科学晋升；目录score=hard胜率42/256，不能与历史协议score比较。

JSON仅补齐发布元数据，flat/tensors与候选权重一致；75个张量与真实policy逐个精确校验。网页fp32与训练bf16存在数值/动作差异，不宣称逐位等价。发布metadata改变JSON文件hash，原始flat/tensors保持逐字节一致；ONNX文件复用并核对既有评估导出，hash未变。

- JSON：`bun392-a3-u800-455d40c702ff444e3c8cd5c01275c67c7dbe31817ec690986a67d576bafca928.json`；40294692 bytes；SHA256 `455d40c702ff444e3c8cd5c01275c67c7dbe31817ec690986a67d576bafca928`。
- ONNX：`bun392-a3-u800-d4d718999397d3c2a56201f09705f9ab542602a9ade5bfc7b08f4579b0ee182d.onnx`；30242279 bytes；SHA256 `d4d718999397d3c2a56201f09705f9ab542602a9ade5bfc7b08f4579b0ee182d`。

## 验证

- `node web/test_stock_window_release.js`：16个固定seed，3,840个nativeItems/nativeTrap逐帧对照；不涉及库存取包时 snapshotReplay完全一致；自由存活敌人旧版允许、新版拒绝库存取包；死亡/原生糖浆两版均允许。物理/道具实现的源码差异仅新增库存窗口gate及历史规则显式导出开关。新sim完整SHA256 `2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45`。
- `verify_bun_release.js`和`verify_bun_release.py`：8个双座位样本，75张量精确，Python fp32对JS/ONNX最大误差6.42e-6，masked greedy一致。
- Chromium + ORT WASM：8样本最大误差7.37e-6。实际对战页`index.html`中模型以P1执行24次finite动作，nativeItems/nativeTrap=true，1v1、worker-wasm、greedy=true、console/pageerror均0；24次推理约18–22ms。设置口令未提供，因此用本地测试脚本程序化选择模型并触发既有change事件，未声称验证口令交互；没有改生产口令或鉴权。所有数据库写入被拦截，未写排行榜。
- `npm test`和独立`npm run test:review`均PASS。Review原有外部模型文件可选检查因缺模型依原契约SKIP，其来源hash检查仍执行；本次u800模型单独完成精确导出/浏览器验证。
- `check_publish_privacy`：git追踪文件+本次资产元数据/Release文本0 findings；二进制私有标识另检。`git diff --check`通过，Git文件最大7,958,608 bytes，小于10MB；大模型仅放Release。历史Review/training/sim逐字节不变。

## 排行榜影响

当前match/record schema仅有client_version，没有独立rule_version字段；本次不擅加数据库字段、不删改历史记录。页面静态提示“自 2026-10-30 起对战采用新偷包规则”，同时明示新旧规则成绩混排。后续可独立迁移增加ruleset_version列、旧行标legacy、新局写stock-window-v1、按规则过滤榜单；需另行决定。本次没有执行数据库迁移。

## 发布过程与线上验收

生产提交 `090e40c2f36c6da7124e3f14a71f71bd3b3432de`；tag `web-bun392-a3-u800-stockwindow-20261030`。
[Release](https://github.com/allwellll/qqt-RL/releases/tag/web-bun392-a3-u800-stockwindow-20261030) 已公开，四份资产上传完成，GitHub返回两模型资产digest与本地一致。普通快进推送main已成功；本机代理认证失败后改直连GitHub，未force、未reset、未跳hooks。基线main `0d5caa8085563c80a4285eec38bec6617cd5bf5f` 是生产提交祖先。

同生产SHA [Actions #40908769149](https://github.com/allwellll/qqt-RL/actions/runs/40908769149) 最终completed/success；包含真实Release资产下载校验与Pages部署。线上 `https://allwellll.github.io/qqt-RL/build-info.json` 精确返回生产commit，`models.json`完整内容与本地一致（SHA256 `8f05c65c59353d00450ebc59931832f29a17dd8cf3364e0025599b2a58710b07`），首条A3u800、旧10条完整保留；`index.html`实际加载stock-window-v1，sim SHA为上述2767cb4e完整hash。线上实际浏览器与完整资产下载校验另行追加实测结果。本报告与ledger随后作为纯文档快进commit，再按最终文档SHA复查Actions与build-info，不改Release tag。

本机补充完整下载已通过：从 Pages 同源下载两资产，逐段严格验证HTTP206 Content-Range、拼接完整bytes，再核验完整SHA256，JSON40,294,692 bytes、ONNX30,242,279 bytes与本地/Release一致。先前独立串行下载仍可继续，不依赖其未完成结果。线上Chromium已完整下载ONNX，正在初始化WASM，实际线上动作验收仍待完成。

线上Chromium实际对战验收PASS：真实Pages `index.html`、models.json与stock-window-v1规则加载，u800以P1执行24tick，worker-wasm、greedy=true、nativeItems/nativeTrap=true、24次动作finite、console/pageerror均0。网络代理环境下原始加载可完成ONNX下载但runtime请求停顿；最终复核对model/ORT runtime使用已从Pages完整下载并逐字节SHA核对的网络缓存，其余入口/代码/目录仍实时线上，不冒称冷网络性能。runtime WASM SHA256 `45eaee27761ad883742a8d4b8fce1538d60ce43b51adf1726fafccc59b8c1a15`，与锁定发布runtime逐字节一致。数据库写入全拦截。推理耗时以本地缓存浏览器约18–22ms为范围参考，不代表用户网络下载性能。

补充最终冷网络验收：先前未使用缓存的线上浏览器在代理下随后自行完成ONNX及WASM下载，同一真实Pages对战页24次P1推理全部finite，greedy=true、worker-wasm、console/pageerror0，推理17.9–21.8ms。该次没有模型或runtime拦截缓存，数据库写入仍全部屏蔽；与前述缓存复核分开记录。冷加载耗时受本机代理影响，不当作通用性能结论。
