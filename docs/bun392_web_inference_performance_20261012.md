# Bun392 网页推理性能修复（20261012）

根因已复现：原网页在主线程运行纯 JS 标量 Transformer，392 模型推理超过100ms；`setInterval(tick, 100)` 的重入锁丢掉正在推理期间的tick，且同步计算阻塞rAF。保持模型、状态编码、greedy、动作及100ms物理tick，改为专用 Worker 内 ONNX Runtime Web 1.23.2 单线程 SIMD WASM。模型对应既有 g3072 raw Release，不改为EMA，不重新训练或改导出权重。

| 真实 Chromium 检查 | 结果 |
| --- | --- |
| 旧192纯JS，16状态 p95 | 45.9ms |
| 392纯JS，16状态 p95 | 183.4ms |
| 392 Worker/WASM，48次 p95 / median | 27.8 / 14.3ms |
| logits/value 最大绝对误差 | 2.3222e-5 |
| 原生24平面/global24，16状态 greedy差异 | 0 |
| 2个固定seed×16tick完整轨迹动作/编码差异 | 0 |
| 装配站，人类对模型10秒 | 100次调用 / 100完成 / 100逻辑tick |
| 装配站，模型对规则10秒 | 100次调用 / 100完成 / 100逻辑tick |
| 两模式推理 p95 | 14.8 / 36.0ms |
| 两模式 rAF p95 | 16.8 / 16.7ms |
| 两模式稳定推理窗口主线程longtask | 无 |

上述性能测试与JE04 GPU训练及CPU评估同时运行，浏览器使用独立CPU额度。测量会受机器负载影响，不保证所有终端都达到相同耗时。原生Bun物理及观战模式原有道具开关保持原样；此前网页导出JS/ONNX一致性最大误差5.72e-6见发布报告，本轮新增WASM和完整轨迹验证，不声称全状态空间数学完全相等。

方案比较：仅将旧JS迁入Worker可以改善绘制，但不能解决其单次约180ms造成的10Hz损失；WASM同时解决阻塞和吞吐。测试浏览器WebGPU API可见但无可用adapter，故选择实测可靠WASM，未使用训练GPU。GitHub Pages无跨源隔离，因此numThreads=1、不依赖SharedArrayBuffer或COOP/COEP。既有ONNX为固定batch1；model-vs-rule只有一个模型，规则Bot无需神经推理，模型对模型按原seat/RNG顺序请求，不能直接将该资产冒充batch2。

下载JSON与ONNX都校验完整大小及SHA256；CI有界瞬态重试和原子写入，不跳过失败模型。Worker先下载校验ONNX，再初始化运行时；HTTP/大小/hash/非finite/数值不一致硬失败。运行时不支持WASM才允许同一已验证policy的worker-JS兼容回退，回退不算性能通过。专用Worker只算神经网络，sim.js编码、mask、动作解码、缓存和RNG均复用，不修改sim.js。

ONNX及JSON沿用 `web-bun392-selfplay3072-v1` Release；目录 `web/models.json` 声明ONNX大小和SHA。WASM依赖通过锁定npm完整性在CI装配同源资源，生成binary不入Git，包含许可证及第三方声明。相关故障测试涵盖下载HTTP500、hash错误、数值不一致、运行时回退、消息乱序、pending销毁，以及CI模型校验不可降级。

复现：`npm ci && npm test`；`python3 -m pytest -q tests/test_fetch_web_models.py tests/test_bun_release.py`（25项）；启动网页后 `WEB_URL=<WEB_URL> npm run benchmark:inference` 和 `WEB_URL=<WEB_URL> node scripts/verify_bun_inference_browser.js`，须安装Playwright Chromium。benchmark性能阈值只在真实浏览器执行，不拿静态代码断言替代性能测量。完整静态装配验证6个JSON+1个ONNX，复用已公开下载的cache时仍重新检查全部bytes/SHA，不跳过任何资产。

机器证据：`reports/je03_bun_stage2_20261012/bun392_web_inference_performance_20261012.json`。部署后按精确提交核验Actions和Pages终态，再验证公网build-info、目录、运行时字节、同一Release ONNX完整性与真实两模式10Hz。

公网首次实测暴露下载阶段的短超时：模型正在持续下载但当地连接约0.04MiB/s，初始化原3分钟上限会误终止健康下载。初始化上限改为有界20分钟，推理请求独立保留30秒超时；不把慢下载当推理成功，不改变SHA/大小或性能阈值。公网验收仍必须真实下载并开局，不能用mock绕过。

进一步减少初始化重复权重传输：manifest冻结与已校验JSON完全相同的原生meta及零输入参考logits/value，正常Worker仅下载并完整校验30.2MB ONNX；初始化仍比较参考数值，完整benchmark继续验证JSON/ONNX元数据逐字段一致及原生轨迹。只有运行时不可用的兼容路径才下载并校验40.3MB原JSON权重。CI依旧下载校验两种完整Release资产，未跳过模型校验，不改权重和动作。公开资产逐字节验收的大文件下载上限与初始化同样有界20分钟，避免45秒测量工具超时误判部署失败。
