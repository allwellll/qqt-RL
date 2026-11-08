# 模型下载改为Release镜像（20261108）

前序纯击杀开局道具版本 `a6c632e502b4fdb8d7209c0b9a2e7c5d18a2cd15` 已完成同SHA Actions/Pages成功与线上桌面、手机视口验收，再顺序实施本任务；其24件道具、各类8件、真实拾取增强和正常运包不变均通过。

## 改动

- `web/model_loader.js` 的来源固定为 `models/<file>` Pages同源镜像 → manifest中的Release直链。完全不读取 `oss_url` 作为下载来源；历史OSS字段和资产保留，旧Review不变。
- 同源镜像和Release均沿用原分段下载/重试，进度以manifest大小为总量；每个来源仍须通过原大小及SHA-256校验。Release受CORS限制时仍可能不可用，失败会明确显示错误和重试。
- `web/model_catalog.js` 返回同源路径；遮罩显示“下载来源：Release 镜像”，失败时显示尝试Release直链，不再显示主线路/OSS。
- 四模型目录、SP默认、当前v4玩法、手机控制均保持。README和当前交付文档同步来源顺序，也纠正交付规范里过期的G_frozen默认描述。
- 不新增资产或额外回读Release。现有CI从Release下载四模型的JSON/ONNX八份文件，按manifest逐份校验大小和SHA后才上传Pages；此构建检查用于确认镜像一致性。

## 最小验证

```sh
node web/test_model_loader.js
node web/test_model_catalog.js
node web/test_model_worker.js
node web/test_web_feel.js
node web/test_http.js
python3 scripts/audit_combat_training_path.py
PLAYWRIGHT_MODULE=<PLAYWRIGHT_MODULE> RELEASE_MIRROR_STAGED=1 node scripts/verify_release_mirror_browser.js
```

以上通过。来源回归覆盖镜像正常、网络失败、HTTP失败、大小错误、SHA错误、Release兜底和全部来源失败；每种场景均禁止OSS请求。源码闭包仅更新网页app指纹，不运行训练。

真实Chromium桌面最小检查：以Pages资源配合本次待发布JS替换，默认SP实际取得Pages同源ONNX HTTP 200，经过verified/runtime/warming到首帧，遮罩来源正确，无OSS请求、无Release兜底请求、无pageerror。此为待发布源码检查，不能代替发布后同SHA线上检查；未声称实体iOS实测。

隐私扫描限定Git追踪文件及本次明确新增报告、浏览器脚本。正常提交push main后，等待同SHA测试/构建/Pages最终成功，再去掉JS替换运行同一脚本，核对build-info.commit、实际模型请求与开局。最终SHA、Actions和线上收据写忽略目录 `.hermes/release-mirror-published.json`，在交付消息回报；不为报告写入自身SHA循环发布。

## 发布后发现与修复

首次发布 `203f5c1264d56c6cc07ca4f3831c19561fc2857a` 的同SHA Actions/Pages成功，但最终线上浏览器三次未完成开局：Pages返回200，慢链路正文下载被单次请求的120秒总超时中止；带逐段进度诊断的一次到81%仍被截断，随后Release直链出现网络/CORS失败。发布前待发布JS检查虽成功，不能代替此结果。

修复是恢复Pages此前已有的分段下载机制，而非提高验证结论或改变模型。Pages与Release均调用同一 `downloadVerifiedAsset`，256KiB分段、最多8并发、每段120秒及原有重试；所有字节凑齐后仍严格大小/SHA校验。来源顺序和不请求OSS保持。再次执行相关测试、隐私扫描并正常追加提交，待同SHA Pages最终成功后做实际线上开局验收。

分段修复的待发布浏览器诊断已收到合法206和匹配Content-Range，但本机链路仍慢；180秒检查预算不足，未记录为完整开局通过。停止本任务的重复诊断以避免争抢带宽，最终线上最小检查允许10分钟等待，不修改生产物理或模型，不再重复预下载。
