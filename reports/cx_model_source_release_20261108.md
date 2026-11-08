# 模型下载改为Release镜像（20261108）

前序纯击杀开局道具版本 `a6c632e502b4fdb8d7209c0b9a2e7c5d18a2cd15` 已完成同SHA Actions/Pages成功与线上桌面、手机视口验收，再顺序实施本任务；其24件道具、各类8件、真实拾取增强和正常运包不变均通过。

## 改动

- `web/model_loader.js` 的来源固定为 `models/<file>` Pages同源镜像 → manifest中的Release直链。完全不读取 `oss_url` 作为下载来源；历史OSS字段和资产保留，旧Review不变。
- 同源镜像使用可缓存的流式下载，进度以manifest大小为总量；每个来源仍须通过原大小及SHA-256校验。Release保留既有分段下载/重试，受CORS限制时仍可能不可用，失败会明确显示错误和重试。
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
