# 网页模型 OSS 下载发布验收（20261106）

执行日期：2026-11-06 至 2026-11-07。基于最新 origin/main 的独立分支完成，只修改网页模型来源和对应测试。已用本机凭据；凭据不进入前端、仓库、报告或输出。不调用 CC，不修改训练及其他 CX 工作树，不 force。

## 模型来源与校验

20 个模型、35 个 JSON/ONNX 资产，共 1,221,843,513 字节，全部新增 OSS URL。原 Release URL、文件名、大小、SHA256、模型顺序与元数据保持原值，已与基线 manifest 深比较。对缺少哈希的原文件名，仅 OSS 对象键追加完整 SHA256；Release 文件名不变。

下载顺序为 OSS → 原 GitHub Release → Pages 同源 Release 镜像。每个来源均独立校验大小和 SHA256，任一失败自动尝试下一来源；全失败则明确报错。OSS 使用单次流式匿名 GET，保留百分比进度和超时。JSON 页面路径及 ONNX worker 均使用同一下载器。没有改变模型、温度、采样、规则、物理或推理节拍。

Pages 构建继续从原 GitHub Release 获取并校验模型，保留既有同源镜像。当前网络下真实浏览器无法成功访问 Release 跨域直链，因此额外保留这份同源兜底；不能把这项结果描述为“Release 直链浏览器下载成功”。

## OSS 权限与缓存

- bucket：`qqt-model`；endpoint：`oss-cn-beijing.aliyuncs.com`；公开模型主机：`https://qqt-model.oss-cn-beijing.aliyuncs.com`。
- bucket ACL 回读仍为 `private`；仅本次 `web-models/` 下的 35 个对象逐个设置 `public-read`。无关 PNG 的 `default` ACL 不变。
- 初次设置对象 ACL 被 bucket 的 BlockPublicAccess 拒绝。为允许授权的模型对象公开，关闭该 bucket 的 BlockPublicAccess；没有改账户级设置，也没有把 bucket ACL 设为 public-read。
- CORS origins 仅为 `https://allwellll.github.io`、`http://localhost:*`、`http://127.0.0.1:*`；methods 仅 GET、HEAD；允许头 `range`；暴露 Content-Length、ETag、Content-Range；max-age 3600。
- JSON：`application/json`；ONNX：`application/octet-stream`；全部为 `Cache-Control: public, max-age=31536000, immutable`；没有启用 gzip。
- 35 个对象均已 HEAD 核对大小、Content-Type、Cache-Control，再匿名重新下载核对大小和 SHA256。
- 无签名 curl：模型 200、bucket 列表 403、原无关 PNG 403。

## 测试与真实浏览器

`npm test` 全部通过；catalog、loader、worker 测试通过；`git diff --check` 通过。新增测试覆盖 OSS 健康时不访问后备、网络/403/大小/SHA 错误回退 Release、Release 故障再回退 Pages 镜像、来源顺序、进度，以及全部失败时不能标记 verified。

真实 Playwright Chromium 本地验收：默认 G_frozen ONNX（30,276,775 字节）从 OSS 加载约 1.589 秒，worker-wasm 后端，数值 parity 最大误差 0.000001430511474609375，至少 3 次真实推理，进度 100%，对局正常开局并出招，无 pageerror。三次推理约 16.6 / 16.2 / 16.1 ms。

拦截 OSS 请求后，真实尝试原 GitHub Release；该直链在当前网络返回 Failed to fetch，随后 Pages 同源镜像通过大小/SHA 校验，约 5.831 秒完成加载，至少 3 次真实推理，进度 100%，正常开局并出招，无 pageerror。两种场景截图已人工查看。

## 下载测速

本机通过环境代理访问外网；耗时仅供当前网络参考。真实 Chromium 中以公开 Pages 为 origin，禁用下载缓存，每来源各测 3 次，下载默认 G_frozen ONNX 并核对完整 SHA256，45 秒超时。

| 来源 | 第 1 次 | 第 2 次 | 第 3 次 | 成功下载中位数 |
| --- | --- | --- | --- | --- |
| OSS | 0.9157 s，200/SHA 通过 | 0.6955 s，200/SHA 通过 | 0.6770 s，200/SHA 通过 | 0.6955 s |
| 原 GitHub Release | 0.6981 s，TypeError | 0.4686 s，TypeError | 45.0053 s，超时 | 无，0/3 成功 |

补充 curl 原 Release 三次：15.2892 s 连接失败、4.6982 s 成功且 SHA 通过、15.6802 s 连接失败。不能据此声称三次成功中位数或可靠加速比。

## 发布与隐私验收

已完成推送前隐私扫描：`git ls-files -z` 范围及本次报告，私有标识/内网 IP/机器 PID 检查 0 项；两个 AK ID 前缀及四个完整凭据值额外扫描 0 命中。模型二进制、凭据文件、本地编排和日志均未提交。commit 使用中性身份，未含本机私有标识。

正常 push main 成功，代码 SHA：`b42f8d12bf711977940ad642dfc901fac1e34394`。发布前已 fetch 最新 origin/main 并确认无新增提交，主工作树无发布锁，未修改另一 CX 的文件。

Actions：[42251660818](https://github.com/allwellll/qqt-RL/actions/runs/42251660818)，`head_sha` 与上述 main SHA 完全一致，最终 `completed / success`；`test-and-build` 和 `deploy` 均 success。

线上 [GitHub Pages](https://allwellll.github.io/qqt-RL/) 无缓存回读 `build-info.json`，`commit` 为上述同一 SHA。`models.json`、`model_catalog.js`、`model_loader.js`、`model_inference_worker.js`、`app.js` 与本次提交逐字节一致；线上 35 个资产均指向 OSS，且保留原 Release URL。线上真实 Chromium：首轮 OSS 场景加载成功；故障回退场景已自动尝试 OSS、原 Release（Failed to fetch）并进入 Pages 同源镜像，但代理网络下 180 秒验收时限到达时仅下载 63%，未宣称该轮通过。仅延长本地验收脚本等待上限至 600 秒后重测；生产代码未改。重测最终两场景均 PASS：OSS 加载 53.388 秒，拦截 OSS 后 12.462 秒经 Release 失败再由 Pages 镜像完成加载。两场景均通过大小和 SHA 校验、worker-wasm 推理、数值 parity（最大误差 0.000001430511474609375）、至少三次真实推理、进度 100%，正常人机开局并出招，无 pageerror；线上两张截图已人工查看。该代理环境波动显著，这些页面加载耗时不等于前述下载测速中位数。

交付报告保留在本地 `reports/cx_model_oss_20261106.md`；代码发布 SHA 如上，最终验收回填没有另行触发一次部署。

用户在聊天里贴过明文 key，建议之后在阿里云控制台轮换。

## 完整对象清单

以下全部为 `public-read`，共 35 个；SHA256 与 manifest 和本地资产完全一致。

| 对象键 | 字节 | SHA256 |
| --- | ---: | --- |
| `web-models/bun392-g-frozen-b0601e29ca06556d8fb561a1baa43dd6b723a8d6996fa72c879a3d1e31af24ce.json` | 40340827 | `b0601e29ca06556d8fb561a1baa43dd6b723a8d6996fa72c879a3d1e31af24ce` |
| `web-models/bun392-g-frozen-0cca2473fb0b6af900d523a04fb65526d22f289486c4ec31fb9875cee671f480.onnx` | 30276775 | `0cca2473fb0b6af900d523a04fb65526d22f289486c4ec31fb9875cee671f480` |
| `web-models/bun392-v2-risk-u9809-6365c568b594d0dfd3c83f74d3a14afb4d9512d519579f0492d4781f1e6a2ddd.json` | 40381088 | `6365c568b594d0dfd3c83f74d3a14afb4d9512d519579f0492d4781f1e6a2ddd` |
| `web-models/bun392-v2-risk-u9809-3127be3f37f24e8ea81fafd9c8f02ea79a3ed9ec0dbe84d570fe3a43a33e0673.onnx` | 30306583 | `3127be3f37f24e8ea81fafd9c8f02ea79a3ed9ec0dbe84d570fe3a43a33e0673` |
| `web-models/bun392-v2-risk-u9969-a68f3fcc71d07e9da922c22b9187ad21f4c37279f734a1e18d5c03e73950f278.json` | 40381088 | `a68f3fcc71d07e9da922c22b9187ad21f4c37279f734a1e18d5c03e73950f278` |
| `web-models/bun392-v2-risk-u9969-0d92d0a93303bf33310d7be298d2267e81223908ef3418ab5eb482f28935d49f.onnx` | 30306583 | `0d92d0a93303bf33310d7be298d2267e81223908ef3418ab5eb482f28935d49f` |
| `web-models/bun392-v2-r4-k1-u5282-88241ba1cd7a14756d6be51a29d564a071df578c18489a750b4cafdf42daeec2.json` | 40340752 | `88241ba1cd7a14756d6be51a29d564a071df578c18489a750b4cafdf42daeec2` |
| `web-models/bun392-v2-r4-k1-u5282-2491959a098051bdecd97aebf00f00168923d0a47b79dfe3b11db9d2d03bebe6.onnx` | 30276775 | `2491959a098051bdecd97aebf00f00168923d0a47b79dfe3b11db9d2d03bebe6` |
| `web-models/bun392-v2-r3-a-u4002-u4002-763ae64a325215de60fa4af403f4e03aeb27ad888bd859d54a097144e816c0a7.json` | 40340774 | `763ae64a325215de60fa4af403f4e03aeb27ad888bd859d54a097144e816c0a7` |
| `web-models/bun392-v2-r3-a-u4002-u4002-f11461f32ba29e3aed09c280e9cf66f974f22cce5b9cfb157839a0e3fad59656.onnx` | 30276775 | `f11461f32ba29e3aed09c280e9cf66f974f22cce5b9cfb157839a0e3fad59656` |
| `web-models/bun392-v2-r3-a-u3040-u3040-2a45af6ab8231d71e5104626ef58d08b0590385b1d0b9238fc73bf3b178590eb.json` | 40340766 | `2a45af6ab8231d71e5104626ef58d08b0590385b1d0b9238fc73bf3b178590eb` |
| `web-models/bun392-v2-r3-a-u3040-u3040-d9ef89f4b6454eae02395b0884079e04d77258f1827e2c52103a11e2af5afbd2.onnx` | 30276775 | `d9ef89f4b6454eae02395b0884079e04d77258f1827e2c52103a11e2af5afbd2` |
| `web-models/bun392-r1-slot03-u2760-cbd1069b6456b14ce577156a006715c89a5a1742d84871a7e00137e1c75a5b40.json` | 40294802 | `cbd1069b6456b14ce577156a006715c89a5a1742d84871a7e00137e1c75a5b40` |
| `web-models/bun392-r1-slot03-u2760-d724280ee7052bdb04d4f59fe9b9bba541e682901cee3324db19c717a0c615b1.onnx` | 30242279 | `d724280ee7052bdb04d4f59fe9b9bba541e682901cee3324db19c717a0c615b1` |
| `web-models/bun392-a3sr-slot00-u2400-887079d10fc7b99647c847e40a39fb6a8240ddbccdad797722b326362a201732.json` | 40294794 | `887079d10fc7b99647c847e40a39fb6a8240ddbccdad797722b326362a201732` |
| `web-models/bun392-a3sr-slot00-u2400-c0145d534bc57ae1f07e068d440106c1de25345da0939ee7126bb5a14726910f.onnx` | 30242279 | `c0145d534bc57ae1f07e068d440106c1de25345da0939ee7126bb5a14726910f` |
| `web-models/bun392-a3sr-u1280-736bc0c2d458682d771879be0cca80e25b90779336440beb40cbd800de0b7fed.json` | 40294780 | `736bc0c2d458682d771879be0cca80e25b90779336440beb40cbd800de0b7fed` |
| `web-models/bun392-a3sr-u1280-f3ab40a46f935796b50185a63f5b43ed28ee66abd67c57d1d23a9b6af592abe1.onnx` | 30242279 | `f3ab40a46f935796b50185a63f5b43ed28ee66abd67c57d1d23a9b6af592abe1` |
| `web-models/bun392-a3-u800-455d40c702ff444e3c8cd5c01275c67c7dbe31817ec690986a67d576bafca928.json` | 40294692 | `455d40c702ff444e3c8cd5c01275c67c7dbe31817ec690986a67d576bafca928` |
| `web-models/bun392-a3-u800-d4d718999397d3c2a56201f09705f9ab542602a9ade5bfc7b08f4579b0ee182d.onnx` | 30242279 | `d4d718999397d3c2a56201f09705f9ab542602a9ade5bfc7b08f4579b0ee182d` |
| `web-models/bun392-pbt-slot05-u720-ed52b79d7a1cc138a1458e275364dff0369e6510c928c09461d4e46b6fe82088.json` | 40294669 | `ed52b79d7a1cc138a1458e275364dff0369e6510c928c09461d4e46b6fe82088` |
| `web-models/bun392-pbt-slot05-u720-2380ddc1c44afc3ad75831bbfdc6bd201a7257db18687d107cdeb2ee591cffa7.onnx` | 30242279 | `2380ddc1c44afc3ad75831bbfdc6bd201a7257db18687d107cdeb2ee591cffa7` |
| `web-models/bun392-je10-v38-r01-g8-19b98f2192a237e994ffd29801de7934dfe6168de3507530b19b3c7dd4d5ca45.json` | 40296711 | `19b98f2192a237e994ffd29801de7934dfe6168de3507530b19b3c7dd4d5ca45` |
| `web-models/bun392-je10-v38-r01-g8-b1da0c13915e1093d18b07d1f059f579b68a7f47c8469c3b49e378c2c117dc16.onnx` | 30242279 | `b1da0c13915e1093d18b07d1f059f579b68a7f47c8469c3b49e378c2c117dc16` |
| `web-models/bun392-je09-stage-c-g8-raw-b80373c1a34cceb6eaeb549074c4fa154313b7b31272b42698f4430e2c7c50d2.json` | 40294827 | `b80373c1a34cceb6eaeb549074c4fa154313b7b31272b42698f4430e2c7c50d2` |
| `web-models/bun392-je09-stage-c-g8-raw-b606294fc8bc1e137c55db7600e4a1eb0284b03a1721279869978ba3bf82764d.onnx` | 30242279 | `b606294fc8bc1e137c55db7600e4a1eb0284b03a1721279869978ba3bf82764d` |
| `web-models/bun392-transport1536-raw-2ec354adfd72c65f0b5b601b40a13d7f879729350572897c660cbc2866710dc0.json` | 40294842 | `2ec354adfd72c65f0b5b601b40a13d7f879729350572897c660cbc2866710dc0` |
| `web-models/bun392-transport1536-raw-d5c8eb84936c18610d50f7044a13ec7d2ad480d5024b96ee75f82e1441b7b60a.onnx` | 30242279 | `d5c8eb84936c18610d50f7044a13ec7d2ad480d5024b96ee75f82e1441b7b60a` |
| `web-models/bun392-selfplay3072-raw-c100cc13aa01315b9263b300448c18a5d9a534af07438ae6b86e07ca7a02e94a.json` | 40294795 | `c100cc13aa01315b9263b300448c18a5d9a534af07438ae6b86e07ca7a02e94a` |
| `web-models/bun392-selfplay3072-7618ec5f6c23b2e2f0c9f45d086a6ca1429e73808aab3276c8b5cdbf29af5639.onnx` | 30242279 | `7618ec5f6c23b2e2f0c9f45d086a6ca1429e73808aab3276c8b5cdbf29af5639` |
| `web-models/bun-tf-it4000-830cdf89747123a9ba1e112ccc43cce55adce0f57d203dab574c84b76905f084.json` | 9907406 | `830cdf89747123a9ba1e112ccc43cce55adce0f57d203dab574c84b76905f084` |
| `web-models/bun-kill-mlp4-1de4cd682680966e46fe0d11eb2075a5d1f5ca3e5512499031af8572d40a5c70.json` | 32372445 | `1de4cd682680966e46fe0d11eb2075a5d1f5ca3e5512499031af8572d40a5c70` |
| `web-models/safe-high-c16-cabd0dad115cd888dec4f715d999f862296ef4ce3dc1e6bfad7c0aea70d37aa1.json` | 40294224 | `cabd0dad115cd888dec4f715d999f862296ef4ce3dc1e6bfad7c0aea70d37aa1` |
| `web-models/safe-medium-c16-9908caa24d39fab7d34ab0047754c22a78ae2be267142290fbaad41a09bfe3f8.json` | 40294230 | `9908caa24d39fab7d34ab0047754c22a78ae2be267142290fbaad41a09bfe3f8` |
| `web-models/safe-high-c24-9ba571c45791aa01dc03ddc8b1a5ae7916be1f5f24e30dec86bdc0c4f8700b5b.json` | 40294224 | `9ba571c45791aa01dc03ddc8b1a5ae7916be1f5f24e30dec86bdc0c4f8700b5b` |
