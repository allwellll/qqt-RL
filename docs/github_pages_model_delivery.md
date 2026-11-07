# 当前网页模型发布规范（20261107）

本页替换早期Top-N设计草案的执行说明。当前为manifest固定身份、按需加载的可选目录；专项长处可单独上架，不要求全面胜出，默认更换仍需完整局证据。不会因为整理文档删除任何旧模型、run、checkpoint或录像。

## 资产与身份

- Git保存 `web/models.json`、`web/model_inference_defaults.json`、模型说明和公开证据，不提交JSON权重/ONNX或秘密文件。`web/models/`、runs与本地配置保持忽略。
- 独立核对可信checkpoint/COMMITTED/policy SHA、JSON/ONNX SHA及导出等价；新候选身份使用明确READY固定点，不能用后来更新替换同名资产。
- 内容寻址资产首先上传OSS `web-models/`，只有此前缀对象public-read；bucket private、CORS只允许GitHub Pages及本地预览。凭据仅本机加载，不入前端、报告或日志。
- 同字节资产保留GitHub Release回退，CI根据manifest逐资产下载并验证大小与SHA后放入临时Pages `_site/models/`。没有权重Git分支或LFS历史。
- 浏览器实际链为 **OSS → GitHub Release → Pages同源镜像**；每个来源都必须先过SHA验证。ONNX/JSON具体字段见manifest与 `web/model_loader.js`，不另造第二份资产列表。

当前候选Release为 [web-items-specialists-20261107](https://github.com/allwellll/qqt-RL/releases/tag/web-items-specialists-20261107)，旧资产沿manifest各自release URL保留。本机代理Release匿名下载失败会如实记录；本次CI从Release实际下载并SHA通过，不以API元数据代替全部下载验证。

## 证据与选择

DUEL u019902 τ0.3、SP u023302 τ0.3为可选专项；默认G_frozen τ0.3保留。原v3生产JS选模856局、网页各对手每方式8局；当前v4额外针对快筛与真实浏览器复核分别展示。必须披露DUEL自死更多、两候选完整局静止偏高，不能把v3数字当v4成绩或把专项胜率当完整局默认替换依据。[完整交付](../reports/cx_web_specialists_items_v3_20261107.md)

历史Top-N、固定release名称、N=3/5、部署后自动清理/删除的草案均不是当前策略；旧代码只作历史复现，未重新授权不执行清理、历史重写或训练。

## 正常发布与验收

1. 基于最新origin/main隔离分支，核对owner的READY与review清单；一个writer，只整合已就绪改动，保留旧Review物理/资产。
2. 测试与隐私扫描覆盖 `git ls-files` 和待提交/明确公开产物；不扫描本地训练run/checkpoint，密钥不打印。独立验桌面/手机选择、OSS verified、真实worker推理、长短板与默认不变。
3. 正常push main，不force；`.github/workflows/pages.yml`执行 `npm test`、Review回归、静态站拼装、Release下载/SHA及Pages部署。实际触发为push main或workflow_dispatch，不声称未配置的repository_dispatch可用。
4. 按同一SHA等待test-and-build、deploy与workflow最终success，回读build-info.commit及关键文件字节；再在线上重复浏览器验收。
5. 发布回执写本地 `.hermes/WEB_CANDIDATE_PUBLISHED.json`，记录SHA/Actions/资产与浏览器证据，并回填报告；仅报告回填无需制造额外部署。

## 本地验收入口

安装Node依赖后 `npm run serve`；Playwright Chromium及其运行库可用时运行：

```bash
node scripts/verify_web_specialists_browser.js
node scripts/verify_mobile_controls_browser.js
node scripts/verify_facing_range_browser.js
```

分别设置 `SPECIALISTS_PUBLIC_URL`、`MOBILE_PUBLIC_URL`、`PASS_PUBLIC_URL` 为Pages根URL即可在线复验；`PLAYWRIGHT_MODULE`可指向本机Playwright包。Chromium手机触摸环境不等于实体iOS。导出器和浏览器脚本见 [有效入口索引](current_web_and_training_status_20261107.md)。不运行训练或旧自动晋级队列来完成网页交付。
