# D 网页模型发布 — 2026-10-28

## 变更与资产

- 基线：`3710c6db7a33d300d7ec849261b01866bbfa10fa`；只新增网页默认D，旧9条模型对象原样保留。历史JE10测试改为按id定位；Review默认及录像不变。
- tag：`web-bun392-pbt-slot05-u720-20261028`；[Release](https://github.com/allwellll/qqt-RL/releases/tag/web-bun392-pbt-slot05-u720-20261028)。
- 来源：`runs/iter_pbt_v1/formal/slot05_c000005/u000720`，policy SHA256 `ed5a2b1860e8256fa34ac5a5830ff9b5c81e67a0b33d5c8a73101b3149796663`，COMMITTED SHA256 `10f87427e8cd6bdf199f58a40e9874bfacdd3db8e89606859b9fe9e1364c19b0`。
- 原JSON SHA256 `ab6fa204a1c867081c71950de3ff472701bd8351c89fc12e385449c346d7266a`；在本worktree副本补齐strategy、COMMITTED hash、state_dim/heads/update、来源引用及显示名，flat/tensors完全未改。ONNX由同一已验policy用现行CPU exporter导出，故与历史导出文件hash不同。未重新训练或生成录像。

- JSON: [bun392-pbt-slot05-u720-ed52b79d7a1cc138a1458e275364dff0369e6510c928c09461d4e46b6fe82088.json](https://github.com/allwellll/qqt-RL/releases/download/web-bun392-pbt-slot05-u720-20261028/bun392-pbt-slot05-u720-ed52b79d7a1cc138a1458e275364dff0369e6510c928c09461d4e46b6fe82088.json)；40294669 bytes；SHA256 `ed52b79d7a1cc138a1458e275364dff0369e6510c928c09461d4e46b6fe82088`。
- ONNX: [bun392-pbt-slot05-u720-2380ddc1c44afc3ad75831bbfdc6bd201a7257db18687d107cdeb2ee591cffa7.onnx](https://github.com/allwellll/qqt-RL/releases/download/web-bun392-pbt-slot05-u720-20261028/bun392-pbt-slot05-u720-2380ddc1c44afc3ad75831bbfdc6bd201a7257db18687d107cdeb2ee591cffa7.onnx)；30242279 bytes；SHA256 `2380ddc1c44afc3ad75831bbfdc6bd201a7257db18687d107cdeb2ee591cffa7`。

## 验证命令与结果

以下命令中 `<CHECKPOINT>` 为只读来源，`<JSON>` / `<ONNX>` 为上述本地资产，`<FIXTURES>` / `<RESULT>` 为本worktree忽略目录内输出。

- `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <PYTHON> scripts/je03_native_vit_export_20261011.py <CHECKPOINT>/policy.pt <ONNX>`：成功，CPU导出。
- `CUDA_VISIBLE_DEVICES= QQT_ONNX_RUNTIME=<LOCAL_CPU_RUNTIME> node scripts/verify_bun_release.js <JSON> <ONNX> <FIXTURES>`：PASS，8个双座位样本。
- `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <PYTHON> scripts/verify_bun_release.py --checkpoint <CHECKPOINT> --model <JSON> --fixtures <FIXTURES> --out <RESULT>`：PASS，75张量完全一致；Python fp32对JS/ONNX最大误差7.94e-6，掩码greedy一致。bf16仅作诊断；历史57样本中3次greedy差异，不能称训练/网页严格等价。
- `CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=<ISOLATED_BROWSER_LIBS> node .hermes/publish-best-d/browser.js`：PASS，真实Chromium+ORT WASM，8样本最大误差5.51e-6。直接启动缺libnspr4.so，使用本机现有隔离库解决。此检查覆盖浏览器实际推理，未覆盖完整网页口令交互对战；未提供网页设置口令。
- `npm test`：PASS（含内嵌test:review）。
- `npm run test:review`：PASS；其可选外部Review模型检查因本地模型未提供而按原契约SKIP，历史Actor证据与录像hash检查通过；本次D资产单独完成精确张量校验。
- `PRIVATE_DENY_TOKENS=<LOCAL_IDENTIFIER> python3 scripts/check_publish_privacy.py --include <REPORT> --include <RELEASE_MANIFEST> --include <RELEASE_BODY> --include <JSON>`：PASS，0 findings；范围为git已追踪文件、本次报告及显式发布文本/JSON，排除未追踪run/checkpoint/编排。ONNX二进制另查本机标识为0。
- `git ls-files -z`加待提交报告的大小检查：最大7,958,608 bytes，文件 `web/review-replays/h8-v36-g32-tactical_v2-2026101786.json`，低于10MB。
- `git diff --check`：PASS；原9条目录对象完全相同，原D导出的flat/tensors完全相同。
- catalog的score明确为JS safe_win合并比例77/192，不冒称PBT选择分或与旧协议分数可比。

## 发布与线上验收

- Release已公开，发布时间 `2026-10-28T15:31:02Z`；tag指向生产提交 `da6a11845b303d1e974584274ce486200eaf495b`。
- 生产提交 `da6a11845b303d1e974584274ce486200eaf495b` 已通过指定命令快进push到main；首次代理CONNECT 503，第二次成功。未force、未跳hooks。
- 同SHA [Actions #40716694289](https://github.com/allwellll/qqt-RL/actions/runs/40716694289) 最终 `completed/success`，`test-and-build`及`deploy`均success，结束时间2026-10-28T15:34:23Z。
- `git ls-remote origin refs/heads/main refs/tags/web-bun392-pbt-slot05-u720-20261028`：两者均匹配上述生产提交；基线origin/main是其祖先，推送为快进。
- 线上 `https://allwellll.github.io/qqt-RL/build-info.json` 精确匹配生产提交；`models.json`完整内容与本地一致，共10条，第一条为D，旧9条保持原样。
- Release完整下载SHA验收：同生产SHA的Actions `test-and-build` → `Assemble static site` 最终success。该步骤执行 `python3 scripts/fetch_web_models.py --manifest web/models.json --output _site/models`，实际下载所有Release JSON/ONNX，`download_models()`逐个校验bytes及完整SHA256；workflow随后再次逐文件校验，覆盖本次D两资产。公开资产HEAD实测HTTP 200；GitHub返回两资产digest也与上列SHA一致。因此“可下载、完整内容SHA正确”由同SHA成功CI证明。
- 本机附加下载复核单独披露：公开URL 256KiB Range探测HTTP206成功；8路1MiB分段复核遭代理CONNECT 503，未完成、不记PASS。另有先前启动的只读串行下载仍在缓慢传输；遵守不杀进程，不终止它，不依赖其未完成结果。它只可能在本worktree忽略目录保存下载收据，不会修改发布内容。
- 本报告与ledger实际发布时间作为后续验收记录提交；Release tag保持指向生产提交。该后续提交仅含文档，将再次快进push并按其HEAD复查Actions及线上build-info；最终交付消息提供该提交及workflow结果，避免在文档中预写尚未发生的结论。

## 边界

未改来源checkpoint目录、任何runs或Review worktree；未占GPU、未终止进程。大模型仅放Release，不入Git。

验证环境说明：本worktree预置的 `web/vendor` 是共享vendor缓存链接；`npm test` 的 `build:runtime` 经该链接重写了5个运行时资产（3个ORT文件、2个许可证文件）。已逐字节确认它们与锁定的onnxruntime-web 1.23.2及仓库许可证副本一致；共享缓存目标没有Git追踪的vendor文件。没有改动其Git追踪源码。这项共享缓存写入应与来源checkpoint/runs只读边界分别记录。
