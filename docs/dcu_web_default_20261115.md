# 网页对拼默认模型 · 20261115

本页保留 u840 首次发布记录。当前默认已按同日三点复核更新为 u930，
最新情况见 [收敛复核及最新发布说明](dcu_convergence_and_latest_20261115.md)。

用户要求：把当前较好的对战模型更新到网页，作为对拼默认。
入口：<https://allwellll.github.io/qqt-RL/>，选择“纯击杀”。模型 ID
`dcu-kill-u000840-tau03`，显示“对拼强化 · u840（对拼默认）”。
进入页面、切回纯击杀或保留纯击杀后刷新，都选 u840；正常运包使用原 SP。
旧模型仍在选择菜单。推理温度 0.3，合法动作内采样，非贪心。

## 选择依据与边界

u000840，220200960 环境步；固定 DUEL_latest/SP_latest/G_frozen 各 16 局，
8 个种子双席、每局 1800 tick，共 44 胜 1 平 3 负。对 DUEL_latest 为
12 胜 1 平 3 负；SP/G 也是训练对手，小样本不保证真人或长期胜率。
最后保存的 u001030 尚未同协议评测，不能默认当更强。
48 局录像：<https://allwellll.github.io/qqt-RL/dcu-replays/u000840-20261115.html>。
原始结果在 `runs/kill-eval-u000840-20261115-v2/RESULT.json`。

## 权重和网页推理

冻结 Actor SHA256：
`dcc148a17acea599555c468c4295c8578bd515ff584035397b9920d684bda7f5`。
训练输入 obs25/global38，导出仅去掉乘零的最后 mode 行，网页 obs25/global37、3 能力，
现有 R3 编码器与 v4-facing 模拟器。部署张量逐项与源 Actor FP32 精确相等，固定模式投影误差 0。
完整源 Actor FP32 / ONNX 随机输入最大误差约 1.63e-5；
16 个真实纯击杀双席状态加零输入，JS/ONNX 最大误差约 1.34e-5，合法贪心动作一致。
这些是 FP32 部署等价校验，训练评测采用 JAX BF16，不能宣称两者逐位一致或将
48 局 BF16 成绩直接当浏览器 FP32 的胜率。

JSON/ONNX 内容寻址资产在 Release：
<https://github.com/allwellll/qqt-RL/releases/tag/dcu-kill-u000840-20261115>。
Git 只保存 manifest、配置、代码和评测摘要，权重不入 Git。Pages CI 下载所有
Release 资产并验证 SHA/大小后部署；浏览器同源下载再次验证 SHA。
公开证据：`web/specialists/dcu-kill-u000840-20261115.json`。

## 复现与接手

不要更换系统或训练依赖。导出所需 CPU Torch/ONNX 位于单独 `.tools/web-export-env`。
以下仅 CPU 导出和验收，输出目录须为新的：

```bash
source scripts/dcu_env.sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' taskset -c 120-127 \
  .tools/web-export-env/bin/python -m scripts.export_dcu_kill_web_model \
  --source runs/kill-eval-u000840-20261115-v2 --output runs/dcu-web-new-export
.tools/node/bin/node scripts/check_dcu_kill_web_model.js \
  runs/dcu-web-new-export/model.json runs/dcu-web-new-export/samples.json
.tools/web-export-env/bin/python -m scripts.verify_web_specialist_onnx \
  --onnx runs/dcu-web-new-export/model.onnx --samples runs/dcu-web-new-export/samples.json \
  --output runs/dcu-web-new-export/PARITY.json
PLAYWRIGHT_BROWSERS_PATH="$PWD/.tools/replay-preview/browsers" \
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' taskset -c 120-127 \
  .tools/node/bin/node scripts/verify_dcu_kill_default_browser.js \
  runs/dcu-web-u000840-20261115-v3
```

本次回执目录 `runs/dcu-web-u000840-20261115-v3`：
`EXPORT.json`、`PARITY.json`、`RELEASE.json`、`LOCAL_BROWSER.json`、
`PAGES_PUBLISH.json`、`PUBLIC_BROWSER.json`、`npm-test.log`。
部署完成必须同时检查 Actions success、线上 build-info.commit、线上配置和文件 SHA、
桌面/手机真实 worker-wasm 推理。仅上传 Release 或提交 main 不算完成。
线上浏览器脚本设置 `DCU_DEFAULT_PUBLIC_URL=https://allwellll.github.io/qqt-RL/`；
本容器 Chromium 网络若受限，可设 `REPLAY_HTTP_BRIDGE=1`，由标准 HTTP 客户端
原样传递线上字节与安全头，仍使用真实部署文件和模型推理。排行榜非本次 E2E 范围。

## 分支边界

main 网页发布提交 `1989d4fbb86449d91833ac17b18231642c59a72a`，仅网页默认模型、
目录/配置/测试/公开证据与对应源码指纹。main 原有角色 magician、队伍标记和模拟器
保持原分支实现，未把训练分支的模拟器或训练入口覆盖到 main。
本地 dcu 的页面渲染版本不同，保留其实现；两个分支只共用本次模型默认选择改动。
公开 Pages 构建是最终 main 集成验收，本地桌面/手机结果单独保存。

浏览器验收的容器传输适配在 `scripts/dcu_public_http_bridge.py`：仅允许本仓库公开
Pages URL，保留线上响应类型/安全头与 Range，WASM 使用分块 HTTP 下载，所有网页
代码与模型来自线上，不替换为本地生产文件。缓存保存本次从相同公开 URL 取得的字节，
再次加载仍重跑浏览器模型 SHA 校验。它只用于验收，不进入生产网页。

## 最终验收

2026-11-15 12:14 UTC，Pages [Actions 43711444819](https://github.com/allwellll/qqt-RL/actions/runs/43711444819)
构建及部署均 success；线上 build-info 为 `1989d4fbb86449d91833ac17b18231642c59a72a`，
8 份发布网页文件逐字节大小与 SHA 均匹配。CI 校验五模型的全部十份 JSON/ONNX 资产。

线上 1280/390 宽度均完成真实 worker-wasm 推理；u840 的 WASM/JS 零输入误差
约 2.10e-6，温度 .3，纯击杀模式、至少 3 次推理、tick 正常推进；
切到运包加载 SP，切回纯击杀与刷新均恢复 u840，页面错误 0。
容器传输适配过程遇到首次 MIME 推断错误和整文件 WASM 下载超时，均在验收工具
修复后完整重跑，未因此更改生产模型或跳过线上验收。
汇总回执：[dcu_web_default_20261115.json](../reports/dcu_web_default_20261115.json)。
