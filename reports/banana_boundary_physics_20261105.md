# 香蕉边界物理修复（20261105）

用户已明确授权修改实际香蕉物理与必要wrapper，正常测试发布；原确认阻塞解除。网页为唯一发布owner，模型与穿泡图案保留；原训练区只读，不操作GPU/训练/未知进程。

## 根因与最小修复

480px/s高速香蕉使 `_tryMove/resolveAxis` 探针返回界外位置，实际step夹回地图边界，但滑行结束判断与legalMask仍读取未夹紧探针。因此原右下fixture位置 `(12.5,14.64)` 不再变化，滑行状态2/向右锁永久存在。四向均复现；不把受控fixture冒充用户原自然录像。

修复仅在 `_tryMove` 返回前使用原有 `[radius,H-radius] × [radius,W-radius]` 夹紧；新增只读 `PHYSICS_REVISION=banana-boundary-clamp-v1`。step提交位置、动作掩码与清锁依据相同。未调速度、半径、方向、香蕉触发、泡权限、奖励或模型权重。native逐帧人类路径原本即可停滑，本次纳入回归。

|身份|旧 SHA256|新 SHA256|
|---|---|---|
|base sim.js|7d2047b7db2025608b38afec0f8283625b6f98a3d09ac49693f2a0a44d1d485b|674ff0f6cde9123a158ee888b18be5e372d5929709e2d3025b4e8d5823737fda|
|bun-native-v2|f6e346f3022640556efc2a988eb3b9b8e90abd5776af6ca1b4b1b0ee2acd6abc|593b50e8278b5e544132f86dc4011c4dc06d0de1166c8ff8e751d9ac4f553154|

机器可读身份：`web/physics-version.json`。

## JS/JAX语义与parity

当前 `jax_bomb/je05_core_env_v2.py::_move_player` 已在返回前clip；`je07_bun_env_v2.py` 用实际movement phase位移清除滑行。网页改为相同边界坐标语义。JAX native默认香蕉268px/s，网页480px/s，速度差异不变；JAX逐格扫前缘与JS首尾前缘、float32与JS双精度也仍有差异，不宣称全引擎等价。

CPU-only 240个共享移动探针：四向、六档速度、五个边缘距离、相邻阻挡。232个误差小于2e-5；144个直接贴边无障碍探针最大误差3.433227533378158e-7。8个旧有浮点/格线案例差异最大0.19990020751952997，原JS经同样clip与新JS一致，已完整记录并断言非本次新增。未修这些既有差异。回执 `runs/banana_physics_20261105/PARITY.json` / `PARITY_DIFFERENCES.json`，范围限定为坐标探针，不当作训练与网页整体规则一致。

已通过Herdr向主CX `cx-v37-kill-route-learning` 通知新旧SHA、精确改动、默认速度差异与训练迁移边界。多区当前1280与下一320继续使用旧 `PHYSICS_VERSION.json`，未来迁移另版本且完整保存；本次不热改训练。

## 历史录像与模型行为

- 原native字节封存 `web/review-sim/bun-native-v2-20261104.js`，SHA仍为f6e346…；旧base已封存 `stock-legacy-20261028.js`，SHA仍为7d2047…。
- Review协议与浏览器按录像SHA分派模拟器；五槽Review不再硬编码当前引擎；旧八局manifest、完整动作、状态检查点和终局均保持原样。
- 默认目录u9809、可选u9969与K1、19项模型manifest及模型资产均不变。历史成绩仍属原规则，不移植成新物理性能结论。
- 物理清锁只保证可响应反向合法输入。806受控fixture中u9809清锁后仍可选择合法idle（上0.1461、左0.3685、idle0.4854），属于策略偏好。物理发布不等待长训，策略层另作明确版本与有界验收；不能把外置恢复冒充模型能力。

## 验证与发布收据

96个base/native×双席×四向×角落×268/480/600速度用例，真实香蕉触发、反向输入恢复、墙/砖/泡阻挡、死亡/复活、有效动作精确复放，以及12条非边界普通轨迹与旧版逐帧一致通过。

最终完整npm、桌面/手机、全部历史Review、公开Pages同head、规则与归档SHA结果保存于 `runs/banana_physics_20261105/` 和 `.hermes/RECEIPT_banana_physics_20261105.json`。不为回填发布head再追加改变已验收提交的文档commit；每日最终Markdown收据也保存在该项目产物目录。


## 已部署物理版与独立恢复辅助

物理版 `7a750752aa39d6a466f806384ff66bb6dbe95997` 已正常快进 main，GitHub Actions `41867613354` 构建和 Pages 发布成功；公开 `build-info.json` 同 head，桌面1440×1000、手机390×844、规则/归档/模型清单 SHA、四向反向恢复、原穿泡提示、原八局完整Review与通用Review校验/实播通过。物理阶段独立收据 `runs/banana_physics_20261105/PHYSICS_RECEIPT.json`，原始验收备份为该目录 `PHYSICS_*`。已再次通知主CX正式部署与训练边界。

模型原始策略在右下806清障碍受控场景中，物理已清锁、上/左输入合法，仍偏好idle。实际网页加载u9809 WASM worker，关闭辅助45 tick仍在 `(12.5,14.64)`；开启辅助第12 tick原始idle被明确记录为实际向左，45 tick到 `(12.5,13.86)`。不是更换权重，也不是模型自主脱困性能声明。

独立辅助版本 `post-banana-boundary-recovery-v1`，入口仅真人对战、双席、native、显式五槽模型。必须先观察真实香蕉滑行，再在世界边界清锁；连续10 tick原始idle且没有位移或泡/道具请求才介入。清锁后50 tick窗口、最多4次覆盖、离锚点0.5格停止。只选原掩码合法且原模型概率最高的向内移动，用现有 `_steer` 预测实际路径，扫描完整身体扫过区域，拒绝危险图/余威、泡、墙砖、地面道具与宝箱，步长超过0.5格也拒绝；死亡、被困、出生保护、多席、重开及关闭开关不介入。没有放泡、用道具、传送或改物理。

网页直接展示“前端辅助，可关闭”和本局介入次数。`lastDecision`保留原模型概率，`lastDeploymentDecision`和带版本事件分别记录raw/effective；复放必须用实际有效动作。观战、K1、旧模型、旧Review及训练均不接入辅助。模型manifest和历史录像文件保持原字节。

验收：8个双席四向边界、17个拒绝场景、同tick幂等、4次硬上限、开关/重开、原决策保留、辅助不修改物理快照、有效动作精确复放；12个正常seed共960 tick与无辅助逐帧相同且零介入。真实u9809 Node和浏览器WASM推理均恢复并精确复放。实际手机网页选择原模型，启用/关闭两条真实对局输入路径均通过；回执 `RECOVERY_UNIT.json`、`RECOVERY_MODEL.json`、`RECOVERY_BROWSER.json`、`LOCAL_RECOVERY_APP.json`。不将这些有界fixture当作整体真人胜率评估。

仍有的实际差异：JAX/JS默认香蕉速度及8个浮点格线探针差异未迁移；模型自身idle偏好并未通过训练修正。辅助只解决上述入口的有限停滞，危险区、障碍区或窗口外可能继续idle；没有进行无界策略改写或长训。最终独立前端发布head、在线SHA和开关实测记录于最终项目收据。
