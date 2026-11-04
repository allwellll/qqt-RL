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
