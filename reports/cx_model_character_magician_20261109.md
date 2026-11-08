# 模型P1礼帽魔术师（20261109）

按 `TASK_model_character_magician_20261108.md` 实施；前序Release镜像版本 `f5868656e067486147f1cef1aa7c8e3ab0288138` 已发布验收完成后顺序执行。

## 改动与来源

- 原版 #55 礼帽魔术师来自 kuuhaku1314/qqtang v1.3.2 Release 的QQTang-Local.zip，角色配置为 `object/player/Player55.ini`，body=15501、leg=1。
- 扩展 `scripts/export_client_actor_sprites.py` 支持 `--roles` 和 `--actors-only`，配置文件按大小写无关定位；修复单方向共享层的四方向复用，保留原图层顺序、裁切和脚底锚点。
- 新增 `web/assets/native/magician_stand.png`、`magician_walk.png`、`magician_portrait.png` 及sprites.json角色身份。站立/行走均为85×85帧、4方向、6帧（站立帧包含共享层序列），未改现有毛毛/皮皮/通用效果资产。
- 仅实战网页motion.characters把模型座位pid1设为magician；本地真人pid0继续使用原人物选择与本地保存设置。新增实战teamRings标记，沿用原阵营光圈及对手名牌，真人箭头保留。
- 无显式角色的旧Review/录像继续原精灵；未改任何训练、物理、模型菜单、默认模型或下载来源。

导出命令：

```sh
python3 scripts/export_client_actor_sprites.py <QQTANG_CLIENT_ROOT> --roles 55 --actors-only
```

## 验证与上线凭证

```sh
node web/test_web_feel.js
node web/test_native_restoration.js
node web/test_visual_parity.js
node web/test_http.js
python3 scripts/audit_combat_training_path.py
MAGICIAN_FIXTURE_ONLY=1 PLAYWRIGHT_MODULE=<PLAYWRIGHT_MODULE> node scripts/verify_model_magician_browser.js
```

以上通过。PNG尺寸与清单一致；渲染回归覆盖模型四方向行走/松键站立、死亡隐藏/复活重现、旧回放无显式皮肤时仍用旧模型精灵。

本地真实Chromium使用真实导出的PNG与实际Canvas渲染器、当前v4原生模拟器，在纯击杀和正常运包两种受控场景检查每个方向的实际drawImage帧；困泡画通用trap，原生trap到期触发死亡pop，原生复活流程重新画出魔术师。真人毛毛/皮皮均可渲染。该前置检查只验证素材和受控生命周期，不冒称完整模型开局。

正常commit/push main并通过同SHA Actions/Pages后，运行不带fixture-only的同一浏览器脚本，实际默认模型加载并开局，切换两模式核查实战motion角色及阵营标记，切真人皮皮确认模型仍为magician，回读build-info SHA。线上截图路径为 `.hermes/model-magician-public-desktop.png`、`.hermes/model-magician-public-mobile.png`，收据 `.hermes/model-magician-public.json` 与 `.hermes/model-magician-published.json`。手机范围为Chromium手机视口截图，未声称实体iOS实测；四方向与困泡/死亡/复活为真实Canvas受控场景检查。

隐私扫描仅覆盖Git追踪文件与本次新增素材、浏览器脚本及报告，禁止本机来源绝对路径入库；源根仅使用中性占位符。不删除资产或旧录像、不调用CC、不force、不启动训练。最终SHA与同SHA部署链接在交付消息回报，不为报告记录自身SHA额外循环部署。
