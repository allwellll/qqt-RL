# 自己的复活位置预告（20261108）

在加载进度版本成功上线后顺序实施，只有一个 writer 和顺序 push。

## 行为与实现

- 原纯击杀代码在30tick结束时才选随机点，`killRespawnPos`保存的是死亡位置；直接画该字段会预告错误位置。本次新增 `web/self_respawn_preview.js`，仅对网页新建的真人对局实例启用，死亡时即用该局的确定性RNG均匀选定合法安全点。
- 每个逻辑tick复查预定点；新泡射线、爆炸余威、人物占用或物品/障碍出现时，按相同候选规则重选并转移标记。仍安全则不重选、不额外消耗RNG。复活时直接使用已选点，保留原倒计时、属性回收、掉落和重生保护。
- 候选沿用原 `_itemSupplyCells()`，排除墙、砖、笼/遮挡、物品、泡、余威、落物、玩家等不可用格；再使用原 `_rays()`排除当前所有泡的原生射线覆盖，不以引信长短放宽。候选不足为零时隐藏预告并延后复活，逐tick重试，绝不强行在危险格出现。
- 正常运包使用同样的安全复查：原固定出生点可用时保留，不可用时重选；原复活时间和保护保持。
- 渲染器要求实例显式启用且 `humanPid >= 0`，只返回该真人一条预告。淡色本方光圈和“自己 · N秒复活”在压暗层之后绘制，限定在棋盘内部，不覆盖道具HUD。复活后立即消失。对方坐标不进入状态面板或显示文字。
- 观战/旧Review未安装适配器且humanPid为-1，不显示任一方预告。原模拟器文件和按SHA选择的旧Review引擎均未修改，避免替换历史录像物理。模型的观察编码函数不变；新字段也没有加入编码或录像快照。

## 最小验证

通过：

```sh
node web/test_self_respawn_preview.js
node web/test_kill_mode.js
node web/test_death_overlay.js
node web/test_web_feel.js
node web/test_visual_parity.js
node web/test_review_source_hash_compatibility.js
python3 scripts/audit_combat_training_path.py
PLAYWRIGHT_MODULE=<PLAYWRIGHT_MODULE> node scripts/verify_self_respawn_browser.js
```

单测覆盖固定seed、死亡即选点、稳定点不重采样、30tick精确到达、泡射线/余威/玩家占用重选、全图无合法格时延期与恢复、正常运包复活、R3及五槽输入不含预告坐标、未启用的历史引擎不显示。加入npm测试链。

真实Chromium桌面1280×900和手机390×900各一次，加载线上默认模型后注入双方死亡：仅出现自己标记，初始预告为[12.5,0.5]；注入该格爆炸后转到[1.5,7.5]，第31tick实际位置精确为[1.5,7.5]，预告消失。两种视口均无pageerror；状态面板未出现双方预告坐标。标记在棋盘内、HUD之外。截图与实际位置收据存于忽略目录 `.hermes/self-respawn-local-*.png`、`.hermes/self-respawn-local.json`。

范围为真实Chromium和手机视口，未做真机iOS验证。没有模型额外评测、资产SHA回读或训练。最小本地检查通过后执行隐私扫描，零发现才提交；正常push main，等同SHA Actions/Pages成功，并在线上复用脚本核对版本、显示及实际复活点。最终发布收据保存在忽略目录并在消息回报，避免为回填自身SHA重复发布。
