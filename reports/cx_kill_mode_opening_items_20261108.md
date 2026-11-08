# 纯击杀开局能力道具（20261108）

在开局/侧栏版本 `364c687f01469b53b999dbe3bc55432c6b86e429` 同SHA Pages及线上验收完成后顺序实施。不开新训练，不改模型，不动旧Review引擎。

## 精确定义

- 数量只有一个配置入口：`web/kill_opening_items.js` 的 `OPENING_ITEM_COUNT=24`。
- 仅真人网页新建的纯击杀对局，在原清砖reset之后应用；正常运包直接返回，不增加字段、不消耗RNG、不修改物品。
- 从原 `_itemSupplyCells()` 合法空格中筛选，排除双方出生格及其八邻格（切比雪夫距离≤1）；再取左右镜像格对（列c与14-c），同对同类型。
- 806原来双方各自独立随机挑出生点，部分组合的格距奇偶不同，使完全对称物品与相等最近距离不能同时满足。因此仅纯击杀开局保留P1原随机出生点，将P2置于地图已有的对应镜像出生点；无新增出生格。后续死亡的随机安全复活保持原逻辑，正常运包出生不变。
- 物品候选逐对检查双方到该对的最短可通行格距相等，再按本局RNG不放回采样12对。距离使用原生静态可通行墙/砖规则，物品放置合法性仍用原 `_itemSupplyCells()`，不会把可通行装饰遮挡错误当成阻路。
- 每对按泡数、威力、速度循环，实际12对/24件，三种各8件，镜像左右各12件。每一种物品的双方最近距离也相等，不只整体最近距离相等。
- 候选不足时只放可用整对并记录actual/shortage；不覆盖障碍、不强凑、不增殖。当前生产806检查32个seed均为24件，无缺额，同seed完全可复现。
- 使用已有crate/crateType/superCrate和原拾取链路，保持生产上限泡数10、威力8、速度2.4；未改编码函数、种类或上限。死亡8个额外能力掉落+原库存回收保持，未修改模拟器源文件。

## 最小验证

```sh
node web/test_kill_opening_items.js
node web/test_kill_mode.js
node web/test_self_respawn_preview.js
python3 scripts/audit_combat_training_path.py
PLAYWRIGHT_MODULE=<PLAYWRIGHT_MODULE> node scripts/verify_kill_opening_items_browser.js
```

单测覆盖32个确定性seed、左右同类对称、各类型数量、出生区排除、合法性、双方整体及分类型最近路径距离、重复调用不增殖、原生拾取增强/上限、正常运包snapshot完全不变、全图无格时缺额24、死亡仍掉8件。

真实Chromium桌面1280×900及手机390×900实际应用各一轮，首帧记录24个物品，三类各8个。测试专用代码仅把人类定位到真实物品格，随后执行原生step拾取：泡数2→3、威力2→3、速度1.3→1.457142857，物品消失；不是直接篡改能力。切到正常运包，以相同seed重建原模拟器对照，开局crate及crateType逐格一致。没有pageerror。

截图与收据在忽略目录 `.hermes/kill-opening-items-local-*.png` 和 `.hermes/kill-opening-items-local.json`。范围为真实Chromium及手机视口，未做真机iOS测试。

隐私扫描覆盖Git追踪文件和本次新增源码、测试、报告与浏览器脚本，零发现后正常commit/push main。等待同SHA Actions/Pages成功后，用相同浏览器脚本核对线上版本、数量、真实拾取和运包对照。最终收据在忽略目录并在消息回报，不为写入自身SHA重复发布。不force、不CC、无额外模型评测或资产SHA回读。
