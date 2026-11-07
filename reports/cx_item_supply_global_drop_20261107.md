# 道具供给与全图死亡掉落（20261107）

## 核实与规则定义

基线 main 为 `97c07bddd15074ec82817db4cc1f913bde8d87d6`。现行 face 规则在 `web/sim-bun-native-v2-face.js`：806 正常图 reset 后 81 块可炸砖、初始地面道具 0，砖内概率 0.582011，预期供给 47.142891 件；类型沿用随机泡/威/速及 tacticalItemFraction=0.3 的香蕉/胶，超级比例 0.090909。现行对拼场由 `web/tactical_arena.js` 设置 id=240、无砖、no_items=true、item_field=0，所以实际供给为 0。

现有 `_grow` 普通属性 +1 档、超级 +5 档并裁剪到地图上限；满属性溢出进入原墓地。生产 806 及当前对拼场的实际上限为泡数 10、威力 8、速度属性 2.4；没有增加上限，没有无上限能力。地图文件注释中的旧上限说明不替代运行时实际配置。

现有死亡 `_dropDeathInventory` 回收超过 loBombs/loBlast/loSpeed 的实际增益及道具栏实际数量，回退属性、清空道具栏。旧 `_releaseDeathDrops` 将候选格按尸体曼哈顿距离排序后选最近格，故确实集中在人物附近。被炸地面道具保留 type/isSuper/count 转入 graveyard，由既有鸟投回收，不凭空补发。

## 新版本：bun-native-v3-items

权威实现为 `web/sim-bun-native-v3-items.js`，独立复制冻结 face 版本后仅修改物品规则与新状态；旧 face/v2/stock 和 Review 引擎不改。网页与 JS 评估选择器 `scripts/native_eval_simulator_v1.js` 默认 current 指向新规则，也可显式 `QQT_EVAL_SIMULATOR=items`；保留 face/native-v2/legacy 的显式选择。

供给单位为一个可拾取道具格/砖内掉落，不将胶的三件礼包、超级成长档数或死亡堆叠数量误当作新增格数。精确公式：

- B = reset 后原地面道具格数 + 可炸砖数 × clamp(原砖掉率, 0, 1)。原砖掉率保持。
- B>0 时额外请求 ceil(1.5×B−B) 件；B=0 的可用道具场，明确使用参考供给 10，额外请求 15 件。对拼 id=240 在新引擎启用道具，其他显式禁用道具的地图仍禁用。
- 额外道具在 reset 时从全图合法空格无放回抽取，类型与超级比例沿用原 `_rollCrateType`；空格不足只放可容纳数量，记录 unplaced，不继续凭空补发。
- `sim.itemSupply` 保存 baselineInitial/baselineBricks/brickRate/expectedBaseline、requestedExtra/added/unplaced、initialActual、expectedTotal、brickDropsActual。页面显示开局实际数量和本局砖内实际生成数量；预期值与实际值分开。

| 场景 | 旧实际开局 | 新实际开局 | 旧预期总供给 | 新预期总供给 |
| --- | ---: | ---: | ---: | ---: |
| 806 默认正常图 | 0 | 24 | 47.142891 | 71.142891（约 1.509 倍） |
| 当前近距对拼场 | 0 | 15 | 0 | 15（零供给特例，参考 10×1.5） |

死亡掉落精确定义：保留原聚合栈的种类、count、isSuper，泡/威/速回收和道具栏清理规则不变。每 tick 枚举全图 0..194 的合法候选，对每个待投栈取 floor(rng×候选数)，移除选中格后继续，完全不读取尸体坐标、距离、出生侧。一个原栈仍占一个道具格，不拆分或乘倍。

合法性排除：越界、墙、砖、推箱、遮蔽、已有道具格、场地道具、存活泡、火焰残留、砖残留、已待生成砖道具、散落包子、空投预留格，以及存活人物 AABB（包括糖泡笼中人物）。不会覆盖这些格。候选不足时，完整保留尚未投放的原栈，每逻辑 tick 重试；满图不丢失、不复制。被炸道具继续原墓地/空投流程；落地时重新检查同一合法格条件，只排除空投自己的预留标记。如果起飞后新出现火焰、糖泡笼等占用，完整原栈退回墓地，等待下一轮。没有改成免炸或复制再生。

新引擎 reset 用 sim.seed 重置 mulberry32。新快照保存 itemSupply、itemRngState 与原 pendingDeathDrops，restore 恢复随机流继续位置。相同 seed/地图/选项/动作可复现 reset 布局及后续全图投放。旧版本随机语义与旧回放不受影响。

## 训练对齐交接

已尽早写 `.hermes/ITEM_RULE_READY.json`，原任务工作树和本发布工作树各一份，文件内用相对路径指出唯一源码位置。最终 revision 2，包含版本、精确定义、配置、源码/地图/评估入口/关键测试 SHA256 和测试命令。没有修改训练 CX 工作树、调用 CC 或停止训练。`.hermes` 为本地编排产物，不入库。

## 测试与本地真实浏览器

`npm test` 全部通过，另在最终关键测试扩充后执行 `node web/test_item_rules.js` 通过；`git diff --check` 通过。证据：`.hermes/item-npm-test-final.log`（落地安全复核后的完整测试）。

关键用例包括：供给实际数量与公式、零供给对拼特例、其他禁物品地图、合法属性上限、成功拾取真正增长；死亡回退及五种栈数守恒、重复 release 不复制；候选不足/满图/解除占用后的重试；非法格与糖泡笼 AABB 排除；固定 seed 与 reset 一致、随机流快照恢复；砖掉落实际计数不重复，炸毁堆叠转墓地及整轮鸟投数量保留、起飞后新火焰/糖泡笼的落地复检；旧引擎 SHA 与版本分派。

全图无偏测试将四角、两侧及中心共 7 个合法格逐一作为等宽随机区间的结果；移动尸体至四角并改出生侧后，相同随机流的布局完全相同。1,024 个固定 seed 的单次采样落入四象限为 212/279/250/283（象限格数不同，不要求四数相等）。代码直接对全部合法格等概率无放回选择，无距离排序。

真实 Chromium 桌面 1280px 与手机触屏 390px 均 PASS。原生浏览器处理真实逻辑 tick，测试观察器不加入生产文件：

- 正常图开局 24 件；对拼场开局 15 件。截图已实际查看，390px 无横向溢出。
- 成功拾取属性道具，泡数 2→3、威力 2→3、速度 1.3→1.457142857；边界及满上限覆盖于单测。
- 实际糖泡超时死亡：尸体 `[3.5,1.5]`，固定投放 seed=20261107，五个栈分别在 cell 101/164/113/114/90，类型 0/1/2/3/4 数量 3/2/2/4/6；包含远离尸体的合法区域，待投队列为空，原总量守恒。手机死亡截图已实际查看。
- 当前线上 G_frozen+τ0.3 真实模型加载、推理与动作执行，在正常图运行 32–33 tick、对拼场 29–30 tick，无页面错误；新规则标识和 10/8/2.4 上限正确，正常图本局真实砖掉落数为 2。此为可运行性检查，不宣称胜率或训练能力提升。
- 真人与模型 P1 拾取音效、静音/解锁及满库存静音回归通过。face 原地 160ms 充能、512ms 满圈放泡、实际穿越到 x=7.1，通过桌面键盘和手机触屏验收。
- 旧 Review 固定 8 局在真实浏览器完整校验、逐局 seek 终局并实际播放通过；完整 npm 还覆盖历史 D Review 15 个 seed。
- 排行榜继续暂缓，新局本地结算且不上传，不转投旧 ruleset；旧榜读取保留。合成测试拦截后端避免写入成绩，提交/资料写请求为 0。

本地证据：`.hermes/item-audio-local-browser.json`、`.hermes/item-facing-local-browser-results.json`、`.hermes/face-old-review-local.json`。截图：`.hermes/items-local-1280-supply.png`、`.hermes/items-local-390-death.png` 及对应桌面/手机版本。模型供给状态与死亡栈坐标数量保存在 JSON 中。

## 发布与隐私

玩法实现已正常推送 main，提交为 `5c621f6f8e413550fca92433d9e9691478594cd5`，没有 force。对应 [Actions 42333256159](https://github.com/allwellll/qqt-RL/actions/runs/42333256159)；首轮测试/构建成功，Pages 部署因 600 秒等待超时失败；第二轮 deploy-only 返回 Deployment cancelled，第三轮 deploy-only 再次等待超时。没有其他并发发布工作流。第四轮同 SHA 完整工作流重新测试、构建、上传成功，但 Pages 仍在 600 秒超时。另以独立提交 `e04646ef74f44dccc3380e91fba8bbe49a66d9a3` 将 deploy-pages 等待延长为 1,200,000ms；未改变玩法源码和物品定义。该提交已正常推送 main，最终 head 为 `e04646ef74f44dccc3380e91fba8bbe49a66d9a3`，对应 [Actions 42336959698](https://github.com/allwellll/qqt-RL/actions/runs/42336959698)。最终 Actions 的 test-and-build、deploy 均 completed/success；Pages 与 main 为同一 SHA，线上已完成真实浏览器验收。

发布前隐私扫描 PASS：tracked/staged 文件及本次报告、明确交接 JSON 共 2,227 文件，本机用户路径、内网 IP、阿里云 Key、GitHub token 均无命中。没有加载或输出新凭据，没有操作训练进程、训练工作树和其他 CX 的网页文件。

此报告保存本地交付结果；新引擎和关键测试随代码提交，所有引用使用仓库相对路径。手机验收使用真实 Chromium 的触屏模拟，不宣称实体 iOS 设备验证。

## 最终线上验收

[正式网页](https://allwellll.github.io/qqt-RL/) 回读 `build-info.json`：commit=`e04646ef74f44dccc3380e91fba8bbe49a66d9a3`、ruleset_version=`bun-native-v3-items`。19 个关键网页文件逐字节与提交一致，包括新引擎、旧 face/v2、三个冻结 Review 引擎、OSS loader、音效和小键盘控件。发布证据为 `.hermes/item-release-actions.json` 和 `.hermes/item-release-public-readback.json`。

线上真实 Chromium 桌面 1280px、手机触屏 390px 均 PASS：正常图 24 件，对拼 15 件；泡/威/速真实拾取增长与本地一致；实际死亡后的五栈坐标和数量与固定 seed 本地结果完全相同，待投队列为空。G_frozen+τ0.3 在桌面正常图/对拼实际运行 34/31 tick，手机 32/33 tick，砖内实际生成 2/0 件，属性均未超过 10/8/2.4。没有页面错误，模型成功加载并推理行动。此为运行冒烟，不替代胜率评估。

真人与模型的真实拾取播放调用各一次、满库存无调用、静音设置保持，音频资源四项 HTTP 200，AudioContext=running，真人拾取音效自然结束。face 在桌面键盘及手机触屏均原地 160ms 达 32%、512ms 满圈，真实放泡后穿过至 x=7.1，页面无横向溢出。旧 Review 8 局逐局校验/seek 终局通过，并实际播放 4 tick，冻结引擎 SHA256=`f6e346f3022640556efc2a988eb3b9b8e90abd5776af6ca1b4b1b0ee2acd6abc`。旧榜读取仍可用，新规则本地结算、重新开局正常，合成验收的成绩/资料写请求为 0。

首轮线上验收在异步切图尚未完成时读取了旧模拟器；本地验收脚本增加了“新模拟器引用已生成且 itemDuel 与选择地图一致”的等待，保留 24/15 数量断言，随后完整重验通过。生产玩法源码没有因此变更。

线上证据：`.hermes/item-audio-public-browser.json`、`.hermes/item-facing-public-browser-results.json`、`.hermes/face-old-review-public.json`；截图 `.hermes/items-public-1280-supply.png`、`.hermes/items-public-390-death.png`、`.hermes/item-facing-public-390-passed.png` 已实际查看。两份 `.hermes/ITEM_RULE_READY.json` 已更新为 published-verified，包含最终 main SHA、Actions/Pages 结果及精确源码 SHA256。
