# 手机方向区点按即时移动（20261116）

## 选择的交互与RED

前序 `547bb3e82208ede74e4cd01b031c464ce42aaf74` 已发布验收。工作树干净、fetch确认远端相同，新建独立分支，唯一writer；未启动训练。

旧版RED：createFloatingStick.start不论触点都返回null；pointerdown/up在同一rAF前发生后held为空，下一帧没有移动。它与“触点本身定中心”的径向设计一致，却不能支持即时点按。

本次采用可见方向盘，不增加隐藏推断：

- 空闲参考中心固定为方向区几何中心，四向区域由可见对角分割线标示；松手圆盘恢复CSS中心，不随历次触点漂移。
- 圈内、距中心大于等于18%半径的点：pointerdown当事件就生成四向方向，按住持续移动。四向初次选择取相对中心的绝对分量较大者；完全相等稳定取水平（左/右），不随机、不继承旧方向。
- 中心18%半径内或可见圆盘外的方向区：按下不移动，触点成为浮动拖动中心，后续径向拖动。区外沿8CSS px不捕获，防屏幕/区域边缘误触。其他区域和右手按钮没有移动绑定。
- 拖动继续使用18%径向死区、58°滞回、25–65°次方向、超半径中心跟随与160ms路口缓冲；明确反向不被上次触摸覆盖。松手清除方向、次向、路口缓冲。
- 若正常pointerup之前没有任何rAF读取方向，保留最后一个有效意图，100ms过期，覆盖而非累计；下一帧只对真人调用一次1/60秒frameStep，此后立即空闲。长按已采样就不补走。cancel/lostcapture/blur/隐藏/resize全部丢弃短意图，键盘或新触摸也会覆盖它。未推进整局step、未增加自动走路。
- 困泡、死亡、香蕉强制状态不消费短意图。明确单方向碰到墙/泡泡仍提交朝向，原生frameStep决定碰撞和充能，不因不可达查询吞掉输入。

实际390×900视口圆盘直径97.5px、半径48.75px，中心区约8.775px。沿用原物理6px转角容差与原生移动速度；模型、训练、回放引擎不变。

## 验证

```sh
node web/test_controls.js
node web/test_web_feel.js
node web/test_pass_facing_range.js
node web/test_banana_boundary.js
node web/test_http.js
node web/test_replay.js
node web/test_review_source_hash_compatibility.js
python3 scripts/audit_combat_training_path.py
STICK_STAGED=1 PLAYWRIGHT_MODULE=<PLAYWRIGHT_MODULE> node scripts/verify_mobile_stick_browser.js
```

均通过。新 `web/test_mobile_tap_move.js` 由controls测试入口执行，覆盖四向同步点按、中心/外沿/盘外、可见中心归位、浮动拖动/跟随、四个对角稳定tie和轻抖、连续点按/明确反向、同帧短意图只消费一次/最后一次优先/100ms过期、清理、键盘优先、泡泡面向充能及香蕉强制方向。上一版径向慢/快划、路口和只读碰撞测试继续执行。

最小真实390×900移动Chromium/CDP：加载当前默认模型，用待发布controls/app/CSS检查；Supabase全部拦截。四向pointerdown直接返回方向，按住实际移动、松手后位置不再变化。测试包装暂停游戏rAF，真实CDP先down再up，位置不变；恢复后真人仅从x=4.5到4.55（0.05格），80ms后仍不变，证明没有短点按丢失或自动持续移动。中心/3px抖动为空闲，方向区外沿误触不激活，拖动/超半径跟随、斜向路口实际转弯通过。

多指移动+放泡、三槽分别减1且其他库存不变、cancel/blur/lostcapture/resize、输入框、小键盘隔离、禁选中/contextmenu均通过，无pageerror。人为给真实推理额外300ms延迟时，采样50帧，其中39帧推理等待中仍移动，帧间隔P95为16.7ms、输入选择P95约0.10ms。只为确认事件/rAF链路持续，不声称模型速度改善或实体iPhone性能。

## 发布及局限

正常提交push main；隐私扫描只覆盖Git追踪+新增测试和本报告，禁止用户标识、本机绝对路径、内网IP/PID。等待同SHA Actions/Pages最终成功后，回读build-info、controls/app/CSS/index字节，并以已发布代码再次做相同最小移动检查。截图与收据在忽略目录 `.hermes/mobile-stick-public.png`、`.hermes/mobile-stick-public.json`，最终发布收据 `.hermes/mobile-tap-move-published.json`；待发布证据同名staged文件。最终SHA/Actions在交付消息回报，不为记录自身SHA循环部署。

极短点按最少一帧是明确交互例外：正常抬手立即停，未采样短意图允许100ms内一次受限移动；超过100ms宁可丢弃，避免失焦后迟到走动。测试仅Chromium手机视口，未声称iPhone/Safari真机；不扩大物理转角、不改模型或训练，不force、不调用CC。
