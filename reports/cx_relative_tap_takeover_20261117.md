# 手机相对点按接管（20261117）

## 同步、根因与RED

实际发布工作树为上次任务使用的工作树，HEAD `4196d54a87010382acc700115384b30f03a7e38e`，无未提交改动；fetch确认origin/main相同，从其新建分支。另一个历史工作树仍在旧face提交，其5份未跟踪历史报告和本地分支全部保留，未把旧物理改动重新合入。唯一writer，未启动/恢复训练，不修改模型、训练或模拟器源码。

根因可复现：`bindFloatingDirection`在pointer非空时忽略所有新pointerdown；点击还采用固定zone中心，且小圆外一律重新归零。新增真实绑定路径测试先在旧代码失败：右向按住后第二指点上方，预期MOVE_UP=0，实际MOVE_RIGHT=3。不是按规则名或只看UI推断问题。

## 最终交互和阈值

- 当前显示圆心就是相对点按参考。初始为方向区中心，拖动超过半径时仍按原径向方式移动可见圆心；松手清空方向但保留圆心显示，下一次单指或多指按下都用该位置。没有不可见长期漂移；中心死区内点击不改变圆心，blur/隐藏/resize/旋转等全局清理后恢复初始中心。
- 只在左方向区处理，外沿8CSS px拒绝，中心死区半径18%保留。小圆外但仍在方向区的明确方向也立即接受，不再以“小圆外”强制归零。
- 每个有效方向pointerdown直接接管。水平夹角小于35°取水平，大于55°取垂直，不沿用旧58°滞回；明确上/下/反向优先相对圆心。35–55°模糊对角先看相对旧主触点（或松手200ms内最后触点）的位移：长度≥8px且某轴为另一轴≥1.5倍，并与相对圆心象限一致，才用该轴。否则保留同象限的当前有效方向；无依据取较大轴、完全等长取水平。此为明确算法，不声称猜中真实意图。次方向仍保留给既有只读可达性/路口处理。
- 方向所有者是最新有效点按。中心或外沿误触不抢正在控制的方向。旧手指保留捕获但挂起，微抖不会抢回；累计离开其挂起位置达到max(12px,半径25%)才可明确再拖动接管，且以挂起位置重新定拖动中心，避免旧绝对偏移把角色拉回原向。390×900实测圆盘直径97.5px，阈值约12.19px。
- 接管指松开停止，不自动恢复旧向右。所有指针松开不持续移动。未被rAF读取的快速点按仅保留最后一个，100ms过期、最多补一次1/60秒真人frameStep；旧指抬起不丢弃新指的补帧，也不会恢复它。
- pointercancel清全部方向指针/缓冲/待补意图；blur、隐藏、旋转/resize同样清理。旧指lostcapture只移除旧指，当前指lostcapture全清；不在接管时release旧capture，清理前先清Map和owner以防重入。stale move/up找不到有效记录就忽略，复用ID的新down建立新记录，早于该down的时间戳不接受。
- 拖动58°滞回、18%径向死区、25–65°次向和160ms路口缓冲保持；新点按清旧方向缓冲，立即提交新意图。原生墙/泡泡、面向充能、香蕉强制状态、键盘及旧Review语义保持。右手泡和道具继续使用独立pointer映射。

## 针对性测试和真实浏览器

```sh
node web/test_controls.js
node web/test_web_feel.js
node web/test_pass_facing_range.js
node web/test_banana_boundary.js
node web/test_model_worker.js
node web/test_http.js
node web/test_replay.js
node web/test_review_source_hash_compatibility.js
python3 scripts/audit_combat_training_path.py
STICK_STAGED=1 PLAYWRIGHT_MODULE=<PLAYWRIGHT_MODULE> node scripts/verify_mobile_stick_browser.js
```

均通过。`web/test_relative_tap_takeover.js`由controls入口执行：真实绑定事件→held→moveFor→原生frameStep，检查第二down即时接管及人物坐标；浮动中心162px后小圆外上方点按；旧指抖动/明确恢复；owner up/旧指up相互不误清；同帧和分帧短tap；明确四向；斜向惯性与触点位移tie；中心/边缘；stale move/up、ID复用、lostcapture和全量取消。上一版点按/径向/路口及泡泡面向、香蕉测试继续保留，旧版“松手归固定中心/小圆外中性”断言按新授权更新。

最小真实移动Chromium/CDP（390×900）：从Pages加载现有默认模型，待发布代码替换仅controls/app/CSS。Supabase RPC全拦截，无生产写。原手指向右拖后显示圆心从x87.75移动到约109；第二指在该圆心上方65px（小圆半径48.75px之外）pointerdown立即显示ArrowUp，人物y从6.5下降至6.375。第二指松开后旧指2px/1px微抖不移动，再向下20px实际恢复向下。测试期间的帧推进/跨CDP调用延迟不当作零延迟承诺。

暂停游戏rAF时，右向旧指仍按住，第二指真实down/up在帧前完成；恢复只向上移动0.05格，x保持，随后位置不变。连续后续下/左点按、第二指取消、中心/外沿不抢、旋转清全部指针通过。既有四向点按、中心/边缘、径向拖动/跟随、原生路口实际转弯、多指泡和三槽各减1且其他库存不变、blur/lostcapture、键盘/输入框均通过，无pageerror。最初浏览器脚本误将CDP touchEnd列表中的旧指作为要释放的指，已纠正为实际接管指后完整通过；这是测试事件修正，不伪称此前验收成功。

额外300ms真实推理等待下仍观察到50帧，其中38帧等待中移动，P95帧间隔16.7ms、输入选择约0.10ms；仅证实主线程/rAF持续，不是新模型评估或真机性能结论。

## 发布与回执

隐私扫描只看Git追踪及本次新增测试/报告。正常commit/push main，不force。等待同SHA CI/Pages最终成功后，回读build-info与controls/app/CSS/index线上字节，不替换代码再做最小移动Chromium点按接管验收。

真实浏览器证据路径 `.hermes/mobile-stick-staged.json`、`.hermes/mobile-stick-public.json`，截图对应同名png；文件核对 `.hermes/relative-tap-files-public.json`。简明最终回执 `.hermes/relative_tap_takeover_receipt.json` 记录SHA、Actions、报告和证据路径、线上状态。最终SHA/Actions在交付消息回报，不为报告记录自身SHA循环部署。

范围仅移动Chromium/CDP，无iPhone/Safari真机验收；新算法是固定几何/阈值选择，边界点依然可能与用户主观意图不符。旧指需要明确再拖动才能恢复是有意设计，松手后可见圆心保留也是本次授权的交互变化。历史报告、资产、录像保留；未恢复训练、不CC、不force。
