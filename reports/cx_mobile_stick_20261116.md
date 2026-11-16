# 手机径向摇杆跟手优化（20261116）

## 基线、原因和实际链路

工作树原HEAD `64eb579`、无未提交内容；fetch后安全快进至 `a36107d`，新建独立分支。同步保留已有模型菜单/默认、魔术师形象、Pages模型镜像和沉浸模式。未运行训练或修改模拟器源码。

确定性复现旧输入：触点(100,100)，拖到(130,100)后向下仅抖4px，旧算法输出ArrowDown；回到原中心(100,100)输出ArrowLeft。原中心仅装饰，判向取相邻pointer事件的短stroke，故当前手指相对中心的位置与移动方向不一致，也没有斜向次方向。

实际链为pointer事件 → held/径向状态 → rAF的stepHumanFrame → 原生frameStep。真人动作第4位仍为1，不由10Hz模拟tick重复移动。模型ONNX在Worker，tick等待act时，rAF可继续推进真人；本次没有修改推理代码或声称提升模型吞吐。事件含合并样本时逐个消费，输入状态立即更新，只有合法方向选择/物理提交在rAF执行。

## 参数和边界

- 半径为实际显示圆盘宽度的一半，随现有80–112px圆盘自适应；本次390×900视口实际直径97.5px、半径48.75px。
- pointerdown位置成为真实中心；拖出半径后，中心跟随到距触点一个半径的位置，非装饰跟随。
- 径向死区18%，上述视口约8.775CSS px；中心内立即idle，松手不恢复前次方向。反向必须拖到当前中心另一侧并超死区，同一pointer事件接受，不再用3px短stroke反转。
- 当前四向保持到偏角超过58°；43–47°抖动不来回切向。相对水平轴25–65°时保留相邻垂直/水平方向作为secondary。
- 转向缓冲160ms，从方向意图变化事件计时，不每帧续期。已沿一路行走时优先尝试垂直候选，未通则短暂沿旧路；主路受阻时尝试次方向。成功转入后保持选中方向，避免同一斜向每帧左右轮换。过期缓冲不在下个路口复活；持续主方向及被挡时的次向仍属当前有效输入。
- 可达性查询复用当前原生 `_nativeResolve`，查询一个25ms投影距离，限制1–40原生像素；使用私有native状态和wallExit副本，不调用step/frameStep、触点计时、推箱累计或随机数。原有合法转角修正可被视为可达，实际位置仍仅由frameStep提交。
- 原生转角容差保持6px/40px=0.15格，未采用参考建议的0.3–0.4格，因为这将改变生产物理。原始墙/砖/泡泡阻挡、穿点权限及香蕉强制方向保持。
- 触控松手、pointercancel、lostcapture、失焦、隐藏与旋转清空方向/次方向/缓冲；键盘优先级继续由held顺序决定，小键盘不触发道具。触控清理不删除独立键盘状态。主箭头/实际选择方向高亮，次方向有独立颜色提示。

## 针对性验证

```sh
node web/test_controls.js
node web/test_web_feel.js
node web/test_pass_facing_range.js
node web/test_banana_boundary.js
node web/test_model_worker.js
node web/test_native_restoration.js
node web/test_replay.js
node web/test_review_source_hash_compatibility.js
node web/test_http.js
python3 scripts/audit_combat_training_path.py
STICK_STAGED=1 PLAYWRIGHT_MODULE=<PLAYWRIGHT_MODULE> node scripts/verify_mobile_stick_browser.js
```

均通过。`web/test_mobile_stick.js`由现有controls测试入口执行，覆盖1/4/16/120种采样密度、慢/快划、微抖、45°滞回、反向、中心归零、超半径跟随、斜向路口/阻挡次向、缓冲过期与取消、原生墙角实际移动、查询前后完整模拟器序列化不变、泡泡阻挡及香蕉强制运动。既有手机验收脚本同步其过期的4px/短stroke断言；本次不重复执行整个旧大脚本。

真实Chromium/CDP 390×900待发布检查：从Pages加载当前默认模型，替换本次待发布controls/app/CSS；其余游戏及模型资源为实际资产。Supabase请求全部拦截，无生产写入。在测试控制的原生开放场景检查真实多指移动+放泡、三个槽位各减少1且其余库存不变、松手/取消/blur/lostcapture/resize清理、小键盘及输入框隔离、禁选中与contextmenu。真实斜向输入从横向通道转入上方窄通道，人物位置到(6.05,6.5)，实际选择ArrowUp。

人为对每次真实Worker推理额外延迟300ms，约750ms采样窗口内观察52帧，其中39帧在推理等待中、真人前进2.775格；帧间隔P95为16.7ms，方向选择CPU耗时P95为0.10ms。该数据是本机模拟手机结果，不代表真机性能，也不是修改前后推理速度对比。首轮脚本在touchCancel后误发touchEnd退出，修正测试事件序列后完整通过。

## 发布

隐私扫描仅覆盖Git追踪与本次新增测试、浏览器脚本、报告，禁止用户标识、本机绝对路径、内网IP和PID。正常commit/push main，不force。等待同SHA Actions测试/构建/Pages成功后，不替换代码再次运行同一最小移动浏览器脚本，回读build-info SHA与线上controls/app/CSS字节，确认真实功能。

本机验收收据/截图为 `.hermes/mobile-stick-staged.json`、`.hermes/mobile-stick-staged.png`；最终线上为 `.hermes/mobile-stick-public.json`、`.hermes/mobile-stick-public.png`，发布收据 `.hermes/mobile-stick-published.json`。最终SHA与Actions链接在交付消息回报，不为写入自身SHA循环部署。范围仅移动Chromium/CDP，未声称iPhone/Safari真机；未增加十字键备选或改写历史录像。
