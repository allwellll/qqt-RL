# 手机浮动方向与触控容错交付（20261108）

在已发布的纯击杀/隐藏复活地点版本之后，独立实现手机触控布局。保留默认 SP、四模型菜单、v4-facing、OSS、音效和旧 Review；没有改游戏物理、模型或训练，没有启动训练。

## 实际规则与尺寸

- 左侧方向区占游戏舞台宽45%、高60%，位于左下。手机舞台为一屏高，说明/昵称编辑区位于舞台之外，触控层不会覆盖输入框。
- 每次按下点就是本次中心；当前位移决定四方向。闲时外观显示习惯中心：每次按下按75%旧中心+25%新点更新；距离超过1.5半径则直接采用新点。
- 方向盘直径 `clamp(80px,25vmin,112px)`，死区为半径×10/48，范围约8.33–11.67 CSS px。390×900及844×390实测死区10.156px。
- 松开距下次按下严格小于250ms时沿用最后方向；其余初始按下不移动。滑回死区停走。四个90°扇区，切向需超过原方向中线60°，即边界外再15°滞回。
- 超出方向区仍由 pointer capture 保持移动。离中心超过半径时，中心每次移动事件追随超出部分18%，单步上限为半径25%。方向高亮，支持时请求8ms震动，不支持静默。
- 放泡可见直径 `clamp(88px,25vmin,104px)`；命中区直径为其1.25倍。下方三个道具键横排，尺寸 `clamp(48px,14vmin,64px)`。放泡扩展区与道具行之间10px间隔，不能串键；道具只按可见区命中。空槽不派发动作。
- 上述两种验收视口实测：放泡97.5px，命中区121.875px，道具54.594px。三槽对应当前页真实库存；4–7槽原分页保留，不删除或重排库存。
- 安全区使用 `env(safe-area-inset-*)`。触控方向使用独立 held 标识，松手不会释放同时按住的桌面方向键。主数字道具和小键盘移动隔离保持。

## 最小验证

通过：

```sh
node web/test_controls.js
node web/test_web_feel.js
node web/test_visual_parity.js
python3 scripts/audit_combat_training_path.py
PLAYWRIGHT_MODULE=<PLAYWRIGHT_MODULE> node scripts/verify_mobile_touch_tolerance_browser.js
git diff --check
```

真实 Chromium CDP 触摸，390×900竖屏及844×390横屏各一轮：

- 偏心按下不强行拉回旧中心；死区内不动；51°斜滑保持右，72°切下；250ms内短按续下，等待300ms后不续向。
- 手指滑出方向区仍持续真实位置移动；长按无文字选中；控制区取消 contextmenu。
- 第二指命中可见放泡圆之外的扩展区，真实放泡一次；抬动作指不释放移动指。
- 三个道具分别真实放置一次，仅扣对应槽数量，无放泡；空槽无请求。翻页后第7槽可用，其余6槽不变。
- touch cancel、真实 lostpointercapture、blur、切后台清理；同时键盘+触控时触控释放保留键盘；NumLock关闭对应物理小键盘码不触发道具；输入框正常输入。
- 两种视口无横向溢出、无 pageerror。截图与详细动作收据保存在忽略的 `.hermes/touch-tolerance-local-*` / `.hermes/touch-tolerance-local.json`。

验收脚本等待实际指针事件派发后检查状态。Chromium 对同坐标移动不生成新 pointermove；丢失捕获用例通过释放真实捕获并继续移动1px触发通知，没有以固定指针编号伪造通过。

范围：桌面 Chromium 的手机视口与真实 CDP 触摸，不是真机 iOS/Safari，不声明已验证物理震动或 Safari callout。CSS已保留局部 `-webkit-touch-callout:none`、禁选中及手势抑制；普通输入区不受该样式影响。

## 发布与隐私

隐私扫描仅覆盖Git追踪文件与本次新增脚本、报告，必须零发现后提交。正常 push main，不 force；同一提交的 Actions `test-and-build` 和 `deploy` 成功后才算发布完成。最终 SHA/运行链接及线上触摸收据另存忽略目录并在交付消息回报，避免为回写同一文档反复触发发布。

线上复用同一脚本，设置 `MOBILE_PUBLIC_URL` 和 `EXPECTED_SHA`，先确认 `build-info.json` 的 commit 与发布 SHA 一致，然后在实际 Pages 应用重复竖横触摸验收。没有额外模型评测或资产SHA回读。
