# 当前网页、模型与暂停状态（20261107）

本页是11月7日状态索引。[在线游戏](https://allwellll.github.io/qqt-RL/)菜单仅保留SP、G_frozen、DUEL与风险线u9969四个选择；[模型manifest](../web/models.json)固定资产身份，[推理默认](../web/model_inference_defaults.json)固定默认与温度。旧报告中的“当前”“训练中”“下一段”仅代表记录当时，不是继续运行授权。

## 模型选择与证据

|菜单/温度|长处|短板|
|---|---|---|
|G_frozen / τ0.3，均衡|保留现有完整局参照|专项不代表全面领先|
|DUEL u019902 / τ0.3，近身击杀|v3 近身dev/holdout因果杀8/7，各16局；同批G为1/0。网页五对手40胜0负，交付40|网页自死9次，G为4；完整局可行动静止23.94%，G为18.83%；不称全局更流畅|
|SP u023302 / τ0.3，默认，猎手对抗与喷泉|v3 网页39胜0负1平，自死1；困难猎手8局因果杀7、交付10；有连续喷泉信号|完整局可行动静止34.08%；近身优势未建立；喷泉串弱于同点argmax|

训练侧生产JS评测共856局：筛查216、独立确认320、五对手网页320。网页各对手各方式8局；两个已发布τ0.3组合各有80局证据（近身dev16+holdout16、完整局holdout8、网页40）。argmax与采样分开，专项不等于完整抢包能力；不重复计算8局为更大分母。

原规则是 `bun-native-v3-items`；线上已经是 `bun-native-v4-facing`。独立v4快筛总24局（候选和G各8），关键指标与原v3同seed子集一致，未观察到说明的明显反转，仍只是兼容性与有限能力核查。网页说明卡独立列出原v3证据和v4快筛，不混写数字。[原v3证据](../web/specialists/items-20261107.json)、[v4独立快筛](../web/specialists/v4-screen-20261107.json)、[导出/发布报告](../reports/cx_web_specialists_items_v3_20261107.md)。

只消费 READY 固定checkpoint；CPU独立重导出、37/38维固定mode0投影、JSON/ONNX SHA与JS/ORT parity核对均已完成。源码资产不含权重；JSON/ONNX在OSS与Release。此次默认按用户最新明确授权改为SP，未新增评测，不将原v3数字称作v4全面优胜。风险线u9969保留为旧规则录像认可点；其余18项只移出菜单，资产和旧Review身份元数据保留于 `web/retired-models-20261107.json`。

## 当前游戏与手机操作

- **v4面朝穿点**：按主行/主列找前方中心距离最多3格的首个泡；墙、砖、推箱与cover遮挡。原地松键仍计时，不必接触或挤边，身后/斜向/超距不激活。保留原生 `500 < charged < 600ms` 与freshness小于100ms的窗口；计时圆环和放泡权限共用原生判断，满圈面朝放泡可实际穿过。转向新泡时目标转移，失去目标过期，不是永久权限。
- **v3道具继承**：正常图及对拼默认供给约原1.5倍，记录实际初始/砖内生成数；能力保持生产上限。死亡掉落在全图合法空格无放回采样，不带尸体距离或出生侧偏置；候选不足保留整叠待后续格可用时重试，不复制、不覆盖墙砖笼或已有物品。[规则报告](../reports/cx_item_supply_global_drop_20261107.md)
- **输入**：方向键；小键盘5上/1左/2下/3右，NumLock开关均按物理code识别；Space/W放泡；主键盘1–7对应道具槽，E/Shift为首槽；R重开。小键盘移动不触发道具，输入框/对话框不劫持。
- **手机**：方向与动作按钮最小56CSS px、空间充足时64px，画布外布局与safe-area；三键直接放置当前页对应槽位，显示真实图标/数量/空槽。4–7槽有换页，原库存不丢。只有香蕉皮、慢慢胶两种库存物理，三键不是新增三种道具。
- **长按/多指**：仅控制区禁selection、Safari callout、contextmenu与拖拽，保留普通文字和昵称编辑；pointer capture保持滑出滑回一致。逐指释放，cancel/lostcapture、blur、visibilitychange清理；长按道具一次，空槽不扣减/不伪播拾取音效。
- **音效**：拾取按真实成功事件播放，人类与模型均走现有事件链，保留静音/音量及用户手势AudioContext解锁；满库存未拾取不冒充成功。香蕉滑行显示平滑修复不改物理。

线上已验320/360/390竖屏及844×390横屏，实际长按超过2秒、多指移动＋放泡/槽位、世界道具与数量、七槽换页、松手和取消清理；另验桌面模型及v4无接触起计。范围是Linux Chromium真实CDP触摸及手机viewport，**不是实体iOS Safari**。系统callout和真机安全区仍需真机补验。[手机报告](../reports/cx_mobile_controls_20261107.md)

## 有效入口与历史兼容

|用途|有效入口与边界|
|---|---|
|网页真人游戏|`web/index.html` → `app.js` → `ruleset_selector.js`，当前 `sim-bun-native-v4-facing.js`|
|模型菜单/推理|`models.json`、`model_catalog.js`、`model_inference_defaults.json`、`model_loader.js`、model worker；manifest和SHA是资产身份|
|本地预览/测试|`npm ci`、`npm run serve`；`npm test`包含旧Review兼容，测试不表示恢复训练|
|版本化Review|`review.html`、`review_protocol.js`、`web/review-sim/`及明确版本simulator；按原ruleset/SHA回放，不改旧录像|
|历史G/联赛训练闭包|`model_training.py`已由`training_launch_policy.json`默认关闭；`model_entry.py`在导入payload前阻断训练/GPU并清理CPU任务继承的GPU环境，`scripts/model_launch.py`保留main可移植实现；[11月6日闭包](combat_league_v2_training_path_20261106.md)是快照，不含所有本地后续实验|
|评估/诊断|`model_evaluation.py`、`model_diagnostics.py`、`model_evaluation.js`读取本地配置；必须核对配置目标是评估而非训练，物理/模型SHA与输出分母单列|
|历史r6/cycle/JE10/R1–R4|保留源码、报告与原协议；旧命令/自动后继说明仅供追溯，不是当前启动入口或授权|

当前静态依赖闭包见 [20261107 manifest](combat_league_v2_dependency_manifest_20261107.json)，执行 `python3 scripts/audit_combat_training_path.py` 只解析源码、不导入训练或启动worker；135个根入口、348个本地依赖。11月6日manifest保留为历史快照。

不把文件名相似当作可互换的入口；历史 `sim-bun-native-v2.js`、v2-face、v3-items 与归档Review仍保留，不通过覆盖旧文件“升级”录像。当前物理由selector和实际加载文件决定，不单凭规则名判断生效。

## Release镜像与部署

浏览器按 **Pages同源Release镜像 → Release直链** 下载，逐资产验证大小/SHA256后才实例化；失败明确报错。OSS下载已停用，manifest历史OSS字段仅作元数据保留；不删除旧资产，不使用或公开凭据。

正常push main触发 `.github/workflows/pages.yml`；构建从Release取得manifest资产并SHA核验。只有同一SHA的Actions测试/部署最终成功、Pages build-info.commit和关键文件字节一致、浏览器Pages镜像加载与推理通过、Network无OSS请求才记交付。[当前发布规范](github_pages_model_delivery.md)

排行榜暂缓；后端尚不支持当前ruleset时显示暂不可用，仍能开局、本地结算，不发旧ruleset、不混榜、不删除旧数据。无需Supabase管理凭据或迁移阻塞玩法。

## 训练暂停与保存

用户最新指令覆盖此前“跑完预算段”的安排：**立即停止当前训练，撤销全部后继；未来明确重新授权前禁止启动、恢复、延预算或自动续跑**。停止与保存由训练owner实施；其最终回执确认worker为0、GPU计算进程为空、八卡锁释放，七臂最后完整COMMITTED的state/policy SHA通过。网页owner只接收其回执，不发GPU命令。[立即停训报告](../reports/cx_training_stopped_20261107.md)。保留最后完整 COMMITTED 和历史失败，未完整的瞬时状态不冒充已提交终点。

持久停止状态在本地 `.hermes/TRAINING_STOP_STATE.json`，它是编排文件而非公开凭据/协议。历史“持续”“下一1200”等ledger条目不构成新授权。CPU选模可继续，但不得以评估/恢复验证触发训练。不得删除runs、checkpoints、旧录像、秘密文件或大范围整理无关目录。

[实验ledger](training_experiment_ledger.md)保留原条目并新增有日期的覆盖记录；[报告索引](../reports/README.md)区分研究、规则、模型、手机与发布。训练owner已review的46项交付已核对封存包与逐文件SHA，非破坏性合并停训策略、专项/物品/CPU评估工具和文档；保留main可移植launcher、增量合并ledger、AGENTS仅合本地忽略规则，不整树覆盖其未就绪网页文件。[训练代码与封存索引](training_code_map_20261107.md)说明历史入口、外部冻结输入与当前禁用边界。
