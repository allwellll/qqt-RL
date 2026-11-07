# 报告索引（20261107）

本索引先列当前交付，历史报告按原日期保留。运行/规则/模型身份以较晚日期、明确SHA及真实回执为准；历史“继续训练”不覆盖用户最新立即停训指令。

## 当前网页交付

|主题|报告|范围|
|---|---|---|
|本次文档整理与整合|[统一交付](cx_web_and_docs_delivery_20261107.md)|最新main、已review来源清单、测试、隐私、同SHA部署与线上回读|
|DUEL/SP可选模型|[候选模型交付](cx_web_specialists_items_v3_20261107.md)|可信READY导出/身份、OSS/Release、v3原证据与v4快筛、默认G保留、桌面手机真实加载|
|手机操作|[手机触控](cx_mobile_controls_20261107.md)|局部防长按菜单、56–64px、三槽直放及4–7换页、多指/释放、线上四尺寸；非实体iOS|
|无接触面朝穿点|[v4距离修复](cx_pass_facing_range_fix_20261107.md)|前方首泡最多3格、遮挡/目标/闲帧、半格整格真实间隙、圆环与真实穿过|
|道具与全图掉落|[v3物品规则](cx_item_supply_global_drop_20261107.md)|约1.5倍供给、生产属性上限、全图合法掉落/数量守恒、旧Review版本隔离|
|OSS下载|[OSS交付](cx_model_oss_20261106.md)|对象前缀ACL、bucket私有、受限CORS、SHA与回退、无前端凭据|
|face首版与拾取音效|[face交付](cx_pass_facing_rule_20261107.md)|历史v2-face与拾取声音修复；面朝范围以较晚v4修复为准|
|立即停训与保存|[停训报告](cx_training_stopped_20261107.md)|七臂最后COMMITTED、worker归零、八卡释放、持久禁止新训练及后继|
|历史主线闭包/轨迹|[main合入](cx_main_merge_20261106.md)、[时间轴证据](mainline_timeline_evidence_20261106.json)|G→联赛v2闭包与截至11月6日轨迹，不冒充最新网页模型|

模型与手机的浏览器JSON/截图在各报告同名证据目录。新增报告回填会说明哪些验证是本次fresh、哪些是原发布后已完成证据；没有重跑时不改称新的实测。

## 研究与历史兼容

- [实验ledger](../docs/training_experiment_ledger.md)：保留正/负结果、协议修订及停止授权，最新状态在顶部与末尾日期条目。
- [当前网页与有效入口](../docs/current_web_and_training_status_20261107.md)：v4、模型、手机、OSS、暂停边界。
- [R1–R4快照](../docs/r1_r4_training_guide_20261101.md)、[G/联赛v2闭包快照](../docs/combat_league_v2_training_path_20261106.md)：历史代码与外部产物要求，非恢复授权。
- [JE10长期总结](../docs/je10_control_route_training_summary.md)、[旧Review与D证据](../docs/README.md)：旧规则、hash、奖励口径继续按原版本读取。
- `runs/`、`checkpoints/`、原始录像和秘密文件保持原地、不因索引整理删除或发布。训练owner停止/COMMITTED的本地原始回执仅在明确review、脱敏后进入公开报告，不复制私有编排状态。
