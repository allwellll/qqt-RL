# 网页、停训与代码文档整理（20261107）

## 已在线交付与本次基线

本次基于最新origin/main `c2ed1cb26b4c9546c7e8554f10dceae2eb451c74`，独立分支整理文档；没有等待新模型/训练拖延上架。

- 模型首发 `cd9cddf0af110f7825a31181e519e08ee1b8921f`，[Actions](https://github.com/allwellll/qqt-RL/actions/runs/42346134048)最终success。
- 手机修复 `c2ed1cb26b4c9546c7e8554f10dceae2eb451c74`，[Actions/Pages](https://github.com/allwellll/qqt-RL/actions/runs/42347826789)最终success；同SHA线上文件回读、四尺寸触控、多指与槽位、桌面手机两候选OSS SHA和真实worker推理通过。
- 默认G_frozen τ0.3保留；DUEL u019902 τ0.3与SP u023302 τ0.3可选，长处短板/原v3证据/当前v4快筛明确分列。DUEL网页自死更多，两者完整局静止偏高，不声明全面优胜。

## 立即停训

用户最新要求覆盖先前“跑完active段”：立即停止训练、所有后继撤销，未来重新授权前不启动/恢复/延预算。训练owner已完成安全停止；最终状态STOPPED_RELEASED_PERSISTENT_USER_HOLD，worker=0、GPU计算进程为空、八卡锁释放。七臂最后COMMITTED的state/policy SHA由owner重新核验；不为新保存点增加训练，不补跑已中止的fresh。[owner停训报告](cx_training_stopped_20261107.md)

网页交付侧只读停止状态与既有模型契约，仅按明确授权写交付回执，不改训练源码、闸门或进程；本次也不运行训练/恢复/smoke。所有runs、checkpoint、秘密文件、旧录像和旧Git历史保留，没有大清理。

## 文档与整合范围

- README更新实际模型菜单、温度、短板、手机/小键盘键位、v4面朝首泡最多3格、v3物品继承、OSS回退、停训与排行榜暂缓。
- `docs/current_web_and_training_status_20261107.md`集中当前有效入口/历史兼容边界；旧R1–R4及联赛v2指南保留并标记日期快照。历史训练示例不是当前授权，README默认验证不再包含训练smoke。
- `docs/github_pages_model_delivery.md`改为实际OSS→Release→同源、manifest/SHA与同SHA验收流程，撤回早期Top-N自动删除/历史重写草案作为现行执行说明。
- ledger保留已有全部实验记录，顶部加当前覆盖状态、末尾追加模型/v4手机/立即停训与整理条目；`docs/README.md`与`reports/README.md`统一索引。
- 收入各发布工作树最终报告和明确公开的浏览器JSON/代表截图，修复原报告跨工作树证据缺失链接。隐私边界为tracked与本次明确交付文件，不复制本地run、完整模型或私有编排状态。
- Pages仅增加当前简明说明与已review训练代码图文档；模型、网页运行代码、旧Review、已发布物理字节保留。训练owner后续已review代码文档仅按明确清单/SHA非破坏性接收，不整树拷贝其未就绪网页改动。

## 测试与最终发布

完整 `npm test` 已成功（含全部旧Review）；controls/model catalog/v4 range定向测试通过。文档当前入口与所有新增索引本地链接核对0缺失；tracked隐私扫描0命中，另做待提交文本绝对路径/凭据扫描。修复ledger历史任务名串中的路径样式分隔，保留原数字与结论。

入口审计发现v3/v4继承的受try/catch保护可选TimeAStarAI未被旧例外表登记；仅补明确版本例外，未创建Bot或放宽普通依赖检查。新日期manifest `docs/combat_league_v2_dependency_manifest_20261107.json`为当前默认，保留11月6日原manifest字节。静态审计109roots/321dependencies通过；`python3 -m pytest -q tests/test_combat_source_audit.py` 两项通过，包含非可选缺依赖仍拒绝。没有JAX导入、训练更新或fresh运行。原训练路径测试改读新manifest，不声明缺外部JAX的整套Python训练测试已跑。

文档首提交 `12e4f4cd7c4b2c4380f7cad0b67845e1b4331ba8` 正常push后，训练owner的REVIEWED_READY_FOR_INTEGRATION封存交付到达。已逐项核对46文件与tar包SHA，不等待训练或新模型；owner原CPU复核7停训闸门/69防hack/5结算及2578协议源码检查记录在逐项整合收据。

新增默认关闭的 `training_launch_policy.json` 与 `model_entry.py` 的导入前闸门；即使没有私有停止状态，训练角色/GPU配置/CUDA环境仍拒绝，CPU任务清理继承GPU环境。引入明确清单内DUEL混合/同步、v3物品版本迁移、CPU选模工具和代码图，全部只作封存来源或显式CPU用途，不执行prep/migration/owner/训练/fresh。外部protocol、Bank、场景、冻结副本和模型仍是必需本地输入，不冒称干净checkout可恢复。

冲突处理：保留main已修复的可移植 `scripts/model_launch.py`，不回退到训练树机器专属别名；AGENTS规则只合本地忽略文件；ledger提取相对来源base的纯新增历史段追加到main，保留两侧记录，公开副本脱敏一个机器进程标识；当前停训覆盖所有历史后继文本。两个历史物品准备/CPU选模工具的契约定位改为QQT_ITEM_RULE_READY（默认当前仓库私有契约），去掉兄弟工作树绑定；来源报告外部契约位置改中性占位符。算法与原封存协议未改，源/结果SHA分列；其余清单文件按封存字节接收；训练树全部web改动排除，旧协议/Review归档不改。[逐文件源SHA/处理/结果SHA](cx_web_and_docs_delivery_20261107/training-code-integration.json)。

静态闭包加入已review入口后扩为135roots/348dependencies，显式package import无缺文件。当前实际合并源码的停训闸门与闭包pytest9项、既有训练路径4项、69项CPU纯结算/防hack、5项独立DUEL纯结算检查均通过；allowlist内Python/Node语法通过，包内显式import本地依赖无缺失。完整npm在整合后再次全部通过。无collector、训练、优化器更新或fresh启动。

CI新增标准库unittest停训闸门与静态源码闭包步骤，不安装JAX、不初始化GPU。

本次最终SHA/Actions/线上验收在完成后回填。此前浏览器证据不冒充本次新的实测。
