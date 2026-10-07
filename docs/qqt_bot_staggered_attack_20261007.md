# QQ堂 Bot 错峰连续进攻（20261007）

基线为 `991416fc028e1050efeef94f0b40ae4235bb29e8`。在同一 worktree 接续 dirty 草稿，未 reset、覆盖或丢弃修改。本项不修改数据库/API、人物、结算或榜单。完成后停止，不启动排行榜第三项。

## 审计与实现

决策入口是 `BunCoopHunterBot.analyzeSim → analyze → chooseGoal/considerBomb/pickMove`。`BunHunterBot.predict` 展开每 tick 危险图及连锁；`escape` 搜索时间路径，`hypoSurvivors` 考虑敌人新增泡；Coop 的 `physicalEscape` 校验真实连续坐标与转角。新泡相对决策状态第 31 tick 爆炸，放置后引信 30，预测火焰 3 tick；Sim 连锁严格同 tick 引爆。原 reserve/connector 会把几颗泡压成同一爆炸阶段，且两泡容量缺乏主动延长线续爆。

新增仅 hard 的 STAGGER 计划：沿已有本人泡的射线/延长线选择可达候选，在临爆窗口补泡，保留不同爆炸阶段。延长线自然续爆；射线内连接泡必须新增敌人或敌方路线覆盖，且另一枚本人晚爆泡不被提前连掉。候选必须具备目标压力、足够接近与撤离时间；不是机械填满容量。原即时攻击机会优先，原 reserve/chain、搬包、救援等任务保持优先级。

泡上限仍为 hard 原值 4；场景实际容量 2/3，引信仍为 30；原 robust 和三 tick 保险保留。接近与放置检查真实坐标、提前三 tick 危险、敌方可放泡格、道具与滑行；放置后继续撤离验证。队友静止、搬包路线、救援及逃生空间仍受保护，STAGGER 首步经过原共用道具/队友扫掠间距过滤。

## TDD 与评估口径

过程证据位于 `runs/qqt_staggered_attack_20261007/`。`tdd-fixed-behavior-red.log`、`tdd-carrier-reference-red.log` 在原实现/内存 Git 基线确认实际错峰行为缺口；`tdd-immediate-opportunity-red.log` 复现新计划抢占即时捕获机会；`tdd-spacing-red.log` 复现接近路线绕过队友间距。修复后定向测试 GREEN。

`web/test_bot_staggered_attack.js` 覆盖实际错峰、临爆延长线、连接泡同 tick 引爆但保留独立后爆阶段、撤离余量、墙角/香蕉/慢慢胶/提前连锁拒绝、静止队友/救援/搬包优先、即时攻击机会、队友间距与确定性回放。既有 Coop 套件继续覆盖分区开墙、reserve/三泡连接、搬包拦截、救援护送、道具等。

独立审查发现并实际复现三个评估器问题：半身几何覆盖不等于真实判伤，无敌状态不应算命中；糖泡延迟死亡要保留初次捕获归因，救援/清理后撤销；自己的泡阻塞不能计为队友阻塞。`tdd-oracle-red.log` 和 `web/test_staggered_attack_oracle.js` 确认修正。现在几何覆盖单独报告，真实命中基于同 tick 总覆盖归因及实际捕获/掉血/死亡。

同批 seed 固定 `2026100701..2026100704`，四方向、solo/静止队友、容量/敌距/搬包三组合，共 24 场；另用前两 seed、806 地图、size 1/3、team 0/1 共 8 场。双方基线和候选同配置，不筛 seed。场景攻击窗口 72 tick，追加 91 tick 静止跟踪；地图攻击窗口 360 tick，追加 91 tick 仅移动撤离跟踪。尾期决策前禁用双方新增泡能力，不产生被截断的队友 phantom 泡；仅用于关闭最后放泡的引信与糖泡死亡观察窗口，不改变主动阶段容量。历史探索文件保留，最终结果以 `comparison-final.json` 为准。

每枚泡记录放置/预期/实际爆炸 tick、自然或连锁来源及父泡、覆盖/真实命中、逐 tick 真实位置撤离路径和最后射线接触到爆炸的保守余量。链式补泡不把同 tick 的连接泡计成独立阶段。ACCEPT 同时检查总体时序改善以及每场真实命中/捕获/死亡和安全指标不退化。

## 同批最终结果

`ACCEPT=1 node scripts/eval_staggered_attack.js` 实际 exit 0。32 场总体：

| 指标 | 基线 | 候选 |
| --- | ---: | ---: |
| 连续多泡场次 / 比例 | 12 / 37.5% | 31 / 96.875% |
| 实际放泡 | 202 | 222 |
| 实际错峰共存配对 | 70 | 106 |
| 各场不同实际爆炸 tick 数之和 | 155 | 192 |
| 临爆空间衔接的不同 tick 续爆配对 | 27 | 56 |
| 续爆配对 / 错峰配对 | 38.57% | 52.83% |
| 几何覆盖 / 真实命中 | 28 / 26 | 28 / 26 |
| 敌方捕获 / 延迟死亡 | 26 / 26 | 26 / 26 |
| 本人泡自陷 / 自困 tick | 1 / 28 | 1 / 28 |
| 队友泡伤害 / 队友阻塞 / 重叠 tick | 0 / 0 / 0 | 0 / 0 / 0 |

全部 32 行安全及真实命中/捕获/死亡逐行无退化。19 枚 STAGGER 泡均有实际爆炸事件，预期与实际 tick 一致，保守撤离余量最小 6 tick。搬包敌人正例放置 tick 1/20/40，实际自然爆炸 tick 31/50/70，后两泡撤离余量 11/30 tick；射线内 connector 正例 tick 1 放置、tick 10 被 anchor 引爆，另一枚仍 tick 24 爆炸，未被提前连掉。

额外 connector 完整轨迹保存为 `linked-connector.json`。此例保守几何余量为 0，因为撤离后身体一半仍接触射线；按 tick 10 所有来源总覆盖及原版完整身体判伤，最后危险位置为 tick 3，判伤余量为 7。两项分别保留为 `safetyMarginTicks` / `damageSafetyMarginTicks`，不隐藏几何零值。`tdd-contact-margin-red.log` 先确认缺少完整判伤余量字段，新增计算后测试直接调用真实 `Sim._isHitByExplosion` 逐位置核对，确认三 tick 保险仍满足；独立审查也实际核对通过。

改善主要出现在 24 个固定进攻场景。8 个真实地图短局中错峰配对 45→44、续爆配对 7→8、不同爆炸 tick 116→116，真实命中/捕获/死亡 2→2；不宣称自然对局胜率提高。基线已有一次本人泡自陷与 28 tick 自困，候选同样保持，不能宣称整个批次零风险。观测窗口延长仅用于补齐实际事件，历史短窗口结果和被拒绝的中间实现保留在证据目录。

## 验证与发布证据

新测试纳入完整 `npm test`，最终日志为 `npm-test-final.log`；定向日志为 `targeted-final.log`。本地 SQL 回归仅在 PGlite 执行，没有正式数据库写入。

真实 Chromium 149，1440x1000 和 390x844：`browser-local/checks.json` 使用实际网页加载的 Sim/Bot 执行确定性事件轨迹，另检查网页 Canvas 非空、布局无横向溢出、默认毛毛。网页截图展示实际应用，轨迹使用独立 Sim，不宣称截图展示该轨迹战斗。`browser-local-loading/checks.json` 验证延迟素材无错误 loss 提示及真实 loss；`browser-local-round4/checks.json` 验证胜利感言、loss 无资料请求、成功提交/X 一次新局、失败不重开、资料重试/幂等/榜单版本保护。全部远端写请求 mock，非预期远端请求为零。截图人工检查通过。

独立只读审查 `round4_review` 审阅实现并实际复现问题、执行定向/评估口径测试及逐行指标门槛核对；修复后复审。完整审查记录为 `independent_review_20261007.md`。提交前执行 `git diff --check`。

最终 SHA、push、同 head Pages 工作流成功、build-info 与 18 个关键资源逐字节核验、线上双视口 Bot/加载/结算 mock 功能验收及额外真实只读榜单检查，记录于 `runs/qqt_staggered_attack_20261007/release_status_20261007.md`。对应 `pages-final.json`、`browser-pages/checks.json`、`browser-pages-loading/checks.json`、`browser-pages-round4/checks.json`、`pages-readonly/checks.json`。正式 SQL/数据库写入及排行榜污染始终禁止。
