# 严格串行玩法验收 20261006

## 工作区与执行约束

在固定 `cx_bot_cooperation_20261004` worktree 从 `1c913b7` 恢复；没有 reset、新建 workspace/worktree 或 goal。五个已有 dirty 文件先完整保存至 `runs/coop_win_20261006/recovered_candidate/`，含 SHA-256 清单与 tracked patch。原 CPU 进程 PID 594528 已结束（defunct），没有终止该进程。后续阶段按 Bot → 10% 半身爆炸 → 香蕉墙角滑行 → 终局贴图内结算执行。

## Bot 策略审查与实验

- 恢复后的首条命令为 `git diff --check && node web/test_coop_win_strategy.js`，实际通过；随后完整 `npm test` 通过，日志 `/tmp/coop-win-resumed-npm-20261006.log`。
- 当前候选五项策略：提前拦截携包敌人的必经路线；接触前与救出后的定时安全救援；开局按出生点划分发育区域；必要时穿越胶水或安全炸除堵路胶水；错峰储备与实际三泡连锁。默认未启用，通过 `winStrategy` 显式测试。
- 当前候选 SHA-256：`aae7333d28058b3c3c7074b80e10628a0b9934143d8c8098e66eea5704841860`。早期 B 为 `e6af7583bc631b89506b4d223dc9f45031c164828ce7c94501a2dfa6127b0886`，不可混用其收益。
- A 用 `git show 1c913b7:web/bun_coop_hunter_bot.js` 动态加载。模拟器、真实 JS Hunter、Tactical v2、适配器及地图的文件哈希均核实一致。原生道具/糖泡、完整806地图、2400 tick、CPU，无模型/Python/JAX改动。
- 每个对手、每个策略臂32固定 seeds（`2026100600 + i*7919`）×双向 seat swap，共64局。出生组水平对称，同一个 seed/seat 使用相同 actor seed，行动顺序 pid 升序。按对手独立配对，seed 作为 bootstrap 聚类单位，10000次重采样。
- 当前 Tactical C 已有完整64局且代码哈希吻合：33胜31平0负；A同为33胜31平0负，逐局非CPU计时字段一致。matched better/worse/same=0/0/64，净胜负差0、95%区间[0,0]。
- 恢复后实际执行 Hunter 当前候选64局：`node scripts/eval_coop_win_strategy.js --opponent hunter --size 1 --arm B --pairs 32 --out runs/coop_win_20261006/hunter_C.json`。A为15胜30平19负，当前候选16胜30平18负；matched better/worse/same=2/1/61，净胜负差+0.03125，95%区间[-0.0625,0.125]。双方自困均18、被困/死亡均34；1v1友伤为零不构成团队安全证据。seat0从7胜17平8负变8胜17平7负，seat1均8胜13平11负。
- `runs/coop_win_20261006/compare_current.js` 实际验证完整性、64个唯一seed/seat、相同协议/依赖哈希，并生成 `matched_current.json`。最终门槛结果 `release=false`：没有明确改善，因此不发布策略。先保存本报告，再仅将两个tracked Bot文件写回 `git show 1c913b7` 的内容，三个新增实验脚本移入已有备份目录；不使用reset，不删除实验产物，不撤回其他改动。
- 早期候选 Tactical B 为25胜39平0负，相比33胜基线退化；其 Hunter B 为22胜30平12负，但不能用另一份候选的结果证明当前代码改善。
- 真实 Chromium 1440×1000与390×844已重跑：`scripts/verify_coop_win_browser.js`。真实页面→HunterAdapter→候选→Sim→Canvas，开局目标分离；第0、8、19 tick放泡，三泡实际连锁且提前引爆2个，己方/队友安全。无JS错误或横向溢出，截图与JSON位于 `runs/coop_win_20261006/browser/`。这是受控行为场景，不是自然对局胜率统计；网络写入禁用。
- 独立只读 Agent `bot_review` 实跑行为测试并审查协议、策略与证据。指出1v1不能证明2v2救援、分区、友伤收益；现有team smoke不完整，不作为通过证据。评估器的 `captures` 读取不存在的 `info.capture`，该列恒零、不可用；夺包只能参考真实每局score，不能宣称无夺包。
- 发布门槛：至少一个对手 matched 95%区间下界>0、另一个无退化，同时无安全回归；行为测试通过不能替代收益证据。未达到门槛只保留报告与本地实验备份，撤回本轮Bot策略，不发布候选。

## 外部边界

可信IP记录持续禁用，未修改SQL/Edge或伪造IP/服务端排名。尚未部署的数据库排名能力仍以明确等待/升级提示呈现。
