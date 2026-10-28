# PPO batch 扩容与长程观察（20261025）

实验 `PPO-SCALE-LONGRUN-20261025-R1` 已实际进入正式训练。短程没有显著提升不阻止长训，短点能力是观察指标；数值/恢复故障、可靠安全退化与用户暂停单独处理。本报告持续更新，不代表六 run 已完成。

## 冻结设计与起点

共同干净父为 r19 causal-clean g32，policy SHA `a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95`，原完整 state SHA `2099e2288b663e1a6007eeb2d4932e3324ab45987b87a9a4d194f25aaf87cae5`；提取的共同 params/Adam input SHA `61829b5a4a94ff1c468cb2a8ab055b6412415262f89ade97a75376dc32315b8e`。选择干净父，未择峰继承退化 DAgger 候选。R16 PPO holdout 17/17/12 对父15（各64局）只作背景证据，不能据跨零 CI 宣称等效安全或已获稳定提升。

两臂均为 PPO/value/entropy，DAgger=0、combat KL=0。奖励、网络、输入、mask、固定 JAX rule3 对手、真实取包 Bank 与自然 reset 配方保持；没有人工预置携包训练。Bank 保持旧单图/P0/all-far 局限。名义 reset 混合 natural 75% / real Bank 25%，实际 tick 暴露另记。current/ring/fixed 为 0/0/100%，不称动态联赛。

| 配置 | base | scaled |
|---|---:|---:|
| GPU | 8 | 8 |
| env/device | 8 | 16 |
| rollout 时间步 | 128 | 128 |
| global rollout batch | 8,192 | 16,384 |
| global/local minibatch | 256/32 | 512/64 |
| epochs | 1 | 1 |
| lr | 7.5e-5 | 7.5e-5 |
| outer updates/run | 128 | 64 |
| 环境步/run | 1,048,576 | 1,048,576 |
| optimizer steps/run | 4,096 | 2,048 |

gamma=.995、GAE lambda=.95、clip=.1、value=.5、entropy=.005、梯度裁剪=.5。配对 seed `2026102751/52/53`，六 run 共6,291,456环境步/18,432 optimizer steps，单样本每更新采用一次。同环境步比较，优化步数不相等，不能称等优化步实验。

首 seed 各65,536环境步为有界吞吐 prefix（base u8、scaled u4），直接计入长预算；之后按既定队列恢复完成六 run，不以短点能力筛选配置。声明点 base u0/1/8/16/32/64/96/128，scaled u0/1/4/8/16/32/48/64；u1仅启动诊断，跨臂能力比较按环境步对齐。

## 技术证据与真实执行

两臂 smoke 均 finite，真实 Actor/Critic/ability 更新非零。base ΔL2=.0838971/.1528114/.00119553，scaled=.0830937/.1505534/.00121927。

首次接线有技术错误：在完整 fresh 恢复比较之前启动了 base u1。发现后在 u1 原子保存边界暂停，保留8192正式环境步，不重跑、不抹去。随后另一个只验证加载后可更新的 precheck 同样不足，已保留为 `LOAD_ONLY_RESTORE_PRECHECK.json`。最终由两个独立新进程对比两 batch 的完整下一步 env/aux/RNG、batch、history、loss、params、Adam，所有摘要 exact；`FRESH_RESTORE.json` 与 `TECHNICAL_ACCEPTANCE.json` 是当前验收依据，旧 smoke 的 pending 字段已被后者明确覆盖。

正式恢复从原子 u1 继续，u2 state SHA `3bd0bfaf2c4c656e72dd40ec51796f36b08505f4df759a4bfe0fffc2c733c603`；Actor/Critic ΔL2=.8721883/.7612747，ability=.0145871，累计16,384环境步。base u8已原子完成65,536环境步；u4–8稳态约1484–1543环境步/秒，u2含启动编译约104步/秒。活跃8卡快照约70–80%，编译/保存间隙有0%，不以显存占用冒充有效利用率。扩容 batch 编译与稳态另记，不能将 base 吞吐代入 scaled。

## 动作与测量边界

共同父真实 categorical collector 审计执行65,536步、0优化步：释放972、greedy代理890、老师标记机会427、contact labels0、香蕉库存21,820 ticks、持包12,760 ticks、死亡20；natural56,596 / real Bank8,940。它证明释放按钮可采样，不证明完整有益香蕉/胶链。协议原 sampling_audit 写8192，实际是8×8192，已在测量 amendment 勘误；胶库存字段0是未采集占位，不能作为零库存观测。

v2只有接触至滑行片段，漏准备成本、覆盖旧事件且无终局 censor。被动v3加入放置事件ID、准备成本、重叠/终局 censor，物理/RNG逐帧 parity通过，核心 evaluator 已采用该冻结版本。进一步真实取包微测试发现 v3 会把滑行中交付后的 cargo 清空误作丢包：v4单独锁存真实交付身份，保留原 v3 证据，其 chain_harm/beneficial 暂不用于收益结论。v4通过真实取包→放置→离开→自吃→完整滑行/交付、重叠不覆盖、终局 censor 和无物理/RNG变化测试。

v4同固定 dev64自然对局按 production JS categorical/greedy分别运行，原始 chain events独立保存在 `natural_action_audit_v4`。这与 JAX training categorical、生产 JS greedy/Tactical/Hunter 都分开报告。观察到的全链距离净值仍只是 proxy，不冒充经验 Q 或真实优势。无道具完美门槛，无新增老师/奖励干预。

历史训练行 `carry_inventory_ticks` 实为全阶段香蕉库存；需与 carry 交集区分，不能直接算持包道具机会率。`COMMITTED.optimizer_steps` 是本更新步数，累计数看 metrics.cumulative_optimizer_steps。GAE、loss数组、奖励分量保留原始数据，不将奖励/Value替代能力或优势。

## 异步、恢复与接续

唯一 GPU owner `scripts.ppo_scale_longrun_run` 持有 namespace 锁，逐更新原子保存完整状态与 SHA。唯一固定 CPU/JS owner `scripts.ppo_scale_longrun_cpu` 已实际启动并消费父点/u1/u8，声明点各执行自然64、Python Tactical v2 64、真实 JS Hunter难中易各64；最终自然独立 holdout64，不用holdout调参。CPU评估与有界v4动作审计不控制GPU。

每个run预算终态先接续声明队列，再消费完整曲线/尾部/最差点，不择峰。六run结束后只依据证据提出并预注册新的有界PPO问题；无合理后继则主动报告，不恢复旧O5/continuous/储备，不默认追加DAgger。故障与终态写持久通知并调用已授权CLI通知。

复现入口：`.venv/bin/python -m scripts.ppo_scale_longrun_run`；CPU：`.venv/bin/python -m scripts.ppo_scale_longrun_cpu`；有界动作审计：`.venv/bin/python -m scripts.ppo_scale_natural_audit runs/ppo_scale_20261025/r1_longrun/formal/no_update/u000`。启动前先确认唯一锁和 scoped 授权，不与现有 owner 重复运行。安全暂停为 namespace 内 `USER_STOP.json`，worker在下一更新前检查，保留最近原子点；CPU另用 `CPU_STOP.json`。

focused Python与Node验证、原始统计、协议/源码/恢复hash均保存在实验namespace。无push、无发布。

## 等步数 pilot 已完成及首批异步证据

base u8与scaled u4均已完成各65,536环境步并自动接续长预算。base u4–8迭代内稳态中位1,521.6步/秒，scaled u3–4为2,622.0步/秒（约1.72×）；计入每次原子保存的相邻COMMITTED wall-clock中位分别1,146.2与2,067.0步/秒（约1.80×）。尾部样本数5/2较小，不外推整段；scaled u1/u2分别274.1/251.8秒，含首次签名编译，base恢复也发生重新编译，不能把steady速度当端到端速度。u3虽startup_compile_included=false也可能含编译，该flag不是完整编译检测。

共同父v4同dev64真实自然对局：categorical交付26/64、死亡对局36/64（43死亡事件）；greedy交付12/64，动作选择协议严格分开。categorical持包香蕉库存8400 ticks/首槽合法5780，实际释放36、自有接触7、完整settlement5、censor31，其中1条扣全准备成本后距离净值为正；持包胶库存9962/首槽合法5405、释放23、归因敌接触14、settlement3/censor20，其中1条满足追击减速且向家推进的观察条件。它是稀少真实支持，不等同稳定高技能，不证明因果Q。chain_complete还包含提前carrier-harm结算，不能全部解释为成功完成；原始reason和owned_contact必须同时查看。greedy自然主要目标任务与道具链不足仍要跟踪。

核心固定评估早期dev：父交付12/64，base u1/u8为19/19，scaled u1为8；Tactical安全胜父/base u1/base u8/scaled u1/scaled u4分别10/10/9/13/7，JS Hunter各难度64独立保留。u1并非跨臂等环境步，不据单seed早期波动择峰、晋升或停训；最终比较须使用全部配对seed、对齐点与独立保留集。

统计补正：smoke实际collector长度8，base512/scaled1024环境步且各2优化步；旧smoke元数据误乘长rollout128/错误分母，已写TECHNICAL_SMOKE_ACCOUNTING_CORRECTION，仅更正统计，不重启smoke或影响正式预算。训练EV未单独采集，现有value loss与GAE方差不能冒充EV；该遥测缺口披露，不作为停训门槛。

当前唯一GPU owner、worker、CPU固定评估owner已分别核验；只读monitor另写LIVE_SUPERVISION和milestone收据，不具备启动GPU/旧队列能力。所有固定run终态由原owner主动通知并立即接续声明run，完整实验就绪另有持久通知。

### 归因复查与撤回

v4所谓1条独立完整正净值香蕉链的结论撤回：原始seed2026102512的第二次放置发生在第一条香蕉滑行持续时，准备期混入旧滑行进展。v5隔离测量保留先前活跃链到真正结束/新接触，并在新放置已有本人移动效果时排除正收益标签；真实取包/交付/准备成本/重叠/终局/无状态变化测试通过，固定父点同64 categorical/greedy补审计正在执行，不改变PPO。v4与v3原始数据保留，不能将其1条正值当独立动作收益证据。胶的1条减速推进也只称观察proxy，可能与先前敌方减速重叠，不证明独立因果收益；不要求道具完美，不追加老师，也不停止健康长训。

自动比较报告 `reports/ppo_scale_longrun_comparison_20261025.md` 消费固定保存点，完整漏斗保存在LIVE_COMPARISON；生成与监控均不控制GPU。

v5同固定128局补审计完成：所有非道具episode指标精确等于v4，两个动作模式各固定首seed的完整诊断trace（动作/状态hash/事件）精确一致，其余126局不声称有完整逐帧trace。v5保留旧滑行到真正结束后，两个独立自然来源局seed2026102480/2026102502记录完整banana正距离净值+1.5601/+5.1202格，均为真实持包、放置时无既有本人移动效果，含24/13 tick准备成本，未损失原包。共同父categorical持包36释放/7接触/2条正净值链，greedy7释放/1接触/0正净值。仍只是观察净位移proxy，不等同因果Q或稳定技能；胶1条减速推进可能重叠旧敌方减速，继续保留归因限制。v4独立完整链结论仍撤回，不能把两个版本相加。原始128局与CLOSED_LOOP_PARITY收据保存。

## 首个完整长预算终态与接续

`ppo_base_batch_s2026102751/u128` 已完整完成1,048,576环境步/4096优化步，128/128正式更新finite、Actor/Critic/ability均非零；state SHA `06d47ce4f3e28c9fcf3421ce6df9bba4bba95f292c0caac54f1c1883467b0b61`，policy SHA `f7380f576632ccc51790835d7ecf1a8939cfd5fcb66e5d7318c12160e1399037`，两者独立复核。未等待CPU即接续同seed scaled，现已从u4原子点恢复并提交u5：累计81920环境步、finite，Actor/Critic ΔL2=.6215814/1.1142536、ability=.01267944。其后继续scaled至u64及另两配对seed四run，不重跑base或pilot、不恢复旧队列。u128固定评估由唯一CPU owner独立消费，尚不宣称最终能力改善。FIRST_LONGRUN_TERMINAL_VERIFIED与FIRST_LONGRUN_SUCCESSOR_UPDATED保存验收证据。

### 首完整run最终dev下降，未晋升

base seed2751 u128的固定dev自然交付2/64，父12/64，固定尾部u64/96/128为13/7/2（均值7.33），不择早期u32=24峰值。categorical同dev64交付15，父26，说明不只是greedy阈值差异；greedy差距更大。死亡事件25→10不能单独称安全改善，Tactical炸弹/局4.125→2.7344，提示行动减少。

机会分母：自然窗口84→47，归因控制34→16；合法窗口库存拾取21→5，按窗口归一化.25→.1064；同链窗口交付12/21=.5714→1/5=.20。故下降同时涉及机会创造、库存转化及拾取后的交付，不能只怪道具或无路。训练params/Adam/env/aux全部finite，原始Adam count实际增加4096，技术上有效；不是NaN/OOM/坏恢复。当前只一个seed的base终态，不宣称重复seed可靠退化，所有固定run继续。

按证据排序的待验证假设为累计PPO漂移×固定rule3训练暴露/自然JS迁移差异、critic/credit校准、batch方差。EV未采集，不能把value loss/GAE方差当EV或断言PPO噪声。若后续配对seed重复确认漂移，下一最小问题优先PPO学习率单变量对照，不默认追加老师；当前奖励/对手/课程/网络均不改，后继只有独立冻结协议后才执行。完整诊断收据FIRST_LONGRUN_DEV_DIAGNOSIS保存。
