# COMBAT-MIX-V1 对拼实验（20261027）

冻结协议 runs/combat_mix_v1/PROTOCOL.json，澄清 protocols/amendment01.json。用户授权网页对比→近似Hunter审计→训练接线→12run。旧暂停队列不恢复、不发布。原A/B/C冻结源码不改，独立collector/worker。

## 配方与预算

三seed2026103051/52/53，各自env2048 pilot u160完整state（参数、Adam、env、aux、RNG）为共同父。四组control、opp_mix、opp_mix_curriculum、opp_mix_curriculum_shaping，各新增160更新/41,943,040环境步/5,120Adam步，固定u320不择峰，总503,316,480新增步。env2048、rollout128、mb8192、epochs1、lr7.5e-5，DAgger/KL=0。每组两更新技术smoke可纳入预算，新进程full-state与next-step验收后接续。

池权重20/25/20/15/10/10%，原rule3/Hunter/hard/current/ring/固定父。原rule3本身是Hunter index3，前两槽策略相同，不当独立bot。记录实际tick/live/new-episode分母。current每更新动态传入；ring四槽每16更新替换，逐更新轮换，完整checkpoint保留。

后两组25%固定行下次真实reset进入map806清砖/hp1/300tick对拼，官方64spawn。保留native trap和原奖励及capture终止，不声称与JS快速协议完全一致。课程行替代该行Bank抽样，整体Bank曝光改变，属于课程包；不传送已携包状态。第四组系数.1、单局cap.5：旧泡所有者/格/出生tick、真实时间先后、无遮挡blast射线、首次新增敌方格覆盖和安全放置，危险完整结算后一次付费。死亡/受控取消pending，同旧事件/敌方格去重，不按存量tick付费。原threat/pressure/safe-resolution塑形保留，存在事件重叠。

## 实测

CPU网页12模型已启动，runs/combat_mix_v1/js_web_compare_v1/OWNER.json。每模型生产JS Hunter难中易各64，自然完整categorical/greedy各64，共3840局。父点已完成，其他按冻结顺序继续。线上release HTTP503，本地selfplay3072/transport1536 checkpoint SHA与本地manifest精确一致，尚未验证线上默认及发布raw/ONNX字节相同，不冒称已完成。

96固定状态（含32明示近接诊断、无人工携包），非法JAX动作0；easy/normal/hard移动一致58/69/34，放泡一致80/92/96。PASS_APPROXIMATE_TRAINING_OPPONENT，仅作近似对手，非JS策略/物理等价。父点JAX各64局kill/game=.547/.250/.141，death=.281/.375/.344，bombs/game=10.03/9.66/9.95。生产JS快速各64局safe_win=3/2/0，death_episode=0/4/0。初态/规则/终止不全相同，不作单因素引擎因果解释。

连泡RED日志 tests/SHAPING_RED.log；初轮focused15通过。真实动作/RNG/logprob/value不变、池划分、遮挡/所有者归因和相关worker回归见 tests/INTEGRATION_GREEN.log。三条control已正式u161+，finite、Actor/Critic非零；暖周期约13秒/262144步，约2万env/s/卡。完整SHA各u*/COMMITTED.json。独立唯一owner/watchdog已adopt，状态STATUS.json；旧C完成前GPU6/7保留。

## settled_kill=0解释

je04_bun_ops.retention_reward按16×surviving_kill累pending，危险settled且未死亡才付。je10_control_credit.parent_reward_events有意排除敌方受控死亡，防首次控制4/残余.25再叠旧16。因此controlled-pop的settled_kill=0是明确去重设计，不能据此断言击杀丢失；直接未受控击杀仍可进入旧pending。本轮不改奖励。此前305/304是事件数/局，可超过1，不能写成305独立胜局。

## 异步验收

u240/u320提交CPU请求：JS Hunter难中易64、自然完整categorical/greedy、JAX holdout304；dev/holdout、JAX/JS、采样/argmax分列。窗口归因→取包→同源交付；死亡/掉包/censor/未知拆分，香蕉持包自吃和胶单列。缺失报null+原因，评估不阻塞GPU。当前尚无本轮能力结论。每run完成验收原子SHA并京ME通知，技术故障保留证据；12run完结汇总，不擅扩预算。

## 自动实测状态

训练状态：RUNNING。
- control_s2026103051：GPU3，u295，finite=True，env/s=20271.72711920266。
- control_s2026103052：GPU4，u293，finite=True，env/s=19890.581369004012。
- control_s2026103053：GPU5，u294，finite=True，env/s=20033.138248211744。
- opp_mix_s2026103051：GPU2，u185，finite=True，env/s=20271.636368337786。
- opp_mix_s2026103052：GPU1，u183，finite=True，env/s=20412.421981019445。
- opp_mix_s2026103053：GPU0，u173，finite=True，env/s=20506.886540164276。
已验收固定run：0/12；完整更新/SHA/吞吐见 STATUS.json 和各run updates.jsonl。
CPU网页完成：2/12；统计见 reports/js_web_compare_20261027.md。
