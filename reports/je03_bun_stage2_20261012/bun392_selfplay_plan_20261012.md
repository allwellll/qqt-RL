# 历史冻结计划与终态说明

本计划已执行结束。动态 g3072 仅保留为扩大样本确认候选，g4096未形成稳定优势；完整曲线见 `bun392_final_acceptance_20261012.md`。代码整合后的网页 Hunter 沿用最新 main，历史结果必须按JSON记录的旧源码hash理解，不能与新版本结果直接混合。

# Bun392 保守动态自对弈续训（20261012）

实验 `JE03-BUN392-SELFPLAY12-20261012-v2`；namespace `runs/je03_bun_stage2_20261012/bun392_selfplay12_v2`。用户明确授权阶段2，覆盖旧能力门禁，不代表旧第一阶段完全通过。retention_v2两臂健康训练继续4096，不主动打断；旧CPU447316继续3072/4096。唯一handoff owner在实际TRAINING_COMPLETE、worker/owner退出和显存无其他owner后接管，不等待能力评测或技术READY。首组4卡释放立即启动相应臂；另臂退出后接其4卡，共同输入在首次释放时冻结，不让卡等CPU或另一臂。

## 曲线与起点

教师g0 Hunter0/4/15=19/192、Tactical22/64；g512/768/1024/1536/2048/3072 Hunter总safe17/11/18/14/16/13，Tactical19/12/19/14/21/16。g512 hard/normal均0，g1536及2048 hard/normal均2/4，g3072为2/1。教师的安全/任务优于对照，但full/route仍0，不称完整任务学会或全面收敛；64/档仅快筛。

预声明选择：纳入checkpoint与权威完整marker一致的教师512/768/1024/1536/2048/3072/4096，先最大化相邻点共同hard/normal安全击杀下界，再最大化`min(score_a,score_b)-.25*abs(score_a-score_b)`，score=Hunter safe+Tactical safe+.5*(carry capture+steal)−.75*death−2*self−2*trade；取区间中效用距最好≤3的最早点，否则较强点。当前偏好1536–2048中的2048；首次GPU释放用最新完整曲线重算，不默认4096，迟到CPU不阻塞，起点冻结后不追选winner。v1只CPU原型，偏重总安全选g768，已SUPERSEDED_BEFORE_GPU，更新/恢复证据保留。

同一selected raw Actor和训练独立Critic保留，Adam因新LR/clip/epoch重新初始化，两臂配对seed2026101201与fresh形状。Bun392 g0教师另存，完整Bun24/global24、392×4、三能力/道具/复活保留。

## A/B与奖励

共同LR7.5e-5（旧2e-4）、clip.1（旧.2）、1epoch（旧2）=64实际updates/gen（旧128）、global grad clip.5、entropy.005、g0教师KL.10（旧.08）、GAE.995/.95、45%战斗reset。课程danger_arena .45/full_ambush .20/full_contact .10/combat .10/full .10/carry_home .03/near_steal .02不变。KL=current||g0，全部learner Actor minibatch合法move×3能力联合分布；不约束Critic、不硬锁道具头、不训练对手。

35%响应规则、33%冻结g0、20%冻结Stage1，另两槽各6%。Control两槽恒为Stage1；Dynamic使用recent/history raw actor，每128代完成更新后recent=current、history=previous recent，初始同Stage1。快照版本/digest、RNG/Adam/EMA/环境完整存取；这是响应实际obs的神经网络滚动历史联赛，所有版本checkpoint保留，不是定时巡逻。learner按env偶/奇P0/P1各50%，两臂相同出生reset RNG；角色平衡是相对旧P0-only的共同变化，跨轮收益不能单归因self-play。实际lesson/opponent tick占比记录。

**奖励披露**：现行retention_v2构造后仍被lesson路由覆盖，旧plan的统一低任务reward并未实际生效。当前run不热改，新A/B冻结当前实际奖励语义，不捆绑reward修复；component末列是actual total，前7项是retention候选分项，不能混称已生效。奖励修复后续独立实验。

## 预算、评估和监控

4096代/臂×524288独立fresh ticks=2,147,483,648 ticks/臂；4卡×1024env×128tick，mb4096/card，1epoch。checkpoint0/8/16/32/64/128/256/512/768/1024/1536/2048/3072/4096，每64代完整恢复，每128代快照。早期强点只作为初始化，不冒称winner。

CPU独立CUDA空：真实web/sim.js+bun_hunter_bot.js Hunter64/档300+64残余危险，Tactical64原生300tick独立；carry/steal/route/full各16局idle对手任务诊断，另真实网页item/trap/rescue探针。同seed/出生，有界并发；64只是快筛，正向后扩大样本与独立seed，JAX任务不冒充完整网页能力。

每代监控固定每卡64个fresh样本KL_to_g0/update KL/entropy、预clip梯度范数、Actor漂移、角色与opponent/lesson ticks、非零finite/replica、fresh ticks/s、updates/s、rollout/update和GPU采样。技术检查仅监控NaN/崩溃/损坏，保存failed attempt、最新有效原子状态恢复；CPU非零+fresh restore并行，绝非启动门禁。

连续三个完成点若Hunter safe较起点≤−6或death≥+10或self≥+4，TERM该臂至原子更新边界，owner退出后同4卡立即启动独立recovery namespace：共同冻结起点、冻结对手、LR再减半，全部恶化证据保留。单点不操作、无能力启动或扩训阻塞。自动恢复不选择winner。

启动命令：`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 128-135 <PYTHON> scripts/je03_bun_selfplay_20261012.py handoff`。旧库只借解释器，不写回；现行worktree独立提交、不改main、不push。

## 已安装证据

tmux `je03-bun392-selfplay12-v2-20261012`；launcher shell PID（本地记录），唯一handoff PID（本地记录）；内核flock第二writer实际获取失败。正式plan SHA256 `4e89f524c9e428da48417403f0326cc049d6f5e4a36ed2f216f6c322c4b4f5f5`。状态WAIT_PARENT_ATOMIC_EXIT、assigned空、checks_block_launch=false，候选2048。当前retention worker（本地记录）仍占GPU0-3/4-7，没有被抢占，不伪造self-play已启动。

八项CPU TDD通过（稳定区间/困难档支持/缺失后期点/配对参数/席位/联赛refresh恢复/动作来源选择/占卡拒绝）。独立CPU monitor_v1两臂g1/g2 Actor及Critic非零finite，fresh-process digest均`334b331cf283cc9aa2cf089d0a12c9c4bb58a2edca1cee6bf7848cdcd8c455a0`；测试refresh=1明确不同于正式128，版本1recent/2history相等，已证明滚动恢复。两臂小样本动作未显著分流（digest相同），不冒称该tiny smoke证明实际12%覆盖。另真实network诊断在同物理状态对比初始和实际更新后快照，最大动作概率差.930254、KL(new||old)=1.967522，CUDA空、原生Bun；这是功能诊断，不是正式shape稳定性结论。最终代码快照另在cpu_monitor_v2重跑并行验证，正式调度不读取其passed作为门禁。

最终cpu_monitor_v2两臂nonzero/finite及fresh-process全状态恢复均已通过，全部sealed source hash与最终源码一致，进程自然退出；真实完整证据另存`selfplay_cpu_validation_20261012.json`，不是阻塞门禁。首次GPU释放时handoff将生成正式共同初始化hash、实际GPU更新与吞吐markers，当前绝不伪造已占卡。
