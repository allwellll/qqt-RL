# 历史预注册设计，须结合实际执行语义阅读

原设计的 retention_v2 合成奖励会被后续 lesson 分支覆盖，未按名义量级普遍执行。下面是保留的设计记录，不是生效奖励或成功机制证明；最终结论以 `../je03_bun_stage2_20261012/bun392_final_acceptance_20261012.md` 为准。

# Bun392 Retention v2 训练设计

实验 ID：`JE03-BUN392-RETENTION-V2-20261011-v1`。本轮由审计提交 `862303f` 直接驱动，目标是先保留共同 Bun392 gen0 的战斗与安全能力，再学习 Bun 任务。两个臂都从同一冻结 `inputs/bun392_gen0.pkl` 独立初始化，动态自对弈关闭。

Control 使用 `retention_v2` 奖励和战斗优先分层课程；Teacher 只额外加入冻结 gen0 actor KL=0.08 与 45% 战斗 replay reset。这样 reward/course 是共同底座，教师保持与 replay 是唯一轮内处理变量；二者都覆盖同一冻结对手组合 `rule_combat=.35 + old=.65`。

`retention_v2` 的安全与击杀项为：自身命中 -2、死亡 -12、自炸 -16、换命 -12；安全放泡 +0.5、危险引爆 -1.5；敌方受伤 +1、存活击杀 +16、因果击杀 +4；终局胜负 +6/-6。任务项严格有界且一次性：进入敌半场 +0.75、进入敌基地 +1、抢包 +2、携包过家线 +1、捕获 +4、追回 +1、回包 +1，掉包 -2；progress 仅 `0.05*clip(progress,-1,1)`，关闭 crate/brick/explore 奖励。所有事件来自环境一次性标志，不能靠重复进度或偷包刷分。

课程权重为 danger_arena .45、full_ambush .20、full_contact .10、combat .10、full .10、carry_home .03、near_steal .02。Teacher 的 45% reset 强制从 danger_arena/combat/combat_moving/full_contact 采样，Control 为 0%。每代遥测保存 reward component、lesson 计数、KL 系数、replay 比例、rollout/update wall time、fresh ticks/s 和 actor/critic update L2。

预算为每臂 4096 代、524288 fresh ticks/gen（2,147,483,648 fresh ticks/臂），保存 0/8/16/32/64/128/192/256/512/768/1024/1536/2048/3072/4096 和每 64 代恢复点。真实网页 Bun Hunter 三档、Tactical、携包/抢包/完整任务和死亡/自炸/换命评估均由独立 CPU owner 异步执行，不构成 GPU 启动或扩训门禁。第一阶段未通过前不启用动态自对弈。

最小验证仅用于故障监控：两个 CPU smoke/recovery 已 finite、非零更新、replica 一致、fresh digest 一致；GPU smoke 在无旧 owner 后并行启动。评估失败或训练异常保留原子证据，不覆盖旧 run。
