# JE10 v38 击杀信用与搬包路线训练总结

**2026-10-19：用户要求安全暂停并集成。r19已在g58原子提交后PAUSED，r20未启动；恢复训练须用户显式指令。** 121,634,816 fresh ticks，固定预算尚余70代。训练与独立评估均退出，8卡compute contexts为0。最后完整能力评估为g32，暂停checkpoint不冒充已评估点。

历史网页综合折中发布点仍为r01 attempt04 g8；本次保留既有模型和多模型Reward/Advantage回放。最近完整轮r18g128与r19g32及所有专项最高点分别报告，不重造综合冠军。

| 固定点 | JS Hunter难/中/易，各64 | Python Tactical v2 /64 | D/S/Drop/Near/Mid/Far/Recovery/Full交付，各16 | 自然拾包/16 | 自伤：Hunter三档;Tactical;八路线 |
|---|---:|---:|---|---:|---|
| r18 g128 | 0/2/7 | 5 | 0/3/12/16/15/12/3/2 | 5 | 0/0/0;0;0/0/0/0/0/0/0/0 |
| r19 g32 | 2/1/7 | 6 | 6/3/11/16/16/12/5/6 | 8 | 0/0/0;4;0/0/0/0/0/0/0/0 |

八路线依序D/S/Drop/Near/Mid/Far/Recovery/Full。Near/Mid/Far为人工预置携包诊断；库存/恢复是控制窗口reset题；只有natural full为原生完整局。Hunter为JS生产bot，Tactical为Python bot/CPU JAX环境，与训练JAX bot分列；不能把各模拟器或任务合成成功率。

## 因果规则与修复

**真实走到敌笼并取到敌包，存在刚才进入的可逆通路。旧“真实取包后无路／动态封路／需要预先开返程通道”结论撤回。** 假不可达来自库存fixture直接把learner放在敌笼组件而没有真实入站；death/sugar/recovery各16题有8题初态home距离INF。人工carrier末状态审计也不能证明真实取包后无路。

训练scene2/3/4/6无事件预携包、课程采样入口、旧lesson1/11及scene-ID custody授权全部移除；固定return replay拒绝缺少hash-bound原生拾取证明的数据，合法2058行保留。16,384固定seed测试learner全部空手，同时真实库存取包、掉包拾取及接触转移链仍成立。r11g18 EXCLUDED_INVALID，不入能力结论。干净父点为用户指定r10g128，后继r12–r19累计61个固定点逐shard哈希核验。

## 实验方法与边界

首控信用以enemy/episode/life/entry稳定身份在受控边沿归因，主信用4、主动爆破残余.25、单局gross上限8；死亡扣回不重开预算。关闭受控死亡继承的16重复击杀，保留非受控战斗16和交付24。测试覆盖施害者、去重、reset、复活、链爆、队友/自伤及原生接线。

路线采用共享连续拓扑势能和阶段目标、set-valued合法native动作oracle、learner-state DAgger/FIFO；合法返程贡献×2。r14补每episode chunk第二不同格且间隔8tick查询，r15返程error加权，r16加入回放暴露遥测，r17实际replay64→128，r18仅认证返程移动免g0/stage1移动KL（ability保持），r19 FIFO2048→4096。数值单因素接续仍受父策略及分布变化影响，不构成同父多seed因果实验。r12因果清理是整合改动。

8设备×1024环境×256tick，每代2,097,152 fresh ticks，128代268,435,456；Adam、EMA-to-raw、RNG、env、FIFO及计数每新run重置。只用最小技术smoke防无效run，不设能力/admission/promotion门禁。固定CPU/JS owner独立跟随g0/4/8/16/32/64/96/128；每点Tactical及三档Hunter各64、八路线各16，固定seed不择优。用户暂停指令已替代此前自动接续策略。

r18自然交付曲线3/3/3/2/4/3/5/2；r19已完成点2/2/2/4/6，掉包12/9/11/6/11。r19g32自然交付相对父点+4，但Tactical仅6/64且自伤4，未证明战斗安全保持或跨场景稳定提升；只有一张原生806，不能称跨地图泛化。

r19g32自然漏斗：窗口→目标13/13局级共现，首oracle3/9、持续1257/1557、推进1298/4312、拾包8/16→交付6/16、恢复25/30；归因首控9/32→最终pop31（主动19）→因果kill9/存活9。局级共现不能当事件因果转换。

早期610个返程输入的合法joint mass由g4 .948回落至g32 .725；float32同向回退，不能仅归因CPU bf16。FIFO延长确实保留到g16，g32早期原记录已过期；复习次数与保持不单调，瞬时监督方向也不持续冲突。证据优先支持研究覆盖、持续拟合与共享目标竞争，尚未证明单一根因。未安装的optimizer快照草稿不发布。

详见[全部61点、专项最高与证据限制](../reports/je10_training_20261019.md)、[冻结计数和哈希](../reports/je10_pause_evidence_20261019.json)、[实验总账](training_experiment_ledger.md)。历史发布选择保留在[发布报告](../reports/je10_best_model_review_20261018.md)。公开代码含已安装训练控制/评估依赖，外部bootstrap、冻结模型及replay输入仍需提供，不含run/checkpoint。线上只读[训练结果页](https://allwellll.github.io/qqt-RL/training.html)与模型回放独立。
