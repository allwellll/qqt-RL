# COMBAT-MIX 固定终点配对结果（20261028）

结果按冻结u320、三seed配对，不择峰。JAX与JS、categorical与greedy、dev与holdout分别输出；缺数据不按0。完整分子分母与输入SHA见 runs/combat_mix_v1/PAIRED_RESULTS.json。

|比较|已齐配对seed|交付差 vs control|交付差 vs 父|
|---|---|---|---|
|opp_mix:categorical:dev|3/3|0.0|0.0|
|opp_mix:categorical:holdout|3/3|0.0|0.0|
|opp_mix:greedy:dev|3/3|-0.0625|-0.057291666666666664|
|opp_mix:greedy:holdout|3/3|-0.0317982456140351|-0.036184210526315784|
|opp_mix_curriculum:categorical:dev|3/3|0.0|0.0|
|opp_mix_curriculum:categorical:holdout|3/3|-0.002192982456140339|-0.002192982456140339|
|opp_mix_curriculum:greedy:dev|3/3|-0.015625|-0.010416666666666666|
|opp_mix_curriculum:greedy:holdout|3/3|-0.009868421052631563|-0.014254385964912242|
|opp_mix_curriculum_shaping:categorical:dev|3/3|0.0|0.0|
|opp_mix_curriculum_shaping:categorical:holdout|3/3|-0.0010964912280701695|-0.0010964912280701695|
|opp_mix_curriculum_shaping:greedy:dev|3/3|-0.026041666666666668|-0.020833333333333332|
|opp_mix_curriculum_shaping:greedy:holdout|3/3|-0.02302631578947371|-0.027412280701754388|
|legacy_mix:categorical:dev|1/3|0.0|0.0|
|legacy_mix:categorical:holdout|1/3|-0.003289473684210509|-0.003289473684210509|
|legacy_mix:greedy:dev|1/3|-0.546875|-0.546875|
|legacy_mix:greedy:holdout|1/3|-0.5460526315789473|-0.549342105263158|

只有3训练seed，训练seed bootstrap区间证据弱；目前不宣称总体最佳或发布。局级配对原始episodes保留，可重算；未完成条件明确标记。

连泡奖励/候选数量从训练exposure原始字段独立列出，不把塑形收益等同能力。legacy_mix为旧设置包复原，不作单因素归因。
