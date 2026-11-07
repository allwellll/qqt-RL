# 新物理网页可选模型筛选（20261107）

## 首批网页P1组合对照

每组合40局（五对手各8），仅为有限快筛；全部使用生产bun-native-v3-items。默认不改。

|模型+τ|胜/负/平|交付|因果杀|自死|
|---|---|---:|---:|---:|
|G_frozen+τ0.3|37/3/0|37|20|4|
|L2_S05g_u019302+τ0.3|33/4/3|31|22|6|
|L2_DUEL_u019902+τ0.3|40/0/0|40|22|9|
|L2_SP_u023302+τ0.3|39/0/1|39|24|1|

已冻结七臂最新COMMITTED与G_frozen、健康u019302；所有比赛使用生产bun-native-v3-items引擎。旧物理评测不复用为新证据。模型固定网页P1（engine seat1），分别argmax/τ0.3，对照G_frozen τ0.3；不改训练、不push、不调用CC。协议与逐局RESULT/review保存在 `runs/items_candidates_20261107/v1`。专项长处可作为可选，不要求全面更强，不改变默认模型。

状态：216局七臂筛查、320局独立确认、320局五对手网页P1快筛全部完成；已完成856局逐项审计。交付契约状态 READY。

## smoke 实际完成

|模型/方式/任务/分集|局数|胜/负/平|score|交付|因果杀|自死|携包失败|可行动静止%/最长|c tick|喷泉串|
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
|L2_DUEL_u019902/argmax/bun.coop_hunter@hard/web|1|1/0/0|1.000|2|1|1|0|29.94/29|0|2.000|
|L2_DUEL_u019902/tau0.3/duel/dev|1|1/0/0|1.000|0|0|0|0|9.43/8|2|1.000|
|L2_DUEL_u019902/argmax/duel/dev|1|1/0/0|1.000|0|1|0|0|12.94/10|0|2.000|
|L2_DUEL_u019902/tau0.3/bun.coop_hunter@hard/web|1|1/0/0|1.000|3|4|3|0|16.75/11|45|2.000|

## 冻结checkpoint与实际运行

|臂|冻结COMMITTED|完整policy SHA|
|---|---|---|
|L2_DUEL|runs/cc_train_20261105/training/items_v3_20261107_l2_duel/L2_DUEL/u019902|c2058a0f2dbf0bb2ecd4ea9b17be975a2bd51caf4e47c1713d17c9c6947f0a70|
|L2_DUEL75|runs/cc_train_20261105/training/items_v3_20261107_l2_duel75/L2_DUEL75/u019902|b47f55f9dbb891f17a6a11e2241f34f50766540c03816a7588ee93168a91f655|
|L2_DUEL25|runs/cc_train_20261105/training/items_v3_20261107_l2_duel25/L2_DUEL25/u019862|16be6a94f7414484d220bc9e9f82a22a0eadf4876c221a073f10687e3a66bd85|
|L2_DUEL_SYNC2|runs/cc_train_20261105/training/items_v3_20261107_l2_duel_sync2/L2_DUEL_SYNC2/u020022|7afcebb0ddb4e93d0a5fae1152d8f69743bf00699bb3c57e63f00c7458a21e32|
|L2_FTN2|runs/cc_train_20261105/training/items_v3_20261107_l2_ftn2/L2_FTN2/u021262|5a666c77be385ec8b8e30fe966d4d9aa6411949cca887cd4cd1797cab567eaf9|
|L2_ATK|runs/cc_train_20261105/training/items_v3_20261107_l2_atk/L2_ATK/u024582|3b8bcf396bc476c7c4157216874d783a602191b16520b094864e82fa3d88e4c0|
|L2_SP|runs/cc_train_20261105/training/items_v3_20261107_l2_sp/L2_SP/u023302|39b0dea7327221891bdd4cfba93ab917c897aa86fd78d83401241ad945a982cf|

四局接线smoke均已通过实际生产JS独立重放，不能据此宣称能力通过。评测核从被旧队列占用的24–31调度到36–43，最多8个进程，复用已完成精确缓存；只替换本次CPU evaluator且等待其子局完成，八卡训练未动。独立调度回执 `runs/items_candidates_20261107/v1/CPU_SCHEDULING.json`，物理/模型/seed协议不改。

## screen 实际完成

|模型/方式/任务/分集|局数|胜/负/平|score|交付|因果杀|自死|携包失败|可行动静止%/最长|c tick|喷泉串|
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
|L2_SP_u023302/argmax/strong_full/screen|4|2/1/1|0.625|3|1|1|0|45.97/75|131|1.500|
|L2_DUEL_u019902/argmax/strong_full/screen|4|2/2/0|0.500|2|1|2|0|33.41/31|66|3.000|
|L2_DUEL75_u019902/argmax/strong_full/screen|4|3/1/0|0.750|3|1|2|2|14.40/22|3|2.000|
|L2_ATK_u024582/argmax/strong_full/screen|4|0/4/0|0.000|0|0|3|1|61.05/40|30|3.000|
|L2_DUEL25_u019862/argmax/strong_full/screen|4|1/3/0|0.250|1|1|1|1|21.06/28|40|2.750|
|L2_S05g_u019302/argmax/strong_full/screen|4|1/2/1|0.375|1|2|0|1|50.83/64|37|1.500|
|G_frozen/argmax/strong_full/screen|4|1/1/2|0.500|1|2|1|1|28.30/24|4|1.500|
|L2_DUEL_SYNC2_u020022/argmax/strong_full/screen|4|2/2/0|0.500|2|2|3|0|53.41/101|120|3.000|
|L2_FTN2_u021262/argmax/strong_full/screen|4|0/3/1|0.125|0|2|4|0|49.93/71|74|2.000|
|L2_DUEL25_u019862/tau0.3/strong_full/screen|4|3/1/0|0.750|4|3|1|1|13.69/19|47|2.500|
|L2_DUEL75_u019902/tau0.3/strong_full/screen|4|2/2/0|0.500|2|2|3|1|17.45/22|24|2.750|
|L2_DUEL_SYNC2_u020022/tau0.3/strong_full/screen|4|1/3/0|0.250|1|1|0|1|37.03/28|94|1.500|
|L2_DUEL_u019902/tau0.3/strong_full/screen|4|3/1/0|0.750|3|3|2|0|29.81/21|268|2.000|
|L2_ATK_u024582/tau0.3/strong_full/screen|4|1/2/1|0.375|2|2|2|0|34.47/26|109|2.500|
|L2_FTN2_u021262/tau0.3/strong_full/screen|4|1/2/1|0.375|3|3|0|1|36.33/17|29|1.750|
|L2_SP_u023302/tau0.3/strong_full/screen|4|3/1/0|0.750|4|2|1|0|31.28/46|101|1.500|
|G_frozen/tau0.3/strong_full/screen|4|2/2/0|0.500|2|2|0|1|16.96/12|296|1.750|
|L2_S05g_u019302/tau0.3/strong_full/screen|4|0/3/1|0.125|1|3|2|0|37.56/30|56|2.000|
|L2_DUEL25_u019862/argmax/duel/screen|8|3/0/5|0.688|0|3|0|0|8.31/20|9|1.375|
|L2_DUEL75_u019902/argmax/duel/screen|8|7/0/1|0.938|0|6|0|0|5.52/7|7|1.875|
|L2_FTN2_u021262/argmax/duel/screen|8|0/0/8|0.500|0|0|0|0|24.00/23|0|1.000|
|L2_DUEL_SYNC2_u020022/argmax/duel/screen|8|7/0/1|0.938|0|7|0|0|12.16/19|26|1.500|
|L2_DUEL_u019902/argmax/duel/screen|8|7/0/1|0.938|0|7|0|0|9.77/17|9|1.375|
|L2_SP_u023302/argmax/duel/screen|8|0/1/7|0.438|0|0|0|0|16.29/28|4|1.000|
|L2_ATK_u024582/argmax/duel/screen|8|1/0/7|0.562|0|0|0|0|39.06/23|5|1.125|
|G_frozen/argmax/duel/screen|8|1/2/5|0.438|0|0|0|0|16.43/20|41|1.000|
|L2_S05g_u019302/argmax/duel/screen|8|0/3/5|0.312|0|0|2|0|24.73/25|6|0.875|
|L2_DUEL_u019902/tau0.3/duel/screen|8|6/0/2|0.875|0|5|1|0|8.90/15|4|1.625|
|L2_DUEL25_u019862/tau0.3/duel/screen|8|6/0/2|0.875|0|4|0|0|6.99/10|19|1.250|
|L2_DUEL75_u019902/tau0.3/duel/screen|8|6/0/2|0.875|0|4|0|0|7.07/12|7|1.500|
|L2_ATK_u024582/tau0.3/duel/screen|8|0/1/7|0.438|0|0|0|0|37.42/17|8|1.125|
|L2_DUEL_SYNC2_u020022/tau0.3/duel/screen|8|4/1/3|0.688|0|4|1|0|7.05/12|22|1.500|
|L2_SP_u023302/tau0.3/duel/screen|8|0/1/7|0.438|0|0|1|0|14.79/19|2|1.125|
|G_frozen/tau0.3/duel/screen|8|0/1/7|0.438|0|0|0|0|13.93/31|46|1.250|
|L2_S05g_u019302/tau0.3/duel/screen|8|1/2/5|0.438|0|1|2|0|20.43/23|1|1.625|
|L2_FTN2_u021262/tau0.3/duel/screen|8|1/0/7|0.562|0|1|0|0|23.99/12|2|1.000|

## confirm 实际完成

|模型/方式/任务/分集|局数|胜/负/平|score|交付|因果杀|自死|携包失败|可行动静止%/最长|c tick|喷泉串|
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
|L2_SP_u023302/argmax/duel/dev|16|3/0/13|0.594|0|2|0|0|22.34/72|22|1.312|
|L2_DUEL_u019902/argmax/duel/dev|16|8/0/8|0.750|0|6|0|0|6.83/14|22|1.188|
|L2_S05g_u019302/argmax/duel/dev|16|2/8/6|0.312|0|1|6|0|34.21/30|6|1.375|
|G_frozen/argmax/duel/dev|16|6/2/8|0.625|0|2|0|0|12.49/24|58|0.750|
|L2_SP_u023302/tau0.3/duel/dev|16|2/4/10|0.438|0|2|2|0|20.26/17|18|1.188|
|L2_DUEL_u019902/tau0.3/duel/dev|16|10/0/6|0.812|0|8|0|0|9.26/16|56|1.000|
|L2_S05g_u019302/tau0.3/duel/dev|16|1/4/11|0.406|0|1|2|0|27.29/22|2|1.312|
|G_frozen/tau0.3/duel/dev|16|4/2/10|0.562|0|1|1|0|15.81/14|112|1.000|
|L2_DUEL_u019902/argmax/duel/holdout|16|11/1/4|0.812|0|9|2|0|12.18/26|101|1.062|
|L2_SP_u023302/argmax/duel/holdout|16|1/2/13|0.469|0|1|1|0|23.35/26|18|1.250|
|L2_S05g_u019302/argmax/duel/holdout|16|3/3/10|0.500|0|2|3|0|32.72/33|8|1.312|
|G_frozen/argmax/duel/holdout|16|1/1/14|0.500|0|1|1|0|13.46/22|81|0.938|
|G_frozen/tau0.3/duel/holdout|16|1/2/13|0.469|0|0|0|0|14.12/15|94|1.000|
|L2_DUEL_u019902/tau0.3/duel/holdout|16|10/0/6|0.812|0|7|0|0|9.70/14|64|1.188|
|L2_SP_u023302/tau0.3/duel/holdout|16|4/1/11|0.594|0|2|1|0|20.67/18|28|1.125|
|L2_S05g_u019302/tau0.3/duel/holdout|16|1/5/10|0.375|0|1|3|0|26.81/20|9|1.562|
|L2_SP_u023302/argmax/strong_full/holdout|8|3/3/2|0.500|3|1|0|0|30.12/68|329|1.500|
|L2_DUEL_u019902/argmax/strong_full/holdout|8|4/4/0|0.500|4|1|3|1|37.47/318|995|1.750|
|G_frozen/argmax/strong_full/holdout|8|4/3/1|0.562|4|1|0|2|9.95/140|46|1.500|
|L2_S05g_u019302/argmax/strong_full/holdout|8|3/3/2|0.500|3|8|2|2|47.93/32|214|2.125|
|L2_DUEL_u019902/tau0.3/strong_full/holdout|8|4/3/1|0.562|5|4|1|0|23.94/26|124|2.000|
|L2_SP_u023302/tau0.3/strong_full/holdout|8|4/4/0|0.500|4|2|1|1|34.08/30|101|1.875|
|G_frozen/tau0.3/strong_full/holdout|8|3/5/0|0.375|4|2|1|2|18.83/23|206|1.500|
|L2_S05g_u019302/tau0.3/strong_full/holdout|8|4/4/0|0.500|4|3|1|1|34.40/24|136|1.875|

网页完整局快筛阶段使用12个空闲物理评估核36–47，实际最多12个CPU子进程；共享逐核锁仍生效。该纯调度修订覆盖初始8进程元数据，原比赛/seed/模型/物理与所有精确缓存不改。调度源码与收据 `scripts/cx_items_candidate_cpu_v3.py`、`runs/items_candidates_20261107/v1/CPU_SCHEDULING_V3.json`；训练CPU绑定保持。

## web 实际完成

|模型/方式/任务/分集|局数|胜/负/平|score|交付|因果杀|自死|携包失败|可行动静止%/最长|c tick|喷泉串|
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
|L2_SP_u023302/argmax/bun.tactical_v2/web|8|8/0/0|1.000|8|0|0|0|5.80/13|18|0.625|
|L2_DUEL_u019902/argmax/bun.tactical_v2/web|8|8/0/0|1.000|4|3|0|4|12.27/56|35|0.750|
|G_frozen/argmax/bun.tactical_v2/web|8|8/0/0|1.000|6|1|0|0|20.37/49|597|0.250|
|L2_DUEL_u019902/tau0.3/bun.tactical_v2/web|8|8/0/0|1.000|4|2|1|0|7.60/14|83|1.000|
|L2_S05g_u019302/argmax/bun.tactical_v2/web|8|8/0/0|1.000|6|2|0|0|16.76/394|123|0.625|
|G_frozen/tau0.3/bun.tactical_v2/web|8|8/0/0|1.000|8|1|0|0|11.95/15|212|0.375|
|L2_SP_u023302/tau0.3/bun.tactical_v2/web|8|8/0/0|1.000|7|0|0|0|16.28/15|23|0.500|
|L2_S05g_u019302/tau0.3/bun.tactical_v2/web|8|8/0/0|1.000|5|0|1|2|16.22/21|67|0.750|
|L2_DUEL_u019902/argmax/bun.hunter@hard/web|8|6/0/2|0.875|6|3|0|0|30.65/111|113|1.750|
|L2_S05g_u019302/argmax/bun.hunter@hard/web|8|6/1/1|0.812|6|2|1|0|47.87/211|162|1.875|
|G_frozen/argmax/bun.hunter@hard/web|8|6/1/1|0.812|5|0|0|1|23.63/70|261|1.250|
|L2_SP_u023302/argmax/bun.hunter@hard/web|8|5/1/2|0.750|6|5|1|1|39.74/1025|197|2.125|
|G_frozen/tau0.3/bun.hunter@hard/web|8|7/1/0|0.875|7|1|1|0|25.82/15|134|1.125|
|L2_DUEL_u019902/tau0.3/bun.hunter@hard/web|8|8/0/0|1.000|8|4|2|0|29.35/21|73|2.125|
|L2_SP_u023302/tau0.3/bun.hunter@hard/web|8|8/0/0|1.000|10|7|0|0|23.65/21|91|1.875|
|L2_S05g_u019302/tau0.3/bun.hunter@hard/web|8|4/2/2|0.625|4|2|2|1|32.68/22|165|1.500|
|L2_DUEL_u019902/argmax/bun.hunter@normal/web|8|5/2/1|0.688|5|1|2|0|39.36/101|79|1.875|
|L2_SP_u023302/argmax/bun.hunter@normal/web|8|8/0/0|1.000|7|8|0|1|41.59/37|41|2.000|
|G_frozen/argmax/bun.hunter@normal/web|8|8/0/0|1.000|7|7|0|0|21.64/43|88|1.500|
|L2_S05g_u019302/argmax/bun.hunter@normal/web|8|7/0/1|0.938|7|7|0|0|44.93/26|283|1.750|
|G_frozen/tau0.3/bun.hunter@normal/web|8|7/1/0|0.875|7|7|1|0|22.17/29|65|1.375|
|L2_DUEL_u019902/tau0.3/bun.hunter@normal/web|8|8/0/0|1.000|10|5|2|0|26.49/21|88|2.625|
|L2_SP_u023302/tau0.3/bun.hunter@normal/web|8|8/0/0|1.000|7|7|0|0|26.41/22|53|2.125|
|L2_S05g_u019302/tau0.3/bun.hunter@normal/web|8|7/0/1|0.938|7|8|1|1|26.25/20|143|1.750|
|L2_DUEL_u019902/argmax/bun.hunter@easy/web|8|8/0/0|1.000|8|5|0|0|23.17/26|27|1.375|
|L2_SP_u023302/argmax/bun.hunter@easy/web|8|8/0/0|1.000|8|5|0|0|31.63/330|32|1.500|
|G_frozen/argmax/bun.hunter@easy/web|8|8/0/0|1.000|8|8|0|0|23.19/25|19|1.000|
|L2_S05g_u019302/argmax/bun.hunter@easy/web|8|8/0/0|1.000|8|7|0|0|31.48/34|3|1.250|
|L2_SP_u023302/tau0.3/bun.hunter@easy/web|8|8/0/0|1.000|8|6|0|0|25.50/24|48|1.500|
|L2_DUEL_u019902/tau0.3/bun.hunter@easy/web|8|8/0/0|1.000|8|6|0|0|21.64/14|35|1.250|
|G_frozen/tau0.3/bun.hunter@easy/web|8|8/0/0|1.000|8|8|1|0|20.61/21|94|1.000|
|L2_S05g_u019302/tau0.3/bun.hunter@easy/web|8|8/0/0|1.000|8|8|1|0|23.38/21|90|1.250|
|G_frozen/argmax/bun.coop_hunter@hard/web|8|8/0/0|1.000|8|1|0|0|30.82/24|83|1.250|
|L2_S05g_u019302/argmax/bun.coop_hunter@hard/web|8|8/0/0|1.000|9|3|0|0|38.45/36|155|1.875|
|L2_DUEL_u019902/argmax/bun.coop_hunter@hard/web|8|8/0/0|1.000|9|4|1|1|28.47/29|43|1.750|
|L2_SP_u023302/argmax/bun.coop_hunter@hard/web|8|7/1/0|0.875|7|4|0|1|40.88/346|103|2.375|
|L2_SP_u023302/tau0.3/bun.coop_hunter@hard/web|8|7/0/1|0.938|7|4|1|0|32.82/30|326|1.875|
|L2_S05g_u019302/tau0.3/bun.coop_hunter@hard/web|8|6/2/0|0.750|7|4|1|1|29.77/19|118|2.125|
|G_frozen/tau0.3/bun.coop_hunter@hard/web|8|7/1/0|0.875|7|3|1|0|18.01/18|52|1.125|
|L2_DUEL_u019902/tau0.3/bun.coop_hunter@hard/web|8|8/0/0|1.000|10|5|4|0|26.09/26|136|1.875|

## 七臂专项代表筛查

|方向|筛查代表|因果杀合计|交付合计|喷泉串分组均值之和|可行动静止%分组之和|
|---|---|---:|---:|---:|---:|
|击杀|L2_DUEL_u019902|16|5|8.000|81.88|
|进攻流畅|L2_DUEL75_u019902|13|5|8.125|44.45|
|喷泉|L2_DUEL75_u019902|13|5|8.125|44.45|
|运包|L2_SP_u023302|3|7|5.125|108.32|

筛查各组局数相同，上表仅用于候选排序；短局和完整局分组和不是生产线上统一得分。首批只对两个筛查候选追加独立dev/holdout与网页快筛，其他代表未被冒充独立通过。

## 可选交付结论

所有逐局RESULT、生产物理SHA、模型SHA及独立重放hash已审计；argmax与τ0.3分别保留，失败不删。首批网页每对手每方式8局为快筛，不冒称历史64局已完成。独立专项dev与holdout各16局/方式，完整局holdout8局/方式；仅P1，有选择偏差及小样本限制。

指标口径：因果杀按原只读控制/死亡事件链统计，自死导致的胜场不冒充因果击杀。可行动静止=存活、未受控且真实坐标不变；c类为现有危险场下有安全推进证据的保守分类，不是全未来安全证明。喷泉串为有效跨tick批次、间隔≤24tick的最长串均值；严格串另保留在JSON的strict_fountain_streak，不按放泡量代替。人工专项300tick或首死亡终止、空手无运包，是任务设置而非网页完整局规则；网页和strong_full均为生产2400tick完整抢包。

可选组合 **L2_DUEL_u019902 tau0.3**；checkpoint `runs/cc_train_20261105/training/items_v3_20261107_l2_duel/L2_DUEL/u019902`，policy SHA `c2058a0f2dbf0bb2ecd4ea9b17be975a2bd51caf4e47c1713d17c9c6947f0a70`。

- 长处：独立dev/holdout人工短局真实因果击杀均高于G_frozen τ0.3。
- 长处：独立完整局holdout及网页五对手同批交付均高于G_frozen τ0.3。
- 长处：独立完整局holdout及网页五对手有效喷泉串均高于G_frozen τ0.3。
- 长处：独立dev/holdout人工短局真实因果击杀均高于L2_S05g_u019302 τ0.3。
- 长处：独立完整局holdout及网页五对手同批交付均高于L2_S05g_u019302 τ0.3。
- 长处：独立完整局holdout及网页五对手有效喷泉串均高于L2_S05g_u019302 τ0.3。
- 局限：完整局可行动静止占比高于线上参照：23.94% vs 18.83%，不能称全局更流畅。
- 局限：网页40局自死9次，高于线上参照4次。
- 局限：只验证网页P1；每个网页对手每方式8局，独立短局dev/holdout各16局，样本有限，未证明全面或统计显著优势。
- 局限：人工短局有任务隔离和属性/地形变化，导出固定完整局mode0；专项成绩不等于完整抢包胜率。
- 局限：此处只推荐可选模型；训练持续变化，发布必须使用本收据固定checkpoint和SHA。

可选组合 **L2_SP_u023302 tau0.3**；checkpoint `runs/cc_train_20261105/training/items_v3_20261107_l2_sp/L2_SP/u023302`，policy SHA `39b0dea7327221891bdd4cfba93ab917c897aa86fd78d83401241ad945a982cf`。

- 长处：独立dev/holdout人工短局真实因果击杀均高于G_frozen τ0.3。
- 长处：独立完整局holdout及网页五对手有效喷泉串均高于G_frozen τ0.3。
- 长处：独立dev/holdout人工短局真实因果击杀均高于L2_S05g_u019302 τ0.3。
- 局限：完整局可行动静止占比高于线上参照：34.08% vs 18.83%，不能称全局更流畅。
- 局限：网页有效喷泉串弱于同checkpoint的argmax；采样胜负更好不能掩盖这一混合结果。
- 局限：只验证网页P1；每个网页对手每方式8局，独立短局dev/holdout各16局，样本有限，未证明全面或统计显著优势。
- 局限：人工短局有任务隔离和属性/地形变化，导出固定完整局mode0；专项成绩不等于完整抢包胜率。
- 局限：此处只推荐可选模型；训练持续变化，发布必须使用本收据固定checkpoint和SHA。
- 局限：人工短局holdout因果杀的配对差95%区间含0，不宣称已建立近身击杀优势；本点的可选依据主要是网页猎手表现及重复喷泉信号。

网页交付契约 `.hermes/WEB_CANDIDATE_READY.json` 为私有编排文件，按用户要求保留绝对checkpoint/导出路径；公开报告仅相对路径。网页CX负责发布，本owner不push、不修改默认。

最终实际评测调度为12个空闲物理CPU核36–47，共享逐核锁；纯调度回执 `runs/items_candidates_20261107/v1/CPU_SCHEDULING_V3.json`，八卡训练未更动。

## 最终交付核验

候选checkpoint/COMMITTED、policy、JSON/ONNX、协议及交付源码SHA全部一致，私有READY权限0600。最终现场七臂均有180秒内真实finite更新，覆盖八卡，训练worker均为通用入口。核验收据 `runs/items_candidates_20261107/v1/FINAL_DELIVERY_CHECK.json`；未改变训练、网页默认或发布状态。

## 最新训练停止与已发布状态

最新用户停止授权已执行：七臂和全部GPU训练已停止，原后继及fresh授权撤销。前文“训练继续/未push”仅描述评测交付当时。两候选已由网页CX发布为可选，默认仍G_frozen τ0.3；线上为v4-facing，原856局v3证据与网页owner的v4兼容小样本分开，不跨版本冒称成绩。发布SHA `c2ed1cb26b4c9546c7e8554f10dceae2eb451c74`，Actions success，桌面/移动加载SHA与τ已验证，发布回执 `.hermes/WEB_CANDIDATE_PUBLISHED.json`。READY保留当时交付原件与SHA，训练是否允许仅以停止策略/最新用户授权为准。
