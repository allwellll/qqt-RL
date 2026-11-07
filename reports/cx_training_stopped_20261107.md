# 训练已按用户要求立即停止（20261107）

最新用户指令覆盖等待预算完成的旧收尾授权。持久禁止新run、新预算、后继、自动恢复及GPU诊断更新。先核实实际argv/私有配置与已捕获worker身份，终止原有限收尾owner和未完成fresh诊断，再以USER_STOP安全保存退出；安全窗口最多40秒，未退出只向已确认本任务进程发信号。未知进程信号数0。未为保存新点增加训练。

|臂|GPU|最后完整checkpoint|最后观测更新|未保存更新|当前观测更新已保存|停止方式|
|---|---|---|---:|---:|---|---|
|L2_DUEL_SYNC2|[0, 1]|`runs/cc_train_20261105/training/duel_sync2_valid_20261107/L2_DUEL_SYNC2/u020048`|20048|0|True|COOPERATIVE_USER_STOP_OR_ALREADY_FINISHED|
|L2_DUEL|[6]|`runs/cc_train_20261105/training/items_v3_20261107_l2_duel/L2_DUEL/u020905`|20905|0|True|COOPERATIVE_USER_STOP_OR_ALREADY_FINISHED|
|L2_DUEL75|[3]|`runs/cc_train_20261105/training/items_v3_20261107_l2_duel75/L2_DUEL75/u020903`|20903|0|True|COOPERATIVE_USER_STOP_OR_ALREADY_FINISHED|
|L2_FTN2|[4]|`runs/cc_train_20261105/training/items_v3_20261107_l2_ftn2/L2_FTN2/u022292`|22292|0|True|COOPERATIVE_USER_STOP_OR_ALREADY_FINISHED|
|L2_DUEL25|[2]|`runs/cc_train_20261105/training/items_v3_20261107_l2_duel25/L2_DUEL25/u020844`|20844|0|True|COOPERATIVE_USER_STOP_OR_ALREADY_FINISHED|
|L2_SP|[7]|`runs/cc_train_20261105/training/items_v3_20261107_l2_sp/L2_SP/u024155`|24155|0|True|COOPERATIVE_USER_STOP_OR_ALREADY_FINISHED|
|L2_ATK|[5]|`runs/cc_train_20261105/training/items_v3_20261107_l2_atk/L2_ATK/u025623`|25623|0|True|COOPERATIVE_USER_STOP_OR_ALREADY_FINISHED|

全部最后COMMITTED的state/policy SHA均重新核验。尚未提交的进行中更新不作已保存声明；未完成fresh保留原日志并中止，不补跑。旧runs/checkpoints/protocol均保留。

最终训练worker数：0；GPU计算进程：`[]`；八卡锁：`{"0": true, "1": true, "2": true, "3": true, "4": true, "5": true, "6": true, "7": true}`。

GPU实测：

```text
0, 0 MiB, 0 %
1, 0 MiB, 0 %
2, 0 MiB, 0 %
3, 0 MiB, 0 %
4, 0 MiB, 0 %
5, 0 MiB, 0 %
6, 0 MiB, 0 %
7, 0 MiB, 0 %
```

统一持久状态 `.hermes/TRAINING_STOP_STATE.json`，全局worker停止闸门 `runs/cc_train_20261105/training/USER_STOP.json`，每臂USER_STOP/NO_SUCCESSOR及namespace SUCCESSOR_CANCELLED均保留。原预算目标已撤销，不再等待终点。通用入口model_entry在GPU导入前拒绝新训练，fresh白名单清空；恢复必须用户以后明确重新授权。

网页候选沿用856局与READY契约：DUEL u019902 τ0.3近身击杀、SP u023302 τ0.3猎手与喷泉。发布回执 `.hermes/WEB_CANDIDATE_PUBLISHED.json`；不等待大规模新评测。

## CPU代码复核与网页发布

7项停止闸门测试、69项FTN2防刷分测试、5项DUEL结算测试、生产道具JS回归、Python/JS语法及diff检查均通过。所有active协议来源SHA仍一致；未改旧protocol，未启动GPU更新。新`training_launch_policy.json`可入版本库，缺失私有停止文件也默认禁止训练。代码索引 `docs/training_code_map_20261107.md`，审计 `runs/code_review_20261107/REVIEW_TESTS.json`。

两候选已由网页CX发布并在桌面/移动端核验，默认仍G_frozen τ0.3。线上为v4-facing，v3评测与v4兼容快筛分开。线上URL https://allwellll.github.io/qqt-RL/；SHA `c2ed1cb26b4c9546c7e8554f10dceae2eb451c74`；Actions https://github.com/allwellll/qqt-RL/actions/runs/42347826789 success。本owner独立读回线上模型目录确认两可选存在。

review后的代码/文档按文件SHA清单交网页集成owner合main，不整树覆盖、不并发push。原始所有runs/checkpoints/协议保留。
