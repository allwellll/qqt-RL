# JE10 已完成 checkpoint 发布契约与验收（20261018）

## RED 发布契约（生成新录像前冻结）

仅在独立发布 worktree 修改发布文件。训练源及 runs 只读；不启动代理、训练、GPU任务，不等待 r05，不修改或中断既有进程。候选快照限本次读取时已有 COMPLETE 且 summary/checkpoint hash 可核验、固定协议一致的 v38 异步评估。排除无效和未完成评估；综合路线交付、natural_full2400、Tactical/Hunter 安全击杀与自伤选择，不使用训练 reward 排名。报告全部候选和明确 tradeoff。

固定诊断矩阵：seed 2026101800，real_drop、native_death_stock、native_sugar_stock、natural_full2400 各一条；若复用同协议已经预声明的该 seed 录像，保留全部四条、不筛胜局。使用同一个所选 checkpoint、生产模拟器和原始动作；场景初始化、截断/终局、训练历史录像与新生成策略诊断的区别须明示。不得将重演诊断冒称历史训练 rollout。

发布前必须有自动失败的契约测试，要求：models.json 首项与 review-catalog.json 首批指向同 checkpoint SHA-256；模型导出/ONNX/录像字节 hash 校验通过；目录身份、seed、场景、大小和摘要完整；Review 可见 raw/shaped A1、ΔA 与各相邻 tick 增量。优势仅允许真实反事实策略加权 Q1 或明确公式的离线近似，缺失严格字段为 null 并说明原因；禁止 Reward/Value/Logit 改名为优势。逐 tick 展示动作、前后状态 hash、事件、reward 分量、路线 phase/target/Phi；无法恢复字段明确未记录。

仅通过 review-catalog.json + review-replays/ 发布诊断；旧 replays 链路只读。模型沿用既有 Release 内容寻址 JSON/ONNX，权重不进 Git。生成后逐动作确定性复算；Review 和网页全量测试、相关 Python 测试、桌面及手机全部深链、真实模型加载、隐私扫描通过后仅提交相关文件。普通快进推送 main，远端前进则 fetch/rebase。等待精确提交 SHA 的 Pages completed/success，并实测 build-info、目录首项、模型及每条录像 HTTP 200/hash、深链真实渲染与零 JS 错误；全部通过才称网页已更新。部署后证据存本项目忽略的 runs 目录，避免自引用提交产生再次部署。

## 执行结果

候选快照共37个已完成固定点（r01–r04全部，r05仅已完成g0/4/8/16/32）；不等待后续点。五轮 PROTOCOL SHA-256 一致：`f2fcf50022fbb61e9630855ca3d08ccef4b1d390d89a783a5bdf998b9205a936`。逐项核验 COMPLETE 的全部shard/summary/DIAGNOSIS hash及policy hash。选择脚本只读且不调用训练owner。

选择规则：先优先所有协议零观测自伤的点，再最大化四个分列指标的等权几何均值：natural_full2400交付/16、其余七类路线交付/112、Python Tactical安全击杀/64、JS Hunter三档安全击杀合计/192。几何均值用于避免某一维过弱被加法抵消；指标分列报告，不将不同模拟器合并成胜率。不使用训练reward。这是本次发布的探索性权衡，不是训练门禁、统计显著性或综合晋升。

所选 **v38 r01 attempt04 g8 raw**，分数0.26410648；policy SHA-256 `d392bf8caeb0ce149a91ce5af1c69806c556cadd69d89bef3c00758d2d3a89fc`；COMMITTED完整state SHA-256 `fbdc05d14ab682553a4040198ed05d3993531dabfc8d862e802a157f1ba5f0fc`（流式读取全文件核验）。8卡已提交checkpoint、16,777,216 fresh ticks；不操作训练进程。

| 候选 | 自然交付/16 | 七类路线/112 | Tactical/64 | Hunter/192 | 几何均值 |
|---|---:|---:|---:|---:|---:|
| r01 g8（选择） | 6 | 62 | 12 | 24 | .26410648 |
| r02 g16 | 6 | 64 | 10 | 27 | .26195033 |
| r02 g32 | 8 | 67 | 11 | 16 | .25583771 |
| r01 g96 | 7 | 73 | 11 | 16 | .25280089 |
| r03 g8 | 8 | 74 | 10 | 11 | .23319745 |

不存在单点全优。选择点相对r02 g16路线少2局、Hunter少3局，Tactical多2局；相对r02 g32和r03 g8自然交付少2局，但Hunter保持较高。父点Tactical17/64、Hunter32/192，所选点仍有保持退化，尤其Hunter hard0/64；有限固定样本和多点选择不支持显著性断言。

所选点路线八项交付依序：native_death_stock2/16、native_sugar_stock3/16、real_drop8/16、carry_near16/16、carry_mid16/16、carry_far14/16、recovery_threat3/16、natural_full2400 6/16。对应拾包10/11/8/0/0/1/11/9（各16局；预置持包不需拾包）。Hunter hard/normal/easy安全击杀0/2/22（各64局），Tactical12/64；这四组自伤均0，八类路线self_death亦0，不能据此断言总体安全风险为零。real_drop交付8局均来自team1，team0仍0/8；库存原墙组和开墙组不应混称已解决。

RED：在模型、目录、录像生成前运行`node web/test_review_je10.js`，以“selected JE10 must lead model selector”按预期失败。契约同时约束hash身份、字段完整、Review可见优势和全部动作状态复算；后续转GREEN。

复用冻结评估最早声明seed2026101800的四场全部录像，初始状态/原动作不变；三类任务为合法reset fixture，非模型自然制造初态；natural_full2400为原生完整终局。此为真实已完成训练checkpoint的策略诊断录像，未找到历史训练rollout，页面明确不冒称历史训练轨迹。

raw/shaped A1使用同checkpoint masked softmax(move)×softmax(ability)在合法联合动作上归一化，并固定真实对手动作、克隆生产Sim全部状态及相同随机流。对每个候选真正推进一步；每个实际动作分支与原录像state hash完全一致。Q1=r+γ×0，A1=Q1(taken)−Σπlegal Q1，ΔA=A1shaped−A1raw，相邻增量x[t]−x[t−1]。V=0为明确的零后续值近似，未用Value/Logit代替优势。当前可观测raw奖励子集为24×交付−12×死亡−16×自炸−12×同死−.5×携包死亡；shaped再加独占归因首控4与可观测连续路线.5×(.995Φafter−Φbefore)。当前系数源文件hash写入meta；未复原settlement延迟、cap/clawback、avoidable danger与其他hidden aux，故不是完整当前/训练Reward，更不是长期优势或严格Critic差。完整训练reward、严格raw/shaped A1均为null并显示未记录。

四场4624 tick的raw/shaped A1和ΔA近似覆盖100%；每场首tick相邻增量null，其余4620/4624=99.9135%；严格训练A1覆盖0%。动作、合法策略、前后状态hash、事件、可观测reward子分量与路线phase/target/Phi逐tick覆盖100%；COMBAT目标为null表示无激活路线目标，不是缺失。全部反事实action/probability/reward/next_state_hash均随录像发布。原始reward-GAE保留其近似命名，未当A1。

模型保持既有浏览器JSON及CPU fp32 ONNX格式，均复用已经评估的原字节，权重放Release/Pages且不进Git。JSON SHA-256 `19b98f2192a237e994ffd29801de7934dfe6168de3507530b19b3c7dd4d5ca45`；ONNX SHA-256 `b1da0c13915e1093d18b07d1f059f579b68a7f47c8469c3b49e378c2c117dc16`。models.json首项和Review四场checkpoint/模型/ONNX身份严格一致。37点完整指标和出处hash保存在review-catalog.json的je10_selection中。


固定录像事实：real_drop在tick1真实掉包，模型首偏航为动作前tick3/结果帧tick4，A_ENEMY_DROP cell84/nonce1，oracle下/左，实际右；模型未拾取/交付，最终tick451负。两类库存均tick5真实取包，但1200 tick未交付；自然原生局tick1542取包、tick1773真实交付并终局胜，首偏航动作前tick1425/结果帧1426为B_STOCK。所有失败与偏航保留；不把固定seed自然胜局当作16局总体结果，不把预置窗口当自然制造窗口。

独立验证：75个导出JSON张量与policy.pt逐数组精确一致且finite；全部4624实际动作同checkpoint CPU ONNX重推理一致，生产Sim每tick前后状态与冻结录像一致；46774个合法反事实分支在另一脚本中逐个重建随机流并复算next_state_hash、可观测奖励与路线Phi/公式，全部精确一致。模型JSON和ONNX内容寻址资产已发布至`web-bun392-je10-v38-r01-g8-20261018` Release，公开HTTP回读大小/hash通过。未提交任何权重。

本地验收：桌面1600×1100及手机390×844四深链全部实际渲染，逐局tick1/2/中点/终点优势与三类增量精确显示、刷新回到tick2、hash验证、无横向溢出、零pageerror/console error/HTTP错误。两端实际模型选择器加载并额外独立worker推理均为worker-wasm，parity最大误差0.0000044788011452912。修复新诊断长文本换行；WASM使用error级日志避免CPU厂商识别的原生warning被浏览器作为console.error输出，实际推理和错误处理保留。相关Python模型导出/下载/seed泄漏23项通过。Review全量包含旧版录像hash回放、伪造A1/概率/字段/fixture拒绝、新版820项schema/hash/seek/标注检查。

最终`npm test`、单独`npm run test:review`与相关Python23项通过；隐私扫描覆盖所有Git跟踪文件及新增发布产物，用户标识/含标识绝对路径/内网IP/机器PID零发现，旧replays.json及replays/无改动。精确提交SHA、Pages run和线上逐文件HTTP/hash/浏览器结果另存本项目`runs/je10_release/`，避免自引用提交循环。


| 固定seed2026101800场景 | tick | 录像SHA-256 |
|---|---:|---|
| real_drop | 451 | `27aa70746fd03ee13f8de548f1fdb3d3a0609ef77bd8b7bde9b10231f161e315` |
| native_death_stock | 1200 | `7f2a80cd4a855a20b22498c0e56f5af5837fc97a73312220a8b204150b7c51a5` |
| native_sugar_stock | 1200 | `319accbe85507dfb5690e5bcc69c06a720ba1d088293452bbb4ac0e37dba5f6b` |
| natural_full2400 | 1773 | `3b7ab090983b18b66b7c521aeb5f1cd631836067caf1225c6bb9bd01f563e4d5` |
