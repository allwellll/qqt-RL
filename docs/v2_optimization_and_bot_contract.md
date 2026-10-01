# v2 训练效率与通用机器人接入任务

## 边界

仅修改当前canonical仓库。旧仓库正在运行的 r6 四组训练完全不可触碰、不可重启、不可改运行时。

## 目标 A：训练效率优化

按 TDD 实施并真实基准测试：

1. 为 JAX 启用可配置、进程间共享的持久编译缓存；缓存必须位于运行目录或用户显式目录，不进入 Git。
2. 优化反事实数据生成：消除外层逐 state 的重复同步/编译，按固定 shape 批量处理多个 state；保持每个 declared/effective seed、候选动作顺序、标签字段与旧实现一致。若 bitwise 不可达，必须逐字段报告最大误差并解释，不得静默改变语义。
3. 设计可复用/常驻执行器，使一个 candidate 的反事实生成、Critic 校准、联合更新尽量复用 Python/JAX 进程与已编译 kernel。若一次完整实现风险过高，先交付可运行的持久 worker 协议与 runner 集成，不接受只有设计文档。
4. Critic batch size 只做显式可配置与 benchmark，不更改默认 64，不用于当前旧 r6；测试 64/128/256 的吞吐和数值行为。
5. 提供 benchmark CLI，输出冷启动、热缓存、反事实 states/s、Critic samples/s、联合更新 steps/s、GPU利用率采样（可用时）。必须与当前默认路径做 A/B。
6. 所有优化先在小规模固定 seed 下验证 replay内容、标签、loss、checkpoint可读性和seed泄漏检查。

## 目标 B：可扩展规则机器人

设计并实现统一接入契约，Python 与浏览器 JS 语义对齐：

- `BotSpec`：稳定 id、version、display_name、runtime、capabilities、config schema/defaults、identity/provenance hash。
- `Bot` 生命周期：`reset(context)`、`act(observation, player_id, rng)`、可选 `observe_transition(event, next_observation, player_id)`、`close()`。
- `BotContext`/`BotObservation`/`BotAction` 使用明确、可序列化schema；动作必须统一为 `(move, ability)` 并经过合法性验证。
- `BotRegistry`：register/create/list/validate；拒绝重复ID、未知配置、版本不兼容和能力不匹配；禁止任意模块路径动态import，插件必须显式注册。
- 至少适配：现有 Tactical v2规则机器人、随机/漫游示例机器人；模型机器人通过同一接口接入但可声明async/browser-only。
- 训练、评估、Web均从同一稳定 bot id + config 创建；run manifest必须保存bot spec、配置、实现hash及fixture hash。
- 支持新增机器人而无需修改核心runner的if/else；给出Python和JS各一个最小插件例子和中文接入文档。
- 对手冻结且不进入梯度；声明deterministic、batched/JIT、async、transition-observer等capability。
- 跨语言fixture验证Python/JS同一机器人给出一致动作和身份hash。

## 质量门禁

- 严格TDD：测试先失败，再最小实现，再全量回归。
- 新代码加必要中文注释，重点解释batch shape、PRNG映射、同步边界、机器人生命周期/身份与冻结约束。
- 不引入旧仓库绝对路径、日志、checkpoint、缓存或凭据。
- CPU全量测试、Node/Web测试、真实小训练smoke必须通过；GPU不可用则明确跳过，不能冒充通过。
- 性能优化必须有真实旧/新A/B数字；若没有加速，保留正确性实现但不得宣称提升。
- 完成规格审查与质量审查后提交并推送main。