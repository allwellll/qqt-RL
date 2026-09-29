# Bun Bot 插件接入

## 稳定契约

- `BotSpec` 固定 `id`、语义版本、显示名、可运行端、capabilities、配置 schema/defaults，以及 identity、implementation、fixture、provenance hash。
- `BotContext` 在每局 `reset(context)` 时传入；`BotObservation` 只包含可序列化状态、tick 和合法动作集合。
- `BotAction` 永远是 `{move, ability}`；Bun 范围为 move `0..4`、ability `0..2`，并在进入环境前再次校验。
- 生命周期为 `reset` → 多次 `act`/可选 `observe_transition` → `close`。浏览器统一用 `Promise.resolve` 包装同步和异步 Bot。
- `batched` capability 使用 `none`、`emulated`、`native_host`、`native_device`，避免布尔值无法表达执行位置。
- 训练对手必须声明 `frozen=true`，只能返回动作，不能持有或更新 learner 参数，也不进入梯度图。

## 注册规则

Python 使用 `qqt_rl.bots.BotRegistry`，浏览器使用 `QQTBots.BotRegistry`。插件只能由应用显式调用 `register()`；注册表不接受模块路径或字符串 import，重复 ID、未知配置、版本不匹配、runtime/capability 不匹配都会失败。

内置稳定 ID：

- `bun.tactical_v2`：Python/浏览器安全战术 Bot，deterministic、`native_host` batch、支持 transition observer。
- `bun.random_roam`：Python/浏览器随机漫游示例，随机性只来自传入 RNG。
- `bun.browser_model`：浏览器模型适配器，声明 `async=true` 和 `browser` runtime。

## Python 最小插件

参考 `examples/python_bot_plugin.py`：定义生命周期对象，并由启动代码显式执行 `register(registry)`。生产插件应以源码内容和 fixture 内容计算 hash；修改行为时必须提升 version 并更新跨语言 fixture。

## JavaScript 最小插件

参考 `web/example_bot_plugin.js`：脚本只暴露显式注册函数，不扫描全局对象、不拼接模块路径。若 `act` 返回 Promise，Web tick-in-flight 门禁会阻止同一 tick 重入。

## 训练与评估

`prepare_bun_safe_aggression_v7.py` 将完整 bot spec、解析后的配置、implementation hash 和 fixture hash 写入 run/candidate manifest。runner 通过 `BUN_OPPONENT_BOT_ID` 与 `BUN_OPPONENT_BOT_CONFIG_JSON` 创建冻结对手；新增已注册 Bot 不需要增加核心 runner 的类型分支。

Python/JS 共用 `tests/fixtures/bot_contract_cases.json`，测试稳定 identity hash 和动作一致性。新增跨语言 Bot 时，应在该 fixture 或独立同格式 fixture 中加入固定状态、player id 与期望 `(move, ability)`。
