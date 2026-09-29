# v2 补充审计约束

1. 当前 runner 的 JAX cache 含 `cycle_XXX`，必须改为 candidate 级稳定目录：`counterfactual/critic/joint/eval`，从而跨 cycle 复用。
2. Critic 默认 batch 仍为 64；尾 batch 采用 valid_mask/padding 时，所有 loss 分子和分母必须排除 padding，并验证数学等价。128/256 只用于 benchmark，不改正式默认。
3. 反事实批量化必须保持旧顺序：split train→validation→test，state item递增，action id递增，MC sample最内层；每state继续使用原seed异或树，禁止global key flatten split。
4. 当前规则Bot有FSM状态。batch emulation必须每env独立Bot session或显式`[B,...]`状态，严禁一个有状态实例串行处理多个环境导致phase/cache串扰。
5. `batched`能力枚举为`none/emulated/native_host/native_device`，不使用含糊布尔值。
6. Web统一通过Promise-normalize调用同步/异步Bot，并用tick-in-flight门禁防止异步推理重入或重复`sim.step`。
7. 常驻worker必须保留cycle原子提交边界；worker崩溃时不能发布半成品checkpoint，runner可从最后完整cycle恢复。
