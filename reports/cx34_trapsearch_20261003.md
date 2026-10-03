# CX-34 trap-search 诊断

## 目的与边界

CX-34 测试一个局部候选格搜索：hard JAX 代理在距离 Hunter 约三格的候选格上比较 danger 与逃生空间。它是机制诊断，不是生产入口变更。训练通过隔离 worktree 的 inline monkeypatch 注入，不能把训练 metadata 中的旧 opponent 名称当成已验证实现来源。

## 协议

- control 与 treatment 各两个训练 seed，保存 21-update run 的 `it10` 与 `final`。
- 四 Bot 快筛：固定评估 seed `20261001`、相同出生、greedy、300 tick、每 Bot 64 局、Tactical v2 与真实网页 hunter `hard`/`normal`/`easy`，启用 attack trace。
- `S` 是 surviving non-trade kill；64 局只作筛选，不能替代独立 seed 的 512 局确认。

## 原始计数

下表四列依次为 tactical、hard、normal、easy，数值均为 `S/64`。

| checkpoint | 四 Bot S/64 |
|---|---|
| cx32_control_baseline | 24, 0, 0, 5 |
| control_s20340001_it10 | 30, 0, 0, 5 |
| control_s20340001_final | 22, 0, 0, 4 |
| control_s20340002_it10 | 23, 0, 0, 4 |
| control_s20340002_final | 29, 0, 0, 6 |
| trapsearch_s20340001_it10 | 27, 0, 0, 5 |
| trapsearch_s20340001_final | 32, 0, 1, 5 |
| trapsearch_s20340002_it10 | 27, 0, 0, 3 |
| trapsearch_s20340002_final | 31, 0, 0, 4 |

## 结论

trap-search 在代理侧的改善没有形成跨 seed 的真实网页 Hunter 攻击收益：hard 全部为 `0/64`，normal 只有一个 treatment 保存点为 `1/64`。Tactical 的短程点估计上升不足以证明归因或泛化。该结果不满足长训门槛，不合入代理实现，也不启动 101-update/512 局扩展；下一步应先对齐 JAX 标签与网页 Hunter 的时间展开和可达性。

原始 run 与矩阵不进入版本库；统一门槛和前序结果见[训练实验总账](../docs/training_experiment_ledger.md)、[CX-01 至 CX-29 经验索引](../docs/cx01_29_reproduction_lessons_20261003.md)及[CX-29 报告](cx29_trap_routes_20261003.md)。
