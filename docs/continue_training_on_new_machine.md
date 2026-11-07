# 在新机器继续 DUEL / SP：模型权重新开一轮

截至 2026-11-07，所有训练已停止，自动续跑与后继已撤销。此文档是迁移操作说明，不是恢复训练授权。必须先获得用户重新授权，再由新机器的操作者解除停止策略；当前仓库默认 `training_launch_policy.json` 的 `training_enabled: false`，开训入口会在导入 JAX 前拒绝运行。

**验证范围：仅 DUEL 本机 CPU 加载检查通过，SP 与干净目录未验证。** 用户随后要求直接上传，因此跳过后续 CPU、干净目录和 GPU 验证。上传前隐私扫描已完成；上传后SHA256回读仅完成 **11/21**，用户随后明确要求停止回读、节约时间，其余10项未复核，不再继续下载或额外验证。未宣称已在第二台机器或 GPU 上完成训练。

## 当前有效入口与版本

- [Release：training-bundle-20261107](https://github.com/allwellll/qqt-RL/releases/tag/training-bundle-20261107)：19 个运行数据/模型文件，加 `MANIFEST.json` 与 `SHA256SUMS`。只下载这些即可，不下载全部历史 Release。
- 代码以本次合入 main 的 `scripts/portable_training.py` 为新机器入口；加载 `training_runtime/items_v3/` 下四个封存的实际物理/混合任务模块。旧 `scripts.cx_items_install_v2` 的历史训练包装器、旧协议及 state 恢复命令属于历史入口，不能直接拿来迁移。
- 训练包物理是 **bun-native-v3-items**，与原 DUEL/SP 训练及专项评测一致。网页当前是 **bun-native-v4-facing**；没有将 v3 训练包称作 v4，也没有悄悄变更训练物理。
- DUEL 从 **u020905**、SP 从 **u024155** 的 `policy.pt` 开新一轮；额外保留网页上线点 **u019902 / u023302** 的完整 actor 权重。默认网页模型不因打包变更。
- `policy.pt` 只含 actor。旧 critic、Adam、RNG、环境和对手池状态没有恢复。新 critic 确定性随机初始化、Adam 从零、环境从 fresh reset、RNG按协议seed重新初始化、对手池重建；这不是精确断点续训。不得恢复或重建旧 `state.pkl`。

## 硬件和环境

本机来源环境：Linux x86-64、Python **3.12.12**、JAX/JAXlib **0.11.2**、Optax **0.2.8**、NumPy **2.5.4**；CUDA 12 插件/PJRT **0.11.2**，CUDA runtime 包 **12.9.79**。其他版本未在本次验收。CPU 加载无需 CUDA；建议至少 16 GB 内存，预留 2 GB 下载/安装空间，训练还需额外日志与模型空间。

原配置为单 GPU：DUEL 2048 env、128 ticks、minibatch 8192；SP 3072 env、128 ticks、minibatch 12288。原 SP 4096 env 曾在 143 GB 卡上 OOM，不能据此保证 80 GB 卡跑原配置。建议按显存从更小 batch 开始；32/48/80 GB 卡的可用 batch 没有本次实测。减 env/minibatch 会改变样本规模，必须记录为新配置。env 必须是 4 的倍数，`envs × 128` 必须能整除 minibatch。

CPU 安装示例（在仓库根目录）：

```bash
git clone --branch main https://github.com/allwellll/qqt-RL.git
cd qqt-RL
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'jax==0.11.2' 'jaxlib==0.11.2' 'optax==0.2.8' 'numpy==2.5.4' scipy
```

仅在重新授权并准备 GPU 训练后，按 JAX 官方兼容矩阵安装对应 CUDA 插件，例如 `python -m pip install 'jax[cuda12]==0.11.2'`。CUDA12 wheel 的 Linux 驱动通常至少 525 系列，实际以该 wheel 的要求为准；不要将本机 runtime 包版本当作新机器已经具备的系统 CUDA。CPU-only 环境无需安装 GPU 包。此轮没有启动 GPU 验证。

## 下载、SHA256 校验与目录

```bash
python scripts/fetch_training_bundle.py
```

下载器只获取本 Release 的19个 payload，先校验 `SHA256SUMS` 中的 manifest，再逐个校验文件大小及 SHA256，最后写入包内相对路径。已有目标文件时拒绝覆盖。不要把凭据、`.hermes` 或原机器目录复制到新机器。

手工下载时：打开 Release 取 `MANIFEST.json`、`SHA256SUMS` 和19个资产，在下载目录运行 `sha256sum -c SHA256SUMS`；按 manifest 的 `files[].path` 放置文件。Release 名称用 `--` 编码目录，例如 `models--DUEL_latest--policy.pt` 对应 `data/portable-training-20261107/models/DUEL_latest/policy.pt`。所有协议路径均相对于仓库根目录，不能只把权重散放在当前目录。

```text
data/portable-training-20261107/
  MANIFEST.json
  ACCEPTED_RULE.json
  protocols/DUEL.json
  protocols/SP.json
  canonical/web/assets/maps/levels.json
  datasets/BANK.json
  datasets/p0_bank.pkl
  datasets/p1_bank.pkl
  datasets/train_spawns.json
  datasets/train_starts.json
  datasets/SCENARIOS.json
  models/<MODEL_NAME>/policy.pt
```

19个文件共 **275,698,799 bytes**（约262.93 MiB；manifest另计）。9个actor包括4个当前/网页点，以及 G_health、G_j、G_i、G_frozen、SP_fixed 五个实际使用的对手锚点。G_frozen 在这里是训练对手依赖，因此需要带；网页 JSON/ONNX 不属于训练输入，没有重复上传。真实 Bank 裁剪后只保留运行读取的 `states`、`aux`、`attribute_highwater`，两物理座位分别25/16条，保留其真实属性历史，未伪造新样本。

manifest 的 `code_sha256` 校验封存模块和入口等关键代码；应使用对应发布版本的 main。若未来 main 修改这些文件而校验失败，checkout 本 Release tag 对应提交，再使用该版本说明，不删除校验来绕过。

## 可选的 CPU 检查（本轮按用户要求跳过后续执行）

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -m scripts.portable_training --line DUEL
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -m scripts.portable_training --line SP
```

预期只做私有物理安装、Bank/九个模型/六个对手槽加载、4环境初始化和有限数值推理，输出 `PASS_CPU_ONLY`；rollout及优化器更新为0。此命令不解除停训闸门。这里列出命令供接收方使用，不代表 SP 或干净目录已经通过。

## 重新授权之后的开训命令

当前运行下述命令会被 `training_launch_policy.json` 拒绝。只有用户明确重新授权后，操作者才可在自己的机器修改停止策略，并确认没有私有持久停止状态仍在生效。包没有自动授权、自动选卡、跨机器进程终止或后继编排。

原预算大小示例（每条线显式指定单卡及新输出目录）：

```bash
python -m scripts.portable_training --line DUEL --train --gpu 0 --updates 1200 --out runs/portable/DUEL
python -m scripts.portable_training --line SP --train --gpu 1 --updates 1200 --out runs/portable/SP
```

小显存起步示例仅是待授权配置，不是经过本轮 GPU 验收的推荐性能值：

```bash
python -m scripts.portable_training --line DUEL --train --gpu 0 --updates 40 --envs 128 --minibatch 512 --out runs/portable/DUEL_small
```

协议保留两条线的实际九槽权重、SIL系数0.5与原奖励；DUEL为一半完整局、一半单命对拼。初始 own current/ring 都使用该线最后actor；固定历史与 C 回退使用随包锚点。新入口不扫描旧机器 live-peer registry，C 对手为显式封存回退；own ring 随本轮保存点更新，SP保留入池gate/PFSP函数。未来接入其他活跃同伴需显式扩展协议，不能把静态回退说成动态跨线自博弈。

前几轮 critic/优势估计和旧值不同、loss或胜率波动均可能出现，因为只续 actor，不续旧critic/Adam/样本流。计数从本轮0开始，原update写在POLICY元数据里；不得把新一轮当作原 COMMITTED 的无缝恢复。

## 只保存模型

当前入口每40个update及最后一轮只保存 `runs/portable/<LINE>/uNNNNNN/policy.pt` 和 `POLICY.json`（actor SHA、原来源点、规则与本轮update）；**目前未保存 Critic**。只写模型与日志，不写 `state.pkl`、优化器、环境、旧RNG或完整恢复token。保存后的再次启动仍是新的 policy-only round，不是本轮的精确恢复。输出目录已存在时拒绝覆盖，避免混写。

`--updates` 是固定上限，结束后不启动后继。下次重新授权的停止方式：关闭 tracked训练策略，或在该任务输出目录创建 `USER_STOP`；入口在每轮开始检查，当前已开始的一轮结束后才会再次检查。此轮不执行这些开训/停止命令，也不触碰其他机器或未知进程。

## Critic：迁移事实、预热建议和今后保存策略

`policy.pt` 只含 Actor。训练使用独立 Critic：`r4_k4_observation.actor_critic_forward` 从 `params['critic']` 计算价值。旧 Critic 仅存在于已经删除的 `state.pkl`，当前 DUEL/SP 两线都无可用旧 Critic。不能从 actor 文件恢复旧 Critic，也不会为此重建旧 `state.pkl`；新机器 Critic 从零初始化。

**建议的第一段**：冻结 Actor，仅训练 Critic **40次 update**，随后在价值损失的滑动均值/波动稳定、预测值保持有限且没有持续发散时，再进行 Actor/Critic 联合训练。40只是“几十次”的经验起点，**未实测、不是已验证阈值**，不能保证40次后一定稳定；若仍不稳定，先检查回报尺度、样本与初始化，而非自动解除冻结。Actor冻结应同时确保不执行Actor优化器更新；Critic预热阶段可以用固定Actor采样新的on-policy数据。首段价值曲线抖动、联合训练后短暂小幅回退均属可能现象，不应宣称与旧状态无缝接续。

**当前实现边界与待办**：本入口没有现成的“冻结Actor／仅更新Critic／预热结束切换”开关，上面的开训命令直接进入联合训练，尚未实现该推荐流程。须在未来用户授权的代码任务中增加并核验这些开关；不能把上文40次当成当前命令已经执行的预热。本轮按用户要求只改文档，不添加开关、不运行预热或训练。

**今后每个保存点的目标策略**：在 `policy.pt` 之外附带独立 **`critic.pt`**，保存真实的 `params['critic']`（约30MB，实际以参数文件大小为准），分别记录Actor/Critic的SHA256、规则、模型结构和update。只增加Critic模型参数，仍不保存优化器、环境、RNG或对手池状态。这样以后可加载Actor+Critic并重建Adam/环境/池；仍不是完整状态恢复。

**当前保存器待办**：本轮代码尚未写出或加载 `critic.pt`；现有Release也没有Critic资产。落实此保存策略及加载配对校验需要后续单独授权修改代码。在实现之前，不得把当前policy-only保存点描述成已经保留Critic。

## 常见问题与已知边界

- `Persistent user stop`：当前预期行为，必须先有用户重新授权，不能用删停止文件规避。
- 校验失败：核对 Release、Git版本和路径；不要改掉预期SHA或混用v4网页物理。
- 缺少 `state.pkl`：本包设计如此；不要运行历史full-state恢复入口。
- 找不到旧 K1/bomb_threat/authorized_campaign/r3 lineage 模型：新入口不读取这些历史引用，所以没有为凑齐历史协议而打包；真正使用的9个actor均随包提供。
- OOM：降低 env/minibatch 并另记新配置；保留原物理与奖励，不补造旧状态。
- CPU检查耗时：首次JAX编译可能较慢；本次用户要求不再等待SP或干净目录验收，没有宣称此类检查通过。
- 下载19个资产与SHA回读证明传输一致，不等于训练正确性或新机器可训练保证。GPU collector/优化器路径本轮未执行。

对应交付报告：[cx_portable_training_bundle_20261107.md](../reports/cx_portable_training_bundle_20261107.md)。
