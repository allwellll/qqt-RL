# 网页模型菜单精简（20261107）

按用户任务及最新更正，菜单只保留4个，默认由G_frozen改为SP u023302 τ0.3；这是用户明确授权，未新增评测或声称SP全面胜出。

- SP · 少自死/猎手强/偶停（默认）
- G_frozen · 均衡
- DUEL · 近身击杀强 / 自死偏多
- 风险线 u9969 · 旧规则录像认可点

SP名称为手机窄屏略缩短措辞，完整description、长处/短板与原v3评测/既有v4快筛证据原样保留。推理默认配置改SP，τ0.3保持；G_frozen仍按其原τ0.3可选。

其余18项只从web/models.json移除。历史身份元数据另存web/retired-models-20261107.json供旧Review/原发布证据检查使用；不删Release/OSS资产、Git历史、旧录像，不触碰训练。原有Review身份检查从只读历史清单读取，不要求旧模型继续占当前菜单。

已检查引用：游戏URL原先只读取seed；localStorage原先保存人物与榜单数据，没有模型id恢复路径，未新增模型存储/深链协议。菜单事件遇到已下架model:id或失效空值，归一化回SP，不把未知model传给Bot registry。保留既有规则Bot和初始协作猎手对局设置；“默认模型”指模型组首选及推理默认，不强迫首屏下载模型。

目录/默认/失效选择及HTTP轻量契约检查通过。静态源码闭包元数据随受影响文件刷新。未运行额外选模评测、模型推理基准或模型资产SHA下载回读；GitHub既有发布流水线按原步骤运行。

隐私扫描覆盖tracked与本次明确待提交文件；发布后按同SHA等待Actions/Pages最终结果，再只读models.json/推理默认配置并在桌面/手机尺寸打开菜单查看。结果交付后回填本地，不为收据额外改变验收SHA。

## 发布验收回执

- main / Pages SHA：`37f5af884e92f65ce268730c5831662b5014503f`。
- [Actions 42387670616](https://github.com/allwellll/qqt-RL/actions/runs/42387670616)：workflow、test-and-build、deploy 全部 success；既有暂停训练策略/源码闭包和网页测试通过。
- 线上 build-info 同 SHA；models.json 仅4项，顺序为 SP、G_frozen、DUEL、风险线u9969；推理配置默认 SP τ0.3。仅读取菜单元数据，没有模型资产 SHA 回读。
- 已用真实 Chromium 打开线上菜单，查看1280×900桌面及390×900手机尺寸截图：四项特点和SP默认标签可读，无横向溢出、无pageerror。测试预置既有设备授权标记后通过高阶设置按钮打开菜单，未测试口令验证；未选择/下载/推理模型。手机为Chromium触控视口模拟，不是实体iOS。
- 本地验收证据：`.hermes/menu-public.json`、`.hermes/menu-browser-results.json`、`.hermes/menu-public-1280.png`、`.hermes/menu-public-390.png`。
- 发布前隐私扫描0项；回填后再次扫描tracked文件边界。最终回执保留本地，已发布报告正文随上述SHA入库。
