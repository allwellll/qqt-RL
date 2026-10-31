# bun-native-v2 条件发布验收（20261031）

网页真人P0、模型P1。bun-native-v2自由取包，生产JS Hunter、argmax、老师关闭。固定三档每档256局（各席128），P1三档胜负差{'hunter_hard': 99, 'hunter_normal': 116, 'hunter_easy': 123}，合计+338（网页旧Actor在v2下+320）；P0合计+353（基线+345）；自伤0。逐档/P0/自伤护栏全部通过。协作Hunter hard单列压力测试，不混排名。16出生几何重复；不宣称统计显著或真人泛化。旧v1可选、旧录像按原规则回放，成绩分榜不混排。R3如有使用25平面/37维可见输入；真人P0 frameStep与评估Sim.step范围分开，不宣称逐位等价。仍可能有零泡、振荡与持包停滞。

{
  "status": "PUBLISHED_VERIFIED",
  "head": "6fff35a249089b555207dc2a72ccd0227408fdee",
  "tag": "web-bun392-v2-r3-a-u4002-u4002-20261031",
  "policy_sha256": "f12ab745ca746ae364bcb172fb425399cbf47038f60f7129617036a4051d6a30",
  "model_sha256": "763ae64a325215de60fa4af403f4e03aeb27ad888bd859d54a097144e816c0a7",
  "onnx_sha256": "f11461f32ba29e3aed09c280e9cf66f974f22cce5b9cfb157839a0e3fad59656",
  "ruleset": "bun-native-v2",
  "webpage_url": "https://allwellll.github.io/qqt-RL/",
  "replay_url": "<EXTERNAL_REPLAY_URL>
  "release_url": "https://github.com/allwellll/qqt-RL/releases/tag/web-bun392-v2-r3-a-u4002-u4002-20261031",
  "database_ruleset_contract": 2,
  "training_untouched": true,
  "unix": 1793457367.1907659
}

完整6144局比较见本轮本侧出生复核报告；原始SHA、完整state来源、导出逐张量、fp32/JS/ONNX/浏览器、回归、分榜、非force Git、同head Pages和冷加载完整P1真人模式证据见本轮独立选型namespace的publication目录。数据库写入在浏览器测试中屏蔽。真人键盘探针1秒后空手等待，不能冒充真人对抗能力评估。旧模型/规则/回放/数据保留；未修改训练协议或停止训练。
