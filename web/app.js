'use strict';

(async function startBunArena() {
  const canvas = document.getElementById('game');
  const status = document.getElementById('status');
  const restart = document.getElementById('restart');
  const opponentSelect = document.getElementById('opponent');
  const matchMode = document.getElementById('match-mode');
  const publishedModel = document.getElementById('published-model');
  const modelDetails = document.getElementById('model-details');
  const modelFile = document.getElementById('model-file');
  const modelStatus = document.getElementById('model-status');
  const levelList = await fetch('assets/maps/levels.json').then((response) => response.json());
  const baseLevel = levelList.find((item) => item.qqt_id === 806);
  if (!baseLevel) throw new Error('Bun06 level missing');
  const level = QQTTacticalArena.buildTacticalArena(baseLevel);
  const visualAssets = await QQTVisual.loadAssets(level);
  const renderer = QQTVisual.createRenderer(canvas, level, visualAssets);

  const held = new Set();
  const modelRng = QQT.mulberry32(0x515154);
  let bombQueued = false;
  let loadedModel = null;
  let publishedModels = [];
  let activeBot = null;
  let ticking = false;
  let sim;

  function instantiateModel(document) {
    const arch = document.meta && document.meta.arch;
    const model = arch === 'transformer' ? new QQT.TransformerModel(document)
      : arch === 'cnn' ? new QQT.CNNModel(document) : new QQT.MLPModel(document);
    model.greedy = true;
    return model;
  }

  function createRegistry() {
    return QQTBots.createDefaultRegistry({
      BunRuleTacticalBot,
      modelFactory: loadedModel ? () => ({
        reset(context) {},
        async act(observation, playerId, rng) {
          const raw = await Promise.resolve(loadedModel.act(sim, playerId, modelRng));
          return QQTBots.validateAction({ move: Number(raw[0]), ability: Number(raw[1]) });
        },
        close() {},
      }) : null,
    });
  }

  function resetBot() {
    if (activeBot && activeBot.close) activeBot.close();
    const registry = createRegistry();
    const botId = matchMode.value === 'model-vs-rule' ? 'bun.tactical_v2' : opponentSelect.value;
    activeBot = registry.create(botId, {});
    activeBot.reset({
      schema: 'qqt.bot.context/v1', episode_id: `web-${Date.now()}`,
      seed: 0x515154, ruleset: 'bun', max_ticks: null, metadata: {},
    });
  }

  function reset() {
    sim = new QQT.Sim(Date.now() >>> 0);
    sim.reset(level);
    if (opponentSelect.value === 'bun.browser_model' && !loadedModel) {
      opponentSelect.value = 'bun.tactical_v2';
    }
    if (matchMode.value === 'model-vs-rule' && !loadedModel) {
      matchMode.value = 'human-vs-opponent';
      modelStatus.textContent = '请先从模型列表选择并加载一个模型';
    }
    resetBot();
    bombQueued = false;
    renderer.reset();
  }

  function humanAction() {
    const move = QQTControls.moveForHeld(held);
    const action = [move, bombQueued ? 1 : 0, 0, 0];
    bombQueued = false;
    return action;
  }

  async function loadPublishedModel(row) {
    modelStatus.textContent = `正在下载：${row.display_name}…`;
    const response = await fetch(QQTModelCatalog.modelUrl(row), { cache: 'no-store' });
    if (!response.ok) throw new Error(`模型下载失败 HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== row.bytes) throw new Error('模型大小校验失败');
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)))
      .map((value) => value.toString(16).padStart(2, '0')).join('');
    if (digest !== row.sha256) throw new Error('模型 SHA-256 校验失败');
    loadedModel = instantiateModel(JSON.parse(new TextDecoder().decode(buffer)));
    modelStatus.textContent = `已加载：${row.display_name}`;
    modelDetails.textContent = `${row.candidate} · cycle ${row.cycle} · score ${row.score.toFixed(4)} · ${(row.bytes / 1048576).toFixed(1)} MiB`;
    matchMode.value = 'model-vs-rule';
    reset();
  }

  async function loadCatalog() {
    try {
      const response = await fetch('models.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      publishedModels = QQTModelCatalog.validateManifest(await response.json());
      publishedModel.replaceChildren(new Option('选择已发布模型', ''));
      for (const row of publishedModels) publishedModel.add(new Option(row.display_name, row.id));
      modelDetails.textContent = `当前提供 ${publishedModels.length} 个评估候选；选择后才下载权重。`;
    } catch (error) {
      publishedModel.replaceChildren(new Option('暂无已发布模型', ''));
      modelDetails.textContent = `模型列表读取失败：${error.message}`;
    }
  }

  function render(now = performance.now()) {
    renderer.render(sim, now);
    status.textContent = JSON.stringify({
      mode: QQTModelCatalog.matchLabel(matchMode.value),
      tick: sim.t,
      player_or_model_alive: sim.alive[0],
      rule_bot_alive: sim.alive[1],
      bun_score: sim.bunScore,
      carrying: sim.bunCarried,
      winner: sim.done ? sim.winner : null,
    }, null, 2);
  }

  async function tick() {
    if (ticking) return;
    ticking = true;
    try {
      if (!sim.done) {
        const first = matchMode.value === 'model-vs-rule'
          ? await Promise.resolve(loadedModel.act(sim, 0, modelRng)) : humanAction();
        const state = QQTBunRuleBot.stateFromSim(sim);
        const observation = {
          schema: 'qqt.bot.observation/v1', tick: sim.t, state,
          legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2], metadata: {},
        };
        const action = await Promise.resolve(activeBot.act(observation, 1, modelRng));
        const info = sim.step([[Number(first[0]), Number(first[1])], [action.move, action.ability]]);
        renderer.addExplosion(info, performance.now());
        if (activeBot.observe_transition) {
          activeBot.observe_transition(info, {
            ...observation, tick: sim.t, state: QQTBunRuleBot.stateFromSim(sim),
          }, 1);
        }
      }
    } finally {
      ticking = false;
    }
  }

  window.addEventListener('keydown', (event) => {
    if ([...QQTControls.MOVEMENT_KEYS, 'KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space'].includes(event.code)) event.preventDefault();
    held.add(event.code);
    if (event.code === 'Space') bombQueued = true;
    if (event.code === 'KeyR') reset();
  });
  window.addEventListener('keyup', (event) => held.delete(event.code));
  restart.addEventListener('click', reset);
  modelFile.addEventListener('change', async () => {
    const file = modelFile.files && modelFile.files[0];
    if (!file) return;
    try {
      const document = JSON.parse(await file.text());
      loadedModel = instantiateModel(document);
      opponentSelect.value = 'bun.browser_model';
      modelStatus.textContent = `已加载：${document.meta.display_name || document.meta.name || file.name}`;
      reset();
    } catch (error) {
      loadedModel = null;
      opponentSelect.value = 'bun.tactical_v2';
      modelStatus.textContent = `加载失败：${error.message}`;
    }
  });
  publishedModel.addEventListener('change', async () => {
    const row = publishedModels.find((item) => item.id === publishedModel.value);
    if (!row) return;
    try { await loadPublishedModel(row); }
    catch (error) { loadedModel = null; modelStatus.textContent = `加载失败：${error.message}`; reset(); }
  });
  matchMode.addEventListener('change', reset);
  opponentSelect.addEventListener('change', reset);
  await loadCatalog();
  reset();
  function animationFrame(now) {
    render(now);
    requestAnimationFrame(animationFrame);
  }
  requestAnimationFrame(animationFrame);
  setInterval(tick, 100);
})();
