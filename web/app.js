'use strict';

(async function startBunArena() {
  const canvas = document.getElementById('game');
  const status = document.getElementById('status');
  const restart = document.getElementById('restart');
  const opponentSelect = document.getElementById('opponent');
  const mapSelect = document.getElementById('map-select');
  const matchMode = document.getElementById('match-mode');
  const publishedModel = document.getElementById('published-model');
  const modelDetails = document.getElementById('model-details');
  const modelProgressWrap = document.getElementById('model-progress-wrap');
  const modelProgress = document.getElementById('model-progress');
  const modelProgressText = document.getElementById('model-progress-text');
  const modelFile = document.getElementById('model-file');
  const modelStatus = document.getElementById('model-status');
  const replaySelect = document.getElementById('replay-select');
  const replayToggle = document.getElementById('replay-toggle');
  const replayRestart = document.getElementById('replay-restart');
  const replaySpeed = document.getElementById('replay-speed');
  const replaySeek = document.getElementById('replay-seek');
  const replayStatus = document.getElementById('replay-status');
  const levelList = await fetch('assets/maps/levels.json').then((response) => response.json());
  const baseLevel = levelList.find((item) => item.qqt_id === 806);
  if (!baseLevel) throw new Error('Bun06 level missing');
  const trainingLevel = baseLevel;                                    // 训练实际采样的真实 806 抢包子图（带砖/道具/包子屋）
  const arenaLevel = QQTTacticalArena.buildTacticalArena(baseLevel);  // 清空砖块的空场能力测试
  let level = trainingLevel;
  let renderer = null;
  function selectedLevel() { return mapSelect.value === 'arena' ? arenaLevel : trainingLevel; }
  // 切换地图需重建资源与渲染器（不同图砖块/精灵集合不同）；调用方随后自行 reset。
  async function useMap(target) {
    level = target;
    const visualAssets = await QQTVisual.loadAssets(level);
    renderer = QQTVisual.createRenderer(canvas, level, visualAssets);
  }
  await useMap(trainingLevel);

  const held = new Set();
  const modelRng = QQT.mulberry32(0x515154);
  let bombQueued = false;
  let loadedModel = null;
  let publishedModels = [];
  let activeBot = null;
  let ticking = false;
  let replayCatalog = [];
  let replayDocument = null;
  let replayIndex = 0;
  let replayPlaying = false;
  let replayAccumulator = 0;
  let sim;
  // 渲染插值：两个逻辑 tick(100ms) 之间线性插值角色位置 → 60fps 顺滑。
  const TICK_MS = 100;
  const prevPos = new Float64Array(4);
  const curPos = new Float64Array(4);
  let lastTickT = performance.now();
  function snapMotion() { prevPos.set(sim.pos); curPos.set(sim.pos); lastTickT = performance.now(); }
  function motionState() {
    // model-vs-rule 观战 / 回放无本地人类 → 两个角色都插值；人类对战时 pid0 不插值(输入即时)。
    const humanPid = (replayDocument || matchMode.value === 'model-vs-rule') ? -1 : 0;
    return { prevPos, curPos, lastTickT, tickMs: TICK_MS, humanPid };
  }

  function instantiateModel(document) {
    const arch = document.meta && document.meta.arch;
    const model = arch === 'transformer' ? new QQT.TransformerModel(document)
      : arch === 'cnn' ? new QQT.CNNModel(document)
      : arch === 'mlp4' ? new QQT.MLP4Model(document) : new QQT.MLPModel(document);
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

  async function reset() {
    replayDocument = null;
    replayPlaying = false;
    replayToggle.textContent = '播放';
    if (level !== selectedLevel()) await useMap(selectedLevel());
    sim = new QQT.Sim(Date.now() >>> 0);
    sim.reset(level);
    // 人类手动控制 pid0 时启用降敏拐角滑移（对齐原 gpu sim 手感，减少贴墙/贴泡误滑）；
    // 观战(model-vs-rule) pid0 是模型 → 保持完整 _steer 与训练一致。
    sim.steerReduced = [matchMode.value !== 'model-vs-rule', false];
    if (opponentSelect.value === 'bun.browser_model' && !loadedModel) {
      opponentSelect.value = 'bun.tactical_v2';
    }
    if (matchMode.value === 'model-vs-rule' && !loadedModel) {
      matchMode.value = 'human-vs-opponent';
      modelStatus.textContent = '请先从模型列表选择并加载一个模型';
    }
    resetBot();
    bombQueued = false;
    snapMotion();
    renderer.reset();
  }

  function resetReplay() {
    if (!replayDocument) return;
    sim = new QQT.Sim(replayDocument.meta.seed);
    sim.reset(level);
    replayIndex = 0;
    replayAccumulator = 0;
    replaySeek.max = String(replayDocument.actions.length);
    replaySeek.value = '0';
    snapMotion();
    renderer.reset();
  }

  function stepReplay() {
    if (!replayDocument || replayIndex >= replayDocument.actions.length) {
      replayPlaying = false;
      replayToggle.textContent = '播放';
      return;
    }
    const row = replayDocument.actions[replayIndex++];
    prevPos.set(sim.pos);
    const info = sim.step([[row[0], row[1], row[2]], [row[3], row[4], row[5]]]);
    curPos.set(sim.pos); lastTickT = performance.now();
    renderer.addExplosion(info, performance.now());
    replaySeek.value = String(replayIndex);
    if (replayIndex >= replayDocument.actions.length) {
      replayPlaying = false;
      replayToggle.textContent = '播放';
    }
  }

  function seekReplay(target) {
    resetReplay();
    while (replayIndex < target) stepReplay();
  }

  function humanAction() {
    const move = QQTControls.moveForHeld(held);
    const action = [move, bombQueued ? 1 : 0, 0, 0];
    bombQueued = false;
    return action;
  }

  async function loadPublishedModel(row) {
    publishedModel.disabled = true;
    modelProgressWrap.hidden = false;
    modelProgress.removeAttribute('value');
    modelProgressText.textContent = '正在连接模型文件…';
    modelStatus.textContent = `正在加载：${row.display_name}`;
    const response = await fetch(QQTModelCatalog.modelUrl(row), { cache: 'no-store' });
    const buffer = await QQTModelLoader.readResponseWithProgress(response, (loaded, total) => {
      if (total > 0) modelProgress.value = loaded * 100 / total;
      else modelProgress.removeAttribute('value');
      modelProgressText.textContent = QQTModelLoader.progressText(loaded, total);
    });
    if (buffer.byteLength !== row.bytes) throw new Error('模型大小校验失败');
    modelProgress.removeAttribute('value');
    modelProgressText.textContent = '正在校验模型 SHA-256…';
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)))
      .map((value) => value.toString(16).padStart(2, '0')).join('');
    if (digest !== row.sha256) throw new Error('模型 SHA-256 校验失败');
    modelProgressText.textContent = '正在解析并初始化模型…';
    await new Promise((resolve) => setTimeout(resolve, 0));
    loadedModel = instantiateModel(JSON.parse(new TextDecoder().decode(buffer)));
    modelProgress.value = 100;
    modelProgressText.textContent = '模型加载完成，正在开始观战';
    publishedModel.disabled = false;
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

  async function loadReplayCatalog() {
    try {
      const response = await fetch('replays.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const document = await response.json();
      if (document.schema !== 'qqt.replays/v1' || !Array.isArray(document.replays)) throw new Error('录像目录格式错误');
      replayCatalog = document.replays;
      replaySelect.replaceChildren(new Option('选择离线实战录像', ''));
      for (const row of replayCatalog) {
        replaySelect.add(new Option(`${row.model_id} · seed ${row.seed} · ${row.summary.ticks} ticks`, row.id));
      }
      replayStatus.textContent = `当前提供 ${replayCatalog.length} 局固定seed离线推理录像；每局仅约数KB动作流。`;
    } catch (error) {
      replaySelect.replaceChildren(new Option('录像列表读取失败', ''));
      replayStatus.textContent = `录像列表读取失败：${error.message}`;
    }
  }

  async function loadReplay(row) {
    const response = await fetch(`replays/${row.file}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    replayDocument = QQTReplay.validateReplay(await response.json());
    await useMap(arenaLevel);   // 离线录像在空场竞技场上生成，回放必须用同一张图
    resetReplay();
    replayPlaying = true;
    replayToggle.textContent = '暂停';
    const s = replayDocument.summary;
    replayStatus.textContent = `${row.model_id}｜seed ${row.seed}｜模型/规则Bot放泡 ${s.model_bombs}/${s.rule_bombs}｜活动格 ${s.model_unique_cells}/${s.rule_unique_cells}｜胜者 ${s.winner == null ? '未决' : s.winner}`;
  }

  function render(now = performance.now()) {
    renderer.render(sim, now, motionState());
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
      if (replayDocument) {
        if (replayPlaying) {
          replayAccumulator += Number(replaySpeed.value);
          while (replayAccumulator >= 1 && replayPlaying) { stepReplay(); replayAccumulator -= 1; }
        }
      } else if (!sim.done) {
        const first = matchMode.value === 'model-vs-rule'
          ? await Promise.resolve(loadedModel.act(sim, 0, modelRng)) : humanAction();
        const state = QQTBunRuleBot.stateFromSim(sim);
        const observation = {
          schema: 'qqt.bot.observation/v1', tick: sim.t, state,
          legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2], metadata: {},
        };
        const action = await Promise.resolve(activeBot.act(observation, 1, modelRng));
        prevPos.set(sim.pos);
        const info = sim.step([[Number(first[0]), Number(first[1])], [action.move, action.ability]]);
        curPos.set(sim.pos); lastTickT = performance.now();
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
  replaySelect.addEventListener('change', async () => {
    const row = replayCatalog.find((item) => item.id === replaySelect.value);
    if (!row) return;
    try { await loadReplay(row); } catch (error) { replayStatus.textContent = `录像加载失败：${error.message}`; }
  });
  replayToggle.addEventListener('click', () => {
    if (!replayDocument) return;
    if (replayIndex >= replayDocument.actions.length) resetReplay();
    replayPlaying = !replayPlaying;
    replayToggle.textContent = replayPlaying ? '暂停' : '播放';
  });
  replayRestart.addEventListener('click', () => {
    if (!replayDocument) return;
    resetReplay(); replayPlaying = true; replayToggle.textContent = '暂停';
  });
  replaySeek.addEventListener('input', () => { if (replayDocument) seekReplay(Number(replaySeek.value)); });
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
    catch (error) {
      loadedModel = null;
      publishedModel.disabled = false;
      modelProgressWrap.hidden = false;
      modelProgress.removeAttribute('value');
      modelProgressText.textContent = `加载失败：${error.message}`;
      modelStatus.textContent = `加载失败：${error.message}`;
      reset();
    }
  });
  matchMode.addEventListener('change', reset);
  opponentSelect.addEventListener('change', reset);
  mapSelect.addEventListener('change', reset);
  await loadCatalog();
  reset();
  await loadReplayCatalog();
  function animationFrame(now) {
    render(now);
    requestAnimationFrame(animationFrame);
  }
  requestAnimationFrame(animationFrame);
  setInterval(tick, 100);
})();
