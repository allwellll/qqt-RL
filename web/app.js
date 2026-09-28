'use strict';

(async function startBunArena() {
  const canvas = document.getElementById('game');
  const status = document.getElementById('status');
  const restart = document.getElementById('restart');
  const opponentSelect = document.getElementById('opponent');
  const modelFile = document.getElementById('model-file');
  const modelStatus = document.getElementById('model-status');
  const levelList = await fetch('assets/maps/levels.json').then((response) => response.json());
  const level = levelList.find((item) => item.qqt_id === 806);
  if (!level) throw new Error('Bun06 level missing');
  const visualAssets = await QQTVisual.loadAssets(level);
  const renderer = QQTVisual.createRenderer(canvas, level, visualAssets);

  const held = new Set();
  const modelRng = QQT.mulberry32(0x515154);
  let bombQueued = false;
  let loadedModel = null;
  let activeBot = null;
  let ticking = false;
  let sim;

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
    const botId = opponentSelect.value;
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
    resetBot();
    bombQueued = false;
    renderer.reset();
  }

  function humanAction() {
    let move = QQT.MOVE_IDLE;
    if (held.has('KeyW')) move = QQT.MOVE_UP;
    else if (held.has('KeyS')) move = QQT.MOVE_DOWN;
    else if (held.has('KeyA')) move = QQT.MOVE_LEFT;
    else if (held.has('KeyD')) move = QQT.MOVE_RIGHT;
    const action = [move, bombQueued ? 1 : 0, 0, 0];
    bombQueued = false;
    return action;
  }

  function render(now = performance.now()) {
    renderer.render(sim, now);
    status.textContent = JSON.stringify({
      tick: sim.t,
      human_alive: sim.alive[0],
      bot_alive: sim.alive[1],
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
        const human = humanAction();
        const state = QQTBunRuleBot.stateFromSim(sim);
        const observation = {
          schema: 'qqt.bot.observation/v1', tick: sim.t, state,
          legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2], metadata: {},
        };
        const action = await Promise.resolve(activeBot.act(observation, 1, modelRng));
        const opponent = [action.move, action.ability];
        const info = sim.step([human, opponent]);
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
    if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space'].includes(event.code)) event.preventDefault();
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
      const arch = document.meta && document.meta.arch;
      loadedModel = arch === 'transformer' ? new QQT.TransformerModel(document)
        : arch === 'cnn' ? new QQT.CNNModel(document) : new QQT.MLPModel(document);
      opponentSelect.value = 'bun.browser_model';
      modelStatus.textContent = `已加载：${document.meta.display_name || document.meta.name || file.name}`;
      reset();
    } catch (error) {
      loadedModel = null;
      opponentSelect.value = 'bun.tactical_v2';
      modelStatus.textContent = `加载失败：${error.message}`;
    }
  });
  opponentSelect.addEventListener('change', reset);
  reset();
  function animationFrame(now) {
    render(now);
    requestAnimationFrame(animationFrame);
  }
  requestAnimationFrame(animationFrame);
  setInterval(tick, 100);
})();
