'use strict';

(async function startBunArena() {
  const canvas = document.getElementById('game');
  const status = document.getElementById('status');
  const restart = document.getElementById('restart');
  const opponentSelect = document.getElementById('opponent');
  const mapSelect = document.getElementById('map-select');
  const matchMode = document.getElementById('match-mode');
  const teamMode = document.getElementById('team-mode');
  const characterSelect = document.getElementById('character');
  const characterPortrait = document.getElementById('character-portrait');
  try {
    const savedCharacter = localStorage.getItem('qqt.character');
    if (['pipi', 'maomao'].includes(savedCharacter)) characterSelect.value = savedCharacter;
  } catch (_) {}
  function updateCharacterPortrait() {
    characterPortrait.src = `assets/native/${characterSelect.value}_portrait.png`;
    characterPortrait.alt = characterSelect.selectedOptions[0].textContent;
  }
  updateCharacterPortrait();
  const modelOptions = document.getElementById('model-options');
  const modelDetails = document.getElementById('model-details');
  const modelProgressWrap = document.getElementById('model-progress-wrap');
  const modelProgress = document.getElementById('model-progress');
  const modelProgressText = document.getElementById('model-progress-text');
  const modelStatus = document.getElementById('model-status');
  const replaySelect = document.getElementById('replay-select');
  const replayToggle = document.getElementById('replay-toggle');
  const replayRestart = document.getElementById('replay-restart');
  const replaySpeed = document.getElementById('replay-speed');
  const replaySeek = document.getElementById('replay-seek');
  const replayStatus = document.getElementById('replay-status');
  const loading = document.getElementById('loading');
  const loadingText = document.getElementById('loading-text');
  const loadingProgress = document.getElementById('loading-progress');
  const leaderboard = QQTLeaderboard.mount(document, { config: QQTLeaderboardConfig,
    storage: { getItem: key => localStorage.getItem(key), setItem: (key, value) => localStorage.setItem(key, value) },
    crypto, fetch: (...args) => fetch(...args) });
  let leaderboardMatch = null, clientVersion = 'dev';
  fetch('build-info.json', { cache: 'no-store' }).then(r => r.json()).then(info => {
    if (/^[a-f0-9]{40}$/.test(info.commit)) clientVersion = info.commit;
  }).catch(() => {});
  function showLoading(text) {
    loading.classList.remove('done');
    loadingText.textContent = text;
    loadingProgress.value = 0;
  }
  function loadingStep(done, total) {
    loadingProgress.value = total ? done * 100 / total : 0;
    loadingText.textContent = `正在加载素材 ${done}/${total}`;
  }
  // 首帧真正画出后才撤掉遮罩，避免先露出一块黑画布。
  let assetsPending = true;
  function hideLoading() { if (!assetsPending) loading.classList.add('done'); }
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
    assetsPending = true;
    showLoading('正在加载地图与素材…');
    try {
      const visualAssets = await QQTVisual.loadAssets(level, loadingStep);
      renderer = QQTVisual.createRenderer(canvas, level, visualAssets);
    } catch (error) {
      loadingText.textContent = `加载失败：${error.message}，请刷新重试`;
      throw error;
    }
    loadingText.textContent = '素材就绪，正在开局…';
    assetsPending = false;
  }
  await useMap(trainingLevel);

  const held = new Set();
  const modelRng = QQT.mulberry32(0x515154);
  // 按键瞬间记下所在格（-1=未按）；下一个 10Hz tick 在该格放泡/放道具。
  let bombCell = -1;
  let itemCell = -1;
  let itemSlot = 0;
  let loadedModel = null;
  let loadedModelId = null;
  let publishedModels = [];
  let activeBot = null;
  // 组队模式下 pid>=2 的猎手（pid1 仍是 activeBot）。
  let extraBots = [];
  let ticking = false;
  let replayCatalog = [];
  let replayDocument = null;
  let replayIndex = 0;
  let replayPlaying = false;
  let replayAccumulator = 0;
  let sim;
  const sound = QQTSound.createPlayer();
  sound.load();
  const soundToggle = document.getElementById('sound-toggle');
  // 每名玩家当前的移动意图（MOVE_*），渲染器据此定朝向：顶墙时也面朝按键方向。
  let intents = [QQT.MOVE_IDLE, QQT.MOVE_IDLE];
  // 渲染插值：对手(pid1，10Hz)在两个逻辑 tick 间线性插值 → 60fps 顺滑。
  // 本地人类(pid0)不插值：由 rAF 逐帧 frameStep 连续移动，本身就是每帧真实位置。
  const TICK_MS = 100;
  let prevPos = new Float64Array(4);
  let curPos = new Float64Array(4);
  let lastTickT = performance.now();
  let prevFrame = performance.now();
  function snapMotion() {
    if (prevPos.length !== sim.pos.length) {
      prevPos = new Float64Array(sim.pos.length); curPos = new Float64Array(sim.pos.length);
      intents = Array.from({ length: sim.nPlayers }, () => QQT.MOVE_IDLE);
    }
    intents.fill(QQT.MOVE_IDLE);
    prevPos.set(sim.pos); curPos.set(sim.pos); lastTickT = performance.now(); prevFrame = performance.now();
  }
  // 1v2 → [红, 蓝, 蓝]；2v2 → [红, 蓝, 红, 蓝]（pid1 恒为第一个敌人）。
  const TEAM_LAYOUTS = { '1v1': [0, 1], '1v2': [0, 1, 1], '2v2': [0, 1, 0, 1] };
  function teamLayout() { return localHumanControls() ? (TEAM_LAYOUTS[teamMode.value] || TEAM_LAYOUTS['1v1']) : TEAM_LAYOUTS['1v1']; }
  // 本地人类操控 pid0（非回放、非观战 model-vs-rule）时 humanPid=0 → 渲染 raw；否则 -1（两方皆插值）。
  function localHumanControls() { return !replayDocument && matchMode.value !== 'model-vs-rule'; }
  function motionState() {
    return { prevPos, curPos, lastTickT, tickMs: TICK_MS, humanPid: localHumanControls() ? 0 : -1, intents,
      characters: localHumanControls() ? [characterSelect.value] : [] };
  }

  // rAF 逐帧推进本地人类 pid0：sim.frameStep 内走原版像素移动（含 6px 拐角修正、泡泡 3px 入口带），
  // 空闲帧也调用以推进穿泡计时。
  function stepHumanFrame(now) {
    if (!sim || !localHumanControls()) { prevFrame = now; return; }
    const dt = Math.min((now - prevFrame) / 1000 || 0, 0.25);
    prevFrame = now;
    if (sim.done) return;
    const move = QQTControls.moveForHeld(held);
    intents[0] = sim.alive[0] ? sim.playerMoveDirection(0, move) : QQT.MOVE_IDLE;
    sim.frameStep(0, move, dt);
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
      hunter: QQTBunHunterBot,
      coopHunter: QQTBunCoopHunterBot,
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

  const MODEL_PREFIX = 'model:';
  const isModelChoice = (value) => value.startsWith(MODEL_PREFIX);
  // 组队只支持猎手；规则 Bot 与训练模型按 1v1 训练。
  const teamCapable = (value) => value.startsWith('bun.hunter') || value.startsWith('bun.coop_hunter');

  function resetBot() {
    if (activeBot && activeBot.close) activeBot.close();
    for (const bot of extraBots) if (bot && bot.close) bot.close();
    const registry = createRegistry();
    // 选项值形如 "bun.hunter@hard"：@ 后为难度配置。
    const choice = matchMode.value === 'model-vs-rule' ? 'bun.tactical_v2'
      : isModelChoice(opponentSelect.value) ? 'bun.browser_model' : opponentSelect.value;
    const [botId, difficulty] = choice.split('@');
    const context = (seed) => ({
      schema: 'qqt.bot.context/v1', episode_id: `web-${Date.now()}`,
      seed, ruleset: 'bun', max_ticks: null, metadata: {},
    });
    activeBot = registry.create(botId, difficulty ? { difficulty } : {});
    activeBot.reset(context(Date.now() >>> 0));
    extraBots = [];
    for (let pid = 2; pid < sim.nPlayers; pid++) {
      extraBots[pid] = registry.create(botId, difficulty ? { difficulty } : {});
      extraBots[pid].reset(context((Date.now() + pid * 7919) >>> 0));
    }
  }

  async function reset() {
    leaderboard.clearSettlement();
    QQTLeaderboard.renderSettlement(document, leaderboard.state());
    replayDocument = null;
    replayPlaying = false;
    replayToggle.textContent = '播放';
    if (level !== selectedLevel()) await useMap(selectedLevel());
    sim = new QQT.Sim(Date.now() >>> 0);
    const wantModel = isModelChoice(opponentSelect.value);
    if (wantModel && loadedModelId !== opponentSelect.value.slice(MODEL_PREFIX.length)) opponentSelect.value = 'bun.coop_hunter@hard';
    if (matchMode.value === 'model-vs-rule' && !loadedModel) {
      matchMode.value = 'human-vs-opponent';
      modelStatus.textContent = '请先在「策略」中选择一个训练模型';
    }
    if (teamMode.value !== '1v1' && !teamCapable(opponentSelect.value)) opponentSelect.value = 'bun.coop_hunter@hard';
    // 原版道具栏/糖泡只在真人对局开启；模型评测与录像保持训练规则。
    const native = localHumanControls();
    sim.reset(level, { nativeItems: native, nativeTrap: native, teams: teamLayout(),
      bananaSlideSpeedPx: native ? 480 : undefined });
    const [opponent, difficulty] = opponentSelect.value.split('@');
    leaderboardMatch = native ? leaderboard.begin({ seed: sim.seed, opponent,
      difficulty: difficulty || (isModelChoice(opponent) ? 'model' : 'fixed'),
      mode: teamMode.value, map_id: mapSelect.value }) : null;
    resetBot();
    bombCell = -1;
    itemCell = -1;
    itemSlot = 0;
    snapMotion();
    renderer.reset();
  }

  function resetReplay() {
    if (!replayDocument) return;
    leaderboardMatch = null;
    leaderboard.clearSettlement();
    sim = new QQT.Sim(replayDocument.meta.seed);
    sim.reset(level);
    replayIndex = 0;
    replayAccumulator = 0;
    replaySeek.max = String(replayDocument.actions.length);
    replaySeek.value = '0';
    snapMotion();
    renderer.reset();
  }

  function setTickIntents(move0, move1) {
    const moves = [move0, move1];
    for (let pid = 0; pid < 2; pid++) {
      intents[pid] = sim.alive[pid] ? sim.playerMoveDirection(pid, Number(moves[pid])) : QQT.MOVE_IDLE;
    }
  }

  function playEvents(before, info, listenerPid) {
    for (const name of QQTSound.detectEvents(before, sim, info, listenerPid)) sound.play(name);
  }

  function stepReplay(silent = false) {
    if (!replayDocument || replayIndex >= replayDocument.actions.length) {
      replayPlaying = false;
      replayToggle.textContent = '播放';
      return;
    }
    const row = replayDocument.actions[replayIndex++];
    prevPos.set(sim.pos);
    const before = QQTSound.snapshot(sim);
    const info = sim.step([[row[0], row[1], row[2]], [row[3], row[4], row[5]]]);
    curPos.set(sim.pos); lastTickT = performance.now();
    setTickIntents(row[0], row[3]);
    renderer.addExplosion(info, performance.now());
    if (!silent) playEvents(before, info, 0);
    replaySeek.value = String(replayIndex);
    if (replayIndex >= replayDocument.actions.length) {
      replayPlaying = false;
      replayToggle.textContent = '播放';
    }
  }

  function seekReplay(target) {
    resetReplay();
    while (replayIndex < target) stepReplay(true);
  }

  function humanCell() {
    if (!sim || !localHumanControls() || !sim.alive[0]) return -1;
    const [row, column] = sim.centerCell(0);
    return row * 15 + column;
  }

  function humanAction() {
    // 第4位=1：跳过 10Hz 逻辑移动（移动改由 rAF 逐帧 frameStep 连续处理）；放泡仍在中心格生效。
    const action = [QQT.MOVE_IDLE, bombCell >= 0 ? 1 : 0, itemCell >= 0 ? 1 : 0, 1, bombCell, itemCell, itemSlot];
    bombCell = -1;
    itemCell = -1;
    itemSlot = 0;
    return action;
  }

  async function loadPublishedModel(row) {
    opponentSelect.disabled = true;
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
    loadedModelId = row.id;
    modelProgressText.textContent = '模型加载完成';
    opponentSelect.disabled = false;
    modelStatus.textContent = `已加载：${row.display_name}`;
    modelDetails.textContent = `${row.candidate} · cycle ${row.cycle} · score ${row.score.toFixed(4)} · ${(row.bytes / 1048576).toFixed(1)} MiB`;
  }

  async function loadCatalog() {
    try {
      const response = await fetch('models.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      publishedModels = QQTModelCatalog.validateManifest(await response.json());
      modelOptions.replaceChildren(...publishedModels.map((row) => new Option(row.display_name, MODEL_PREFIX + row.id)));
      modelDetails.textContent = `「策略」中提供 ${publishedModels.length} 个训练模型；选择后才下载权重。`;
    } catch (error) {
      modelOptions.replaceChildren();
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
    if (sim.done && localHumanControls() && leaderboardMatch) {
      const match = { ...leaderboardMatch, client_version: clientVersion };
      leaderboardMatch = null;
      void leaderboard.finish(match, { result: sim.winner == null ? 'draw' : sim.winner === sim.team[0] ? 'win' : 'loss',
        gameDurationMs: sim.t * TICK_MS });
    }
    renderer.render(sim, now, motionState());
    QQTLeaderboard.renderSettlement(document, leaderboard.state());
    hideLoading();
    status.textContent = JSON.stringify({
      mode: QQTModelCatalog.matchLabel(matchMode.value),
      tick: sim.t,
      player_or_model_alive: sim.alive[0],
      rule_bot_alive: sim.alive[1],
      bun_score: sim.bunScore,
      carrying: sim.bunCarried,
      held_item: sim.nativeItems
        ? sim.itemSlots.map((slots) => slots.map((slot) => `${['无', '香蕉皮', '慢慢胶'][slot.item] || '?'}×${slot.count}`))
        : sim.heldItem.map((item) => ['无', '香蕉皮', '慢慢胶'][item] || '无'),
      trapped: sim.nativeTrap ? sim.trapped : undefined,
      spawn_protection: sim.nativeTrap ? sim.spawnProtection : undefined,
      team: sim.nPlayers > 2 ? sim.team : undefined,
      bot: [activeBot, ...extraBots.slice(2)].map((bot) => (bot && bot.bot && bot.bot.lastDecision
        ? `${bot.bot.lastDecision.mode} / ${bot.bot.lastDecision.reason}` : undefined)).filter(Boolean).join(' | ') || undefined,
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
          legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2], metadata: { sim },
        };
        const action = await Promise.resolve(activeBot.act(observation, 1, modelRng));
        const extra = [];
        for (let pid = 2; pid < sim.nPlayers; pid++) extra[pid] = await Promise.resolve(extraBots[pid].act(observation, pid, modelRng));
        const botActions = [null, action, ...extra.slice(2)];
        prevPos.set(sim.pos);
        const before = QQTSound.snapshot(sim);
        const info = sim.step([
          [Number(first[0]), Number(first[1]), Number(first[2]) || 0, Number(first[3]) || 0,
            first[4] == null ? -1 : first[4], first[5] == null ? -1 : first[5], first[6] == null ? 0 : first[6]],
          ...botActions.slice(1).map((a) => [a.move, a.ability === 1 ? 1 : 0, a.ability === 2 ? 1 : 0,
            0, -1, -1, a.itemSlot == null ? 0 : a.itemSlot]),
        ]);
        curPos.set(sim.pos); lastTickT = performance.now();
        if (localHumanControls()) {
          for (let pid = 1; pid < sim.nPlayers; pid++) {
            intents[pid] = sim.alive[pid] ? sim.playerMoveDirection(pid, botActions[pid].move) : QQT.MOVE_IDLE;
          }
        } else setTickIntents(first[0], action.move);
        renderer.addExplosion(info, performance.now());
        playEvents(before, info, 0);
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

  const unlockAudio = () => sound.unlock();
  window.addEventListener('pointerdown', unlockAudio);
  window.addEventListener('keydown', (event) => {
    if (event.target.closest && event.target.closest('input, textarea, [contenteditable="true"]')) return;
    unlockAudio();
    if ([...QQTControls.MOVEMENT_KEYS, ...QQTControls.ITEM_KEYS, ...QQTControls.ITEM_SLOT_KEYS,
      ...QQTControls.BOMB_KEYS].includes(event.code)) event.preventDefault();
    held.add(event.code);
    // 同一 tick 内多次按键只保留第一次的位置。
    if (QQTControls.BOMB_KEYS.includes(event.code) && bombCell < 0) bombCell = humanCell();
    if (QQTControls.ITEM_KEYS.includes(event.code) && itemCell < 0) { itemCell = humanCell(); itemSlot = 0; }
    const slotKey = QQTControls.ITEM_SLOT_KEYS.indexOf(event.code);
    if (slotKey >= 0 && itemCell < 0) { itemCell = humanCell(); itemSlot = slotKey; }
    if (event.code === 'KeyR') reset();
  });
  window.addEventListener('keyup', (event) => held.delete(event.code));
  document.addEventListener('focusin', (event) => {
    if (event.target.matches('input, textarea')) { held.clear(); bombCell = -1; itemCell = -1; }
  });
  window.addEventListener('blur', () => {
    held.clear(); bombCell = -1; itemCell = -1;
  });
  restart.addEventListener('click', reset);
  document.getElementById('play-again').addEventListener('click', () => { if (sim.done) reset(); });
  soundToggle.addEventListener('change', () => sound.setEnabled(soundToggle.checked));
  characterSelect.addEventListener('change', () => {
    updateCharacterPortrait();
    try { localStorage.setItem('qqt.character', characterSelect.value); } catch (_) {}
  });
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
  async function onStrategyChange() {
    const value = opponentSelect.value;
    if (!teamCapable(value)) teamMode.value = '1v1';
    const row = isModelChoice(value) ? publishedModels.find((item) => MODEL_PREFIX + item.id === value) : null;
    if (row && loadedModelId !== row.id) {
      try { await loadPublishedModel(row); }
      catch (error) {
        loadedModel = null;
        loadedModelId = null;
        opponentSelect.disabled = false;
        modelProgressWrap.hidden = false;
        modelProgress.removeAttribute('value');
        modelProgressText.textContent = `加载失败：${error.message}`;
        modelStatus.textContent = `加载失败：${error.message}`;
      }
    }
    reset();
  }
  matchMode.addEventListener('change', reset);
  teamMode.addEventListener('change', reset);
  opponentSelect.addEventListener('change', onStrategyChange);
  mapSelect.addEventListener('change', reset);
  await loadCatalog();
  reset();
  await loadReplayCatalog();
  function animationFrame(now) {
    stepHumanFrame(now);
    render(now);
    requestAnimationFrame(animationFrame);
  }
  requestAnimationFrame(animationFrame);
  setInterval(tick, 100);
})();
