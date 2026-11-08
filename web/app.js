'use strict';

(async function startBunArena() {
  performance.mark('qqt:startup');
  QQTPanels.mount(document);
  const canvas = document.getElementById('game');
  const status = document.getElementById('status');
  const restart = document.getElementById('restart');
  const opponentSelect = document.getElementById('opponent');
  const gameMode = document.getElementById('game-mode');
  const characterSelect = document.getElementById('character');
  const characterPortrait = document.getElementById('character-portrait');
  try {
    const savedGameMode = localStorage.getItem('qqt.game-mode');
    if (savedGameMode === 'kill' || savedGameMode === 'carry') gameMode.value = savedGameMode;
  } catch (_) {}
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
  const loadingSource = document.getElementById('loading-source');
  const loadingRetry = document.getElementById('loading-retry');
  let modelPending = false, modelFailed = false;
  loadingRetry.addEventListener('click', () => onStrategyChange());
  const leaderboard = QQTLeaderboard.mount(document, { config: QQTLeaderboardConfig,
    storage: { getItem: key => localStorage.getItem(key), setItem: (key, value) => localStorage.setItem(key, value) },
    crypto, fetch: (...args) => fetch(...args), settlementReady: false, onRestart: () => requestRestart() });
  let leaderboardMatch = null, clientVersion = 'dev';
  fetch('build-info.json', { cache: 'no-store' }).then(r => r.json()).then(info => {
    if (/^[a-f0-9]{40}$/.test(info.commit)) clientVersion = info.commit;
  }).catch(() => {});
  function showLoading(text) {
    leaderboard.setSettlementReady(false);
    loading.classList.remove('done');
    loadingText.textContent = text;
    loadingProgress.value = 0;
    loadingRetry.hidden = true;
    loadingSource.textContent = '';
  }
  function loadingStep(done, total) {
    loadingProgress.value = total ? done * 100 / total : 0;
    loadingText.textContent = `正在加载素材 ${done}/${total}`;
  }
  // 首帧真正画出后才撤掉遮罩，避免先露出一块黑画布。
  let assetsPending = true;
  function hideLoading() {
    if (!assetsPending && !modelPending && !modelFailed) {
      if (!loading.classList.contains('done')) performance.mark('qqt:first-frame');
      loading.classList.add('done');
      leaderboard.setSettlementReady(true);
    }
  }
  const levelList = await fetch('assets/maps/levels.json').then((response) => response.json());
  const baseLevel = levelList.find((item) => item.qqt_id === 806);
  if (!baseLevel) throw new Error('Bun06 level missing');
  const trainingLevel = baseLevel;                                    // 训练实际采样的真实 806 抢包子图（带砖/道具/包子屋）
  const arenaLevel = QQTTacticalArena.buildTacticalArena(baseLevel);  // 清空砖块的空场能力测试
  let level = trainingLevel;
  let renderer = null, touchItemAssets = null;
  function selectedLevel() { return trainingLevel; }
  // 切换地图需重建资源与渲染器（不同图砖块/精灵集合不同）；调用方随后自行 reset。
  async function useMap(target) {
    level = target;
    performance.mark('qqt:assets-start');
    assetsPending = true;
    showLoading('正在加载地图与素材…');
    try {
      const visualAssets = await QQTVisual.loadAssets(level, loadingStep);
      touchItemAssets = visualAssets.items;
      renderer = QQTVisual.createRenderer(canvas, level, visualAssets);
    } catch (error) {
      loadingText.textContent = `加载失败：${error.message}，请刷新重试`;
      throw error;
    }
    performance.mark('qqt:assets-ready');
    loadingText.textContent = '正在读取模型目录…';
    assetsPending = false;
  }
  await useMap(trainingLevel);

  const held = new Set();
  let modelRng = QQT.mulberry32(0x515154);
  let botRng = QQT.mulberry32(0x515154 ^ 0x5CA17);
  // 按键瞬间记下所在格（-1=未按）；下一个 10Hz tick 在该格放泡/放道具。
  let bombCell = -1;
  let itemCell = -1;
  let itemSlot = 0;
  const touchControls = document.getElementById('touch-controls');
  let touchItemPage = 0;
  const touchItemButtons = [...touchControls.querySelectorAll('[data-touch-item]')];
  const touchItemPager = document.getElementById('touch-item-page');
  touchItemPager.addEventListener('click', () => {
    const view = QQTControls.touchItemPage(sim.itemSlots?.[0] || [], touchItemPage);
    touchItemPage = (view.page + 1) % view.pages; updateTouchItems();
  });
  function updateTouchItems() {
    const view = QQTControls.touchItemPage(sim.itemSlots?.[0] || [], touchItemPage);
    touchItemPage = view.page;
    for (const [i, value] of view.buttons.entries()) {
      const button = touchItemButtons[i], slot = value.slot;
      button.dataset.controlKey = value.key;
      button.dataset.slotIndex = String(value.index);
      button.setAttribute('aria-disabled', slot ? 'false' : 'true');
      const name = slot ? (['', '香蕉皮', '慢慢胶'][slot.item] || '道具') : '空槽';
      button.setAttribute('aria-label', `槽位${value.number || '无'}，${name}${slot ? `，${slot.count}个，点击直接放置` : ''}`);
      button.querySelector('.touch-slot-number').textContent = value.number || '—';
      button.querySelector('.touch-slot-count').textContent = slot ? `×${slot.count}` : '空';
      const icon = button.querySelector('canvas'), context = icon.getContext('2d');
      context.clearRect(0, 0, icon.width, icon.height);
      const image = touchItemAssets?.[QQTVisual.heldItemSpriteKey(slot?.item)]?.frames[0];
      if (image) { const k = 30 / Math.max(image.width, image.height);
        context.drawImage(image, (32-image.width*k)/2, (32-image.height*k)/2, image.width*k, image.height*k); }
    }
    touchItemPager.hidden = view.pages <= 1;
    touchItemPager.textContent = `槽${view.page*3+1}–${Math.min(view.page*3+3, 7)} / ${sim.itemSlots?.[0]?.length || 0}　换页 ▶`;
  }
  const releaseTouchControls = QQTControls.bindTouchControls(touchControls, held, key => {
    if (!localHumanControls() || sim.done) { held.delete(key); return; }
    if (QQTControls.BOMB_KEYS.includes(key) && bombCell < 0) bombCell = humanCell();
    const slot = QQTControls.ITEM_SLOT_KEYS.indexOf(key);
    if (slot >= 0 && sim.itemSlots?.[0]?.[slot] && itemCell < 0) { itemCell = humanCell(); itemSlot = slot; }
  });
  let loadedModel = null;
  let loadedModelId = null;
  let publishedModels = [];
  let defaultModelId = 'items-l2-sp-u023302-tau03';
  let activeBot = null;
  let ticking = false;
  let sessionRevision = 0;
  let replayCatalog = [];
  let replayDocument = null;
  let legacyReplayApi = null;
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
  function localHumanControls() { return !replayDocument; }
  function motionState() {
    return { prevPos, curPos, lastTickT, tickMs: TICK_MS, humanPid: localHumanControls() ? 0 : -1, intents,
      settlement: QQTLeaderboard.settlementText(leaderboard.state().settlement),
      hideResult: localHumanControls() && !!leaderboard.state().settlement?.closed,
      characters: localHumanControls() ? [characterSelect.value, 'magician'] : [],
      teamRings: localHumanControls() };
  }

  // rAF 逐帧推进本地人类 pid0：sim.frameStep 内走原版像素移动（含 6px 拐角修正、泡泡 3px 入口带），
  // 空闲帧也调用以推进穿泡计时。
  function stepHumanFrame(now) {
    if (!sim || modelPending || modelFailed || !localHumanControls()) { prevFrame = now; return; }
    const dt = Math.min((now - prevFrame) / 1000 || 0, 0.25);
    prevFrame = now;
    if (sim.done) return;
    const move = QQTControls.moveForHeld(held);
    intents[0] = sim.alive[0] ? sim.playerMoveDirection(0, move) : QQT.MOVE_IDLE;
    sim.frameStep(0, move, dt);
  }

  function instantiateModel(document, row) {
    if(document.meta?.ability_actions>3 && document.meta.schema_version!=='five-item-slots-v1')throw Error('Unknown model action schema');
    const arch = document.meta && document.meta.arch;
    const model = arch === 'transformer' ? new QQT.TransformerModel(document)
      : arch === 'cnn' ? new QQT.CNNModel(document)
      : arch === 'mlp4' ? new QQT.MLP4Model(document) : new QQT.MLPModel(document);
    model.greedy = true;
    const consumer = document.meta?.schema_version==='five-item-slots-v1'
      ? QQTFiveSlotsModel.attach(model,document.meta) : QQTR3Inputs.attach(model);
    return QQTModelSampling.attach(consumer,row);
  }

  function createRegistry() {
    return QQTBots.createDefaultRegistry({
      modelFactory: loadedModel ? () => ({
        reset(context) {},
        async act(observation, playerId, rng) {
          const raw = await Promise.resolve(loadedModel.act(sim, playerId, modelRng));
          return QQTBots.validateAction({ move: Number(raw[0]), ability: raw[2] ? 2 : Number(raw[1]),
            ...(raw[2] && raw[6] >= 0 ? {itemSlot: Number(raw[6])} : {}) });
        },
        close() {},
      }) : null,
    });
  }

  const MODEL_PREFIX = 'model:';
  const isModelChoice = (value) => value.startsWith(MODEL_PREFIX);
  function resetBot() {
    if (activeBot?.close) activeBot.close();
    activeBot = loadedModel ? createRegistry().create('bun.browser_model') : null;
    activeBot?.reset({schema:'qqt.bot.context/v1', episode_id:`web-${Date.now()}`,
      seed:sim.seed >>> 0, ruleset:'bun', max_ticks:null, metadata:{}});
  }

  let requestedRuleset = QQTRulesets.current;
  async function reset(preserveCard = false) {
    performance.mark('qqt:reset-start');
    if (leaderboard.state().submitting || leaderboard.state().profileSubmitting) return;
    sessionRevision++;
    leaderboardMatch = null;
    leaderboard.setSettlementReady(false);
    if (preserveCard !== true) leaderboard.clearSettlement();
    QQTLeaderboard.renderSettlement(document, leaderboard.state());
    QQTLeaderboard.renderUpgradeProfile(document, leaderboard.state());
    replayDocument = null;
    replayPlaying = false;
    replayToggle.textContent = '播放';
    if (level !== selectedLevel()) await useMap(selectedLevel());
    // Switch rules only when rebuilding the whole match, after any map load.
    QQTRulesets.select(requestedRuleset);
    const requestedSeed = new URLSearchParams(location.search).get('seed');
    const episodeSeed = requestedSeed != null && /^\d+$/.test(requestedSeed) ? Number(requestedSeed) >>> 0 : Date.now() >>> 0;
    sim = new QQT.Sim(episodeSeed);
    modelRng = QQT.mulberry32((episodeSeed ^ (loadedModel?.samplingProfile?.seed ?? 0x515154)) >>> 0);
    botRng = QQT.mulberry32((episodeSeed ^ 0x515154 ^ 0x5CA17) >>> 0);
    // Fresh model matches use the selected native rules; archived Review keeps its frozen engine.
    const native = localHumanControls();
    sim.reset(level, { nativeItems: true, nativeTrap: true, killMode: gameMode.value === 'kill', teams: [0, 1],
      bananaSlideSpeedPx: native ? 480 : undefined });
    if (native) { QQTSelfRespawnPreview.attach(sim); QQTKillOpeningItems.apply(sim); }
    sim.maxSteps = gameMode.value === 'kill' ? 1800 : 2400;
    QQTModelSampling.watch(sim);
    const [opponent, difficulty] = opponentSelect.value.split('@');
    leaderboardMatch = native && gameMode.value !== 'kill' ? leaderboard.begin({ seed: sim.seed, opponent,
      ruleset_version: QQTRulesets.current,
      difficulty: difficulty || (isModelChoice(opponent) ? 'model' : 'fixed'),
      mode: '1v1', map_id: 'training806', game_mode: gameMode.value }) : null;
    resetBot();
    releaseTouchControls();
    bombCell = -1;
    itemCell = -1;
    itemSlot = 0;
    snapMotion();
    renderer.reset();
    performance.mark('qqt:reset-ready');
  }

  function resetReplay() {
    if (!replayDocument) return;
    releaseTouchControls();
    sessionRevision++;
    leaderboardMatch = null;
    leaderboard.setSettlementReady(false);
    leaderboard.clearSettlement();
    sim = new legacyReplayApi.Sim(replayDocument.meta.seed);
    sim.reset(level);
    replayIndex = 0;
    replayAccumulator = 0;
    replaySeek.max = String(replayDocument.actions.length);
    replaySeek.value = '0';
    snapMotion();
    renderer.reset();
    performance.mark('qqt:reset-ready');
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
    performance.mark('qqt:model-start');
    modelPending = true; modelFailed = false;
    showLoading(`正在初始化模型 ${row.display_name}…`);
    loadingProgress.removeAttribute('value');
    clearHeldControls();
    opponentSelect.disabled = true;
    modelProgressWrap.hidden = false;
    modelProgress.removeAttribute('value');
    modelStatus.textContent = `正在加载：${row.display_name}`;
    modelProgressText.textContent = '正在连接并初始化模型…';
    // Fast cached reads finish before this delay, avoiding a flash of 0–100%.
    let downloadStarted = performance.now(), latest = null, timer = null;
    const display = () => {
      if (!latest) return;
      const {loaded,total} = latest;
      const mb = n => (n / 1048576).toFixed(1);
      const text = `正在下载模型 ${row.display_name}… ${mb(loaded)}${total > 0 ? ` / ${mb(total)} MB（${Math.min(100, Math.floor(loaded*100/total))}%）` : ' MB'}`;
      loadingText.textContent = modelProgressText.textContent = text;
      for (const bar of [loadingProgress, modelProgress]) {
        if (total > 0) bar.value = loaded * 100 / total;
        else bar.removeAttribute('value');
      }
    };
    const progress = event => {
      const {phase,loaded,total,index} = event;
      if (phase && phase !== 'download') performance.mark('qqt:model-' + phase);
      if (phase === 'loading') {
        downloadStarted = performance.now(); latest = null;
        clearTimeout(timer);
        loadingSource.textContent = index > 0
          ? 'Release 镜像不可用，正在尝试 Release 直链…' : '下载来源：Release 镜像';
        timer = setTimeout(display, 600);
      } else if (phase === 'download') {
        // Measure body streaming time, excluding connection/header latency.
        if (loaded === 0) {
          downloadStarted = performance.now(); clearTimeout(timer);
          timer = setTimeout(display, 600);
        }
        latest = {loaded,total};
        if (performance.now() - downloadStarted >= 600) display();
      } else if (['verifying', 'initializing', 'runtime', 'warming'].includes(phase)) {
        clearTimeout(timer); latest = null;
        loadingText.textContent = modelProgressText.textContent = phase === 'verifying'
          ? '下载完成，正在校验并初始化模型…' : phase === 'runtime' ? '正在加载推理运行时…'
          : phase === 'warming' ? '正在预热模型…' : '正在初始化模型…';
        loadingProgress.removeAttribute('value'); modelProgress.removeAttribute('value');
      }
    };
    let model;
    try {
      if (row.onnx) model = await QQTModelWorker.load(row, progress);
      else {
        const buffer = await QQTModelLoader.downloadModelAsset(row,
          (loaded,total)=>progress({phase:'download',loaded,total}),
          {onSource:({status,index})=>progress({phase:status,index})});
        progress({phase:'initializing'});
        await new Promise(resolve => setTimeout(resolve, 0));
        model = instantiateModel(JSON.parse(new TextDecoder().decode(buffer)),row);
      }
    } finally { clearTimeout(timer); }
    const previousModel = loadedModel;
    loadedModel = model.meta?.schema_version === 'five-item-slots-v1'
      ? QQTBoundaryRecovery.attach(model, {enabled: () => localHumanControls() && document.getElementById('boundary-recovery').checked}) : model;
    if (previousModel && previousModel.close) {
      // A request may still belong to the old session. Let its obsolete-tick
      // guard resolve before terminating the worker; never reject a live tick.
      const retire = () => ticking ? setTimeout(retire, 25) : previousModel.close();
      setTimeout(retire, 0);
    }
    modelProgress.value = 100;
    loadedModelId = row.id;
    modelProgressText.textContent = '模型加载完成';
    modelProgressWrap.hidden = true;
    performance.mark('qqt:model-ready');
    loadingText.textContent = '模型就绪，正在开局…';
    loadingProgress.value = 100;
    modelPending = false;
    opponentSelect.disabled = false;
    modelStatus.textContent = '';

  }

  async function loadCatalog() {
    try {
      const response = await fetch('models.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      publishedModels = QQTModelCatalog.validateManifest(await response.json());
      const defaultsResponse = await fetch('model_inference_defaults.json', {cache:'no-store'});
      if (!defaultsResponse.ok) throw new Error('模型推理配置不可用');
      const defaults = await defaultsResponse.json();
      if (defaults.schema !== 'qqt.model-inference-defaults/v1' || defaults.ruleset !== 'bun-native-v2' || defaults.max_ticks !== 2400 || !publishedModels.some(row=>row.id===defaults.default_model_id)) throw new Error('模型推理配置无效');
      if (!defaults.policies || Object.keys(defaults.policies).some(id=>!publishedModels.some(row=>row.id===id))) throw new Error('模型推理策略无效');
      defaultModelId = defaults.default_model_id;
      publishedModels = publishedModels.map(row => defaults.policies[row.id]
        ? {...row,inference:QQTModelSampling.profile({inference:defaults.policies[row.id]})} : row);
      publishedModels.sort((a,b) => Number(b.id===defaults.default_model_id)-Number(a.id===defaults.default_model_id));
      modelOptions.replaceChildren(...publishedModels.map((row) => new Option(QQTModelCatalog.optionLabel(row), MODEL_PREFIX + row.id)));
    } catch (error) {
      modelOptions.replaceChildren();
      modelFailed = true; loadingText.textContent = `模型列表读取失败：${error.message}`;
      loadingRetry.hidden = false;
      loadingRetry.onclick = () => location.reload();
    }
  }

  async function loadReplayCatalog() {
    performance.mark('qqt:replays-start');
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
      const legacyNotice = document.deprecated === true ? ' · 旧版只读兼容；新诊断录像请使用 Review' : '';
      performance.mark('qqt:replays-ready');
      replayStatus.textContent = `可选录像 ${replayCatalog.length} 局${legacyNotice}`;
    } catch (error) {
      replaySelect.replaceChildren(new Option('录像列表读取失败', ''));
      replayStatus.textContent = `录像列表读取失败：${error.message}`;
    }
  }

  async function loadReplay(row) {
    if (leaderboard.state().submitting || leaderboard.state().profileSubmitting) return;
    const response = await fetch(`replays/${row.file}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    replayDocument = QQTReplay.validateReplay(await response.json());
    // Preserve the previously deployed pixel/tick behavior of this deprecated
    // action format; it predates diagnostic SHA-based simulator dispatch.
    if (!legacyReplayApi) {
      const path='review-sim/stock-legacy-20261028.js', r=await fetch(path);
      const bytes=await r.arrayBuffer(), digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
      if(digest!=='7d2047b7db2025608b38afec0f8283625b6f98a3d09ac49693f2a0a44d1d485b')throw Error('历史模拟器 SHA 不一致');
      const current=window.QQT;try{await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=path;script.onload=resolve;script.onerror=()=>reject(Error('历史模拟器加载失败'));document.head.append(script);});legacyReplayApi=window.QQT;}finally{window.QQT=current;}
    }
    await useMap(arenaLevel);   // 离线录像在空场竞技场上生成，回放必须用同一张图
    resetReplay();
    replayPlaying = true;
    replayToggle.textContent = '暂停';
    const s = replayDocument.summary;
    replayStatus.textContent = `${row.model_id}｜seed ${row.seed}｜模型/规则Bot放泡 ${s.model_bombs}/${s.rule_bombs}｜活动格 ${s.model_unique_cells}/${s.rule_unique_cells}｜胜者 ${s.winner == null ? '未决' : s.winner}`;
  }

  function render(now = performance.now()) {
    if (!sim) return;
    touchControls.hidden = !localHumanControls() || sim.done;
    if (!touchControls.hidden) updateTouchItems();
    if (sim.done && localHumanControls() && leaderboardMatch) {
      const match = { ...leaderboardMatch, client_version: clientVersion };
      leaderboardMatch = null;
      void leaderboard.finish(match, { result: sim.winner == null ? 'draw' : sim.winner === sim.team[0] ? 'win' : 'loss',
        autoSubmit: false, gameDurationMs: sim.t * TICK_MS });
    }
    renderer.render(sim, now, motionState());
    QQTLeaderboard.renderSettlement(document, leaderboard.state());
    const rect = canvas.getBoundingClientRect(), stage = canvas.parentElement.getBoundingClientRect();
    document.getElementById('settlement').style.top = window.matchMedia('(max-width: 820px)').matches
      ? '' : `${rect.top - stage.top + rect.height * .65}px`;
    hideLoading();
  }

  async function tick() {
    if (!sim || ticking || assetsPending || modelPending || modelFailed) return;
    ticking = true;
    const tickSim = sim, revision = sessionRevision;
    const obsolete = () => sim !== tickSim || revision !== sessionRevision;
    try {
      if (replayDocument) {
        if (replayPlaying) {
          replayAccumulator += Number(replaySpeed.value);
          while (replayAccumulator >= 1 && replayPlaying) { stepReplay(); replayAccumulator -= 1; }
        }
      } else if (!sim.done) {
        if (!activeBot) return;
        const first = humanAction();
        if (obsolete()) return;
        const action = await Promise.resolve(activeBot.act(null, 1, botRng));
        if (obsolete()) return;
        const botActions = [null, action];
        prevPos.set(sim.pos);
        const before = QQTSound.snapshot(sim);
        const joint = [
          [Number(first[0]), Number(first[1]), Number(first[2]) || 0, Number(first[3]) || 0,
            first[4] == null ? -1 : first[4], first[5] == null ? -1 : first[5], first[6] == null ? 0 : first[6]],
          ...botActions.slice(1).map((a) => [a.move, a.ability === 1 ? 1 : 0, a.ability === 2 ? 1 : 0,
            0, -1, -1, a.itemSlot == null ? 0 : a.itemSlot]),
        ];
        const info = sim.step(joint);
        curPos.set(sim.pos); lastTickT = performance.now();
        if (localHumanControls()) {
          for (let pid = 1; pid < sim.nPlayers; pid++) {
            intents[pid] = sim.alive[pid] ? sim.playerMoveDirection(pid, botActions[pid].move) : QQT.MOVE_IDLE;
          }
        } else setTickIntents(action.move, first[0]);
        renderer.addExplosion(info, performance.now());
        playEvents(before, info, 0);

      }
    } finally {
      ticking = false;
    }
  }

  const unlockAudio = () => sound.unlock();
  const restartGame = QQTControls.createRestartGate(reset, () => sim.done);
  function requestRestart(finishedOnly = false) {
    if (leaderboard.state().submitting || leaderboard.state().profileSubmitting) {
      document.getElementById('settlement-status').textContent = '正在提交，请稍候～';
      return Promise.resolve(false);
    }
    return restartGame(finishedOnly).catch(error => { status.textContent = `重新开局失败：${error.message}`; });
  }
  canvas.addEventListener('click', () => { if (sim.done) leaderboard.reopenSettlement(); });
  window.addEventListener('pointerdown', unlockAudio);
  window.addEventListener('keydown', (event) => {
    if (document.querySelector('dialog[open]') || event.target.closest?.('#advanced-settings') || QQTControls.isTypingTarget(event.target)) return;
    unlockAudio();
    if (event.code === 'KeyR') {
      if (QQTControls.isRestartKey(event, held)) { event.preventDefault(); requestRestart(); }
      held.add(event.code);
      return;
    }
    if ([...QQTControls.MOVEMENT_KEYS, ...QQTControls.ITEM_KEYS, ...QQTControls.ITEM_SLOT_KEYS,
      ...QQTControls.BOMB_KEYS].includes(event.code)) event.preventDefault();
    held.add(event.code);
    // 同一 tick 内多次按键只保留第一次的位置。
    if (QQTControls.BOMB_KEYS.includes(event.code) && bombCell < 0) bombCell = humanCell();
    if (QQTControls.ITEM_KEYS.includes(event.code) && itemCell < 0) { itemCell = humanCell(); itemSlot = 0; }
    const slotKey = QQTControls.ITEM_SLOT_KEYS.indexOf(event.code);
    if (slotKey >= 0 && itemCell < 0) { itemCell = humanCell(); itemSlot = slotKey; }
  });
  window.addEventListener('keyup', (event) => held.delete(event.code));
  function clearHeldControls() {
    releaseTouchControls();
    held.clear(); bombCell = -1; itemCell = -1; itemSlot = 0;
  }
  document.addEventListener('focusin', (event) => {
    if (QQTControls.isTypingTarget(event.target) || event.target.closest?.('dialog, #advanced-settings')) clearHeldControls();
  });
  window.addEventListener('blur', clearHeldControls);
  document.addEventListener('visibilitychange', () => { if (document.hidden) clearHeldControls(); });
  restart.addEventListener('click', () => requestRestart());
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
    if (modelPending) return;
    const value = QQTModelCatalog.resolveSelection(opponentSelect.value, publishedModels, defaultModelId);
    opponentSelect.value = value;
    const row = isModelChoice(value) ? publishedModels.find((item) => MODEL_PREFIX + item.id === value) : null;
    if (row && loadedModelId !== row.id) {
      try { await loadPublishedModel(row); }
      catch (error) {
        modelPending = false; modelFailed = true;
        loadingText.textContent = `模型加载失败：${error.message}`;
        loadingProgress.removeAttribute('value');
        loadingRetry.hidden = false;
        opponentSelect.disabled = false;
        modelProgressWrap.hidden = false;
        modelProgress.removeAttribute('value');
        modelProgressText.textContent = `加载失败：${error.message}`;
        modelStatus.textContent = `加载失败：${error.message}`;
        return;
      }
    }
    modelFailed = false;
    await reset();
  }
  gameMode.addEventListener('change', async () => {
    try { localStorage.setItem('qqt.game-mode', gameMode.value); } catch (_) {}
    await reset();
  });
  opponentSelect.addEventListener('change', onStrategyChange);
  await loadCatalog();
  if (publishedModels.length) {
    opponentSelect.value = `${MODEL_PREFIX}${defaultModelId}`;
    await onStrategyChange();
  } else reset(true);
  void loadReplayCatalog();
  function animationFrame(now) {
    stepHumanFrame(now);
    render(now);
    requestAnimationFrame(animationFrame);
  }
  requestAnimationFrame(animationFrame);
  setInterval(tick, 100);
})();
