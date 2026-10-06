'use strict';

(function initVisualRenderer(root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.QQTVisual = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function visualRendererFactory() {
  const CELL = 60;
  const SOURCE_CELL = 40;
  const SCALE = CELL / SOURCE_CELL;
  const BOARD_OFFSET = 20 * SCALE;
  const BOARD_H = 13 * CELL;
  const Z_ROW_STRIDE = 24;
  const DIR_KEYS = ['U', 'D', 'L', 'R'];
  const MOVE_TO_SPRITE_ROW = [3, 0, 1, 2];
  const MOVE_IDLE = 4;

  const BUN_ELEMENT_IDS = [8001, 8002, 8003, 8004, 8005, 8006, 8009, 8010, 8011, 8012, 8013, 8028];
  // DIMG 无逐帧时长；与上游 GM 预览一致取 100ms/帧。
  const ITEM_FRAME_MS = 100;
  // Visual-only: the simulation lands at 3 ticks while this trail finishes at 5 ticks.
  const AIRDROP_ANIMATION_TICKS = 5;
  const CRATE_SPRITES = ['bomb', 'power', 'speed', 'banana_pickup', 'glue_pickup', 'fast_shoe'];
  const SUPER_CRATE_SPRITES = ['bomb_super', 'power_super', 'speed_super'];
  const FIELD_SPRITES = [null, 'banana_field', 'glue_field'];

  function crateSpriteKey(crateType, isSuper) {
    if (crateType < 0) return 'random';
    if (isSuper && crateType < 3) return SUPER_CRATE_SPRITES[crateType];
    return CRATE_SPRITES[crateType] || 'random';
  }

  // 道具 DIMG 帧坐标相对格子左上、画布原点为 (ox, oy)；原版帧底边落在格底下方 10~18px（含浮动余量），
  // 上移 14px 使其与泡泡/地图元素的格底对齐。
  const ITEM_ANCHOR_DY = 14;

  function itemFrame(sprite, now) {
    return sprite.frames[Math.floor(Math.max(0, now) / ITEM_FRAME_MS) % sprite.frames.length];
  }

  // 源像素坐标下，道具画布左上相对格子左上的位置。
  function itemCanvasOrigin(sprite) {
    return [sprite.ox, sprite.oy - ITEM_ANCHOR_DY];
  }

  function levelElementIds(level) {
    const found = new Set();
    if (level && level.layers) {
      for (const layer of level.layers) for (const value of layer) if (value) found.add(Math.abs(value));
    }
    return Array.from(found).sort((a, b) => a - b);
  }

  function bombFrame(ageSeconds, count) {
    return Math.floor(Math.max(0, ageSeconds) * count) % count;
  }

  function bombAgeSeconds(fuseTicks, fuseMaxTicks = 30, tickHz = 10) {
    return Math.max(0, fuseMaxTicks - fuseTicks) / tickHz;
  }

  // 脚底落在逻辑中心下方 9 源像素：若贴碰撞盒下沿(+19px)，脚和影子会比逻辑格低近半格，
  // 玩家按脚判断所在格，放泡就会看起来落在上一格。
  const FOOT_BELOW_CENTER_PX = 9;
  function playerVisualY(gridY, imageHeight) {
    return gridY * CELL + FOOT_BELOW_CENTER_PX * SCALE - imageHeight;
  }

  // 角色帧顶部约 30% 为透明留白（精灵图 26/85）；point.png 箭头尖在图高 75% 处（30/40）。
  const SPRITE_HEAD_FRAC = 0.3;
  const POINT_TIP_FRAC = 0.75;
  // 携带包子：包子底边压进头顶的比例（相对帧高）。
  const CARRY_BUN_SINK_FRAC = 0.06;

  const HELD_ITEM_SPRITES = [null, 'banana_pickup', 'glue_pickup'];
  function heldItemSpriteKey(item) { return HELD_ITEM_SPRITES[item] || null; }
  const ITEM_SLOT_COUNT = 7;
  const ITEM_BAR_SLOT_PX = 50;

  // 顶部溢出带：本图地面最上一行原纹理 + 半透明黑，表示界外；首行元件上溢仍画在它上面。
  const TOP_BAND_SHADE = 0.5;
  function makeTopBand(background) {
    const band = document.createElement('canvas');
    band.width = background.width; band.height = BOARD_OFFSET;
    const g = band.getContext('2d');
    g.drawImage(background, 0, 0, background.width, BOARD_OFFSET, 0, 0, band.width, BOARD_OFFSET);
    g.fillStyle = `rgba(0,0,0,${TOP_BAND_SHADE})`;
    g.fillRect(0, 0, band.width, BOARD_OFFSET);
    return band;
  }

  // 终局结果（以 viewerPid 视角）：win/lose/draw；viewerPid<0（观战）时报蓝/红方胜。
  function matchResult(sim, viewerPid = 0) {
    if (!sim || !sim.done) return null;
    const timeout = sim.maxSteps != null && sim.t >= sim.maxSteps;
    const reason = !sim.isBun ? '' : timeout ? '时间到' : '运包成功';
    // 超时按基地存包总数判胜负，运包成功按夺包数；比分与判定口径一致。
    const tally = !sim.isBun ? null : timeout && sim.bunStored
      ? sim.bunStored.map((row) => row.reduce((sum, count) => sum + count, 0)) : sim.bunScore;
    const score = tally ? `${tally[0]} : ${tally[1]}` : '';
    if (sim.winner == null) return { kind: 'draw', title: '平局', reason, score };
    if (viewerPid < 0) return { kind: 'win', title: sim.winner === 0 ? '蓝方胜利' : '红方胜利', reason, score };
    const viewerTeam = sim.team ? sim.team[viewerPid] : viewerPid;
    return sim.winner === viewerTeam
      ? { kind: 'win', title: '胜利', reason, score }
      : { kind: 'lose', title: '失败', reason, score };
  }

  function playerCount(sim) { return sim.nPlayers || 2; }
  function teamOf(sim, pid) { return sim.team ? sim.team[pid] : pid; }
  // 名牌：自己=“你”，同队=“队友”，敌方按序号“敌1/敌2”。
  function playerLabel(sim, pid, humanPid, humanTeam) {
    if (pid === humanPid) return '你';
    if (teamOf(sim, pid) === humanTeam) return '队友';
    let k = 0;
    for (let q = 0; q <= pid; q++) if (teamOf(sim, q) !== humanTeam) k++;
    return `敌${k}`;
  }
  function respawnSeconds(ticks, tickHz = 10) {
    return Math.max(0, Math.ceil(ticks / tickHz));
  }

  function deathOverlayAlpha(sim, viewerPid) {
    if (!sim || viewerPid == null || viewerPid < 0 || !sim.alive || sim.alive[viewerPid]) return 0;
    if (sim.isBun && sim.bunRespawn && !(sim.bunRespawn[viewerPid] > 0)) return 0;
    const remaining = sim.isBun && sim.bunRespawn ? sim.bunRespawn[viewerPid] : 1;
    const total = sim.bunRespawnTicks || 1;
    return Math.max(0.18, Math.min(0.72, sim.isBun ? 0.34 + 0.38 * remaining / total : 0.58));
  }

  function withContextState(ctx, draw) {
    ctx.save();
    try { return draw(); } finally { ctx.restore(); }
  }

  function respawnMarkers(sim, viewerPid = -1) {
    if (!sim || !sim.isBun || sim.done) return [];
    const markers = [];
    for (let pid = 0; pid < playerCount(sim); pid++) {
      const spawn = sim.bunSpawnPos && sim.bunSpawnPos[pid];
      if (sim.alive[pid] || !(sim.bunRespawn[pid] > 0) || !spawn ||
          !Number.isFinite(spawn[0]) || !Number.isFinite(spawn[1])) continue;
      if (spawn[0] < 0 || spawn[0] >= 13 || spawn[1] < 0 || spawn[1] >= 15) continue;
      markers.push({ pid, row: spawn[0], col: spawn[1], team: teamOf(sim, pid),
        seconds: respawnSeconds(sim.bunRespawn[pid]),
        label: viewerPid < 0 ? `${teamOf(sim, pid) === 0 ? '蓝' : '红'}${pid + 1}`
          : teamOf(sim, pid) === teamOf(sim, viewerPid)
            ? `${playerLabel(sim, pid, viewerPid, teamOf(sim, viewerPid))}${pid + 1}`
            : playerLabel(sim, pid, viewerPid, teamOf(sim, viewerPid)) });
    }
    return markers;
  }

  function bunTokens(sim) {
    const tokens = [];
    if (!sim || !sim.isBun) return tokens;
    for (let baseTeam = 0; baseTeam < (sim.bunBases || []).length; baseTeam++) {
      const base = sim.bunBases[baseTeam];
      for (let team = 0; team < 2; team++) {
        const count = sim.bunStored && sim.bunStored[baseTeam]
          ? sim.bunStored[baseTeam][team] || 0 : 0;
        if (count) tokens.push({
          row: base[0] + 1, column: base[1] + 1, team, count, size: 0.82,
          xOffset: (team - 0.5) * 0.26,
        });
      }
    }
    for (let i = 0; i < 195; i++) for (let team = 0; team < 2; team++) {
      const count = sim.bunLoose[i * 2 + team] || 0;
      if (count) tokens.push({
        row: Math.floor(i / 15), column: i % 15, team, count, size: 0.78,
        xOffset: (team - 0.5) * 0.24,
      });
    }
    return tokens;
  }

  function explosionFrame(ageSeconds) {
    const age = Math.max(0, ageSeconds);
    let tip = 1;
    if (age >= 0.20 && age < 0.26) tip = 5;
    else if (age >= 0.33) tip = 6;
    const body = age < 0.15 ? 2 : age < 0.24 ? 3 : age < 0.33 ? 4 : 3;
    return { center: Math.floor(age * 12) % 2 ? 2 : 1, body, tip, activeScale: age < 0.06 || age >= 0.39 ? 1 : Infinity };
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`素材加载失败: ${src}`));
      image.src = src;
      // Cached images can be complete before an onload listener is delivered.
      if (image.complete) {
        if (image.naturalWidth > 0) resolve(image);
        else reject(new Error(`素材加载失败: ${src}`));
      }
    });
  }

  function sliceSheet(sheet, rows, columns, target) {
    const result = [];
    const sw = sheet.width / columns;
    const sh = sheet.height / rows;
    for (let row = 0; row < rows; row++) {
      const frames = [];
      for (let column = 0; column < columns; column++) {
        // 先按原尺寸裁出单帧再缩放：直接从整张图缩放采样时，插值会把上一行帧贴底的脚
        // 渗进本帧顶边，形成头顶一条横线。
        const cell = document.createElement('canvas');
        cell.width = sw; cell.height = sh;
        cell.getContext('2d').drawImage(sheet, column * sw, row * sh, sw, sh, 0, 0, sw, sh);
        const frame = document.createElement('canvas');
        frame.width = target;
        frame.height = target;
        frame.getContext('2d').drawImage(cell, 0, 0, sw, sh, 0, 0, target, target);
        frames.push(frame);
      }
      result.push(frames);
    }
    return result;
  }

  function scaleImage(image, width = Math.round(image.width * SCALE), height = Math.round(image.height * SCALE)) {
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    canvas.getContext('2d').drawImage(image, 0, 0, width, height);
    return canvas;
  }

  // 素材并行下载：原先逐张串行 await，首屏要等几十次往返。onProgress(done, total) 驱动加载动画。
  async function loadAssets(level, onProgress = null) {
    const [elements, itemMeta, nativeMeta] = await Promise.all([
      fetch('assets/maps/elements.json').then((r) => r.json()),
      fetch('assets/item/items.json').then((r) => r.json()),
      fetch('assets/native/sprites.json').then((r) => r.json()),
    ]);
    const elementIds = levelElementIds(level).filter((id) => elements[String(id)] || elements[id]);
    const flameFiles = ['C_1', 'C_2'];
    const nativeSprites = Object.values(nativeMeta.actors).flatMap((actor) => Object.values(actor.actions))
      .concat(Object.values(nativeMeta.effects));
    for (const dir of DIR_KEYS) for (let f = 1; f <= 6; f++) flameFiles.push(`${dir}_${f}`);
    const sources = [
      level.bg || 'assets/bg/抢包子.png', 'assets/角色4×4精灵图.png', 'assets/角色c4×4.png',
      'assets/bomb-custom/经典黄泡泡.png', 'assets/shadow.png', 'assets/point.png',
      ...flameFiles.map((name) => `assets/flame/flame_${name}.png`),
      ...Object.values(itemMeta).map((meta) => meta.file),
      ...elementIds.map((id) => (elements[String(id)] || elements[id]).file),
      ...nativeSprites.map((meta) => meta.file),
    ];
    let done = 0;
    const total = sources.length;
    if (onProgress) onProgress(0, total);
    const images = await Promise.all(sources.map((src) => loadImage(src).then((image) => {
      done++;
      if (onProgress) onProgress(done, total);
      return image;
    })));
    let cursor = 0;
    const take = (count) => images.slice(cursor, (cursor += count));
    const [background, humanSheet, botSheet, bombStrip, shadow, point] = take(6);
    const flameImages = take(flameFiles.length);
    const itemImages = take(Object.keys(itemMeta).length);
    const elementSheets = take(elementIds.length);
    const nativeImages = take(nativeSprites.length);
    const nativeFrames = new Map(nativeSprites.map((meta, index) => [meta.file,
      sliceSheet(nativeImages[index], meta.directions || 1, meta.frames, Math.round(meta.w * SCALE))]));
    const characters = {};
    for (const [key, actor] of Object.entries(nativeMeta.actors)) {
      characters[key] = Object.fromEntries(Object.entries(actor.actions).map(([action, meta]) =>
        [action, nativeFrames.get(meta.file)]));
    }
    const effects = {};
    for (const [key, meta] of Object.entries(nativeMeta.effects)) {
      effects[key] = { frames: nativeFrames.get(meta.file)[0], aspect: meta.h / meta.w,
        ox: meta.ox || 0, oy: meta.oy || 0, frameMs: meta.frameMs || 100 };
    }
    const humanSize = Math.round((humanSheet.width / 4) * SCALE);
    const botSize = Math.round((botSheet.width / 4) * SCALE * 0.85);
    const bombSplits = [0, 38, 73, 112, 156];
    const bombs = [];
    for (let i = 0; i < 4; i++) {
      const sx = bombSplits[i], sw = bombSplits[i + 1] - sx;
      const frame = document.createElement('canvas');
      frame.width = Math.round(sw * SCALE);
      frame.height = Math.round(bombStrip.height * SCALE);
      frame.getContext('2d').drawImage(bombStrip, sx, 0, sw, bombStrip.height, 0, 0, frame.width, frame.height);
      bombs.push(frame);
    }
    const flames = { C: [], U: [], D: [], L: [], R: [] };
    flameFiles.forEach((name, index) => {
      const [dir, f] = name.split('_');
      if (dir === 'C') { flames.C[Number(f)] = scaleImage(flameImages[index], CELL, CELL); return; }
      const scaled = scaleImage(flameImages[index]);
      const frame = document.createElement('canvas');
      frame.width = CELL; frame.height = CELL;
      frame.getContext('2d').drawImage(scaled, dir === 'L' ? CELL - scaled.width : 0, dir === 'U' ? CELL - scaled.height : 0);
      flames[dir][Number(f)] = frame;
    });
    const items = {};
    Object.entries(itemMeta).forEach(([key, meta], index) => {
      const strip = itemImages[index];
      items[key] = { ox: meta.ox || 0, oy: meta.oy || 0, frames: [] };
      for (let f = 0; f < meta.frames; f++) {
        const frame = document.createElement('canvas');
        frame.width = Math.round(meta.w * SCALE);
        frame.height = Math.round(meta.h * SCALE);
        frame.getContext('2d').drawImage(strip, f * meta.w, 0, meta.w, meta.h, 0, 0, frame.width, frame.height);
        items[key].frames.push(frame);
      }
    });
    const elementImages = new Map();
    elementIds.forEach((id, index) => elementImages.set(id, scaleImage(elementSheets[index])));
    const scaledBackground = scaleImage(background);
    return {
      elements, background: scaledBackground, baseBand: makeTopBand(scaledBackground),
      players: [sliceSheet(humanSheet, 4, 4, humanSize), sliceSheet(botSheet, 4, 4, botSize)],
      bombs, flames, shadow: scaleImage(shadow), elementImages, items,
      characters, effects,
      point: scaleImage(point, Math.round(point.width * SCALE * 0.5), Math.round(point.height * SCALE * 0.5)),
    };
  }

  function createRenderer(canvas, level, assets) {
    const ctx = canvas.getContext('2d', { alpha: false });
    ctx.imageSmoothingEnabled = false;
    const explosions = [];
    const faces = [1, 1, 1, 1];
    const movingUntil = [0, 0, 0, 0];
    const pops = [];
    const airdropVisuals = [];
    let previousTraps = [];
    let previousTrapPositions = [];
    let trapGeneration = null;
    let lastPositions = null;

    function tileZ(row, column) { return row * Z_ROW_STRIDE + (15 - 1 - column); }
    function addExplosion(info, now) {
      if (info && info.covered && info.triggered && info.covered.some((value) => value > 0)) {
        explosions.push({ covered: Uint8Array.from(info.covered), triggered: Uint8Array.from(info.triggered), t0: now });
      }
    }
    function reset() {
      explosions.length = 0;
      faces.fill(1);
      movingUntil.fill(0);
      pops.length = 0;
      airdropVisuals.length = 0;
      previousTraps = []; previousTrapPositions = []; trapGeneration = null;
      lastPositions = null;
    }
    // 有移动意图时朝向跟随意图（顶墙也要面朝墙）；无意图时才按位移推断（如香蕉皮滑行）。
    function updateFaces(sim, intents) {
      const previous = lastPositions;
      for (let pid = 0; pid < playerCount(sim); pid++) {
        const intent = sim.movementStatus && sim.movementStatus[pid] === 2
          ? sim.slideDir[pid] : (intents ? intents[pid] : MOVE_IDLE);
        if (intent >= 0 && intent < MOVE_IDLE) { faces[pid] = intent; continue; }
        const dy = previous ? sim.pos[pid * 2] - previous[pid * 2] : 0;
        const dx = previous ? sim.pos[pid * 2 + 1] - previous[pid * 2 + 1] : 0;
        if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 1e-5) faces[pid] = dx < 0 ? 2 : 3;
        else if (Math.abs(dy) > 1e-5) faces[pid] = dy < 0 ? 0 : 1;
      }
      lastPositions = Array.from(sim.pos);
      return previous;
    }
    // (x, bottom) = 所在格底边中点（像素）。原版包子 item11 按道具锚点绘制并以格底为缩放基准，保留自带浮动帧。
    function drawBun(x, bottom, team, count, size = 1, now = 0) {
      const bun = assets.items && assets.items.bun;
      if (bun) {
        const image = itemFrame(bun, now);
        const [ox, oy] = itemCanvasOrigin(bun);
        const k = SCALE * size;
        const left = x + (ox - SOURCE_CELL / 2) * k, top = bottom + (oy - SOURCE_CELL) * k;
        ctx.drawImage(image, Math.round(left), Math.round(top), Math.round(image.width * size), Math.round(image.height * size));
        drawBunBadge(left + image.width * size * 0.82, top + image.height * size * 0.3, team, count);
        return;
      }
      const y = bottom - CELL * 0.38;
      const radius = 15 * size;
      ctx.save(); ctx.translate(x, y); ctx.shadowColor = 'rgba(0,0,0,0.35)'; ctx.shadowBlur = 4 * size; ctx.shadowOffsetY = 3 * size;
      ctx.fillStyle = '#f6c745'; ctx.beginPath(); ctx.ellipse(0, 2 * size, radius, radius * .72, 0, 0, Math.PI * 2); ctx.fill();
      ctx.shadowColor = 'transparent'; ctx.fillStyle = '#ffe78a'; ctx.beginPath(); ctx.ellipse(-4 * size, -3 * size, radius * .52, radius * .36, -.25, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = team ? '#3887e8' : '#e5484d'; ctx.lineWidth = Math.max(2, 3 * size); ctx.beginPath(); ctx.arc(0, size, radius * .72, .15, Math.PI - .15); ctx.stroke();
      if (count > 1) { ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(radius * .72, -radius * .55, 8 * size, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = '#fff'; ctx.font = `bold ${Math.round(10 * size)}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(count), radius * .72, -radius * .55); }
      ctx.restore();
    }
    // 手持道具：角色右上角的小图标 + 白底圆框，提示按 E/Shift 放置。
    function drawHeldItem(sprite, cx, cy, now) {
      const image = itemFrame(sprite, now);
      const size = 30, k = size / Math.max(image.width, image.height);
      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.strokeStyle = 'rgba(40,30,10,0.8)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, size * 0.62, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.drawImage(image, Math.round(cx - image.width * k / 2), Math.round(cy - image.height * k / 2),
        Math.round(image.width * k), Math.round(image.height * k));
      ctx.restore();
    }
    function drawNativeEffect(sprite, cx, bottom, now, age = null) {
      const index = age == null ? Math.floor(now / 100) % sprite.frames.length
        : Math.min(sprite.frames.length - 1, Math.floor(age / 120));
      const image = sprite.frames[index];
      ctx.drawImage(image, Math.round(cx - image.width / 2), Math.round(bottom - image.width * sprite.aspect),
        image.width, Math.round(image.width * sprite.aspect));
    }
    // Native syrup frames replace the fallback bubble when client assets are loaded.
    function drawTrapBubble(cx, cy, ticks, now) {
      const r = CELL * 0.62;
      const urgency = 1 - Math.min(1, ticks / 60);
      const wobble = 1 + Math.sin(now / (120 - urgency * 70)) * (0.03 + urgency * 0.04);
      if (assets.effects && assets.effects.trap) {
        drawNativeEffect(assets.effects.trap, cx, cy + r, now);
      } else {
      ctx.save(); ctx.translate(cx, cy); ctx.scale(wobble, 2 - wobble);
      const fill = ctx.createRadialGradient(-r * 0.3, -r * 0.35, r * 0.1, 0, 0, r);
      fill.addColorStop(0, 'rgba(255,255,255,0.55)');
      fill.addColorStop(0.55, 'rgba(150,215,255,0.28)');
      fill.addColorStop(1, 'rgba(90,170,255,0.5)');
      ctx.fillStyle = fill; ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.beginPath(); ctx.ellipse(-r * 0.4, -r * 0.45, r * 0.18, r * 0.1, -0.6, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      }
      ctx.save();
      ctx.font = 'bold 18px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
      ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,0.75)'; ctx.fillStyle = urgency > 0.65 ? '#ff7a7a' : '#ffffff';
      const label = String(respawnSeconds(ticks));
      ctx.strokeText(label, cx, cy - r - 8); ctx.fillText(label, cx, cy - r - 8);
      ctx.restore();
    }
    // 地图下方底栏：左起依次为本人速度 / 威力 / 糖泡数量，随后是 7 格道具栏（格内图标 + 数量 + 数字键）。
    function speedLevel(sim, pid) {
      const step = sim.speedStep || 0, base = sim.speedMax !== undefined && step ? sim.speedMax - 7 * step : 0;
      return step ? 1 + Math.round((sim.spdG[pid] - base) / step) : Number(sim.spdG[pid]).toFixed(1);
    }
    function drawStatusBar(sim, pid, now) {
      const size = ITEM_BAR_SLOT_PX, gap = 4, pad = 6;
      const top = BOARD_OFFSET + BOARD_H + Math.round((canvas.height - BOARD_OFFSET - BOARD_H - size) / 2);
      const stats = [
        ['速度', 'speed', speedLevel(sim, pid)],
        ['威力', 'power', sim.blastCap ? sim.blastCap[pid] : 0],
        ['糖泡', 'bomb', sim.bombsCap ? sim.bombsCap[pid] : 0],
      ];
      const statW = 76;
      let x = pad + 4;
      ctx.save();
      ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fillRect(x - pad, top - pad, stats.length * (statW + gap) - gap + pad * 2, size + pad * 2);
      for (const [label, key, value] of stats) {
        ctx.fillStyle = 'rgba(20,40,48,0.85)'; ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 1.5;
        ctx.fillRect(x, top, statW, size); ctx.strokeRect(x + 0.5, top + 0.5, statW - 1, size - 1);
        const sprite = assets.items && assets.items[key];
        if (sprite) {
          const image = itemFrame(sprite, 0), k = (size - 14) / Math.max(image.width, image.height);
          ctx.drawImage(image, Math.round(x + 4), Math.round(top + (size - image.height * k) / 2),
            Math.round(image.width * k), Math.round(image.height * k));
        }
        ctx.textAlign = 'right'; ctx.fillStyle = '#cfe9ee'; ctx.font = 'bold 12px sans-serif';
        ctx.fillText(label, x + statW - 6, top + 12);
        ctx.font = 'bold 22px sans-serif'; ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,0.85)'; ctx.fillStyle = '#ffd54a';
        ctx.strokeText(String(value), x + statW - 6, top + size - 15); ctx.fillText(String(value), x + statW - 6, top + size - 15);
        x += statW + gap;
      }
      ctx.restore();
      if (sim.nativeItems && sim.itemSlots) drawItemBar(sim.itemSlots[pid], now, x + pad * 2 + 6, top);
    }
    function drawItemBar(slots, now, left, top) {
      const size = ITEM_BAR_SLOT_PX, gap = 4, count = ITEM_SLOT_COUNT, pad = 6;
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fillRect(left - pad, top - pad, count * (size + gap) - gap + pad * 2, size + pad * 2);
      ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
      for (let i = 0; i < count; i++) {
        const x = left + i * (size + gap), slot = slots[i];
        ctx.fillStyle = slot ? 'rgba(255,248,220,0.82)' : 'rgba(0,0,0,0.4)';
        ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1.5;
        ctx.fillRect(x, top, size, size); ctx.strokeRect(x + 0.5, top + 0.5, size - 1, size - 1);
        const key = slot ? heldItemSpriteKey(slot.item) : null;
        if (key && assets.items && assets.items[key]) {
          const image = itemFrame(assets.items[key], now), k = (size - 8) / Math.max(image.width, image.height);
          ctx.drawImage(image, Math.round(x + (size - image.width * k) / 2), Math.round(top + (size - image.height * k) / 2),
            Math.round(image.width * k), Math.round(image.height * k));
        }
        ctx.font = 'bold 13px sans-serif'; ctx.textAlign = 'left';
        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,0.85)'; ctx.fillStyle = '#ffffff';
        ctx.strokeText(String(i + 1), x + 4, top + 9); ctx.fillText(String(i + 1), x + 4, top + 9);
        if (slot) {
          ctx.font = 'bold 17px sans-serif'; ctx.textAlign = 'right'; ctx.lineWidth = 4; ctx.fillStyle = '#ffd54a';
          ctx.strokeText(String(slot.count), x + size - 4, top + size - 10); ctx.fillText(String(slot.count), x + size - 4, top + size - 10);
        }
      }
      ctx.restore();
    }
    function drawTeamRing(cx, cy, team) {
      ctx.save();
      ctx.strokeStyle = team ? 'rgba(56,135,232,0.9)' : 'rgba(229,72,77,0.9)';
      ctx.fillStyle = team ? 'rgba(56,135,232,0.22)' : 'rgba(229,72,77,0.22)';
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.ellipse(cx, cy, CELL * 0.4, CELL * 0.16, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.restore();
    }
    function drawNameTag(cx, y, label, team) {
      ctx.save();
      ctx.font = 'bold 13px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
      ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.fillStyle = team ? '#9cc8ff' : '#ffaaa8';
      ctx.strokeText(label, cx, y); ctx.fillText(label, cx, y);
      ctx.restore();
    }
    function drawBunBadge(x, y, team, count) {
      ctx.save();
      ctx.fillStyle = team ? '#3887e8' : '#e5484d';
      ctx.beginPath(); ctx.arc(x, y, count > 1 ? 9 : 5, 0, Math.PI * 2); ctx.fill();
      if (count > 1) {
        ctx.fillStyle = '#fff'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(String(count), x, y);
      }
      ctx.restore();
    }
    function groundItems(sim, now, items, coveredCells) {
      if (!assets.items) return;
      for (let i = 0; i < 195; i++) {
        const debris = brickDebris(sim, i);
        if (coveredCells.has(i) || (sim.brick && sim.brick[i] && !debris) || (sim.wall && sim.wall[i])) continue;
        const row = Math.floor(i / 15), column = i % 15;
        const fieldKey = sim.fieldItem ? FIELD_SPRITES[sim.fieldItem[i]] : null;
        const place = (sprite, z) => {
          const [ox, oy] = itemCanvasOrigin(sprite);
          items.push([z, itemFrame(sprite, now), Math.round(column * CELL + ox * SCALE), Math.round(row * CELL + oy * SCALE)]);
        };
        if (fieldKey && assets.items[fieldKey]) place(assets.items[fieldKey], row * Z_ROW_STRIDE + 14);
        if (sim.crate && sim.crate[i] && !airdropVisuals.some((entry) => entry.drop.cell === i)) {
          const sprite = assets.items[crateSpriteKey(sim.crateType ? sim.crateType[i] : -1, sim.superCrate && sim.superCrate[i] === 1)];
          if (sprite) place(sprite, row * Z_ROW_STRIDE + 16);
          if (sim.crateCount && sim.crateCount[i] > 1) items.push([row * Z_ROW_STRIDE + 17, () => {
            ctx.save(); ctx.font = 'bold 16px sans-serif'; ctx.textAlign = 'right';
            ctx.lineWidth = 3; ctx.strokeStyle = '#202020'; ctx.fillStyle = '#fff';
            const x = (column + 1) * CELL - 4, y = (row + 1) * CELL - 8;
            ctx.strokeText(String(sim.crateCount[i]), x, y); ctx.fillText(String(sim.crateCount[i]), x, y);
            ctx.restore();
          }]);
        } else if (debris && sim.pendingCrateType && sim.pendingCrateType[i] >= 0) {
          const sprite = assets.items[crateSpriteKey(sim.pendingCrateType[i], sim.pendingSuperCrate && sim.pendingSuperCrate[i] === 1)];
          if (sprite) place(sprite, row * Z_ROW_STRIDE + 16);
        }
      }
    }
    // 被炸砖在残骸期仍是碰撞体，但画面上立即消失。
    function brickDebris(sim, i) {
      return !!(sim.brick && sim.brick[i] && sim.brickLinger && sim.brickLinger[i] > 0);
    }
    function structureItems(sim, items) {
      const coveredCells = new Set();
      for (let layerIndex = 0; layerIndex < 2; layerIndex++) {
        const layer = level.layers[layerIndex];
        for (let row = 0; row < 13; row++) for (let column = 0; column < 15; column++) {
          const index = row * 15 + column, value = layer[index];
          if (!value || value < 0) continue;
          if (layerIndex === 1 && !sim.wall[index] && !sim.brick[index] && !sim.cover[index]) continue;
          if (layerIndex === 1 && !sim.wall[index] && brickDebris(sim, index)) continue;
          const id = Math.abs(value), meta = assets.elements[String(id)] || assets.elements[id], image = assets.elementImages.get(id);
          if (!meta || !image) continue;
          for (let rr = row; rr < Math.min(13, row + meta.h); rr++) {
            for (let cc = column; cc < Math.min(15, column + meta.w); cc++) {
              coveredCells.add(rr * 15 + cc);
            }
          }
          items.push([tileZ(row, column), image, column * CELL - meta.xo * SCALE, row * CELL - meta.yo * SCALE]);
        }
      }
      return coveredCells;
    }
    function drawExplosion(exp, now, items) {
      const age = (now - exp.t0) / 1000;
      if (age > 0.45) return false;
      const frame = explosionFrame(age);
      for (let i = 0; i < 195; i++) {
        if (!exp.triggered[i]) continue;
        const sr = Math.floor(i / 15), sc = i % 15;
        items.push([sr * Z_ROW_STRIDE + 19, assets.flames.C[frame.center], sc * CELL, sr * CELL]);
        for (let d = 0; d < 4; d++) {
          const delta = [[-1,0],[1,0],[0,-1],[0,1]][d], key = DIR_KEYS[d];
          let length = 0;
          for (let k = 1; k <= 8; k++) {
            const row = sr + delta[0] * k, column = sc + delta[1] * k;
            if (row < 0 || row >= 13 || column < 0 || column >= 15 || !exp.covered[row * 15 + column]) break;
            length++;
          }
          const activeLength = Math.min(length, frame.activeScale);
          for (let k = 1; k <= activeLength; k++) {
            const row = sr + delta[0] * k, column = sc + delta[1] * k;
            if (exp.triggered[row * 15 + column]) continue;
            const image = assets.flames[key][k === activeLength ? frame.tip : frame.body];
            items.push([row * Z_ROW_STRIDE + 19, image, column * CELL, row * CELL]);
          }
        }
      }
      return true;
    }
    function renderUnsafe(sim, now = performance.now(), motion = null) {
      if (trapGeneration !== sim._gen) {
        pops.length = 0; airdropVisuals.length = 0; previousTraps = []; previousTrapPositions = []; trapGeneration = sim._gen;
      }
      for (let pid = 0; pid < playerCount(sim); pid++) {
        if (previousTraps[pid] > 0 && !sim.alive[pid] && assets.effects && assets.effects.pop) {
          pops.push({ x: previousTrapPositions[pid * 2 + 1] * CELL,
            y: previousTrapPositions[pid * 2] * CELL + FOOT_BELOW_CENTER_PX * SCALE, t0: now });
        }
      }
      previousTraps = sim.trapped ? sim.trapped.slice() : [];
      previousTrapPositions = Array.from(sim.pos);
      const intents = motion && motion.intents ? motion.intents : null;
      const previousPositions = updateFaces(sim, intents);
      // 逻辑节拍 10Hz，渲染 60Hz：对手(10Hz)在两个 sim tick 之间线性插值位置 → 顺滑。
      // 本地人类(motion.humanPid)不插值：由 rAF 逐帧 frameStep 连续移动，sim.pos 本身即每帧真实位置。
      // 复活/传送(位移>1格)时不插值直接吸附。
      const alpha = motion ? Math.min(1, Math.max(0, (now - motion.lastTickT) / motion.tickMs)) : 1;
      const viewerPid = motion ? motion.humanPid : 0;
      const deathAlpha = deathOverlayAlpha(sim, viewerPid);
      ctx.filter = 'none';
      ctx.fillStyle = '#0c0e13'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      const band = assets.baseBand;
      ctx.drawImage(band, 0, Math.max(0, (band.height - BOARD_OFFSET) / 2), band.width, BOARD_OFFSET,
        0, 0, canvas.width, BOARD_OFFSET);
      ctx.save();
      try {
        ctx.translate(0, BOARD_OFFSET);
        ctx.drawImage(assets.background, 0, 0);
        const tickMs = motion && motion.tickMs ? motion.tickMs : 100;
        for (const drop of sim.airdropFalls || []) if (!airdropVisuals.some((entry) => entry.key === `${drop.tick}:${drop.cell}`)) {
          airdropVisuals.push({ key: `${drop.tick}:${drop.cell}`, drop: { ...drop }, startedAt: now - Math.max(0, sim.t - drop.tick) * tickMs });
        }
        for (let i = airdropVisuals.length - 1; i >= 0; i--) {
          const entry = airdropVisuals[i];
          if (now - entry.startedAt >= AIRDROP_ANIMATION_TICKS * tickMs ||
              (sim.t >= entry.drop.tick + 3 && (!sim.crate[entry.drop.cell] || sim.crateType[entry.drop.cell] !== entry.drop.type))) {
            airdropVisuals.splice(i, 1);
          }
        }
        const items = [];
        const coveredCells = structureItems(sim, items);
        groundItems(sim, now, items, coveredCells);
        for (const token of bunTokens(sim)) items.push([
          token.row * Z_ROW_STRIDE + 15,
          () => drawBun((token.column + .5 + token.xOffset) * CELL, (token.row + 1) * CELL,
            token.team, token.count, token.size, now),
        ]);
        for (let i = explosions.length - 1; i >= 0; i--) if (!drawExplosion(explosions[i], now, items)) explosions.splice(i, 1);
        for (let i = 0; i < 195; i++) if (sim.fuse[i] > 0) {
          if (coveredCells.has(i)) continue;
          const row = Math.floor(i / 15), column = i % 15;
          const age = bombAgeSeconds(sim.fuse[i]);
          const image = assets.bombs[bombFrame(age, assets.bombs.length)];
          items.push([row * Z_ROW_STRIDE + 17, image, column * CELL + (CELL - image.width) / 2, (row + 1) * CELL - image.height]);
        }
        // 本地玩家头顶的原版 point.png 箭头；观战/回放时不画。
        const arrowPid = motion ? motion.humanPid : 0;
        let arrow = null;
        const humanTeam = teamOf(sim, motion ? motion.humanPid : 0);
        for (let pid = 0; pid < playerCount(sim); pid++) if (sim.alive[pid]) {
          let gy = sim.pos[pid * 2], gx = sim.pos[pid * 2 + 1];
          if (motion && pid !== motion.humanPid) {
            const py = motion.prevPos[pid * 2], px = motion.prevPos[pid * 2 + 1];
            const cy = motion.curPos[pid * 2], cx = motion.curPos[pid * 2 + 1];
            // 位移过大（复活/传送）不插值，避免角色横扫全图
            if (Math.abs(cy - py) + Math.abs(cx - px) <= 1.0) { gy = py + (cy - py) * alpha; gx = px + (cx - px) * alpha; }
            else { gy = cy; gx = cx; }
          }
          const row = MOVE_TO_SPRITE_ROW[faces[pid]];
          const pushing = intents && intents[pid] >= 0 && intents[pid] < MOVE_IDLE;
          const moved = pushing || (previousPositions && (Math.abs(sim.pos[pid * 2] - previousPositions[pid * 2]) + Math.abs(sim.pos[pid * 2 + 1] - previousPositions[pid * 2 + 1]) > 1e-5));
          if (moved) movingUntil[pid] = now + 150;
          const trapTicks = sim.trapped ? sim.trapped[pid] : 0;
          const spawnProtection = sim.spawnProtection ? sim.spawnProtection[pid] : 0;
          const team = teamOf(sim, pid);
          const character = motion && motion.characters && assets.characters && assets.characters[motion.characters[pid]];
          const walking = !trapTicks && now < movingUntil[pid];
          const frames = character ? character[walking ? 'walk' : 'stand'][row] : assets.players[team][row];
          const image = frames[walking ? Math.floor(now / 100) % frames.length : 0];
          const x = Math.round(gx * CELL - image.width / 2), y = Math.min(Math.round(playerVisualY(gy, image.height)), 780 - image.height);
          // 箭头先记录：进包子笼等遮挡被隐藏时仍要标出位置。
          if (pid === arrowPid) arrow = { x: gx * CELL, y: y + image.height * SPRITE_HEAD_FRAC };
          const playerCell = Math.floor(gy) * 15 + Math.floor(gx);
          const onWall = sim.wall[playerCell] || sim.brick[playerCell];
          if (coveredCells.has(playerCell) && !onWall) continue;
          const z = Math.floor(gy) * Z_ROW_STRIDE + (onWall ? 23 : 18);
          items.push([z - 1, assets.shadow, Math.round(gx * CELL - assets.shadow.width / 2), y + image.height - assets.shadow.height + 16]);
          // 多人时同队共用精灵：脚下队伍色光圈 + 头顶名牌区分。
          if (playerCount(sim) > 2) {
            items.push([z - 1, () => drawTeamRing(gx * CELL, y + image.height - 6, team)]);
            // 真人只靠箭头标识，不另画名牌。
            if (pid !== (motion ? motion.humanPid : 0)) {
              const label = playerLabel(sim, pid, motion ? motion.humanPid : 0, humanTeam);
              items.push([z + 4, () => drawNameTag(gx * CELL, y + image.height * SPRITE_HEAD_FRAC - 6, label, team)]);
            }
          }
          items.push([z, image, x, y]);
          if (spawnProtection > 0 && assets.effects && assets.effects.protection) {
            const halo = assets.effects.protection;
            const frame = halo.frames[Math.floor(now / halo.frameMs) % halo.frames.length];
            // magic0139 的原点来自 DIMG，换算为角色中心锚点，保持官方环形光效偏移。
            items.push([z + 3, () => ctx.drawImage(frame,
              Math.round(gx * CELL + halo.ox * SCALE),
              Math.round(gy * CELL + halo.oy * SCALE), frame.width,
              Math.round(frame.width * halo.aspect))]);
          }
          // 包子底边落在头顶附近（帧顶约 30% 为透明留白），贴着头顶而不是悬在上方。
          if (sim.bunCarried[pid] >= 0) {
            const bunBottom = y + image.height * (SPRITE_HEAD_FRAC + CARRY_BUN_SINK_FRAC);
            items.push([z + 1, () => drawBun(x + image.width / 2, bunBottom, sim.bunCarried[pid], 1, 0.8, now)]);
          }
          const heldKey = sim.heldItem ? heldItemSpriteKey(sim.heldItem[pid]) : null;
          if (heldKey && assets.items && assets.items[heldKey] && !trapTicks) {
            items.push([z + 2, () => drawHeldItem(assets.items[heldKey], x + image.width * 0.8, y + image.height * 0.25, now)]);
          }
          if (trapTicks > 0) items.push([z + 3, () => drawTrapBubble(gx * CELL,
            gy * CELL + FOOT_BELOW_CENTER_PX * SCALE - CELL * 0.62, trapTicks, now)]);
        }
        for (let i = pops.length - 1; i >= 0; i--) {
          const pop = pops[i], age = now - pop.t0;
          if (age >= assets.effects.pop.frames.length * 120) { pops.splice(i, 1); continue; }
          items.push([Math.floor(pop.y / CELL) * Z_ROW_STRIDE + 23,
            () => drawNativeEffect(assets.effects.pop, pop.x, pop.y, now, age)]);
        }
        items.sort((a, b) => a[0] - b[0]);
        for (const item of items) typeof item[1] === 'function' ? item[1]() : ctx.drawImage(item[1], item[2], item[3]);
        const birdFraction = motion && !sim.done ? Math.max(0, Math.min(1, (now - motion.lastTickT) / motion.tickMs)) : 0;
        const bird = sim.birdFlight ? sim.birdFlight(birdFraction) : null;
        if (assets.items) for (const { drop, startedAt } of airdropVisuals) {
          const progress = Math.min(1, Math.max(0, (now - startedAt) / (AIRDROP_ANIMATION_TICKS * tickMs)));
          const sprite = assets.items[crateSpriteKey(drop.type, drop.isSuper)];
          if (sprite) drawHeldItem(sprite, ((1 - progress) * drop.x + progress * (drop.cell % 15 + .5)) * CELL,
            ((1 - progress) * (drop.row || 3) + progress * (Math.floor(drop.cell / 15) + .5)) * CELL, now);
        }
        if (bird && assets.effects && assets.effects.bird) {
          drawNativeEffect(assets.effects.bird, bird.x * CELL, bird.row * CELL, now);
        }
        if (arrow && assets.point) drawArrow(arrow, now);
        // 死亡只压暗游戏画布，右侧 DOM 控件仍可读可点；观战/回放 viewerPid=-1 不遮挡。
        if (deathAlpha > 0) {
          ctx.fillStyle = `rgba(0,0,0,${deathAlpha.toFixed(3)})`;
          ctx.fillRect(0, 0, canvas.width, BOARD_H);
        }
        ctx.filter = 'none';
        if (sim.isBun && !sim.done) drawRespawnCountdowns(sim, motion ? motion.humanPid : 0);
      } finally { ctx.restore(); }
      // Only the playfield was dimmed; HUD/result rendering remains legible.
      ctx.filter = 'none';
      const barPid = motion ? motion.humanPid : 0;
      if (barPid >= 0 && sim.spdG) drawStatusBar(sim, barPid, now);
      drawResult(matchResult(sim, motion ? motion.humanPid : 0));
    }
    function render(sim, now = performance.now(), motion = null) {
      return withContextState(ctx, () => renderUnsafe(sim, now, motion));
    }
    const RESULT_COLORS = { win: '#ffd54a', lose: '#ff7a7a', draw: '#d8e6ea' };
    function drawResult(result) {
      if (!result) return;
      const cx = canvas.width / 2, cy = (BOARD_OFFSET + BOARD_H) / 2;
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
      ctx.font = 'bold 96px sans-serif'; ctx.lineWidth = 10; ctx.strokeStyle = 'rgba(0,0,0,0.85)';
      ctx.fillStyle = RESULT_COLORS[result.kind];
      ctx.strokeText(result.title, cx, cy - 30); ctx.fillText(result.title, cx, cy - 30);
      const sub = [result.reason, result.score && `包子 ${result.score}`].filter(Boolean).join('  ·  ');
      ctx.font = 'bold 28px sans-serif'; ctx.lineWidth = 6; ctx.fillStyle = '#ffffff';
      if (sub) { ctx.strokeText(sub, cx, cy + 50); ctx.fillText(sub, cx, cy + 50); }
      ctx.font = 'bold 20px sans-serif'; ctx.lineWidth = 5; ctx.fillStyle = '#cfe9ee';
      ctx.strokeText('按 R 重新开局', cx, cy + 100); ctx.fillText('按 R 重新开局', cx, cy + 100);
      ctx.restore();
    }
    // 箭头不参与 Z 排序，永远画在所有地图元件之上；轻微上下浮动便于辨认。
    function drawArrow(anchor, now) {
      const image = assets.point;
      const bob = Math.round(Math.sin(now / 160) * 3);
      const ax = Math.round(anchor.x - image.width / 2);
      const ay = Math.max(-BOARD_OFFSET, Math.round(anchor.y - 3 - image.height * POINT_TIP_FRAC + bob));
      ctx.drawImage(image, ax, ay);
    }
    // Compact spawn markers keep paths visible, including in spectator/replay views.
    function drawRespawnCountdowns(sim, humanPid) {
      for (const marker of respawnMarkers(sim, humanPid)) {
        const x = marker.col * CELL, y = marker.row * CELL;
        ctx.save();
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = marker.team === 0 ? '#8ee9ff' : '#ffa6bf'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.ellipse(x, y + CELL * .2, CELL * .36, CELL * .16, 0, 0, Math.PI * 2); ctx.stroke();
        const text = `${marker.label} · ${marker.seconds}秒复活`;
        const labelX = Math.max(96, Math.min(canvas.width - 96, x));
        const labelY = Math.max(16, y - CELL * .35);
        ctx.font = 'bold 24px sans-serif'; ctx.lineWidth = 5;
        ctx.strokeStyle = 'rgba(0,0,0,.85)'; ctx.fillStyle = marker.pid === humanPid ? '#ffe477' : '#ffffff';
        ctx.strokeText(text, labelX, labelY); ctx.fillText(text, labelX, labelY);
        ctx.restore();
      }
    }
    return { render, addExplosion, reset };
  }

  return { CELL, BOARD_OFFSET, BUN_ELEMENT_IDS, ITEM_FRAME_MS, AIRDROP_ANIMATION_TICKS, crateSpriteKey, levelElementIds, bombFrame, bombAgeSeconds, playerVisualY, respawnSeconds, respawnMarkers, deathOverlayAlpha, withContextState, heldItemSpriteKey, playerLabel, matchResult, bunTokens, explosionFrame, loadAssets, createRenderer };
});
