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
  const Z_ROW_STRIDE = 24;
  const DIR_KEYS = ['U', 'D', 'L', 'R'];
  const MOVE_TO_SPRITE_ROW = [3, 0, 1, 2];
  const MOVE_IDLE = 4;

  const BUN_ELEMENT_IDS = [8001, 8002, 8003, 8004, 8005, 8006, 8009, 8010, 8011, 8012, 8013, 8028];
  // DIMG 无逐帧时长；与上游 GM 预览一致取 100ms/帧。
  const ITEM_FRAME_MS = 100;
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

  const HELD_ITEM_SPRITES = [null, 'banana_pickup', 'glue_pickup'];
  function heldItemSpriteKey(item) { return HELD_ITEM_SPRITES[item] || null; }

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
    return sim.winner === viewerPid
      ? { kind: 'win', title: '胜利', reason, score }
      : { kind: 'lose', title: '失败', reason, score };
  }

  function respawnSeconds(ticks, tickHz = 10) {
    return Math.max(0, Math.ceil(ticks / tickHz));
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

  async function loadAssets(level) {
    const [elements, background, humanSheet, botSheet, bombStrip, shadow] = await Promise.all([
      fetch('assets/maps/elements.json').then((r) => r.json()),
      loadImage(level.bg || 'assets/bg/抢包子.png'),
      loadImage('assets/角色4×4精灵图.png'),
      loadImage('assets/角色c4×4.png'),
      loadImage('assets/bomb-custom/经典黄泡泡.png'),
      loadImage('assets/shadow.png'),
    ]);
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
    for (const f of [1, 2]) flames.C[f] = scaleImage(await loadImage(`assets/flame/flame_C_${f}.png`), CELL, CELL);
    for (const dir of DIR_KEYS) for (let f = 1; f <= 6; f++) {
      const scaled = scaleImage(await loadImage(`assets/flame/flame_${dir}_${f}.png`));
      const frame = document.createElement('canvas');
      frame.width = CELL; frame.height = CELL;
      frame.getContext('2d').drawImage(scaled, dir === 'L' ? CELL - scaled.width : 0, dir === 'U' ? CELL - scaled.height : 0);
      flames[dir][f] = frame;
    }
    const itemMeta = await fetch('assets/item/items.json').then((r) => r.json());
    const items = {};
    for (const [key, meta] of Object.entries(itemMeta)) {
      const strip = await loadImage(meta.file);
      items[key] = { ox: meta.ox || 0, oy: meta.oy || 0, frames: [] };
      for (let f = 0; f < meta.frames; f++) {
        const frame = document.createElement('canvas');
        frame.width = Math.round(meta.w * SCALE);
        frame.height = Math.round(meta.h * SCALE);
        frame.getContext('2d').drawImage(strip, f * meta.w, 0, meta.w, meta.h, 0, 0, frame.width, frame.height);
        items[key].frames.push(frame);
      }
    }
    const elementImages = new Map();
    for (const id of levelElementIds(level)) {
      const meta = elements[String(id)] || elements[id];
      if (meta) elementImages.set(id, scaleImage(await loadImage(meta.file)));
    }
    const scaledBackground = scaleImage(background);
    return {
      elements, background: scaledBackground, baseBand: makeTopBand(scaledBackground),
      players: [sliceSheet(humanSheet, 4, 4, humanSize), sliceSheet(botSheet, 4, 4, botSize)],
      bombs, flames, shadow: scaleImage(shadow), elementImages, items,
    };
  }

  function createRenderer(canvas, level, assets) {
    const ctx = canvas.getContext('2d', { alpha: false });
    ctx.imageSmoothingEnabled = false;
    const explosions = [];
    const faces = [1, 1];
    const movingUntil = [0, 0];
    let lastPositions = null;

    function tileZ(row, column) { return row * Z_ROW_STRIDE + (15 - 1 - column); }
    function addExplosion(info, now) {
      if (info && info.covered && info.triggered && info.covered.some((value) => value > 0)) {
        explosions.push({ covered: Uint8Array.from(info.covered), triggered: Uint8Array.from(info.triggered), t0: now });
      }
    }
    function reset() {
      explosions.length = 0;
      faces[0] = faces[1] = 1;
      movingUntil[0] = movingUntil[1] = 0;
      lastPositions = null;
    }
    // 有移动意图时朝向跟随意图（顶墙也要面朝墙）；无意图时才按位移推断（如香蕉皮滑行）。
    function updateFaces(sim, intents) {
      const previous = lastPositions;
      for (let pid = 0; pid < 2; pid++) {
        const intent = intents ? intents[pid] : MOVE_IDLE;
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
        if (sim.crate && sim.crate[i]) {
          const sprite = assets.items[crateSpriteKey(sim.crateType ? sim.crateType[i] : -1, sim.superCrate && sim.superCrate[i] === 1)];
          if (sprite) place(sprite, row * Z_ROW_STRIDE + 16);
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
    function render(sim, now = performance.now(), motion = null) {
      const intents = motion && motion.intents ? motion.intents : null;
      const previousPositions = updateFaces(sim, intents);
      // 逻辑节拍 10Hz，渲染 60Hz：对手(10Hz)在两个 sim tick 之间线性插值位置 → 顺滑。
      // 本地人类(motion.humanPid)不插值：由 rAF 逐帧 frameStep 连续移动，sim.pos 本身即每帧真实位置。
      // 复活/传送(位移>1格)时不插值直接吸附。
      const alpha = motion ? Math.min(1, Math.max(0, (now - motion.lastTickT) / motion.tickMs)) : 1;
      ctx.fillStyle = '#0c0e13'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      const band = assets.baseBand;
      ctx.drawImage(band, 0, Math.max(0, (band.height - BOARD_OFFSET) / 2), band.width, BOARD_OFFSET,
        0, 0, canvas.width, BOARD_OFFSET);
      ctx.save(); ctx.translate(0, BOARD_OFFSET);
      ctx.drawImage(assets.background, 0, 0);
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
      for (let pid = 0; pid < 2; pid++) if (sim.alive[pid]) {
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
        if (coveredCells.has(Math.floor(gy) * 15 + Math.floor(gx))) continue;
        if (moved) movingUntil[pid] = now + 150;
        const frames = assets.players[pid][row], image = frames[now < movingUntil[pid] ? Math.floor(now / 125) % 4 : 0];
        const x = Math.round(gx * CELL - image.width / 2), y = Math.min(Math.round(playerVisualY(gy, image.height)), 780 - image.height);
        const z = Math.floor(gy) * Z_ROW_STRIDE + 18;
        items.push([z - 1, assets.shadow, Math.round(gx * CELL - assets.shadow.width / 2), y + image.height - assets.shadow.height + 16]);
        items.push([z, image, x, y]);
        if (sim.bunCarried[pid] >= 0) items.push([z + 1, () => drawBun(x + image.width / 2, y + 8, sim.bunCarried[pid], 1, 0.8, now)]);
        const heldKey = sim.heldItem ? heldItemSpriteKey(sim.heldItem[pid]) : null;
        if (heldKey && assets.items && assets.items[heldKey]) {
          items.push([z + 2, () => drawHeldItem(assets.items[heldKey], x + image.width * 0.8, y + image.height * 0.25, now)]);
        }
      }
      items.sort((a, b) => a[0] - b[0]);
      for (const item of items) typeof item[1] === 'function' ? item[1]() : ctx.drawImage(item[1], item[2], item[3]);
      // 玩家(pid0)被炸掉→复活期间压暗画面：alpha 随复活倒计时消退，复活瞬间恢复。
      if (sim.isBun && !sim.alive[0] && sim.bunRespawn[0] > 0) {
        const frac = Math.max(0, Math.min(1, sim.bunRespawn[0] / (sim.bunRespawnTicks || 1)));
        ctx.fillStyle = `rgba(0,0,0,${(0.62 * frac).toFixed(3)})`;
        ctx.fillRect(0, 0, canvas.width, canvas.height - BOARD_OFFSET);
      }
      if (sim.isBun && !sim.done) drawRespawnCountdowns(sim, motion ? motion.humanPid : 0);
      ctx.restore();
      drawResult(matchResult(sim, motion ? motion.humanPid : 0));
    }
    const RESULT_COLORS = { win: '#ffd54a', lose: '#ff7a7a', draw: '#d8e6ea' };
    function drawResult(result) {
      if (!result) return;
      const cx = canvas.width / 2, cy = canvas.height / 2;
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
    // 本地玩家阵亡：画面中央大字倒计时；其余阵亡者：在复活点显示小号秒数。
    function drawRespawnCountdowns(sim, humanPid) {
      for (let pid = 0; pid < 2; pid++) {
        if (sim.alive[pid] || !(sim.bunRespawn[pid] > 0)) continue;
        const seconds = respawnSeconds(sim.bunRespawn[pid]);
        ctx.save();
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.lineJoin = 'round';
        if (pid === humanPid) {
          const cx = canvas.width / 2, cy = (canvas.height - BOARD_OFFSET) / 2;
          ctx.font = 'bold 34px sans-serif';
          ctx.lineWidth = 6; ctx.strokeStyle = 'rgba(0,0,0,0.75)'; ctx.fillStyle = '#ffffff';
          ctx.strokeText('你被炸中了', cx, cy - 48); ctx.fillText('你被炸中了', cx, cy - 48);
          ctx.font = 'bold 72px sans-serif'; ctx.lineWidth = 8; ctx.fillStyle = '#ffd54a';
          ctx.strokeText(String(seconds), cx, cy + 20); ctx.fillText(String(seconds), cx, cy + 20);
          ctx.font = 'bold 22px sans-serif'; ctx.lineWidth = 5; ctx.fillStyle = '#ffffff';
          ctx.strokeText('秒后复活', cx, cy + 78); ctx.fillText('秒后复活', cx, cy + 78);
        } else {
          const spawn = (sim.bunSpawnPos && sim.bunSpawnPos[pid]) || null;
          if (!spawn) { ctx.restore(); continue; }
          const x = spawn[1] * CELL, y = spawn[0] * CELL - CELL * 0.35;
          ctx.fillStyle = 'rgba(0,0,0,0.6)';
          ctx.beginPath(); ctx.arc(x, y, 20, 0, Math.PI * 2); ctx.fill();
          ctx.font = 'bold 22px sans-serif'; ctx.fillStyle = '#ff8a8a';
          ctx.fillText(String(seconds), x, y + 1);
        }
        ctx.restore();
      }
    }
    return { render, addExplosion, reset };
  }

  return { CELL, BOARD_OFFSET, BUN_ELEMENT_IDS, ITEM_FRAME_MS, crateSpriteKey, levelElementIds, bombFrame, bombAgeSeconds, playerVisualY, respawnSeconds, heldItemSpriteKey, matchResult, bunTokens, explosionFrame, loadAssets, createRenderer };
});
