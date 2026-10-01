#!/usr/bin/env node
'use strict';

// 网页手感回归（连续逐帧移动重做）：
//  1) sim.frameStep：本地人类逐帧连续移动（真实 dt 缩放 / 撞墙夹紧 / realtime 位跳过 10Hz 移动）。
//  2) frameStep 原版像素物理：±19px 前缘、6px 拐角修正、泡泡 3px 入口带、穿泡计时。
//  3) 渲染：本地人类(humanPid)raw 不插值、对手 10Hz 插值、大位移吸附、复活压暗。
//  4) app.js 接线契约：humanAction 第4位=1、rAF frameStep、无 autoTurn/steerReduced。
//  5) parity 隔离：_steer 默认 new 口径不变（真外拐角仍侧滑），sim.js 已无 steerReduced。
// 模型推理数值 parity 由 test_mlp4_parity.js 独立守护，本测试不得触碰 step/_steer/legalMask。

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const QQT = require('./sim.js');
const visual = require('./visual_renderer.js');

const { MOVE_UP, MOVE_RIGHT, MOVE_IDLE, N, W, CFG } = QQT;

function openSim() {
  const s = new QQT.Sim(1);
  s.reset('open');
  s.wall.fill(0); s.brick.fill(0); s.fuse.fill(0);
  return s;
}

// ---------- 1) frameStep 连续逐帧移动 ----------
{
  // 位移随 dt 线性缩放：dt=0.1 应约为 dt=0.05 的 2 倍（同一起点开阔地向右）。
  const a = openSim(); a.pos[0] = 5.5; a.pos[1] = 5.5;
  a.frameStep(0, MOVE_RIGHT, 0.1);
  const d1 = a.pos[1] - 5.5;
  const b = openSim(); b.pos[0] = 5.5; b.pos[1] = 5.5;
  b.frameStep(0, MOVE_RIGHT, 0.05);
  const d2 = b.pos[1] - 5.5;
  assert(d1 > 0 && d2 > 0, 'frameStep 必须在开阔地向前推进 pos');
  assert(Math.abs(d1 / d2 - 2) < 0.05, `位移应随 dt 线性缩放(≈2x)，实得 ${(d1 / d2).toFixed(3)}`);
  // dt 上限 0.1：dt=0.5 与 dt=0.1 位移相同（防丢帧瞬移穿墙）。
  const c = openSim(); c.pos[0] = 5.5; c.pos[1] = 5.5;
  c.frameStep(0, MOVE_RIGHT, 0.5);
  assert(Math.abs((c.pos[1] - 5.5) - d1) < 1e-9, 'dt 应被夹到 0.1 上限，大 dt 不得瞬移');
}

{
  // 撞墙夹紧：前缘 +19px 顶墙即停，停在格中心 +1px（x=260px）。
  const s = openSim();
  s.pos[0] = 5.5; s.pos[1] = 5.5;
  s.wall[5 * W + 7] = 1;
  for (let k = 0; k < 20; k++) s.frameStep(0, MOVE_RIGHT, 0.05);
  assert.equal(Math.round(s.pos[1] * QQT.NATIVE_CELL_PX), 260, '贴墙时 frameStep 必须停在 260px，不得穿墙');
}

{
  // realtime 位=1 → 10Hz step 不移动该玩家（移动交给 rAF frameStep）；=0 → 正常移动。
  const skip = openSim(); skip.pos[0] = 5.5; skip.pos[1] = 5.5;
  skip.step([[MOVE_RIGHT, 0, 0, 1], [MOVE_IDLE, 0, 0, 1]]);
  assert(Math.abs(skip.pos[1] - 5.5) < 1e-9, 'realtime 位=1 时 step 不得移动 pid0');
  const move = openSim(); move.pos[0] = 5.5; move.pos[1] = 5.5;
  move.step([[MOVE_RIGHT, 0, 0, 0], [MOVE_IDLE, 0, 0, 1]]);
  assert(move.pos[1] > 5.5 + 1e-3, 'realtime 位=0 时 step 应照常移动 pid0（模型/训练口径）');
}

// ---------- 2) 原版像素物理 ----------
const PX = QQT.NATIVE_CELL_PX;
function hold(s, mv, ms) { for (let t = 0; t < ms; t += 16) s.frameStep(0, mv, 0.016); }
{
  // 6px 拐角容差：x=236px 向上，左角点在被堵格(4,5)、右角点已进开放格(4,6)，余数 36≥34 → 滑到 260 再上行。
  const s = openSim(); s.pos[0] = 5.5; s.pos[1] = 236 / PX;
  s.wall[4 * W + 5] = 1;
  hold(s, MOVE_UP, 600);
  assert(Math.abs(s.pos[1] * PX - 260) <= 1 && s.pos[0] * PX < 200, `6px 容差内必须拐角滑入开放列，实得 (${s.pos[1] * PX}, ${s.pos[0] * PX})`);
}
{
  // 超出容差(x=230px)：两个角点都在被堵格 → 原版不修正，顶墙停在格中心行。
  const s = openSim(); s.pos[0] = 5.5; s.pos[1] = 230 / PX;
  s.wall[4 * W + 5] = 1;
  hold(s, MOVE_UP, 600);
  assert(Math.abs(s.pos[1] * PX - 230) <= 1 && Math.abs(s.pos[0] * PX - 220) <= 1, '两角都被挡时不得侧滑');
}
{
  // 只有一个角被挡且余数 <20：向开放侧修正到格中心（x=210 → 220 后上行需前方开放）。
  const s = openSim(); s.pos[0] = 5.5; s.pos[1] = 210 / PX;
  s.wall[4 * W + 4] = 1;
  hold(s, MOVE_UP, 600);
  assert(Math.abs(s.pos[1] * PX - 220) <= 1 && s.pos[0] * PX < 200, '单角被挡时必须修正到本格中心后前进');
}
{
  // 泡泡 3px 入口带：脚下泡可走出；正前方泡在前缘 +19px 处挡住（x=220px）。
  const own = openSim(); own.pos[0] = 5.5; own.pos[1] = 5.5; own.fuse[5 * W + 5] = 30;
  hold(own, MOVE_RIGHT, 400);
  assert(own.pos[1] * PX > 300, '必须能走出脚下刚放的泡泡');
  const ahead = openSim(); ahead.pos[0] = 5.5; ahead.pos[1] = 5.5; ahead.fuse[5 * W + 6] = 30;
  hold(ahead, MOVE_RIGHT, 800);
  assert.equal(Math.round(ahead.pos[1] * PX), 220, '正前方泡泡必须在入口带挡住');
  // 刚放的泡泡：中心还在泡泡格内可自由来回；中心越过格边界后回走即被挡住（无需整个身体离开）。
  const stepOff = (px) => {
    const s = openSim(); s.pos[0] = 5.5; s.pos[1] = 5.5; s.fuse[5 * W + 5] = 30;
    while (s.pos[1] * PX < px) s.frameStep(0, MOVE_RIGHT, 0.004);
    return s;
  };
  const inside = stepOff(236);
  hold(inside, QQT.MOVE_LEFT, 300);
  assert(inside.pos[1] * PX < 215, `中心仍在泡泡格内时必须能往回走，实得 ${inside.pos[1] * PX}`);
  const edge = stepOff(242);
  const edgeX = edge.pos[1] * PX;
  assert(edgeX - 19 < 240, '前提：身体仍与泡泡格重叠');
  hold(edge, QQT.MOVE_LEFT, 400);
  assert(Math.abs(edge.pos[1] * PX - edgeX) <= 1, `中心越过边界后回走必须被泡泡挡住，实得 ${edge.pos[1] * PX}`);
  for (const [mv, dy, dx] of [[QQT.MOVE_UP, -1, 0], [QQT.MOVE_DOWN, 1, 0], [QQT.MOVE_LEFT, 0, -1]]) {
    const s = openSim(); s.pos[0] = 5.5; s.pos[1] = 5.5; s.fuse[5 * W + 5] = 30;
    const back = [QQT.MOVE_DOWN, MOVE_UP, MOVE_RIGHT][[QQT.MOVE_UP, QQT.MOVE_DOWN, QQT.MOVE_LEFT].indexOf(mv)];
    const axis = dy ? 0 : 1;
    while (Math.abs(s.pos[axis] - 5.5) * PX < 23) s.frameStep(0, mv, 0.004);
    const at = s.pos[axis];
    hold(s, back, 400);
    assert(Math.abs(s.pos[axis] - at) * PX <= 1, `方向 ${mv} 离开后回走必须被挡`);
  }
  // 顶泡 500~600ms 后放泡 → 获得一格穿泡。
  const pass = openSim(); pass.pos[0] = 5.5; pass.pos[1] = 5.5; pass.fuse[5 * W + 6] = 30;
  hold(pass, MOVE_RIGHT, 560);
  assert(pass._activateNativePass(0), '顶泡 500~600ms 后放泡必须激活穿泡');
  hold(pass, MOVE_RIGHT, 400);
  assert(pass.pos[1] * PX > 260, '穿泡激活后必须能穿过前方泡泡');
  const early = openSim(); early.pos[0] = 5.5; early.pos[1] = 5.5; early.fuse[5 * W + 6] = 30;
  hold(early, MOVE_RIGHT, 300);
  assert(!early._activateNativePass(0), '顶泡不足 500ms 不得激活穿泡');
}

// ---------- 3) parity 隔离：_steer 默认 new 口径不变 ----------
{
  // 真外拐角(目标格被堵、仅下侧开放)：默认 _steer 仍应被动向开放侧圆角滑移。
  const s = new QQT.Sim(1); s.reset('open'); s.wall.fill(0); s.brick.fill(0); s.fuse.fill(0);
  const DIST = 0.3;
  const blocked = new Uint8Array(N);
  blocked[4 * W + 6] = 1; blocked[5 * W + 6] = 1;  // (5,6) 目标堵、(4,6) 上侧堵 → 仅下侧开放
  const out = s._steer(5.1, 5.5, 3, blocked, DIST, 0);
  assert(out[0] > 5.1 + 0.05, '真外拐角：默认 _steer 仍向开放(下)侧被动滑移（parity 口径不变）');
}

// ---------- 4) 渲染（mock canvas）：人类 raw / 对手插值 / 大位移吸附 / 复活压暗 ----------
const CELL = visual.CELL;
function tagImg(tag, w = 40, h = 40) { return { tag, width: w, height: h }; }
function mockAssets() {
  const players = [];
  for (let pid = 0; pid < 2; pid++) {
    const rows = [];
    for (let r = 0; r < 4; r++) {
      const frames = [];
      for (let c = 0; c < 4; c++) { const img = tagImg(`p${pid}`, 40, 40); img.row = r; frames.push(img); }
      rows.push(frames);
    }
    players.push(rows);
  }
  const flames = { C: [null, tagImg('flameC1'), tagImg('flameC2')], U: [], D: [], L: [], R: [] };
  return {
    elements: {}, elementImages: new Map(),
    background: tagImg('bg', 900, 780), baseBand: tagImg('band', 900, 30),
    players, bombs: [tagImg('bomb')], flames, shadow: tagImg('shadow', 30, 20),
  };
}
function mockCanvas() {
  const draws = [];
  const rects = [];
  const texts = [];
  let fillStyle = '';
  const ctx = {
    set fillStyle(v) { fillStyle = v; }, get fillStyle() { return fillStyle; },
    imageSmoothingEnabled: true,
    shadowColor: '', shadowBlur: 0, shadowOffsetY: 0,
    strokeStyle: '', lineWidth: 0, font: '', textAlign: '', textBaseline: '',
    save() {}, restore() {}, translate() {}, scale() {}, beginPath() {}, ellipse() {}, strokeRect() {},
    createRadialGradient() { return { addColorStop() {} }; },
    fill() {}, arc() {}, stroke() {}, strokeText() {},
    fillText(text, x, y) { texts.push({ text: String(text), x, y }); },
    drawImage(img, a, b) { if (arguments.length === 3 && img && img.tag) draws.push({ tag: img.tag, x: a, y: b, w: img.width, row: img.row }); },
    fillRect(x, y, w, h) { rects.push({ fillStyle, x, y, w, h }); },
  };
  return { canvas: { width: 900, height: 810, getContext: () => ctx }, draws, rects, texts };
}
function fakeSim(opts) {
  return {
    pos: Float64Array.from(opts.pos),
    alive: opts.alive || [1, 1],
    fuse: new Uint8Array(N),
    bunCarried: [-1, -1],
    wall: new Uint8Array(N), brick: new Uint8Array(N), cover: new Uint8Array(N),
    isBun: !!opts.isBun,
    bunBases: [], bunStored: [], bunLoose: new Uint8Array(N * 2),
    bunRespawn: opts.bunRespawn || [0, 0], bunRespawnTicks: 20,
    bunSpawnPos: [[6.5, 2.5], [6.5, 12.5]],
  };
}
const level = { layers: [new Int16Array(N), new Int16Array(N)] };
function playerGx(draws, pid) {
  const hit = draws.find((d) => d.tag === `p${pid}`);
  assert(hit, `pid${pid} 精灵必须被绘制`);
  return (hit.x + hit.w / 2) / CELL;
}

// 4a) 本地人类(humanPid=0)raw 不插值；对手(pid1)按 alpha 插值。
{
  const { canvas, draws } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 6.8] });   // sim.pos = 每帧连续真实位置
  const motion = {
    prevPos: Float64Array.from([5.0, 2.0, 5.0, 6.0]),
    curPos: Float64Array.from([5.0, 3.0, 5.0, 6.8]),
    lastTickT: 1000, tickMs: 100, humanPid: 0,
  };
  r.render(s, 1050, motion);                            // alpha = 0.5
  assert(Math.abs(playerGx(draws, 0) - 3.0) < 0.02, 'pid0(本地人类)必须用 sim.pos 原始值(3.0)，不插值');
  assert(Math.abs(playerGx(draws, 1) - 6.4) < 0.02, 'pid1(对手)应插值到中点(6.4)');
}

// 4b) 观战/回放(humanPid=-1)：两方都插值。
{
  const { canvas, draws } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 6.8] });
  const motion = {
    prevPos: Float64Array.from([5.0, 2.0, 5.0, 6.0]),
    curPos: Float64Array.from([5.0, 3.0, 5.0, 6.8]),
    lastTickT: 1000, tickMs: 100, humanPid: -1,
  };
  r.render(s, 1050, motion);
  assert(Math.abs(playerGx(draws, 0) - 2.5) < 0.02, '观战 pid0 应插值到中点(2.5)');
  assert(Math.abs(playerGx(draws, 1) - 6.4) < 0.02, '观战 pid1 应插值到中点(6.4)');
}

// 4c) 大位移(复活/传送 >1格)不插值，直接吸附 curPos。
{
  const { canvas, draws } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 12.0] });
  const motion = {
    prevPos: Float64Array.from([5.0, 3.0, 5.0, 2.0]),
    curPos: Float64Array.from([5.0, 3.0, 5.0, 12.0]),
    lastTickT: 1000, tickMs: 100, humanPid: 0,
  };
  r.render(s, 1050, motion);
  assert(Math.abs(playerGx(draws, 1) - 12.0) < 0.02, '大位移必须吸附 curPos(12.0)');
}

// 4d) 无 motion（回退）：使用原始 sim.pos。
{
  const { canvas, draws } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 4.0, 5.0, 9.0] });
  r.render(s, 1050, null);
  assert(Math.abs(playerGx(draws, 0) - 4.0) < 0.02, '无 motion 时 pid0 用原始位置');
  assert(Math.abs(playerGx(draws, 1) - 9.0) < 0.02, '无 motion 时 pid1 用原始位置');
}

function dimRects(rects, canvas) {
  return rects.filter((rc) => /^rgba\(0,0,0,/.test(String(rc.fillStyle)) &&
    rc.x === 0 && rc.y === 0 && rc.w === canvas.width);
}

// 4e) pid0 阵亡且复活倒计时中 → 压暗一层。
{
  const { canvas, rects } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 8.0], alive: [0, 1], isBun: true, bunRespawn: [15, 0] });
  r.render(s, 1050, null);
  const dims = dimRects(rects, canvas);
  assert(dims.length === 1, '阵亡复活期间必须压暗一层');
  const alphaVal = parseFloat(String(dims[0].fillStyle).match(/rgba\(0,0,0,([0-9.]+)\)/)[1]);
  assert(alphaVal > 0 && alphaVal <= 0.62, `压暗透明度应在 (0,0.62]，实得 ${alphaVal}`);
}

// 4f) 存活时不压暗 / 复活倒计时归零瞬间恢复亮度。
{
  const { canvas, rects } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 8.0], alive: [1, 1], isBun: true, bunRespawn: [0, 0] });
  r.render(s, 1050, null);
  assert(dimRects(rects, canvas).length === 0, '存活时不得压暗');
}
{
  const { canvas, rects } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 8.0], alive: [0, 1], isBun: true, bunRespawn: [0, 0] });
  r.render(s, 1050, null);
  assert(dimRects(rects, canvas).length === 0, '复活倒计时归零瞬间必须恢复亮度');
}

// 4g) 复活倒计时：本地玩家阵亡显示中央秒数；对手阵亡在复活点显示秒数。
{
  const { canvas, texts } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 8.0], alive: [0, 0], isBun: true, bunRespawn: [15, 42] });
  r.render(s, 1050, { prevPos: s.pos, curPos: s.pos, lastTickT: 1000, tickMs: 100, humanPid: 0 });
  assert(texts.some((t) => t.text === '2'), '本地玩家剩 15 tick 应显示 2 秒');
  assert(texts.some((t) => t.text === '秒后复活'), '本地玩家阵亡应显示复活提示');
  const bot = texts.find((t) => t.text === '5');
  assert(bot && Math.abs(bot.x - 12.5 * CELL) < 1e-6, '对手倒计时(42 tick→5 秒)应画在其复活点');
}
{
  const { canvas, texts } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 8.0], alive: [1, 1], isBun: true });
  r.render(s, 1050, null);
  assert.equal(texts.length, 0, '双方存活时不显示倒计时');
}

// 4h) 朝向跟随移动意图：顶墙不动时按上也必须面朝上（精灵行 3），松手后保持朝向。
{
  const { canvas, draws } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.5, 3.5, 5.5, 8.5] });
  const motion = (intents) => ({ prevPos: s.pos, curPos: s.pos, lastTickT: 1000, tickMs: 100, humanPid: 0, intents });
  const faceRow = (pid) => draws.filter((d) => d.tag === `p${pid}`).pop().row;
  r.render(s, 1000, motion([QQT.MOVE_IDLE, QQT.MOVE_IDLE]));
  assert.equal(faceRow(0), 0, '初始朝下');
  for (const [mv, row] of [[QQT.MOVE_UP, 3], [QQT.MOVE_LEFT, 1], [QQT.MOVE_RIGHT, 2], [QQT.MOVE_DOWN, 0]]) {
    r.render(s, 1016, motion([mv, mv]));
    assert.equal(faceRow(0), row, `位置不变(顶墙)时意图 ${mv} 必须朝向精灵行 ${row}`);
    assert.equal(faceRow(1), row, `对手顶墙时意图 ${mv} 同样须转向`);
  }
  r.render(s, 1032, motion([QQT.MOVE_UP, QQT.MOVE_IDLE]));
  r.render(s, 1048, motion([QQT.MOVE_IDLE, QQT.MOVE_IDLE]));
  assert.equal(faceRow(0), 3, '松手后保持最后朝向');
}

// ---------- 4i) 音效事件：放泡/爆炸全场可闻，拾取只对监听者 ----------
{
  const sound = require('./sound.js');
  const s = openSim(); s.pos[0] = 5.5; s.pos[1] = 5.5; s.pos[2] = 1.5; s.pos[3] = 1.5;
  s.crate[5 * W + 5] = 1; s.crateType[5 * W + 5] = 0;
  const before = sound.snapshot(s);
  const info = s.step([[MOVE_IDLE, 1, 0, 1], [MOVE_IDLE, 0, 0, 1]]);
  const events = sound.detectEvents(before, s, info, 0);
  assert(events.includes('place'), '放泡应触发 place 音效');
  assert(events.includes('pickup'), '踩到道具应触发 pickup 音效');
  assert(!events.includes('boom'), '未爆炸不应触发 boom');
  assert(!sound.detectEvents(before, s, info, 1).includes('pickup'), '对手的拾取不对监听者播放');
  assert(sound.detectEvents(null, s, { covered: Uint8Array.from([0, 1]) }, 0).includes('boom'), '有火焰覆盖应触发 boom');
}

// 4j) 香蕉皮/慢慢胶：拾取后手持，真人动作第3位=1 放到脚下格；角色旁绘制手持图标。
{
  const s = openSim(); s.isBun = false; s.pos[0] = 5.5; s.pos[1] = 5.5; s.pos[2] = 1.5; s.pos[3] = 1.5;
  for (const [crate, held, field] of [[3, 1, 1], [4, 2, 2]]) {
    const cell = 5 * W + 5;
    s.crate[cell] = 1; s.crateType[cell] = crate; s.heldItem[0] = 0; s.fieldItem[cell] = 0;
    s.step([[MOVE_IDLE, 0, 0, 1], [MOVE_IDLE, 0, 0, 1]]);
    assert.equal(s.heldItem[0], held, `拾取道具 ${crate} 后必须手持 ${held}`);
    s.step([[MOVE_IDLE, 0, 1, 1], [MOVE_IDLE, 0, 0, 1]]);
    assert.equal(s.heldItem[0], 0, '按放置键后手持道具清空');
    assert.equal(s.fieldItem[cell], field, '道具必须落在脚下格');
  }
  assert.equal(visual.heldItemSpriteKey(1), 'banana_pickup');
  assert.equal(visual.heldItemSpriteKey(2), 'glue_pickup');
  assert.equal(visual.heldItemSpriteKey(0), null);
  const { canvas, draws } = mockCanvas();
  const assets = mockAssets();
  assets.items = { banana_pickup: { ox: 0, oy: 0, frames: [tagImg('heldBanana', 40, 46)] } };
  const r = visual.createRenderer(canvas, level, assets);
  const sim = fakeSim({ pos: [5.5, 3.5, 5.5, 8.5] });
  sim.heldItem = [1, 0];
  const origDraw = canvas.getContext().drawImage;
  let heldDrawn = 0;
  canvas.getContext().drawImage = function (img) { if (img && img.tag === 'heldBanana') heldDrawn++; return origDraw.apply(this, arguments); };
  r.render(sim, 1000, null);
  assert.equal(heldDrawn, 1, '手持香蕉皮必须在角色旁绘制一次图标');
}
// 4j1) 多人组队：按队伍选精灵，名牌“你/队友/敌1/敌2”，胜负按队伍判。
{
  const { canvas, draws, texts } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const sim = fakeSim({ pos: [5.5, 3.5, 5.5, 8.5, 9.5, 3.5, 9.5, 8.5], alive: [1, 1, 1, 1] });
  sim.nPlayers = 4; sim.team = [0, 1, 0, 1];
  sim.bunCarried = [-1, -1, -1, -1]; sim.bunRespawn = [0, 0, 0, 0];
  const motion = { prevPos: sim.pos, curPos: sim.pos, lastTickT: 1000, tickMs: 100, humanPid: 0, intents: [4, 4, 4, 4] };
  r.render(sim, 1000, motion);
  assert.equal(draws.filter((d) => d.tag === 'p0').length, 2, '红队两人共用红方精灵');
  assert.equal(draws.filter((d) => d.tag === 'p1').length, 2, '蓝队两人共用蓝方精灵');
  const labels = texts.map((t) => t.text);
  for (const label of ['你', '队友', '敌1', '敌2']) assert(labels.includes(label), `多人名牌 ${label}`);
  assert.equal(visual.playerLabel({ team: [0, 1, 1] }, 2, 0, 0), '敌2');
  assert.equal(visual.matchResult({ done: true, isBun: true, t: 10, maxSteps: 2400, winner: 0, team: [0, 1, 0, 1], bunScore: [1, 0] }, 0).title, '胜利');
  assert.equal(visual.matchResult({ done: true, isBun: true, t: 10, maxSteps: 2400, winner: 1, team: [0, 1, 1], bunScore: [0, 1] }, 0).title, '失败');
  const one = mockCanvas();
  visual.createRenderer(one.canvas, level, mockAssets()).render(fakeSim({ pos: [5.5, 3.5, 5.5, 8.5] }), 1000, null);
  assert(!one.texts.some((t) => t.text === '你'), '1v1 不画名牌');
}
const appTeamSource = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
assert(/teams: teamLayout\(\)/.test(appTeamSource) && /'2v2': \[0, 1, 0, 1\]/.test(appTeamSource), 'app.js 按队伍模式创建多人对局');

// 4j3) 携带包子贴在头顶：包子底边落在帧顶透明留白以下，而不是悬在帧顶上方。
{
  const { canvas } = mockCanvas();
  const assets = mockAssets();
  assets.items = { bun: { ox: 3, oy: 12, frames: [tagImg('bun', 52, 63)] } };
  const r = visual.createRenderer(canvas, level, assets);
  const sim = fakeSim({ pos: [5.5, 3.5, 5.5, 8.5], isBun: true });
  sim.bunCarried = [1, -1];
  const ctx = canvas.getContext();
  const plain = ctx.drawImage;
  let bunTop = null, bunH = 0, spriteY = null;
  ctx.drawImage = function (img, x, y, w, h) {
    if (img && img.tag === 'bun') { bunTop = y; bunH = h; }
    if (img && img.tag === 'p0' && spriteY === null) spriteY = y;
    return plain.apply(this, arguments);
  };
  r.render(sim, 1000, null);
  assert(bunTop !== null && spriteY !== null, '应绘制携带包子与角色');
  const sink = bunTop + bunH - spriteY;
  assert(sink > 40 * 0.3 && sink < 40 * 0.5, `携带包子底边应落在头顶附近（实际 ${sink}px）`);
}

// 4j2) 原版道具栏（顶部带右侧 7 格 + 数量/键位）与糖泡包裹（倒计时秒数）。
{
  const { canvas, texts } = mockCanvas();
  const assets = mockAssets();
  assets.items = { glue_pickup: { ox: 0, oy: 0, frames: [tagImg('glue', 40, 46)] } };
  const r = visual.createRenderer(canvas, level, assets);
  const sim = fakeSim({ pos: [5.5, 3.5, 5.5, 8.5] });
  sim.heldItem = [2, 0];
  sim.nativeItems = true;
  sim.itemSlots = [[{ item: 2, count: 12 }], []];
  sim.trapped = [0, 45];
  r.render(sim, 1000, { prevPos: sim.pos, curPos: sim.pos, lastTickT: 1000, tickMs: 100, humanPid: 0, intents: [4, 4] });
  const labels = texts.map((t) => t.text);
  for (let i = 1; i <= 7; i++) assert(labels.includes(String(i)), `道具栏必须标出数字键 ${i}`);
  const countText = texts.find((t) => t.text === '12');
  assert(countText && countText.y > canvas.height - 70 && countText.x < 120, '道具栏在画面左下角并显示数量');
  assert(texts.some((t) => t.text === '5' && t.y > visual.BOARD_OFFSET * 3), '糖泡在角色头顶显示剩余秒数');
}
const controls = require('./controls.js');
assert(controls.ITEM_KEYS.includes('KeyE') && controls.ITEM_KEYS.includes('ShiftLeft'), 'E/Shift 必须是放道具键');
assert.deepEqual(controls.ITEM_SLOT_KEYS, ['Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5', 'Digit6', 'Digit7'], '数字键 1-7 对应道具栏');
assert(/ITEM_SLOT_KEYS\.indexOf\(event\.code\)/.test(fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8')), 'app.js 必须把数字键接到道具栏格位');
assert(/nativeItems: native, nativeTrap: native/.test(fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8')),
  '真人对局开启原版道具栏/糖泡，模型评测保持训练规则');

// 4k) 终局提示：运包成功/时间到 → 胜利/失败/平局。
{
  const base = { isBun: true, done: true, t: 50, maxSteps: 2400, bunScore: [1, 0], bunStored: [[1, 1], [1, 0]] };
  assert.deepEqual(visual.matchResult({ ...base, winner: 0 }, 0), { kind: 'win', title: '胜利', reason: '运包成功', score: '1 : 0' });
  assert.equal(visual.matchResult({ ...base, winner: 1 }, 0).title, '失败');
  const timeout = visual.matchResult({ ...base, t: 2400, winner: null, bunStored: [[1, 0], [0, 1]] }, 0);
  assert.deepEqual([timeout.title, timeout.reason, timeout.score], ['平局', '时间到', '1 : 1']);
  assert.equal(visual.matchResult({ ...base, winner: 1 }, -1).title, '红方胜利', '观战按阵营报胜者');
  assert.equal(visual.matchResult({ ...base, done: false, winner: null }, 0), null, '未结束不提示');
  const { canvas, texts } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const sim = fakeSim({ pos: [5.5, 3.5, 5.5, 8.5], isBun: true });
  Object.assign(sim, { done: true, winner: 0, t: 30, maxSteps: 2400, bunScore: [1, 0] });
  r.render(sim, 1000, null);
  assert(texts.some((t) => t.text === '胜利') && texts.some((t) => t.text === '按 R 重新开局'), '终局必须在画面显示结果');
}

// 4l) 炸砖：砖体残骸期仍挡路，但画面立即不画砖；掷出的道具立即可见，残骸结束才可拾取。
{
  const s = new QQT.Sim(3); s.reset('open');
  s.wall.fill(0); s.brick.fill(0); s.fuse.fill(0); s.crate.fill(0);
  s.crateRate = 1; s.itemsEnabled = true;
  const cell = 5 * W + 7;
  s.brick[cell] = 1; s.pos[0] = 5.5; s.pos[1] = 4.5; s.pos[2] = 11.5; s.pos[3] = 11.5;
  s.fuse[5 * W + 6] = 1; s.owner[5 * W + 6] = 1; s.bombBlast[5 * W + 6] = 2;
  s.step([[MOVE_IDLE, 0, 0, 1], [MOVE_IDLE, 0, 0, 1]]);
  assert(s.brick[cell] === 1 && s.brickLinger[cell] > 0, '炸砖后残骸期内砖仍是碰撞体');
  assert(s.pendingCrateType[cell] >= 0 && !s.crate[cell], '道具在炸砖瞬间掷定但尚不可拾取');
  const lv = { layers: [new Int16Array(N), new Int16Array(N)] };
  lv.layers[1][cell] = 8001;
  const { canvas, draws } = mockCanvas();
  const assets = mockAssets();
  assets.elements = { 8001: { w: 1, h: 1, xo: 0, yo: 0 } };
  assets.elementImages = new Map([[8001, tagImg('brick')]]);
  const key = visual.crateSpriteKey(s.pendingCrateType[cell], s.pendingSuperCrate[cell] === 1);
  assets.items = { [key]: { ox: 0, oy: 0, frames: [tagImg('crateItem')] } };
  const r = visual.createRenderer(canvas, lv, assets);
  r.render(s, 1000, null);
  assert(!draws.some((d) => d.tag === 'brick'), '被炸砖在残骸期内不得再画出');
  assert(draws.some((d) => d.tag === 'crateItem'), '炸出的道具必须立即显示');
  for (let k = 0; k < QQT.CFG.brickLingerTicks; k++) s.step([[MOVE_IDLE, 0, 0, 1], [MOVE_IDLE, 0, 0, 1]]);
  assert(s.brick[cell] === 0 && s.crate[cell] === 1 && s.pendingCrateType[cell] === -1, '残骸结束后才开放通行并落为可拾取道具');
}

// 4m) 放泡落在按键瞬间的格子：按空格后在同一 tick 内走回右格，泡仍放在左格。
{
  const s = openSim(); s.pos[0] = 5.5; s.pos[1] = 4.5; s.pos[2] = 1.5; s.pos[3] = 1.5;
  const pressCell = 5 * W + 4;
  s.pos[1] = 5.5;                                   // tick 前已回到右格
  s.step([[MOVE_IDLE, 1, 0, 1, pressCell, -1], [MOVE_IDLE, 0, 0, 1]]);
  assert(s.fuse[pressCell] > 0 && !(s.fuse[5 * W + 5] > 0), '泡必须落在按键时的左格');
  const d = openSim(); d.pos[0] = 5.5; d.pos[1] = 5.5;
  d.step([[MOVE_IDLE, 1, 0, 1], [MOVE_IDLE, 0, 0, 1]]);
  assert(d.fuse[5 * W + 5] > 0, '未给按键格时仍按当前中心格放泡（训练/模型口径不变）');
  const it = openSim(); it.pos[0] = 5.5; it.pos[1] = 5.5; it.heldItem[0] = 1;
  it.step([[MOVE_IDLE, 0, 1, 1, -1, pressCell], [MOVE_IDLE, 0, 0, 1]]);
  assert.equal(it.fieldItem[pressCell], 1, '道具同样落在按键时的格子');
}

// ---------- 5) sim.js / app.js 接线契约 ----------
const simSource = fs.readFileSync(path.join(__dirname, 'sim.js'), 'utf8');
assert(!/steerReduced/.test(simSource), 'sim.js 必须已彻底移除 steerReduced（连续移动下作废）');
assert(/frameStep\s*\(pid, mv, dtSec\)/.test(simSource), 'sim.js 必须提供 frameStep(pid, mv, dtSec)');
assert(/NATIVE_HALF_PX = 19/.test(simSource) && /NATIVE_CORNER_TOLERANCE_PX = 6/.test(simSource),
  'sim.js 必须使用原版 ±19px 前缘与 6px 拐角容差');

const appSource = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
assert(!/steerReduced/.test(appSource), 'app.js 必须已移除 steerReduced 接线');
assert(/\[QQT\.MOVE_IDLE, bombCell >= 0 \? 1 : 0, itemCell >= 0 \? 1 : 0, 1, bombCell, itemCell, itemSlot\]/.test(appSource),
  'humanAction 必须返回第4位=1（跳过 10Hz 移动，改由 rAF frameStep）');
assert(!/autoTurn|turnSlide/.test(appSource), 'app.js 必须移除 autoTurn（拐角修正由原版物理负责）');
assert(/const move = QQTControls\.moveForHeld\(held\);[\s\S]*?sim\.frameStep\(0, move, dt\)/.test(appSource), 'rAF 必须逐帧调用 sim.frameStep(0, ...) 连续移动本地人类');
assert(/intents\[0\] = sim\.alive\[0\] \? sim\.playerMoveDirection\(0, move\)/.test(appSource), 'rAF 必须把按键意图交给渲染器定朝向');
assert(/humanPid: localHumanControls\(\) \? 0 : -1/.test(appSource),
  'motionState 必须传 humanPid（本地人类 raw / 观战回放插值）');
assert(/stepHumanFrame\(now\)/.test(appSource), 'rAF 回调必须先驱动 stepHumanFrame(now)');

assert(/if \(QQTControls\.BOMB_KEYS\.includes\(event\.code\) && bombCell < 0\) bombCell = humanCell\(\);/.test(appSource), '放泡键按下瞬间必须记录所在格');
console.log('网页手感回归（连续逐帧移动 / 原版像素物理 / 人类raw渲染 / 复活压暗）通过');
