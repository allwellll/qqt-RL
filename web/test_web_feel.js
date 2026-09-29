#!/usr/bin/env node
'use strict';

// 针对三项网页手感修复的定向回归：
//  1) 降敏拐角滑移（sim.steerReduced，人类 pid0）——去掉门框归中误滑，保留真外拐角圆角。
//  2) 渲染插值（visual_renderer 在 10Hz tick 间线性插值 → 60fps；本地人类不插值；大位移吸附）。
//  3) 复活压暗（isBun 且 pid0 阵亡复活倒计时中，画面变暗；复活后恢复）。
// 三者均不得改动模型推理数值（parity 由 test_mlp4_parity.js 独立守护）。

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const QQT = require('./sim.js');
const visual = require('./visual_renderer.js');

const W = 15, N = 195, DIST = 0.3;

// ---------- 1) steerReduced 行为回归 ----------
function blockedGrid(cells) {
  const b = new Uint8Array(N);
  for (const [r, c] of cells) b[r * W + c] = 1;
  return b;
}

const sim = new QQT.Sim(1);

// 门框卡位：向右走，目标格 (5,6) 是通路，但 (4,6) 挡住上半身贴门框 → 直行受阻。
// 完整 _steer（模型/训练口径）朝行中心竖直归中滑入；降敏模式不再竖直归中，改为沿原轴继续。
const doorway = blockedGrid([[4, 6]]);
sim.steerReduced = [false, false];
const doorFull = sim._steer(5.1, 5.5, 3, doorway, DIST, 0);
sim.steerReduced = [true, true];
const doorReduced = sim._steer(5.1, 5.5, 3, doorway, DIST, 0);
assert(Math.abs(doorFull[0] - 5.1) > 0.1,
  '完整 _steer 必须在门框卡位时竖直归中滑移（模型/训练口径不变）');
assert(Math.abs(doorReduced[0] - 5.1) < 1e-6,
  '降敏模式必须取消门框归中误滑（人类朝前走贴泡/贴墙不再侧滑）');
assert(doorReduced[1] > 5.5 + 1e-3,
  '降敏模式仍应沿原前进轴推进，而非原地卡死');

// 真外拐角：目标格 (5,6) 被堵、仅下侧 (6,6)/(6,5) 开放 → 两种模式都应被动向下圆角滑移。
const corner = blockedGrid([[4, 6], [5, 6]]);
sim.steerReduced = [false, false];
const cornerFull = sim._steer(5.1, 5.5, 3, corner, DIST, 0);
sim.steerReduced = [true, true];
const cornerReduced = sim._steer(5.1, 5.5, 3, corner, DIST, 0);
assert(Math.abs(cornerFull[0] - cornerReduced[0]) < 1e-9 &&
       Math.abs(cornerFull[1] - cornerReduced[1]) < 1e-9,
  '真外拐角圆角滑移不得被降敏模式误删（正常走仍会侧滑）');
assert(cornerReduced[0] > 5.1 + 0.1,
  '真外拐角必须仍向开放侧滑移');

// 默认（未设 steerReduced 或 [false,false]）＝完整口径，保证模型对手/训练一致。
const simDefault = new QQT.Sim(1);
assert.deepStrictEqual(Array.from(simDefault.steerReduced), [false, false],
  'Sim 默认 steerReduced 必须为 [false,false]（模型/训练不受影响）');
const defaultOut = simDefault._steer(5.1, 5.5, 3, doorway, DIST, 0);
assert.deepStrictEqual(defaultOut, doorFull,
  '默认口径必须与完整 _steer 逐位一致（parity 前提）');

// ---------- 渲染插值 + 复活压暗（mock canvas） ----------
const CELL = visual.CELL;

function tagImg(tag, w = 40, h = 40) { return { tag, width: w, height: h }; }

function mockAssets() {
  const players = [];
  for (let pid = 0; pid < 2; pid++) {
    const rows = [];
    for (let r = 0; r < 4; r++) {
      const frames = [];
      for (let c = 0; c < 4; c++) frames.push(tagImg(`p${pid}`, 40, 40));
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
  let fillStyle = '';
  const ctx = {
    set fillStyle(v) { fillStyle = v; }, get fillStyle() { return fillStyle; },
    imageSmoothingEnabled: true,
    shadowColor: '', shadowBlur: 0, shadowOffsetY: 0,
    strokeStyle: '', lineWidth: 0, font: '', textAlign: '', textBaseline: '',
    save() {}, restore() {}, translate() {}, beginPath() {}, ellipse() {},
    fill() {}, arc() {}, stroke() {}, fillText() {},
    drawImage(img, a, b) { if (arguments.length === 3 && img && img.tag) draws.push({ tag: img.tag, x: a, y: b, w: img.width }); },
    fillRect(x, y, w, h) { rects.push({ fillStyle, x, y, w, h }); },
  };
  return { canvas: { width: 900, height: 810, getContext: () => ctx }, draws, rects };
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
  };
}

const level = { layers: [new Int16Array(N), new Int16Array(N)] };

function playerGx(draws, pid) {
  const hit = draws.find((d) => d.tag === `p${pid}`);
  assert(hit, `pid${pid} 精灵必须被绘制`);
  return (hit.x + hit.w / 2) / CELL; // 反解 gx：x = round(gx*CELL - w/2)
}

// 2a) 观战/回放（humanPid=-1）：两个角色都按 alpha 插值。
{
  const { canvas, draws } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 6.8] }); // 当前(=cur)位置；两角色位移均 <=1.0 格
  const motion = {
    prevPos: Float64Array.from([5.0, 2.0, 5.0, 6.0]),
    curPos: Float64Array.from([5.0, 3.0, 5.0, 6.8]),
    lastTickT: 1000, tickMs: 100, humanPid: -1,
  };
  r.render(s, 1050, motion); // alpha = (1050-1000)/100 = 0.5
  assert(Math.abs(playerGx(draws, 0) - 2.5) < 0.02, 'pid0 观战应插值到 prev/cur 中点(2.5)');
  assert(Math.abs(playerGx(draws, 1) - 6.4) < 0.02, 'pid1 观战应插值到 prev/cur 中点(6.4)');
}

// 2b) 人类对战（humanPid=0）：pid0 用原始 sim.pos（输入即时），pid1 插值。
{
  const { canvas, draws } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 6.8] });
  const motion = {
    prevPos: Float64Array.from([5.0, 2.0, 5.0, 6.0]),
    curPos: Float64Array.from([5.0, 3.0, 5.0, 6.8]),
    lastTickT: 1000, tickMs: 100, humanPid: 0,
  };
  r.render(s, 1050, motion);
  assert(Math.abs(playerGx(draws, 0) - 3.0) < 0.02, 'pid0 人类不插值，须用原始 sim.pos(3.0)');
  assert(Math.abs(playerGx(draws, 1) - 6.4) < 0.02, 'pid1 对手仍插值(6.4)');
}

// 2c) 大位移（复活/传送 > 1.0 格）不插值，直接吸附 curPos，避免横扫全图。
{
  const { canvas, draws } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 12.0] });
  const motion = {
    prevPos: Float64Array.from([5.0, 3.0, 5.0, 2.0]),  // pid1 位移 10 格
    curPos: Float64Array.from([5.0, 3.0, 5.0, 12.0]),
    lastTickT: 1000, tickMs: 100, humanPid: -1,
  };
  r.render(s, 1050, motion);
  assert(Math.abs(playerGx(draws, 1) - 12.0) < 0.02, '大位移必须吸附 curPos(12.0)，不得插值到中点');
}

// 2d) 无 motion（回退）：使用原始 sim.pos，alpha=1。
{
  const { canvas, draws } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 4.0, 5.0, 9.0] });
  r.render(s, 1050, null);
  assert(Math.abs(playerGx(draws, 0) - 4.0) < 0.02, '无 motion 时 pid0 用原始位置');
  assert(Math.abs(playerGx(draws, 1) - 9.0) < 0.02, '无 motion 时 pid1 用原始位置');
}

function dimRects(rects, canvas) {
  // 复活压暗：覆盖整块棋盘、fillStyle 为半透明黑的 fillRect。
  return rects.filter((rc) => /^rgba\(0,0,0,/.test(String(rc.fillStyle)) &&
    rc.x === 0 && rc.y === 0 && rc.w === canvas.width);
}

// 3a) pid0 阵亡且复活倒计时中 → 压暗一层。
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

// 3b) 存活时不压暗。
{
  const { canvas, rects } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 8.0], alive: [1, 1], isBun: true, bunRespawn: [0, 0] });
  r.render(s, 1050, null);
  assert(dimRects(rects, canvas).length === 0, '存活时不得压暗');
}

// 3c) 复活瞬间（倒计时归 0）恢复亮度。
{
  const { canvas, rects } = mockCanvas();
  const r = visual.createRenderer(canvas, level, mockAssets());
  const s = fakeSim({ pos: [5.0, 3.0, 5.0, 8.0], alive: [0, 1], isBun: true, bunRespawn: [0, 0] });
  r.render(s, 1050, null);
  assert(dimRects(rects, canvas).length === 0, '复活倒计时归零瞬间必须恢复亮度');
}

// ---------- app.js 接线契约 ----------
const appSource = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
assert(/sim\.steerReduced = \[matchMode\.value !== 'model-vs-rule', false\]/.test(appSource),
  'app 必须仅对人类 pid0 启用降敏，观战(模型)保持完整 _steer');
assert(/humanPid = \(replayDocument \|\| matchMode\.value === 'model-vs-rule'\) \? -1 : 0/.test(appSource),
  'app 观战/回放须令 humanPid=-1（两角色都插值），人类对战 humanPid=0');
assert(appSource.includes('renderer.render(sim, now, motionState())'),
  'app 必须把 motion 传入 renderer 以启用插值');
assert(appSource.includes('snapMotion()'), 'reset/回放必须吸附 motion，避免首帧横扫');

console.log('网页手感修复（降敏侧滑 / 渲染插值 / 复活压暗）定向回归通过');
