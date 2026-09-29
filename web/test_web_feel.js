#!/usr/bin/env node
'use strict';

// 网页手感回归（连续逐帧移动重做）：
//  1) sim.frameStep：本地人类逐帧连续移动（真实 dt 缩放 / 撞墙夹紧 / realtime 位跳过 10Hz 移动）。
//  2) sim.probeMoveDist：三态判定（完全可走 / 部分可走 / 贴墙被挡）——autoTurn 侧滑门控的基石。
//  3) 渲染：本地人类(humanPid)raw 不插值、对手 10Hz 插值、大位移吸附、复活压暗。
//  4) app.js 接线契约：humanAction 第4位=1、rAF frameStep、autoTurn+MIN_OFF=0.25、无 steerReduced。
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
  // 撞墙夹紧：右侧贴墙(box 前缘顶墙)时 frameStep 不得穿墙。
  const s = openSim();
  s.pos[0] = 5.5; s.pos[1] = 6 - CFG.radius - 1e-4;
  s.wall[5 * W + 6] = 1;
  s.frameStep(0, MOVE_RIGHT, 0.1);
  assert(s.pos[1] <= 6 - CFG.radius + 1e-6, '贴墙时 frameStep 不得穿墙');
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

// ---------- 2) probeMoveDist 三态 ----------
{
  const s = openSim(); s.pos[0] = 5.5; s.pos[1] = 5.5;
  const full = s.probeMoveDist(0, MOVE_RIGHT);
  const stepLen = CFG.stepLen * s.spdG[0] * s.playerMoveScale(0);
  assert(Math.abs(full - stepLen) < 1e-6, '完全可走：probe ≈ 满步长');
  s.wall[5 * W + 6] = 1;
  const partial = s.probeMoveDist(0, MOVE_RIGHT);
  assert(partial > stepLen * 0.05 && partial < stepLen * 0.95, `部分可走：5%~95% 步长，实得 ${partial.toFixed(3)}`);
  s.pos[1] = 6 - CFG.radius - 1e-4;
  const blocked = s.probeMoveDist(0, MOVE_RIGHT);
  assert(blocked < stepLen * 0.05, `贴墙被挡：probe < 5% 步长，实得 ${blocked.toFixed(4)}`);
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

// ---------- 5) sim.js / app.js 接线契约 ----------
const simSource = fs.readFileSync(path.join(__dirname, 'sim.js'), 'utf8');
assert(!/steerReduced/.test(simSource), 'sim.js 必须已彻底移除 steerReduced（连续移动下作废）');
assert(/frameStep\s*\(pid, mv, dtSec\)/.test(simSource), 'sim.js 必须提供 frameStep(pid, mv, dtSec)');
assert(/probeMoveDist\s*\(pid, mv\)/.test(simSource), 'sim.js 必须提供 probeMoveDist(pid, mv)');

const appSource = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
assert(!/steerReduced/.test(appSource), 'app.js 必须已移除 steerReduced 接线');
assert(/\[QQT\.MOVE_IDLE, bombQueued \? 1 : 0, 0, 1\]/.test(appSource),
  'humanAction 必须返回第4位=1（跳过 10Hz 移动，改由 rAF frameStep）');
assert(/const MIN_OFF = 0\.25;/.test(appSource), 'autoTurn 触发阈值 MIN_OFF 必须为 0.25（对齐边长 1/4）');
assert(/if \(off >= MIN_OFF\) return move;/.test(appSource), 'autoTurn：偏移≥MIN_OFF(居中/正前方)不得侧滑');
assert(/if \(moved > stepLen \* 0\.05\) return move;/.test(appSource),
  'autoTurn：仅贴墙被挡(<5%步长)才触发，部分可走不侧滑');
assert(/sim\.frameStep\(0, eff, dt\)/.test(appSource), 'rAF 必须逐帧调用 sim.frameStep(0, ...) 连续移动本地人类');
assert(/humanPid: localHumanControls\(\) \? 0 : -1/.test(appSource),
  'motionState 必须传 humanPid（本地人类 raw / 观战回放插值）');
assert(/stepHumanFrame\(now\)/.test(appSource), 'rAF 回调必须先驱动 stepHumanFrame(now)');

console.log('网页手感回归（连续逐帧移动 / autoTurn 侧滑 / 人类raw渲染 / 复活压暗）通过');
