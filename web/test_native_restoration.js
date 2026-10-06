'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const Q = require('./sim.js');
const Sound = require('./sound.js');

function scene(options = {}) {
  const sim = new Q.Sim(1);
  sim.reset('open', { nativeItems: true, nativeTrap: true, ...options });
  for (const key of ['wall', 'brick', 'crate', 'fuse']) sim[key].fill(0);
  sim.pos.set([4.5, 3.5, 10.5, 10.5]);
  sim.invuln.fill(0); sim.spawnProtection.fill(0);
  return sim;
}

// Native x86 mixed-corner cases from upstream native_wall_pass_test.go.
const cases = [
  { walls: [[0, 1]], bombs: [[1, 1]], pos: [21, 40], active: true, allowed: true },
  { walls: [[1, 1]], bombs: [[0, 1]], pos: [21, 40], active: true, allowed: false },
  { walls: [[0, 1]], bombs: [[1, 1]], pos: [21, 40], active: false, allowed: false },
  { walls: [[1, 1]], bombs: [[1, 1]], pos: [21, 60], active: true, allowed: true },
  { walls: [[0, 2]], bombs: [[1, 1]], pos: [21, 40], active: true, allowed: false },
  { walls: [], bombs: [[1, 1]], pos: [21, 60], active: true, allowed: true },
  { walls: [], bombs: [[1, 1], [1, 2]], pos: [21, 60], active: true, allowed: false },
  { walls: [[1, 1]], bombs: [], pos: [21, 60], active: true, allowed: false },
];
for (let rotation = 0; rotation < 4; rotation++) {
  const move = [Q.MOVE_RIGHT, Q.MOVE_UP, Q.MOVE_LEFT, Q.MOVE_DOWN][rotation];
  const rotateCell = ([r, c]) => {
    for (let i = 0; i < rotation; i++) [r, c] = [9 - c, r];
    return r * Q.W + c;
  };
  for (const row of cases) {
    const sim = scene();
    row.walls.forEach((c) => { sim.wall[rotateCell(c)] = 1; });
    row.bombs.forEach((c) => { sim.fuse[rotateCell(c)] = 30; });
    let [x, y] = row.pos;
    for (let i = 0; i < rotation; i++) [x, y] = [y, 399 - x];
    const state = sim._nativeState(0); state.passActive = row.active;
    assert.equal(sim._nativePositionWalkable(state, x, y, move), row.allowed,
      `native mixed-corner case ${JSON.stringify(row)} rotation ${rotation}`);
  }
}

// Ordinary inputs enter a wall only after successful placement in the strict pass window.
for (const place of [false, true]) {
  const sim = scene();
  const wall = 3 * Q.W + 4;
  sim.wall[wall] = 1;
  sim.spdG[0] = 80 / 120;
  sim.fuse[4 * Q.W + 4] = 30; sim.owner[4 * Q.W + 4] = 1;
  const advance = (move, count) => { for (let i = 0; i < count; i++) sim.frameStep(0, move, 0.02); };
  advance(Q.MOVE_RIGHT, 26); advance(Q.MOVE_UP, 2);
  if (place) {
    const info = sim.step([[Q.MOVE_IDLE, 1, 0, 1], [Q.MOVE_IDLE, 0, 0, 1]]);
    assert(info.placed[0] && sim._nativeState(0).passActive, 'actual successful placement activates passage');
  }
  advance(Q.MOVE_UP, 11); advance(Q.MOVE_RIGHT, 13);
  const [r, c] = sim.centerCell(0);
  assert.equal(r * Q.W + c === wall, place, 'wall entry depends on the placement window');
  if (!place) continue;
  assert.deepStrictEqual(Array.from(sim.pos.slice(0, 2)).map((x) => Math.round(x * 40)), [159, 161]);
  sim.invuln[0] = 0;
  for (let tick = 0; tick < 60; tick++) {
    sim.frameStep(0, Q.MOVE_IDLE, 0.1);
    sim.step([[Q.MOVE_IDLE, 0, 0, 1], [Q.MOVE_IDLE, 0, 0, 1]]);
    assert(sim.alive[0] && !sim.trapped[0], 'wall occupant survives actual explosions after pass expires');
  }
  assert(!sim._nativeState(0).passActive && sim.fuse.every((f) => f === 0));
  advance(Q.MOVE_LEFT, 10);
  assert.notEqual(sim.centerCell(0).join(','), '3,4', 'wall occupant can leave');
}

for (const [ms, allowed] of [[500, false], [501, true], [599, true], [600, false]]) {
  const sim = scene(), state = sim._nativeState(0);
  Object.assign(state, { touchValid: true, touchStart: 0, touchLast: ms, clock: ms });
  assert.equal(sim._activateNativePass(0), allowed, `strict ${ms}ms boundary`);
}
{
  const sim = scene(), state = sim._nativeState(0);
  Object.assign(state, { touchValid: true, touchStart: 0, touchLast: 450, clock: 550 });
  assert(!sim._activateNativePass(0), '100ms stale contact must not activate passage');
}
{
  const sim = scene({ bananaSlideSpeedPx: 480 });
  sim.spdG[0] = 2.4;
  const cell = sim.centerCell(0)[0] * Q.W + sim.centerCell(0)[1];
  sim.fieldItem[cell] = Q.ITEM_BANANA; sim.fieldArmed[cell] = 1; sim.fieldOwner[cell] = 1;
  sim.step([[Q.MOVE_RIGHT, 0, 0, 1], [Q.MOVE_IDLE, 0, 0, 1]]);
  assert.equal(sim.movementStatus[0], Q.MOVE_STATUS_SLIDE, 'actual banana contact activates sliding');
  const x = sim.pos[1];
  for (let i = 0; i < 10; i++) sim.frameStep(0, Q.MOVE_LEFT, 0.02);
  assert(Math.abs((sim.pos[1] - x) * 40 - 96) <= 1, '480px/s slide ignores steering within native pixel rounding');
  sim.wall[4 * Q.W + 7] = 1;
  for (let i = 0; i < 100 && sim.movementStatus[0] === Q.MOVE_STATUS_SLIDE; i++) sim.frameStep(0, Q.MOVE_LEFT, 0.02);
  assert.equal(sim.movementStatus[0], Q.MOVE_STATUS_NONE, 'fast slide stops at obstacles');
  assert(sim.pos[1] < 7, 'slide cannot tunnel through a wall');
  const restored = scene();
  restored.restoreReplay(sim.snapshotReplay());
  assert.equal(restored.bananaSlideSpeedPx, 480, 'replay preserves custom banana speed');
  const original = scene().snapshotReplay();
  assert(!Object.hasOwn(original, 'bananaSlideSpeedPx'), 'default speed preserves legacy replay schema');
  restored.restoreReplay(original);
  assert.equal(restored.bananaSlideSpeedPx, 268, 'legacy replay restores original banana speed');
}
// A one-sided real blast does not trap a native actor visibly straddling the boundary;
// visual overlap and expired flames do not accumulate into a later hit either.
for (const x of [4.999, 5]) {
  const sim = scene();
  sim.pos[1] = x; sim.invuln[0] = 0;
  const cell = 4 * Q.W + 5;
  sim.fuse[cell] = 1; sim.bombBlast[cell] = 1; sim.owner[cell] = 1;
  sim.wall[cell - 1] = 1;
  const info = sim.step([[Q.MOVE_IDLE, 0], [Q.MOVE_IDLE, 0]]);
  assert(info.covered[cell] && !info.covered[cell - 1], 'fixture blast stops at the adjacent wall');
  assert.equal(sim.trapped[0] > 0, false, `native half-tile survives one-sided coverage at x=${x}`);
  sim.pos[1] = 5;
  sim.step([[Q.MOVE_IDLE, 0], [Q.MOVE_IDLE, 0]]);
  assert(!sim.trapped[0], 'entering expired flame visuals does not trap the native actor');
}
{
  const sim = scene(); sim.trapped[0] = 1;
  const before = Sound.snapshot(sim);
  const info = sim.step([[Q.MOVE_IDLE, 0], [Q.MOVE_IDLE, 0]]);
  assert(Sound.detectEvents(before, sim, info).includes('pop'), 'trap timeout plays native pop sound');
  const touched = scene();
  touched.trapped[0] = 20; touched.pos[2] = touched.pos[0]; touched.pos[3] = touched.pos[1];
  const touchBefore = Sound.snapshot(touched);
  const touchInfo = touched.step([[Q.MOVE_IDLE, 0], [Q.MOVE_IDLE, 0]]);
  assert(touchInfo.died[0] && Sound.detectEvents(touchBefore, touched, touchInfo).includes('pop'),
    'enemy contact plays the native pop sound');
  const rescued = scene({ teams: [0, 0] });
  rescued.trapped[0] = 20; rescued.pos[2] = rescued.pos[0]; rescued.pos[3] = rescued.pos[1];
  const rescueBefore = Sound.snapshot(rescued);
  const rescueInfo = rescued.step([[Q.MOVE_IDLE, 0], [Q.MOVE_IDLE, 0]]);
  assert(!Sound.detectEvents(rescueBefore, rescued, rescueInfo).includes('pop'), 'rescue does not play death sound');
}
const manifest = require('./assets/native/sprites.json');
assert.equal(manifest.actors.maomao.roleId, 9);
assert.equal(manifest.effects.trap.source, 'object/misc/misc111_trigger.img');
assert.equal(manifest.sound.source, 'sound/X12_01.wav');
for (const actor of Object.values(manifest.actors)) for (const sprite of Object.values(actor.actions)) {
  assert(sprite.frames > 0 && sprite.directions === 4);
  assert(fs.existsSync(path.join(__dirname, sprite.file)));
}
assert(fs.existsSync(path.join(__dirname, manifest.sound.file)));
console.log('Native syrup, pop sound, Maomao assets, banana sliding and wall passage: passed');
