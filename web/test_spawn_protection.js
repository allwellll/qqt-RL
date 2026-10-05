'use strict';
const assert = require('assert');
const Q = require('./sim.js');
const Hunter = require('./bun_hunter_bot.js');
const meta = require('./assets/native/sprites.json').effects.protection;
const level = require('./assets/maps/levels.json').find((l) => l.qqt_id === 806);
const idle = (sim) => sim.step(sim.team.map(() => [4, 0, 0, 0]));
function scene() {
  const sim = new Q.Sim(19);
  sim.reset(level, { nativeItems: true, nativeTrap: true, teams: [0, 1, 0, 1] });
  for (const key of ['wall', 'brick', 'crate', 'fuse']) sim[key].fill(0);
  sim.pos.set([5.5, 5.5, 10.5, 10.5, 11.5, 1.5, 11.5, 13.5]);
  return sim;
}
function hit(sim, owner = 1) {
  const [r, c] = sim.centerCell(0), cell = r * Q.W + c;
  sim.fuse[cell] = 1; sim.owner[cell] = owner; sim.bombBlast[cell] = 1;
  return idle(sim);
}
for (const owner of [0, 1, 2]) {
  const sim = scene();
  assert.deepStrictEqual(sim.spawnProtection, [30, 30, 30, 30]);
  assert.equal(Hunter.hunterStateFromSim(sim).players[0].invuln, 30);
  for (let tick = 0; tick < 30; tick++) {
    hit(sim, owner);
    assert(sim.alive[0] && !sim.trapped[0], `immune to own/enemy/ally blast at tick ${tick}`);
    assert.equal(sim.spawnProtection[0], 29 - tick);
  }
  hit(sim, owner);
  assert(sim.trapped[0] > 0, 'first tick after exact 3000ms expiry admits harm');
}
{
  const sim = scene();
  const cell = 5 * Q.W + 5;
  sim.fieldItem[cell] = Q.ITEM_BANANA; sim.fieldOwner[cell] = 1; sim.fieldArmed[cell] = 1;
  idle(sim);
  assert.equal(sim.movementStatus[0], Q.MOVE_STATUS_NONE, 'protection rejects banana field contact');
  assert.equal(sim.fieldItem[cell], Q.ITEM_BANANA, 'immune contact does not consume the field');
  sim.invuln[0] = 0; sim.spawnProtection[0] = 0; idle(sim);
  assert.equal(sim.movementStatus[0], Q.MOVE_STATUS_SLIDE, 'expired protection admits banana contact');
  sim._clearMovementStatus(0);
  sim.invuln[0] = 30; sim.spawnProtection[0] = 30;
  sim.fieldItem[cell] = Q.ITEM_SLOW_GLUE; sim.fieldOwner[cell] = 1; sim.fieldArmed[cell] = 1;
  idle(sim);
  assert.equal(sim.movementStatus[0], Q.MOVE_STATUS_NONE, 'protection rejects slow glue contact');
  sim.invuln[0] = 0; sim.spawnProtection[0] = 0; idle(sim);
  assert.equal(sim.movementStatus[0], Q.MOVE_STATUS_SLOW, 'expired protection admits slow glue');
}
{
  const sim = scene();
  sim._killPlayer(0);
  assert.equal(sim.spawnProtection[0], 0, 'death removes halo');
  sim.bunRespawn[0] = 1; sim.lastDied.fill(false);
  idle(sim);
  assert(sim.alive[0]);
  assert.equal(sim.spawnProtection[0], 30, 'respawn starts a complete new 3000ms interval');
  for (let n = 0; n < 30; n++) { hit(sim); assert(!sim.trapped[0]); }
  hit(sim); assert(sim.trapped[0] > 0);
}
{
  const sim = scene(); idle(sim);
  const frame = JSON.parse(JSON.stringify(sim.snapshotReplay(null)));
  const restored = new Q.Sim(3); restored.restoreReplay(frame);
  assert.deepStrictEqual(restored.spawnProtection, sim.spawnProtection);
  assert.deepStrictEqual(restored.invuln, sim.invuln);
  frame.spawnProtection[0] = 0;
  assert.equal(restored.spawnProtection[0], 29, 'replay owns its state');
  const expired = sim.snapshotReplay(null); expired.spawnProtection.fill(0); expired.invuln.fill(0);
  restored.restoreReplay(expired); assert.equal(restored.spawnProtection[0], 0);
  delete expired.spawnProtection; restored.restoreReplay(expired);
  assert.deepStrictEqual(restored.spawnProtection, [0, 0, 0, 0], 'old replays receive no invented protection');
}
const training = new Q.Sim(19); training.reset(level);
assert(training.spawnProtection.every((n) => n === 0));
assert(!('spawnProtection' in training.snapshotReplay(null)), 'training replay hashes remain unchanged');
{
  const sim = scene();
  const cell = 5 * Q.W + 5;
  sim.crate[cell] = 1; sim.crateType[cell] = 1;
  const before = sim.blastCap[0];
  const info = sim.step([[4, 1, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]]);
  assert(info.placed[0] && sim.blastCap[0] > before, 'protection permits active bubbles and ordinary upgrade pickups');
  sim.fuse[5 * Q.W + 4] = 20; sim.owner[5 * Q.W + 4] = 1;
  const x = sim.pos[1];
  for (let n = 0; n < 3; n++) sim.frameStep(0, Q.MOVE_LEFT, .1);
  assert(sim.pos[1] >= 5 && sim.pos[1] <= x, 'harm immunity does not make ordinary bubbles passable');
}
assert.equal(meta.source, 'object/magic/magic0139.img');
assert.equal(meta.effect, 'effect/flash2.eff');
assert.deepStrictEqual([meta.w, meta.h, meta.ox, meta.oy, meta.frames, meta.cycleMs, meta.protectionMs],
  [100, 113, -48, -90, 15, 1500, 3000]);
console.log('native spawn/respawn protection, harm scope, expiry and replay checks passed');
