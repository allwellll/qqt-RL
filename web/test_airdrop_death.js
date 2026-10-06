'use strict';
const assert = require('assert');
const Q = require('./sim.js');
const visual = require('./visual_renderer.js');
const level = require('./assets/maps/levels.json').find(x => x.qqt_id === 806);
const idle = [4, 0, 0, 1];
assert.equal(Q.AIRDROP_LANDING_TICKS, 3, 'airdrop simulation still lands after 3 ticks');
assert.equal(visual.AIRDROP_ANIMATION_TICKS, 5, 'airdrop rendering may animate for 5 ticks');
function scene(options = {}) {
  const sim = new Q.Sim(19);
  sim.reset(level, { nativeItems: true, nativeTrap: true, ...options });
  for (const key of ['wall', 'brick', 'cover', 'pushable', 'crate', 'fuse', 'blastLinger']) sim[key].fill(0);
  sim.pos.set([6.5, 6.5, 11.5, 13.5]);
  sim.itemSlots[0] = [{ item: Q.ITEM_BANANA, count: 2 }, { item: Q.ITEM_SLOW_GLUE, count: 5 }];
  sim._syncHeldItem(0);
  return sim;
}
{
  const sim = scene(); sim._manualBird = true;
  const cell = 5 * Q.W + 5;
  sim.wall[cell] = sim.brick[cell] = sim.pushable[cell] = sim.crate[cell] = sim.fieldItem[cell] = sim.fuse[cell] = 0;
  sim.airdropFalls = [{ type: 4, isSuper: false, cell, tick: sim.t, x: 5.5, row: 3 }];
  for (let elapsed = 1; elapsed <= 3; elapsed++) {
    sim.t++; sim._nativeAirdropStep();
    assert.equal(!!sim.crate[cell], elapsed === 3, 'pickup state becomes available on original tick 3, not visual tick 5');
    assert.equal(sim.airdropFalls.length, elapsed < 3 ? 1 : 0);
  }
}
function tick(sim) { return sim.step(sim.alive.map(() => idle)); }
function quantities(sim) {
  const totals = {};
  for (let i = 0; i < Q.N; i++) if (sim.crate[i]) {
    const type = sim.crateType[i];
    totals[type] = (totals[type] || 0) + (sim.crateCount[i] || 1);
  }
  return totals;
}
// Real blast -> trapped -> rescue keeps possessions; only actual death releases them.
{
  const sim = scene({ teams: [0, 0, 1] });
  sim.pos.set([6.5, 6.5, 11.5, 11.5, 11.5, 13.5]);
  sim.invuln.fill(0);
  const cell = 6 * Q.W + 6;
  sim.fuse[cell] = 1; sim.bombBlast[cell] = 1; sim.owner[cell] = 2;
  tick(sim);
  assert(sim.trapped[0] > 0 && sim.alive[0]);
  assert.equal(sim.itemSlots[0][1].count, 5);
  assert.deepEqual(quantities(sim), {});
  sim.pos[2] = sim.pos[0]; sim.pos[3] = sim.pos[1];
  tick(sim);
  assert.equal(sim.trapped[0], 0);
  assert.equal(sim.itemSlots[0][1].count, 5);
}
for (const cause of ['timeout', 'enemy', 'direct-blast']) {
  const sim = scene({ nativeTrap: cause !== 'direct-blast' });
  sim.bombsCap[0] += 2; sim.blastCap[0] += 3;
  sim.spdG[0] += 2 * (sim.speedStep || Q.CFG.growthSpeedStep);
  if (cause === 'direct-blast') {
    sim.invuln[0] = 0;
    const cell = 6 * Q.W + 6;
    sim.fuse[cell] = 1; sim.bombBlast[cell] = 1; sim.owner[cell] = 1;
  } else {
    sim.trapped[0] = cause === 'timeout' ? 1 : 20;
    if (cause === 'enemy') { sim.pos[2] = sim.pos[0]; sim.pos[3] = sim.pos[1]; }
  }
  tick(sim);
  assert(!sim.alive[0] && sim.lastDied[0], cause);
  assert.deepEqual(sim.itemSlots[0], []);
  assert.equal(sim.heldItem[0], Q.ITEM_NONE);
  assert.equal(sim.bombsCap[0], sim.loBombs[0]);
  assert.equal(sim.blastCap[0], sim.loBlast[0]);
  assert.equal(sim.spdG[0], sim.loSpeed[0]);
  assert.deepEqual(quantities(sim), { 0: 2, 1: 3, 2: 2, 3: 2, 4: 5 });
  const copy = JSON.parse(JSON.stringify(sim.snapshotReplay()));
  sim._killPlayer(0);
  assert.deepEqual(sim.snapshotReplay(), copy, 'duplicate death is idempotent');
  for (let cell = 0; cell < Q.N; cell++) if (sim.crate[cell]) {
    assert(!sim.wall[cell] && !sim.brick[cell] && !sim.fuse[cell] && !sim.blastLinger[cell]);
    sim._collectCrate(1, cell);
  }
  assert.equal(sim.itemSlots[1].find(x => x.item === Q.ITEM_SLOW_GLUE).count, 5, 'glue never multiplies by three');
  assert.equal(sim.itemSlots[1].find(x => x.item === Q.ITEM_BANANA).count, 2);
  for (let i = 0; i < sim.bunRespawnTicks + 1; i++) tick(sim);
  assert(sim.alive[0] && !sim.itemSlots[0].length && sim.heldItem[0] === 0, 'respawn does not restore dropped inventory');
}
// Saturated map and simultaneous deaths preserve every bundle without overwriting objects.
{
  const sim = scene(); sim.wall.fill(1);
  sim.itemSlots[1] = [{ item: Q.ITEM_SLOW_GLUE, count: 4 }];
  sim.trapped.fill(1); tick(sim);
  assert.equal(sim._pendingDeathDrops.length, 3);
  const replay = sim.snapshotReplay();
  const restored = scene(); restored.restoreReplay(replay);
  assert.deepEqual(restored._pendingDeathDrops, sim._pendingDeathDrops);
  sim.wall[80] = 0; tick(sim);
  assert.equal(sim._pendingDeathDrops.length, 2);
  assert.equal(sim.crateCount[80], 2);
  sim._collectCrate(1, 80); tick(sim);
  assert.equal(sim.crateCount[80], 5);
  sim._collectCrate(1, 80); tick(sim);
  assert.equal(sim.crateCount[80], 4);
}
// A large payload lands beneath the visible bird, never flushes invisibly, and preserves counts.
{
  const sim = scene(); sim.t = 269;
  sim.graveyard = Array.from({ length: 60 }, (_, i) => ({ type: i % 5, isSuper: false, count: i % 5 === 4 ? 2 : 1 }));
  const initial = sim.graveyard.reduce((n, x) => n + x.count, 0);
  let flights = 0, restored;
  for (let i = 0; i < 34; i++) {
    tick(sim);
    for (const drop of sim.airdropFalls) {
      assert.equal(drop.cell % Q.W, Math.max(0, Math.min(14, Math.floor(drop.x))));
      assert(!sim.crate[drop.cell], 'airborne item cannot be picked up');
    }
    if (sim.birdFlight()) flights++;
    if (sim.airdropFalls.length && !restored) {
      restored = scene(); restored.restoreReplay(JSON.parse(JSON.stringify(sim.snapshotReplay())));
      assert.deepEqual(restored.airdropFalls, sim.airdropFalls);
      assert.deepEqual(restored._airdropQueue, sim._airdropQueue);
      assert.deepEqual(restored.birdFlight(), sim.birdFlight());
      // No RNG is used for landing an already dispatched item.
      restored._manualBird = true;
      for (let n = 0; n < Q.AIRDROP_LANDING_TICKS; n++) { restored.t++; restored._nativeAirdropStep(); }
      assert(restored.crate.some(x => x > 0));
    }
  }
  assert.equal(flights, 30);
  assert(!sim.airdropFalls.length && !sim._airdropQueue.length);
  assert.equal(Object.values(quantities(sim)).reduce((a,b) => a+b, 0) + sim.graveyard.reduce((a,x) => a+x.count, 0), initial);
}
{
  const sim = scene(); sim.wall.fill(1); sim.t = 269;
  sim.graveyard = [{ type: 4, isSuper: false, count: 5 }];
  for (let i = 0; i < 34; i++) tick(sim);
  assert.deepEqual(sim.graveyard, [{ type: 4, isSuper: false, count: 5 }]);
  assert.equal(sim.crate.reduce((a,b) => a+b, 0), 0);
}
// Destroyed held-item bundles enter the bird queue with their exact quantity.
{
  const sim = scene();
  const cell = 5 * Q.W + 5;
  sim.crate[cell] = 1; sim.crateType[cell] = 4; sim.crateCount[cell] = 5;
  sim.fuse[cell] = 1; sim.owner[cell] = 1; sim.bombBlast[cell] = 1;
  tick(sim);
  assert.deepEqual(sim.graveyard, [{ type: 4, isSuper: false, count: 5 }]);
  assert.equal(sim.crateCount[cell], 0);
}
{
  const sim = scene();
  const legacy = sim.snapshotReplay();
  for (const key of ['crateCount', 'airdropTotal', 'airdropDropped', 'airdropQueue', 'airdropFalls', 'pendingDeathDrops']) delete legacy[key];
  sim.airdropFalls = [{ type: 4, cell: 80, tick: 1 }];
  sim._pendingDeathDrops = [{ type: 3, count: 2 }];
  sim.restoreReplay(legacy);
  assert(!sim.airdropFalls.length && !sim._pendingDeathDrops.length && !sim._airdropQueue.length);
  assert(sim.crateCount.every(x => x === 0), 'old snapshots clear new state instead of leaking the previous match');
}
console.log('Native bird flight, landing, full inventory death drops and quantity conservation: passed');
