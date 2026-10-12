'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const QQT = require('../web/sim.js');
const authority = require('../scripts/eval_web_hunter.js');
const probe = require('../scripts/eval_web_native_bun_probe.js');

test('native CLI does not require a replaced spawn manifest', () => {
  const options = probe.parseOptions(['--model','model.json','--difficulty','hard',
    '--seed','392010731','--checkpoint-sha','p','--out','out.json']);
  assert.equal(options.modelPath,'model.json');
  assert.equal(options.checkpointSha,'p');
  assert.throws(() => probe.parseOptions(['--model','model.json']), /missing/);
  assert.throws(() => probe.parseOptions(['--model','m','--difficulty','hard',
    '--seed','1','--out','o']), /seed/);
});

test('native probe preserves bricks/items/traps and exposes its separate scope', () => {
  const sim = probe.setup(authority.loadLevel(), 392010731, 'native_1v1');
  assert.ok(sim.nativeItems && sim.nativeTrap);
  assert.equal(sim.nPlayers, 2);
  assert.equal(sim.maxSteps, 300);
  assert.ok(Array.from(sim.brick).some(Boolean));
  assert.deepEqual(sim.bunInitial, [1, 1]);
});

test('rescue stress state has an explicit live friendly player', () => {
  const sim = probe.setup(authority.loadLevel(), 392010731, 'rescue_2v1');
  assert.equal(sim.nPlayers, 3);
  assert.deepEqual(sim.team, [0, 1, 0]);
  assert.ok(sim.alive[2] && sim.trapped[2] === sim.trapTicks);
  const cell = Math.floor(sim.pos[4])*QQT.W+Math.floor(sim.pos[5]);
  assert.ok(!sim.wall[cell] && !sim.brick[cell]);
});

test('observable alias audit is exact and does not mutate the original state', () => {
  const sim = probe.setup(authority.loadLevel(), 392010731, 'rescue_2v1');
  const result = probe.aliasAudit(sim);
  assert.ok(result.teammate_location_unobserved);
  assert.ok(result.trap_state_unobserved);
  assert.equal(sim.trapped[2], sim.trapTicks);
});

test('friendly harm requires a new trap with actual hero damage source', () => {
  assert.equal(probe.newHeroTrap([0, 0, 0], [0, 0, 60],
    {physicalDamageSource: [[], [], [false, true, false]], causalDamageSource: [[], [], []]}, 2), 0);
  assert.equal(probe.newHeroTrap([0, 0, 0], [0, 0, 60],
    {physicalDamageSource: [[], [], [true, false, false]], causalDamageSource: [[], [], []]}, 2), 1);
  assert.equal(probe.newHeroTrap([0, 0, 60], [0, 0, 59],
    {physicalDamageSource: [[], [], [true, false, false]], causalDamageSource: [[], [], []]}, 2), 0);
});
