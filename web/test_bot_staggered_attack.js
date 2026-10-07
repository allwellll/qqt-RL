'use strict';
const assert = require('assert');
const QQT = require('./sim');
const Coop = require('./bun_coop_hunter_bot');
const { scene, runScene, runMap, trace, SEEDS } = require('../scripts/eval_staggered_attack');
const mode = process.argv[2] || 'all';
if (mode === 'all' || mode === 'stagger') {
  const linked = scene(SEEDS[0], 3, false, 3, 6, true);
  linked.pos.set([6.5,12.5,6.5,10.5]);
  for (const [y,x,fuse] of [[6,8,10],[2,8,24]]) {
    const cell = y * QQT.W + x; linked.fuse[cell] = fuse; linked.owner[cell] = 1; linked.bombBlast[cell] = 3;
  }
  const chainBot = new Coop.BunCoopHunterBot({ difficulty: 'hard' });
  const chain = trace(linked, [null, chainBot], 1, (s,bs,decisions) => {
    const action = chainBot.act(s,1); decisions.set(1,chainBot.lastDecision); return [[4,0,0,0],action];
  }, 35);
  const connector = chain.bubbles.find(b => b.reason === 'bomb_stagger_chain');
  assert(connector && connector.triggerSource.kind === 'chain');
  assert.equal(connector.actualExplosionTick, 10);
  assert(connector.damageSafetyMarginTicks >= 3, 'RED: full-body escape retains three ticks even at half-contact positions');
  const combined = new Uint8Array(linked.fuse.length);
  for (const bubble of chain.bubbles.filter(b => b.actualExplosionTick === 10)) for (const cell of bubble.covered) combined[cell] = 1;
  let lastHit = 0;
  for (const point of connector.retreatPath.filter(p => p.tick <= 10)) {
    linked.pos[2] = point.y; linked.pos[3] = point.x;
    if (linked._isHitByExplosion(1, combined, null, null, false)) lastHit = point.tick;
  }
  assert.equal(connector.safetyMarginTicks, 0, 'retain conservative geometry evidence');
  assert.equal(connector.damageSafetyMarginTicks, 10 - lastHit);
  assert.equal(connector.damageSafetyMarginTicks, 7, 'actual native full-contact immunity is checked by Sim');
  assert(chain.bubbles.some(b => b.actualExplosionTick === 24), 'connector preserves another later explosion phase');
  assert.equal(chain.stats.selfTraps + chain.stats.friendlyTraps, 0);
  const result = runScene({ capacity: 2, foeDistance: 4, carrier: true });
  assert(result.stats.multiBubblePairs > 0, 'RED: safe coexisting bubbles must actually explode on different ticks');
  assert(result.stats.chainContinuations > 0, 'RED: approach a late ray extension, place and withdraw into a later explosion phase');
  assert(result.bubbles.some(b => b.reason === 'bomb_stagger_extension'), 'must deliberately schedule useful extension pressure');
  assert(result.stats.enemyTraps > 0, 'continuation must actually affect the enemy');
  assert.equal(result.stats.selfTraps + result.stats.friendlyTraps + result.stats.selfConfinedTicks, 0);
  for (const b of result.bubbles.filter(b => b.reason === 'bomb_stagger_extension')) {
    assert.equal(b.expectedExplosionTick, b.actualExplosionTick);
    assert(b.safetyMarginTicks >= 3, 'actual retreat must preserve at least the existing three-tick slack');
  }
}
if (mode === 'all' || mode === 'guards') {
  const bot = () => new Coop.BunCoopHunterBot({ difficulty: 'hard', seed: SEEDS[0] });
  for (const hazard of ['corner', 'banana', 'glue', 'early-chain', 'stationary-ally', 'rescue', 'carrier']) {
    const sim = scene(SEEDS[0], 3, true), b = bot(), W = QQT.W;
    sim.pos[2] = 6.5; sim.pos[3] = 12.5; sim.pos[0] = 6.5; sim.pos[1] = 14.5;
    const anchor = 6 * W + 8; sim.fuse[anchor] = 9; sim.owner[anchor] = 1; sim.bombBlast[anchor] = 3;
    if (hazard === 'corner' || hazard === 'glue') {
      sim.wall.fill(1); for (const c of [6 * W + 12, 6 * W + 13, 6 * W + 14]) sim.wall[c] = 0;
      if (hazard === 'glue') sim.fieldItem[6 * W + 13] = 2;
    }
    if (hazard === 'banana') { sim.movementStatus[1] = QQT.MOVE_STATUS_SLIDE; sim.slideDir[1] = QQT.MOVE_RIGHT; }
    if (hazard === 'early-chain') { const c = 6 * W + 10; sim.fuse[c] = 1; sim.owner[c] = 0; sim.bombBlast[c] = 5; }
    if (hazard === 'stationary-ally') { sim.pos[4] = 6.5; sim.pos[5] = 13.5; }
    if (hazard === 'rescue') { sim.pos[4] = 7.5; sim.pos[5] = 12.5; sim.trapped[2] = 40; }
    if (hazard === 'carrier') sim.bunCarried[1] = 0;
    const d = b.analyzeSim(sim, 1);
    assert.equal(d.action[1], 0, `${hazard}: unsafe or higher-priority duties must veto attack`);
    assert.notEqual(d.mode, 'STAGGER', `${hazard}: do not commit an unsafe approach`);
  }
  assert.equal(bot().cfg.maxLiveBombs, 4, 'existing live-bubble ceiling stays unchanged');
}
if (mode === 'all') {
  const s = scene(SEEDS[0], 3, true), spacingBot = new Coop.BunCoopHunterBot({ difficulty: 'hard' });
  s.wall.fill(1);
  for (const [y, x] of [[2,11],[3,11],[4,11],[5,9],[5,10],[5,11],[6,10],[6,11],[6,12],[6,13],[6,14],[7,11],[7,12],[8,11]]) s.wall[y * QQT.W + x] = 0;
  s.pos.set([6.5,14.5,5.5,9.5,5.5,10.5]);
  const anchor = 2 * QQT.W + 11; s.fuse[anchor] = 18; s.owner[anchor] = 1; s.bombBlast[anchor] = 3;
  for (let n = 0; n < 3; n++) {
    s.step([[4,0,0,0], spacingBot.act(s,1), [4,0,0,0]]);
    assert(Math.hypot(s.pos[2]-s.pos[4], s.pos[3]-s.pos[5]) >= 0.75, 'RED: stagger approach preserves teammate spacing');
  }
  assert(runMap(SEEDS[0], 3, 0, Coop).stats.enemyTraps >= 1,
    'RED: a timing plan must preserve the existing immediate attack opportunity');
  const a = runScene({ seed: SEEDS[1], direction: 1, team: true });
  const b = runScene({ seed: SEEDS[1], direction: 1, team: true });
  assert.deepEqual(a, b, 'same seed produces identical decisions, movements, placement and actual explosion trace');
}
console.log(`Staggered attack ${mode}: actual separated explosions, late extension, retreat, hazard/team vetoes and deterministic trace passed`);
