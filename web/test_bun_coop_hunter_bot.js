'use strict';
const assert = require('assert');
const QQT = require('./sim.js');
const Hunter = require('./bun_hunter_bot.js');
const Coop = require('./bun_coop_hunter_bot.js');
const Bots = require('./bot_contract.js');
const W = QQT.W;

function scene(teams = [0, 1, 1], coords = [[10.5, 1.5], [6.5, 8.5], [6.5, 12.5]]) {
  const sim = new QQT.Sim(11);
  sim.reset('open', { nativeItems: true, nativeTrap: true, teams });
  for (const name of ['wall', 'brick', 'crate', 'fuse', 'blastLinger']) sim[name].fill(0);
  sim.isBun = true;
  sim.bunBases = [[1, 4], [1, 8]];
  sim.bunStored = [[1, 0], [0, 1]];
  sim.bunScore = [0, 0];
  sim.bunTarget = 1;
  sim.bunInitial = [Infinity, Infinity];
  coords.forEach(([y, x], p) => { sim.pos[p * 2] = y; sim.pos[p * 2 + 1] = x; sim.invuln[p] = 0; });
  return sim;
}
const bot = () => new Coop.BunCoopHunterBot({ difficulty: 'hard', seed: 7 });
const decide = (sim, pid) => bot().analyze(Hunter.hunterStateFromSim(sim), pid);

{
  const sim = scene();
  assert.equal(decide(sim, 1).role, 'ATTACKER');
  assert.equal(decide(sim, 2).mode, 'GUARD');
  sim.alive[1] = false;
  assert.equal(decide(sim, 2).role, 'SOLO', 'support takes over when attacker is dead');
}
{
  const sim = scene([0, 1, 1], [[11.5, 1.5], [5.5, 9.5], [5.5, 5.5]]);
  sim.trapped[1] = 60;
  const rescue = bot();
  assert.equal(rescue.analyzeSim(sim, 2).mode, 'RESCUE');
  for (let t = 0; t < 40 && sim.trapped[1]; t++) {
    const action = rescue.act(sim, 2);
    assert.equal(action[1], 0, 'rescue must not be delayed by bombing');
    sim.step([[4, 0, 0, 0], [4, 0, 0, 0], [...action, 0]]);
  }
  assert(sim.alive[1] && !sim.trapped[1], 'actual contact rescues the attacker');
}
{
  const sim = scene();
  sim.bunCarried[1] = 0;
  assert.equal(decide(sim, 1).mode, 'DELIVER');
  assert.equal(decide(sim, 1).action[1], 0);
  assert.equal(decide(sim, 2).mode, 'ESCORT');
  sim.bunCarried[1] = -1;
  sim.bunCarried[2] = 0;
  assert.equal(decide(sim, 1).mode, 'ESCORT', 'either role escorts a carrier');
}
{
  const sim = scene([0, 1, 0, 1], [[11.5, 1.5], [6.5, 8.5], [1.5, 10.5], [6.5, 12.5]]);
  sim.bunCarried[2] = 1;
  const b = bot(), state = Hunter.hunterStateFromSim(sim);
  assert.strictEqual(b.foeOf(state, 1), state.players[2], 'select the thief among multiple enemies');
  assert.equal(b.analyze(state, 3).mode, 'INTERCEPT');
}
{
  const sim = scene([0, 1, 1], [[5.5, 4.5], [5.5, 11.5], [5.5, 5.5]]);
  sim.trapped[0] = 60;
  assert.equal(decide(sim, 2).mode, 'POP', 'nearer support finishes a trapped enemy');
  assert.equal(decide(sim, 1).mode, 'RAID', 'attacker uses the suppression window to steal');
}
{
  const sim = scene([0, 1, 1], [[3.5, 9.5], [9.5, 8.5], [5.5, 12.5]]);
  assert.equal(decide(sim, 2).mode, 'DEFEND', 'support responds to an enemy at the home base');
}
{
  const sim = scene([0, 1, 1], [[11.5, 1.5], [5.5, 9.5], [5.5, 5.5]]);
  sim.trapped[1] = 60;
  for (const [r, c] of [[4, 9], [6, 9], [5, 8], [5, 10]]) sim.brick[r * W + c] = 1;
  assert.notEqual(decide(sim, 2).mode, 'RESCUE', 'an unreachable rescue cannot replace a useful task');
}
{
  const sim = scene([0, 1], [[11.5, 1.5], [5.5, 6.5]]);
  const cell = 5 * W + 5;
  sim.fuse[cell] = 8; sim.owner[cell] = 0; sim.bombBlast[cell] = 3;
  const b = bot();
  for (let t = 0; t < 14; t++) sim.step([[4, 0, 0, 0], [...b.act(sim, 1), 0]]);
  assert(sim.alive[1] && !sim.trapped[1], 'single cooperative hunter actually dodges the explosion');
}
{
  const sim = scene([0, 1], [[5.5, 6.5], [5.5, 5.5]]);
  for (const [r, c] of [[4, 7], [6, 7], [5, 8], [6, 6], [6, 5]]) sim.wall[r * W + c] = 1;
  assert.equal(decide(sim, 1).action[1], 1, 'single bot still attacks a trapped escape route');
}
{
  const sim = scene([0, 1, 1], [[5.5, 6.5], [5.5, 5.5], [3.5, 5.5]]);
  for (const [r, c] of [[3, 4], [3, 6], [2, 5], [4, 4], [4, 6], [4, 7], [6, 7], [5, 8], [6, 6], [6, 5]]) sim.wall[r * W + c] = 1;
  assert.equal(decide(sim, 1).action[1], 0, 'cannot trap a teammate with friendly fire');
}
{
  const sim = scene([0, 1, 1], [[5.5, 6.5], [5.5, 5.5], [5.5, 4.5]]);
  sim.bunCarried[2] = 0;
  assert.equal(decide(sim, 1).action[1], 0, 'protect the slower carrier from new blast lines');
}
{
  const registry = Bots.createDefaultRegistry({ coopHunter: Coop });
  assert.equal(registry.describe('bun.coop_hunter', {}).config.difficulty, 'hard');
  assert.throws(() => registry.create('bun.coop_hunter', { difficulty: 'bad' }), /invalid config/);
  const adapter = registry.create('bun.coop_hunter', {});
  adapter.reset({ seed: 8 });
  const sim = scene();
  const obs = { metadata: { sim }, legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2] };
  Bots.validateAction(adapter.act(obs, 1));
}
{
  const sim = scene([0, 1, 1], [[11.5, 1.5], [5.5, 8.5], [5.5, 11.5]]);
  sim.bunCarried[1] = 0;
  const bots = [null, bot(), bot()];
  let escorted = 0;
  for (let t = 0; t < 160 && sim.bunCarried[1] >= 0; t++) {
    const actions = [[4, 0, 0, 0]];
    for (const p of [1, 2]) {
      const a = bots[p].act(sim, p);
      actions[p] = [...a, 0];
      if (bots[p].lastDecision.mode === 'ESCORT') escorted++;
    }
    sim.step(actions);
    assert(sim.alive[1] && !sim.trapped[1], 'carrier remains safe during actual escorted delivery');
  }
  assert(escorted > 0, 'escort persisted across ticks');
  assert.equal(sim.bunCarried[1], -1, 'carrier actually deposits the bun');
  assert.equal(sim.bunScore[1], 1);
}
{
  const make = () => scene([0, 1, 1], [[5.5, 6.5], [5.5, 5.5], [3.5, 8.5]]);
  const a = make(), b = make();
  const botsA = [bot(), bot(), bot()], botsB = [bot(), bot(), bot()];
  for (let t = 0; t < 80; t++) {
    const actionsA = botsA.map((item, pid) => [...item.act(a, pid), 0]);
    const actionsB = botsB.map((item, pid) => [...item.act(b, pid), 0]);
    assert.deepStrictEqual(actionsA, actionsB, 'joint decisions are deterministic');
    a.step(actionsA); b.step(actionsB);
    assert.deepStrictEqual(Array.from(a.pos), Array.from(b.pos));
    assert.deepStrictEqual(a.trapped, b.trapped);
  }
}
{
  const sim = scene([0, 1, 1], [[5.5, 6.5], [5.5, 5.5], [3.5, 9.5]]);
  for (const [r, c] of [[4, 7], [6, 7], [5, 8], [6, 6], [6, 5]]) sim.wall[r * W + c] = 1;
  const first = bot(), second = bot();
  assert.equal(first.act(sim, 1)[1], 1, 'first teammate commits an attack');
  assert.equal(sim.fuse[5 * W + 5], 0, 'committed attack has not yet entered physics');
  const original = second.analyze.bind(second);
  let observed;
  second.analyze = (state, pid) => { observed = state.bombs.slice(); return original(state, pid); };
  second.act(sim, 2);
  assert(observed.some((b) => b.owner === 1 && b.cell === 5 * W + 5), 'teammate sees the pending attack');
  sim.t++;
  second.act(sim, 2);
  assert.equal(observed.length, 0, 'commitments expire at the next tick');
}
{
  const sim = scene([0, 1, 1], [[5.5, 8.5], [10.5, 12.5], [3.5, 8.5]]);
  sim.bunStored = [[0, 0], [0, 0]];
  sim.spdG.fill(1); sim.blastCap.fill(1);
  for (let row = 1; row < 7; row++) for (const col of [7, 9]) sim.wall[row * W + col] = 1;
  sim.wall[3 * W + 9] = 0;
  const defender = bot(), first = defender.analyzeSim(sim, 2);
  assert.equal(first.mode, 'DEFEND');
  assert.equal(first.reason, 'bomb_block', 'defender blocks the only base approach before a direct shot');
  const info = sim.step([[QQT.MOVE_UP, 0, 0, 0], [4, 0, 0, 0], [...first.action, 0, 0]]);
  assert(info.placed[2] && sim.fuse[3 * W + 8] > 0, 'blocking bubble enters actual physics');
  for (let t = 0; t < 12; t++) {
    sim.step([[QQT.MOVE_UP, 0, 0, 0], [4, 0, 0, 0], [...defender.act(sim, 2), 0]]);
    assert(sim.pos[0] >= 4 && sim.alive[2] && !sim.trapped[2], 'bubble stops the enemy while defender escapes');
  }
}
{
  const sim = scene([0, 1, 1], [[7.5, 7.5], [5.5, 5.5], [10.5, 12.5]]);
  sim.bunStored = [[0, 0], [0, 0]];
  sim.spdG.fill(1); sim.blastCap.fill(3); sim.bombsCap.fill(5);
  const attacker = bot();
  let anchor, connector, connectedTick, chainExploded = false;
  for (let tick = 0; tick < 34; tick++) {
    const decision = attacker.analyzeSim(sim, 1);
    const cell = sim.centerCell(1)[0] * W + sim.centerCell(1)[1];
    if (tick === 0) {
      assert.equal(decision.reason, 'bomb_reserve', 'first bubble prepares a corner connection');
      assert.equal(decision.action[1], 1);
      anchor = cell;
    }
    if (decision.reason === 'bomb_chain') { connector = cell; connectedTick = tick; }
    const info = sim.step([[4, 0, 0, 0], [...decision.action, 0, 0], [4, 0, 0, 0]]);
    assert(sim.alive[1] && !sim.trapped[1], 'attacker survives its real chain attack');
    if (info.triggered[anchor] && connector != null && info.triggered[connector]) {
      chainExploded = true;
      assert(tick < connectedTick + 30, 'connector detonates early through the reserve bubble');
      assert(sim.trapped[0] > 0, 'connected ray actually traps the enemy outside the first ray');
    }
  }
  assert(connector != null && chainExploded, 'reserve plan leads to a second placement and real chain explosion');
}
for (const difficulty of ['easy', 'normal', 'hard']) {
  const sim = scene([0, 1, 1], [[11.5, 1.5], [5.5, 9.5], [5.5, 5.5]]);
  sim.trapped[1] = 60;
  const rescue = new Coop.BunCoopHunterBot({ difficulty, seed: 3, overrides: { mistakeRate: 0 } });
  for (let t = 0; t < 40 && sim.trapped[1]; t++) sim.step([[4, 0, 0, 0], [4, 0, 0, 0], [...rescue.act(sim, 2), 0]]);
  assert(sim.alive[1] && !sim.trapped[1], `${difficulty} completes actual rescue`);
}
{
  const sim = scene([0, 1, 1], [[7.5, 7.5], [5.5, 5.5], [10.5, 12.5]]);
  sim.bunStored = [[0, 0], [0, 0]];
  sim.spdG.fill(1); sim.blastCap.fill(3); sim.bombsCap.fill(5);
  const attacker = bot(), decision = attacker.analyzeSim(sim, 1);
  assert.equal(decision.reason, 'bomb_reserve');
  sim.step([[4, 0, 0, 0], [...decision.action, 0, 0], [4, 0, 0, 0]]);
  sim.pos[0] = 11.5; sim.pos[1] = 1.5;
  attacker.analyzeSim(sim, 1);
  assert.equal(attacker.chainPlan, null, 'cancel the reserve connection after the enemy leaves its target ray');
}
{
  const sim = scene([0, 1, 1], [[10.5, 1.5], [6.5, 8.5], [6.5, 9.5]]);
  sim.bunStored = [[0, 0], [0, 0]]; sim.spdG.fill(1);
  const attacker = bot(), defender = bot();
  let guardTicks = 0;
  for (let tick = 0; tick < 30; tick++) {
    const first = attacker.act(sim, 1), second = defender.act(sim, 2);
    if (defender.lastDecision.mode === 'GUARD') guardTicks++;
    sim.step([[4, 0, 0, 0], [...first, 0], [...second, 0]]);
  }
  const separation = Math.abs(sim.pos[2] - sim.pos[4]) + Math.abs(sim.pos[3] - sim.pos[5]);
  assert(guardTicks >= 20 && separation >= 4, 'actual attack and defense routes split initially adjacent teammates');
  assert(sim.pos[4] < 6 && sim.alive[2] && !sim.trapped[2], 'defender stays near the base approach');
}
{
  const sim = scene([0, 1, 1], [[8.8, 7.5], [5.5, 5.5], [10.5, 12.5]]);
  sim.bunStored = [[0, 0], [0, 0]];
  sim.spdG.fill(1); sim.blastCap.fill(3); sim.bombsCap.fill(5);
  sim.wall[4 * W + 5] = 1; sim.wall[5 * W + 4] = 1;
  const attacker = bot();
  attacker.observeEnemyTrends(Hunter.hunterStateFromSim(sim), 1);
  sim.step([[QQT.MOVE_UP, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]]);
  let connected = false, exploded = false, waited = 0;
  for (let tick = 0; tick < 34; tick++) {
    const decision = attacker.analyzeSim(sim, 1);
    if (tick === 0) {
      assert.equal(decision.mode, 'HUNT');
      assert.equal(decision.reason, 'bomb_reserve', 'observed motion enables a corner reserve before the enemy enters its ray');
      const state = Hunter.hunterStateFromSim(sim), g = attacker.geometry(state);
      const ray = attacker.blastCells(state, g, attacker.chainPlan.cell, 3);
      assert(!ray.has(state.players[0].cell) &&
        ray.has(attacker.predictedEnemyCell(state, g, 0)), 'the reserve aims at projected movement, not the current cell');
    }
    if (decision.mode === 'CHAIN' && decision.action[0] === 4 && !decision.action[1]) waited++;
    if (decision.reason === 'bomb_chain') {
      assert(sim.fuse[5 * W + 5] <= 12, 'late connector waits until the reserve is near detonation');
      connected = true;
    }
    const info = sim.step([[tick < 3 ? QQT.MOVE_UP : 4, 0, 0, 0], [...decision.action, 0, 0], [4, 0, 0, 0]]);
    assert(sim.alive[1] && !sim.trapped[1], 'late corner attack keeps its own escape route');
    if (info.triggered[5 * W + 5] && info.triggered[7 * W + 5]) {
      assert(sim.trapped[0] > 0, 'motion-predicted late connection actually traps the arriving enemy');
      exploded = true;
    }
  }
  assert(connected && exploded && waited >= 5, 'reserve, wait, connect and explosion all occur in real simulation');
}
{
  const sim = scene([0, 1, 1], [[7.5, 7.5], [5.5, 5.5], [10.5, 12.5]]);
  sim.spdG.fill(1);
  const hunter = bot();
  hunter.observeEnemyTrends(Hunter.hunterStateFromSim(sim), 1);
  sim.step([[QQT.MOVE_RIGHT, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]]);
  hunter.observeEnemyTrends(Hunter.hunterStateFromSim(sim), 1);
  sim.wall[7 * W + 8] = 1;
  const state = Hunter.hunterStateFromSim(sim), g = hunter.geometry(state);
  assert.equal(hunter.predictedEnemyCell(state, g, 0), state.players[0].cell, 'movement projection stops at the wall');
  sim.t++; sim.pos[0] = 1.5; sim.pos[1] = 12.5;
  hunter.observeEnemyTrends(Hunter.hunterStateFromSim(sim), 1);
  assert(!hunter.enemyTrends.get(0).confidence, 'respawn-size jumps do not become a confident movement trend');
  hunter.reset();
  assert.equal(hunter.enemyTrends.size, 0, 'new match clears learned movement trends');
}
{
  const sim = scene([0, 1, 1], [[11.5, 1.5], [10.5, 4.5], [6.5, 10.5]]);
  sim.bunStored = [[0, 0], [0, 0]];
  sim.bombsCap[2] = 1;
  sim.brick[6 * W + 11] = 1;
  sim.crateRate = 1;
  sim._rollCrateType = () => ({ type: 0, isSuper: false });
  const support = bot(), capacity = sim.bombsCap[2];
  let farmed = false, collected = false, placed = false;
  for (let t = 0; t < 150 && sim.bombsCap[2] === capacity; t++) {
    const action = support.act(sim, 2);
    farmed ||= support.lastDecision.mode === 'FARM';
    collected ||= support.lastDecision.mode === 'COLLECT';
    const info = sim.step([[4, 0, 0, 0], [4, 0, 0, 0], action]);
    placed ||= !!info.placed[2];
    assert(sim.alive[2] && !sim.trapped[2], 'resource farming preserves the defender escape');
  }
  assert(farmed && collected && placed && !sim.brick[6 * W + 11] && sim.bombsCap[2] > capacity,
    'idle defender actually bombs a brick and collects its revealed upgrade');
  sim.pos[0] = 3.5; sim.pos[1] = 9.5;
  assert.equal(support.analyzeSim(sim, 2).mode, 'DEFEND', 'base threat immediately preempts farming');
}
{
  const sim = scene([0, 1, 1], [[11.5, 1.5], [7.5, 8.5], [5.2, 8.5]]);
  sim.bunCarried[1] = 0;
  sim.spdG.fill(1);
  sim.itemSlots[2] = [{ item: 2, count: 1 }, { item: 1, count: 1 }];
  sim._syncHeldItem(2);
  const carrier = bot(), support = bot();
  const registry = Bots.createDefaultRegistry({ coopHunter: Coop });
  const adapter = registry.create('bun.coop_hunter', {});
  adapter.reset({ seed: 7 });
  const obs = { metadata: { sim }, legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2] };
  const first = adapter.act(obs, 2);
  assert.equal(first.ability, 2, 'registry forwards cooperative item use');
  assert.equal(first.itemSlot, 1, 'registry selects the banana in the second inventory slot');
  Bots.validateAction(first);
  assert.throws(() => Bots.validateAction({ move: 4, ability: 2, itemSlot: 7 }), /item slot/);
  let placed = false, slid = false, slideDistance = 0;
  for (let t = 0; t < 80 && sim.bunCarried[1] >= 0; t++) {
    const a = carrier.act(sim, 1), b = support.act(sim, 2);
    if (b[2]) { placed = true; assert.equal(b[6], 1); }
    const y = sim.pos[2], sliding = sim.movementStatus[1] === 2;
    sim.step([[4, 0, 0, 0], a, b]);
    slid ||= sim.movementStatus[1] === 2;
    if (sliding) slideDistance = Math.max(slideDistance, Math.abs(sim.pos[2] - y));
    assert(sim.alive[1] && !sim.trapped[1], 'banana escort does not injure the carrier');
  }
  assert(placed && slid && slideDistance > 0.2 && sim.bunScore[1] === 1,
    'support leaves its planted banana, carrier triggers acceleration and actually delivers');
  assert.equal(sim.itemSlots[2][0].item, 2, 'banana use preserves the glue in the first slot');
}
{
  const sim = scene([0, 1, 1], [[11.5, 1.5], [6.5, 8.5], [6.5, 8.5]]);
  sim.bunStored = [[0, 0], [0, 0]]; sim.spdG.fill(1);
  const a = bot(), b = bot();
  for (let t = 0; t < 20; t++) sim.step([[4, 0, 0, 0], a.act(sim, 1), b.act(sim, 2)]);
  assert(Math.hypot(sim.pos[2] - sim.pos[4], sim.pos[3] - sim.pos[5]) > 2,
    'two teammates starting at exactly the same coordinates split their movement');
}
{
  const sim = scene([0, 1, 1], [[11.5, 8.5], [6.5, 8.5], [8.8, 8.5]]);
  sim.bunCarried[1] = 0; sim.spdG.fill(1);
  sim.itemSlots[2] = [{ item: 2, count: 1 }]; sim._syncHeldItem(2);
  const carrier = bot(), support = bot();
  let placed = false, slowed = false;
  for (let t = 0; t < 12; t++) {
    const a = carrier.act(sim, 1), b = support.act(sim, 2);
    if (b[2]) { placed = true; assert.equal(support.lastDecision.reason, 'approach_glue'); }
    sim.step([[QQT.MOVE_UP, 0, 0, 0], a, b]);
    slowed ||= sim.movementStatus[0] === QQT.MOVE_STATUS_SLOW;
    assert.equal(sim.movementStatus[1], 0, 'glue never slows the friendly delivery route');
    assert.equal(sim.movementStatus[2], 0, 'support does not step back onto its own glue');
  }
  assert(placed && slowed, 'support plants glue ahead of the pursuer and the enemy actually triggers it');
}
{
  const sim = scene([0, 1, 1], [[11.5, 1.5], [7.5, 8.5], [5.2, 8.5]]);
  sim.bunCarried[1] = 0; sim.spdG.fill(1);
  sim.itemSlots[2] = [{ item: 1, count: 1 }]; sim._syncHeldItem(2);
  sim.wall[4 * W + 8] = 1;
  const support = bot();
  support.analyzeSim(sim, 2);
  assert(!support.itemPlan || support.itemPlan.cell !== 5 * W + 8,
    'a straight slide that stops before reaching home is rejected');
  sim.wall[4 * W + 8] = 0; sim.fuse[5 * W + 10] = 6; sim.owner[5 * W + 10] = 0; sim.bombBlast[5 * W + 10] = 3;
  support.analyzeSim(sim, 2);
  assert(!support.itemPlan || support.itemPlan.cell !== 5 * W + 8,
    'a banana lane crossing a predicted explosion is rejected');
}
{
  const sim = scene([0, 1, 1], [[11.5, 1.5], [6.5, 7.9], [6.5, 8.9]]);
  sim.spdG.fill(1);
  const a = bot(), b = bot();
  for (const [hunter, target] of [[a, 6 * W + 10], [b, 6 * W + 6]]) {
    hunter.chooseGoal = () => ({ mode: 'HUNT', seeds: [{ cell: target, cost: 0 }], threat: false });
    hunter.considerBomb = () => null;
  }
  sim.step([[4, 0, 0, 0], a.act(sim, 1), b.act(sim, 2)]);
  assert(Math.hypot(sim.pos[2] - sim.pos[4], sim.pos[3] - sim.pos[5]) >= 0.75,
    'same-tick movement commitments stop teammates walking into each other');
}
{
  const { trackItemEvents } = require('../scripts/eval_bot_cooperation.js');
  const sim = scene([0, 1, 1], [[5.5, 5.5], [6.5, 8.5], [7.5, 8.5]]);
  const stats = [0, 1].map(() => ({ bananas: 0, glue: 0, carrierSlides: 0, enemySlows: 0, friendlySlows: 0 }));
  trackItemEvents(sim, stats);
  sim.itemSlots[2] = [{ item: QQT.ITEM_SLOW_GLUE, count: 2 }]; sim._syncHeldItem(2);
  sim.wall[7 * W + 8] = 1;
  assert(!sim._placeHeldItem(2), 'blocked item placement fails');
  assert.equal(stats[1].glue, 0, 'failed placement does not inflate usage');
  sim.wall[7 * W + 8] = 0;
  assert(sim._placeHeldItem(2));
  assert.equal(stats[1].glue, 1, 'confirmed placement is counted');
  sim.pos[4] = 8.5;
  sim.pos[0] = 7.5; sim.pos[1] = 8.5;
  sim._updateFieldItems([[4], [4], [4]]);
  assert.equal(stats[1].enemySlows, 1, 'enemy hit is attributed to the actual glue owner');
  assert.equal(stats[0].enemySlows, 0);
  sim.pos[4] = 7.5;
  assert(sim._placeHeldItem(2));
  sim.pos[4] = 8.5; sim.pos[0] = 10.5;
  sim.pos[2] = 7.5;
  sim._updateFieldItems([[4], [4], [4]]);
  assert.equal(stats[1].friendlySlows, 1, 'friendly glue hits are recorded separately');
  assert.equal(stats[1].enemySlows, 1, 'friendly hits do not inflate enemy hits');
}
console.log('Cooperative hunter: roles, resource farming, inventory escort, separation and friendly safety passed');
