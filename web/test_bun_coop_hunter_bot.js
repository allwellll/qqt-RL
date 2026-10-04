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
  assert.equal(decide(sim, 2).mode, 'SUPPORT');
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
console.log('Cooperative hunter: roles, takeover, rescue, delivery, escort, interception and friendly safety passed');
