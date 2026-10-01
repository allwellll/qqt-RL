#!/usr/bin/env node
'use strict';

// 猎手 Bot 回归：躲泡、不自杀、击杀后偷包搬回、注册表/难度配置。
const assert = require('assert');
const QQT = require('./sim.js');
const Hunter = require('./bun_hunter_bot.js');
const QQTBots = require('./bot_contract.js');
const { BunRuleTacticalBot, stateFromSim } = require('./bun_rule_bot.js');
const levels = require('./assets/maps/levels.json');

const level = levels.find((item) => item.qqt_id === 806);
const W = QQT.W;

function playMatch(difficulty, seed, opponent = 'idle', maxSteps = 2400) {
  const sim = new QQT.Sim(seed);
  sim.reset(level);
  sim.maxSteps = maxSteps;
  const bot = new Hunter.BunHunterBot({ difficulty, seed });
  const rule = opponent === 'tactical' ? new BunRuleTacticalBot() : null;
  const stats = { selfDeaths: 0, kills: 0, delivered: 0, winner: null, ticks: 0 };
  while (!sim.done) {
    const foe = rule ? rule.act(sim, 0) : [QQT.MOVE_IDLE, 0, 0];
    const action = bot.act(sim, 1);
    const score = sim.bunScore[1];
    const info = sim.step([[foe[0], foe[1], 0, 0], [action[0], action[1], 0, 0]]);
    if (rule) rule.observeTransition(info, stateFromSim(sim), 0);
    if (sim.lastDied[1] && !info.creditedKill[0]) stats.selfDeaths++;
    if (info.creditedKill[1]) stats.kills++;
    if (sim.bunScore[1] > score) stats.delivered++;
  }
  stats.winner = sim.winner;
  stats.ticks = sim.t;
  return stats;
}

// 1) 躲泡：真人泡泡即将爆炸，Bot 站在火线上，必须逃出火区。
function dodgeScene() {
  const sim = new QQT.Sim(3);
  sim.reset('open');
  sim.wall.fill(0); sim.brick.fill(0); sim.crate.fill(0); sim.fuse.fill(0);
  sim.isBun = true;
  sim.pos[0] = 11.5; sim.pos[1] = 1.5;
  sim.pos[2] = 5.5; sim.pos[3] = 6.5;
  const cell = 5 * W + 5;
  sim.fuse[cell] = 8; sim.owner[cell] = 0; sim.bombBlast[cell] = 3;
  return sim;
}
for (const difficulty of ['easy', 'normal', 'hard']) {
  const sim = dodgeScene();
  const bot = new Hunter.BunHunterBot({ difficulty, seed: 1, overrides: { mistakeRate: 0 } });
  for (let t = 0; t < 14; t++) {
    const action = bot.act(sim, 1);
    sim.step([[QQT.MOVE_IDLE, 0, 0, 0], [action[0], action[1], 0, 0]]);
    assert(sim.alive[1], `${difficulty}: 猎手必须躲开即将爆炸的泡泡 (t=${t})`);
  }
}

// 2) 对挂机对手：三档都能击杀、从对方包子笼搬包子回家并获胜，且不自杀。
for (const difficulty of ['easy', 'normal', 'hard']) {
  for (const seed of [1000, 1001, 1002]) {
    const stats = playMatch(difficulty, seed);
    assert.strictEqual(stats.selfDeaths, 0, `${difficulty}/${seed}: 不得被自己的泡炸死`);
    assert(stats.kills >= 1, `${difficulty}/${seed}: 应主动进攻并击杀对手`);
    assert.strictEqual(stats.delivered, 1, `${difficulty}/${seed}: 应偷到包子并送回己方包子屋`);
    assert.strictEqual(stats.winner, 1, `${difficulty}/${seed}: 应在限时内获胜`);
  }
}

// 3) 对会放泡的战术规则 Bot：困难档仍不输。
for (const seed of [1000, 1001]) {
  const stats = playMatch('hard', seed, 'tactical');
  assert.notStrictEqual(stats.winner, 0, `hard/${seed}: 不应输给战术规则 Bot`);
}

// 4) 注册表：难度配置校验、依赖 metadata.sim。
{
  const registry = QQTBots.createDefaultRegistry({ BunRuleTacticalBot, hunter: Hunter });
  const spec = registry.describe('bun.hunter', {});
  assert.strictEqual(spec.config.difficulty, 'normal');
  assert.throws(() => registry.create('bun.hunter', { difficulty: 'insane' }), /invalid config value/);
  const bot = registry.create('bun.hunter', { difficulty: 'hard' });
  bot.reset({ seed: 5 });
  const sim = new QQT.Sim(5);
  sim.reset(level);
  const observation = { state: {}, legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2] };
  assert.throws(() => bot.act(observation, 1), /metadata\.sim/);
  const action = bot.act({ ...observation, metadata: { sim } }, 1);
  QQTBots.validateAction(action);
  assert.throws(() => new Hunter.BunHunterBot({ difficulty: 'insane' }), /unknown hunter difficulty/);
}

// 5) 组队协作（网页 1v2/2v2，糖泡规则）：救队友、爆破被困敌人、不误伤队友。
function teamScene(teams, coords) {
  const sim = new QQT.Sim(11);
  sim.reset('open', { nativeItems: true, nativeTrap: true, teams });
  sim.wall.fill(0); sim.brick.fill(0); sim.crate.fill(0); sim.fuse.fill(0);
  sim.isBun = true;
  coords.forEach(([y, x], p) => { sim.pos[p * 2] = y; sim.pos[p * 2 + 1] = x; sim.invuln[p] = 0; });
  return sim;
}
function runTeam(sim, bots, ticks, until) {
  for (let t = 0; t < ticks; t++) {
    const actions = sim.team.map((_, p) => bots[p] ? bots[p].act(sim, p) : [QQT.MOVE_IDLE, 0, 0]);
    sim.step(actions.map((a) => [a[0], a[1], 0, 0]));
    if (until(sim)) return t;
  }
  return -1;
}
for (const difficulty of ['easy', 'normal', 'hard']) {
  // 1v2：蓝方 pid1 被困，队友 pid2 必须赶来救出；真人 pid0 远离。
  const sim = teamScene([0, 1, 1], [[11.5, 1.5], [5.5, 9.5], [5.5, 5.5]]);
  sim.trapped[1] = 60;
  const bot = new Hunter.BunHunterBot({ difficulty, seed: 2, overrides: { mistakeRate: 0 } });
  assert.strictEqual(bot.analyze(Hunter.hunterStateFromSim(sim), 2).mode, 'RESCUE', `${difficulty}: 队友被困时进入 RESCUE`);
  const t = runTeam(sim, [null, null, bot], 50, (s) => s.trapped[1] === 0);
  assert(t >= 0 && sim.alive[1], `${difficulty}: 猎手在糖泡爆破前救出队友 (t=${t})`);
}
for (const difficulty of ['easy', 'normal', 'hard']) {
  // 2v2：红方真人 pid0 被困，敌方猎手 pid1 必须去碰爆。
  const sim = teamScene([0, 1, 0, 1], [[5.5, 4.5], [5.5, 9.5], [11.5, 1.5], [11.5, 13.5]]);
  sim.trapped[0] = 60;
  const bot = new Hunter.BunHunterBot({ difficulty, seed: 3, overrides: { mistakeRate: 0 } });
  assert.strictEqual(bot.analyze(Hunter.hunterStateFromSim(sim), 1).mode, 'POP', `${difficulty}: 敌人被困时进入 POP`);
  const t = runTeam(sim, [null, bot, null, null], 50, (s) => !s.alive[0]);
  assert(t >= 0 && sim.trapped[0] === 0, `${difficulty}: 猎手赶在自动爆破前碰爆被困敌人 (t=${t})`);
}
{
  // 赶不到（剩余糖泡时间太短）就不追。
  const sim = teamScene([0, 1, 1], [[11.5, 1.5], [1.5, 13.5], [11.5, 13.5]]);
  sim.trapped[1] = 3;
  const bot = new Hunter.BunHunterBot({ difficulty: 'hard', seed: 4 });
  assert.notStrictEqual(bot.analyze(Hunter.hunterStateFromSim(sim), 2).mode, 'RESCUE', '赶不到不追');
}
{
  // 友军伤害：敌人 pid0 困在死角、放泡必杀；但队友 pid2 在正上方的死胡同里，出口就在火线上 → 不放。
  const corridor = (teams) => {
    const sim = teamScene(teams, [[5.5, 6.5], [5.5, 5.5], [3.5, 5.5]].slice(0, teams.length));
    for (const [r, c] of [[3, 4], [3, 6], [2, 5], [4, 4], [4, 6], [4, 7], [6, 7], [5, 8], [6, 6], [6, 5]]) sim.wall[r * W + c] = 1;
    return sim;
  };
  const decide = (sim) => {
    const bot = new Hunter.BunHunterBot({ difficulty: 'hard', seed: 5 });
    const state = Hunter.hunterStateFromSim(sim);
    const g = bot.geometry(state);
    const perceive = () => true;
    const pred = bot.predict(state, g, [], perceive);
    const field = bot.goalField(state, g, bot.attackSeeds(state, g, 1, pred, perceive), state.players[1], pred, null);
    return bot.considerBomb(state, g, 1, pred, perceive, field);
  };
  assert.strictEqual(decide(corridor([0, 1])).reason, 'bomb_kill', '对照：无队友时会放泡击杀');
  assert.strictEqual(decide(corridor([0, 1, 1])), null, '放泡会困死队友时不放');
}

console.log('猎手 Bot：躲泡/进攻/偷包搬运/难度注册/组队救援 回归通过');
