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

console.log('猎手 Bot：躲泡/进攻/偷包搬运/难度注册 回归通过');
