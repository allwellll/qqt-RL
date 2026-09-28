'use strict';

const assert = require('assert');
const fs = require('fs');
const QQT = require('./sim.js');
const { BunRuleTacticalBot, stateFromSim } = require('./bun_rule_bot.js');
const { buildTacticalArena } = require('./tactical_arena.js');

const levels = JSON.parse(fs.readFileSync(require('path').join(__dirname, 'assets/maps/levels.json'), 'utf8'));
const base = levels.find((level) => level.qqt_id === 806);
const arena = buildTacticalArena(base);
assert(arena.brick.every((value) => value === 0), '网页战术对局必须清空可炸砖，匹配训练竞技场');
assert(arena.layers[1].every((value) => value === 0), '网页视觉砖层也必须清空');

function run(seed, humanPolicy) {
  const sim = new QQT.Sim(seed);
  sim.reset(arena);
  const bot = new BunRuleTacticalBot();
  let moved = 0, placed = 0, escapedThreat = false;
  for (let tick = 0; tick < 240 && !sim.done; tick++) {
    const before = stateFromSim(sim);
    const decision = bot.analyze(before, 1);
    const old = [sim.pos[2], sim.pos[3]];
    const human = humanPolicy(sim, tick);
    const info = sim.step([human, [decision.action[0], decision.action[1], 0]]);
    bot.observeTransition(info, stateFromSim(sim), 1);
    if (Math.abs(sim.pos[2] - old[0]) + Math.abs(sim.pos[3] - old[1]) > 1e-6) moved++;
    if (info.placed[1]) placed++;
    if (before.bombs.some((bomb) => bomb.physical_owner === 0) && decision.reason === 'escape_immediate' && decision.action[0] !== 4) escapedThreat = true;
  }
  return { moved, placed, escapedThreat };
}

const active = run(17, () => [4, 0, 0]);
assert(active.moved >= 10, `战术机器人应持续接近而非原地不动，实际移动tick=${active.moved}`);
assert(active.placed >= 1, `战术机器人应找到安全机会放泡，实际=${active.placed}`);

const threatened = run(29, (sim, tick) => {
  const distance = Math.abs(sim.pos[0] - sim.pos[2]) + Math.abs(sim.pos[1] - sim.pos[3]);
  if (tick % 25 === 0 && distance <= sim.blastCap[0] + 1) return [4, 1, 0];
  let best = 4, score = Infinity;
  for (const move of [0, 1, 2, 3, 4]) {
    const d = move < 4 ? QQT.DIRS[move] : [0, 0];
    const value = Math.abs(sim.pos[0] + d[0] - sim.pos[2]) + Math.abs(sim.pos[1] + d[1] - sim.pos[3]);
    if (value < score) { score = value; best = move; }
  }
  return [best, 0, 0];
});
assert(threatened.escapedThreat, '面对真人糖泡时应识别危险并采取非待机逃生动作');
console.log('网页 Tactical v2 训练竞技场行为回归通过', { active, threatened });
