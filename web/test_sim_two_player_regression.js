#!/usr/bin/env node
'use strict';

// 1v1 回归金样：多人组队改造不得改变默认 2 人路径（训练/模型/录像口径）的任何状态或随机流。
// 生成：node web/test_sim_two_player_regression.js --record
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const QQT = require('./sim.js');
const Hunter = require('./bun_hunter_bot.js');

const level = require('./assets/maps/levels.json').find((item) => item.qqt_id === 806);
const FIXTURE = path.join(__dirname, 'test_fixtures/sim_two_player_golden.json');
const TICKS = 1200;

function hashFrame(sim) {
  const frame = sim.snapshotReplay(null);
  for (const key of ['nPlayers', 'team']) delete frame[key];
  return crypto.createHash('sha256').update(JSON.stringify(frame)).digest('hex').slice(0, 16);
}

function runCase({ seed, driver, native }) {
  const sim = new QQT.Sim(seed);
  sim.reset(level, native ? { nativeItems: true, nativeTrap: true } : undefined);
  const rng = QQT.mulberry32(seed ^ 0x9e37);
  const bots = [0, 1].map((p) => new Hunter.BunHunterBot({ difficulty: p ? 'hard' : 'normal', seed: seed + p }));
  const hashes = [];
  for (let t = 0; t < TICKS && !sim.done; t++) {
    let actions;
    if (driver === 'random') {
      actions = [0, 1].map(() => [Math.floor(rng() * 5), rng() < 0.15 ? 1 : 0, rng() < 0.05 ? 1 : 0]);
    } else {
      actions = [0, 1].map((p) => bots[p].act(sim, p));
    }
    const info = sim.step(actions);
    if (t % 50 === 49) {
      const summary = JSON.stringify([info.placed, info.died, info.creditedKill, info.causalKill, info.ownBombDefeat]);
      hashes.push(hashFrame(sim) + ':' + crypto.createHash('sha256').update(summary).digest('hex').slice(0, 8));
    }
  }
  hashes.push(`end:${sim.t}:${sim.done}:${sim.winner}:${hashFrame(sim)}`);
  return hashes;
}

const CASES = [];
for (const seed of [1, 7, 42, 2026]) {
  for (const driver of ['random', 'hunter']) {
    // 猎手在糖泡规则下会主动去碰被困者（有意的行为变化），所以 native 只用随机动作锁定 sim 本身。
    for (const native of driver === 'hunter' ? [false] : [false, true]) CASES.push({ seed, driver, native });
  }
}

const key = (c) => `${c.seed}/${c.driver}/${c.native ? 'native' : 'train'}`;
if (process.argv.includes('--record')) {
  const out = {};
  for (const c of CASES) out[key(c)] = runCase(c);
  fs.writeFileSync(FIXTURE, JSON.stringify(out, null, 1) + '\n');
  console.log(`recorded ${CASES.length} cases -> ${FIXTURE}`);
} else {
  const golden = JSON.parse(fs.readFileSync(FIXTURE, 'utf8'));
  for (const c of CASES) {
    const got = runCase(c), want = golden[key(c)];
    for (let i = 0; i < Math.max(got.length, want.length); i++) {
      if (got[i] !== want[i]) throw new Error(`1v1 回归偏离 ${key(c)} 检查点 ${i}: ${got[i]} != ${want[i]}`);
    }
  }
  console.log(`1v1 默认路径逐 tick 回归金样一致（${CASES.length} 例）`);
}
