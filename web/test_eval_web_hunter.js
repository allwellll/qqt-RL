'use strict';

// scripts/eval_web_hunter.js：难度选择、固定出生/seed、danger_arena 等价设置与逐局指标口径。
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const QQT = require('./sim.js');
const Eval = require('../scripts/eval_web_hunter.js');

// 1) 三档难度必须走真实网页猎手（bot_contract 注册的 bun.hunter），配置逐项对应。
for (const difficulty of ['easy', 'normal', 'hard']) {
  const hunter = Eval.createHunter(difficulty);
  assert.strictEqual(hunter.spec.id, 'bun.hunter');
  assert.strictEqual(hunter.adapter.bot.constructor.name, 'BunHunterBot');
  assert.strictEqual(hunter.adapter.bot.difficulty, difficulty);
  assert.deepStrictEqual(hunter.config, { difficulty });
  const expected = require('./bun_hunter_bot.js').DIFFICULTIES[difficulty];
  assert.deepStrictEqual(hunter.resolved, Object.assign({}, expected));
}
assert.notDeepStrictEqual(Eval.createHunter('easy').resolved, Eval.createHunter('hard').resolved);
assert.throws(() => Eval.createHunter('hunter_hard'), /unknown hunter difficulty/);

// 2) 环境：清砖、hp=1、300 tick、出生点按清单放置、抓包不终局。
const level = Eval.loadLevel();
const sim = Eval.setupSim(level, [[7, 0], [4, 0]], 123, 300);
assert.strictEqual(sim.brick.reduce((a, b) => a + b, 0), 0);
assert.deepStrictEqual(Array.from(sim.hp), [1, 1]);
assert.strictEqual(sim.maxSteps, 300);
assert.deepStrictEqual([Math.floor(sim.pos[0]), Math.floor(sim.pos[1]), Math.floor(sim.pos[2]), Math.floor(sim.pos[3])], [7, 0, 4, 0]);
assert.deepStrictEqual(sim.bunSpawnPos, [[7.5, 0.5], [4.5, 0.5]]);
assert.strictEqual(sim._bunHasCapturedAll(0), false);
sim.bunStored[0][1] = 1;
assert.strictEqual(sim._bunHasCapturedAll(0), false, 'capture must not end a danger_arena-style episode');
assert.strictEqual(Eval.episodeSeed(20261001, 3), (20261001 + 3 * 7919) >>> 0);

// 3) 用确定性的“桩模型”跑真实分片：每局恰好覆盖 [start,end)、出生一致、元数据齐全、可复现。
class StubModel {
  constructor() { this.obsShape = [24, 13, 15]; }
  act(s, pid) { return [s.t % 7 === 0 ? 1 : 4, s.t % 20 === 5 ? 1 : 0, 0]; }
  forward() { return { move: [0, 0, 0, 0, 1], bomb: [1, 0, 0] }; }
  _abilityMask() { return [1, 1, 0]; }
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hunter-eval-'));
const spawnsPath = path.join(tmp, 'spawns.json');
const spawnCells = [[[7, 0], [4, 0]], [[2, 1], [6, 0]], [[2, 12], [7, 14]], [[5, 14], [2, 13]]];
fs.writeFileSync(spawnsPath, JSON.stringify({ seed: 77, spawn_sha256: 'abc', spawn_cells: spawnCells }));
const stub = { model: new StubModel(), sha: 'f'.repeat(64), meta: { source: 'stub.pt', embed: 4, depth: 1, patch: 3 } };
const options = { model: stub, difficulty: 'normal', spawnsPath, start: 1, end: 4, seed: 77, maxSteps: 60,
  checkpointSha: 'e'.repeat(64) };
const shard = Eval.runShard(options);
assert.strictEqual(shard.schema, Eval.SCHEMA);
assert.deepStrictEqual(shard.game_range, [1, 4]);
assert.deepStrictEqual(shard.episodes.map((e) => e.game), [1, 2, 3]);
shard.episodes.forEach((e) => assert.deepStrictEqual(e.spawn_cells, spawnCells[e.game]));
assert.strictEqual(shard.bot.id, 'bun.hunter');
assert.strictEqual(shard.bot.version, '1.0.0');
assert.deepStrictEqual(shard.bot.config, { difficulty: 'normal' });
assert.strictEqual(shard.checkpoint_sha256, 'e'.repeat(64));
assert.strictEqual(shard.model.model_json_sha256, 'f'.repeat(64));
assert.strictEqual(shard.spawn_manifest_sha256, 'abc');
for (const name of ['web/sim.js', 'web/bun_hunter_bot.js', 'web/bot_contract.js']) {
  assert.match(shard.source_hashes[name], /^[0-9a-f]{64}$/);
}
for (const e of shard.episodes) {
  assert(e.ticks <= 60);
  for (const key of ['surviving_causal_kill', 'surviving_physical_kill', 'own_bomb_defeat', 'killed_by_bot',
    'mutual_death', 'deaths', 'bombs', 'bot_deaths', 'bot_own_bomb_defeat', 'bot_bombs']) {
    assert(Number.isInteger(e.counts[key]) && e.counts[key] >= 0, key);
  }
  assert(e.counts.bombs > 0, 'stub places bombs');
  assert(e.counts.deaths >= e.counts.own_bomb_defeat + e.counts.killed_by_bot);
}
const again = Eval.runShard(options);
assert.deepStrictEqual(again.episodes.map((e) => [e.counts, e.ticks]), shard.episodes.map((e) => [e.counts, e.ticks]));
assert.throws(() => Eval.runShard(Object.assign({}, options, { seed: 78 })), /seed/);
assert.throws(() => Eval.runShard(Object.assign({}, options, { end: 5 })), /bad game range/);

// 4) 击杀归因：模型泡单独炸死猎手 => surviving kill；同 tick 双亡 => 只计换命。
function forcedKill(mutual) {
  const s = Eval.setupSim(level, [[9, 5], [9, 7]], 5, 300);
  const cell = 9 * 15 + 6;
  s.fuse[cell] = 1; s.owner[cell] = 0; s.bombBlast[cell] = 3;
  if (!mutual) { s.pos[0] = 6.5; s.pos[1] = 0.5; }
  const info = s.step([[4, 0, 0], [4, 0, 0]]);
  return { info, alive: Array.from(s.alive) };
}
const solo = forcedKill(false);
assert.deepStrictEqual(solo.info.died, [false, true]);
assert.strictEqual(solo.info.causalKill[0] && solo.alive[0], true);
const trade = forcedKill(true);
assert.strictEqual(trade.info.mutualDeath, true);
assert.strictEqual(trade.info.causalKill[0], false);

// 5) 命令行参数：缺参报错，parity 模式独立。
assert.throws(() => Eval.parseArgs(['--model', 'x']), /missing --difficulty/);
assert.strictEqual(Eval.parseArgs(['--parity-dump', 'o.json', '--model', 'm', '--spawns', 's', '--seed', '1']).parity, true);
fs.rmSync(tmp, { recursive: true, force: true });
console.log('网页猎手评估器：难度/出生/seed/元数据/归因 检查通过');
