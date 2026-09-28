#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const QQT = require('../web/sim.js');
const Arena = require('../web/tactical_arena.js');
const { BunRuleTacticalBot, stateFromSim } = require('../web/bun_rule_bot.js');
const Replay = require('../web/replay.js');

const ROOT = path.resolve(__dirname, '..');
const EXPORT_ROOT = process.env.QQT_MODEL_EXPORT_ROOT || '/tmp/qqt-model-export';
const specs = [['safe-high-c16', 'high16/actor.json'], ['safe-medium-c16', 'medium16/actor.json'], ['safe-high-c24', 'high24/actor.json']];
const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/models.json')));
const models = new Map(manifest.models.map((x) => [x.id, x]));
const levels = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/assets/maps/levels.json')));
const level = Arena.buildTacticalArena(levels.find((x) => x.qqt_id === 806));
const catalog = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/replays.json')));
assert.strictEqual(catalog.schema, 'qqt.replays/v1');

for (const row of catalog.replays) {
  const spec = specs.find(([id]) => id === row.model_id);
  assert(spec, `${row.id}: unknown model`);
  const bytes = fs.readFileSync(path.join(EXPORT_ROOT, spec[1]));
  const modelRow = models.get(row.model_id);
  assert.strictEqual(crypto.createHash('sha256').update(bytes).digest('hex'), modelRow.sha256);
  const model = new QQT.TransformerModel(JSON.parse(bytes)); model.greedy = true;
  const bot = new BunRuleTacticalBot();
  const sim = new QQT.Sim(row.seed); sim.reset(level);
  const rng = QQT.mulberry32(row.seed ^ 0x515154);
  const doc = Replay.validateReplay(JSON.parse(fs.readFileSync(path.join(ROOT, 'web/replays', row.file))));
  for (let t = 0; t < doc.actions.length; t++) {
    const expectedModel = model.act(sim, 0, rng);
    const expectedBot = bot.act(sim, 1);
    const recorded = doc.actions[t];
    assert.deepStrictEqual(recorded, [Number(expectedModel[0]), Number(expectedModel[1]), Number(expectedModel[2] || 0), Number(expectedBot[0]), Number(expectedBot[1]), Number(expectedBot[2] || 0)], `${row.id}: action mismatch at tick ${t}`);
    const info = sim.step([[recorded[0], recorded[1], recorded[2]], [recorded[3], recorded[4], recorded[5]]]);
    bot.observeTransition(info, stateFromSim(sim), 1);
  }
  assert.strictEqual(sim.t, doc.summary.ticks);
  assert.strictEqual(sim.done ? sim.winner : null, doc.summary.winner);
  console.log(`verified ${row.id}: ${sim.t} ticks`);
}
