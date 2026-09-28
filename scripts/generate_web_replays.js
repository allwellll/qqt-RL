#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const QQT = require('../web/sim.js');
const Arena = require('../web/tactical_arena.js');
const { BunRuleTacticalBot, stateFromSim } = require('../web/bun_rule_bot.js');

const ROOT = path.resolve(__dirname, '..');
const EXPORT_ROOT = process.env.QQT_MODEL_EXPORT_ROOT || '/tmp/qqt-model-export';
const specs = [
  ['safe-high-c16', 'high16/actor.json'],
  ['safe-medium-c16', 'medium16/actor.json'],
  ['safe-high-c24', 'high24/actor.json'],
];
const seeds = [202609280032]; // 同一预声明seed用于三个checkpoint的配对观察
const levels = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/assets/maps/levels.json')));
const level = Arena.buildTacticalArena(levels.find((x) => x.qqt_id === 806));
const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/models.json')));
const modelById = new Map(manifest.models.map((x) => [x.id, x]));
const outDir = path.join(ROOT, 'web/replays');
fs.mkdirSync(outDir, { recursive: true });
const sourceHashes = {};
for (const name of ['web/sim.js', 'web/bun_rule_bot.js', 'web/tactical_arena.js', 'web/assets/maps/levels.json']) {
  sourceHashes[name] = crypto.createHash('sha256').update(fs.readFileSync(path.join(ROOT, name))).digest('hex');
}

function makeReplay(modelId, relative, seed) {
  const modelRow = modelById.get(modelId);
  const source = path.join(EXPORT_ROOT, relative);
  const bytes = fs.readFileSync(source);
  if (crypto.createHash('sha256').update(bytes).digest('hex') !== modelRow.sha256) throw new Error(`${modelId}: model hash mismatch`);
  const model = new QQT.TransformerModel(JSON.parse(bytes));
  model.greedy = true;
  const rule = new BunRuleTacticalBot();
  const sim = new QQT.Sim(seed);
  sim.reset(level);
  const rng = QQT.mulberry32(seed ^ 0x515154);
  const actions = [];
  const cells = [new Set(), new Set()];
  const bombs = [0, 0];
  const deaths = [0, 0];
  const maxTicks = 300;
  while (!sim.done && actions.length < maxTicks) {
    const a0 = model.act(sim, 0, rng);
    const a1 = rule.act(sim, 1);
    const row = [Number(a0[0]), Number(a0[1]), Number(a0[2] || 0), Number(a1[0]), Number(a1[1]), Number(a1[2] || 0)];
    actions.push(row);
    bombs[0] += row[1] === 1 ? 1 : 0;
    bombs[1] += row[4] === 1 ? 1 : 0;
    const info = sim.step([[row[0], row[1], row[2]], [row[3], row[4], row[5]]]);
    rule.observeTransition(info, stateFromSim(sim), 1);
    deaths[0] += info.died[0] ? 1 : 0;
    deaths[1] += info.died[1] ? 1 : 0;
    for (let p = 0; p < 2; p++) cells[p].add(sim.centerCell(p).join(','));
  }
  return {
    schema: 'qqt.replay/v1',
    meta: { id: `${modelId}-seed-${seed}`, model_id: modelId, opponent: 'bun.tactical_v2', seed, tick_ms: 100,
      max_ticks: maxTicks, policy: 'greedy', source_sha256: modelRow.source_sha256,
      model_json_sha256: modelRow.sha256, source_hashes: sourceHashes },
    summary: { ticks: actions.length, model_bombs: bombs[0], rule_bombs: bombs[1], model_unique_cells: cells[0].size, rule_unique_cells: cells[1].size, model_deaths: deaths[0], rule_deaths: deaths[1], winner: sim.done ? sim.winner : null },
    actions,
  };
}

const catalog = [];
for (const [modelId, relative] of specs) {
  for (const seed of seeds) {
    const doc = makeReplay(modelId, relative, seed);
    const file = `${doc.meta.id}.json`;
    fs.writeFileSync(path.join(outDir, file), JSON.stringify(doc));
    catalog.push({ id: doc.meta.id, model_id: modelId, seed, file, summary: doc.summary });
  }
}
fs.writeFileSync(path.join(ROOT, 'web/replays.json'), JSON.stringify({ schema: 'qqt.replays/v1', replays: catalog }, null, 2) + '\n');
console.log(JSON.stringify(catalog, null, 2));
