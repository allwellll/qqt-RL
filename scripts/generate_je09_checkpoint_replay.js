'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const QQT = require('../web/sim.js');
const Arena = require('../web/tactical_arena.js');
const { BunRuleTacticalBot, stateFromSim } = require('../web/bun_rule_bot.js');

const ROOT = path.resolve(__dirname, '..');
const SEED = 2026101736;
const MAX_TICKS = 300;
const MODEL_ID = 'Bun392 JE09路线势能连续g32';
const REPLAY_ID = 'bun392-je09-route-pbrs-continuous-v36-g32-seed-2026101736';
const REPLAY_FILE = `${REPLAY_ID}.json`;
const SOURCE_FILES = ['web/sim.js', 'web/tactical_arena.js', 'web/bun_rule_bot.js',
  'web/assets/maps/levels.json', 'scripts/generate_je09_checkpoint_replay.js'];

function sha(bytes) { return crypto.createHash('sha256').update(bytes).digest('hex'); }

function simulate(model, seed) {
  const levels = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/assets/maps/levels.json')));
  const level = Arena.buildTacticalArena(levels.find((row) => row.qqt_id === 806));
  const sim = new QQT.Sim(seed);
  sim.reset(level);
  const bot = new BunRuleTacticalBot();
  const rng = QQT.mulberry32(seed ^ 0x515154);
  const actions = [];
  const cells = [new Set(), new Set()];
  const bombs = [0, 0];
  const deaths = [0, 0];
  while (!sim.done && actions.length < MAX_TICKS) {
    const a0 = model.act(sim, 0, rng);
    const a1 = bot.act(sim, 1);
    const row = [Number(a0[0]), Number(a0[1]), Number(a0[2] || 0),
      Number(a1[0]), Number(a1[1]), Number(a1[2] || 0)];
    actions.push(row);
    bombs[0] += row[1] === 1 ? 1 : 0;
    bombs[1] += row[4] === 1 ? 1 : 0;
    const info = sim.step([[row[0], row[1], row[2]], [row[3], row[4], row[5]]]);
    bot.observeTransition(info, stateFromSim(sim), 1);
    for (let pid = 0; pid < 2; pid++) {
      deaths[pid] += info.died[pid] ? 1 : 0;
      cells[pid].add(sim.centerCell(pid).join(','));
    }
  }
  return { actions, summary: { ticks: actions.length, model_bombs: bombs[0], rule_bombs: bombs[1],
    model_unique_cells: cells[0].size, rule_unique_cells: cells[1].size,
    model_deaths: deaths[0], rule_deaths: deaths[1], winner: sim.done ? sim.winner : null } };
}

function makeDocument(modelPath) {
  const modelBytes = fs.readFileSync(modelPath);
  const modelSha = sha(modelBytes);
  const document = JSON.parse(modelBytes);
  const model = new QQT.TransformerModel(document);
  model.greedy = true;
  const result = simulate(model, SEED);
  const sourceHashes = Object.fromEntries(SOURCE_FILES.map((name) =>
    [name, sha(fs.readFileSync(path.join(ROOT, name)))]));
  const checkpointSha = document.meta.checkpoint_sha256;
  if (!/^[0-9a-f]{64}$/.test(checkpointSha || '')) throw new Error('checkpoint hash missing from exported model');
  return { schema: 'qqt.replay/v1', meta: { id: REPLAY_ID, model_id: MODEL_ID,
    opponent: 'bun.tactical_v2', seed: SEED, tick_ms: 100, max_ticks: MAX_TICKS,
    policy: 'greedy', source_sha256: checkpointSha, checkpoint_sha256: checkpointSha,
    model_json_sha256: modelSha, source_hashes: sourceHashes },
  summary: result.summary, actions: result.actions };
}

function main() {
  const [modelArg, outputArg] = process.argv.slice(2);
  if (!modelArg || !outputArg) throw new Error('usage: node scripts/generate_je09_checkpoint_replay.js MODEL.json OUTPUT.json');
  const modelPath = path.resolve(modelArg);
  const outputPath = path.resolve(outputArg);
  const replay = makeDocument(modelPath);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const encoded = JSON.stringify(replay);
  fs.writeFileSync(outputPath, encoded);

  const catalogPath = path.join(ROOT, 'web/replays.json');
  const catalog = JSON.parse(fs.readFileSync(catalogPath, 'utf8'));
  catalog.replays = catalog.replays.filter((row) => row.id !== REPLAY_ID);
  catalog.replays.push({ id: REPLAY_ID, model_id: MODEL_ID, seed: SEED,
    file: REPLAY_FILE, summary: replay.summary });
  fs.writeFileSync(catalogPath, `${JSON.stringify(catalog, null, 2)}\n`);
  console.log(JSON.stringify({ replay: REPLAY_FILE, replay_sha256: sha(encoded),
    model_json_sha256: replay.meta.model_json_sha256, checkpoint_sha256: replay.meta.checkpoint_sha256,
    seed: SEED, ticks: replay.summary.ticks, summary: replay.summary }));
}

if (require.main === module) main();
module.exports = { makeDocument, REPLAY_ID, SEED, sha };
