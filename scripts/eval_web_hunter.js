#!/usr/bin/env node
'use strict';

// Model (player 0, web JS TransformerModel, greedy) vs real web BunHunterBot
// (player 1, via bot_contract HunterAdapter exactly as web/app.js) in web/sim.js.
// danger_arena-equivalent setup: map 806, destructible bricks cleared, hp=1,
// 300 ticks, no capture termination, spawn pair i from a JAX-generated manifest.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const HunterApi = require('../web/bun_hunter_bot.js');
const QQTBots = require('../web/bot_contract.js');

const ROOT = path.resolve(__dirname, '..');
const SCHEMA = 'web_hunter_eval_shard_v1';
const BOT_ID = 'bun.hunter';
const DIFFICULTIES = Object.keys(HunterApi.DIFFICULTIES);
const SOURCE_FILES = ['web/sim.js', 'web/bun_hunter_bot.js', 'web/bot_contract.js',
  'web/assets/maps/levels.json', 'scripts/eval_web_hunter.js'];

function sha256(bytes) { return crypto.createHash('sha256').update(bytes).digest('hex'); }

function sourceHashes() {
  const out = {};
  for (const name of SOURCE_FILES) out[name] = sha256(fs.readFileSync(path.join(ROOT, name)));
  return out;
}

function loadLevel() {
  const levels = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/assets/maps/levels.json'), 'utf8'));
  const level = levels.find((item) => item.qqt_id === 806);
  if (!level) throw new Error('map 806 missing');
  return level;
}

function createHunter(difficulty) {
  if (!DIFFICULTIES.includes(difficulty)) throw new Error(`unknown hunter difficulty: ${difficulty}`);
  const registry = QQTBots.createDefaultRegistry({ hunter: HunterApi });
  const spec = registry.list().find((item) => item.id === BOT_ID);
  const adapter = registry.create(BOT_ID, { difficulty });
  if (adapter.bot.difficulty !== difficulty) throw new Error('hunter difficulty not applied');
  return { adapter, spec, config: { difficulty }, resolved: Object.assign({}, adapter.bot.cfg) };
}

function episodeSeed(seed, game) { return (Number(seed) + game * 7919) >>> 0; }

function setupSim(level, spawn, simSeed, maxSteps) {
  const sim = new QQT.Sim(simSeed);
  sim.reset(level);
  sim.brick.fill(0);
  sim.brickLinger.fill(0);
  sim.maxSteps = maxSteps;
  sim.bunInitial = [Infinity, Infinity];
  sim.initialHp = 1;
  for (let p = 0; p < 2; p++) {
    const y = spawn[p][0] + 0.5, x = spawn[p][1] + 0.5;
    sim.pos[p * 2] = y; sim.pos[p * 2 + 1] = x;
    sim.bunSpawnPos[p] = [y, x];
    sim.hp[p] = 1;
    sim.invuln[p] = 0;
  }
  sim._gen += 1;
  return sim;
}

function observationFor(sim) {
  return { schema: 'qqt.bot.observation/v1', tick: sim.t, state: null,
    legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2], metadata: { sim } };
}

// Mirrors bun_env opponent_physical_defeat | opponent_causal_defeat (sole source, no trade).
function killedByOpponent(info, victim) {
  const killer = 1 - victim;
  const phys = info.physicalDamageSource[victim], causal = info.causalDamageSource[victim];
  const physCount = Number(phys[0]) + Number(phys[1]);
  const causalCount = Number(causal[0]) + Number(causal[1]);
  return !!info.died[victim] && !info.mutualDeath
    && ((phys[killer] && physCount === 1) || (causal[killer] && causalCount === 1));
}

function runEpisode({ level, model, hunter, spawn, game, seed, maxSteps, rngSeed, trace }) {
  const simSeed = episodeSeed(seed, game);
  const sim = setupSim(level, spawn, simSeed, maxSteps);
  const botSeed = episodeSeed(seed ^ 0x68756e74, game);
  hunter.adapter.reset({ schema: 'qqt.bot.context/v1', episode_id: `eval-${game}`, seed: botSeed,
    ruleset: 'bun', max_ticks: maxSteps, metadata: {} });
  const rng = QQT.mulberry32(rngSeed == null ? simSeed ^ 0x515154 : rngSeed);
  const counts = {
    surviving_causal_kill: 0, surviving_physical_kill: 0, own_bomb_defeat: 0,
    killed_by_bot: 0, mutual_death: 0, deaths: 0, bombs: 0,
    bot_deaths: 0, bot_own_bomb_defeat: 0, bot_bombs: 0,
  };
  const startSpawn = [[Math.floor(sim.pos[0]), Math.floor(sim.pos[1])],
    [Math.floor(sim.pos[2]), Math.floor(sim.pos[3])]];
  let modelMs = 0, botMs = 0;
  while (!sim.done && sim.t < maxSteps) {
    if (trace && trace.rows.length < trace.limit) {
      const obs = sim.encodeObsJAX(0, model.obsShape[0]);
      const state = sim.encodeStateJAX(0);
      const masks = sim.legalMask();
      const logits = model.forward(obs, state);
      trace.rows.push({ tick: sim.t, obs: Array.from(obs), state: Array.from(state),
        move_mask: masks.mm[0], ability_mask: model._abilityMask(sim, masks, 0),
        move: Array.from(logits.move), ability: Array.from(logits.bomb) });
    }
    let started = performance.now();
    const raw = model.act(sim, 0, rng);
    modelMs += performance.now() - started;
    started = performance.now();
    const botAction = hunter.adapter.act(observationFor(sim), 1, rng);
    botMs += performance.now() - started;
    const info = sim.step([
      [Number(raw[0]), Number(raw[1]), Number(raw[2]) || 0],
      [botAction.move, botAction.ability === 1 ? 1 : 0, botAction.ability === 2 ? 1 : 0],
    ]);
    counts.bombs += info.placed[0] ? 1 : 0;
    counts.bot_bombs += info.placed[1] ? 1 : 0;
    counts.deaths += info.died[0] ? 1 : 0;
    counts.bot_deaths += info.died[1] ? 1 : 0;
    counts.mutual_death += info.mutualDeath ? 1 : 0;
    counts.own_bomb_defeat += info.ownBombDefeat[0] ? 1 : 0;
    counts.bot_own_bomb_defeat += info.ownBombDefeat[1] ? 1 : 0;
    counts.killed_by_bot += killedByOpponent(info, 0) ? 1 : 0;
    counts.surviving_causal_kill += info.causalKill[0] && sim.alive[0] ? 1 : 0;
    counts.surviving_physical_kill += info.creditedKill[0] && sim.alive[0] ? 1 : 0;
  }
  return { game, sim_seed: simSeed, bot_seed: botSeed, spawn_cells: startSpawn,
    ticks: sim.t, counts, model_ms: modelMs, bot_ms: botMs };
}

function loadModel(file) {
  const bytes = fs.readFileSync(file);
  const doc = JSON.parse(bytes.toString('utf8'));
  if (!doc.meta || doc.meta.arch !== 'transformer') throw new Error('expected a transformer web model');
  const model = new QQT.TransformerModel(doc);
  model.greedy = true;
  return { model, sha: sha256(bytes), meta: doc.meta };
}

function runShard(options) {
  const { modelPath, difficulty, spawnsPath, start, end, seed, maxSteps, checkpointSha } = options;
  const manifest = JSON.parse(fs.readFileSync(spawnsPath, 'utf8'));
  if (manifest.seed !== seed) throw new Error(`spawn manifest seed ${manifest.seed} != ${seed}`);
  if (!(Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end > start
    && end <= manifest.spawn_cells.length)) throw new Error(`bad game range [${start},${end})`);
  const level = loadLevel();
  const { model, sha: modelSha, meta } = options.model ? options.model : loadModel(modelPath);
  const hunter = createHunter(difficulty);
  const started = Date.now();
  const episodes = [];
  for (let game = start; game < end; game++) {
    const spawn = manifest.spawn_cells[game];
    const record = runEpisode({ level, model, hunter, spawn, game, seed, maxSteps });
    if (JSON.stringify(record.spawn_cells) !== JSON.stringify(spawn)) throw new Error(`spawn mismatch game ${game}`);
    episodes.push(record);
  }
  return {
    schema: SCHEMA, seed, max_steps: maxSteps, game_range: [start, end],
    spawn_manifest_sha256: manifest.spawn_sha256,
    bot: { id: BOT_ID, version: hunter.spec.version, display_name: hunter.spec.display_name,
      config: hunter.config, resolved_config: hunter.resolved },
    model: { runtime: 'web/sim.js TransformerModel', greedy: true, model_json_sha256: modelSha,
      source: meta.source, embed: meta.embed, depth: meta.depth, patch: meta.patch },
    checkpoint_sha256: checkpointSha || null,
    protocol: { map: 806, destructible_bricks_cleared: true, hp: 1, capture_terminates: false,
      native_items: false, native_trap: false, actor_player: 0, bot_player: 1 },
    source_hashes: sourceHashes(),
    node_version: process.version,
    elapsed_seconds: (Date.now() - started) / 1000,
    episodes,
  };
}

// t=0 observations for every manifest spawn plus a short traced episode (obs/state/
// JS logits per tick) so the Python side can check encoder and network parity vs JAX.
function parityDump({ modelPath, spawnsPath, seed, maxSteps, games, ticks, difficulty }) {
  const manifest = JSON.parse(fs.readFileSync(spawnsPath, 'utf8'));
  const level = loadLevel();
  const loaded = loadModel(modelPath);
  const initial = manifest.spawn_cells.slice(0, games).map((spawn, game) => {
    const sim = setupSim(level, spawn, episodeSeed(seed, game), maxSteps);
    return { game, obs: Array.from(sim.encodeObsJAX(0, loaded.model.obsShape[0])),
      state: Array.from(sim.encodeStateJAX(0)), move_mask: sim.legalMask().mm[0] };
  });
  const trace = { rows: [], limit: ticks };
  runEpisode({ level, model: loaded.model, hunter: createHunter(difficulty), spawn: manifest.spawn_cells[0],
    game: 0, seed, maxSteps, trace });
  return { schema: 'web_hunter_parity_dump_v1', model_json_sha256: loaded.sha, initial, trace: trace.rows };
}

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i].startsWith('--')) throw new Error(`unexpected argument ${argv[i]}`);
    args[argv[i].slice(2)] = argv[i + 1];
  }
  if (args['parity-dump']) {
    return { parity: true, modelPath: args.model, spawnsPath: args.spawns, seed: Number(args.seed),
      maxSteps: Number(args['max-steps'] || 300), games: Number(args.games || 64),
      ticks: Number(args.ticks || 120), difficulty: args.difficulty || 'hard', out: args['parity-dump'] };
  }
  for (const key of ['model', 'difficulty', 'spawns', 'start', 'end', 'seed', 'out']) {
    if (args[key] == null) throw new Error(`missing --${key}`);
  }
  return {
    modelPath: args.model, difficulty: args.difficulty, spawnsPath: args.spawns,
    start: Number(args.start), end: Number(args.end), seed: Number(args.seed),
    maxSteps: Number(args['max-steps'] || 300), checkpointSha: args['checkpoint-sha'] || null,
    out: args.out,
  };
}

if (require.main === module) {
  const options = parseArgs(process.argv.slice(2));
  const result = options.parity ? parityDump(options) : runShard(options);
  const tmp = `${options.out}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(result));
  fs.renameSync(tmp, options.out);
  if (!options.parity) {
    console.log(`${options.difficulty} games [${options.start},${options.end}) ${result.elapsed_seconds.toFixed(0)}s`);
  }
}

module.exports = { runShard, runEpisode, createHunter, setupSim, parseArgs, loadLevel, loadModel,
  episodeSeed, parityDump, SCHEMA, DIFFICULTIES };
