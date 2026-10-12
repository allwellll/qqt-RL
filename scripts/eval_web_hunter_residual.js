'use strict';
const assert = require('node:assert/strict');
const fs = require('fs');
const crypto = require('crypto');
const path = require('path');
const authority = require('./eval_web_hunter.js');

function hasHazards(sim) {
  return sim.fuse.some(x => x > 0) || sim.blastLinger.some(x => x > 0) || sim.trapped.some(x => x > 0);
}

function killedByOpponent(info, victim) {
  const killer = 1 - victim;
  const physical = info.physicalDamageSource[victim], causal = info.causalDamageSource[victim];
  return !!info.died[victim] && !info.mutualDeath &&
    ((physical[killer] && Number(physical[0]) + Number(physical[1]) === 1) ||
     (causal[killer] && Number(causal[0]) + Number(causal[1]) === 1));
}

function settle(sim, model, hunter, rng, counts, limit) {
  const started = sim.t;
  sim.done = false;
  sim.maxSteps = started + limit + 1;
  while (hasHazards(sim) && sim.t < started + limit) {
    const raw = model.act(sim, 0, rng);
    const bot = hunter.adapter.act({ schema: 'qqt.bot.observation/v1', tick: sim.t, state: null,
      legal_moves: [0, 1, 2, 3, 4], legal_abilities: [0, 1, 2], metadata: { sim } }, 1, rng);
    const info = sim.step([[Number(raw[0]), 0, 0], [bot.move, 0, 0]]);
    assert.ok(!info.placed[0] && !info.placed[1], 'settlement cannot place new bombs');
    counts.deaths += Number(info.died[0]);
    counts.bot_deaths += Number(info.died[1]);
    counts.mutual_death += Number(info.mutualDeath);
    counts.own_bomb_defeat += Number(info.ownBombDefeat[0]);
    counts.bot_own_bomb_defeat += Number(info.ownBombDefeat[1]);
    counts.killed_by_bot += Number(killedByOpponent(info, 0));
    counts.surviving_causal_kill += Number(info.causalKill[0] && sim.alive[0]);
    counts.surviving_physical_kill += Number(info.creditedKill[0] && sim.alive[0]);
  }
  return { ticks: sim.t - started, hazards_remaining: hasHazards(sim), alive: Array.from(sim.alive) };
}

function runShard(options) {
  const manifest = JSON.parse(fs.readFileSync(options.spawnsPath));
  assert.equal(manifest.seed, options.seed);
  const loaded = authority.loadModel(options.modelPath);
  const level = authority.loadLevel();
  const hunter = authority.createHunter(options.difficulty);
  const started = Date.now();
  const episodes = [];
  for (let game = options.start; game < options.end; game++) {
    let sim, rng;
    // Delegate the entire original 300-tick episode without changing actions,
    // observations, RNG or time normalization; retain its live simulator only.
    const wrapper = Object.create(loaded.model);
    wrapper.act = (current, player, random) => {
      sim = current; rng = random;
      return loaded.model.act(current, player, random);
    };
    const record = authority.runEpisode({ level, model: wrapper, hunter,
      spawn: manifest.spawn_cells[game], game, seed: options.seed, maxSteps: 300 });
    assert.equal(record.ticks, 300);
    const primary = JSON.parse(JSON.stringify(record.counts));
    const residual = settle(sim, loaded.model, hunter, rng, record.counts, 64);
    assert.equal(residual.hazards_remaining, false, 'residual hazards exceed declared bound');
    episodes.push({ ...record, primary_counts: primary, primary_ticks: 300,
      ticks: sim.t, residual, residual_deaths: record.counts.deaths - primary.deaths });
  }
  const root = path.resolve(__dirname, '..');
  const files = ['web/sim.js', 'web/bun_hunter_bot.js', 'web/bot_contract.js',
    'web/assets/maps/levels.json', 'scripts/eval_web_hunter.js', 'scripts/eval_web_hunter_residual.js'];
  return { schema: 'web_hunter_residual_shard_v1', seed: options.seed, game_range: [options.start, options.end],
    primary_steps: 300, residual_limit: 64, new_bombs_during_residual: false,
    source_hashes: Object.fromEntries(files.map(name => [name,
      crypto.createHash('sha256').update(fs.readFileSync(path.join(root, name))).digest('hex')])),
    checkpoint_sha256: options.checkpointSha, spawn_manifest_sha256: manifest.spawn_sha256,
    bot: { id: 'bun.hunter', version: hunter.spec.version, config: hunter.config, resolved_config: hunter.resolved },
    protocol: { map: 806, destructible_bricks_cleared: true, hp: 1, capture_terminates: false,
      native_items: false, native_trap: false, primary_unchanged: true, residual_new_bombs_disabled: true },
    difficulty: options.difficulty, model_json_sha256: loaded.sha, real_web_hunter: true,
    elapsed_seconds: (Date.now() - started) / 1000, episodes };
}

if (require.main === module) {
  const options = authority.parseArgs(process.argv.slice(2));
  const result = runShard(options);
  fs.writeFileSync(`${options.out}.tmp`, JSON.stringify(result));
  fs.renameSync(`${options.out}.tmp`, options.out);
}

module.exports = { hasHazards, settle, runShard };
