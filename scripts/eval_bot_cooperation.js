#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const Module = require('module');
const QQT = require('../web/sim.js');
const Hunter = require('../web/bun_hunter_bot.js');
const Coop = require('../web/bun_coop_hunter_bot.js');
const levels = require('../web/assets/maps/levels.json');
const ROOT = path.resolve(__dirname, '..');

function trackItemEvents(sim, stats) {
  const place = sim._placeHeldItem;
  sim._placeHeldItem = function(player, cell, slotIndex) {
    const slot = this.itemSlots[player][slotIndex || 0];
    const item = slot ? slot.item : this.heldItem[player];
    const placed = place.call(this, player, cell, slotIndex);
    if (placed) stats[this.team[player]][item === QQT.ITEM_BANANA ? 'bananas' : 'glue']++;
    return placed;
  };
  const update = sim._updateFieldItems;
  sim._updateFieldItems = function(actions) {
    const items = [];
    for (let cell = 0; cell < this.fieldItem.length; cell++) {
      if (!this.fieldItem[cell]) continue;
      const player = this.team.findIndex((_, p) => this.alive[p] && !this.trapped[p] &&
        this.centerCell(p)[0] * (this.W || QQT.W) + this.centerCell(p)[1] === cell);
      if (player >= 0) items.push({ cell, player, item: this.fieldItem[cell], owner: this.fieldOwner[cell] });
    }
    update.call(this, actions);
    for (const event of items) {
      if (this.fieldItem[event.cell]) continue;
      const team = this.team[event.player], ownerTeam = this.team[event.owner];
      if (event.item === QQT.ITEM_BANANA && this.bunCarried[event.player] >= 0) stats[team].carrierSlides++;
      if (event.item === QQT.ITEM_SLOW_GLUE && ownerTeam != null) {
        stats[ownerTeam][ownerTeam === team ? 'friendlySlows' : 'enemySlows']++;
      }
    }
  };
}

function referenceBot(ref) {
  if (!/^[0-9a-f]{7,40}$/.test(ref)) throw new Error('reference must be a commit hash');
  const source = execFileSync('git', ['show', `${ref}:web/bun_coop_hunter_bot.js`], { cwd: ROOT, encoding: 'utf8' });
  const filename = path.join(ROOT, 'web', 'reference_coop_hunter_bot.js');
  const snapshot = new Module(filename, module);
  snapshot.filename = filename;
  snapshot.paths = Module._nodeModulePaths(path.dirname(filename));
  snapshot._compile(source, filename);
  return { api: snapshot.exports, sha256: crypto.createHash('sha256').update(source).digest('hex'), ref };
}

function episode(seed, size, candidateTeam, maxSteps, clearBricks = false, strategy = 'coop', reference = null) {
  const teams = size === 1 ? [0, 1] : size === 3 ? [1 - candidateTeam, candidateTeam, candidateTeam] : [0, 1, 0, 1];
  const sim = new QQT.Sim(seed);
  sim.reset(levels.find((l) => l.qqt_id === 806), { nativeItems: true, nativeTrap: true, teams });
  sim.maxSteps = maxSteps;
  if (clearBricks) { sim.brick.fill(0); sim.brickLinger.fill(0); }
  const candidateBot = strategy === 'coop' ? Coop.BunCoopHunterBot
    : strategy === 'previous' ? reference.api.BunCoopHunterBot : Hunter.BunHunterBot;
  const bots = teams.map((team, pid) => new (team === candidateTeam ? candidateBot : Hunter.BunHunterBot)(
    { difficulty: 'hard', seed: (seed + pid * 7919) >>> 0 }));
  const stats = [0, 1].map(() => ({ bombs: 0, threateningBombs: 0, deaths: 0,
    trapped: 0, selfTraps: 0, friendlyTraps: 0, rescues: 0, carrierTicks: 0, escortTicks: 0,
    bananas: 0, glue: 0, carrierSlides: 0, enemySlows: 0, friendlySlows: 0, overlapTicks: 0, modes: {} }));
  for (const s of stats) Object.assign(s, { attackIntentBombs: 0, earlyChainBombs: 0, crates: 0,
    capacityUpgrades: 0, damageUpgrades: 0, speedUpgrades: 0, multiReserves: 0, reserveSeconds: 0, connectors: 0, screens: 0 });
  trackItemEvents(sim, stats);
  const collect = sim._collectCrate;
  sim._collectCrate = function(player, cell) {
    const before = [this.bombsCap[player], this.blastCap[player], this.spdG[player]];
    const picked = collect.call(this, player, cell);
    if (picked) {
      const s = stats[this.team[player]];
      s.crates++;
      if (this.bombsCap[player] > before[0]) s.capacityUpgrades++;
      if (this.blastCap[player] > before[1]) s.damageUpgrades++;
      if (this.spdG[player] > before[2]) s.speedUpgrades++;
    }
    return picked;
  };
  const spawns = teams.map((_, p) => Array.from(sim.pos.slice(p * 2, p * 2 + 2)));
  const ordered = teams.map((_, p) => p).sort((a, b) => teams[a] - teams[b] || a - b);
  while (!sim.done) {
    const wasTrapped = sim.trapped.slice();
    const actions = teams.map(() => [4, 0, 0, 0]);
    for (const p of ordered) {
      const raw = bots[p].act(sim, p);
      actions[p] = raw;
      const mode = bots[p].lastDecision.mode;
      const s = stats[teams[p]];
      s.modes[mode] = (s.modes[mode] || 0) + 1;
      if (mode === 'ESCORT') s.escortTicks++;
      if (sim.bunCarried[p] >= 0 && sim.alive[p]) s.carrierTicks++;
      if (teams.some((team, q) => q > p && team === teams[p] && sim.alive[p] && sim.alive[q] &&
          Math.hypot(sim.pos[p * 2] - sim.pos[q * 2], sim.pos[p * 2 + 1] - sim.pos[q * 2 + 1]) < 0.75)) s.overlapTicks++;
    }
    const state = Hunter.hunterStateFromSim(sim);
    const threat = teams.map((_, p) => {
      if (!actions[p][1]) return false;
      const b = bots[p], g = b.geometry(state), me = state.players[p];
      const cells = new Set([me.cell]);
      for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        for (let k = 1; k <= me.blast; k++) {
          const r = me.row + dy * k, c = me.col + dx * k;
          if (r < 0 || r >= g.H || c < 0 || c >= g.W || state.wall[r * g.W + c]) break;
          const cell = r * g.W + c;
          cells.add(cell);
          if (state.brick[cell] || g.bombAt[cell] >= 0) break;
        }
      }
      return b.enemiesOf(state, p).some((q) => state.players[q].alive && cells.has(state.players[q].cell));
    });
    const fuseBefore = sim.fuse.slice(), ownersBefore = sim.owner.slice();
    const info = sim.step(actions);
    for (let c = 0; c < sim.fuse.length; c++) if (info.triggered[c] && fuseBefore[c] > 1 && ownersBefore[c] >= 0) {
      stats[teams[ownersBefore[c]]].earlyChainBombs++;
    }
    for (let p = 0; p < teams.length; p++) {
      const s = stats[teams[p]];
      if (info.placed[p]) {
        s.bombs++; if (threat[p]) s.threateningBombs++;
        const reason = bots[p].lastDecision.reason;
        if (reason !== 'bomb_dig') s.attackIntentBombs++;
        if (reason === 'bomb_reserve_multi') s.multiReserves++;
        if (reason === 'bomb_reserve_second') s.reserveSeconds++;
        if (reason === 'bomb_chain') s.connectors++;
        if (reason === 'bomb_screen' || reason === 'bomb_block') s.screens++;
      }
      if (info.died[p]) s.deaths++;
      if (!wasTrapped[p] && sim.trapped[p]) {
        s.trapped++;
        if (info.physicalDamageSource[p][p]) s.selfTraps++;
        if (teams.some((team, q) => q !== p && team === teams[p] && info.physicalDamageSource[p][q])) s.friendlyTraps++;
      }
      if (wasTrapped[p] && !sim.trapped[p] && sim.alive[p]) s.rescues++;
    }
  }
  return { seed, size, strategy, clearBricks, candidateTeam, spawns, winner: sim.winner, ticks: sim.t,
    candidate: stats[candidateTeam], baseline: stats[1 - candidateTeam], score: sim.bunScore.slice(),
    outcome: sim.winner === candidateTeam ? 'win' : sim.winner === 1 - candidateTeam ? 'loss' : 'draw' };
}

function summarize(rows) {
  const summary = { games: rows.length, wins: 0, losses: 0, draws: 0, candidate: {}, baseline: {} };
  for (const row of rows) {
    summary[row.outcome === 'win' ? 'wins' : row.outcome === 'loss' ? 'losses' : 'draws']++;
    for (const side of ['candidate', 'baseline']) {
      for (const [key, value] of Object.entries(row[side])) {
        if (typeof value === 'number') summary[side][key] = (summary[side][key] || 0) + value;
      }
    }
  }
  return summary;
}

function pairedInterval(rows) {
  const groups = new Map();
  for (const row of rows) {
    const value = row.outcome === 'win' ? 1 : row.outcome === 'loss' ? -1 : 0;
    if (!groups.has(row.seed)) groups.set(row.seed, []);
    groups.get(row.seed).push(value);
  }
  const values = [...groups.values()].map((g) => g.reduce((a, b) => a + b, 0) / g.length);
  if (!values.length) return null;
  const rng = QQT.mulberry32(761034);
  const samples = [];
  for (let i = 0; i < 10000; i++) {
    let sum = 0;
    for (let j = 0; j < values.length; j++) sum += values[Math.floor(rng() * values.length)];
    samples.push(sum / values.length);
  }
  samples.sort((a, b) => a - b);
  return { method: 'seed-cluster-bootstrap', pairs: values.length,
    meanNetWin: values.reduce((a, b) => a + b, 0) / values.length,
    lower95: samples[250], upper95: samples[9749] };
}

function run({ pairs = 8, seed = 2026100400, maxSteps = 1200, sizes = [1, 2, 3], referenceRef = null } = {}) {
  const rows = [];
  const started = performance.now();
  const hashes = {};
  const reference = referenceRef ? referenceBot(referenceRef) : null;
  for (const name of ['web/sim.js', 'web/bun_hunter_bot.js', 'web/bun_coop_hunter_bot.js',
    'web/assets/maps/levels.json', 'scripts/eval_bot_cooperation.js']) {
    hashes[name] = crypto.createHash('sha256').update(fs.readFileSync(path.join(ROOT, name))).digest('hex');
  }
  for (const size of sizes) {
    for (let i = 0; i < pairs; i++) {
      for (const candidateTeam of [0, 1]) {
        rows.push(episode(seed + i * 7919, size, candidateTeam, maxSteps));
        if (reference) rows.push(episode(seed + i * 7919, size, candidateTeam, maxSteps, false, 'previous', reference));
        if (size === 3) rows.push(episode(seed + i * 7919, size, candidateTeam, maxSteps, false, 'legacy'));
      }
        process.stderr.write(`size=${size} pair=${i + 1}/${pairs} ${JSON.stringify(summarize(rows.filter((r) => r.size === size && r.strategy === 'coop')))}\n`);
    }
  }
  return { schema: 'bot_cooperation_eval/v5', protocol: { pairs, seed, maxSteps, sizes, map: 806,
    nativeItems: true, nativeTrap: true, difficulty: 'hard', candidate: 'bun.coop_hunter', baseline: 'bun.hunter' },
    reference: reference ? { ref: reference.ref, sha256: reference.sha256, baseline: 'bun.hunter' } : null,
    hashes, elapsedSeconds: (performance.now() - started) / 1000,
    solo: summarize(rows.filter((r) => r.size === 1 && r.strategy === 'coop')),
    team: summarize(rows.filter((r) => r.size === 2 && r.strategy === 'coop')),
    previousSolo: summarize(rows.filter((r) => r.size === 1 && r.strategy === 'previous')),
    previousTeam: summarize(rows.filter((r) => r.size === 2 && r.strategy === 'previous')),
    previousDuoVsSolo: summarize(rows.filter((r) => r.size === 3 && r.strategy === 'previous')),
    duoVsSolo: summarize(rows.filter((r) => r.size === 3 && r.strategy === 'coop')),
    legacyDuoVsSolo: summarize(rows.filter((r) => r.size === 3 && r.strategy === 'legacy')),
    soloInterval: pairedInterval(rows.filter((r) => r.size === 1 && r.strategy === 'coop')),
    teamInterval: pairedInterval(rows.filter((r) => r.size === 2 && r.strategy === 'coop')), episodes: rows };
}

if (require.main === module) {
  const options = {};
  let output = null;
  for (let i = 2; i < process.argv.length; i += 2) {
    const flag = process.argv[i], value = process.argv[i + 1];
    if (flag === '--out') output = value;
    else if (flag === '--reference-ref') options.referenceRef = value;
    else if (flag === '--sizes') {
      options.sizes = value.split(',').map(Number);
      if (!options.sizes.length || options.sizes.some((n) => ![1, 2, 3].includes(n))) throw new Error('invalid --sizes');
    }
    else if (['--pairs', '--seed', '--max-steps'].includes(flag)) {
      const n = Number(value);
      if (!Number.isSafeInteger(n) || n <= 0) throw new Error(`invalid ${flag}`);
      options[flag === '--max-steps' ? 'maxSteps' : flag.slice(2)] = n;
    } else throw new Error(`unknown option ${flag}`);
  }
  const result = run(options);
  if (output) { fs.mkdirSync(path.dirname(output), { recursive: true }); fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n'); }
  console.log(JSON.stringify({ solo: result.solo, team: result.team, previousSolo: result.previousSolo,
    previousTeam: result.previousTeam, previousDuoVsSolo: result.previousDuoVsSolo, duoVsSolo: result.duoVsSolo,
    legacyDuoVsSolo: result.legacyDuoVsSolo, elapsedSeconds: result.elapsedSeconds }));
}
module.exports = { episode, summarize, pairedInterval, run, trackItemEvents, referenceBot };
