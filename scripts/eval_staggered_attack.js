'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const QQT = require('../web/sim');
const Hunter = require('../web/bun_hunter_bot');
const Coop = require('../web/bun_coop_hunter_bot');
const { referenceBot } = require('./eval_bot_cooperation');
const levels = require('../web/assets/maps/levels.json');
const SEEDS = [2026100701, 2026100702, 2026100703, 2026100704];
const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const idle = () => [4, 0, 0, 0];

function scene(seed, direction = 3, team = false, capacity = 3, foeDistance = 6, carrier = false) {
  const sim = new QQT.Sim(seed), teams = team ? [0, 1, 1] : [0, 1];
  sim.reset('open', { nativeItems: true, nativeTrap: true, teams });
  for (const key of ['wall', 'brick', 'crate', 'fuse', 'blastLinger']) sim[key].fill(0);
  sim.isBun = true; sim.bunBases = [[1, 4], [1, 8]];
  sim.bunStored = [[0, 0], [0, 0]]; sim.bunInitial = [Infinity, Infinity];
  sim.bunTarget = 1; sim.bunScore = [0, 0]; sim.maxSteps = 163;
  sim.spdG.fill(1); sim.blastCap.fill(3); sim.bombsCap.fill(capacity); sim.invuln.fill(0);
  const [dy, dx] = DIRS[direction];
  sim.pos[2] = 6.5; sim.pos[3] = 8.5;
  sim.pos[0] = 6.5 + dy * foeDistance; sim.pos[1] = 8.5 + dx * foeDistance;
  if (carrier) sim.bunCarried[0] = 1;
  if (team) { sim.pos[4] = 10.5; sim.pos[5] = direction < 2 ? 14.5 : 4.5; }
  return sim;
}

function trace(sim, bots, candidateTeam, getActions, maxSteps) {
  const candidates = sim.team.map((t, p) => t === candidateTeam ? p : -1).filter(p => p >= 0);
  const frames = [], bubbles = [], active = new Map(), decisions = new Map(), trapOwners = new Map();
  const stats = { bombs: 0, multiBubblePairs: 0, differentExplosionTicks: 0, chainContinuations: 0,
    geometricContacts: 0, pressureContacts: 0, enemyTraps: 0, enemyDeaths: 0, selfTraps: 0, friendlyTraps: 0,
    selfConfinedTicks: 0, teammateBlockedTicks: 0, ownBubbleBlockedTicks: 0, overlapTicks: 0 };
  const resolve = sim._resolveExplosions;
  let predicted = new Map(), explosionSources = [];
  for (let cell = 0; cell < sim.fuse.length; cell++) if (sim.fuse[cell] > 0 && sim.owner[cell] >= 0) {
    const owner = sim.owner[cell], entry = { id: `${owner}:${sim.t}:${cell}`, cell, owner, placedTick: sim.t,
      naturalExplosionTick: sim.t + sim.fuse[cell], expectedExplosionTick: sim.t + sim.fuse[cell], reason: 'fixture',
      anchorCell: null, otherFusesAtPlacement: [], actualExplosionTick: null };
    active.set(cell, entry); bubbles.push(entry);
  }
  sim._resolveExplosions = function (...args) {
    for (let cell = 0; cell < this.fuse.length; cell++) if (this.owner[cell] >= 0 && !active.has(cell)) {
      const owner = this.owner[cell], id = `${owner}:${this.t + 1}:${cell}`;
      const entry = { id, cell, owner, placedTick: this.t + 1, naturalExplosionTick: this.t + Hunter.NEW_BOMB_TICK,
        expectedExplosionTick: predicted.get(cell) ?? this.t + Hunter.NEW_BOMB_TICK,
        reason: decisions.get(owner)?.reason || 'fixture', anchorCell: decisions.get(owner)?.attackTiming?.anchor ?? null,
        otherFusesAtPlacement: Array.from(active.values()).filter(b => b.owner === owner)
          .map(b => ({ id: b.id, remaining: this.fuse[b.cell] })), actualExplosionTick: null };
      active.set(cell, entry); bubbles.push(entry);
    }
    const fuse = this.fuse.slice(), result = resolve.apply(this, args);
    const depth = new Map(result.sources.filter(s => fuse[s.cell] === 0).map(s => [s.cell, 0]));
    for (let round = 0; round < result.sources.length; round++) for (const s of result.sources) {
      if (depth.has(s.cell)) continue;
      const parents = result.sources.filter(p => depth.has(p.cell) && p.covered[s.cell]);
      if (parents.length) depth.set(s.cell, Math.min(...parents.map(p => depth.get(p.cell))) + 1);
    }
    explosionSources = result.sources;
    for (const s of result.sources) {
      const entry = active.get(s.cell); assert(entry, 'actual explosion must have a placement event');
      entry.actualExplosionTick = this.t + 1;
      entry.triggerSource = fuse[s.cell] === 0 ? { kind: 'natural', parents: [] } : { kind: 'chain',
        parents: result.sources.filter(p => depth.get(p.cell) < depth.get(s.cell) && p.covered[s.cell]).map(p => active.get(p.cell).id) };
      entry.causeMask = s.causeMask;
      entry.covered = Array.from(s.covered.entries()).filter(([, v]) => v).map(([c]) => c);
      entry.coveredEnemyContacts = this.team.map((t, p) => t !== this.team[s.owner] && this.alive[p] &&
        this._explosionContactCells(p).some(c => s.covered[c]) ? p : -1).filter(p => p >= 0);
    }
    for (const s of result.sources) active.delete(s.cell);
    return result;
  };
  while (!sim.done && sim.t < maxSteps) {
    const tick = sim.t, wasTrapped = sim.trapped.slice(), hpBefore = sim.hp.slice(), before = Array.from(sim.pos);
    for (let p = 0; p < sim.nPlayers; p++) if (!wasTrapped[p]) trapOwners.delete(p);
    decisions.clear(); const actions = getActions(sim, bots, decisions);
    const state = Hunter.hunterStateFromSim(sim), extras = [];
    for (let p = 0; p < sim.nPlayers; p++) if (actions[p][1]) extras.push({ cell: state.players[p].cell,
      e: Hunter.NEW_BOMB_TICK, blast: state.players[p].blast });
    predicted = new Map();
    if (extras.length) {
      const b = bots[candidates[0]], g = b.geometry(state), prediction = b.predict(state, g, extras, () => true);
      for (const extra of extras) predicted.set(extra.cell, tick + prediction.bombGone[extra.cell]);
    }
    for (const p of candidates) {
      const b = bots[p];
      if (decisions.get(p)?.mode && sim.alive[p] && !sim.trapped[p]) {
        const physical = b.decisionState || state, g = b.geometry(physical), pred = b.predict(physical, g, [], () => true);
        if (!b.escape(g, pred, physical.players[p], false).surv && !b.physicalEscape(physical, g, p, pred)) stats.selfConfinedTicks++;
      }
    }
    explosionSources = []; const info = sim.step(actions);
    const contacts = sim.team.map((_, p) => sim._explosionContactCells(p));
    frames.push({ tick: sim.t, actions, decisions: Array.from(decisions.entries()), before, after: Array.from(sim.pos), contacts });
    for (const source of explosionSources) if (candidates.includes(source.owner)) {
      const bubble = bubbles.find(b => b.cell === source.cell && b.actualExplosionTick === sim.t);
      stats.geometricContacts += bubble.coveredEnemyContacts.length;
      bubble.hitEnemies = bubble.coveredEnemyContacts.filter(p => info.physicalDamageSource[p][source.owner] &&
        ((!wasTrapped[p] && sim.trapped[p]) || hpBefore[p] > sim.hp[p] || info.died[p]));
      stats.pressureContacts += bubble.hitEnemies.length;
    }
    for (let p = 0; p < sim.nPlayers; p++) {
      if (candidates.includes(p)) {
        if (info.placed[p]) stats.bombs++;
        if (!wasTrapped[p] && sim.trapped[p]) {
          if (info.physicalDamageSource[p][p]) stats.selfTraps++;
          if (candidates.some(q => q !== p && info.physicalDamageSource[p][q])) stats.friendlyTraps++;
        }
        if (actions[p][0] !== 4 && sim.alive[p] && !sim.trapped[p] &&
            Math.hypot(sim.pos[p * 2] - before[p * 2], sim.pos[p * 2 + 1] - before[p * 2 + 1]) < 0.02) {
          const [dy, dx] = DIRS[actions[p][0]], cell = state.players[p].cell + dy * QQT.W + dx;
          if (cell >= 0 && cell < sim.fuse.length && sim.fuse[cell] > 0) {
            const owner = sim.owner[cell];
            if (owner === p) stats.ownBubbleBlockedTicks++;
            else if (candidates.includes(owner) && sim.alive[owner] && !sim.trapped[owner]) stats.teammateBlockedTicks++;
          }
        }
        if (candidates.some(q => q > p && sim.alive[q] && Math.hypot(sim.pos[p * 2] - sim.pos[q * 2], sim.pos[p * 2 + 1] - sim.pos[q * 2 + 1]) < 0.75)) stats.overlapTicks++;
      } else {
        if (!wasTrapped[p] && sim.trapped[p]) {
          const owners = candidates.filter(q => info.physicalDamageSource[p][q]);
          trapOwners.set(p, owners);
          if (owners.length) stats.enemyTraps++;
        }
        if (info.died[p] && (trapOwners.get(p)?.length || candidates.some(q => info.physicalDamageSource[p][q]))) stats.enemyDeaths++;
        if (!sim.trapped[p] || info.died[p]) trapOwners.delete(p);
      }
    }
  }
  const own = bubbles.filter(b => candidates.includes(b.owner) && b.actualExplosionTick != null);
  stats.differentExplosionTicks = new Set(own.map(b => b.actualExplosionTick)).size;
  for (const b of own) {
    const ray = new Set(b.covered);
    b.retreatPath = frames.filter(f => f.tick >= b.placedTick && f.tick <= b.actualExplosionTick + Hunter.FLAME_TICKS - 1)
      .map(f => ({ tick: f.tick, move: f.actions[b.owner][0], y: f.after[b.owner * 2], x: f.after[b.owner * 2 + 1], contacts: f.contacts[b.owner] }));
    const exposed = b.retreatPath.filter(p => p.tick <= b.actualExplosionTick && p.contacts.some(c => ray.has(c)));
    b.safetyMarginTicks = b.actualExplosionTick - (exposed.at(-1)?.tick ?? b.placedTick);
    // Native half-contact immunity depends on total coverage in the same tick.
    // Keep the conservative per-ray contact margin as a separate measurement.
    const combined = new Set(bubbles.filter(a => a.actualExplosionTick === b.actualExplosionTick).flatMap(a => a.covered));
    const hitPositions = b.retreatPath.filter(p => p.tick <= b.actualExplosionTick &&
      (sim.nativeTrap ? p.contacts.every(c => combined.has(c)) : p.contacts.some(c => combined.has(c))));
    b.damageSafetyMarginTicks = b.actualExplosionTick - (hitPositions.at(-1)?.tick ?? b.placedTick);
    for (const a of own) if (a.owner === b.owner && a.placedTick < b.placedTick && a.actualExplosionTick >= b.placedTick &&
        a.actualExplosionTick !== b.actualExplosionTick) {
      stats.multiBubblePairs++;
      if (b.otherFusesAtPlacement.some(f => f.id === a.id && f.remaining <= 12) && b.covered.some(c => a.covered.includes(c))) stats.chainContinuations++;
    }
  }
  return { stats, bubbles, frames, outcome: { ticks: sim.t, winner: sim.winner, score: sim.bunScore } };
}

function runScene({ seed = SEEDS[0], direction = 3, team = false, capacity = 3, foeDistance = 6, carrier = false, api = Coop } = {}) {
  const sim = scene(seed, direction, team, capacity, foeDistance, carrier);
  const bot = new api.BunCoopHunterBot({ difficulty: 'hard', seed });
  assert.equal(bot.cfg.maxLiveBombs, 4);
  const bots = sim.team.map((_, p) => p === 1 ? bot : null);
  const result = trace(sim, bots, 1, (s, bs, decisions) => {
    const actions = s.team.map(idle);
    if (s.t >= 72) return actions;
    if (s.t < 12) actions[1] = [s.t < 7 ? direction : 4, s.t === 0 ? 1 : 0, 0, 0];
    else { actions[1] = bot.act(s, 1); decisions.set(1, bot.lastDecision); }
    return actions;
  }, 163);
  return { kind: 'ray-extension', seed, direction, team, capacity, foeDistance, carrier, ...result };
}

function runMap(seed, size, candidateTeam, api) {
  const sim = new QQT.Sim(seed), teams = size === 1 ? [0, 1] : [1 - candidateTeam, candidateTeam, candidateTeam];
  sim.reset(levels.find(l => l.qqt_id === 806), { nativeItems: true, nativeTrap: true, teams }); sim.maxSteps = 451;
  const bots = teams.map((t, p) => new (t === candidateTeam ? api.BunCoopHunterBot : Hunter.BunHunterBot)({ difficulty: 'hard', seed: seed + p * 7919 }));
  const result = trace(sim, bots, candidateTeam, (s, bs, decisions) => {
    // Close the attack window before decisions so cooperative peers never see
    // phantom planned bubbles from an action that the follow-up would strip.
    if (s.t >= 360) s.bombsCap.fill(0);
    const actions = teams.map(idle);
    for (const p of teams.map((_, p) => p).sort((a, b) => teams[a] - teams[b] || a - b)) {
      actions[p] = bs[p].act(s, p); decisions.set(p, bs[p].lastDecision);
      if (s.t >= 360) {
        assert.equal(actions[p][1], 0, 'follow-up must not plan or place another bubble');
        actions[p] = [actions[p][0], 0, 0, 0];
      }
    }
    return actions;
  }, sim.maxSteps);
  return { kind: 'map806', seed, size, candidateTeam, ...result };
}

function summarize(rows) {
  const result = { episodes: rows.length, continuousMultiBubbleEpisodes: rows.filter(r => r.stats.multiBubblePairs > 0).length };
  for (const row of rows) for (const [k, v] of Object.entries(row.stats)) result[k] = (result[k] || 0) + v;
  result.continuousMultiBubbleRate = result.continuousMultiBubbleEpisodes / rows.length;
  return result;
}
function run(api, maps = true) {
  const rows = [];
  for (let i = 0; i < SEEDS.length; i++) for (const team of [false, true]) for (const [capacity, foeDistance, carrier] of [[2, 4, false], [3, 6, false], [2, 4, true]])
    rows.push(runScene({ seed: SEEDS[i], direction: i, team, capacity, foeDistance, carrier, api }));
  if (maps) for (const seed of SEEDS.slice(0, 2)) for (const size of [1, 3]) for (const candidateTeam of [0, 1]) rows.push(runMap(seed, size, candidateTeam, api));
  return { summary: summarize(rows), scenes: summarize(rows.filter(r => r.kind !== 'map806')), maps: summarize(rows.filter(r => r.kind === 'map806')), rows };
}
if (require.main === module) {
  const baselineRef = process.env.BASELINE_REF || '991416fc028e1050efeef94f0b40ae4235bb29e8';
  const baseline = referenceBot(baselineRef), maps = process.env.SCENES_ONLY !== '1';
  const before = run(baseline.api, maps), after = process.env.BASELINE_ONLY === '1' ? null : run(Coop, maps);
  const source = fs.readFileSync(path.resolve(__dirname, '../web/bun_coop_hunter_bot.js'));
  const result = { baselineRef, baselineHash: baseline.sha256, candidateHash: crypto.createHash('sha256').update(source).digest('hex'),
    protocol: { seeds: SEEDS, directions: [0, 1, 2, 3], teams: [false, true], warmupTicks: 12, sceneActiveTicks: 72,
      sceneTrapFollowupTicks: 91, mapSeeds: SEEDS.slice(0, 2), mapActiveTicks: 360, mapMovementOnlyFollowupTicks: 91,
      fuse: 30, botMaxLiveBombs: 4, sceneCapacityDistanceCarrier: [[2, 4, false], [3, 6, false], [2, 4, true]], nativeItems: true, nativeTrap: true }, before, after };
  const out = process.env.EVIDENCE_FILE;
  if (out) { fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, JSON.stringify(result, null, 2)); }
  console.log(JSON.stringify({ before: before.summary, after: after?.summary, scenesBefore: before.scenes, scenesAfter: after?.scenes, mapsBefore: before.maps, mapsAfter: after?.maps }, null, 2));
  if (after && process.env.ACCEPT === '1') {
    assert(after.summary.continuousMultiBubbleRate > before.summary.continuousMultiBubbleRate);
    assert(after.summary.multiBubblePairs > before.summary.multiBubblePairs);
    assert(after.summary.differentExplosionTicks > before.summary.differentExplosionTicks);
    assert(after.summary.chainContinuations > before.summary.chainContinuations);
    assert(after.summary.enemyTraps >= before.summary.enemyTraps && after.summary.enemyDeaths >= before.summary.enemyDeaths &&
      after.summary.pressureContacts >= before.summary.pressureContacts);
    for (const key of ['selfTraps', 'friendlyTraps', 'selfConfinedTicks', 'teammateBlockedTicks', 'overlapTicks'])
      assert(after.summary[key] <= before.summary[key], `${key} regression: ${before.summary[key]} -> ${after.summary[key]}`);
    for (let i = 0; i < before.rows.length; i++) {
      const a = before.rows[i].stats, b = after.rows[i].stats;
      for (const key of ['selfTraps', 'friendlyTraps', 'selfConfinedTicks', 'teammateBlockedTicks', 'overlapTicks'])
        assert(b[key] <= a[key], `row ${i} ${key} safety regression`);
      for (const key of ['enemyTraps', 'enemyDeaths', 'pressureContacts']) assert(b[key] >= a[key], `row ${i} ${key} attack regression`);
    }
  }
}
module.exports = { scene, runScene, runMap, trace, SEEDS };
