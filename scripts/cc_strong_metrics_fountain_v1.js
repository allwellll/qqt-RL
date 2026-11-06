'use strict';
// Offline chain / fountain metrics for strong-eval replays (cc_strong_metrics_fountain_v1).
// Replays every review.json in the frozen production simulator web/sim-bun-native-v2.js, verifies every replay
// hash and the winner, and observes Sim._resolveExplosions (read-only wrapper; no rule, RNG or action change).
// Definitions are in scripts/cc_strong_metrics_fountain_v1.py (DEFINITIONS) and fountain_metrics_v1/DEFINITIONS.md.
// usage: node scripts/cc_strong_metrics_fountain_v1.js CONFIG.json   (CONFIG: {files:[review.json...], output})
const fs = require('node:fs'), crypto = require('node:crypto'), assert = require('node:assert/strict');
const Q = require('./native_eval_simulator_v1'), S = require('../web/five-item-slots-v1');
const { setup } = require('./resume_gpu67_eval_v1');
const W = 15, H = 13, N = W * H, DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]], NEAR = 3, GAP = 24, BATCH_GAP = 8;
const SIM_SHA = Q.SIMULATOR_SHA256;
const hash = x => crypto.createHash('sha256').update(typeof x === 'string' ? x : JSON.stringify(x)).digest('hex');
const act = row => [S.engineAction(row[0], row[1]), S.engineAction(row[2], row[3])];
const cellOf = (s, p) => Math.floor(s.pos[2 * p]) * W + Math.floor(s.pos[2 * p + 1]);
const manhattan = (a, b) => Math.abs(((a / W) | 0) - ((b / W) | 0)) + Math.abs((a % W) - (b % W));

// One bomb's cross exactly as Sim._rays: walls stop uncovered; bricks and bombs (blocking(i)) are covered and stop.
function cross(s, at, blast, blocking) {
  const out = [at], r0 = (at / W) | 0, c0 = at % W;
  for (const [dr, dc] of DIRS) {
    let r = r0, c = c0;
    for (let k = 0; k < blast; k++) {
      r += dr; c += dc; if (r < 0 || r >= H || c < 0 || c >= W) break;
      const i = r * W + c; if (s.wall[i]) break; out.push(i); if (blocking(i) || s.brick[i]) break;
    }
  }
  return out;
}
const blastOf = (s, i) => s.bombBlast[i] > 0 ? s.bombBlast[i] : Q.CFG.blast;

function blank() {
  return { placed: 0, placed_with_own_live: 0, on_own_ray: 0, on_own_ray_other_batch: 0, on_ray_triggered_early: 0,
    explosion_ticks: 0, chain_events: 0, chain_links: 0, chain_len_max: 0, chain_near_enemy: 0, chain_hit_enemy_cell: 0,
    effective: [], effective_strict: [] };
}

function install() {
  const P = Q.Sim.prototype, resolve = P._resolveExplosions;
  P._resolveExplosions = function (...args) {
    const x = this.__ftn; if (!x) return resolve.apply(this, args);
    const t = this.t + 1, fuse = Array.from(this.fuse), owner = Array.from(this.owner);
    const brick = Array.from(this.brick), linger = Array.from(this.brickLinger), alive = Array.from(this.alive);
    const trapped = Array.from(this.trapped), cells = [cellOf(this, 0), cellOf(this, 1)];
    // 1. this tick's placements (fuse decremented before placement, so a fresh bomb has fuse === CFG.fuse)
    for (let p = 0; p < 2; p++) if (this.lastReplayPlaced[p]) {
      for (let i = 0; i < N; i++) if (fuse[i] === Q.CFG.fuse && owner[i] === p && !x.bombs.has(i)) {
        if (t - x.last[p] >= BATCH_GAP) x.batch[p]++;
        x.last[p] = t;
        const me = { batch: x.batch[p], tick: t, owner: p, on_ray: false }, m = x.stats[p];
        // own live bombs other than this one (fuse > 0, or exploding this tick)
        const others = [];
        for (let j = 0; j < N; j++) if (j !== i && owner[j] === p && fuse[j] >= 0 && x.bombs.has(j)) others.push(j);
        m.placed++; if (others.length) m.placed_with_own_live++;
        const block = k => k !== i ? fuse[k] > 0 || (fuse[k] === 0 && owner[k] >= 0) : false;
        const hits = others.filter(j => cross(this, j, blastOf(this, j), k => k !== j && (block(k) || k === i)).includes(i));
        if (hits.length) {
          me.on_ray = true; m.on_own_ray++;
          if (hits.some(j => x.bombs.get(j).batch !== me.batch)) m.on_own_ray_other_batch++;
          x.events.push({ kind: 'on_ray_place', t, player: p, cell: i, batch: me.batch, under: hits });
        }
        x.bombs.set(i, me);
      }
    }
    const r = resolve.apply(this, args), sources = r.sources || [];
    if (!sources.length) return r;
    // 2. chain edges among this tick's sources: a covered target whose fuse was still > 0 was triggered early
    const edges = [];
    for (const a of sources) for (const b of sources) if (a.cell !== b.cell && fuse[b.cell] > 0 && a.covered[b.cell]) edges.push([a.cell, b.cell]);
    for (let p = 0; p < 2; p++) {
      const q = 1 - p, m = x.stats[p], mine = sources.filter(a => owner[a.cell] === p);
      if (!mine.length) continue;
      m.explosion_ticks++;
      const enemyNear = i => alive[q] && manhattan(i, cells[q]) <= NEAR;
      // chain: own timer roots, closure over own targets only
      const roots = mine.filter(a => fuse[a.cell] === 0), seen = new Set(roots.map(a => a.cell));
      for (let changed = true; changed;) { changed = false;
        for (const [a, b] of edges) if (seen.has(a) && owner[b] === p && !seen.has(b)) { seen.add(b); changed = true; } }
      // a bomb placed this very tick has no batch yet in r3_items_reward / R3 (batch_update runs after the step)
      const batchOf = i => { const b = x.bombs.get(i); return b && b.tick < t ? b.batch : undefined; };
      const cross_links = edges.filter(([a, b]) => seen.has(a) && seen.has(b) && fuse[b] > 0 && batchOf(a) !== undefined && batchOf(b) !== undefined && batchOf(a) !== batchOf(b));
      const crossed = new Set(cross_links.map(([, b]) => batchOf(b)));
      for (const b of seen) if (fuse[b] > 0 && x.bombs.get(b)?.on_ray) m.on_ray_triggered_early++;
      if (crossed.size) {
        const batches = new Set([...seen].map(batchOf).filter(v => v !== undefined));
        const fire = new Set(); for (const a of sources) if (seen.has(a.cell)) for (let i = 0; i < N; i++) if (a.covered[i]) fire.add(i);
        const near = [...fire].some(enemyNear), onEnemy = alive[q] && fire.has(cells[q]);
        m.chain_events++; m.chain_links += crossed.size; m.chain_len_max = Math.max(m.chain_len_max, batches.size);
        m.chain_near_enemy += Number(near); m.chain_hit_enemy_cell += Number(onEnemy);
        x.events.push({ kind: 'chain', t, player: p, batches: batches.size, cross_batches: crossed.size, near_enemy: near,
          roots: roots.map(a => a.cell), triggered: [...seen].filter(i => fuse[i] > 0) });
      }
      // effective explosion batch: valid own explosion that covers a novel cell within 3 of the live enemy,
      // a fresh brick, or the enemy's cell
      if (!alive[p] || trapped[p] > 0 || !alive[q]) continue;
      const own = sources.filter(a => owner[a.cell] === p && (a.causeMask & (1 << p))), enemyFire = new Uint8Array(N);
      for (const a of sources) if (a.causeMask & (1 << q)) for (let i = 0; i < N; i++) if (a.covered[i]) enemyFire[i] = 1;
      if (!own.length) continue;
      const owned = new Uint8Array(N); for (const a of own) for (let i = 0; i < N; i++) if (a.covered[i] && !enemyFire[i]) owned[i] = 1;
      // novelty: the cell lies on exactly one own live cross (blockers: bombs alive before this tick's chain)
      const count = new Uint8Array(N);
      for (let j = 0; j < N; j++) if (owner[j] === p && fuse[j] >= 0 && x.bombs.has(j))
        for (const i of cross(this, j, blastOf(this, j), k => k !== j && fuse[k] > 0)) count[i]++;
      let novelNear = false, freshBrick = false, hitEnemy = false;
      for (let i = 0; i < N; i++) if (owned[i]) {
        if (count[i] === 1 && enemyNear(i)) novelNear = true;
        if (brick[i] && linger[i] <= 0) freshBrick = true;
        if (i === cells[q]) hitEnemy = true;
      }
      const sb = Math.min(...own.map(a => batchOf(a.cell) ?? 1e9));
      const rec = (list, key) => { if (x.lastEff[key][p] === sb) return false; x.lastEff[key][p] = sb; list.push(t); return true; };
      if ((novelNear || freshBrick || hitEnemy) && rec(m.effective, 'loose'))
        x.events.push({ kind: 'effective', t, player: p, batch: sb, novel_near: novelNear, brick: freshBrick, enemy_cell: hitEnemy });
      if (novelNear && (freshBrick || hitEnemy || [...Array(N).keys()].some(i => owned[i] && (this.crate[i] > 0 || (this.tacticalCrate && this.tacticalCrate[i] > 0)))))
        rec(m.effective_strict, 'strict');
    }
    for (const a of sources) x.bombs.delete(a.cell);
    return r;
  };
}

function streaks(ticks) {
  const out = []; let cur = null;
  for (const t of ticks) { if (cur && t - cur.end <= GAP) { cur.end = t; cur.n++; } else { cur = { start: t, end: t, n: 1 }; out.push(cur); } }
  const long = out.filter(s => s.n >= 2);
  return { effective: ticks.length, streak_max: out.reduce((a, s) => Math.max(a, s.n), 0),
    ticks: long.reduce((a, s) => a + s.end - s.start + 1, 0), streaks_ge2: long.length, streaks_ge3: out.filter(s => s.n >= 3).length, list: long };
}

function replay(file) {
  const review = JSON.parse(fs.readFileSync(file)), seat = review.meta.learner_seat, rows = review.actions;
  assert.equal(review.meta.simulator_sha256, SIM_SHA, 'simulator sha (set QQT_EVAL_SIMULATOR=legacy for archived matches)');
  const s = setup(review.meta.game, review.level), checks = new Map(review.checks);
  assert.equal(hash(s.snapshotReplay()), checks.get(0), 'initial hash');
  s.__ftn = { bombs: new Map(), batch: [0, 0], last: [-1000, -1000], stats: [blank(), blank()], events: [],
    lastEff: { loose: [-1, -1], strict: [-1, -1] } };
  let verified = 1;
  for (let k = 0; k < rows.length; k++) {
    s.step(act(rows[k]));
    if (checks.has(s.t)) { assert.equal(hash(s.snapshotReplay()), checks.get(s.t), `replay hash @${s.t}`); verified++; }
  }
  assert(s.done && s.winner === review.summary.winner, 'terminal mismatch');
  const x = s.__ftn, players = x.stats.map(m => {
    const f = streaks(m.effective), g = streaks(m.effective_strict), { effective, effective_strict, ...rest } = m;
    return { ...rest, effective_batches: f.effective, fountain_streak: f.streak_max, fountain_ticks: f.ticks,
      fountain_streaks_ge2: f.streaks_ge2, fountain_streaks_ge3: f.streaks_ge3,
      strict_effective_batches: g.effective, strict_fountain_streak: g.streak_max, strict_fountain_ticks: g.ticks, _streaks: f.list };
  });
  const timeline = x.events.slice();
  players.forEach((m, p) => { for (const z of m._streaks) timeline.push({ kind: 'fountain', t: z.start, end: z.end, player: p, batches: z.n }); delete m._streaks; });
  timeline.sort((a, b) => a.t - b.t);
  // cross-check against the eval worker's own R3 chain record (batch_count >= 2 == cross-batch chain tick)
  let r3_check = null; const res = file.replace(/review\.json$/, 'RESULT.json');
  if (fs.existsSync(res)) {
    const chains = JSON.parse(fs.readFileSync(res)).r3?.chains;
    if (chains) r3_check = [0, 1].every(p => { const c = chains.filter(z => z.player === p && z.batch_count >= 2);
      return c.length === players[p].chain_events && Math.max(0, ...c.map(z => z.batch_count)) === players[p].chain_len_max; });
  }
  return { r3_chain_check: r3_check, file, review_sha256: hash(fs.readFileSync(file, 'utf8')), label: review.meta.model, opponent: review.meta.opponent, seat,
    mode: review.meta.game.mode, opponent_mode: review.meta.game.opponent_mode, seed: review.meta.seed, ticks: rows.length,
    winner: review.summary.winner, hash_checks_verified: verified, players, timeline };
}

function main() {
  const cfg = JSON.parse(fs.readFileSync(process.argv[2])); install();
  const out = []; let errors = 0;
  for (const f of cfg.files) {
    try { out.push(JSON.stringify(replay(f))); }
    catch (e) { errors++; out.push(JSON.stringify({ file: f, error: String(e && e.stack || e).slice(0, 2000) })); }
  }
  fs.writeFileSync(cfg.output + '.tmp', out.join('\n') + '\n'); fs.renameSync(cfg.output + '.tmp', cfg.output);
  console.log(`FOUNTAIN_WORKER files=${cfg.files.length} errors=${errors}`);
}
if (require.main === module) main();
module.exports = { install, replay, streaks };
