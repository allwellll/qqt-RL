'use strict';

(function coopHunterFactory(root, factory) {
  const base = typeof module === 'object' && module.exports ? require('./bun_hunter_bot.js') : root.QQTBunHunterBot;
  const api = factory(base);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTBunCoopHunterBot = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildCoopHunter(Base) {
  const INF = 1e9;
  const commitments = new WeakMap();
  const distance = (a, b) => Math.abs(a.row - b.row) + Math.abs(a.col - b.col);

  class BunCoopHunterBot extends Base.BunHunterBot {
    constructor(options = {}) {
      super(options);
      this.cfg.maxLiveBombs = Math.min(this.cfg.maxLiveBombs, 4);
    }

    reset(seed) {
      super.reset(seed);
      this.assignment = null;
      this.lastGoalMode = null;
      this.chainPlan = null;
      this.enemyTrends = new Map();
      this.trendTick = -1;
      this.decisionState = null;
      this.resourceTarget = -1;
      this.itemPlan = null;
    }

    analyzeSim(sim, pid) {
      const state = Base.hunterStateFromSim(sim);
      state.itemSlots = sim.nativeItems ? sim.itemSlots : state.heldItem.map((item) => item ? [{ item, count: 1 }] : []);
      state.fieldOwner = sim.fieldOwner;
      state.movementStatus = sim.movementStatus;
      let pending = commitments.get(sim);
      if (!pending || pending.tick !== sim.t || pending.generation !== sim._gen) {
        pending = { tick: sim.t, generation: sim._gen, bombs: [], moves: {}, tasks: {} };
        commitments.set(sim, pending);
      }
      const team = state.players[pid].team;
      // Later teammates see bombs committed earlier in this tick before choosing their escape.
      state.bombs = state.bombs.concat(pending.bombs.filter((b) => state.players[b.owner].team === team));
      state.committedMoves = pending.moves;
      state.committedTasks = pending.tasks;
      const blocked = Uint8Array.from(sim.wall, (wall, c) => wall || sim.brick[c] || sim.fuse[c] > 0 ? 1 : 0);
      for (const b of state.bombs) blocked[b.cell] = 1;
      state.nextPosition = (q, move) => {
        const p = state.players[q], direction = sim.playerMoveDirection(q, move);
        const step = this.stepLen(p, false);
        return sim.movementStatus[q] === 2 ? sim._tryMove(p.y, p.x, direction, blocked, step)
          : sim._steer(p.y, p.x, direction, blocked, step, q);
      };
      const decision = this.analyze(state, pid);
      pending.moves[pid] = decision.action[0];
      pending.tasks[pid] = { mode: decision.mode, cell: this.assignment && this.assignment.cell };
      if (decision.action[1] === 1) {
        const me = state.players[pid];
        if (!pending.bombs.some((b) => b.owner === pid)) pending.bombs.push({
          cell: me.cell, row: me.row, col: me.col, owner: pid, fuse: Base.NEW_BOMB_TICK, blast: me.blast,
        });
      }
      return decision;
    }

    act(sim, pid = 1) {
      const decision = this.analyzeSim(sim, pid);
      return [decision.action[0], decision.action[1] === 1 ? 1 : 0, decision.action[1] === 2 ? 1 : 0,
        0, -1, -1, decision.itemSlot || 0];
    }

    teamRole(state, pid) {
      const active = [pid, ...this.alliesOf(state, pid)].filter((q) => {
        const p = state.players[q];
        return p.alive && !p.trapped;
      }).sort((a, b) => a - b);
      return active.length <= 1 ? 'SOLO' : active[0] === pid ? 'ATTACKER' : 'SUPPORT';
    }

    foeOf(state, pid) {
      const me = state.players[pid], team = me.team == null ? pid : me.team;
      const carrier = this.alliesOf(state, pid).find((q) => state.players[q].alive && state.players[q].carrying >= 0);
      const protectedPlayer = carrier == null ? me : state.players[carrier];
      const enemies = this.enemiesOf(state, pid);
      let best = enemies[0], score = INF;
      for (const q of enemies) {
        const e = state.players[q];
        if (!e.alive || e.trapped) continue;
        const value = distance(protectedPlayer, e) - (e.carrying === team ? 20 : 0);
        if (value < score) { best = q; score = value; }
      }
      return state.players[best];
    }

    chooseGoal(state, g, pid, pred, perceive) {
      let goal = this.selectGoal(state, g, pid, pred, perceive);
      const plan = this.chainPlan;
      if (plan) {
        const ray = this.blastCells(state, g, plan.cell, state.players[pid].blast);
        const anchor = state.bombs.find((b) => b.cell === plan.anchor && b.owner === pid);
        if (state.tick >= plan.expires || !g.open[plan.cell] || g.bombAt[plan.cell] >= 0 ||
            !anchor ||
            !this.enemiesOf(state, pid).some((q) => state.players[q].alive &&
              !state.players[q].trapped && this.enemyPathCells(state, g, q, anchor.fuse)
                .some((cell) => ray.has(cell)))) this.chainPlan = null;
      }
      if (this.chainPlan && ['HUNT', 'SUPPORT', 'RAID'].includes(goal.mode)) {
        goal = { mode: 'CHAIN', seeds: [{ cell: plan.cell, cost: 0 }], threat: false, noDig: true };
      }
      this.lastGoalMode = goal.mode;
      return goal;
    }

    fieldForGoal(state, g, goal, me, pred, foe) {
      if (me.carrying >= 0 && state.fieldOwner) {
        const fieldItem = state.fieldItem.slice();
        for (let c = 0; c < g.N; c++) if (fieldItem[c] === 1 &&
            state.players[state.fieldOwner[c]] && state.players[state.fieldOwner[c]].team === me.team) fieldItem[c] = 0;
        state = Object.assign({}, state, { fieldItem });
      }
      return this.goalField(state, g, goal.seeds, me, pred, goal.threat ? foe : null,
        goal.noDig == null ? me.carrying >= 0 : goal.noDig);
    }

    selectGoal(state, g, pid, pred, perceive) {
      const me = state.players[pid], team = me.team == null ? pid : me.team;
      const allies = this.alliesOf(state, pid);
      const role = this.teamRole(state, pid);
      this.assignment = { role, target: null };
      this.itemPlan = null;
      const rescue = this.trapGoal(state, g, pid, pred, allies, 'RESCUE');
      if (rescue && (me.carrying < 0 || rescue.steps <= 4)) {
        this.assignment.target = rescue.target;
        return rescue;
      }
      if (me.carrying >= 0) return super.chooseGoal(state, g, pid, pred, perceive);
      const home = this.baseCells(state, g, team).map((cell) => ({ cell, cost: 0 }));
      const carrier = allies.find((q) => state.players[q].alive && !state.players[q].trapped && state.players[q].carrying >= 0);
      if (carrier != null) {
        this.assignment.target = carrier;
        const ally = state.players[carrier], foe = this.foeOf(state, pid);
        this.itemPlan = this.escortItemPlan(state, g, pid, carrier, pred);
        if (this.itemPlan) {
          this.assignment.cell = this.itemPlan.cell;
          return { mode: 'ESCORT', seeds: [{ cell: this.itemPlan.cell, cost: 0 }], threat: false, noDig: true };
        }
        // Stay between the carrier and the nearest pursuer without chasing beyond the escort radius.
        const seeds = [];
        const delivery = this.routeCells(state, g, ally, this.baseCells(state, g, team), pred);
        for (let cell = 0; cell < g.N; cell++) {
          if (!g.open[cell] || g.bombAt[cell] >= 0) continue;
          const point = { row: Math.floor(cell / g.W), col: cell % g.W };
          const d = distance(point, ally);
          if (d < 2 || d > 3 || delivery.includes(cell)) continue;
          seeds.push({ cell, cost: foe && foe.alive && !foe.trapped ? distance(point, foe) * 0.4 : d * 0.2 });
        }
        return { mode: 'ESCORT', seeds: seeds.length ? seeds : [{ cell: ally.cell, cost: 0 }], threat: false, noDig: true };
      }
      const thief = this.enemiesOf(state, pid).find((q) => state.players[q].alive && state.players[q].carrying === team);
      if (thief != null) return { mode: 'INTERCEPT', seeds: this.attackSeeds(state, g, pid, pred, perceive), threat: false };
      if (role !== 'SUPPORT') {
        const goal = super.chooseGoal(state, g, pid, pred, perceive);
        if (role === 'ATTACKER' && goal.mode === 'POP') {
          const partner = allies.find((q) => state.players[q].alive && !state.players[q].trapped && state.players[q].carrying < 0);
          const partnerPop = partner == null ? null : this.trapGoal(state, g, partner, pred, [goal.target], 'POP');
          const suppressed = this.enemiesOf(state, pid).every((q) => !state.players[q].alive || state.players[q].trapped);
          if (partnerPop && partnerPop.steps <= goal.steps && suppressed) {
            const stock = state.bunStored[1 - team];
            const homeOpen = this.goalField(state, g, home, me, pred, null, true)[me.cell] < INF;
            if (stock && stock[1 - team] > 0 && homeOpen) return { mode: 'RAID',
              seeds: this.baseCells(state, g, 1 - team).map((cell) => ({ cell, cost: 0 })), threat: false };
          }
        }
        if (role === 'ATTACKER' && goal.mode === 'RECOVER') {
          return { mode: 'HUNT', seeds: this.attackSeeds(state, g, pid, pred, perceive), threat: false };
        }
        return goal;
      }
      const looseOwn = [];
      for (let cell = 0; cell < g.N; cell++) if (state.bunLoose[cell * 2 + team]) looseOwn.push({ cell, cost: 0 });
      if (looseOwn.length) return { mode: 'RECOVER', seeds: looseOwn, threat: true };
      const pop = this.trapGoal(state, g, pid, pred, this.enemiesOf(state, pid), 'POP');
      if (pop) return pop;
      const foe = this.foeOf(state, pid);
      if (foe && foe.alive && !foe.trapped) {
        const enemyHome = this.goalField(state, g, home, foe, pred, null, true)[foe.cell];
        if (enemyHome <= 4) return { mode: 'DEFEND', seeds: this.attackSeeds(state, g, pid, pred, perceive), threat: false };
      }
      const opportunity = super.chooseGoal(state, g, pid, pred, perceive);
      if (['OPEN_ROUTE', 'STEAL'].includes(opportunity.mode)) return opportunity;
      const resource = this.resourceGoal(state, g, pid, pred);
      if (resource) return resource;
      const guards = this.defenseSeeds(state, g, pid);
      if (guards.length && foe && foe.alive && !foe.trapped) {
        return { mode: 'GUARD', seeds: guards, threat: false };
      }
      return Object.assign({}, opportunity, { mode: opportunity.mode === 'HUNT' ? 'SUPPORT' : opportunity.mode });
    }

    attackSeeds(state, g, pid, pred, perceive) {
      const seeds = super.attackSeeds(state, g, pid, pred, perceive);
      const foe = this.foeOf(state, pid);
      const allies = this.alliesOf(state, pid).map((q) => state.players[q]).filter((p) => p.alive && !p.trapped);
      // Prefer close firing positions to distant cells on the same ray.
      return seeds.map((s) => {
        const point = { row: Math.floor(s.cell / g.W), col: s.cell % g.W };
        const spacing = allies.reduce((cost, ally) => cost + Math.max(0, 3 - distance(point, ally)) * 2.5, 0);
        return { cell: s.cell, cost: s.cost + Math.max(0, distance(point, foe) - 2) * 0.7 + spacing };
      });
    }

    threatMap(state, g) {
      const combined = new Uint8Array(g.N);
      const pid = this.currentPid;
      if (pid == null) return combined;
      for (const q of this.enemiesOf(state, pid)) {
        const e = state.players[q];
        if (e.trapped || e.carrying >= 0 || e.bombsLeft <= 0) continue;
        const map = super.threatMap(state, g, e);
        for (let cell = 0; cell < g.N; cell++) combined[cell] |= map[cell];
      }
      return combined;
    }

    analyze(state, pid) {
      this.currentPid = pid;
      this.decisionState = state;
      this.assignment = null;
      this.observeEnemyTrends(state, pid);
      const decision = super.analyze(state, pid);
      const plan = this.itemPlan, me = state.players[pid];
      if (plan && me.alive && !me.trapped && me.cell === plan.cell && decision.action[1] === 0) {
        const g = this.geometry(state), pred = this.predict(state, g, [], () => true);
        const next = state.nextPosition && state.nextPosition(pid, decision.action[0]);
        // An item arms only after its owner leaves: clear the tile before the carrier arrives.
        if (next && Math.floor(next[0]) * g.W + Math.floor(next[1]) !== me.cell &&
            !this.threatened(pred, me, g.N)) {
          decision.action[1] = 2;
          decision.itemSlot = plan.slot;
          decision.reason = plan.item === 1 ? 'escort_banana' : 'approach_glue';
        }
      }
      return Object.assign(decision, this.assignment || { role: 'SOLO', target: null });
    }

    resourceGoal(state, g, pid, pred) {
      const me = state.players[pid], home = this.baseCells(state, g, me.team);
      const allies = this.alliesOf(state, pid).filter((q) => state.players[q].alive && !state.players[q].trapped);
      const claimed = (cell) => allies.some((q) => state.committedTasks && state.committedTasks[q] &&
        state.committedTasks[q].cell === cell);
      const candidates = [];
      const openTravel = this.goalField(state, g, [{ cell: me.cell, cost: 0 }], me, pred, null, true);
      const digTravel = this.goalField(state, g, [{ cell: me.cell, cost: 0 }], me, pred, null, false);
      for (let cell = 0; cell < g.N; cell++) {
        if ((!state.crate[cell] && !state.brick[cell]) || state.wall[cell] || claimed(cell)) continue;
        const point = { row: Math.floor(cell / g.W), col: cell % g.W };
        const d = distance(me, point);
        if (d > 7 || Math.min(...home.map((c) => distance(point, { row: Math.floor(c / g.W), col: c % g.W }))) > 9) continue;
        const type = state.crateType[cell];
        if (state.crate[cell] && ((type === 0 && me.bombsCap >= 5) || (type === 1 && me.blast >= 7) ||
            (type === 2 && me.speed >= 2) || ((type === 3 || type === 4) && state.itemSlots &&
              state.itemSlots[pid].reduce((n, slot) => n + slot.count, 0) >= 6))) continue;
        const travel = (state.crate[cell] ? openTravel : digTravel)[cell];
        if (travel >= INF / 2) continue;
        const partner = allies.some((q) => distance(state.players[q], point) + 2 < d);
        candidates.push({ cell, score: travel + (state.crate[cell] ? -3 : 2) + (partner ? 4 : 0)
          - (cell === this.resourceTarget ? 2 : 0) });
      }
      candidates.sort((a, b) => a.score - b.score || a.cell - b.cell);
      if (!candidates.length) { this.resourceTarget = -1; return null; }
      const cell = candidates[0].cell;
      this.resourceTarget = cell;
      this.assignment.cell = cell;
      return { mode: state.crate[cell] ? 'COLLECT' : 'FARM', seeds: [{ cell, cost: 0 }], threat: false,
        noDig: !!state.crate[cell] };
    }

    routeCells(state, g, player, targets, pred, limit = 20) {
      const field = this.goalField(state, g, targets.map((cell) => ({ cell, cost: 0 })), player, pred, null, true);
      const path = [player.cell];
      for (let i = 0; i < limit && field[path[path.length - 1]] > 0; i++) {
        const cell = path[path.length - 1];
        let next = -1, best = field[cell];
        for (let a = 0; a < 4; a++) {
          const n = g.nb[cell * 4 + a];
          if (n >= 0 && g.open[n] && g.bombAt[n] < 0 && field[n] < best) { next = n; best = field[n]; }
        }
        if (next < 0) break;
        path.push(next);
      }
      return path;
    }

    safeSlide(state, g, path, index, home, danger) {
      const delta = path[index] - path[index - 1];
      const dir = delta === -g.W ? 0 : delta === g.W ? 1 : delta === -1 ? 2 : 3;
      let cell = path[index], reached = false, length = 0;
      while (cell >= 0 && g.open[cell] && g.bombAt[cell] < 0) {
        if (danger[cell] || state.fieldItem[cell] === 2) return false;
        if (home.includes(cell)) reached = true;
        if (!reached && path[index + length] !== cell) return false;
        length++;
        cell = g.nb[cell * 4 + dir];
      }
      return reached && length >= 2;
    }

    escortItemPlan(state, g, pid, carrier, pred) {
      const slots = state.itemSlots && state.itemSlots[pid];
      if (!slots || !slots.length || state.movementStatus[pid] === 2) return null;
      const me = state.players[pid], ally = state.players[carrier];
      const home = this.baseCells(state, g, ally.team), danger = this.dangerCells(pred, g.N);
      const path = this.routeCells(state, g, ally, home, pred);
      const free = (cell) => g.open[cell] && g.bombAt[cell] < 0 && !state.fieldItem[cell] && !danger[cell];
      const slot = slots.findIndex((s) => s.item === 1 && s.count > 0);
      if (slot >= 0 && state.movementStatus[carrier] !== 2) {
        for (let i = 2; i < path.length; i++) {
          const cell = path[i];
          if (!free(cell) || !this.safeSlide(state, g, path, i, home, danger)) continue;
          const travel = this.goalField(state, g, [{ cell, cost: 0 }], me, pred, null, true)[me.cell];
          if (travel * this.moveTicks(me, false) + 2 >= i * this.moveTicks(ally, false)) continue;
          return { cell, slot, item: 1 };
        }
      }
      const glue = slots.findIndex((s) => s.item === 2 && s.count > 0);
      if (glue < 0) return null;
      for (const q of this.enemiesOf(state, pid)) {
        const foe = state.players[q];
        if (!foe.alive || foe.trapped || distance(foe, ally) > 9) continue;
        const approach = this.routeCells(state, g, foe, [ally.cell], pred, 8);
        for (let i = 2; i < approach.length; i++) {
          const cell = approach[i];
          if (!free(cell) || path.some((c) => Math.abs(Math.floor(c / g.W) - Math.floor(cell / g.W)) +
              Math.abs(c % g.W - cell % g.W) < 2)) continue;
          const travel = this.goalField(state, g, [{ cell, cost: 0 }], me, pred, null, true)[me.cell];
          if (travel > 5 || travel * this.moveTicks(me, false) + 2 >= i * this.moveTicks(foe, false)) continue;
          return { cell, slot: glue, item: 2 };
        }
      }
      return null;
    }

    observeEnemyTrends(state, pid) {
      if (!this.alliesOf(state, pid).length) return;
      if (this.trendTick === state.tick) return;
      this.trendTick = state.tick;
      for (const q of this.enemiesOf(state, pid)) {
        const player = state.players[q], previous = this.enemyTrends.get(q);
        const continuous = previous && previous.tick === state.tick - 1 && player.alive && !player.trapped;
        let vy = continuous ? player.y - previous.y : 0;
        let vx = continuous ? player.x - previous.x : 0;
        if (Math.abs(vy) + Math.abs(vx) > this.stepLen(player, false) * 1.5) { vy = 0; vx = 0; }
        this.enemyTrends.set(q, { tick: state.tick, y: player.y, x: player.x, vy, vx,
          confidence: Math.abs(vy) + Math.abs(vx) >= 0.12 });
      }
    }

    enemyPathCells(state, g, q, ticks = 4) {
      const foe = state.players[q], trend = this.enemyTrends.get(q);
      const path = [foe.cell];
      if (!trend || !trend.confidence) return path;
      let y = foe.y, x = foe.x, cell = foe.cell;
      // Short observed-motion projection stops at solid tiles and live bubbles.
      for (let t = 0; t < Math.min(6, ticks); t++) {
        const nextY = y + trend.vy, nextX = x + trend.vx;
        const row = Math.floor(nextY), col = Math.floor(nextX), next = row * g.W + col;
        if (row < 0 || row >= g.H || col < 0 || col >= g.W || !g.open[next] ||
            (next !== foe.cell && g.bombAt[next] >= 0)) break;
        y = nextY; x = nextX; cell = next;
        if (path[path.length - 1] !== cell) path.push(cell);
      }
      return path;
    }

    predictedEnemyCell(state, g, q, ticks = 4) {
      const path = this.enemyPathCells(state, g, q, ticks);
      return path[path.length - 1];
    }

    solidNeighbors(state, g, cell) {
      let score = 0;
      for (let a = 0; a < 4; a++) {
        const next = g.nb[cell * 4 + a];
        if (next < 0 || state.wall[next] || state.brick[next]) score++;
      }
      return score;
    }

    defenseSeeds(state, g, pid) {
      const me = state.players[pid], foe = this.foeOf(state, pid);
      const home = this.baseCells(state, g, me.team);
      const attacker = this.alliesOf(state, pid).map((q) => state.players[q])
        .find((p) => p.alive && !p.trapped && p.carrying < 0);
      const seeds = [];
      for (let cell = 0; cell < g.N; cell++) {
        if (!g.open[cell] || g.bombAt[cell] >= 0 || home.includes(cell)) continue;
        const point = { row: Math.floor(cell / g.W), col: cell % g.W };
        const homeDistance = Math.min(...home.map((base) => distance(point, {
          row: Math.floor(base / g.W), col: base % g.W })));
        const attackerDistance = attacker ? distance(point, attacker) : 99;
        if (homeDistance < 1 || homeDistance > 5 || attackerDistance < 3) continue;
        const foeDistance = foe && foe.alive ? distance(point, foe) : 99;
        const corner = this.solidNeighbors(state, g, cell);
        seeds.push({ cell, cost: homeDistance * 0.4 + Math.max(0, foeDistance - 3) * 0.35
          - Math.min(2, corner) * 0.25 });
      }
      return seeds.sort((a, b) => a.cost - b.cost).slice(0, 18);
    }

    considerBomb(state, g, pid, pred, perceive, field) {
      if (this.lastGoalMode === 'RESCUE') return null;
      if (this.itemPlan) return null;
      if (this.lastGoalMode === 'CHAIN' && this.chainPlan && state.players[pid].cell === this.chainPlan.cell) {
        const anchor = state.bombs.find((b) => b.cell === this.chainPlan.anchor && b.owner === pid);
        if (this.alliesOf(state, pid).length && anchor && anchor.fuse > 12) return null;
      }
      const candidate = (this.lastGoalMode === 'CHAIN' ? this.tacticalBomb(state, g, pid, field) : null)
        || super.considerBomb(state, g, pid, pred, perceive, field)
        || this.tacticalBomb(state, g, pid, field);
      if (!candidate) return null;
      const me = state.players[pid];
      const mine = { cell: me.cell, e: Base.NEW_BOMB_TICK, blast: me.blast };
      // Physical safety uses all visible bombs even when a difficulty has delayed perception.
      const before = this.predict(state, g, [], () => true);
      const after = this.predict(state, g, [mine], () => true);
      if (!(this.escape(g, after, me, false).surv & (1 << candidate.move))) return null;
      for (const q of this.alliesOf(state, pid)) {
        const ally = state.players[q];
        if (!ally.alive) continue;
        if (ally.trapped) {
          for (let t = 1; t < Math.min(ally.trapped, after.T + 1); t++) {
            if (after.lethal[t * g.N + ally.cell] && !before.lethal[t * g.N + ally.cell]) return null;
          }
        } else {
          const safe = this.escape(g, after, ally, false).surv;
          if (!safe) return null;
          const committedMove = state.committedMoves && state.committedMoves[q];
          if (committedMove != null && !(safe & (1 << committedMove))) return null;
          if (ally.carrying >= 0) {
            // Do not force the slower carrier off its delivery route with a new blast line.
            const danger = this.dangerCells(after, g.N), oldDanger = this.dangerCells(before, g.N);
            if (danger[ally.cell] && !oldDanger[ally.cell]) return null;
            const home = this.baseCells(state, g, ally.team).map((cell) => ({ cell, cost: 0 }));
            const homeField = this.goalField(state, g, home, ally, before, null, true);
            for (let a = 0; a < 4; a++) {
              const n = g.nb[ally.cell * 4 + a];
              if (n >= 0 && homeField[n] < homeField[ally.cell] && danger[n] && !oldDanger[n]) return null;
            }
          }
        }
      }
      if (candidate.nextCell != null) this.chainPlan = { cell: candidate.nextCell, anchor: me.cell,
        expires: state.tick + Base.NEW_BOMB_TICK - (this.alliesOf(state, pid).length ? 4 : 8) };
      else if (this.lastGoalMode === 'CHAIN') this.chainPlan = null;
      return candidate;
    }

    blastCells(state, g, cell, blast) {
      const cells = new Set([cell]);
      for (let a = 0; a < 4; a++) {
        let c = cell;
        for (let n = 0; n < blast; n++) {
          c = g.nb[c * 4 + a];
          if (c < 0 || state.wall[c]) break;
          cells.add(c);
          if (state.brick[c] || g.bombAt[c] >= 0) break;
        }
      }
      return cells;
    }

    routeDistance(g, start, targets, blocked = -1) {
      const dist = new Int16Array(g.N).fill(-1), queue = [start];
      dist[start] = 0;
      for (let i = 0; i < queue.length; i++) {
        const c = queue[i];
        if (targets.has(c)) return dist[c];
        for (let a = 0; a < 4; a++) {
          const n = g.nb[c * 4 + a];
          if (n < 0 || n === blocked || !g.open[n] || g.bombAt[n] >= 0 || dist[n] >= 0) continue;
          dist[n] = dist[c] + 1; queue.push(n);
        }
      }
      return INF;
    }

    pickMove(g, me, acts, esc, field, threatened, danger) {
      const state = this.decisionState;
      if (state && state.nextPosition && this.lastGoalMode !== 'RESCUE') {
        let separated = 0;
        let itemSafe = 0;
        for (let a = 0; a < 5; a++) {
          if (!(acts & (1 << a))) continue;
          const next = state.nextPosition(this.currentPid, a);
          const cell = Math.floor(next[0]) * g.W + Math.floor(next[1]);
          if (cell === me.cell || !state.fieldItem[cell] ||
              (state.fieldItem[cell] === 1 && me.carrying >= 0)) itemSafe |= 1 << a;
          const conflict = this.alliesOf(state, this.currentPid).some((q) => {
            const ally = state.players[q];
            if (!ally.alive || ally.trapped) return false;
            const move = state.committedMoves[q], end = move == null ? [ally.y, ally.x] : state.nextPosition(q, move);
            if (me.carrying >= 0 && ally.carrying < 0 && move == null) return false;
            const dy = me.y - ally.y, dx = me.x - ally.x;
            const vy = next[0] - me.y - (end[0] - ally.y), vx = next[1] - me.x - (end[1] - ally.x);
            const t = Math.max(0, Math.min(1, -(dy * vy + dx * vx) / (vy * vy + vx * vx || 1)));
            const closest = Math.hypot(dy + t * vy, dx + t * vx);
            // Existing overlap must be allowed to separate; never eliminate every safe escape.
            return closest < (ally.carrying >= 0 ? 1.4 : 0.75) &&
              Math.hypot(next[0] - end[0], next[1] - end[1]) <= Math.hypot(dy, dx) + 0.01;
          });
          if (!conflict) separated |= 1 << a;
        }
        if (itemSafe) acts = itemSafe;
        if (separated & acts) acts &= separated;
      }
      if (this.itemPlan && me.cell === this.itemPlan.cell) {
        const exits = acts & 15;
        if (exits) acts = exits;
      }
      if (this.lastGoalMode === 'GUARD' && !threatened && (acts & 16) &&
          !(danger && danger[me.cell]) && field[me.cell] <= Math.min(...Array.from(
            g.nb.slice(me.cell * 4, me.cell * 4 + 4)).filter((c) => c >= 0).map((c) => field[c]))) return 4;
      if (this.lastGoalMode === 'CHAIN') {
        if (this.assignment && this.assignment.role !== 'SOLO' && this.chainPlan &&
            me.cell === this.chainPlan.cell && (acts & 16) && esc.count[4] > 0) return 4;
        return super.pickMove(g, me, acts, esc, field, false, null);
      }
      let spacedField = field;
      if (!threatened && this.decisionState && ['HUNT', 'SUPPORT', 'GUARD', 'DEFEND'].includes(this.lastGoalMode)) {
        spacedField = field.slice();
        const allies = this.alliesOf(this.decisionState, this.currentPid)
          .map((q) => this.decisionState.players[q]).filter((p) => p.alive && !p.trapped);
        for (const cell of [me.cell, ...Array.from(g.nb.slice(me.cell * 4, me.cell * 4 + 4))]) {
          if (cell < 0) continue;
          const point = { row: Math.floor(cell / g.W), col: cell % g.W };
          for (const ally of allies) spacedField[cell] += Math.max(0, 3 - distance(point, ally)) * 2;
        }
      }
      return super.pickMove(g, me, acts, esc, spacedField, threatened, danger);
    }

    tacticalBomb(state, g, pid, field) {
      const me = state.players[pid];
      if (this.difficulty === 'easy' || me.carrying >= 0 || me.bombsLeft <= 0 ||
          me.liveBombs >= this.cfg.maxLiveBombs || g.bombAt[me.cell] >= 0 || !g.open[me.cell] ||
          !['HUNT', 'SUPPORT', 'GUARD', 'DEFEND', 'INTERCEPT', 'ESCORT', 'CHAIN'].includes(this.lastGoalMode)) return null;
      const foes = this.enemiesOf(state, pid).map((q) => ({ ...state.players[q], pid: q })).filter((e) =>
        e.alive && !e.trapped && e.invuln < Base.NEW_BOMB_TICK && distance(me, e) <= this.cfg.attackRadius);
      if (!foes.length) return null;
      const mine = { cell: me.cell, e: Base.NEW_BOMB_TICK, blast: me.blast };
      const before = this.predict(state, g, [], () => true);
      const after = this.predict(state, g, [mine], () => true);
      const blast = this.blastCells(state, g, me.cell, me.blast);
      const oldDanger = this.dangerCells(before, g.N), danger = this.dangerCells(after, g.N);
      let reason = null, nextCell = null;
      if (this.chainPlan && me.cell === this.chainPlan.cell && blast.has(this.chainPlan.anchor) &&
          foes.some((e) => blast.has(e.cell) || blast.has(this.predictedEnemyCell(state, g, e.pid,
            after.bombGone[me.cell])))) reason = 'bomb_chain';
      const home = new Set(this.baseCells(state, g, me.team));
      for (const foe of foes) {
        if (reason) break;
        const approach = this.routeDistance(g, foe.cell, home);
        if (approach <= 7 && distance(me, foe) <= 5 && me.cell !== foe.cell &&
            this.routeDistance(g, foe.cell, home, me.cell) >= approach + 3) {
          reason = 'bomb_block'; break;
        }
        const futureCell = this.predictedEnemyCell(state, g, foe.pid, after.bombGone[me.cell]);
        if ((danger[foe.cell] && !oldDanger[foe.cell]) ||
            (danger[futureCell] && !oldDanger[futureCell] && after.bombGone[me.cell] <= 12)) {
          const linked = state.bombs.some((b) => state.players[b.owner] &&
            state.players[b.owner].team === me.team && blast.has(b.cell));
          reason = linked ? 'bomb_chain' : 'bomb_pressure'; break;
        }
      }
      // Reserve a first bubble only when a reachable second placement connects to it and aims at a foe.
      if (!reason && me.bombsLeft >= 2 && me.liveBombs === 0 && me.liveBombs + 1 < this.cfg.maxLiveBombs &&
          ['HUNT', 'SUPPORT'].includes(this.lastGoalMode)) {
        let best = INF;
        for (const cell of blast) {
          if (cell === me.cell || !g.open[cell] || g.bombAt[cell] >= 0) continue;
          const travel = this.routeDistance(g, me.cell, new Set([cell]));
          const ticks = Math.ceil(travel / this.stepLen(me, false));
          if (ticks < 1 || ticks + 10 >= after.bombGone[me.cell]) continue;
          const secondRay = this.blastCells(state, g, cell, me.blast);
          if (!foes.some((e) => {
            const target = this.predictedEnemyCell(state, g, e.pid);
            return (secondRay.has(e.cell) && !blast.has(e.cell)) || (secondRay.has(target) && !blast.has(target));
          })) continue;
          const future = { ...me, cell, row: Math.floor(cell / g.W), col: cell % g.W,
            y: Math.floor(cell / g.W) + 0.5, x: cell % g.W + 0.5 };
          const shifted = { ...state, bombs: state.bombs.map((b) => ({ ...b, fuse: Math.max(1, b.fuse - ticks) })) };
          const connected = this.predict(shifted, g, [
            { ...mine, e: after.bombGone[me.cell] - ticks },
            { cell, e: Base.NEW_BOMB_TICK, blast: me.blast },
          ], () => true);
          if (!this.escape(g, connected, future, false).surv) continue;
          const score = travel - (this.alliesOf(state, pid).length ?
            Math.min(2, this.solidNeighbors(state, g, me.cell)) * 0.5
              + Math.min(2, this.solidNeighbors(state, g, cell)) * 0.8 : 0);
          if (score < best) { best = score; nextCell = cell; reason = 'bomb_reserve'; }
        }
      }
      if (!reason) return null;
      const escape = this.escape(g, after, me, false);
      let safe = escape.surv;
      if (this.cfg.robust) safe &= this.hypoSurvivors(state, g, pid, () => true, [mine]);
      if (!safe) return null;
      const targetField = nextCell == null ? field : this.goalField(state, g,
        [{ cell: nextCell, cost: 0 }], me, after, null, true);
      return { move: this.pickMove(g, me, safe, escape, targetField, true, danger), reason, nextCell };
    }

  }

  return { BunCoopHunterBot, hunterStateFromSim: Base.hunterStateFromSim };
});
