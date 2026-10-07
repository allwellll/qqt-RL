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
      this.reservePlan = null;
      this.reserveRetryTick = 0;
      this.reserveCancellations = 0;
      this.staggerPlan = null;
      this.staggerRetreatUntil = 0;
      this.lastAttackTiming = null;
      this.lastDirectThreat = null;
      this.directPursuit = null;
    }

    analyzeSim(sim, pid) {
      const state = Base.hunterStateFromSim(sim);
      state.itemSlots = sim.nativeItems ? sim.itemSlots : state.heldItem.map((item) => item ? [{ item, count: 1 }] : []);
      state.fieldOwner = sim.fieldOwner;
      state.movementStatus = sim.movementStatus;
      // Attack evidence uses the actual native body rule and movement primitive,
      // not the conservative grid/linger map used to protect our own escape.
      state.threatHit = (q, y, x, covered) => {
        const body = Object.create(sim);
        body.pos = sim.pos.slice(); body.pos[q * 2] = y; body.pos[q * 2 + 1] = x;
        return body._isHitByExplosion(q, covered, null, null, false);
      };
      state.threatPosition = (q, y, x, move, pred, tick) => {
        // A changing status, item contact or push-box route is inconclusive. Do
        // not turn an incomplete forecast into a claimed direct hit.
        if (sim.movementStatus[q] === 2 || sim.pushBoxes.some(b => !b.dead) ||
            (sim.movementStatus[q] && sim.movementStatusTicks[q] < tick)) return null;
        const obstacles = Uint8Array.from(state.wall, (wall, c) => wall ||
          (state.brick[c] && (!pred.brickGone[c] || pred.brickGone[c] >= tick)) || pred.bombGone[c] >= tick ? 1 : 0);
        const next = move === 4 ? [y, x] : sim._steer(y, x, move, obstacles, this.stepLen(state.players[q], false), q);
        // Sim checks the starting tile and the swept movement, including a
        // departure which lands in a different tile during this tick.
        const steps = Math.max(1, Math.ceil(Math.max(Math.abs(next[0] - y), Math.abs(next[1] - x)) * 4));
        for (let k = 0; k <= steps; k++) {
          const cell = Math.floor(y + (next[0] - y) * k / steps) * state.width +
            Math.floor(x + (next[1] - x) * k / steps);
          if (state.fieldItem[cell] || state.crate[cell]) return null;
        }
        return next;
      };
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
      const obstacleCache = new WeakMap();
      state.projectPosition = (q, y, x, move, pred, tick) => {
        if (move === 4 || !state.players[q].alive || state.players[q].trapped) return [y, x];
        let frames = obstacleCache.get(pred);
        if (!frames) { frames = []; obstacleCache.set(pred, frames); }
        if (!frames[tick]) frames[tick] = Uint8Array.from(state.wall,
          (wall, c) => wall || state.brick[c] || pred.bombGone[c] > tick ? 1 : 0);
        const obstacles = frames[tick];
        const player = state.players[q], step = this.stepLen(player, false);
        const direction = sim.playerMoveDirection(q, move);
        return sim.movementStatus[q] === 2 ? sim._tryMove(y, x, direction, obstacles, step)
          : sim._steer(y, x, direction, obstacles, step, q);
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
      const combat = ['HUNT', 'SUPPORT', 'RAID', 'DEFEND', 'GUARD', 'INTERCEPT', 'ESCORT'];
      const me = state.players[pid];
      if ((this.reservePlan || this.chainPlan) && (!combat.includes(goal.mode) ||
          me.carrying >= 0 || state.movementStatus && state.movementStatus[pid] === 2)) {
        this.cancelReserve(state);
      }
      if (this.reservePlan) {
        const reserve = this.reservePlan;
        const targetRay = this.blastCells(state, g, reserve.targetCell, state.players[pid].blast);
        const remaining = this.routeDistance(g, me.cell, new Set([reserve.targetCell]));
        if (remaining < reserve.remaining || reserve.remaining == null) {
          reserve.remaining = remaining; reserve.progressTick = state.tick;
        }
        if (state.tick >= reserve.expires || !g.open[reserve.targetCell] || g.bombAt[reserve.targetCell] >= 0 ||
            state.tick - reserve.progressTick > 8 || me.bombsLeft < 2 ||
            !state.bombs.some((b) => b.owner === pid && b.cell === reserve.anchor) ||
            !this.enemiesOf(state, pid).some((q) => state.players[q].alive && !state.players[q].trapped &&
              this.pressureRouteCells(state, g, pid, q, pred).some((c) => targetRay.has(c)))) {
          this.cancelReserve(state);
        } else {
          goal = { mode: 'RESERVE', seeds: [{ cell: reserve.targetCell, cost: 0 }], threat: false, noDig: true };
        }
      }
      const plan = this.chainPlan;
      if (plan) {
        const ray = this.blastCells(state, g, plan.cell, state.players[pid].blast);
        for (const cell of plan.linkedCells || []) for (const c of this.blastCells(state, g, cell, state.players[pid].blast)) ray.add(c);
        const anchor = state.bombs.find((b) => b.cell === plan.anchor && b.owner === pid);
        const usefulPath = this.enemiesOf(state, pid).some((q) => state.players[q].alive &&
          !state.players[q].trapped && this.enemyPathCells(state, g, q, anchor ? anchor.fuse : 0)
            .some((cell) => ray.has(cell)));
        // Keep a prepared connector through the staging window. Re-evaluate the
        // target only as the earliest reserve approaches detonation; this avoids
        // abandoning a valid ambush because an opponent briefly changes lanes.
        if (state.tick >= plan.expires || !g.open[plan.cell] || g.bombAt[plan.cell] >= 0 ||
            !anchor || (!usefulPath && (anchor.fuse <= 12 || !(plan.linkedCells || []).length ||
              !this.enemiesOf(state, pid).some((q) => state.players[q].alive &&
                distance(me, state.players[q]) <= 7)))) this.cancelReserve(state);
      }
      if (this.chainPlan && [...combat, 'RESERVE'].includes(goal.mode)) {
        goal = { mode: 'CHAIN', seeds: [{ cell: plan.cell, cost: 0 }], threat: false, noDig: true };
      }
      if (!combat.includes(goal.mode) || me.carrying >= 0 || this.itemPlan ||
          (state.movementStatus && state.movementStatus[pid] === 2)) this.staggerPlan = null;
      if (!this.reservePlan && !this.chainPlan && combat.includes(goal.mode)) {
        const physical = this.predict(state, g, [], () => true);
        if (this.staggerPlan) {
          const p = this.staggerPlan, anchor = state.bombs.find(b => b.cell === p.anchor && b.owner === pid);
          const remaining = this.routeDistance(g, me.cell, new Set([p.cell]));
          if (remaining < p.remaining) { p.remaining = remaining; p.progressTick = state.tick; }
          if (!anchor || state.tick >= p.expires || state.tick - p.progressTick > 8 ||
              !this.staggerProposal(state, g, pid, anchor, p.cell, physical)) this.staggerPlan = null;
        }
        if (!this.staggerPlan) this.staggerPlan = this.findStaggeredPlan(state, g, pid, physical);
        if (this.staggerPlan) goal = { mode: 'STAGGER', seeds: [{ cell: this.staggerPlan.cell, cost: 0 }], threat: false, noDig: true };
      }
      this.lastGoalMode = goal.mode;
      return goal;
    }

    cancelReserve(state) {
      this.reservePlan = null;
      this.chainPlan = null;
      this.reserveRetryTick = state.tick + 12;
      this.reserveCancellations++;
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
          if (d < 2 || d > (foe && foe.alive ? 5 : 3) || delivery.includes(cell)) continue;
          seeds.push({ cell, cost: foe && foe.alive && !foe.trapped ? distance(point, foe) * 0.4 : d * 0.2 });
        }
        return { mode: 'ESCORT', seeds: seeds.length ? seeds : [{ cell: ally.cell, cost: 0 }], threat: false, noDig: true };
      }
      const thief = this.enemiesOf(state, pid).find((q) => state.players[q].alive && state.players[q].carrying === team);
      if (thief != null) return { mode: 'INTERCEPT', seeds: this.attackSeeds(state, g, pid, pred, perceive), threat: false };
      if (role !== 'SUPPORT') {
        const goal = super.chooseGoal(state, g, pid, pred, perceive);
        if (goal.mode === 'HUNT' && role === 'ATTACKER') {
          const resource = this.resourceGoal(state, g, pid, pred);
          if (resource && resource.mode === 'COLLECT' && distance(me, {
            row: Math.floor(resource.seeds[0].cell / g.W), col: resource.seeds[0].cell % g.W }) <= 3) return resource;
          if (role === 'ATTACKER') goal.seeds = this.offensiveSeeds(state, g, pid, pred, perceive).concat(goal.seeds || []);
        }
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
      if (this.directPursuit && state.tick < this.directPursuit.until && state.threatPosition && state.threatHit) {
        const q = this.directPursuit.target, target = state.players[q], trend = this.enemyTrends.get(q);
        if (target.alive && !target.trapped) {
          const move = trend?.confidence ? (Math.abs(trend.vy) > Math.abs(trend.vx) ? (trend.vy < 0 ? 0 : 1) : (trend.vx < 0 ? 2 : 3)) : 4;
          let y = target.y, x = target.x, valid = true;
          for (let t = 1; t <= Base.NEW_BOMB_TICK; t++) {
            const next = state.threatPosition(q,y,x,move,pred,t);
            if (!next) { valid = false; break; } [y,x] = next;
          }
          if (valid) for (let cell = 0; cell < g.N; cell++) {
            if (!g.open[cell] || g.bombAt[cell] >= 0) continue;
            const point = {row:Math.floor(cell/g.W),col:cell%g.W};
            if (Math.abs(point.row-Math.floor(y))+Math.abs(point.col-Math.floor(x)) > state.players[pid].blast) continue;
            const ray = new Uint8Array(g.N);
            for(const c of this.blastCells(state,g,cell,state.players[pid].blast)) ray[c]=1;
            if(state.threatHit(q,y,x,ray)) seeds.push({cell,cost:-4});
          }
        }
      }
      return seeds.map((s) => {
        const point = { row: Math.floor(s.cell / g.W), col: s.cell % g.W };
        const spacing = allies.reduce((cost, ally) => cost + Math.max(0, 3 - distance(point, ally)) * 2.5, 0);
        return { cell: s.cell, cost: s.cost + Math.max(0, distance(point, foe) - 2) * 0.7 + spacing };
      });
    }

    offensiveSeeds(state, g, pid, pred, perceive) {
      const me = state.players[pid], team = me.team == null ? pid : me.team;
      const seeds = this.attackSeeds(state, g, pid, pred, perceive).slice();
      const enemyHome = this.baseCells(state, g, 1 - team), foe = this.foeOf(state, pid);
      const route = foe && foe.alive ? this.routeCells(state, g, foe, this.baseCells(state, g, team), pred, 12) : [];
      const add = (cell, cost) => {
        if (cell < 0 || !g.open[cell] || g.bombAt[cell] >= 0) return;
        if (!seeds.some((s) => s.cell === cell)) seeds.push({ cell, cost });
      };
      route.slice(1, 8).forEach((cell, i) => add(cell, 1.5 + i * 0.25));
      for (const base of enemyHome) for (let a = 0; a < 4; a++) add(g.nb[base * 4 + a], 2.2);
      return seeds;
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
      if (decision.action[1] === 1 && this.lastDirectThreat) decision.directThreat = this.lastDirectThreat;
      if (decision.action[1] === 1 && decision.reason.startsWith('bomb_stagger')) decision.attackTiming = this.lastAttackTiming;
      if (decision.reason === 'doomed_max_survival' && state.projectPosition && this.alliesOf(state, pid).length) {
        const g = this.geometry(state), pred = this.predict(state, g, [], () => true);
        const physical = [];
        for (let move = 0; move < 5; move++) if (this.physicalEscape(state, g, pid, pred, move)) physical.push(move);
        if (physical.length) {
          decision.action[0] = physical[0];
          decision.reason = 'physical_escape';
        }
      }
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

    glueClearBomb(state, g, pid, pred, field) {
      const me = state.players[pid];
      const glue = Array.from(g.nb.slice(me.cell * 4, me.cell * 4 + 4))
        .filter(c => c >= 0 && state.fieldItem[c] === 2 && field[c] < field[me.cell]);
      if (!glue.length || me.carrying >= 0 || me.bombsLeft <= 0 ||
          me.liveBombs >= this.cfg.maxLiveBombs || g.bombAt[me.cell] >= 0) return null;
      const slow = this.escape(g, pred, { ...me, speed: Math.min(me.speed, 0.6) }, false);
      const canEat = [0, 1, 2, 3].some(a => glue.includes(g.nb[me.cell * 4 + a]) && (slow.surv & (1 << a)));
      if (canEat) return null;
      const after = this.predict(state, g, [{cell:me.cell,e:Base.NEW_BOMB_TICK,blast:me.blast}], () => true);
      const esc = this.escape(g, after, me, false);
      const danger = this.dangerCells(after, g.N);
      let acts = esc.surv & 15;
      for (let a = 0; a < 4; a++) {
        const c = g.nb[me.cell * 4 + a];
        if (c < 0 || state.fieldItem[c] || !this.physicalEscape(state,g,pid,after,a)) acts &= ~(1 << a);
      }
      if (!acts) return null;
      for (const q of this.alliesOf(state,pid)) {
        const ally = state.players[q];
        if (ally.alive && (danger[ally.cell] || !this.escape(g,after,ally,false).surv)) return null;
      }
      const move = super.pickMove(g,me,acts,esc,field,true,danger);
      return {move,reason:'bomb_clear_glue'};
    }

    considerBomb(state, g, pid, pred, perceive, field) {
      this.lastDirectThreat = null;
      if (this.lastGoalMode === 'RESCUE') return null;
      if (this.itemPlan) return null;
      if (state.movementStatus && state.movementStatus[pid] === 2) return null;
      if (this.lastGoalMode === 'CHAIN' && this.chainPlan && state.players[pid].cell === this.chainPlan.cell) {
        const anchor = state.bombs.find((b) => b.cell === this.chainPlan.anchor && b.owner === pid);
        const window = 12;
        if (this.alliesOf(state, pid).length && anchor && anchor.fuse > window) return null;
      }
      const clear = this.lastGoalMode === 'STAGGER' ? null : this.glueClearBomb(state,g,pid,this.predict(state,g,[],()=>true),field);
      let immediate = null;
      if (this.lastGoalMode === 'STAGGER' && state.players[pid].cell !== this.staggerPlan?.cell) {
        // A future timing plan must not suppress an already useful attack.
        this.lastGoalMode = 'HUNT';
        const opportunity = super.considerBomb(state, g, pid, pred, perceive, field);
        this.lastGoalMode = 'STAGGER';
        if (opportunity && ['bomb_attack', 'bomb_kill'].includes(opportunity.reason)) immediate = opportunity;
      }
      const direct = this.directBomb(state, g, pid, field);
      const candidate = direct || (this.lastGoalMode === 'STAGGER' ? immediate || this.staggerBomb(state, g, pid, field)
        : clear || (this.lastGoalMode === 'RESERVE' ? this.tacticalBomb(state, g, pid, field)
        : (this.lastGoalMode === 'CHAIN' ? this.tacticalBomb(state, g, pid, field) : null)
        || super.considerBomb(state, g, pid, pred, perceive, field)
        || this.tacticalBomb(state, g, pid, field)));
      if (!candidate) return null;
      const me = state.players[pid];
      const cooperative = this.alliesOf(state, pid).length > 0;
      const scheduled = candidate.reason.startsWith('bomb_stagger'), directAttack = candidate.reason === 'bomb_direct';
      const mine = { cell: me.cell, e: Base.NEW_BOMB_TICK, blast: me.blast };
      // Physical safety uses all visible bombs even when a difficulty has delayed perception.
      const before = this.predict(state, g, [], () => true);
      const after = this.predict(state, g, [mine], () => true);
      if (this.directPursuit && state.tick < this.directPursuit.until &&
          (['bomb_attack','bomb_kill','bomb_chain'].includes(candidate.reason) ||
            candidate.reason.startsWith('bomb_reserve') || candidate.reason.startsWith('bomb_stagger')) &&
          !this.directThreat(state,g,pid,me.cell,before,after)) return null;
      const selfEscape = this.escape(g, after, me, false);
      if (!(selfEscape.surv & (1 << candidate.move))) return null;
      const selfSpace = selfEscape.endCount;
      const oldSelfSpace = this.escape(g, before, me, false).endCount;
      if (cooperative && (selfSpace < Math.min(12, oldSelfSpace) || selfSpace < oldSelfSpace * 0.15)) return null;
      // Leave three ticks of slack for changing opponent fire, contact fields and
      // continuous corner alignment. A last-moment theoretical exit is too brittle.
      const safetyState = cooperative || scheduled || directAttack ? { ...state, bombs: state.bombs.map((b) =>
        ({ ...b, fuse: Math.max(1, b.fuse - 3) })) } : state;
      const safety = cooperative || scheduled || directAttack ? this.predict(safetyState, g,
        [{ ...mine, e: mine.e - 3 }], () => true) : after;
      if ((scheduled || directAttack) && !(this.escape(g, safety, me, false).surv & (1 << candidate.move))) return null;
      const retreat = (!cooperative && !scheduled && !directAttack) || this.physicalEscape(state, g, pid, safety, candidate.move);
      if (!retreat) return null;
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
          const allySpace = this.escape(g, after, ally, false).endCount;
          const oldAllySpace = this.escape(g, before, ally, false).endCount;
          if (allySpace < Math.min(12, oldAllySpace) || allySpace < oldAllySpace * 0.3) return null;
          if (state.movementStatus && state.movementStatus[q] === 2 && this.blastCells(state, g, me.cell, me.blast).has(ally.cell)) return null;
          const committedMove = state.committedMoves && state.committedMoves[q];
          // An undeclared or idle teammate (including the human player) must not
          // be assumed to execute a hypothetical escape just because one exists.
          if (committedMove == null || committedMove === 4) {
            for (let t = 1; t <= after.T; t++) {
              const index = t * g.N + ally.cell;
              if (after.lethal[index] && !before.lethal[index]) return null;
            }
          }
          if (committedMove != null && !(safe & (1 << committedMove))) return null;
          if (!this.physicalEscape(state, g, q, safety, committedMove)) return null;
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
            const route = this.routeCells(state, g, ally, home.map((s) => s.cell), before);
            for (let i = 1; i < route.length; i++) {
              if (route[i] === me.cell) return null;
              const arrival = i * this.moveTicks(ally, false);
              for (let t = Math.max(1, arrival - 2); t <= Math.min(after.T, arrival + 2); t++) {
                const offset = t * g.N + route[i];
                if (after.lethal[offset] && !before.lethal[offset]) return null;
              }
            }
          }
        }
      }
      if (candidate.reserveTarget != null) {
        this.reservePlan = { anchor: me.cell, targetCell: candidate.reserveTarget,
          connector: candidate.reserveConnector, expires: state.tick + Base.NEW_BOMB_TICK - 8 };
        this.chainPlan = null;
      } else if (candidate.nextCell != null) this.chainPlan = { cell: candidate.nextCell,
        anchor: candidate.chainAnchor == null ? me.cell : candidate.chainAnchor,
        linkedCells: ['bomb_reserve_second', 'bomb_reserve_replenish'].includes(candidate.reason) ? [me.cell] : [],
        expires: state.tick + (candidate.chainFuse == null ? Base.NEW_BOMB_TICK - (this.alliesOf(state, pid).length ? 4 : 8)
          : candidate.chainFuse - 2) };
      else if (this.lastGoalMode === 'CHAIN') this.chainPlan = null;
      if (candidate.reason === 'bomb_reserve_second') this.reservePlan = null;
      if (immediate) this.staggerPlan = null;
      this.lastDirectThreat = candidate.proof || this.directThreat(state, g, pid, me.cell, before, after);
      if (directAttack) {
        this.reservePlan = null; this.chainPlan = null;
        this.staggerPlan = null; this.staggerRetreatUntil = state.tick + after.T;
        this.directPursuit = {target:candidate.proof.target,until:state.tick+Base.NEW_BOMB_TICK*3};
      }
      if (scheduled) {
        this.lastAttackTiming = { anchor: this.staggerPlan.anchor,
          anchorExplosionTick: state.tick + before.bombGone[this.staggerPlan.anchor],
          expectedExplosionTick: state.tick + after.bombGone[me.cell], insuranceTicks: 3 };
        this.staggerRetreatUntil = state.tick + after.T;
        this.staggerPlan = null;
      }
      return candidate;
    }

    directThreat(state, g, pid, cell, before, after) {
      if (!state.threatPosition || !state.threatHit || !after.impact) return null;
      const me = state.players[pid], explosion = after.bombGone[cell];
      if (!explosion || explosion > Base.NEW_BOMB_TICK) return null;
      const source = after.sources.find(s => s.cell === cell && s.tick === explosion);
      if (!source) return null;
      const ownImpact = new Uint8Array(g.N);
      for (const c of source.covered) ownImpact[c] = 1;
      for (const q of this.enemiesOf(state, pid)) {
        const foe = state.players[q], trend = this.enemyTrends.get(q);
        if (!foe.alive || foe.trapped || foe.invuln >= explosion || distance(me, foe) > this.cfg.attackRadius) continue;
        const move = trend?.confidence ? (Math.abs(trend.vy) > Math.abs(trend.vx) ? (trend.vy < 0 ? 0 : 1) : (trend.vx < 0 ? 2 : 3)) : 4;
        const project = (prediction, requested) => {
          let y = foe.y, x = foe.x; const path = [];
          for (let tick = 1; tick <= explosion; tick++) {
            const next = state.threatPosition(q, y, x, requested, prediction, tick);
            if (!next) return null;
            [y, x] = next; path.push({ tick: state.tick + tick, y, x });
            if (tick > foe.invuln && state.threatHit(q, y, x, prediction.impact.subarray(tick * g.N, (tick + 1) * g.N)))
              return { hitTick: state.tick + tick, path };
          }
          return { hitTick: null, path };
        };
        const old = project(before, move), forecast = project(after, move);
        if (!old || !forecast || old.hitTick != null || forecast.hitTick !== state.tick + explosion) continue;
        const contact = forecast.path.at(-1);
        if (!state.threatHit(q, contact.y, contact.x, ownImpact)) continue;
        const alternatives = Array.from({ length: 5 }, (_, a) => ({ move: a, result: project(after, a) }));
        const safeMoves = alternatives.filter(a => a.result && a.result.hitTick == null).map(a => a.move);
        return { target: q, source: trend?.confidence ? 'observed-motion' : 'held-position',
          prediction: 'conditional-hit', guaranteed: false, intendedMove: move,
          predictedHitTick: forecast.hitTick, expectedExplosionTick: state.tick + explosion,
          targetPath: forecast.path, baselinePath: old.path, safeConstantMoves: safeMoves,
          sourceCell: cell, covered: source.covered.slice() };
      }
      return null;
    }

    directBomb(state, g, pid, field) {
      const me = state.players[pid];
      if (this.difficulty !== 'hard' || me.carrying >= 0 || me.bombsLeft <= 0 ||
          me.liveBombs >= this.cfg.maxLiveBombs || g.bombAt[me.cell] >= 0 ||
          !['HUNT','SUPPORT','DEFEND','GUARD','INTERCEPT','STAGGER','OPEN_ROUTE'].includes(this.lastGoalMode)) return null;
      const mine = { cell: me.cell, e: Base.NEW_BOMB_TICK, blast: me.blast };
      const before = this.predict(state, g, [], () => true), after = this.predict(state, g, [mine], () => true);
      const proof = this.directThreat(state, g, pid, me.cell, before, after);
      // Existing same-line attacks keep their strategy. New attacks deliberately
      // cover an arriving player who is outside this bubble's current ray.
      const following = proof && this.directPursuit && state.tick < this.directPursuit.until &&
        this.directPursuit.target === proof.target;
      if (!proof || (!following && (proof.source !== 'observed-motion' ||
          this.blastCells(state, g, me.cell, me.blast).has(state.players[proof.target].cell)))) return null;
      const esc = this.escape(g, after, me, false);
      let safe = esc.surv & this.hypoSurvivors(state, g, pid, () => true, [mine]);
      if (!safe) return null;
      return { move: this.pickMove(g, me, safe, esc, field, true, this.dangerCells(after, g.N)), reason: 'bomb_direct', proof };
    }

    staggerProposal(state, g, pid, anchor, cell, before) {
      const me = state.players[pid];
      if (this.difficulty !== 'hard' || me.carrying >= 0 || me.bombsLeft <= 0 ||
          me.liveBombs >= this.cfg.maxLiveBombs || (state.movementStatus && state.movementStatus[pid] === 2) ||
          !g.open[cell] || g.bombAt[cell] >= 0 || state.fieldItem[cell] || before.bombGone[anchor.cell] < 4) return null;
      const ray = this.blastCells(state, g, cell, me.blast);
      const useful = this.enemiesOf(state, pid).some(q => {
        const foe = state.players[q];
        return foe.alive && !foe.trapped && foe.invuln < Base.NEW_BOMB_TICK && distance(me, foe) <= this.cfg.attackRadius &&
          (ray.has(foe.cell) || this.pressureRouteCells(state, g, pid, q, before).filter(c => ray.has(c)).length >= 2);
      });
      if (!useful) return null;
      if (state.players.some(p => p.team !== me.team && p.alive && !p.trapped && p.cell === cell && p.bombsLeft > 0)) return null;
      const after = this.predict(state, g, [{ cell, e: Base.NEW_BOMB_TICK, blast: me.blast }], () => true);
      const first = before.bombGone[anchor.cell], next = after.bombGone[cell];
      const extension = next >= first + Base.FLAME_TICKS;
      // A ray connector detonates in the same engine tick. Keep a separate later
      // bubble intact so the connector cannot collapse every pressure phase.
      const continuation = next === first && state.bombs.some(b => b.owner === pid && b.cell !== anchor.cell &&
        before.bombGone[b.cell] >= first + Base.FLAME_TICKS && after.bombGone[b.cell] === before.bombGone[b.cell]);
      if (!extension && !continuation) return null;
      if (!extension && !this.enemiesOf(state, pid).some(q => {
        const foe = state.players[q];
        return foe.alive && !foe.trapped && [foe.cell, ...this.pressureRouteCells(state, g, pid, q, before)]
          .some(c => ray.has(c) && !before.lethal[first * g.N + c]);
      })) return null;
      for (const q of this.alliesOf(state, pid)) {
        const ally = state.players[q];
        if (!ally.alive) continue;
        if (ray.has(ally.cell) && (ally.trapped || state.committedMoves?.[q] == null || state.committedMoves[q] === 4 || ally.carrying >= 0)) return null;
        if (!this.escape(g, after, ally, false).surv) return null;
        if (ally.carrying >= 0 && this.routeCells(state, g, ally, this.baseCells(state, g, ally.team), before).includes(cell)) return null;
      }
      return { after, kind: extension ? 'extension' : 'chain' };
    }

    findStaggeredPlan(state, g, pid, pred) {
      const me = state.players[pid], candidates = [];
      if (this.itemPlan || me.carrying >= 0 || !this.escape(g, pred, me, false).surv) return null;
      for (const anchor of state.bombs.filter(b => b.owner === pid && b.fuse >= 8 && b.fuse <= 22 && pred.bombGone[b.cell] === b.fuse)) {
        for (let a = 0; a < 4; a++) {
          let cell = anchor.cell;
          for (let d = 1; d <= anchor.blast + 2; d++) {
            cell = g.nb[cell * 4 + a];
            if (cell < 0 || !g.open[cell] || g.bombAt[cell] >= 0) break;
            const travel = this.routeDistance(g, me.cell, new Set([cell]));
            const ticks = travel * this.moveTicks(me, false) + (travel ? 1 : 0);
            if (travel > 3 || ticks + 3 >= anchor.fuse) continue;
            const proposal = this.staggerProposal(state, g, pid, anchor, cell, pred);
            if (proposal) candidates.push({ cell, anchor: anchor.cell, kind: proposal.kind, ticks, remaining: travel,
              score: travel + (proposal.kind === 'chain' ? 1 : 0), expires: state.tick + anchor.fuse - 3, progressTick: state.tick });
          }
        }
      }
      candidates.sort((a, b) => a.score - b.score || a.cell - b.cell);
      for (const plan of candidates.slice(0, 4)) {
        const early = { ...state, bombs: state.bombs.map(b => ({ ...b, fuse: Math.max(1, b.fuse - 3) })) };
        const approach = this.predict(early, g, [], () => true);
        const route = this.physicalEscape(state, g, pid, approach, null,
          { cell: plan.cell, deadline: approach.bombGone[plan.anchor] - 1 });
        if (!route) continue;
        const future = { ...me, cell: plan.cell, row: Math.floor(plan.cell / g.W), col: plan.cell % g.W,
          y: Math.floor(plan.cell / g.W) + 0.5, x: plan.cell % g.W + 0.5 };
        const wait = Math.max(plan.ticks, pred.bombGone[plan.anchor] - 12);
        const futureState = { ...state, players: state.players.map((p, q) => q === pid ? future : p),
          bombs: state.bombs.map(b => ({ ...b, fuse: Math.max(1, b.fuse - wait - 3) })) };
        const safety = this.predict(futureState, g, [{ cell: plan.cell, e: Base.NEW_BOMB_TICK - 3, blast: me.blast }], () => true);
        if (this.escape(g, safety, future, false).surv && this.physicalEscape(futureState, g, pid, safety)) return plan;
      }
      return null;
    }

    staggerBomb(state, g, pid, field) {
      const p = this.staggerPlan, me = state.players[pid];
      if (!p || me.cell !== p.cell) return null;
      const before = this.predict(state, g, [], () => true), anchor = state.bombs.find(b => b.cell === p.anchor && b.owner === pid);
      if (!anchor || before.bombGone[anchor.cell] > 12) return null;
      const proposal = this.staggerProposal(state, g, pid, anchor, p.cell, before);
      if (!proposal) return null;
      const extra = [{ cell: me.cell, e: Base.NEW_BOMB_TICK, blast: me.blast }], escape = this.escape(g, proposal.after, me, false);
      let safe = escape.surv;
      if (this.cfg.robust) safe &= this.hypoSurvivors(state, g, pid, () => true, extra);
      if (!safe) return null;
      return { move: this.pickMove(g, me, safe, escape, field, true, this.dangerCells(proposal.after, g.N), true),
        reason: `bomb_stagger_${proposal.kind}` };
    }

    pressureRouteCells(state, g, pid, q, pred) {
      const foe = state.players[q], me = state.players[pid];
      const cells = new Set(this.enemyPathCells(state, g, q));
      const carrier = this.alliesOf(state, pid).find((p) => state.players[p].alive && state.players[p].carrying >= 0);
      const target = carrier == null ? this.baseCells(state, g, me.team) : [state.players[carrier].cell];
      // Only the next few traversable cells are useful for prepositioning. A distant
      // theoretical route must not justify spending capacity across the whole map.
      for (const cell of this.routeCells(state, g, foe, target, pred, 4).slice(0, 4)) cells.add(cell);
      return [...cells];
    }

    physicalEscape(state, g, pid, pred, firstMove, goal = null) {
      if (!state.projectPosition) return goal ? null : true;
      const player = state.players[pid];
      if (goal && player.cell === goal.cell) return [];
      const futureDanger = new Uint8Array((pred.T + 2) * g.N);
      for (let t = pred.T; t >= 1; t--) for (let c = 0; c < g.N; c++) {
        futureDanger[t * g.N + c] = pred.lethal[t * g.N + c] + futureDanger[(t + 1) * g.N + c];
      }
      let points = [{ y: player.y, x: player.x, ...(goal ? { path: [] } : {}) }];
      for (let tick = 1; tick <= (goal ? Math.min(pred.T, goal.deadline) : pred.T); tick++) {
        const next = new Map();
        for (const point of points) {
          for (let move = 0; move < 5; move++) {
            if (tick === 1 && firstMove != null && move !== firstMove) continue;
            if (tick === 1 && goal && goal.firstMask != null && !(goal.firstMask & (1 << move))) continue;
            const [y, x] = state.projectPosition(pid, point.y, point.x, move, pred, tick);
            const cell = Math.floor(y) * g.W + Math.floor(x);
            if (cell < 0 || cell >= g.N || (tick > player.invuln && pred.lethal[tick * g.N + cell])) continue;
            if (cell !== player.cell && state.fieldItem[cell] && tick > player.invuln) continue;
            if (goal && state.players.some(p => p.team !== player.team && p.alive && !p.trapped && p.bombsLeft > 0 && p.cell === cell)) continue;
            const route = goal ? point.path.concat(move) : null;
            if (goal && cell === goal.cell) return route;
            const key = `${Math.round(y * 5)},${Math.round(x * 5)}`;
            if (!next.has(key)) next.set(key, { y, x, ...(goal ? { path: route } : {}) });
          }
        }
        if (!next.size) return goal ? null : false;
        // Keep continuous alternatives near several exits instead of a single greedy retreat.
        points = [...next.values()].sort((a, b) => {
          const ca = Math.floor(a.y) * g.W + Math.floor(a.x), cb = Math.floor(b.y) * g.W + Math.floor(b.x);
          const target = goal ? { row: Math.floor(goal.cell / g.W), col: goal.cell % g.W } : null;
          const approach = target ? distance({ row: Math.floor(a.y), col: Math.floor(a.x) }, target) -
            distance({ row: Math.floor(b.y), col: Math.floor(b.x) }, target) : 0;
          return futureDanger[tick * g.N + ca] - futureDanger[tick * g.N + cb] + approach;
        }).slice(0, 48);
      }
      return goal ? null : true;
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
          if (n < 0 || n === blocked || (blocked instanceof Set && blocked.has(n)) ||
              !g.open[n] || g.bombAt[n] >= 0 || dist[n] >= 0) continue;
          dist[n] = dist[c] + 1; queue.push(n);
        }
      }
      return INF;
    }

    pickMove(g, me, acts, esc, field, threatened, danger, placing = false) {
      const state = this.decisionState;
      let staggerSafety = null;
      if (state && (this.lastGoalMode === 'STAGGER' || state.tick < this.staggerRetreatUntil)) {
        const early = { ...state, bombs: state.bombs.map(b => ({ ...b, fuse: Math.max(1, b.fuse - 3) })) };
        const safety = this.predict(early, g, [], () => true), insured = this.escape(g, safety, me, false).surv;
        let viable = 0;
        for (let a = 0; a < 5; a++) if ((acts & insured & (1 << a)) && this.physicalEscape(state, g, this.currentPid, safety, a)) viable |= 1 << a;
        if (viable) acts &= viable;
        else if (!placing) this.staggerPlan = null;
        if (this.lastGoalMode === 'STAGGER' && this.staggerPlan && !placing && viable) staggerSafety = safety;
      }
      if (state && state.projectPosition && this.alliesOf(state, this.currentPid).length &&
          state.bombs.some((b) => b.fuse <= 20)) {
        const physical = this.predict(state, g, [], () => true);
        let viable = 0;
        for (let a = 0; a < 5; a++) if ((acts & (1 << a)) &&
          this.physicalEscape(state, g, this.currentPid, physical, a)) viable |= 1 << a;
        if (viable) acts &= viable;
      }
      if (state && state.nextPosition && this.lastGoalMode !== 'RESCUE') {
        let separated = 0;
        let itemSafe = 0, glueMoves = 0;
        const physical = this.predict(state, g, [], () => true);
        const slowEscape = this.escape(g, physical, { ...me, speed: Math.min(me.speed, 0.6) }, false);
        for (let a = 0; a < 5; a++) {
          if (!(acts & (1 << a))) continue;
          const next = state.nextPosition(this.currentPid, a);
          const cell = Math.floor(next[0]) * g.W + Math.floor(next[1]);
          const neighbor = a < 4 ? g.nb[me.cell * 4 + a] : me.cell;
          const glue = state.fieldItem[cell] === 2 || (neighbor >= 0 && state.fieldItem[neighbor] === 2);
          if (glue && a < 4 && (slowEscape.surv & (1 << a))) glueMoves |= 1 << a;
          if (!glue && (cell === me.cell || !state.fieldItem[cell] ||
              (state.fieldItem[cell] === 1 && me.carrying >= 0))) itemSafe |= 1 << a;
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
        // Prefer a real bypass that advances the task. Idle is not a bypass.
        const progress = [0, 1, 2, 3].some(a => (itemSafe & (1 << a)) &&
          g.nb[me.cell * 4 + a] >= 0 && field[g.nb[me.cell * 4 + a]] < field[me.cell]);
        if (progress || threatened) { if (itemSafe) acts = itemSafe; }
        else if (glueMoves) acts = (itemSafe | glueMoves) & acts;
        else if (itemSafe & 15) acts = itemSafe & 15;
        else if (itemSafe) acts = itemSafe;
        if (staggerSafety && me.cell === this.staggerPlan?.cell && (itemSafe & separated & 16)) acts |= 16;
        if (separated & acts) acts &= separated;
      }
      if (staggerSafety && this.staggerPlan) {
        const plan = this.staggerPlan;
        if (me.cell === plan.cell && (acts & 16)) return 4;
        const route = this.physicalEscape(state, g, this.currentPid, staggerSafety, null,
          { cell: plan.cell, deadline: staggerSafety.bombGone[plan.anchor] - 1, firstMask: acts & 15 });
        if (route?.length) return route[0];
        this.staggerPlan = null;
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
      if (this.lastGoalMode === 'RESERVE' && this.reservePlan && me.cell === this.reservePlan.targetCell &&
          !threatened && (acts & 16) && esc.count[4] > 0) return 4;
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
          !['HUNT', 'SUPPORT', 'RAID', 'GUARD', 'DEFEND', 'INTERCEPT', 'ESCORT', 'CHAIN', 'RESERVE'].includes(this.lastGoalMode)) return null;
      const foes = this.enemiesOf(state, pid).map((q) => ({ ...state.players[q], pid: q })).filter((e) =>
        e.alive && !e.trapped && e.invuln < Base.NEW_BOMB_TICK);
      const nearFoes = foes.filter((e) => distance(me, e) <= this.cfg.attackRadius);
      if (!foes.length) return null;
      const mine = { cell: me.cell, e: Base.NEW_BOMB_TICK, blast: me.blast };
      const before = this.predict(state, g, [], () => true);
      const after = this.predict(state, g, [mine], () => true);
      const blast = this.blastCells(state, g, me.cell, me.blast);
      const oldDanger = this.dangerCells(before, g.N), danger = this.dangerCells(after, g.N);
      let reason = null, nextCell = null, chainAnchor, chainFuse, reserveTarget, reserveConnector;
      if (this.lastGoalMode === 'RESERVE' && this.reservePlan) {
        const anchor = state.bombs.find((b) => b.owner === pid && b.cell === this.reservePlan.anchor);
        if (!anchor || me.cell !== this.reservePlan.targetCell || anchor.fuse <= 6 || me.bombsLeft < 2) return null;
        const connector = this.reservePlan.connector;
        const connectorBlast = connector == null ? new Set() : this.blastCells(state, g, connector, me.blast);
        const hitCells = new Set([...connectorBlast, ...blast]);
        if (connector == null || !connectorBlast.has(anchor.cell) || !connectorBlast.has(me.cell) ||
            !foes.some((e) => this.pressureRouteCells(state, g, pid, e.pid, before).some((c) => hitCells.has(c)))) return null;
        reason = 'bomb_reserve_second';
        nextCell = connector;
        chainAnchor = anchor.cell;
        chainFuse = anchor.fuse;
      }
      const chainBlast = new Set(blast);
      if (this.chainPlan) for (const cell of this.chainPlan.linkedCells || []) {
        for (const c of this.blastCells(state, g, cell, me.blast)) chainBlast.add(c);
      }
      if (this.chainPlan && me.cell === this.chainPlan.cell && blast.has(this.chainPlan.anchor) &&
          foes.some((e) => chainBlast.has(e.cell) || chainBlast.has(this.predictedEnemyCell(state, g, e.pid,
            after.bombGone[me.cell])))) reason = 'bomb_chain';
      const home = new Set(this.baseCells(state, g, me.team));
      const carrier = this.alliesOf(state, pid).find((q) => state.players[q].alive && state.players[q].carrying >= 0);
      const protectedCells = carrier == null ? home : new Set([state.players[carrier].cell]);
      for (const foe of nearFoes) {
        if (reason) break;
        const approach = this.routeDistance(g, foe.cell, protectedCells);
        if (approach <= 7 && distance(me, foe) <= 5 && me.cell !== foe.cell &&
            this.routeDistance(g, foe.cell, protectedCells, me.cell) >= approach + 3) {
          reason = 'bomb_block'; break;
        }
        const futureCell = this.predictedEnemyCell(state, g, foe.pid, after.bombGone[me.cell]);
        if ((danger[foe.cell] && !oldDanger[foe.cell]) ||
            (danger[futureCell] && !oldDanger[futureCell] && after.bombGone[me.cell] <= 12)) {
          const linked = state.bombs.some((b) => state.players[b.owner] &&
            state.players[b.owner].team === me.team && blast.has(b.cell));
          reason = linked ? 'bomb_chain' : 'bomb_pressure'; break;
        }
        const route = this.routeCells(state, g, foe, [...protectedCells], before, 10);
        const covered = route.filter((c) => blast.has(c));
        if (this.alliesOf(state, pid).length && approach <= 10 && distance(me, foe) <= 6 && covered.length >= 2 &&
            this.routeDistance(g, foe.cell, protectedCells, blast) >= approach + 3 &&
            covered.some((c) => !oldDanger[c])) { reason = 'bomb_screen'; break; }
      }
      // First place an unlinked reserve bubble. A later connector is only accepted if it
      // reaches both reserves and an observed enemy route before either fuse expires.
      if (!reason && state.tick >= this.reserveRetryTick && this.cfg.multiReserve !== false &&
          this.alliesOf(state, pid).length && me.bombsLeft >= 2 &&
          me.liveBombs > 0 && me.liveBombs + 2 <= this.cfg.maxLiveBombs &&
          ['HUNT', 'SUPPORT', 'RAID', 'DEFEND', 'GUARD', 'INTERCEPT', 'ESCORT'].includes(this.lastGoalMode)) {
        const replenishment = this.findReserveReplenishment(state, g, pid, me, foes, before, blast);
        if (replenishment) {
          reason = 'bomb_reserve_replenish';
          nextCell = replenishment.connector;
          chainAnchor = replenishment.anchor;
          chainFuse = replenishment.fuse;
        }
      }
      if (!reason && state.tick >= this.reserveRetryTick && this.cfg.multiReserve !== false &&
          this.alliesOf(state, pid).length &&
          me.bombsLeft >= 3 && me.liveBombs === 0 && nearFoes.some((e) => distance(me, e) <= 7) &&
          ['HUNT', 'SUPPORT', 'RAID', 'DEFEND', 'GUARD', 'INTERCEPT', 'ESCORT'].includes(this.lastGoalMode)) {
        const separate = this.findSeparateReserve(state, g, pid, me, foes, after, blast);
        if (separate) {
          reason = 'bomb_reserve_multi';
          nextCell = separate.targetCell;
          reserveTarget = separate.targetCell;
          reserveConnector = separate.connector;
        }
      }
      // Existing connected reserve fallback remains available where no separated geometry exists.
      if (!reason && me.bombsLeft >= 2 && me.liveBombs === 0 && me.liveBombs + 1 < this.cfg.maxLiveBombs &&
          ['HUNT', 'SUPPORT'].includes(this.lastGoalMode)) {
        let best = INF;
        for (const cell of blast) {
          if (cell === me.cell || !g.open[cell] || g.bombAt[cell] >= 0) continue;
          const travel = this.routeDistance(g, me.cell, new Set([cell]));
          const ticks = Math.ceil(travel / this.stepLen(me, false));
          if (ticks < 1 || ticks + 10 >= after.bombGone[me.cell]) continue;
          const secondRay = this.blastCells(state, g, cell, me.blast);
          if (!nearFoes.some((e) => {
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
      return { move: this.pickMove(g, me, safe, escape, targetField, true, danger), reason, nextCell,
        chainAnchor, chainFuse, reserveTarget, reserveConnector };
    }

    findSeparateReserve(state, g, pid, me, foes, after, firstBlast) {
      const targets = new Set();
      for (const foe of foes) for (const cell of this.pressureRouteCells(state, g, pid, foe.pid, after)) targets.add(cell);
      const current = me.cell;
      let best = null;
      const candidates = [];
      for (let target = 0; target < g.N; target++) {
        if (!g.open[target] || g.bombAt[target] >= 0 || firstBlast.has(target)) continue;
        const travel = this.routeDistance(g, current, new Set([target]));
        const ticks = Math.ceil(travel / this.stepLen(me, false));
        if (travel >= INF || ticks < 1 || ticks + 12 >= after.bombGone[current]) continue;
        const secondBlast = this.blastCells(state, g, target, me.blast);
        if (![...targets].some((cell) => secondBlast.has(cell))) continue;
        let connector = -1, connectorCost = INF;
        for (const c of firstBlast) {
          if (!g.open[c] || g.bombAt[c] >= 0 || c === current || c === target) continue;
          const ray = this.blastCells(state, g, c, me.blast);
          if (!ray.has(current) || !ray.has(target) || ![...targets].some((cell) => ray.has(cell) || secondBlast.has(cell))) continue;
          const toConnector = this.routeDistance(g, target, new Set([c]));
          if (toConnector < connectorCost) { connectorCost = toConnector; connector = c; }
        }
        if (connector < 0 || connectorCost > 2 || ticks + connectorCost + 8 >= after.bombGone[current]) continue;
        const imminent = foes.some((e) => this.enemyPathCells(state, g, e.pid).some((c) => secondBlast.has(c)));
        // Route-only ambushes need nearby opposition and cover at least two cells
        // on its approach; a single speculative intersection is not worth two slots.
        if (!imminent && !foes.some((e) => distance(me, e) <= 5 &&
          this.pressureRouteCells(state, g, pid, e.pid, after).filter((c) => secondBlast.has(c)).length >= 2)) continue;
        const score = travel + connectorCost * 0.5 - this.solidNeighbors(state, g, target) * 0.4
          - (imminent ? 4 : 0);
        candidates.push({ targetCell: target, connector, score });
      }
      candidates.sort((a, b) => a.score - b.score || a.targetCell - b.targetCell);
      for (const candidate of candidates.slice(0, 6)) {
        const { targetCell, connector } = candidate;
        const future = { ...me, cell: connector, row: Math.floor(connector / g.W), col: connector % g.W,
          y: Math.floor(connector / g.W) + 0.5, x: connector % g.W + 0.5 };
        const chain = this.predict(state, g, [
          { cell: current, e: 12, blast: me.blast },
          { cell: targetCell, e: Base.NEW_BOMB_TICK, blast: me.blast },
          { cell: connector, e: Base.NEW_BOMB_TICK, blast: me.blast },
        ], () => true);
        const futureState = { ...state, players: state.players.map((p, q) => q === pid ? future : p) };
        if (!this.escape(g, chain, future, false).surv || this.alliesOf(state, pid).some((q) =>
          state.players[q].alive && !this.escape(g, chain, state.players[q], false).surv) ||
          !this.physicalEscape(futureState, g, pid, chain) || this.alliesOf(state, pid).some((q) =>
            state.players[q].alive && !this.physicalEscape(futureState, g, q, chain))) continue;
        best = candidate; break;
      }
      return best;
    }

    findReserveReplenishment(state, g, pid, me, foes, pred, ray) {
      const targets = new Set(foes.filter((e) => distance(me, e) <= 7).flatMap((e) =>
        this.pressureRouteCells(state, g, pid, e.pid, pred)));
      if (![...targets].some((c) => ray.has(c))) return null;
      const anchors = state.bombs.filter((b) => b.owner === pid && b.fuse > 12 && b.fuse <= 27 &&
        distance(me, b) <= 5 && !ray.has(b.cell) && !this.blastCells(state, g, b.cell, b.blast).has(me.cell));
      for (const anchor of anchors.sort((a, b) => b.fuse - a.fuse)) {
        for (const cell of ray) {
          if (cell === me.cell || !g.open[cell] || g.bombAt[cell] >= 0) continue;
          const connectorRay = this.blastCells(state, g, cell, me.blast);
          if (!connectorRay.has(anchor.cell) || !connectorRay.has(me.cell)) continue;
          const travel = this.routeDistance(g, me.cell, new Set([cell]));
          const ticks = Math.ceil(travel / this.stepLen(me, false));
          if (travel > 2 || ticks + 8 >= anchor.fuse) continue;
          const linked = this.predict(state, g, [
            { cell: me.cell, e: Base.NEW_BOMB_TICK, blast: me.blast },
            { cell, e: Math.max(1, anchor.fuse - 8), blast: me.blast },
          ], () => true);
          const future = { ...me, cell, row: Math.floor(cell / g.W), col: cell % g.W,
            y: Math.floor(cell / g.W) + .5, x: cell % g.W + .5 };
          const futureState = { ...state, players: state.players.map((p, q) => q === pid ? future : p) };
          if (!this.escape(g, linked, future, false).surv || this.alliesOf(state, pid).some((q) =>
            state.players[q].alive && !this.escape(g, linked, state.players[q], false).surv) ||
            !this.physicalEscape(futureState, g, pid, linked) || this.alliesOf(state, pid).some((q) =>
              state.players[q].alive && !this.physicalEscape(futureState, g, q, linked))) continue;
          return { anchor: anchor.cell, connector: cell, fuse: anchor.fuse };
        }
      }
      return null;
    }


  }

  return { BunCoopHunterBot, hunterStateFromSim: Base.hunterStateFromSim };
});
