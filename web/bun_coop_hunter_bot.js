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
      this.cfg.maxLiveBombs = Math.min(this.cfg.maxLiveBombs, 3);
    }

    reset(seed) {
      super.reset(seed);
      this.assignment = null;
      this.lastGoalMode = null;
    }

    analyzeSim(sim, pid) {
      const state = Base.hunterStateFromSim(sim);
      let pending = commitments.get(sim);
      if (!pending || pending.tick !== sim.t || pending.generation !== sim._gen) {
        pending = { tick: sim.t, generation: sim._gen, bombs: [], moves: {} };
        commitments.set(sim, pending);
      }
      const team = state.players[pid].team;
      // Later teammates see bombs committed earlier in this tick before choosing their escape.
      state.bombs = state.bombs.concat(pending.bombs.filter((b) => state.players[b.owner].team === team));
      state.committedMoves = pending.moves;
      const decision = this.analyze(state, pid);
      pending.moves[pid] = decision.action[0];
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
      return [decision.action[0], decision.action[1], 0];
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
      const goal = this.selectGoal(state, g, pid, pred, perceive);
      this.lastGoalMode = goal.mode;
      return goal;
    }

    fieldForGoal(state, g, goal, me, pred, foe) {
      return this.goalField(state, g, goal.seeds, me, pred, goal.threat ? foe : null,
        goal.noDig == null ? me.carrying >= 0 : goal.noDig);
    }

    selectGoal(state, g, pid, pred, perceive) {
      const me = state.players[pid], team = me.team == null ? pid : me.team;
      const allies = this.alliesOf(state, pid);
      const role = this.teamRole(state, pid);
      this.assignment = { role, target: null };
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
        // Stay between the carrier and the nearest pursuer without chasing beyond the escort radius.
        const seeds = [];
        for (let cell = 0; cell < g.N; cell++) {
          if (!g.open[cell] || g.bombAt[cell] >= 0) continue;
          const point = { row: Math.floor(cell / g.W), col: cell % g.W };
          const d = distance(point, ally);
          if (d < 1 || d > 2) continue;
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
      return Object.assign({}, opportunity, { mode: opportunity.mode === 'HUNT' ? 'SUPPORT' : opportunity.mode });
    }

    attackSeeds(state, g, pid, pred, perceive) {
      const seeds = super.attackSeeds(state, g, pid, pred, perceive);
      const foe = this.foeOf(state, pid);
      // Prefer close firing positions to distant cells on the same ray.
      return seeds.map((s) => ({ cell: s.cell, cost: s.cost + Math.max(0,
        Math.abs(Math.floor(s.cell / g.W) - foe.row) + Math.abs(s.cell % g.W - foe.col) - 2) * 0.7 }));
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
      this.assignment = null;
      const decision = super.analyze(state, pid);
      return Object.assign(decision, this.assignment || { role: 'SOLO', target: null });
    }

    considerBomb(state, g, pid, pred, perceive, field) {
      if (this.lastGoalMode === 'RESCUE') return null;
      const candidate = super.considerBomb(state, g, pid, pred, perceive, field);
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
      return candidate;
    }

  }

  return { BunCoopHunterBot, hunterStateFromSim: Base.hunterStateFromSim };
});
