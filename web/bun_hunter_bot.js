/**
 * BunHunterBot - 抢包子网页对战规则 Bot（猎手）。
 * 时间展开爆炸预测 + 逃生 BFS 躲泡；按对手逃生空间收缩评估进攻；挖砖开路；
 * 对手阵亡/远离时偷包并搬回己方包子屋。难度由感知延迟、连锁感知、防围堵、失误率等参数区分。
 */
'use strict';

(function bunHunterFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) {
    root.BunHunterBot = api.BunHunterBot;
    root.QQTBunHunterBot = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildBunHunterBot() {
  const MOVE_IDLE = 4;
  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const STEP_LEN = 0.3;
  const BOMB_FUSE = 30;
  // 放泡发生在下一 tick 内，之后再走 30 步引信：相对当前状态第 31 步爆炸。
  const NEW_BOMB_TICK = BOMB_FUSE + 1;
  const FLAME_TICKS = 3;
  const BRICK_OPEN_DELAY = 5;
  const MAX_T = 64;
  const FOE_HORIZON = NEW_BOMB_TICK + FLAME_TICKS;
  // 火柱按中线判定：身体（半径 0.36）离开格中心约 0.36 格即脱离，留 0.09 余量。
  const LEAVE_DIST = 0.45;
  const INF = 1e9;

  const DIFFICULTIES = {
    easy: {
      label: '简单', reactionDelay: 4, chainAware: false, robust: 0, mistakeRate: 0.1,
      bombHesitation: 0.4, maxLiveBombs: 1, attackRadius: 3, lineOnly: true,
      footRatio: 1.01, pressureRatio: 0, itemRadius: 0, itemSeed: 0, raidMargin: Infinity,
      threatPenalty: 0, trapSearch: false, brickCost: 6,
    },
    normal: {
      label: '普通', reactionDelay: 1, chainAware: true, robust: 1, mistakeRate: 0.02,
      bombHesitation: 0.05, maxLiveBombs: 2, attackRadius: 5, lineOnly: false,
      footRatio: 1.0, pressureRatio: 0.3, itemRadius: 4, itemSeed: 2, raidMargin: 6,
      threatPenalty: 2, trapSearch: false, brickCost: 5,
    },
    hard: {
      label: '困难', reactionDelay: 0, chainAware: true, robust: 2, mistakeRate: 0,
      bombHesitation: 0, maxLiveBombs: 10, attackRadius: 7, lineOnly: false,
      footRatio: 1.0, pressureRatio: 0.4, itemRadius: 6, itemSeed: 1, raidMargin: 2,
      threatPenalty: 3, trapSearch: true, brickCost: 4,
    },
  };

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function next() {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function manhattan(a, b) { return Math.abs(a.row - b.row) + Math.abs(a.col - b.col); }

  function hunterStateFromSim(sim) {
    const height = Number(sim.H || 13);
    const width = Number(sim.W || (sim.wall.length / height));
    const N = height * width;
    const bombs = [];
    const count = sim.nPlayers || 2;
    const teams = sim.team ? sim.team.slice() : [0, 1];
    const live = new Array(count).fill(0);
    for (let cell = 0; cell < N; cell++) {
      if (sim.fuse[cell] <= 0) continue;
      const owner = Number(sim.owner[cell]);
      if (owner >= 0 && owner < count) live[owner]++;
      bombs.push({ cell, row: Math.floor(cell / width), col: cell % width,
        fuse: Number(sim.fuse[cell]), blast: Number(sim.bombBlast[cell] || 2), owner });
    }
    const players = teams.map((team, p) => {
      const y = Number(sim.pos[p * 2]), x = Number(sim.pos[p * 2 + 1]);
      const row = Math.max(0, Math.min(height - 1, Math.floor(y)));
      const col = Math.max(0, Math.min(width - 1, Math.floor(x)));
      const carrying = sim.bunCarried ? Number(sim.bunCarried[p]) : -1;
      const scale = typeof sim.playerMoveScale === 'function' ? sim.playerMoveScale(p) : 1;
      const bombsCap = Number(sim.bombsCap[p]);
      return {
        y, x, row, col, cell: row * width + col, alive: !!sim.alive[p],
        bombsCap, liveBombs: live[p], bombsLeft: carrying >= 0 ? 0 : Math.max(0, bombsCap - live[p]),
        blast: Number(sim.blastCap[p]), speed: Number(sim.spdG[p]) * scale, carrying,
        invuln: sim.invuln ? Number(sim.invuln[p]) : 0,
        respawn: sim.bunRespawn ? Number(sim.bunRespawn[p]) : 0,
        team, trapped: sim.trapped ? Number(sim.trapped[p]) || 0 : 0,
      };
    });
    const zeros = new Uint8Array(N);
    return {
      schema: 'qqt.bun.hunter_state/v1', height, width, tick: sim.t,
      wall: sim.wall, brick: sim.brick, brickLinger: sim.brickLinger || zeros,
      blastLinger: sim.blastLinger || zeros, crate: sim.crate || zeros, fieldItem: sim.fieldItem || zeros,
      crateType: sim.crateType || new Int8Array(N).fill(-1), heldItem: (sim.heldItem || [0, 0]).slice(),
      bombs, players, teams,
      bunBases: (sim.bunBases || [[1, 4], [1, 8]]).map((b) => b.slice()),
      bunStored: (sim.bunStored || [[1, 0], [0, 1]]).map((b) => b.slice()),
      bunLoose: sim.bunLoose || new Uint8Array(N * 2),
    };
  }

  class BunHunterBot {
    constructor(options = {}) {
      const name = options.difficulty || 'normal';
      if (!DIFFICULTIES[name]) throw new Error(`unknown hunter difficulty: ${name}`);
      this.difficulty = name;
      this.cfg = Object.assign({}, DIFFICULTIES[name], options.overrides || {});
      this.seed = options.seed == null ? 0x68756e74 : options.seed >>> 0;
      this.nbKey = '';
      this.nb = null;
      this.reachBuf = null;
      this.reset();
    }

    reset(seed) {
      if (seed != null) this.seed = seed >>> 0;
      this.rng = mulberry32(this.seed);
      this.lastDecision = null;
      this.itemTarget = -1;
    }

    act(sim, playerId = 1) {
      const decision = this.analyze(hunterStateFromSim(sim), playerId);
      return [decision.action[0], decision.action[1], 0];
    }

    // 组队：敌人 = 异队；当前对手 foe = 最近的未被糖泡困住的存活敌人（都不满足时退回第一个敌人）。
    // 1v1 时恒为 1 - pid，与原行为一致。
    enemiesOf(state, pid) {
      const team = state.players[pid].team == null ? pid : state.players[pid].team;
      const out = [];
      state.players.forEach((p, q) => { if (q !== pid && (p.team == null ? q : p.team) !== team) out.push(q); });
      return out;
    }

    alliesOf(state, pid) {
      const team = state.players[pid].team;
      const out = [];
      state.players.forEach((p, q) => { if (q !== pid && team != null && p.team === team) out.push(q); });
      return out;
    }

    foeOf(state, pid) {
      if (!state._foe) state._foe = {};
      if (state._foe[pid] != null) return state.players[state._foe[pid]];
      const me = state.players[pid];
      const enemies = this.enemiesOf(state, pid);
      let best = enemies.length ? enemies[0] : 1 - pid, bestD = INF;
      for (const q of enemies) {
        const e = state.players[q];
        if (!e.alive || e.trapped > 0) continue;
        const d = manhattan(me, e);
        if (d < bestD) { bestD = d; best = q; }
      }
      state._foe[pid] = best;
      return state.players[best];
    }

    analyze(state, pid) {
      const g = this.geometry(state);
      const me = state.players[pid], foe = this.foeOf(state, pid);
      if (!me.alive) return this.finish(MOVE_IDLE, 0, 'dead', 'DEAD');
      if (me.trapped > 0) return this.finish(MOVE_IDLE, 0, 'trapped', 'TRAPPED');
      const cfg = this.cfg;
      // 自己和队友放的泡立即可见（队友间默契），敌方泡按反应延迟感知。
      const friendly = (owner) => owner === pid ||
        (state.players[owner] && me.team != null && state.players[owner].team === me.team);
      const perceive = (bomb) => friendly(bomb.owner) || bomb.fuse <= BOMB_FUSE - cfg.reactionDelay;
      const pred = this.predict(state, g, [], perceive);
      const esc = this.escape(g, pred, me, false);
      const goal = this.chooseGoal(state, g, pid, pred, perceive);
      const field = this.fieldForGoal(state, g, goal, me, pred, foe);
      if (!esc.surv) {
        let best = MOVE_IDLE, bestT = -1;
        for (let a = 0; a < 5; a++) if (esc.maxT[a] > bestT) { bestT = esc.maxT[a]; best = a; }
        return this.finish(best, 0, 'doomed_max_survival', goal.mode);
      }
      const bomb = this.considerBomb(state, g, pid, pred, perceive, field);
      if (bomb) return this.finish(bomb.move, 1, bomb.reason, goal.mode);

      let acts = esc.surv;
      if (cfg.robust) {
        const robust = acts & this.hypoSurvivors(state, g, pid, perceive, []);
        if (robust) acts = robust;
      }
      const threatened = this.threatened(pred, me, g.N);
      if (cfg.mistakeRate > 0 && this.rng() < cfg.mistakeRate) {
        const options = [];
        for (let a = 0; a < 5; a++) if (acts & (1 << a)) options.push(a);
        return this.finish(options[Math.floor(this.rng() * options.length)], 0, 'mistake', goal.mode);
      }
      const move = this.pickMove(g, me, acts, esc, field, threatened, this.dangerCells(pred, g.N));
      const reason = threatened ? 'dodge' : (move === MOVE_IDLE ? 'hold' : 'route');
      return this.finish(move, 0, reason, goal.mode);
    }

    finish(move, ability, reason, mode) {
      this.lastDecision = { action: [move, ability], reason, mode, difficulty: this.difficulty };
      return this.lastDecision;
    }

    fieldForGoal(state, g, goal, me, pred, foe) {
      return this.goalField(state, g, goal.seeds, me, pred, goal.threat ? foe : null,
        goal.noDig !== false && me.carrying >= 0);
    }

    geometry(state) {
      const H = state.height, W = state.width, N = H * W;
      const key = `${H}x${W}`;
      if (key !== this.nbKey) {
        const nb = new Int16Array(N * 4).fill(-1);
        for (let r = 0; r < H; r++) {
          for (let c = 0; c < W; c++) {
            for (let a = 0; a < 4; a++) {
              const nr = r + DIRS[a][0], nc = c + DIRS[a][1];
              if (nr >= 0 && nr < H && nc >= 0 && nc < W) nb[(r * W + c) * 4 + a] = nr * W + nc;
            }
          }
        }
        this.nb = nb;
        this.nbKey = key;
        this.reachBuf = new Uint8Array((MAX_T + 1) * N);
      }
      const open = new Uint8Array(N);
      for (let c = 0; c < N; c++) open[c] = state.wall[c] || state.brick[c] ? 0 : 1;
      const bombAt = new Int16Array(N).fill(-1);
      state.bombs.forEach((bomb, index) => { bombAt[bomb.cell] = index; });
      return { H, W, N, nb: this.nb, open, bombAt };
    }

    // 时间展开爆炸预测：lethal[t*N+c]=1 表示第 t 步伤害判定时 c 有火；bombGone[c]=该格泡泡爆炸 tick。
    predict(state, g, extra, perceive) {
      const { N, W, H } = g;
      const chain = this.cfg.chainAware;
      const bombs = [];
      for (const b of state.bombs) {
        if (perceive && !perceive(b)) continue;
        bombs.push({ cell: b.cell, e: b.fuse, blast: b.blast, done: 0 });
      }
      for (const b of extra) bombs.push({ cell: b.cell, e: b.e, blast: b.blast, done: 0 });
      const bombAt = new Int16Array(N).fill(-1);
      bombs.forEach((b, i) => { bombAt[b.cell] = i; });
      const lethal = new Uint8Array((MAX_T + 1) * N);
      const impact = new Uint8Array((MAX_T + 1) * N);
      const sources = [];
      let last = 0;
      for (let c = 0; c < N; c++) {
        const linger = Math.min(MAX_T, state.blastLinger[c] || 0);
        for (let t = 1; t <= linger; t++) lethal[t * N + c] = 1;
        if (linger > last) last = linger;
      }
      const brickGone = new Int16Array(N);
      for (let c = 0; c < N; c++) {
        if (state.brick[c] && state.brickLinger[c] > 0) brickGone[c] = state.brickLinger[c] + 1;
      }
      const mark = (cell, t) => {
        impact[t * N + cell] = 1;
        const end = Math.min(MAX_T, t + FLAME_TICKS - 1);
        for (let tau = t; tau <= end; tau++) lethal[tau * N + cell] = 1;
        if (end > last) last = end;
      };
      const order = bombs.map((_, i) => i).sort((a, b) => bombs[a].e - bombs[b].e);
      for (const first of order) {
        if (bombs[first].done) continue;
        const t = bombs[first].e;
        if (t > MAX_T) break;
        const queue = [];
        for (const j of order) if (!bombs[j].done && bombs[j].e === t) { bombs[j].done = t; queue.push(j); }
        for (let q = 0; q < queue.length; q++) {
          const b = bombs[queue[q]];
          const covered = [];
          sources.push({ cell: b.cell, tick: t, covered });
          covered.push(b.cell);
          mark(b.cell, t);
          const br = Math.floor(b.cell / W), bc = b.cell % W;
          for (let a = 0; a < 4; a++) {
            for (let k = 1; k <= b.blast; k++) {
              const r = br + DIRS[a][0] * k, c = bc + DIRS[a][1] * k;
              if (r < 0 || r >= H || c < 0 || c >= W) break;
              const cell = r * W + c;
              if (state.wall[cell]) break;
              covered.push(cell);
              mark(cell, t);
              const j = bombAt[cell];
              // 同 tick 自然引爆的泡引信已归零不挡火；其余仍在场的泡挡火并被连锁。
              if (j >= 0 && bombs[j].e !== t && (!bombs[j].done || bombs[j].done === t)) {
                if (!bombs[j].done && chain) { bombs[j].done = t; queue.push(j); }
                break;
              }
              if (state.brick[cell] && !(brickGone[cell] && t >= brickGone[cell])) {
                if (!brickGone[cell] || brickGone[cell] > t + BRICK_OPEN_DELAY) brickGone[cell] = t + BRICK_OPEN_DELAY;
                break;
              }
            }
          }
        }
      }
      const bombGone = new Int16Array(N);
      for (const b of bombs) bombGone[b.cell] = b.done || b.e;
      // 未感知的泡看不到火势，但仍是物理障碍。
      for (const b of state.bombs) if (!bombGone[b.cell]) bombGone[b.cell] = b.fuse;
      const T = Math.min(MAX_T, Math.max(2, last + 1));
      return { lethal, impact, sources, bombGone, brickGone, T };
    }

    moveTicks(player, optimistic) {
      const k = Math.max(1, Math.ceil(1 / (STEP_LEN * player.speed) - 1e-9));
      return optimistic ? Math.max(1, k - 1) : k;
    }

    stepLen(player, optimistic) {
      return STEP_LEN * player.speed * (optimistic ? 1.25 : 1);
    }

    // 从实际坐标朝方向 a 进入相邻格：先回中线（sim 转向会吃掉该 tick 余量），再沿轴前进。
    // leave = 身体越过起点格火柱中线判定区的 tick；arrive = 到达目标格中心的 tick。
    firstMove(player, a, optimistic) {
      const step = this.stepLen(player, optimistic);
      const cy = player.row + 0.5, cx = player.col + 0.5;
      const vertical = a < 2;
      const perp = vertical ? Math.abs(player.x - cx) : Math.abs(player.y - cy);
      const pos = vertical ? player.y : player.x;
      const center = vertical ? cy : cx;
      const sign = a === 0 || a === 2 ? -1 : 1;
      const perpTicks = perp > 0.02 ? Math.ceil(perp / step - 1e-9) : 0;
      const along = Math.abs(center + sign - pos);
      const leaveDist = Math.max(0, sign * (center + sign * LEAVE_DIST - pos));
      return {
        arrive: Math.max(1, perpTicks + Math.ceil(along / step - 1e-9)),
        leave: perpTicks + Math.ceil(leaveDist / step - 1e-9),
      };
    }

    firstTicks(g, player, target, optimistic) {
      const ty = Math.floor(target / g.W) + 0.5, tx = target % g.W + 0.5;
      const d = Math.abs(ty - player.y) + Math.abs(tx - player.x);
      return Math.max(1, Math.ceil(d / this.stepLen(player, optimistic) - 1e-9));
    }

    // 时间展开逃生 BFS：移动过渡期间同时占用起止两格（保守）；optimistic 用于估计真人的逃生能力。
    escape(g, pred, player, optimistic, horizon) {
      const { N, nb, open } = g;
      const { lethal, bombGone: gone } = pred;
      const T = horizon ? Math.min(MAX_T, Math.max(pred.T, horizon)) : pred.T;
      const inv = player.invuln || 0;
      const step = this.stepLen(player, optimistic);
      const k = Math.max(1, Math.ceil(1 / step - 1e-9));
      const leaveK = Math.ceil(LEAVE_DIST / step - 1e-9);
      const reach = this.reachBuf;
      reach.fill(0, 0, (T + 1) * N);
      const start = player.cell;
      const bad = (t, c) => t > inv && lethal[t * N + c] === 1;
      if (!bad(1, start)) reach[N + start] |= 16;
      const startMoves = [0, 1, 2, 3].map((a) => this.firstMove(player, a, optimistic));
      for (let a = 0; a < 4; a++) {
        const n = nb[start * 4 + a];
        if (n < 0 || !open[n] || gone[n] > 1) continue;
        const move = startMoves[a];
        const arrive = Math.min(T, move.arrive);
        let ok = true;
        for (let tau = 1; tau <= arrive && ok; tau++) {
          if (bad(tau, n) || (tau <= move.leave && bad(tau, start))) ok = false;
        }
        if (ok) reach[arrive * N + n] |= 1 << a;
      }
      for (let t = 1; t < T; t++) {
        const base = t * N;
        const arrive = Math.min(T, t + k);
        const leave = t + leaveK;
        for (let c = 0; c < N; c++) {
          const m = reach[base + c];
          if (!m) continue;
          if (!bad(t + 1, c)) reach[base + N + c] |= m;
          for (let a = 0; a < 4; a++) {
            const n = nb[c * 4 + a];
            if (n < 0 || !open[n] || gone[n] > t + 1) continue;
            // 停留在起点格时角色仍在原坐标（可能偏离中线），按真实坐标计时。
            let arriveA = arrive, leaveA = leave;
            if (c === start) {
              const move = startMoves[a];
              arriveA = Math.min(T, t + move.arrive);
              leaveA = t + move.leave;
            }
            let ok = true;
            for (let tau = t + 1; tau <= arriveA && ok; tau++) {
              if (bad(tau, n) || (tau <= leaveA && bad(tau, c))) ok = false;
            }
            if (ok) reach[arriveA * N + n] |= m;
          }
        }
      }
      const count = [0, 0, 0, 0, 0];
      let surv = 0, endCount = 0;
      const endBase = T * N;
      for (let c = 0; c < N; c++) {
        const m = reach[endBase + c];
        if (!m) continue;
        endCount++;
        surv |= m;
        for (let a = 0; a < 5; a++) if (m & (1 << a)) count[a]++;
      }
      const maxT = [-1, -1, -1, -1, -1];
      let seen = 0;
      for (let t = T; t >= 1 && seen !== 31; t--) {
        let m = 0;
        for (let c = 0; c < N; c++) m |= reach[t * N + c];
        const fresh = m & ~seen;
        for (let a = 0; a < 5; a++) if (fresh & (1 << a)) maxT[a] = t;
        seen |= m;
      }
      return { surv, count, endCount, maxT };
    }

    threatened(pred, player, N) {
      for (let t = 1; t <= pred.T; t++) {
        if (t > (player.invuln || 0) && pred.lethal[t * N + player.cell]) return true;
      }
      return false;
    }

    hypoFoeBombs(state, g, pid) {
      const result = [];
      for (const q of this.enemiesOf(state, pid)) result.push(...this.hypoBombsOf(state, g, pid, state.players[q]));
      return result;
    }

    hypoBombsOf(state, g, pid, foe) {
      const me = state.players[pid];
      if (!foe.alive || foe.trapped > 0 || foe.carrying >= 0 || foe.bombsLeft <= 0) return [];
      if (manhattan(me, foe) > foe.blast + 4) return [];
      const result = [];
      if (g.bombAt[foe.cell] < 0 && !state.brick[foe.cell]) {
        result.push({ cell: foe.cell, e: NEW_BOMB_TICK, blast: foe.blast });
      }
      if (this.cfg.robust >= 2) {
        let best = -1, bestD = manhattan(me, foe);
        for (let a = 0; a < 4; a++) {
          const n = g.nb[foe.cell * 4 + a];
          if (n < 0 || !g.open[n] || g.bombAt[n] >= 0) continue;
          const d = Math.abs(Math.floor(n / g.W) - me.row) + Math.abs(n % g.W - me.col);
          if (d < bestD) { bestD = d; best = n; }
        }
        if (best >= 0) {
          result.push({ cell: best, e: NEW_BOMB_TICK + this.firstTicks(g, foe, best, true), blast: foe.blast });
        }
      }
      return result;
    }

    // 假设对手立刻在自身格（困难档再加上朝我方一步的格）放泡后，我方仍能逃生的首步集合。
    hypoSurvivors(state, g, pid, perceive, mine) {
      const me = state.players[pid];
      let mask = 31;
      for (const hypo of this.hypoFoeBombs(state, g, pid)) {
        const pred = this.predict(state, g, mine.concat([hypo]), perceive);
        mask &= this.escape(g, pred, me, false).surv;
        if (!mask) break;
      }
      return mask;
    }

    baseCells(state, g, team) {
      const anchor = state.bunBases[team];
      const cells = [];
      if (!anchor) return cells;
      for (let r = anchor[0]; r < anchor[0] + 3; r++) {
        for (let c = anchor[1]; c < anchor[1] + 3; c++) {
          if (r < 0 || r >= g.H || c < 0 || c >= g.W) continue;
          if (!state.wall[r * g.W + c]) cells.push(r * g.W + c);
        }
      }
      return cells;
    }

    itemSeeds(state, g, me, pid) {
      const seeds = [];
      if (this.cfg.itemRadius <= 0) return seeds;
      const holding = state.heldItem && state.heldItem[pid] > 0;
      const usable = (c) => {
        if (!state.crate[c] || !g.open[c] || c === me.cell) return false;
        // 香蕉/慢慢胶只能持有一个，手里有道具时踩上去不会拾取。
        const type = state.crateType ? state.crateType[c] : -1;
        return !(holding && (type === 3 || type === 4));
      };
      // 锁定一个道具直到拿到或消失，避免道具在半径边缘进出导致来回摆动。
      if (this.itemTarget >= 0 && usable(this.itemTarget)) {
        return [{ cell: this.itemTarget, cost: this.cfg.itemSeed }];
      }
      this.itemTarget = -1;
      let best = -1, bestD = this.cfg.itemRadius + 1;
      for (let c = 0; c < g.N; c++) {
        if (!usable(c)) continue;
        const d = Math.abs(Math.floor(c / g.W) - me.row) + Math.abs(c % g.W - me.col);
        if (d < bestD) { bestD = d; best = c; }
      }
      if (best >= 0) {
        this.itemTarget = best;
        seeds.push({ cell: best, cost: this.cfg.itemSeed });
      }
      return seeds;
    }

    attackSeeds(state, g, pid, pred, perceive) {
      const me = state.players[pid], foe = this.foeOf(state, pid);
      const cells = new Map();
      const add = (cell, cost) => {
        if (!g.open[cell] || g.bombAt[cell] >= 0) return;
        if (!cells.has(cell) || cells.get(cell) > cost) cells.set(cell, cost);
      };
      add(foe.cell, 0);
      for (let a = 0; a < 4; a++) {
        for (let k = 1; k <= me.blast; k++) {
          const r = foe.row + DIRS[a][0] * k, c = foe.col + DIRS[a][1] * k;
          if (r < 0 || r >= g.H || c < 0 || c >= g.W) break;
          const cell = r * g.W + c;
          if (state.wall[cell] || state.brick[cell]) break;
          add(cell, 0);
        }
      }
      if (this.cfg.trapSearch && me.carrying < 0) {
        const candidates = [];
        for (let c = 0; c < g.N; c++) {
          if (!g.open[c] || g.bombAt[c] >= 0) continue;
          const d = Math.abs(Math.floor(c / g.W) - foe.row) + Math.abs(c % g.W - foe.col);
          if (d <= 3) candidates.push([d, c]);
        }
        candidates.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
        const baseCount = Math.max(1, this.escape(g, pred, foe, true, FOE_HORIZON).endCount);
        for (const [, cell] of candidates.slice(0, 16)) {
          const travel = Math.abs(Math.floor(cell / g.W) - me.row) + Math.abs(cell % g.W - me.col);
          const hypo = { cell, e: NEW_BOMB_TICK + travel * this.moveTicks(me, false), blast: me.blast };
          const trapped = this.escape(g, this.predict(state, g, [hypo], perceive), foe, true, FOE_HORIZON).endCount;
          add(cell, trapped === 0 ? 0 : 1 + Math.round(4 * trapped / baseCount));
        }
      }
      return Array.from(cells, ([cell, cost]) => ({ cell, cost }));
    }

    chooseGoal(state, g, pid, pred, perceive) {
      const cfg = this.cfg;
      const me = state.players[pid], foe = this.foeOf(state, pid);
      const team = me.team == null ? pid : me.team, enemy = 1 - team;
      const ownBase = this.baseCells(state, g, team).map((cell) => ({ cell, cost: 0 }));
      const enemyBase = this.baseCells(state, g, enemy).map((cell) => ({ cell, cost: 0 }));
      const looseEnemy = [], looseOwn = [];
      for (let c = 0; c < g.N; c++) {
        if (state.bunLoose[c * 2 + enemy] > 0) looseEnemy.push({ cell: c, cost: 0 });
        if (state.bunLoose[c * 2 + team] > 0) looseOwn.push({ cell: c, cost: 0 });
      }
      const enemyStock = (state.bunStored[enemy] && state.bunStored[enemy][enemy] > 0) ? enemyBase : [];
      // 糖泡协作：赶得到就去碰被困队友（救出）或被困敌人（爆破）。
      const rescue = this.trapGoal(state, g, pid, pred, this.alliesOf(state, pid), 'RESCUE');
      const pop = this.trapGoal(state, g, pid, pred, this.enemiesOf(state, pid), 'POP');
      if (rescue && (me.carrying < 0 || rescue.steps <= 4)) return rescue;
      if (me.carrying >= 0) {
        // 搬运中不能放泡：回家路被砖堵死时退而求其次，靠近可挖路线等待。
        const home = this.goalField(state, g, ownBase, me, pred, null, true);
        return { mode: 'DELIVER', seeds: ownBase, threat: true, noDig: home[me.cell] < INF };
      }
      // 搬运途中无法挖砖，先确保己方包子屋与当前位置连通，再去偷包。
      const homeOpen = this.goalField(state, g, ownBase, me, pred, null, true)[me.cell] < INF;
      const openRoute = { mode: 'OPEN_ROUTE', seeds: ownBase, threat: false };
      if (pop) return pop;
      if (!foe.alive || foe.trapped > 0) {
        const seeds = looseEnemy.concat(enemyStock, looseOwn);
        if (seeds.length) return homeOpen ? { mode: 'RAID', seeds, threat: false } : openRoute;
        return { mode: 'GUARD', seeds: looseOwn.length ? looseOwn : ownBase, threat: false };
      }
      if (foe.carrying === team) {
        return { mode: 'INTERCEPT', seeds: this.attackSeeds(state, g, pid, pred, perceive), threat: false };
      }
      if (looseEnemy.length) return homeOpen ? { mode: 'STEAL', seeds: looseEnemy, threat: true } : openRoute;
      if (looseOwn.length) return { mode: 'RECOVER', seeds: looseOwn, threat: true };
      if (enemyStock.length && Number.isFinite(cfg.raidMargin)) {
        const dist = this.goalField(state, g, enemyStock, me, pred, null);
        if (dist[foe.cell] >= dist[me.cell] + cfg.raidMargin) {
          return homeOpen ? { mode: 'RAID', seeds: enemyStock, threat: true } : openRoute;
        }
      }
      const seeds = this.attackSeeds(state, g, pid, pred, perceive).concat(this.itemSeeds(state, g, me, pid));
      return { mode: 'HUNT', seeds, threat: false };
    }

    // 以被困目标所在格为目标（进入同格必在 41px 接触范围内；邻格只在靠近中心时才够）；
    // 只在预计到达时间早于糖泡自动爆破时才追。
    trapGoal(state, g, pid, pred, candidates, mode) {
      const me = state.players[pid];
      let best = null;
      for (const q of candidates) {
        const target = state.players[q];
        if (!target.alive || !(target.trapped > 0)) continue;
        const seeds = [{ cell: target.cell, cost: 0 }];
        const steps = this.goalField(state, g, seeds, me, pred, null, true)[me.cell];
        if (!(steps < INF / 2)) continue;
        if (steps * this.moveTicks(me, false) >= target.trapped) continue;
        if (!best || steps < best.steps) best = { mode, seeds, threat: false, noDig: true, steps, target: q };
      }
      return best;
    }

    threatMap(state, g, foe) {
      const threat = new Uint8Array(g.N);
      if (!foe || !foe.alive || this.cfg.threatPenalty <= 0) return threat;
      const reach = foe.blast + 1;
      for (let c = 0; c < g.N; c++) {
        const d = Math.abs(Math.floor(c / g.W) - foe.row) + Math.abs(c % g.W - foe.col);
        if (d <= 2) threat[c] = 1;
      }
      for (let a = 0; a < 4; a++) {
        for (let k = 1; k <= reach; k++) {
          const r = foe.row + DIRS[a][0] * k, c = foe.col + DIRS[a][1] * k;
          if (r < 0 || r >= g.H || c < 0 || c >= g.W) break;
          const cell = r * g.W + c;
          if (state.wall[cell] || state.brick[cell]) break;
          threat[cell] = 1;
        }
      }
      return threat;
    }

    enterCost(state, g, cell, me, pred, threat, noDig) {
      if (state.wall[cell]) return INF;
      if (state.brick[cell]) {
        const opens = pred.brickGone[cell];
        if (opens > 0) return 1 + Math.ceil(opens / this.moveTicks(me, false));
        return noDig ? INF : this.cfg.brickCost;
      }
      let cost = 1;
      if (threat && threat[cell]) cost += this.cfg.threatPenalty;
      if (pred.bombGone[cell] > 0) cost += 3;
      if (state.fieldItem[cell]) cost += 3;
      return cost;
    }

    // 反向 Dijkstra：dist[c] = 从 c 走到任一目标的代价（砖按挖掘代价计入）。
    goalField(state, g, seeds, me, pred, foe, noDig = me.carrying >= 0) {
      const { N, nb } = g;
      const dist = new Float64Array(N).fill(INF);
      if (!seeds.length) return dist;
      const threat = foe ? this.threatMap(state, g, foe) : null;
      const cost = new Float64Array(N);
      for (let c = 0; c < N; c++) cost[c] = this.enterCost(state, g, c, me, pred, threat, noDig);
      for (const seed of seeds) if (seed.cost < dist[seed.cell]) dist[seed.cell] = seed.cost;
      const done = new Uint8Array(N);
      for (let iter = 0; iter < N; iter++) {
        let u = -1, best = INF;
        for (let c = 0; c < N; c++) if (!done[c] && dist[c] < best) { best = dist[c]; u = c; }
        if (u < 0) break;
        done[u] = 1;
        if (cost[u] >= INF) continue;
        for (let a = 0; a < 4; a++) {
          const c = nb[u * 4 + a];
          if (c < 0 || done[c] || state.wall[c]) continue;
          const candidate = best + cost[u];
          if (candidate < dist[c]) dist[c] = candidate;
        }
      }
      return dist;
    }

    // 路线上的下一格若是未被预测炸开的砖，返回该砖格：需要放泡挖开。
    digTarget(state, g, me, field, pred) {
      const start = me.cell;
      let bestOpen = field[start], digCell = -1, digValue = INF;
      for (let a = 0; a < 4; a++) {
        const n = g.nb[start * 4 + a];
        if (n < 0 || state.wall[n]) continue;
        const value = field[n] + 1;
        if (state.brick[n]) {
          if (pred.brickGone[n] === 0 && value < digValue) { digValue = value; digCell = n; }
        } else if (value < bestOpen) bestOpen = value;
      }
      return digCell >= 0 && digValue < bestOpen && digValue < INF / 2 ? digCell : -1;
    }

    considerBomb(state, g, pid, pred, perceive, field) {
      const cfg = this.cfg;
      const me = state.players[pid], foe = this.foeOf(state, pid);
      if (me.carrying >= 0 || me.bombsLeft <= 0 || me.liveBombs >= cfg.maxLiveBombs) return null;
      if (g.bombAt[me.cell] >= 0 || state.brick[me.cell]) return null;
      const dig = this.digTarget(state, g, me, field, pred) >= 0;
      const foeNear = foe.alive && !(foe.trapped > 0) && manhattan(me, foe) <= cfg.attackRadius && foe.invuln < NEW_BOMB_TICK;
      if (!dig && !foeNear) return null;
      const mine = { cell: me.cell, e: NEW_BOMB_TICK, blast: me.blast };
      const predB = this.predict(state, g, [mine], perceive);
      const escB = this.escape(g, predB, me, false);
      let surv = escB.surv;
      if (!surv) return null;
      // 友军伤害：这颗泡若让任一活动队友无路可逃（乐观估计）就不放。
      for (const q of this.alliesOf(state, pid)) {
        const ally = state.players[q];
        if (!ally.alive || ally.trapped > 0) continue;
        if (!this.escape(g, predB, ally, true, FOE_HORIZON).surv) return null;
      }
      let attack = false, kill = false;
      if (foeNear) {
        const inFoot = predB.lethal[NEW_BOMB_TICK * g.N + foe.cell] === 1;
        const lineOk = !cfg.lineOnly || (inFoot && manhattan(me, foe) <= me.blast);
        if (lineOk) {
          const base = Math.max(1, this.escape(g, pred, foe, true, FOE_HORIZON).endCount);
          const withBomb = this.escape(g, predB, foe, true, FOE_HORIZON).endCount;
          const ratio = withBomb / base;
          kill = withBomb === 0;
          // 施压泡只在近距离生效；远处的泡只会让自己反复原地等待。
          const pressure = manhattan(me, foe) <= 4 && ratio <= cfg.pressureRatio;
          attack = kill || (inFoot && ratio <= cfg.footRatio) || pressure;
        }
      }
      if (!attack && !dig) return null;
      if (cfg.robust) {
        surv &= this.hypoSurvivors(state, g, pid, perceive, [mine]);
        if (!surv) return null;
      }
      if (cfg.bombHesitation > 0 && this.rng() < cfg.bombHesitation) return null;
      const move = this.pickMove(g, me, surv, escB, field, true, this.dangerCells(predB, g.N));
      return { move, reason: kill ? 'bomb_kill' : (attack ? 'bomb_attack' : 'bomb_dig') };
    }

    dangerCells(pred, N) {
      const danger = new Uint8Array(N);
      for (let t = 1; t <= pred.T; t++) {
        const base = t * N;
        for (let c = 0; c < N; c++) if (pred.lethal[base + c]) danger[c] = 1;
      }
      return danger;
    }

    pickMove(g, me, acts, esc, field, threatened, danger) {
      const start = me.cell;
      const progressWeight = threatened ? 0.3 : 1;
      const robustWeight = threatened ? 2 : 0.2;
      let best = MOVE_IDLE, bestScore = -INF;
      for (let a = 0; a < 5; a++) {
        if (!(acts & (1 << a))) continue;
        const target = a === MOVE_IDLE ? start : g.nb[start * 4 + a];
        const d = Math.min(field[target], 999);
        let score = -progressWeight * d + robustWeight * Math.log2(1 + esc.count[a]);
        if (danger && danger[target]) score -= 8;
        // 受威胁时尽早离开，避免“原地选项最多”导致拖延到来不及。
        if (a === MOVE_IDLE) score -= threatened ? 3 : 0.3;
        if (score > bestScore + 1e-9) { bestScore = score; best = a; }
      }
      return best;
    }
  }

  return { BunHunterBot, DIFFICULTIES, hunterStateFromSim, NEW_BOMB_TICK, FLAME_TICKS };
});
