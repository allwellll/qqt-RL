/**
 * BunRuleTacticalBot - frozen code strategy for Bun mode.
 *
 * Fixed contract: 10 Hz, 40-step (4 second) horizon, time-expanded blast map,
 * bounded bitset BFS, verified post-detonation escape before every bomb drop.
 */
'use strict';

(function bunRuleBotFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) {
    root.BunRuleTacticalBot = api.BunRuleTacticalBot;
    root.QQTBunRuleBot = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildBunRuleBot() {
  const TICK_HZ = 10;
  const HORIZON_STEPS = 40;
  const DEFAULT_FUSE = 30;
  const BLAST_LINGER_STEPS = 2;
  const HEIGHT = 13;
  const WIDTH = 15;
  const MOVE_IDLE = 4;
  const ABILITY_NONE = 0;
  const ABILITY_BOMB = 1;
  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const PHASE_NAMES = ['COMBAT', 'KILL_CONFIRMED', 'OBJECTIVE_RUSH', 'CARRY_RETURN', 'DELIVER', 'RECOVER'];
  const COMPLEXITY = 'O(HORIZON_STEPS * height * width + bombs^2 * blast) time; O(HORIZON_STEPS * height * width) space';

  function cellsToGrid(cells, height, width, numeric) {
    const grid = new Int16Array(height * width);
    for (const cell of cells || []) {
      const row = Number(cell[0]), col = Number(cell[1]);
      if (row >= 0 && row < height && col >= 0 && col < width) grid[row * width + col] = numeric ? 1 : 1;
    }
    return grid;
  }

  function normalizeGrid(value, cells, height, width, numeric) {
    if (value == null) return cellsToGrid(cells, height, width, numeric);
    const flat = Array.from(value.flat ? value.flat(Infinity) : value);
    if (flat.length !== height * width) throw new Error(`grid has ${flat.length} values, expected ${height * width}`);
    return Int16Array.from(flat, Number);
  }

  function parseState(state) {
    const height = Number(state.height || HEIGHT);
    const width = Number(state.width || WIDTH);
    if (!Array.isArray(state.players) || state.players.length !== 2) throw new Error('state.players must contain exactly two players');
    const players = state.players.map((player) => ({
      row: Number(player.row), col: Number(player.col), alive: player.alive !== false,
      bombs: Math.max(0, Number(player.bombs || 0)), blast: Math.max(1, Number(player.blast || 2)),
      speed: Math.max(0.1, Number(player.speed || 1.3)),
    }));
    const bombs = (state.bombs || []).map((bomb) => ({
      row: Number(bomb.row), col: Number(bomb.col),
      fuse: Math.max(1, Math.min(HORIZON_STEPS + 1, Number(bomb.fuse == null ? DEFAULT_FUSE : bomb.fuse))),
      blast: Math.max(1, Number(bomb.blast || 2)),
      physicalOwner: Number(bomb.physical_owner == null ? (bomb.owner == null ? -1 : bomb.owner) : bomb.physical_owner),
      causalOwner: Number(bomb.causal_owner == null
        ? (bomb.physical_owner == null ? (bomb.owner == null ? -1 : bomb.owner) : bomb.physical_owner)
        : bomb.causal_owner),
    }));
    const bunBases = (state.bun_bases || [[1, 4], [1, 8]]).map((base) => [Number(base[0]), Number(base[1])]);
    const bunCarried = Array.from(state.bun_carried || [-1, -1], Number);
    const bunRespawn = Array.from(state.bun_respawn || [0, 0], Number);
    const bunLoose = new Int16Array(height * width * 2);
    if (state.bun_loose != null) {
      const raw = Array.from(state.bun_loose.flat ? state.bun_loose.flat(Infinity) : state.bun_loose, Number);
      if (raw.length !== bunLoose.length) throw new Error('bun_loose must have height*width*2 values');
      bunLoose.set(raw);
    }
    for (const item of state.bun_loose_cells || []) {
      bunLoose[(Number(item.row) * width + Number(item.col)) * 2 + Number(item.origin)] = Number(item.count || 1);
    }
    return {
      height, width,
      wall: normalizeGrid(state.wall, state.wall_cells, height, width, false),
      brick: normalizeGrid(state.brick, state.brick_cells, height, width, false),
      blastLinger: normalizeGrid(state.blast_linger, state.blast_cells, height, width, true),
      players, bombs, bunBases, bunCarried, bunRespawn, bunLoose,
    };
  }

  function bit(cell) { return 1n << BigInt(cell); }

  class BunRuleTacticalBot {
    constructor(options = {}) {
      const horizon = options.horizonSteps == null ? HORIZON_STEPS : Number(options.horizonSteps);
      if (horizon !== HORIZON_STEPS) throw new Error('BunRuleTacticalBot uses the fixed 40-step/4-second contract');
      this.horizonSteps = horizon;
      this.topologyKey = null;
      this.neighbors = [];
      this.routeCache = new Map();
      this.phases = ['COMBAT', 'COMBAT'];
    }

    reset() { this.phases = ['COMBAT', 'COMBAT']; }

    phase(playerId = 0) { return this.phases[playerId]; }

    restorePhase(playerId, phase) {
      if (!PHASE_NAMES.includes(phase)) throw new Error(`unknown Bun rule-bot phase: ${phase}`);
      this.phases[playerId] = phase;
    }

    eventFlag(info, name, playerId) {
      const aliases = {
        causal_kill: ['causal_kill', 'causalKill'], credited_kill: ['credited_kill', 'creditedKill'],
        mutual_death: ['mutual_death', 'mutualDeath'], own_bomb_defeat: ['own_bomb_defeat', 'ownBombDefeat'],
        death: ['death', 'died'], drop: ['drop'], steal: ['steal'], capture: ['capture'],
      };
      for (const key of aliases[name] || [name]) {
        if (!(key in (info || {}))) continue;
        const value = info[key];
        if (name === 'mutual_death') return !!value;
        return Array.isArray(value) || ArrayBuffer.isView(value) ? !!value[playerId] : !!value;
      }
      return false;
    }

    observeTransition(info, nextState, playerId = 0) {
      const state = parseState(nextState);
      const player = state.players[playerId], enemy = 1 - playerId;
      const validKill = this.eventFlag(info, 'causal_kill', playerId) &&
        this.eventFlag(info, 'credited_kill', playerId) &&
        !this.eventFlag(info, 'mutual_death', playerId) && player.alive;
      if (this.eventFlag(info, 'capture', playerId)) this.phases[playerId] = 'DELIVER';
      else if (this.eventFlag(info, 'death', playerId) || this.eventFlag(info, 'drop', playerId) || !player.alive) this.phases[playerId] = 'RECOVER';
      else if (state.bunCarried[playerId] === enemy) this.phases[playerId] = 'CARRY_RETURN';
      else if (validKill) this.phases[playerId] = 'KILL_CONFIRMED';
      else if (this.phases[playerId] === 'KILL_CONFIRMED') this.phases[playerId] = 'OBJECTIVE_RUSH';
      else this.syncPhase(nextState, playerId);
      return this.phases[playerId];
    }

    syncPhase(rawState, playerId = 0) {
      const state = parseState(rawState);
      const player = state.players[playerId], enemy = 1 - playerId;
      const current = this.phases[playerId];
      if (current === 'DELIVER') return current;
      if (!player.alive) this.phases[playerId] = 'RECOVER';
      else if (state.bunCarried[playerId] === enemy) this.phases[playerId] = 'CARRY_RETURN';
      else if (current === 'CARRY_RETURN') this.phases[playerId] = 'RECOVER';
      else if (current === 'RECOVER') {
        let hasLoose = false;
        for (let cell = 0; cell < state.height * state.width; cell++) if (state.bunLoose[cell * 2 + enemy] > 0) hasLoose = true;
        if (!hasLoose) this.phases[playerId] = state.players[enemy].alive ? 'COMBAT' : 'OBJECTIVE_RUSH';
      } else if (current === 'OBJECTIVE_RUSH' && state.players[enemy].alive) this.phases[playerId] = 'COMBAT';
      return this.phases[playerId];
    }

    decide(state, playerId = 0) {
      return this.analyze(state, playerId).action;
    }

    act(sim, playerId = 0) {
      if (!sim || !sim.isBun) return [MOVE_IDLE, 0, 0];
      const action = this.decide(stateFromSim(sim), playerId);
      return [action[0], action[1] === ABILITY_BOMB ? 1 : 0, 0];
    }

    analyze(rawState, playerId = 0) {
      this.syncPhase(rawState, playerId);
      const state = parseState(rawState);
      if (playerId !== 0 && playerId !== 1) throw new Error('playerId must be 0 or 1');
      const player = state.players[playerId];
      if (!player.alive) return this._decision([MOVE_IDLE, ABILITY_NONE], 'dead', false, false, false, [], 0, this.phase(playerId));
      const neighbors = this.getNeighbors(state);
      const predictedState = this.predict(state);
      const basePlan = this.survivalPlan(state, playerId, predictedState, neighbors, false);
      const startCell = player.row * state.width + player.col;
      const exposed = predictedState.dangerMasks.some((mask) => (mask & bit(startCell)) !== 0n);
      if (exposed) {
        return this._decision(
          [basePlan.firstAction, ABILITY_NONE], basePlan.survived ? 'escape_immediate' : 'doomed_max_survival',
          !basePlan.survived, basePlan.survived, false, predictedState.predictedBombs, basePlan.survivalSteps, this.phase(playerId));
      }
      const phase = this.phase(playerId);
      if (['KILL_CONFIRMED', 'OBJECTIVE_RUSH', 'CARRY_RETURN', 'RECOVER', 'DELIVER'].includes(phase)) {
        if (phase === 'DELIVER') return this._decision([MOVE_IDLE, ABILITY_NONE], 'delivered', false, true, false, predictedState.predictedBombs, basePlan.survivalSteps, phase);
        const goals = this.phaseGoals(state, playerId, phase);
        if (goals.size) {
          const move = this.goalMove(state, playerId, goals, basePlan, neighbors);
          return this._decision([move, ABILITY_NONE], move === MOVE_IDLE ? 'objective_wait' : 'objective_route',
            false, basePlan.survived, false, predictedState.predictedBombs, basePlan.survivalSteps, phase);
        }
      }
      if (!state.players[1 - playerId].alive) {
        return this._decision([MOVE_IDLE, ABILITY_NONE], 'unattributed_enemy_down', false, true, false,
          predictedState.predictedBombs, basePlan.survivalSteps, this.phase(playerId));
      }
      const attack = this.safeAttack(state, playerId, neighbors);
      if (attack) {
        return this._decision(
          [attack.plan.firstAction, ABILITY_BOMB], attack.verifiedKill ? 'safe_non_trade_attack' : 'safe_pressure_attack',
          false, true, true, attack.predicted.predictedBombs, attack.plan.survivalSteps, this.phase(playerId));
      }
      const move = this.controlMove(state, playerId, predictedState, neighbors);
      return this._decision(
        [move, ABILITY_NONE], move === MOVE_IDLE ? 'wait_no_safe_attack' : 'control_space',
        false, basePlan.survived, false, predictedState.predictedBombs, basePlan.survivalSteps, this.phase(playerId));
    }

    _decision(action, reason, doomed, claimedEscape, safeEscapeAfterBomb, predictedBombs, survivalSteps, phase = 'COMBAT') {
      return { action, reason, doomed, claimedEscape, safeEscapeAfterBomb, predictedBombs, survivalSteps, phase };
    }

    getNeighbors(state) {
      let staticKey = `${state.height}x${state.width}:`;
      for (let i = 0; i < state.wall.length; i++) staticKey += state.wall[i] || state.brick[i] ? '1' : '0';
      if (staticKey === this.topologyKey) return this.neighbors;
      const result = [];
      for (let row = 0; row < state.height; row++) {
        for (let col = 0; col < state.width; col++) {
          result.push(DIRS.map(([drow, dcol]) => {
            const nextRow = row + drow, nextCol = col + dcol;
            if (nextRow < 0 || nextRow >= state.height || nextCol < 0 || nextCol >= state.width) return -1;
            const index = nextRow * state.width + nextCol;
            return state.wall[index] || state.brick[index] ? -1 : index;
          }));
        }
      }
      this.topologyKey = staticKey;
      this.neighbors = result;
      return result;
    }

    blastCells(state, row, col, blast, liveBombs) {
      const cells = [[row, col]];
      for (const [drow, dcol] of DIRS) {
        for (let distance = 1; distance <= blast; distance++) {
          const nextRow = row + drow * distance, nextCol = col + dcol * distance;
          if (nextRow < 0 || nextRow >= state.height || nextCol < 0 || nextCol >= state.width) break;
          const index = nextRow * state.width + nextCol;
          if (state.wall[index]) break;
          cells.push([nextRow, nextCol]);
          if (state.brick[index] || liveBombs.has(index)) break;
        }
      }
      return cells;
    }

    predict(state) {
      const dangerMasks = Array(this.horizonSteps + 1).fill(0n);
      for (let cell = 0; cell < state.blastLinger.length; cell++) {
        const linger = Math.min(this.horizonSteps + 1, Math.max(0, state.blastLinger[cell]));
        for (let tick = 0; tick < linger; tick++) dangerMasks[tick] |= bit(cell);
      }
      const bombs = state.bombs.map((bomb) => ({ ...bomb, step: bomb.fuse, done: false }));
      const liveBombs = new Set(bombs.map((bomb) => bomb.row * state.width + bomb.col));
      while (true) {
        const pending = bombs.filter((bomb) => !bomb.done);
        if (!pending.length) break;
        const step = Math.min(...pending.map((bomb) => bomb.step));
        const sameStep = pending.filter((bomb) => bomb.step === step);
        while (sameStep.length) {
          const bomb = sameStep.shift();
          if (bomb.done) continue;
          bomb.done = true;
          liveBombs.delete(bomb.row * state.width + bomb.col);
          const footprint = this.blastCells(state, bomb.row, bomb.col, bomb.blast, liveBombs);
          if (step <= this.horizonSteps) {
            const end = Math.min(this.horizonSteps + 1, step + BLAST_LINGER_STEPS);
            for (let tick = step; tick < end; tick++) {
              for (const [row, col] of footprint) dangerMasks[tick] |= bit(row * state.width + col);
            }
          }
          const footprintSet = new Set(footprint.map(([row, col]) => row * state.width + col));
          for (const other of bombs) {
            if (!other.done && other.step > step && footprintSet.has(other.row * state.width + other.col)) {
              other.step = step;
              other.causalOwner = bomb.causalOwner;
              sameStep.push(other);
            }
          }
        }
      }
      const bombUntil = new Int16Array(state.height * state.width);
      const predictedBombs = bombs.map((bomb) => {
        bombUntil[bomb.row * state.width + bomb.col] = Math.min(this.horizonSteps + 1, bomb.step);
        return {
          row: bomb.row, col: bomb.col, explodeStep: bomb.step, blast: bomb.blast,
          physicalOwner: bomb.physicalOwner, causalOwner: bomb.causalOwner,
        };
      }).sort((a, b) => a.explodeStep - b.explodeStep || a.row - b.row || a.col - b.col);
      return { dangerMasks, bombUntil, predictedBombs };
    }

    moveTicks(player) {
      return Math.max(1, Math.ceil(1 / (0.3 * player.speed)));
    }

    survivalPlan(state, playerId, predicted, neighbors, startOnNewBomb) {
      const player = state.players[playerId];
      const start = player.row * state.width + player.col;
      const count = state.height * state.width;
      const allMask = (1n << BigInt(count)) - 1n;
      let staticMask = 0n;
      for (let cell = 0; cell < count; cell++) if (state.wall[cell] || state.brick[cell]) staticMask |= bit(cell);
      const openMask = allMask & ~staticMask;
      const moveTicks = this.moveTicks(player);
      const bombCells = [];
      for (let cell = 0; cell < count; cell++) if (predicted.bombUntil[cell] > 0) bombCells.push([cell, predicted.bombUntil[cell]]);
      const bombMasks = Array(this.horizonSteps + 1).fill(0n);
      for (let tick = 0; tick <= this.horizonSteps; tick++) {
        let mask = 0n;
        for (const [cell, until] of bombCells) if (until >= tick) mask |= bit(cell);
        bombMasks[tick] = mask;
      }
      let leftColumn = 0n, rightColumn = 0n;
      for (let row = 0; row < state.height; row++) {
        leftColumn |= bit(row * state.width);
        rightColumn |= bit(row * state.width + state.width - 1);
      }
      const shift = (mask, action) => {
        if (action === 0) return mask >> BigInt(state.width);
        if (action === 1) return (mask << BigInt(state.width)) & allMask;
        if (action === 2) return (mask & ~leftColumn) >> 1n;
        if (action === 3) return ((mask & ~rightColumn) << 1n) & allMask;
        return mask;
      };
      const intervalCache = new Map();
      const intervalSafe = (startTick, endTick) => {
        const key = `${startTick}:${endTick}`;
        if (intervalCache.has(key)) return intervalCache.get(key);
        let blocked = 0n;
        for (let tick = startTick + 1; tick <= endTick; tick++) blocked |= predicted.dangerMasks[tick];
        const safe = allMask & ~blocked;
        intervalCache.set(key, safe);
        return safe;
      };
      const reached = Array.from({ length: this.horizonSteps + 1 }, () => Array(5).fill(0n));
      const startMask = bit(start);
      for (const action of [0, 1, 2, 3, MOVE_IDLE]) {
        const duration = action === MOVE_IDLE ? 1 : moveTicks;
        const safe = intervalSafe(0, duration);
        const source = startMask & safe;
        let target = shift(source, action) & safe & openMask;
        if (action !== MOVE_IDLE) target &= ~bombMasks[duration];
        else if (startOnNewBomb) target = 0n;
        reached[duration][action] |= target;
      }
      let bestTick = 0, bestActions = [];
      for (let tick = 1; tick <= this.horizonSteps; tick++) {
        const active = [];
        for (let first = 0; first < 5; first++) if (reached[tick][first] !== 0n) active.push(first);
        if (active.length) { bestTick = tick; bestActions = active; }
        if (tick === this.horizonSteps && active.length) {
          const first = this.rankFirstAction(state, playerId, active, start, neighbors);
          return { survived: true, firstAction: first, survivalSteps: this.horizonSteps,
            endCell: this.firstCell(reached[tick][first]), safeActions: active.slice() };
        }
        for (const first of active) {
          const current = reached[tick][first];
          if (tick + 1 <= this.horizonSteps) reached[tick + 1][first] |= current & intervalSafe(tick, tick + 1);
          const end = tick + moveTicks;
          if (end > this.horizonSteps) continue;
          const safe = intervalSafe(tick, end);
          const source = current & safe;
          for (let action = 0; action < 4; action++) {
            reached[end][first] |= shift(source, action) & safe & openMask & ~bombMasks[end];
          }
        }
      }
      const first = bestActions.length ? Math.min(...bestActions) : MOVE_IDLE;
      return { survived: false, firstAction: first, survivalSteps: bestTick,
        endCell: bestActions.length ? this.firstCell(reached[bestTick][first]) : start,
        safeActions: bestActions.slice() };
    }

    baseCells(state, team) {
      const anchor = state.bunBases[team];
      const cells = new Set();
      for (let row = anchor[0]; row < anchor[0] + 3; row++) {
        for (let col = anchor[1]; col < anchor[1] + 3; col++) {
          if (row < 0 || row >= state.height || col < 0 || col >= state.width) continue;
          const cell = row * state.width + col;
          if (!state.wall[cell] && !state.brick[cell]) cells.add(cell);
        }
      }
      return cells;
    }

    phaseGoals(state, playerId, phase) {
      const enemy = 1 - playerId;
      if (phase === 'KILL_CONFIRMED' || phase === 'OBJECTIVE_RUSH') return this.baseCells(state, enemy);
      if (phase === 'CARRY_RETURN') return this.baseCells(state, playerId);
      const cells = new Set();
      if (phase === 'RECOVER') {
        for (let cell = 0; cell < state.height * state.width; cell++) if (state.bunLoose[cell * 2 + enemy] > 0) cells.add(cell);
      }
      return cells;
    }

    distanceToGoals(state, goals, neighbors) {
      let staticKey = `${state.height}x${state.width}:`;
      for (let cell = 0; cell < state.wall.length; cell++) staticKey += state.wall[cell] || state.brick[cell] ? '1' : '0';
      const key = `${staticKey}:${Array.from(goals).sort((a, b) => a - b).join(',')}`;
      if (this.routeCache.has(key)) return this.routeCache.get(key);
      const distance = new Int16Array(state.height * state.width);
      distance.fill(16384);
      const queue = [];
      for (const goal of goals) { distance[goal] = 0; queue.push(goal); }
      for (let offset = 0; offset < queue.length; offset++) {
        const cell = queue[offset], candidate = distance[cell] + 1;
        for (const neighbor of neighbors[cell]) {
          if (neighbor >= 0 && candidate < distance[neighbor]) {
            distance[neighbor] = candidate;
            queue.push(neighbor);
          }
        }
      }
      this.routeCache.set(key, distance);
      return distance;
    }

    goalMove(state, playerId, goals, plan, neighbors) {
      const player = state.players[playerId];
      const start = player.row * state.width + player.col;
      if (goals.has(start)) return MOVE_IDLE;
      const actions = plan.survived ? plan.safeActions : [plan.firstAction];
      const distance = this.distanceToGoals(state, goals, neighbors);
      let bestAction = MOVE_IDLE, bestScore = Infinity;
      for (const action of actions) {
        const target = action === MOVE_IDLE ? start : neighbors[start][action];
        if (target < 0) continue;
        const score = distance[target] * 10 + (action === MOVE_IDLE ? 2 : 0) + action;
        if (score < bestScore) { bestScore = score; bestAction = action; }
      }
      return bestAction;
    }

    firstCell(mask) {
      if (mask === 0n) return -1;
      let cell = 0;
      while ((mask & 1n) === 0n) { mask >>= 1n; cell++; }
      return cell;
    }

    rankFirstAction(state, playerId, actions, start, neighbors) {
      const player = state.players[playerId];
      const opponent = state.players[1 - playerId];
      const bombCells = state.bombs.map((bomb) => [bomb.row, bomb.col]);
      let bestAction = actions[0], bestScore = -Infinity;
      for (const action of actions) {
        const target = action === MOVE_IDLE ? start : neighbors[start][action];
        if (target < 0) continue;
        const row = Math.floor(target / state.width), col = target % state.width;
        let bombDistance = 0, awayAlignment = 0;
        if (bombCells.length) {
          let nearest = bombCells[0], nearestDistance = Infinity;
          for (const cell of bombCells) {
            const distance = Math.abs(player.row - cell[0]) + Math.abs(player.col - cell[1]);
            if (distance < nearestDistance) { nearestDistance = distance; nearest = cell; }
          }
          bombDistance = Math.min(...bombCells.map((cell) => Math.abs(row - cell[0]) + Math.abs(col - cell[1])));
          if (action !== MOVE_IDLE) awayAlignment = (row - player.row) * (player.row - nearest[0]) + (col - player.col) * (player.col - nearest[1]);
        }
        const opponentDistance = Math.abs(row - opponent.row) + Math.abs(col - opponent.col);
        const opponentAlignment = (row - player.row) * (player.row - opponent.row) + (col - player.col) * (player.col - opponent.col);
        const degree = neighbors[target].filter((value) => value >= 0).length;
        let score = 3 * bombDistance + 2 * awayAlignment + opponentDistance + opponentAlignment + 0.1 * degree;
        if (action === MOVE_IDLE) score -= 0.5;
        if (score > bestScore) { bestScore = score; bestAction = action; }
      }
      return bestAction;
    }

    lineOfBlast(state, player, opponent) {
      if (player.row !== opponent.row && player.col !== opponent.col) return false;
      const distance = Math.abs(player.row - opponent.row) + Math.abs(player.col - opponent.col);
      if (distance > player.blast) return false;
      const drow = player.row === opponent.row ? 0 : (opponent.row > player.row ? 1 : -1);
      const dcol = player.col === opponent.col ? 0 : (opponent.col > player.col ? 1 : -1);
      for (let step = 1; step <= distance; step++) {
        const row = player.row + drow * step, col = player.col + dcol * step;
        const index = row * state.width + col;
        if (state.wall[index] || state.brick[index]) return false;
        if (step < distance && state.bombs.some((bomb) => bomb.row === row && bomb.col === col)) return false;
      }
      return true;
    }

    safeAttack(state, playerId, neighbors) {
      const player = state.players[playerId], opponent = state.players[1 - playerId];
      if (player.bombs <= 0 || !opponent.alive) return null;
      if (state.bombs.some((bomb) => bomb.row === player.row && bomb.col === player.col)) return null;
      if (!this.lineOfBlast(state, player, opponent)) return null;
      const attackState = { ...state, bombs: state.bombs.concat([{
        row: player.row, col: player.col, fuse: DEFAULT_FUSE, blast: player.blast,
        physicalOwner: playerId, causalOwner: playerId,
      }]) };
      const predicted = this.predict(attackState);
      const plan = this.survivalPlan(attackState, playerId, predicted, neighbors, true);
      if (!plan.survived || plan.firstAction === MOVE_IDLE) return null;
      const opponentPlan = this.survivalPlan(attackState, 1 - playerId, predicted, neighbors, false);
      return { plan, predicted, verifiedKill: !opponentPlan.survived };
    }

    transitionSafe(state, predicted, source, target, startTick, endTick) {
      const sourceBit = bit(source), targetBit = bit(target);
      if (target !== source && predicted.bombUntil[target] >= endTick) return false;
      for (let tick = startTick + 1; tick <= Math.min(endTick, this.horizonSteps); tick++) {
        if ((predicted.dangerMasks[tick] & (sourceBit | targetBit)) !== 0n) return false;
      }
      return true;
    }

    controlMove(state, playerId, predicted, neighbors) {
      const player = state.players[playerId], opponent = state.players[1 - playerId];
      const start = player.row * state.width + player.col;
      const moveTicks = this.moveTicks(player);
      let bestAction = MOVE_IDLE, bestScore = -Infinity;
      for (const action of [0, 1, 2, 3, MOVE_IDLE]) {
        const target = action === MOVE_IDLE ? start : neighbors[start][action];
        if (target < 0) continue;
        const end = action === MOVE_IDLE ? 1 : moveTicks;
        if (!this.transitionSafe(state, predicted, start, target, 0, end)) continue;
        const row = Math.floor(target / state.width), col = target % state.width;
        const distance = Math.abs(row - opponent.row) + Math.abs(col - opponent.col);
        const degree = neighbors[target].filter((value) => value >= 0).length;
        const score = -distance + 0.08 * degree - (action === MOVE_IDLE ? 0.25 : 0);
        if (score > bestScore) { bestScore = score; bestAction = action; }
      }
      return bestAction;
    }
  }

  function stateFromSim(sim) {
    const bombs = [];
    const live = [0, 0];
    const height = Number(sim.H || HEIGHT);
    const width = Number(sim.W || (sim.wall.length / height));
    for (let cell = 0; cell < sim.fuse.length; cell++) {
      if (sim.fuse[cell] <= 0) continue;
      const owner = Number(sim.owner[cell]);
      if (owner === 0 || owner === 1) live[owner]++;
      bombs.push({
        row: Math.floor(cell / width), col: cell % width,
        fuse: Number(sim.fuse[cell]), blast: Number(sim.bombBlast[cell] || 2),
        physical_owner: owner, causal_owner: owner,
      });
    }
    return {
      height, width,
      wall: sim.wall, brick: sim.brick, blast_linger: sim.blastLinger,
      players: [0, 1].map((pid) => ({
        row: Math.floor(sim.pos[pid * 2]), col: Math.floor(sim.pos[pid * 2 + 1]),
        alive: !!sim.alive[pid],
        bombs: sim.bunCarried && sim.bunCarried[pid] >= 0 ? 0 : Math.max(0, Number(sim.bombsCap[pid]) - live[pid]),
        blast: Number(sim.blastCap[pid]), speed: Number(sim.spdG[pid]),
      })),
      bombs,
      bun_bases: (sim.bunBases || [[1, 4], [1, 8]]).map((base) => base.slice()),
      bun_carried: (sim.bunCarried || [-1, -1]).slice(),
      bun_respawn: (sim.bunRespawn || [0, 0]).slice(),
      bun_loose: sim.bunLoose || new Uint8Array(height * width * 2),
    };
  }

  return { BunRuleTacticalBot, TICK_HZ, HORIZON_STEPS, PHASE_NAMES, COMPLEXITY, stateFromSim };
});
