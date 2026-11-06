'use strict';

const VERSION = 'bun-interceptor-defend-cap4-v1';
const initialized = new WeakMap();

function applyInitialCapacity(sim, defender, capacity) {
  if (!Number.isInteger(defender) || defender < 0 || defender >= sim.nPlayers) {
    throw new RangeError('defender player index');
  }
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > sim.bombsMax) {
    throw new RangeError('initial capacity exceeds native growth maximum');
  }
  if (initialized.get(sim) === sim.fuse) throw new Error('initial capacity already applied for this reset');
  if (sim.liveBombs(defender) !== 0 || sim.t !== 0) {
    throw new Error('initial capacity can only be applied immediately after reset');
  }
  sim.bombsCap[defender] = capacity;
  sim._gen++;
  initialized.set(sim, sim.fuse);
  return {
    version: VERSION,
    defender,
    initial_bombs_cap: capacity,
    death_floor_bombs: sim.loBombs[defender],
    native_growth_max: sim.bombsMax,
    live_bombs_at_reset: 0,
  };
}

function applyReplayCapacity(sim, meta) {
  if (meta.defender_initial_bomb_capacity == null) return null;
  const receipt = applyInitialCapacity(
    sim,
    meta.defender_player,
    meta.defender_initial_bomb_capacity,
  );
  if (receipt.death_floor_bombs !== meta.defender_death_floor_bombs) {
    throw new Error('recorded capacity reset/death-floor mismatch');
  }
  return receipt;
}

module.exports = { VERSION, applyInitialCapacity, applyReplayCapacity };
