'use strict';
(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTSelfRespawnPreview = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  // Opt in only for live web matches. Frozen Review engines and model encoders
  // are never patched. Use the match RNG, never rendering time or Math.random.
  function attach(sim) {
    if (sim.selfRespawnPreview) return sim;
    sim.selfRespawnPreview = true;
    function safeCells() {
      const bombs = Uint8Array.from(sim.fuse, fuse => Number(fuse > 0));
      const blast = Int16Array.from(sim.bombBlast, n => n || 3);
      const covered = sim._rays(bombs, blast, sim.brick).covered;
      return sim._itemSupplyCells().filter(cell => !covered[cell]);
    }
    function select(player, preferExisting = true) {
      const positions = sim.killMode ? sim.killRespawnPos : sim.bunSpawnPos;
      const previous = positions[player];
      const cell = previous ? Math.floor(previous[0]) * 15 + Math.floor(previous[1]) : -1;
      const cells = safeCells();
      if (preferExisting && previous?.every(x => x >= 0) && cells.includes(cell)) return previous;
      if (!cells.length) return positions[player] = [-1, -1];
      const selected = cells[Math.floor(sim.rng() * cells.length)];
      return positions[player] = [Math.floor(selected / 15) + .5, selected % 15 + .5];
    }
    const kill = sim._killPlayer;
    sim._killPlayer = function(player) {
      const wasAlive = this.alive[player];
      kill.call(this, player);
      if (wasAlive && !this.alive[player] && (this.killMode || this.isBun)) select(player, !this.killMode);
    };
    function respawnStep(killMode) {
      if (killMode ? !sim.killMode : !sim.isBun) return;
      const countdown = killMode ? sim.killRespawn : sim.bunRespawn;
      for (let p = 0; p < sim.nPlayers; p++) {
        if (sim.alive[p] || countdown[p] <= 0 || (!killMode && sim.lastDied?.[p])) continue;
        const target = select(p);
        countdown[p]--;
        if (target[0] < 0) { countdown[p] = Math.max(1, countdown[p]); continue; }
        if (countdown[p] > 0) continue;
        sim.pos[p * 2] = target[0]; sim.pos[p * 2 + 1] = target[1];
        sim.hp[p] = sim.initialHp || 1; sim.alive[p] = true; sim.trapped[p] = 0;
        sim.invuln[p] = killMode ? 10 : (sim.nativeTrap ? 30 : 10);
        sim.spawnProtection[p] = killMode ? 10 : (sim.nativeTrap ? 30 : 0);
        sim._clearMovementStatus(p);
        if (killMode) {
          sim.bombsCap[p] = sim.loBombs[p]; sim.blastCap[p] = sim.loBlast[p]; sim.spdG[p] = sim.loSpeed[p];
          sim.itemSlots[p] = []; sim._syncHeldItem(p);
        }
      }
    }
    sim._killRespawnStep = () => respawnStep(true);
    sim._bunRespawnStep = () => respawnStep(false);
    return sim;
  }
  return {attach};
});
