'use strict';
const assert = require('assert');
const visual = require('./visual_renderer.js');

const sim = { alive: [false, true], isBun: true, bunRespawn: [60, 0], bunRespawnTicks: 100 };
assert(visual.deathOverlayAlpha(sim, 0) > 0.4, 'dead human receives a visible dim overlay');
assert.equal(visual.deathOverlayAlpha(sim, 1), 0, 'alive player is not dimmed');
assert.equal(visual.deathOverlayAlpha(sim, -1), 0, 'spectator/replay view remains unobscured');
sim.alive[0] = true;
assert.equal(visual.deathOverlayAlpha(sim, 0), 0, 'respawn restores normal brightness');
sim.alive[0] = false; sim.bunRespawn[0] = 0;
assert.equal(visual.deathOverlayAlpha(sim, 0), 0, 'expired respawn timer does not leave stale overlay');
assert(visual.deathOverlayAlpha({ alive: [false], isBun: false }, 0) > 0, 'non-bun death still dims the game canvas');
const context = {
  filter: 'sepia(1)', saves: [],
  save() { this.saves.push(this.filter); },
  restore() { this.filter = this.saves.pop(); },
};
assert.throws(() => visual.withContextState(context, () => {
  context.filter = 'none';
  throw new Error('draw failed');
}), /draw failed/);
assert.equal(context.filter, 'sepia(1)', 'render boundary restores Canvas filter even when drawing throws');
console.log('death overlay visibility, respawn recovery and replay compatibility passed');

const Q = require('./sim.js');
const level = require('./assets/maps/levels.json').find(x => x.qqt_id === 806);
const game = new Q.Sim(73);
game.reset(level, { nPlayers: 4, teams: [0, 0, 1, 1], nativeTrap: true });
game.alive.fill(false); game.bunRespawn.fill(21);
const before = JSON.stringify(game.snapshotReplay());
const markers = visual.respawnMarkers(game, 0);
assert.deepEqual(markers, [], 'live views hide all respawn locations');
assert.equal(JSON.stringify(game.snapshotReplay()), before, 'markers never mutate authoritative state');
const replay = new Q.Sim(1); replay.restoreReplay(JSON.parse(before));
assert.deepEqual(visual.respawnMarkers(replay, -1), visual.respawnMarkers(game, -1));
game.lastDied.fill(false); game.bunRespawn.fill(1); game._bunRespawnStep();
assert.deepEqual(visual.respawnMarkers(game, 0), [], 'all markers disappear on the respawn tick');
game.alive[0] = false; game.bunRespawn[0] = 10; game.done = true;
assert.deepEqual(visual.respawnMarkers(game, 0), [], 'completed matches never announce stale respawns');
assert.deepEqual(visual.respawnMarkers(replay, -1), [], 'Review/spectator views also hide spawn locations');
const Kill = require('./sim-bun-native-v4-facing');
const kill = new Kill.Sim(73);
kill.reset(level, {nativeItems:true, nativeTrap:true, killMode:true});
kill._killPlayer(1);
for (let tick = 0; tick < 30; tick++) {
  assert(!kill.alive[1]);
  for (const viewer of [-1, 0, 1]) assert.deepEqual(visual.respawnMarkers(kill, viewer), []);
  kill.step([[4,0], [4,0]]);
}
assert(kill.alive[1], 'hidden markers do not delay respawn');
console.log('respawn locations hidden for both sides, spectators and Review; physics unchanged');
