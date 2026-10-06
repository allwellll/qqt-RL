'use strict';
const assert = require('assert');
const Q = require('./sim');
const seed = 20261007;
const idle = n => Array.from({length: n}, () => [Q.MOVE_IDLE, 0, 0, 1]);
function scene() {
  const s = new Q.Sim(seed);
  s.reset('open', { nativeItems: true, nativeTrap: true, teams: [0, 1, 0, 1] });
  for (const k of ['wall', 'brick', 'crate', 'fuse']) s[k].fill(0);
  s.invuln.fill(0); s.spawnProtection.fill(0);
  s.pos.set([4.5, 4.9, 4.5, 5.5, 10.5, 10.5, 11.5, 11.5].slice(0, s.pos.length));
  return s;
}
for (const item of [Q.ITEM_SLOW_GLUE, Q.ITEM_BANANA]) {
  const s = scene(), cell = 4 * Q.W + 5;
  s._addItemToSlots(1, item);
  const actions = idle(s.nPlayers); actions[1][2] = 1;
  s.step(actions);
  assert.equal(s.fieldArmed[cell], 0, 'placer is still standing on unarmed trap');
  assert.equal(s.movementStatus[1], Q.MOVE_STATUS_NONE, 'no self-trigger on release');
  s.frameStep(0, Q.MOVE_RIGHT, .05);
  assert.equal(s.movementStatus[0], item === Q.ITEM_SLOW_GLUE ? Q.MOVE_STATUS_SLOW : Q.MOVE_STATUS_SLIDE,
    `seed ${seed}: opponent trap works before placer leaves`);
  assert.equal(s.fieldItem[cell], Q.ITEM_NONE);
  const ticks = s.movementStatusTicks[0];
  s._updateFieldItems(idle(s.nPlayers));
  assert.equal(s.movementStatusTicks[0], ticks, 'consumed trap cannot reset duration');
}
// A human crosses a whole cell between two 10Hz ticks; contact belongs to frame movement.
{
  const s = scene(), cell = 4 * Q.W + 5;
  s.fieldItem[cell] = Q.ITEM_SLOW_GLUE; s.fieldOwner[cell] = 1; s.fieldArmed[cell] = 1;
  s._setMovementStatus(0, Q.MOVE_STATUS_FAST);
  s.frameStep(0, Q.MOVE_RIGHT, .1);
  s.frameStep(0, Q.MOVE_RIGHT, .1);
  assert.equal(s.movementStatus[0], Q.MOVE_STATUS_SLOW);
  assert.equal(s.fieldItem[cell], 0);
  assert(s.pos[1] < 6, 'slow applies within movement, not after traversing the full cell');
}
// Custom fast banana slide also crosses a complete field cell in a single Bot tick.
{
  const s = scene(), cell = 4 * Q.W + 5;
  s.bananaSlideSpeedPx = 600; s.pos[3] = 10.5;
  s.fieldItem[cell] = Q.ITEM_SLOW_GLUE; s.fieldOwner[cell] = 1; s.fieldArmed[cell] = 1;
  s.slideDir[0] = Q.MOVE_RIGHT; s._setMovementStatus(0, Q.MOVE_STATUS_SLIDE, 0);
  const actions = idle(s.nPlayers); actions[0][3] = 0;
  s.step(actions);
  assert(s.pos[1] > 6 && s.fieldItem[cell] === 0 && s.movementStatus[0] === Q.MOVE_STATUS_SLOW,
    'logic movement cannot tunnel through a trap');
}
// The opponent releases on the player's starting cell in the same fast movement tick.
{
  const s = scene(), cell = 4 * Q.W + 5;
  s.pos[1] = 5.5; s.bananaSlideSpeedPx = 480;
  s.slideDir[0] = Q.MOVE_RIGHT; s._setMovementStatus(0, Q.MOVE_STATUS_SLIDE, 0);
  s._addItemToSlots(1, Q.ITEM_SLOW_GLUE);
  const actions = idle(s.nPlayers); actions[0][3] = 0; actions[1][2] = 1;
  s.step(actions);
  assert.equal(s.fieldItem[cell], 0, 'same-tick opponent release at starting center is consumed');
  assert.equal(s.movementStatus[0], Q.MOVE_STATUS_SLOW, 'starting center contact must not be skipped');
  const own = scene(); own.pos[1] = 5.9; own.pos[3] = 10.5;
  own._addItemToSlots(0, Q.ITEM_SLOW_GLUE);
  const leave = idle(own.nPlayers); leave[0] = [Q.MOVE_RIGHT, 0, 1, 0];
  own.step(leave);
  assert.equal(own.fieldItem[cell], Q.ITEM_SLOW_GLUE, 'placer departure does not consume its own fresh trap');
  assert.equal(own.movementStatus[0], Q.MOVE_STATUS_NONE);
}
// Boundary center, corner turn, spawn protection and stable same-tick overlap.
for (const x of [4.999, 5, 5.001]) {
  const s = scene(), cell = 4 * Q.W + 5;
  s.pos[1] = x; s.fieldItem[cell] = Q.ITEM_SLOW_GLUE; s.fieldOwner[cell] = 1;
  s._updateFieldItems(idle(s.nPlayers));
  assert.equal(s.movementStatus[0] === Q.MOVE_STATUS_SLOW, x >= 5);
}
{
  const s = scene(), cell = 4 * Q.W + 5;
  s.fieldItem[cell] = Q.ITEM_SLOW_GLUE; s.fieldOwner[cell] = 1;
  s.spawnProtection[0] = 10; s.invuln[0] = 10;
  s.frameStep(0, Q.MOVE_RIGHT, .05);
  assert.equal(s.fieldItem[cell], Q.ITEM_SLOW_GLUE, 'protected spawn does not consume');
  s.spawnProtection[0] = 0; s.invuln[0] = 0;
  s.frameStep(0, Q.MOVE_UP, .025);
  assert.equal(s.movementStatus[0], Q.MOVE_STATUS_SLOW, 'turn and protection expiry contact');
}
{
  const s = scene(), cell = 4 * Q.W + 5;
  s.fieldItem[cell] = Q.ITEM_BANANA; s.fieldOwner[cell] = 1;
  s.pos[1] = 5.5; s.pos[3] = 6.5;
  s._updateFieldItems(idle(s.nPlayers));
  assert.equal(s.movementStatus[0], Q.MOVE_STATUS_SLIDE);
  assert.equal(s.movementStatus[1], Q.MOVE_STATUS_NONE);
  const own = scene(); own.fieldItem[cell] = Q.ITEM_SLOW_GLUE; own.fieldOwner[cell] = 1;
  own.pos[3] = 6.5; own._updateFieldItems(idle(own.nPlayers));
  own.pos[3] = 5.5; own._updateFieldItems(idle(own.nPlayers));
  assert.equal(own.movementStatus[1], Q.MOVE_STATUS_SLOW, 'placer returning triggers armed trap');
  const team = scene(); team.fieldItem[cell] = Q.ITEM_SLOW_GLUE; team.fieldOwner[cell] = 1; team.fieldArmed[cell] = 1;
  team.pos[3] = 6.5; team.pos[7] = 5.5; team.pos[6] = 4.5;
  team._updateFieldItems(idle(team.nPlayers));
  assert.equal(team.movementStatus[3], Q.MOVE_STATUS_SLOW, 'existing teammate contact remains enabled');
}
{
  const s = scene(), cell = 4 * Q.W + 5;
  s.fieldItem[cell] = Q.ITEM_BANANA; s.fieldOwner[cell] = 1;
  const copy = scene(); copy.restoreReplay(s.snapshotReplay());
  for (const sim of [s, copy]) {
    sim.frameStep(0, Q.MOVE_RIGHT, .05);
    sim.frameStep(0, Q.MOVE_UP, .05);
    sim.step(idle(sim.nPlayers));
  }
  assert.deepEqual(copy.snapshotReplay(), s.snapshotReplay(), 'snapshot resume has identical contacts/status/positions');
}
console.log(`Field item regression seed ${seed}: opponent, swept frames, owner/team, protection, overlap, corner and snapshot passed`);
