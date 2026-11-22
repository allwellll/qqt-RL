const assert = require('node:assert/strict');
const C = require('./controls');
const { Sim } = require('./sim-bun-native-v5-facing-window');

for (const [x, y] of [[45, 65], [180, 180]]) {
  const handlers = {};
  const dial = { style: {}, getBoundingClientRect: () => ({ width: 96 }) };
  const zone = {
    dataset: {}, matches: () => true, querySelector: () => dial, querySelectorAll: () => [],
    getBoundingClientRect: () => ({ left: 10, top: 20, right: 250, bottom: 260, width: 240, height: 240 }),
    addEventListener: (name, fn) => { handlers[name] = fn; }, setPointerCapture() {},
  };
  const segments = [];
  const controls = C.bindTouchControls(zone, new Set(), () => {}, {
    ...C.TRAJECTORY_TOUCH_PROFILE, onTrajectorySegment: segment => segments.push(segment),
  });
  const sim = new Sim(3);
  sim.reset('open', { nativeItems: true, nativeTrap: true });
  sim.pos.set([6.5, 6.5, 10.5, 10.5]);
  const send = (type, clientX, clientY) => handlers[type]({ type, clientX, clientY, button: 0, pointerId: 1, preventDefault() {} });
  const move = () => controls.moveFor(sim, 0, performance.now());
  send('pointerdown', x, y);
  assert.equal(move(), 4, 'off-centre contact starts neutral');
  send('pointermove', x + 0.25, y);
  assert.equal(move(), 4, 'small movement is filtered');
  send('pointermove', x + 24, y);
  const right = move();
  assert.equal(right, 3);
  sim.frameStep(0, right, controls.frameSeconds(1 / 60));
  assert(sim.pos[1] > 6.5, 'character actually moves right');
  send('pointermove', x + 36, y - 12);
  assert.equal(move(), 3, 'diagonal retains right');
  send('pointermove', x + 36, y - 36);
  const up = move();
  assert.equal(up, 0, 'upward segment turns even while finger is still right of its start');
  assert.equal(zone.dataset.rawAngle, '90');
  assert.equal(zone.dataset.turnAngle, '45');
  assert.deepEqual(segments.at(-1).from, {x: x + 36, y: y - 12});
  assert.deepEqual(segments.at(-1).to, {x: x + 36, y: y - 36});
  sim.frameStep(0, up, controls.frameSeconds(1 / 60));
  assert(sim.pos[0] < 6.5, 'character actually turns up');
  send('pointerup', x + 36, y - 36);
  assert.equal(segments.at(-1), null, 'release clears diagnostics');
  assert.equal(move(), 4);
  send('pointerdown', x, y);
  send('pointermove', x - 12, y - 1);
  assert.equal(zone.dataset.turnAngle, '', 'new stroke cannot inherit the old stroke angle');
  send('pointermove', x - 24, y);
  assert.equal(move(), 2);
  assert.equal(zone.dataset.turnAngle, '10', 'turn uses shortest arc across ±180°');
}
console.log('Trajectory steering PASS');
