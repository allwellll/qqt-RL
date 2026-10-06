'use strict';

const assert = require('assert');
const controls = require('./controls.js');

assert.strictEqual(controls.moveForHeld(new Set(['ArrowUp'])), 0);
assert.strictEqual(controls.moveForHeld(new Set(['ArrowDown'])), 1);
assert.strictEqual(controls.moveForHeld(new Set(['ArrowLeft'])), 2);
assert.strictEqual(controls.moveForHeld(new Set(['ArrowRight'])), 3);
assert.strictEqual(controls.moveForHeld(new Set()), 4);
for (const first of controls.MOVEMENT_KEYS) for (const last of controls.MOVEMENT_KEYS) {
  if (first === last) continue;
  const held = new Set([first, last]);
  assert.equal(controls.moveForHeld(held), controls.MOVEMENT_KEYS.indexOf(last), 'new direction wins');
  held.add(first);
  assert.equal(controls.moveForHeld(held), controls.MOVEMENT_KEYS.indexOf(last), 'repeat cannot steal priority');
  held.delete(last);
  assert.equal(controls.moveForHeld(held), controls.MOVEMENT_KEYS.indexOf(first), 'release restores held direction');
  held.clear();
  assert.equal(controls.moveForHeld(held), 4);
}
for (const key of ['KeyW', 'KeyA', 'KeyS', 'KeyD']) {
  assert.strictEqual(controls.moveForHeld(new Set([key])), 4, `${key} 不再控制移动`);
}
assert.deepStrictEqual(controls.BOMB_KEYS, ['Space', 'KeyW'], 'Space 与 W 都放泡');
assert(controls.MOVEMENT_KEYS.every((code) => code.startsWith('Arrow')), '必须声明并拦截四个方向键');
console.log('网页方向键移动控制测试通过');

(async () => {
  const held = new Set();
  const target = { closest: () => null };
  const event = { code: 'KeyR', target, repeat: false };
  assert(controls.isRestartKey(event, held));
  assert(!controls.isRestartKey({ ...event, repeat: true }, held));
  held.add('KeyR'); assert(!controls.isRestartKey(event, held)); held.clear();
  for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) assert(!controls.isRestartKey({ ...event, [modifier]: true }, held));
  for (const tag of ['input', 'textarea', 'contenteditable']) {
    assert(!controls.isRestartKey({ ...event, target: { closest: () => ({ tag }) } }, held), 'typing targets never restart');
  }
  let restarts = 0, done = true, release;
  const restart = controls.createRestartGate(async () => {
    restarts++;
    await new Promise(resolve => { release = resolve; });
    done = false;
  }, () => done);
  const a = restart(), b = restart(true), c = restart();
  assert.strictEqual(a, b); assert.strictEqual(b, c);
  await Promise.resolve(); assert.equal(restarts, 1, 'R plus button clicks coalesce while async reset is pending');
  release(); assert.equal(await a, true);
  assert.equal(await restart(true), false, 'stale settlement button cannot restart replacement match');
  assert.equal(restarts, 1);
  let attempts = 0;
  const failure = controls.createRestartGate(async () => { if (++attempts === 1) throw Error('load failed'); }, () => true);
  await assert.rejects(failure(), /load failed/);
  assert.equal(await failure(), true, 'failed restart does not permanently lock the gate');
  console.log('R typing/repeat/modifier protection, shared async restart and error recovery passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
