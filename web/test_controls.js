'use strict';

const assert = require('assert');
const controls = require('./controls.js');

assert.strictEqual(controls.moveForHeld(new Set(['ArrowUp'])), 0);
assert.strictEqual(controls.moveForHeld(new Set(['ArrowDown'])), 1);
assert.strictEqual(controls.moveForHeld(new Set(['ArrowLeft'])), 2);
assert.strictEqual(controls.moveForHeld(new Set(['ArrowRight'])), 3);
assert.strictEqual(controls.moveForHeld(new Set()), 4);
for (const key of ['KeyW', 'KeyA', 'KeyS', 'KeyD']) {
  assert.strictEqual(controls.moveForHeld(new Set([key])), 4, `${key} 不再控制移动`);
}
assert.deepStrictEqual(controls.BOMB_KEYS, ['Space', 'KeyW'], 'Space 与 W 都放泡');
assert(controls.MOVEMENT_KEYS.every((code) => code.startsWith('Arrow')), '必须声明并拦截四个方向键');
console.log('网页方向键移动控制测试通过');
