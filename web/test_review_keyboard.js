'use strict';
const assert = require('node:assert/strict');
const { bindReviewKeyboard, steppedTick } = require('./review_keyboard');

function target(tagName = 'DIV', extra = {}) {
  return { tagName, isContentEditable: false, getAttribute: () => null, ...extra };
}
function fire(key, { repeat = false, target: eventTarget = target() } = {}) {
  let prevented = false;
  const event = { key, repeat, target: eventTarget, preventDefault() { prevented = true; } };
  return { event, get prevented() { return prevented; } };
}

const calls = [];
const listener = bindReviewKeyboard({ onStep: (direction, amount) => calls.push([direction, amount]) });
assert.deepEqual(listener(fire('ArrowLeft').event), { handled: true, prevented: true });
assert.deepEqual(listener(fire('ArrowRight').event), { handled: true, prevented: true });
assert.deepEqual(listener(fire('ArrowRight', { repeat: true }).event), { handled: true, prevented: true });
assert.deepEqual(calls, [[-1, 1], [1, 1], [1, 4]], 'press steps one tick and repeat advances quickly');
assert.equal(steppedTick(10, -1, 1, 300), 9);
assert.equal(steppedTick(10, 1, 4, 300), 14);
assert.equal(steppedTick(0, -1, 4, 300), 0, 'start clamps');
assert.equal(steppedTick(300, 1, 4, 300), 300, 'end clamps');

for (const editable of [target('INPUT'), target('TEXTAREA'), target('SELECT'),
  target('INPUT', { type: 'range' }), target('DIV', { isContentEditable: true }),
  target('DIV', { getAttribute: name => name === 'contenteditable' ? 'true' : null })]) {
  const result = listener(fire('ArrowLeft', { target: editable }).event);
  assert.deepEqual(result, { handled: false, prevented: false });
}
assert.deepEqual(listener(fire('ArrowUp').event), { handled: false, prevented: false });
assert.deepEqual(listener(fire('keyup').event), { handled: false, prevented: false }, 'release has no delayed repeat');
const html = require('node:fs').readFileSync('web/review.html', 'utf8');
assert(html.indexOf('src="review_keyboard.js"') < html.indexOf('src="review.js"'), 'keyboard helper loads before review app');
const app = require('node:fs').readFileSync('web/review.js', 'utf8');
assert.match(app, /document\.addEventListener\('keydown',keyboardStep\)/);
assert.match(app, /seek\(ReviewKeyboard\.steppedTick\(player\.sim\.t,direction,amount,doc\.actions\.length\)\)/);
assert.match(app, /function seek\(tick\)\{[^}]*playing=false;[^}]*hashPosition\(\)/);
console.log('Review keyboard RED/GREEN: exact single step, accelerated repeat, editable controls and unrelated keys');
