const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const engine = require('./sim-bun-native-v5-facing-window');
const controls = require('./controls');
const source = name => fs.readFileSync(require.resolve(`./${name}`), 'utf8');
const nodes = new Map();
const fills = [], boxes = [];
const ctx = {
  clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, arc() {}, ellipse() {}, fill() {},
  save() {}, restore() {}, translate() {},
  fillRect(...rect) { fills.push([this.fillStyle, ...rect]); },
  strokeRect(...rect) { boxes.push([this.strokeStyle, ...rect]); },
};
const node = id => {
  if (!nodes.has(id)) nodes.set(id, {
    width: 750, height: 650, dataset: {}, addEventListener() {}, getContext: () => ctx,
  });
  return nodes.get(id);
};
let sim, profile;
class TrainingSim extends engine.Sim { constructor(...args) { super(...args); sim = this; } }
vm.runInNewContext(source('joystick-tuner.js'), {
  document: { getElementById: node }, window: { addEventListener() {} }, requestAnimationFrame() {}, AbortController,
  QQTBunNativeV5FacingWindow: { ...engine, Sim: TrainingSim },
  QQTControls: { ...controls, bindTouchControls(_zone, _held, _press, options) { profile = options; return () => {}; } },
  localStorage: { setItem() {}, getItem: key => key.endsWith('.v1') ? '{"deadzoneRatio":0.18}' : null },
});
const [cellW, cellH] = [750 / engine.W, 650 / engine.H];
const brickFills = fills.filter(([color]) => color === '#8a5540').map(([, ...rect]) => rect);
assert.deepEqual(brickFills, Array.from(sim.brick).flatMap((brick, i) => brick && !sim.wall[i]
  ? [[(i % engine.W) * cellW, Math.floor(i / engine.W) * cellH, cellW, cellH]] : []), 'brick paint covers the same full cells as physics');
const half = engine.NATIVE_HALF_PX / engine.NATIVE_CELL_PX;
assert(boxes.some(([, x,y,w,h]) => x === (sim.pos[1]-half)*cellW && y === (sim.pos[0]-half)*cellH
  && w === 2*half*cellW && h === 2*half*cellH), 'player box uses the unshifted physics center');
assert(!Array.from(sim.fuse).some((fuse, i) => fuse > 0 && sim.brick[i]), 'training bubble occupies an open cell');
for (let i=0;i<60;i++) sim.frameStep(0, engine.MOVE_RIGHT, 1/60);
assert.equal(sim.pos[0], 10.5);
assert(sim.pos[1] > 3 && sim.pos[1]+half < 4 && sim.pos[1]+half >= 4-1/engine.NATIVE_CELL_PX-1e-6, 'player edge stops at the drawn brick boundary');
for (const [key,value] of Object.entries(controls.TRAJECTORY_TOUCH_PROFILE)) assert.equal(profile[key], value);
assert.match(source('app.js'), /QQTControls\.bindTouchControls\(touchControls, held,[\s\S]*?\}, QQTControls\.TRAJECTORY_TOUCH_PROFILE\);/);
console.log('Tuner collision geometry and shared touch profile PASS');
