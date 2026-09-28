'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const web = __dirname;
const required = [
  'assets/bg/抢包子.png',
  'assets/bg/水面.png',
  'assets/maps/elements.json',
  'assets/mapElem/bun/elem1_stand.png',
  'assets/mapElem/bun/elem28_stand.png',
  'assets/角色4×4精灵图.png',
  'assets/角色c4×4.png',
  'assets/bomb-custom/经典黄泡泡.png',
  'assets/shadow.png',
  'assets/flame/flame_C_1.png',
  'assets/flame/flame_U_6.png',
  'assets/flame/flame_D_6.png',
  'assets/flame/flame_L_6.png',
  'assets/flame/flame_R_6.png',
];
for (const relative of required) {
  assert(fs.existsSync(path.join(web, relative)), `missing visual asset: ${relative}`);
}

const visual = require('./visual_renderer.js');
const appSource = fs.readFileSync(path.join(web, 'app.js'), 'utf8');
const htmlSource = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
assert(appSource.includes('QQTVisual.loadAssets(level)'), 'app must load original visual assets');
assert(appSource.includes('renderer.addExplosion(info'), 'app must forward blast events to renderer');
assert(appSource.includes('renderer.render(sim'), 'app must use sprite renderer');
assert(!appSource.includes("context.arc(x, y, 18"), 'legacy circle-player renderer must be removed');
assert(!appSource.includes("drawCell(row, column, '#546a70'"), 'legacy block-map renderer must be removed');
assert(htmlSource.includes('visual_renderer.js'), 'visual renderer must load before app');
assert(htmlSource.includes('height="810"'), 'canvas must reserve the original 30px top overflow band');
assert.equal(visual.CELL, 60);
assert.equal(visual.BOARD_OFFSET, 30);
assert.equal(visual.explosionFrame(0.01).activeScale, 1);
assert.equal(visual.explosionFrame(0.10).tip, 1);
assert.equal(visual.explosionFrame(0.22).tip, 5);
assert.equal(visual.explosionFrame(0.30).tip, 1);
assert.equal(visual.explosionFrame(0.40).tip, 6);
assert.equal(visual.explosionFrame(0.40).activeScale, 1);
assert.equal(visual.bombFrame(0.00, 4), 0);
assert.equal(visual.bombFrame(0.26, 4), 1);
assert.equal(visual.bombFrame(0.99, 4), 3);
assert.equal(visual.bombFrame(1.01, 4), 0);
assert.equal(visual.bombAgeSeconds(30, 30, 10), 0);
assert.equal(visual.bombAgeSeconds(25, 30, 10), 0.5);
assert.equal(visual.bombAgeSeconds(1, 30, 10), 2.9);
assert.equal(visual.playerVisualY(6.5, 127), 282.6,
  'player sprite feet must align to the simulator collision radius');

const bunSim = {
  isBun: true,
  bunBases: [[1, 4], [1, 8]],
  bunStored: [[1, 0], [0, 2]],
  bunLoose: new Uint8Array(195 * 2),
};
bunSim.bunLoose[(6 * 15 + 7) * 2] = 3;
assert.deepEqual(visual.bunTokens(bunSim), [
  { row: 2, column: 5, team: 0, count: 1, size: 0.82, xOffset: -0.13 },
  { row: 2, column: 9, team: 1, count: 2, size: 0.82, xOffset: 0.13 },
  { row: 6, column: 7, team: 0, count: 3, size: 0.78, xOffset: -0.12 },
]);
assert(appSource.includes('requestAnimationFrame(animationFrame)'),
  'sprite and flame animation must render independently of the 10 Hz simulation tick');
assert(appSource.includes('renderer.reset()'), 'restart must discard stale explosion and facing state');

const level = JSON.parse(fs.readFileSync(path.join(web, 'assets/maps/levels.json')))
  .find((item) => item.qqt_id === 806);
const ids = visual.levelElementIds(level);
assert.deepEqual(ids, [8001, 8002, 8003, 8004, 8005, 8006, 8009, 8010, 8011, 8012, 8013, 8028]);
console.log('visual parity assets and animation contract: ok');
