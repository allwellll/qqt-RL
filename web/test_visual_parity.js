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
assert(appSource.includes('QQTVisual.loadAssets(level, loadingStep)'), 'app must load original visual assets with progress');
assert(htmlSource.includes('id="loading"') && htmlSource.includes('loading-progress'), 'page must show a loading overlay');
assert(fs.existsSync(path.join(web, 'assets/point.png')), 'missing original player arrow point.png');
assert(appSource.includes('renderer.addExplosion(info'), 'app must forward blast events to renderer');
assert(appSource.includes('renderer.render(sim'), 'app must use sprite renderer');
assert(!appSource.includes("context.arc(x, y, 18"), 'legacy circle-player renderer must be removed');
assert(!appSource.includes("drawCell(row, column, '#546a70'"), 'legacy block-map renderer must be removed');
assert(htmlSource.includes('visual_renderer.js'), 'visual renderer must load before app');
assert(htmlSource.includes('height="880"'), 'canvas must reserve the 30px top band + 780px board + 70px item strip below');
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
assert.equal(visual.playerVisualY(6.5, 127), 276.5,
  'player feet sit 9 source px below the logical center so the shadow stays inside the occupied cell');
assert.equal(visual.respawnSeconds(100), 10);
assert.equal(visual.respawnSeconds(91), 10);
assert.equal(visual.respawnSeconds(90), 9);
assert.equal(visual.respawnSeconds(1), 1);
for (const name of ['放炮', '爆炸', '吃道具音效']) {
  assert(fs.existsSync(path.join(web, `assets/snd/${name}.wav`)), `missing sound: ${name}`);
}
assert(fs.existsSync(path.join(web, 'assets/native/syrup_pop.wav')), 'missing native syrup pop sound');
assert(fs.existsSync(path.join(web, 'assets/native/sprites.json')), 'missing native actor manifest');
assert(fs.existsSync(path.join(web, 'assets/native/maomao_stand.png')), 'missing Maomao actor');
assert(htmlSource.includes('sound.js') && htmlSource.includes('sound-toggle'), 'page must load sound effects with a toggle');
assert(appSource.includes('QQTSound.detectEvents'), 'app must play sounds from step events');
const items = JSON.parse(fs.readFileSync(path.join(web, 'assets/item/items.json')));
for (const key of ['bun', 'banana_pickup', 'glue_pickup', 'banana_field', 'glue_field', 'random', 'bomb', 'power', 'speed', 'fast_shoe']) {
  assert(items[key] && fs.existsSync(path.join(web, items[key].file)), `missing client item sprite: ${key}`);
}
assert.equal(items.bun.source, 'object/item/item11_stand.img');
assert.equal(items.banana_field.source, 'object/item/item42_stand.img');
assert.equal(items.glue_field.source, 'object/item/item43_stand.img');
assert.equal(visual.crateSpriteKey(-1, false), 'random');
assert.equal(visual.crateSpriteKey(0, true), 'bomb_super');
assert.equal(visual.crateSpriteKey(3, false), 'banana_pickup');
assert.equal(visual.crateSpriteKey(4, false), 'glue_pickup');
assert.equal(visual.crateSpriteKey(5, false), 'fast_shoe');

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
