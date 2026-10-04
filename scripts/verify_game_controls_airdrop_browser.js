'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.WEB_URL || 'http://127.0.0.1:8080/';
const out = path.resolve('runs/game_controls_airdrop_20261005/browser');
const launch = { headless: true, args: ['--no-sandbox'] };
if (process.env.CHROMIUM_PATH) launch.executablePath = process.env.CHROMIUM_PATH;

async function open(browser, viewport, baseline = false) {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('response', r => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
  if (baseline) await page.route('**/controls.js', route => route.fulfill({
    contentType: 'application/javascript', body: execFileSync('git', ['show', '9e26908:web/controls.js'], { encoding: 'utf8' }),
  }));
  await page.addInitScript(() => {
    window.setInterval = (callback) => { window.appTick = callback; return 1; };
    Date.now = () => 19;
  });
  await page.goto(base);
  await page.waitForSelector('.loading.done');
  await page.evaluate(() => {
    const step = QQT.Sim.prototype.frameStep;
    QQT.Sim.prototype.frameStep = function(pid, move, dt) {
      window.appSim = this; window.lastMove = move;
      return step.call(this, pid, move, dt);
    };
  });
  await page.waitForFunction(() => window.appSim);
  return { page, errors };
}
async function moveProbe(page) {
  await page.evaluate(() => {
    const s = appSim;
    for (const key of ['wall', 'brick', 'cover', 'pushable', 'crate', 'fuse']) s[key].fill(0);
    s.pos[0] = 7.5; s.pos[1] = 7.5;
  });
  await page.keyboard.down('ArrowUp'); await page.waitForTimeout(60);
  await page.keyboard.down('ArrowLeft'); await page.waitForTimeout(80);
  const start = await page.evaluate(() => ({ move: lastMove, pos: Array.from(appSim.pos.slice(0,2)) }));
  await page.waitForTimeout(120);
  const end = await page.evaluate(() => ({ move: lastMove, pos: Array.from(appSim.pos.slice(0,2)) }));
  await page.keyboard.up('ArrowLeft'); await page.waitForTimeout(60);
  const release = await page.evaluate(() => lastMove);
  await page.keyboard.up('ArrowUp');
  return { start, end, release };
}

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch(launch);
  const results = [];
  try {
    const baseline = await open(browser, { width: 1440, height: 1000 }, true);
    const old = await moveProbe(baseline.page);
    assert.equal(old.end.move, 0, 'reproduce old fixed up priority');
    results.push({ baseline: old, errors: baseline.errors });
    await baseline.page.close();
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const { page, errors } = await open(browser, viewport);
      const movement = await moveProbe(page);
      assert.equal(movement.end.move, 2);
      assert(movement.end.pos[1] < movement.start.pos[1] && movement.end.pos[0] === movement.start.pos[0]);
      assert.equal(movement.release, 0);
      // Browser repeats use the same code; Set insertion order must not change.
      await page.keyboard.down('ArrowLeft'); await page.keyboard.down('ArrowUp');
      await page.keyboard.down('ArrowLeft'); await page.waitForTimeout(60);
      assert.equal(await page.evaluate(() => lastMove), 0);
      await page.keyboard.up('ArrowLeft'); await page.keyboard.up('ArrowUp');
      await page.keyboard.down('ArrowDown'); await page.keyboard.down('ArrowRight'); await page.waitForTimeout(60);
      assert.equal(await page.evaluate(() => lastMove), 3);
      await page.evaluate(() => window.dispatchEvent(new Event('blur'))); await page.waitForTimeout(60);
      assert.equal(await page.evaluate(() => lastMove), 4);
      await page.keyboard.up('ArrowDown'); await page.keyboard.up('ArrowRight');
      await page.click('#restart');
      await page.waitForTimeout(80);
      await page.evaluate(() => {
        const s = appSim;
        s.t = 269; s.graveyard = Array.from({ length: 12 }, (_, i) => ({ type: i % 5, isSuper: false, count: 1 }));
        s.itemSlots[0] = [{ item: 1, count: 2 }, { item: 2, count: 5 }]; s._syncHeldItem(0);
      });
      const visual = await page.evaluate(async () => {
        for (let n = 0; n < 15; n++) await appTick();
        if (appSim.t !== 284) throw new Error(`airdrop fixture did not advance: ${appSim.t}`);
        const level = (await fetch('assets/maps/levels.json').then(r => r.json())).find(x => x.qqt_id === 806);
        const assets = await QQTVisual.loadAssets(level);
        const canvas = document.createElement('canvas'); canvas.width = 900; canvas.height = 880;
        const renderer = QQTVisual.createRenderer(canvas, level, assets);
        const read = () => canvas.getContext('2d').getImageData(0,0,900,880).data;
        renderer.render(appSim, 1000); const withBird = read();
        const saved = assets.effects.bird; delete assets.effects.bird;
        renderer.render(appSim, 1000); const withoutBird = read(); assets.effects.bird = saved;
        let changed = 0;
        for (let i = 0; i < withBird.length; i += 4) if (withBird[i] !== withoutBird[i] || withBird[i+1] !== withoutBird[i+1] || withBird[i+2] !== withoutBird[i+2]) changed++;
        renderer.render(appSim, 1200); const flapping = read();
        let animated = 0;
        for (let i = 0; i < withBird.length; i+=4) if (withBird[i] !== flapping[i] || withBird[i+1] !== flapping[i+1]) animated++;
        return { changed, animated, bird: appSim.birdFlight(), frames: saved.frames.length,
          falling: appSim.airdropFalls.length, dropped: appSim.airdropDropped,
          landed: appSim.crate.reduce((a,b) => a+b,0), overflow: document.documentElement.scrollWidth > innerWidth };
      });
      assert(visual.changed > 3000 && visual.animated > 100 && visual.frames === 6 && visual.dropped > 0 && !visual.overflow);
      await page.waitForTimeout(100);
      await page.screenshot({ path: `${out}/${viewport.width}-bird.png`, fullPage: true });
      const death = await page.evaluate(async () => {
        const s = appSim; s.trapped[0] = 1;
        await appTick();
        const crates = [];
        for (let cell=0;cell<QQT.N;cell++) if (s.crateCount[cell] > 0) crates.push({ cell, type:s.crateType[cell], count:s.crateCount[cell] });
        return { dead: !s.alive[0], slots: s.itemSlots[0], held: s.heldItem[0], crates,
          pending: s._pendingDeathDrops, status: JSON.parse(document.querySelector('#status').textContent).held_item };
      });
      assert(death.dead && !death.slots.length && !death.held);
      assert(death.crates.some(x=>x.type===3&&x.count===2));
      assert(death.crates.some(x=>x.type===4&&x.count===5));
      await page.waitForTimeout(100);
      death.statusAfterRender = await page.evaluate(() => JSON.parse(document.querySelector('#status').textContent).held_item[0]);
      assert.deepEqual(death.statusAfterRender, []);
      await page.screenshot({ path: `${out}/${viewport.width}-death.png`, fullPage: true });
      assert.deepEqual(errors, []);
      results.push({ viewport, movement, visual, death, errors });
      await page.close();
      console.log(`viewport ${viewport.width}: keyboard, bird pixels and death inventory passed`);
    }
    // Run the actual 2v2 page pipeline on unmodified map 806 through settlement.
    const { page, errors } = await open(browser, { width: 1440, height: 1000 });
    await page.selectOption('#team-mode', '2v2');
    await page.waitForTimeout(100);
    await page.evaluate(() => {
      window.fullMap = { seed: appSim.seed, startBricks: appSim.brick.reduce((a,b)=>a+b,0),
        deaths: 0, placements: 0, flights: 0, drops: 0 };
      const step = QQT.Sim.prototype.step;
      QQT.Sim.prototype.step = function(actions) {
        const info = step.call(this, actions);
        fullMap.deaths += info.died.filter(Boolean).length;
        fullMap.placements += info.placed.filter(Boolean).length;
        if (this.birdFlight()) fullMap.flights++;
        fullMap.drops = Math.max(fullMap.drops, this.airdropDropped);
        for (const slots of this.itemSlots) for (const slot of slots) {
          if (!Number.isInteger(slot.count) || slot.count <= 0) throw new Error('invalid inventory quantity');
        }
        return info;
      };
    });
    let full;
    do {
      full = await page.evaluate(async () => {
        for (let i=0;i<100 && !appSim.done;i++) await appTick();
        return { ...fullMap, tick:appSim.t, done:appSim.done, winner:appSim.winner,
          score:appSim.bunScore, finalBricks:appSim.brick.reduce((a,b)=>a+b,0) };
      });
      console.log(`full map: ${full.tick} ticks, ${full.placements} bubbles, ${full.deaths} deaths`);
    } while (!full.done);
    assert(full.startBricks > 0 && full.tick > 0 && full.placements > 0 && full.finalBricks < full.startBricks);
    assert.deepEqual(errors, []);
    await page.waitForTimeout(100);
    await page.screenshot({ path: `${out}/full-map-result.png`, fullPage:true });
    results.push({ fullMap: full, errors });
    await page.close();
    fs.writeFileSync(`${out}/checks.json`, JSON.stringify(results,null,2)+'\n');
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode=1; });
