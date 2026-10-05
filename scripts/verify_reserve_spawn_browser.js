'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const out = path.resolve('runs/bot_reserve_spawn_20261005/browser');
const launch = { headless: true, args: ['--no-sandbox'] };
if (process.env.CHROMIUM_PATH) launch.executablePath = process.env.CHROMIUM_PATH;
const base = process.env.WEB_URL || 'http://127.0.0.1:8080/';

async function open(browser, viewport, baseline = false) {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('response', (r) => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
  if (baseline) await page.route('**/sim.js', (route) => route.fulfill({ contentType: 'application/javascript',
    body: execFileSync('git', ['show', '5ecb5c9:web/sim.js'], { encoding: 'utf8' }) }));
  await page.addInitScript(() => {
    window.setInterval = (callback) => { window.appTick = callback; return 1; };
    Date.now = () => 19;
  });
  await page.goto(base);
  await page.waitForSelector('.loading.done');
  await page.evaluate(() => {
    const frame = QQT.Sim.prototype.frameStep;
    QQT.Sim.prototype.frameStep = function(...args) { window.appSim = this; return frame.apply(this, args); };
  });
  await page.waitForFunction(() => window.appSim);
  return { page, errors };
}

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch(launch), rows = [];
  try {
    const old = await open(browser, { width: 1440, height: 1000 }, true);
    const reproduction = await old.page.evaluate(() => {
      const s = appSim, birth = s.invuln[0];
      s._killPlayer(0); s.bunRespawn[0] = 1; s.lastDied.fill(false); s._bunRespawnStep();
      return { birth, respawn: s.invuln[0] };
    });
    assert.deepStrictEqual(reproduction, { birth: 0, respawn: 10 });
    rows.push({ oldProtection: reproduction }); await old.page.close();
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const { page, errors } = await open(browser, viewport);
      assert.equal(await page.locator('#opponent').inputValue(), 'bun.coop_hunter@hard');
      const initial = await page.evaluate(() => ({ timer: appSim.spawnProtection, invuln: appSim.invuln }));
      assert(initial.timer.every((t) => t === 30) && initial.invuln.every((t) => t === 30));
      const visual = await page.evaluate(async () => {
        const level = (await fetch('assets/maps/levels.json').then((r) => r.json())).find((l) => l.qqt_id === 806);
        const assets = await QQTVisual.loadAssets(level);
        const canvas = document.createElement('canvas'); canvas.width = 900; canvas.height = 880;
        const renderer = QQTVisual.createRenderer(canvas, level, assets);
        const read = () => canvas.getContext('2d').getImageData(0, 0, 900, 880).data;
        const delta = (a, b) => { let n = 0; for (let i = 0; i < a.length; i += 4)
          if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) n++; return n; };
        appSim.pos.set([5.5, 5.5, 10.5, 10.5]);
        renderer.render(appSim, 1000); const halo = read();
        renderer.render(appSim, 1200); const animated = delta(halo, read());
        const snapshot = appSim.snapshotReplay(null);
        appSim.spawnProtection.fill(0); renderer.render(appSim, 1000); const without = read();
        const pixels = delta(halo, without);
        appSim.restoreReplay(snapshot); renderer.render(appSim, 1000); const replayPixels = delta(halo, read());
        appSim.invuln.fill(0); appSim.spawnProtection.fill(0); renderer.render(appSim, 1000);
        const expiredPixels = delta(without, read());
        appSim._killPlayer(0); appSim.bunRespawn[0] = 1; appSim.lastDied.fill(false); appSim._bunRespawnStep();
        appSim.pos[0] = 5.5; appSim.pos[1] = 5.5; renderer.render(appSim, 1000);
        return { pixels, animated, replayPixels, expiredPixels, respawnPixels: delta(without, read()),
          respawnTimer: appSim.spawnProtection[0], origin: [assets.effects.protection.ox, assets.effects.protection.oy] };
      });
      assert(visual.pixels > 300 && visual.animated > 300 && visual.respawnPixels > 300);
      assert.equal(visual.replayPixels, 0); assert.equal(visual.expiredPixels, 0); assert.equal(visual.respawnTimer, 30);
      await page.waitForTimeout(100);
      await page.screenshot({ path: `${out}/${viewport.width}-halo.png`, fullPage: true });
      await page.selectOption('#team-mode', '1v2'); await page.waitForTimeout(100);
      const attack = await page.evaluate(async () => {
        const s = appSim;
        for (const key of ['wall', 'brick', 'cover', 'pushable', 'crate', 'fuse', 'blastLinger']) s[key].fill(0);
        s.pos.set([7.5, 7.5, 5.5, 5.5, 11.5, 13.5]);
        s.invuln.fill(0); s.spawnProtection.fill(0); s.spdG.fill(1); s.blastCap.fill(3); s.bombsCap.fill(5);
        s.bunStored = [[0, 0], [0, 0]]; s.bunInitial = [Infinity, Infinity];
        const events = [], decisions = [], step = QQT.Sim.prototype.step;
        const analyze = QQTBunCoopHunterBot.BunCoopHunterBot.prototype.analyzeSim;
        QQTBunCoopHunterBot.BunCoopHunterBot.prototype.analyzeSim = function(sim, pid) {
          const d = analyze.call(this, sim, pid); decisions[pid] = { reason: d.reason, mode: d.mode }; return d;
        };
        let safe = true, chain = false, enemyTrapped = false;
        QQT.Sim.prototype.step = function(actions) {
          const cells = this.team.map((_, p) => this.centerCell(p)[0] * QQT.W + this.centerCell(p)[1]);
          const fuse = Array.from(this.fuse), info = step.call(this, actions);
          for (let p = 1; p < this.nPlayers; p++) if (info.placed[p]) events.push({ pid: p,
            cell: cells[p], tick: this.t - 1, ...decisions[p], anchorFuse: fuse[80] });
          const first = events.find((e) => e.pid === 1 && e.reason === 'bomb_reserve_multi');
          const second = events.find((e) => e.pid === 1 && e.reason === 'bomb_reserve_second');
          const connector = events.find((e) => e.pid === 1 && e.reason === 'bomb_chain');
          if (first && second && connector && info.triggered[first.cell] && info.triggered[second.cell] && info.triggered[connector.cell]) {
            chain = true; enemyTrapped = this.trapped[0] > 0;
          }
          safe &&= this.alive[1] && this.alive[2] && !this.trapped[1] && !this.trapped[2];
          return info;
        };
        for (let n = 0; n < 32; n++) await appTick();
        QQT.Sim.prototype.step = step;
        QQTBunCoopHunterBot.BunCoopHunterBot.prototype.analyzeSim = analyze;
        return { events, safe, chain, enemyTrapped };
      });
      assert(attack.safe && attack.chain && attack.enemyTrapped);
      assert(attack.events.find((e) => e.pid === 1 && e.reason === 'bomb_chain').anchorFuse <= 12);
      const layout = await page.evaluate(() => {
        const c = document.querySelector('#game'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        const colors = new Set(); for (let i = 0; i < d.length; i += 400) colors.add(`${d[i]},${d[i + 1]},${d[i + 2]}`);
        return { colors: colors.size, overflow: document.documentElement.scrollWidth > innerWidth };
      });
      assert(layout.colors > 100 && !layout.overflow); assert.deepStrictEqual(errors, []);
      await page.screenshot({ path: `${out}/${viewport.width}-chain.png`, fullPage: true });
      rows.push({ viewport, initial, visual, attack, layout, errors }); await page.close();
    }
    const { page, errors } = await open(browser, { width: 1440, height: 1000 });
    await page.selectOption('#team-mode', '2v2'); await page.waitForTimeout(100);
    await page.evaluate(() => {
      window.full = { seed: appSim.seed, initialBricks: appSim.brick.reduce((a, b) => a + b, 0),
        placements: 0, reserves: 0, replenishments: 0, chains: 0, earlyChainBombs: 0,
        selfTraps: 0, friendlyTraps: 0, botFriendlyTraps: 0, humanFriendlyTraps: 0, deaths: 0 };
      const analyze = QQTBunCoopHunterBot.BunCoopHunterBot.prototype.analyzeSim, decisions = [];
      QQTBunCoopHunterBot.BunCoopHunterBot.prototype.analyzeSim = function(s, p) {
        const d = analyze.call(this, s, p); decisions[p] = d.reason; return d;
      };
      const step = QQT.Sim.prototype.step;
      QQT.Sim.prototype.step = function(actions) {
        const fuse = this.fuse.slice(), trapped = this.trapped.slice(), info = step.call(this, actions);
        for (let c = 0; c < fuse.length; c++) if (info.triggered[c] && fuse[c] > 1) full.earlyChainBombs++;
        for (let p = 0; p < this.nPlayers; p++) {
          if (info.placed[p]) {
            full.placements++;
            if (decisions[p] === 'bomb_reserve_multi' || decisions[p] === 'bomb_reserve_second') full.reserves++;
            if (decisions[p] === 'bomb_reserve_replenish') full.replenishments++;
            if (decisions[p] === 'bomb_chain') full.chains++;
          }
          if (info.died[p]) full.deaths++;
          if (!trapped[p] && this.trapped[p]) {
            if (info.physicalDamageSource[p][p]) full.selfTraps++;
            if (this.team.some((t, q) => q !== p && t === this.team[p] && info.physicalDamageSource[p][q])) {
              full.friendlyTraps++;
              full[p === 0 ? 'humanFriendlyTraps' : 'botFriendlyTraps']++;
            }
          }
        }
        return info;
      };
    });
    let full;
    do {
      full = await page.evaluate(async () => {
        for (let n = 0; n < 100 && !appSim.done; n++) await appTick();
        return { ...window.full, tick: appSim.t, done: appSim.done, winner: appSim.winner,
          score: appSim.bunScore, finalBricks: appSim.brick.reduce((a, b) => a + b, 0) };
      });
      console.log(`real page full map: ${full.tick} ticks, ${full.placements} bubbles`);
    } while (!full.done);
    assert(full.initialBricks === 81 && full.finalBricks < 81 && full.placements > 0);
    assert.deepStrictEqual(errors, []);
    await page.waitForTimeout(100); await page.screenshot({ path: `${out}/full-map.png`, fullPage: true });
    rows.push({ fullMap: full, errors }); await page.close();
    fs.writeFileSync(`${out}/checks.json`, JSON.stringify(rows, null, 2) + '\n');
    console.log(JSON.stringify(rows));
  } finally { await browser.close(); }
})().catch((e) => { console.error(e); process.exitCode = 1; });
