'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.WEB_URL || 'https://allwellll.github.io/qqt-RL/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/leaderboard_20261006/ui-pages');
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] });
  const evidence = [];
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport, isMobile: viewport.width < 600, hasTouch: viewport.width < 600 });
      const page = await context.newPage(), errors = [];
      page.on('pageerror', e => errors.push(e.message));
      await page.addInitScript(() => {
        window.setInterval = cb => { window.appTick = cb; return 1; };
        window.requestAnimationFrame = cb => { window.appFrame = cb; return 1; };
        const proto = CanvasRenderingContext2D.prototype;
        const text = proto.fillText;
        proto.fillText = function(value, ...args) {
          if (String(value).includes('秒复活')) (window.markerTexts ||= []).push(String(value));
          return text.call(this, value, ...args);
        };
      });
      await page.goto(base); await page.waitForFunction(() => window.appFrame, null, { polling: 50 });
      assert.equal(await page.locator('#character').inputValue(), 'maomao');
      assert((await page.locator('#character-portrait').getAttribute('src')).includes('maomao'));
      await page.locator('#character').selectOption('pipi'); await page.reload(); await page.waitForFunction(() => window.appFrame, null, { polling: 50 });
      assert.equal(await page.locator('#character').inputValue(), 'pipi', 'stored explicit character choice survives reload');
      await page.locator('#team-mode').selectOption('2v2');
      await page.evaluate(() => {
        const step = QQT.Sim.prototype.frameStep;
        QQT.Sim.prototype.frameStep = function(...args) { window.appSim = this; return step.apply(this, args); };
        appFrame(performance.now());
        for (let pid = 0; pid < appSim.nPlayers; pid++) appSim._killPlayer(pid);
        window.markerTexts = []; appFrame(performance.now() + 16);
      });
      await page.waitForSelector('.loading.done');
      await page.waitForTimeout(450);
      assert.equal(await page.evaluate(() => QQTVisual.respawnMarkers(appSim, 0).length), 4);
      assert.equal(await page.evaluate(() => new Set(markerTexts).size), 4);
      assert.equal(await page.locator('#game').evaluate(c => c.getContext('2d').filter), 'none');
      await page.screenshot({ path: `${out}/${viewport.width}-dead.png`, fullPage: true });
      await page.evaluate(() => {
        appSim.lastDied.fill(false); appSim.bunRespawn.fill(1); appSim._bunRespawnStep();
        window.markerTexts = []; appFrame(performance.now() + 32);
      });
      assert.equal(await page.evaluate(() => markerTexts.length), 0);
      assert.equal(await page.evaluate(() => QQTVisual.deathOverlayAlpha(appSim, 0)), 0);
      // No mock network routes: isolated client with no finishable identity only renders
      // settlement state. The remote submit/read path is covered by the other verifier.
      await page.evaluate(() => {
        const client = QQTLeaderboard.createClient({ config: {}, storage: { getItem: () => null, setItem() {} }, crypto, fetch });
        const match = client.begin({}); client.finish(match, { result: 'draw', gameDurationMs: 0 });
        QQTLeaderboard.renderSettlement(document, client.state()); appSim.done = true;
      });
      assert.equal(await page.locator('#settlement-title').textContent(), '平局');
      await page.screenshot({ path: `${out}/${viewport.width}-settlement.png`, fullPage: true });
      await page.locator('#play-again').click();
      await page.evaluate(() => appFrame(performance.now() + 64));
      assert.equal(await page.evaluate(() => appSim.done), false);
      assert.equal(await page.locator('#settlement').isVisible(), false);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(errors, []);
      evidence.push({ viewport, networkMock: false, defaultMaomao: true, storedPipi: true,
        fourPlayerSpawnMarkers: true, canvasFilterRestored: true, respawnClearsMarkers: true,
        settlementRestart: true, errors });
      await context.close();
    }
    fs.writeFileSync(`${out}/checks.json`, JSON.stringify({ url: base, evidence }, null, 2) + '\n');
    console.log('Desktop/mobile: Maomao, persisted character, four authoritative spawn markers, filter, respawn and settlement restart passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
