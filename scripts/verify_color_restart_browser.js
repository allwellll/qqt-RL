'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.WEB_URL || 'http://127.0.0.1:8080/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/leaderboard_20261006/color-restart-local');
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] });
  const evidence = [];
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport, isMobile: viewport.width < 600, hasTouch: viewport.width < 600 });
      const errors = [], submissions = new Map();
      page.on('pageerror', e => errors.push(e.message));
      // Only result writes are mocked; actual Page scripts, map, assets and read RPC load.
      const submit = route => {
        const payload = route.request().postDataJSON().p_payload;
        assert(!submissions.has(payload.client_match_id), 'one submission per completed match');
        submissions.set(payload.client_match_id, payload);
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
          level: 1, points: submissions.size, games: submissions.size, wins: [...submissions.values()].filter(p => p.result === 'win').length }) });
      };
      await page.route('**/functions/v1/submit-result', submit);
      await page.route('**/rest/v1/rpc/qqt_submit_result', submit);
      await page.addInitScript(() => {
        window.clockOffset = 0; const realNow = Date.now;
        Date.now = () => realNow() + window.clockOffset;
        window.setInterval = cb => { window.appTick = cb; return 1; };
        window.requestAnimationFrame = cb => { window.appFrame = cb; return 1; };
      });
      await page.goto(base);
      await page.waitForFunction(() => window.appFrame, null, { polling: 50, timeout: 60000 });
      await page.evaluate(() => {
        const reset = QQT.Sim.prototype.reset, frame = QQT.Sim.prototype.frameStep;
        window.resetCount = 0;
        QQT.Sim.prototype.reset = function(...args) { if (args[0] && args[0].qqt_id === 806) resetCount++; return reset.apply(this, args); };
        QQT.Sim.prototype.frameStep = function(...args) { window.appSim = this; return frame.apply(this, args); };
        appFrame(performance.now());
      });
      await page.waitForSelector('.loading.done');
      await page.waitForFunction(() => getComputedStyle(document.getElementById('loading')).opacity === '0', null, { polling: 50 });
      const color = await page.evaluate(async () => {
        const level = (await fetch('assets/maps/levels.json').then(r => r.json())).find(l => l.qqt_id === 806);
        const assets = await QQTVisual.loadAssets(level);
        const canvas = document.createElement('canvas'); canvas.width = 900; canvas.height = 880;
        const renderer = QQTVisual.createRenderer(canvas, level, assets), s = appSim;
        const ctx = canvas.getContext('2d'), read = () => ctx.getImageData(0, 0, 900, 880).data;
        const same = (a, b, start = 0) => { let diff = 0; for (let i = start; i < a.length; i++) if (a[i] !== b[i]) diff++; return diff; };
        renderer.render(s, 1000); const alive = read();
        s.alive[0] = false; s.bunRespawn[0] = 30;
        renderer.render(s, 1000); const dead = read();
        let samples = 0, ratioError = 0, dimRatio = 0;
        // Central terrain, excluding player/respawn-marker regions. A black overlay
        // scales RGB together; grayscale would collapse saturation and fail ratios.
        for (let y = 400; y < 540; y += 3) for (let x = 320; x < 580; x += 3) {
          const i = (y * 900 + x) * 4, a = [alive[i], alive[i + 1], alive[i + 2]], d = [dead[i], dead[i + 1], dead[i + 2]];
          if (Math.min(...a) < 30 || Math.max(...a) - Math.min(...a) < 50) continue;
          const ratios = a.map((v, k) => d[k] / v);
          ratioError += Math.max(...ratios) - Math.min(...ratios); dimRatio += ratios[0]; samples++;
        }
        const hudDiff = same(alive, dead, 900 * (QQTVisual.BOARD_OFFSET + QQT.H * QQTVisual.CELL) * 4);
        s.alive[0] = true; s.bunRespawn[0] = 0;
        renderer.render(s, 1000); const respawnDiff = same(alive, read());
        // A caller's filter is restored after the render; internal drawing uses none.
        ctx.filter = 'sepia(1)'; renderer.render(s, 1000); const restoredFilter = ctx.filter; ctx.filter = 'none';
        const spectatorFrame = s.snapshotReplay(); s.alive[0] = false; s.bunRespawn[0] = 30;
        renderer.render(s, 1000, { humanPid: -1, prevPos: s.pos, curPos: s.pos, tickMs: 100, lastTickT: 1000 });
        const spectatorDim = QQTVisual.deathOverlayAlpha(s, -1);
        s.restoreReplay(spectatorFrame);
        return { samples, meanRatioError: ratioError / samples, meanBrightnessRatio: dimRatio / samples,
          hudDiff, respawnDiff, restoredFilter, spectatorDim };
      });
      assert(color.samples > 200 && color.meanRatioError < .03 && color.meanBrightnessRatio > .2 && color.meanBrightnessRatio < .7);
      assert.equal(color.hudDiff, 0); assert.equal(color.respawnDiff, 0); assert.equal(color.restoredFilter, 'sepia(1)'); assert.equal(color.spectatorDim, 0);
      await page.evaluate(() => { appSim._killPlayer(0); appFrame(performance.now() + 16); });
      assert.equal(await page.locator('#game').evaluate(c => c.getContext('2d').filter), 'none');
      assert.equal(await page.locator('aside').evaluate(el => getComputedStyle(el).filter), 'none');
      await page.screenshot({ path: `${out}/${viewport.width}-dead-color.png`, fullPage: true });
      await page.evaluate(() => {
        appSim.lastDied.fill(false); appSim.bunRespawn[0] = 1; appSim._bunRespawnStep(); appFrame(performance.now() + 32);
      });
      assert.equal(await page.evaluate(() => QQTVisual.deathOverlayAlpha(appSim, 0)), 0);
      await page.screenshot({ path: `${out}/${viewport.width}-respawn-color.png`, fullPage: true });
      const outcomes = [];
      for (const [result, title] of [['win', '胜利'], ['loss', '失败'], ['draw', '平局']]) {
        await page.evaluate(async result => {
          appSim.t = 119; appSim.maxSteps = 120;
          appSim.bunStored = result === 'win' ? [[2, 1], [0, 1]] : result === 'loss' ? [[1, 0], [1, 2]] : [[1, 0], [0, 1]];
          clockOffset += 12000; await appTick(); appFrame(performance.now() + 50);
        }, result);
        await page.waitForFunction(() => JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).queue.length === 0, null, { polling: 50 }).catch(async error => {
          const detail = await page.evaluate(() => ({ status: document.getElementById('leaderboard-status').textContent, pending: JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).queue.map(x => ({ result: x.result, game_duration_ms: x.game_duration_ms })), done: appSim.done, tick: appSim.t }));
          throw Error(JSON.stringify({ result, detail, submitted: [...submissions.values()].map(x => x.result), error: error.message }));
        });
        await page.evaluate(() => appFrame(performance.now() + 51));
        assert.equal(await page.locator('#settlement-title').textContent(), title);
        assert.match(await page.locator('#play-again').textContent(), /再来一局.*R/);
        assert.equal(submissions.size, outcomes.length + 1);
        const before = await page.evaluate(() => resetCount), writes = submissions.size;
        await page.locator('.profile-settings').evaluate(el => { el.open = true; });
        await page.locator('#player-nickname').focus(); await page.keyboard.press('r');
        assert.equal(await page.evaluate(() => resetCount), before, 'focused input blocks R');
        assert(await page.locator('#settlement').isVisible());
        await page.locator('#player-nickname').evaluate(el => el.blur());
        await page.screenshot({ path: `${out}/${viewport.width}-${result}-R.png`, fullPage: true });
        await page.keyboard.down('r');
        await page.evaluate(() => {
          window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyR', repeat: true, bubbles: true }));
          // Simultaneous/stale settlement button clicks share the same guarded path.
          document.getElementById('play-again').click(); document.getElementById('play-again').click();
        });
        await page.keyboard.up('r');
        await page.evaluate(() => appFrame(performance.now() + 80));
        assert.equal(await page.evaluate(() => resetCount), before + 1, 'repeat plus double button clicks perform one reset');
        assert.equal(await page.evaluate(() => appSim.done), false);
        assert.equal(await page.locator('#settlement').isVisible(), false);
        assert.equal(submissions.size, writes, 'restart never resubmits previous result');
        outcomes.push({ result, title, inputFocusBlocksR: true, exactlyOneRestart: true, noDuplicateSubmit: true });
      }
      // Hold a real adapter's decision across R: its old action must never step
      // the replacement Sim (asynchronous model adapters use this same path).
      await page.evaluate(() => {
        const registry = QQTBots.createDefaultRegistry({ coopHunter: QQTBunCoopHunterBot });
        const prototype = Object.getPrototypeOf(registry.create('bun.coop_hunter', { difficulty: 'hard' }));
        const act = prototype.act;
        prototype.act = function(...args) {
          const action = act.apply(this, args);
          prototype.act = act;
          return new Promise(resolve => { window.releaseOldAction = () => resolve(action); });
        };
        window.oldSim = appSim;
        window.pendingOldTick = appTick();
      });
      await page.waitForFunction(() => window.releaseOldAction, null, { polling: 50 });
      const beforePending = await page.evaluate(() => resetCount), writesPending = submissions.size;
      await page.keyboard.press('r');
      await page.evaluate(() => appFrame(performance.now() + 100));
      await page.evaluate(async () => { releaseOldAction(); await pendingOldTick; });
      assert.equal(await page.evaluate(() => resetCount), beforePending + 1);
      assert(await page.evaluate(() => appSim !== oldSim && appSim.t === 0 && !appSim.done && !appSim.fuse.some(x => x > 0)));
      assert.equal(submissions.size, writesPending);
      assert.deepEqual(errors, []); assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      evidence.push({ viewport, realMapAndRenderer: true, resultWritesMocked: true, color, outcomes,
        pendingOldActionCancelled: true, errors, overflow: false });
      await page.close();
    }
    fs.writeFileSync(`${out}/checks.json`, JSON.stringify({ url: base, remoteDatabaseWrites: false, evidence }, null, 2) + '\n');
    console.log('Desktop/mobile color-only dimming, HUD RGB parity, immediate respawn, Canvas state and win/loss/draw R/focus/repeat/idempotence passed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
