'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { webcrypto } = require('crypto');
const LB = require('../web/leaderboard');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createServer } = require('../web/server');
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/qqt_loading_20261007/browser-local-loading');
const lossText = '小伙子，再沉淀沉淀吧';
// Delayed images intentionally keep document.fonts.ready pending in Chromium.
process.env.PW_TEST_SCREENSHOT_NO_FONTS_READY = '1';
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = process.env.WEB_URL || `http://127.0.0.1:${server.address().port}/`;
  let browser;
  const evidence = [];
  try {
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      // Generate the actual persisted shape through the client, without network writes.
      const store = new Map(); let now = Date.now();
      const c = LB.createClient({ config: {}, storage: { getItem: k => store.get(k), setItem: (k, v) => store.set(k, v) },
        crypto: webcrypto, now: () => now, fetch: async () => { throw new Error('Unexpected fixture request'); } });
      const match = c.begin({ opponent: 'bun.coop_hunter', difficulty: 'hard', seed: 7, mode: '1v2', map_id: 'training' });
      now += 5000; await c.finish(match, { result: 'loss', gameDurationMs: 5000, autoSubmit: false });
      const saved = store.get('qqt.leaderboard.v1');
      const context = await browser.newContext({ viewport, isMobile: viewport.width < 600, hasTouch: viewport.width < 600 });
      const page = await context.newPage(), errors = [], writes = [], unexpected = [];
      const held = { levels: [], metadata: [], images: [] }, released = new Set();
      page.on('pageerror', e => errors.push(e.message));
      await context.route('**/*', async route => {
        const request = route.request(), url = request.url();
        if (url.startsWith(base)) {
          let phase;
          if (url.includes('assets/maps/levels.json')) phase = 'levels';
          else if (url.includes('assets/native/sprites.json')) phase = 'metadata';
          else if (url.includes('assets/') && /\.png(?:\?|$)/.test(url)) phase = 'images';
          if (phase && !released.has(phase)) { held[phase].push(route); return; }
          return route.continue();
        }
        const endpoint = new URL(url).pathname;
        if (endpoint.endsWith('/qqt_get_profile')) return route.fulfill({ json: { profile_contract_version: 2, registered: false } });
        if (endpoint.endsWith('/qqt_leaderboard')) return route.fulfill({ json: [] });
        if (endpoint.endsWith('/qqt_my_win_leaderboard')) return route.fulfill({ json: { leaderboard_contract_version: 1, rows: [] } });
        if (/qqt_submit_result|submit-result|qqt_update_profile/.test(endpoint)) {
          writes.push(endpoint); return route.fulfill({ status: 422, json: {} });
        }
        unexpected.push(endpoint); return route.abort();
      });
      await page.addInitScript(saved => {
        localStorage.setItem('qqt.leaderboard.v1', saved);
        window.clockOffset = 0; const now = Date.now; Date.now = () => now() + clockOffset;
        window.setInterval = cb => { window.appTick = cb; return 1; };
        window.requestAnimationFrame = cb => { window.appFrame = cb; return 1; };
        window.lossTrace = [];
        const sample = () => {
          const status = document.getElementById('settlement-status'), card = document.getElementById('settlement');
          const ready = document.getElementById('loading')?.classList.contains('done') || false;
          const value = { ready, hidden: card?.hidden, text: status?.textContent || '' };
          const last = lossTrace.at(-1);
          if (!last || JSON.stringify(last) !== JSON.stringify(value)) lossTrace.push(value);
        };
        new MutationObserver(sample).observe(document, { childList: true, subtree: true, attributes: true, characterData: true });
      }, saved);
      await page.goto(base, { waitUntil: 'domcontentloaded' });
      const snapshots = [];
      for (const phase of ['levels', 'metadata', 'images']) {
        await page.waitForFunction(() => document.getElementById('leaderboard-status'), null, { polling: 50, timeout: 60000 });
        for (let attempt = 0; !held[phase].length && attempt < 600; attempt++) await page.waitForTimeout(100);
        assert(held[phase].length, `missing delayed ${phase} requests`);
        await page.waitForTimeout(700);
        snapshots.push({ phase, ...(await page.evaluate(() => ({ text: document.getElementById('settlement-status').textContent,
          hidden: document.getElementById('settlement').hidden, ready: document.getElementById('loading').classList.contains('done') }))) });
        await page.screenshot({ path: path.join(out, `${viewport.width}-loading-${phase}.png`), fullPage: true });
        released.add(phase); await Promise.all(held[phase].splice(0).map(route => route.continue()));
      }
      const trace = await page.evaluate(() => lossTrace);
      fs.writeFileSync(path.join(out, `${viewport.width}-loading-trace.json`), JSON.stringify({ snapshots, trace }, null, 2));
      assert(snapshots.every(s => !s.ready && s.hidden && !s.text), 'RED: restored loss leaks into delayed asset loading');
      assert(trace.every(s => s.ready || !s.text.includes(lossText)), 'no transient loss text during initialization or loading');
      try { await page.waitForFunction(() => window.appFrame, null, { polling: 50, timeout: 60000 }); }
      catch (error) {
        fs.writeFileSync(path.join(out, `${viewport.width}-init-error.json`), JSON.stringify({ errors, unexpected,
          held: Object.fromEntries(Object.entries(held).map(([k, v]) => [k, v.length])),
          loading: await page.locator('#loading-text').textContent() }, null, 2));
        throw error;
      }
      await page.evaluate(() => {
        const reset = QQT.Sim.prototype.reset;
        QQT.Sim.prototype.reset = function (...args) { window.appSim = this; return reset.apply(this, args); };
        const step = QQT.Sim.prototype.frameStep;
        QQT.Sim.prototype.frameStep = function (...args) { window.appSim = this; return step.apply(this, args); };
        appFrame(performance.now());
      });
      assert.equal(await page.locator('#settlement-status').textContent(), lossText, 'valid saved loss restored only after first frame');
      await page.locator('#settlement-close').click();
      await page.waitForFunction(() => { appFrame(performance.now()); return document.getElementById('settlement').hidden; }, null, { polling: 50 });
      assert.equal(await page.locator('#settlement-status').textContent(), '', 'restart clears old loss DOM');
      await page.evaluate(async () => {
        appSim.t = 119; appSim.maxSteps = 120;
        appSim.bunStored = [[0, 0], [0, 0]]; appSim.bunStored[1 - appSim.team[0]] = [1, 1];
        clockOffset += 12000; await appTick(); appFrame(performance.now());
      });
      assert.equal(await page.locator('#settlement-status').textContent(), lossText, 'real Sim loss still displays exact prompt');
      assert(!await page.locator('#settlement-form').isVisible());
      await page.locator('#settlement-form').evaluate(e => e.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
      await page.waitForTimeout(150);
      await page.screenshot({ path: path.join(out, `${viewport.width}-real-loss.png`), fullPage: true });
      assert.deepEqual(writes, []); assert.deepEqual(unexpected, []); assert.deepEqual(errors, []);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      evidence.push({ viewport, snapshots, trace, realLoss: true, restartClearsDOM: true, writes, unexpected, errors });
      await context.close();
    }
    fs.writeFileSync(path.join(out, 'checks.json'), JSON.stringify({ url: base, browser: await browser.version(), evidence }, null, 2));
    console.log('Delayed levels/metadata/images: no transient loss, restored/real loss, restart DOM cleanup, desktop/mobile, zero writes passed');
  } finally { if (browser) await browser.close(); await new Promise(resolve => server.close(resolve)); }
})().catch(e => { console.error(e); process.exitCode = 1; });
