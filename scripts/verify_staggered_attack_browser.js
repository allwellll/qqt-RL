'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createServer } = require('../web/server');
const { scene, trace, runScene, SEEDS } = require('./eval_staggered_attack');
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/qqt_staggered_attack_20261007/browser-local');
const harness = `(() => { const Hunter=QQTBunHunterBot, Coop=QQTBunCoopHunterBot;
 const assert=(value,message)=>{if(!value)throw Error(message)}; assert.equal=(a,b)=>assert(a===b,'equality');
 const SEEDS=${JSON.stringify(SEEDS)}, DIRS=[[-1,0],[1,0],[0,-1],[0,1]], idle=()=>[4,0,0,0];
 ${scene.toString()} ${trace.toString()} ${runScene.toString()}
 window.attackTrace=runScene({capacity:2,foeDistance:4,carrier:true}); })()`;
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const server = createServer(); await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = process.env.WEB_URL || `http://127.0.0.1:${server.address().port}/`;
  let browser; const evidence = [];
  try {
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport, isMobile: viewport.width < 600, hasTouch: viewport.width < 600 });
      const page = await context.newPage(), errors = [], writes = [];
      page.on('pageerror', e => errors.push(e.message));
      await context.route('**/*', route => {
        const request = route.request(), url = new URL(request.url());
        if (url.href.startsWith(base) && ['GET','HEAD'].includes(request.method())) return route.continue();
        if (url.pathname.endsWith('/qqt_get_profile')) return route.fulfill({ json: { profile_contract_version: 2, registered: false } });
        if (url.pathname.endsWith('/qqt_leaderboard')) return route.fulfill({ json: [] });
        if (url.pathname.endsWith('/qqt_my_win_leaderboard')) return route.fulfill({ json: { leaderboard_contract_version: 1, rows: [] } });
        writes.push({ method: request.method(), pathname: url.pathname });
        return route.fulfill({ status: 422, json: {} });
      });
      await page.addInitScript(() => { window.setInterval = () => 1; });
      await page.goto(base);
      await page.waitForFunction(() => document.getElementById('loading').classList.contains('done'), null, { timeout: 60000 });
      await page.evaluate(harness);
      const result = await page.evaluate(() => {
        const canvas = document.getElementById('game'), data = canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
        const colors = new Set(); for (let i=0; i<data.length; i+=400) colors.add(`${data[i]},${data[i+1]},${data[i+2]}`);
        return { trace: attackTrace, colors: colors.size, overflow: document.documentElement.scrollWidth > innerWidth,
          character: document.getElementById('character').value, settlementHidden: document.getElementById('settlement').hidden };
      });
      assert(result.trace.stats.chainContinuations > 0 && result.trace.stats.enemyTraps > 0);
      assert.equal(result.trace.stats.selfTraps + result.trace.stats.friendlyTraps, 0);
      assert(result.trace.bubbles.some(b => b.reason === 'bomb_stagger_extension'));
      for (const b of result.trace.bubbles.filter(b => b.reason.startsWith('bomb_stagger'))) {
        assert.equal(b.expectedExplosionTick,b.actualExplosionTick); assert(b.safetyMarginTicks >= 3);
      }
      assert(result.colors > 100 && !result.overflow && result.settlementHidden);
      assert.equal(result.character, 'maomao'); assert.deepEqual(errors, []); assert.deepEqual(writes, []);
      await page.screenshot({ path: path.join(out, `${viewport.width}.png`), fullPage: true });
      evidence.push({ viewport, ...result, errors, writes }); await context.close();
    }
  } finally { if (browser) await browser.close(); await new Promise(resolve => server.close(resolve)); }
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({ base, chromium: 'real', allRemoteWritesMocked: true, evidence },null,2));
  console.log('Chromium desktop/mobile: actual timed attacks, retreat, canvas, default character and zero remote writes passed');
})().catch(error => { console.error(error); process.exitCode=1; });
