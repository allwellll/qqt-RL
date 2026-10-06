'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.WEB_URL || 'http://127.0.0.1:8080/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/leaderboard_20261006/menu-local');
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] });
  const evidence = [];
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport, isMobile: viewport.width < 600, hasTouch: viewport.width < 600 });
      const page = await context.newPage(), errors = [], rpc = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('request', r => {
        if (r.url().includes('/rest/v1/rpc/')) rpc.push(r.url().split('/').pop());
        assert(!r.url().includes('qqt_submit_result') && !r.url().includes('/functions/v1/'), 'menu acceptance must not write remote results');
      });
      await page.addInitScript(() => {
        window.setInterval = cb => { window.appTick = cb; return 1; };
        window.requestAnimationFrame = cb => { window.appFrame = cb; return 1; };
      });
      async function load() {
        await page.goto(base);
        await page.waitForFunction(() => window.appFrame, null, { polling: 50 });
        await page.evaluate(() => appFrame(performance.now()));
        await page.waitForSelector('.loading.done');
        await page.waitForFunction(() => getComputedStyle(document.getElementById('loading')).opacity === '0', null, { polling: 50 });
      }
      await load();
      await page.waitForFunction(() => document.getElementById('leaderboard-status').textContent.includes('排行榜已更新'), null, { polling: 50 });
      await page.locator('aside').evaluate(el => { el.scrollTop = 0; });
      const layout = await page.evaluate(() => ({
        first: document.querySelector('aside').firstElementChild.className,
        columns: [...document.querySelectorAll('.leaderboard-table th')].map(el => el.textContent),
        visibleColumns: [...document.querySelectorAll('.leaderboard-table th')].every(el => getComputedStyle(el).display !== 'none'),
        overflow: document.documentElement.scrollWidth > innerWidth,
        tableOverflow: document.querySelector('.leaderboard-table-wrap').scrollWidth > document.querySelector('.leaderboard-table-wrap').clientWidth,
        button: { height: document.getElementById('leaderboard-retry').getBoundingClientRect().height,
          background: getComputedStyle(document.getElementById('leaderboard-retry')).backgroundImage },
        nickname: document.getElementById('player-nickname').value,
        forbiddenCopy: /匿名身份保存在此浏览器|胜 \+3|排名按等级|Edge|未经权威验证|仅真人/.test(document.querySelector('aside').innerText),
      }));
      assert.equal(layout.first, 'leaderboard'); assert.equal(layout.columns.length, 5);
      assert(layout.visibleColumns && !layout.overflow && !layout.tableOverflow && !layout.forbiddenCopy);
      assert.match(layout.nickname, /^QQT玩家·本地[0-9a-f]{3}…[0-9a-f]{3}$/);
      assert(layout.button.height >= (viewport.width < 600 ? 44 : 36));
      await page.screenshot({ path: `${out}/${viewport.width}-menu.png`, fullPage: true });
      await page.locator('aside').screenshot({ path: `${out}/${viewport.width}-panel.png` });
      await page.locator('.profile-settings summary').click();
      await page.locator('#player-nickname').fill('糖堂玩家 & "毛毛"');
      await page.locator('#player-message').fill('准备出发！');
      await page.locator('#leaderboard-profile button').click();
      await load();
      assert.equal(await page.locator('#player-nickname').inputValue(), '糖堂玩家 & "毛毛"');
      // Deterministic presentation checks only: no match submission or fake service rank.
      await page.evaluate(() => {
        const frame = QQT.Sim.prototype.frameStep;
        QQT.Sim.prototype.frameStep = function(...args) { window.appSim = this; return frame.apply(this, args); };
        appFrame(performance.now() + 20);
        appSim.done = true;
        QQTLeaderboard.renderSettlement(document, { status: '结算已提交', settlement: {
          result: 'win', duration_ms: 12000, submitted: true, ranking: null } });
      });
      assert.equal(await page.locator('#settlement-rank').textContent(), '排名待数据库升级');
      await page.screenshot({ path: `${out}/${viewport.width}-settlement.png`, fullPage: true });
      await page.locator('#play-again').click();
      await page.waitForSelector('#settlement', { state: 'hidden' });
      assert.equal(await page.locator('#character').inputValue(), 'maomao');
      assert.deepEqual(errors, []); assert(rpc.includes('qqt_leaderboard'));
      evidence.push({ viewport, mock: false, readOnlyRemote: true, layout, customNamePreserved: true,
        rankingUpgradeHint: true, restart: true, errors });
      await context.close();
    }
    // A separate explicit mock supplies long/malicious strings to stress all five columns.
    const page = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.route('**/rest/v1/rpc/qqt_leaderboard', r => r.fulfill({ status: 200, contentType: 'application/json',
      body: JSON.stringify([{ rank: 1, nickname: '<img onerror=bad>'.repeat(2), victory_message: '<script>bad</script>'.repeat(4),
        player_ip: '2001:*:*:abcd', best_win_duration_ms: 240000 }]) }));
    await page.goto(base); await page.waitForSelector('.loading.done'); await page.waitForSelector('#leaderboard-list tr');
    assert.equal(await page.locator('#leaderboard-list tr script, #leaderboard-list tr img').count(), 0);
    assert.equal(await page.locator('#leaderboard-list td').count(), 5);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth &&
      document.querySelector('.leaderboard-table-wrap').scrollWidth <= document.querySelector('.leaderboard-table-wrap').clientWidth));
    await page.screenshot({ path: `${out}/390-stress-mock.png`, fullPage: true });
    assert.deepEqual(errors, []); evidence.push({ viewport: { width: 390, height: 844 }, mock: true, xssEscaped: true, overflow: false, errors });
    await page.close();
    fs.writeFileSync(`${out}/checks.json`, JSON.stringify({ url: base, noDatabaseWrites: true, evidence }, null, 2) + '\n');
    console.log('Menu: desktop/mobile real RPC reads, local default/persisted custom name, settlement hint, restart and mock XSS layout passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
