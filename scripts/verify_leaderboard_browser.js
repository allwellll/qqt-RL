'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.WEB_URL || 'http://127.0.0.1:8080/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/leaderboard_20261005/browser');
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] });
  const evidence = [];
  try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    const page = await browser.newPage({ viewport, isMobile: viewport.width < 600, hasTouch: viewport.width < 600 });
    const errors = [], rpc = [], received = new Map(); let unavailable = false;
    page.on('pageerror', e => errors.push(e.message));
    await page.route('**/rest/v1/rpc/qqt_leaderboard', route => { rpc.push('read'); return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ rank: 1, player_ip: '123.*.*.45', nickname: '<unsafe>', victory_message: '<script>bad</script>', best_win_duration_ms: 4200, level: 4, progress: 2, level_reached_ms: 12000, last_level_up_ms: 3000, wins: 8, games: 10, win_rate: .8 }]) }); });
    const submitRoute = async route => {
      const p = route.request().postDataJSON().p_payload;
      rpc.push({ write: p.client_match_id, result: p.result, mode: p.mode });
      if (unavailable) return route.fulfill({ status: 503, contentType: 'application/json', body: '{}' });
      const old = received.get(p.client_match_id);
      if (old) assert.deepEqual(p, old);
      received.set(p.client_match_id, p);
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ level: 1, points: 3, wins: 1, games: received.size }) });
    };
    await page.route('**/rest/v1/rpc/qqt_submit_result', submitRoute);
    await page.route('**/functions/v1/submit-result', submitRoute);
    await page.addInitScript(() => {
      window.clockOffset = 0; const realNow = Date.now;
      Date.now = () => realNow() + window.clockOffset;
      window.setInterval = cb => { window.appTick = cb; return 1; };
      window.realRAF = window.requestAnimationFrame;
      window.requestAnimationFrame = cb => { window.appFrame = cb; return 1; };
    });
    await page.goto(base); await page.waitForSelector('#leaderboard-list tr');
    assert.strictEqual(await page.locator('#leaderboard-list tr').count(), 1);
    assert.strictEqual(await page.locator('#leaderboard-list tr td').nth(1).textContent(), '<unsafe>');
    assert.strictEqual(await page.locator('#leaderboard-list tr script').count(), 0);
    await page.locator('#player-nickname').fill('网页玩家'); await page.locator('#player-message').fill('胜利宣言');
    await page.locator('#leaderboard-profile').evaluate(form => form.requestSubmit());
    assert((await page.locator('#leaderboard-status').textContent()).includes('昵称和宣言'));
    const id = await page.evaluate(() => JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).player_id);
    await page.waitForFunction(() => window.appFrame, null, { polling: 50 });
    async function settle(result) {
      await page.evaluate(async result => {
        const step = QQT.Sim.prototype.frameStep;
        QQT.Sim.prototype.frameStep = function(...args) { window.appSim = this; return step.apply(this, args); };
        appFrame(performance.now());
        appSim.t = 119; appSim.maxSteps = 120;
        // Trigger the real timeout settlement branch on the full 806 map.
        appSim.bunStored = result === 'win' ? [[2, 1], [0, 1]] : result === 'loss' ? [[1, 0], [1, 2]] : [[1, 0], [0, 1]];
        window.clockOffset += 12000;
        await appTick(); appFrame(performance.now()); appFrame(performance.now() + 20);
      }, result);
    }
    await settle('win'); await page.waitForFunction(() => document.getElementById('leaderboard-progress').textContent.includes('1/1'), null, { polling: 50 });
    await page.evaluate(() => appFrame(performance.now() + 30));
    assert.equal(await page.locator('#settlement-title').textContent(), '胜利');
    assert.equal(await page.locator('#settlement-rank').textContent(), '排名待数据库升级');
    await page.screenshot({ path: `${out}/${viewport.width}-win.png`, fullPage: true });
    assert.equal(received.size, 1); const payload = [...received.values()][0];
    assert.equal(payload.nickname, '网页玩家'); assert.equal(payload.victory_message, '胜利宣言');
    assert.equal(payload.game_duration_ms, 12000); assert.equal(payload.result, 'win');
    assert.equal(Object.keys(payload).length, 16);
    const idempotentWrites = rpc.filter(x => x.write).length; assert.equal(idempotentWrites, 1);
    await page.reload(); await page.waitForSelector('#leaderboard-list tr');
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).player_id), id);
    assert.equal(await page.locator('#player-nickname').inputValue(), '网页玩家');
    unavailable = true; await page.waitForFunction(() => window.appFrame, null, { polling: 50 });
    await settle('loss'); await page.waitForFunction(() => document.getElementById('leaderboard-status').textContent.includes('待提交'), null, { polling: 50 });
    await page.evaluate(() => appFrame(performance.now() + 30));
    assert.equal(await page.locator('#settlement-title').textContent(), '失败');
    assert.equal(await page.locator('#settlement-rank').textContent(), '排名等待结算提交');
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).queue.length), 1);
    await page.locator('#play-again').evaluate(button => { button.click(); button.click(); });
    await page.evaluate(() => appFrame(performance.now()));
    assert.equal(await page.evaluate(() => appSim.done), false, 'RPC failure does not block restart');
    unavailable = false; await page.locator('#leaderboard-retry').click();
    await page.waitForFunction(() => JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).queue.length === 0, null, { polling: 50 });
    assert.equal(received.size, 2);
    await settle('draw'); await page.waitForFunction(() => JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).queue.length === 0, null, { polling: 50 });
    await page.evaluate(() => appFrame(performance.now() + 30));
    assert.equal(await page.locator('#settlement-title').textContent(), '平局');
    const drawWrites = rpc.filter(x => x.write).length;
    await page.locator('#play-again').evaluate(button => { button.click(); button.click(); });
    await page.evaluate(() => appFrame(performance.now() + 60));
    assert.equal(rpc.filter(x => x.write).length, drawWrites, 'double restart cannot resubmit prior settlement');
    await page.locator('#player-nickname').fill('<img onerror=alert(1)>');
    await page.locator('#leaderboard-profile').evaluate(form => form.requestSubmit());
    assert((await page.locator('#leaderboard-status').textContent()).includes('尖括号'));
    // Profile typing must not move, bomb, or restart the current game.
    const seed = await page.evaluate(() => appSim.seed);
    await page.locator('#player-nickname').fill(''); await page.locator('#player-nickname').type('R W 123');
    assert.equal(await page.evaluate(() => appSim.seed), seed);
    await page.evaluate(() => { window.requestAnimationFrame = realRAF; });
    await page.screenshot({ path: `${out}/${viewport.width}.png`, fullPage: true });
    assert.deepStrictEqual(errors, []); assert((await page.locator('body').evaluate(() => document.documentElement.scrollWidth <= innerWidth)));
    assert(rpc.includes('read')); await page.close();
    evidence.push({ viewport, mock: true, simulatedTimeoutOnMap806: true, received: received.size,
      writes: rpc.filter(x => x.write).length, identityStable: true, escaped: true, errors });
    console.log(`${viewport.width}: mock RPC, real app settlement hook, refresh identity, offline retry, escaping and layout passed`);
  }
  // Formal Pages verification without route mocks. The two pages share one
  // isolated browser context, so only desktop submits one temporary match.
  const liveContext = await browser.newContext();
  const liveEvidence = [];
  let cleanupPlayerId = null;
  try {
    const desktop = await liveContext.newPage({ viewport: { width: 1440, height: 1000 }, isMobile: false });
    const desktopErrors = [], desktopNetwork = [], desktopRpc = [];
    desktop.on('pageerror', error => desktopErrors.push(error.message));
    desktop.on('response', response => {
      if (response.url().includes('/rest/v1/rpc/')) desktopRpc.push({ url: response.url(), status: response.status() });
      if (response.status() >= 400) desktopNetwork.push({ url: response.url(), status: response.status() });
    });
    await desktop.addInitScript(() => {
      window.clockOffset = 0; const realNow = Date.now;
      Date.now = () => realNow() + window.clockOffset;
      window.setInterval = cb => { window.appTick = cb; return 1; };
      window.realRAF = window.requestAnimationFrame;
      window.requestAnimationFrame = cb => { window.appFrame = cb; return 1; };
    });
    console.log('live desktop: goto');
    await desktop.goto(base, { waitUntil: 'domcontentloaded' });
    console.log('live desktop: loaded');
    await desktop.waitForSelector('#leaderboard-list tr', { timeout: 20000 });
    console.log('live desktop: leaderboard loaded');
    await desktop.locator('#player-nickname').fill('Hermes远端验收');
    await desktop.locator('#player-message').fill('临时测试，验收后清理');
    await desktop.locator('#leaderboard-profile').evaluate(form => form.requestSubmit());
    console.log('live desktop: profile saved');
    await desktop.waitForFunction(() => window.appFrame && window.appTick, null, { polling: 50 });
    console.log('live desktop: hooks ready');
    await desktop.evaluate(() => appFrame(performance.now()));
    await desktop.evaluate(async () => {
      const step = QQT.Sim.prototype.frameStep;
      QQT.Sim.prototype.frameStep = function(...args) { window.appSim = this; return step.apply(this, args); };
      appFrame(performance.now());
      appSim.t = 119; appSim.maxSteps = 120;
      appSim.bunStored = [[2, 1], [0, 1]];
      window.clockOffset += 12000;
      await appTick(); appFrame(performance.now()); appFrame(performance.now() + 20);
    });
    console.log('live desktop: settlement triggered');
    await desktop.waitForFunction(() => document.getElementById('leaderboard-progress').textContent.includes('1/1'), null, { polling: 100, timeout: 20000 });
    await desktop.evaluate(() => appFrame(performance.now() + 30));
    assert.equal(await desktop.locator('#settlement-title').textContent(), '胜利');
    assert.equal(await desktop.locator('#settlement-rank').textContent(), '排名待数据库升级', 'old remote schema must never produce a fabricated rank');
    console.log('live desktop: settlement submitted');
    cleanupPlayerId = await desktop.evaluate(() => JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).player_id);
    const canvas = await desktop.locator('#game').evaluate(c => {
      const pixels = c.getContext('2d').getImageData(0, 0, c.width, c.height).data, colors = new Set();
      for (let i = 0; i < pixels.length; i += 400) colors.add(`${pixels[i]},${pixels[i + 1]},${pixels[i + 2]}`);
      return colors.size;
    });
    assert(desktopRpc.some(row => row.url.endsWith('/qqt_leaderboard') && row.status === 200));
    assert(desktopRpc.some(row => row.url.endsWith('/qqt_submit_result') && row.status === 200));
    assert.deepStrictEqual(desktopErrors, []); assert(canvas > 100);
    assert((await desktop.locator('#leaderboard-list').textContent()).includes('Hermes远端验收'));
    assert(await desktop.locator('body').evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await desktop.screenshot({ path: `${out}/1440-live-real.png`, fullPage: true });
    liveEvidence.push({ viewport: { width: 1440, height: 1000 }, mock: false, submitted: true,
      playerId: cleanupPlayerId, canvasColors: canvas, rpc: desktopRpc, networkErrors: desktopNetwork, errors: desktopErrors });
    await desktop.close();

    const mobile = await liveContext.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    const mobileErrors = [], mobileNetwork = [], mobileRpc = [];
    mobile.on('pageerror', error => mobileErrors.push(error.message));
    mobile.on('response', response => {
      if (response.url().includes('/rest/v1/rpc/')) mobileRpc.push({ url: response.url(), status: response.status() });
      if (response.status() >= 400) mobileNetwork.push({ url: response.url(), status: response.status() });
    });
    console.log('live mobile: goto');
    await mobile.goto(base); await mobile.waitForSelector('.loading.done');
    console.log('live mobile: loaded');
    await mobile.waitForSelector('#leaderboard-list tr', { timeout: 20000 });
    await mobile.waitForFunction(() => document.getElementById('leaderboard-list').textContent.includes('Hermes远端验收'), null, { timeout: 20000 });
    assert.equal(await mobile.locator('#player-nickname').inputValue(), 'Hermes远端验收');
    const mobileCanvas = await mobile.locator('#game').evaluate(c => {
      const pixels = c.getContext('2d').getImageData(0, 0, c.width, c.height).data, colors = new Set();
      for (let i = 0; i < pixels.length; i += 400) colors.add(`${pixels[i]},${pixels[i + 1]},${pixels[i + 2]}`);
      return colors.size;
    });
    assert.deepStrictEqual(mobileErrors, []); assert(mobileCanvas > 100);
    assert(await mobile.locator('body').evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert(mobileRpc.some(row => row.url.endsWith('/qqt_leaderboard') && row.status === 200));
    await mobile.screenshot({ path: `${out}/390-live-real.png`, fullPage: true });
    liveEvidence.push({ viewport: { width: 390, height: 844 }, mock: false, submitted: false,
      playerId: cleanupPlayerId, canvasColors: mobileCanvas, rpc: mobileRpc, networkErrors: mobileNetwork, errors: mobileErrors });
    await mobile.close();
  } finally { await liveContext.close(); }
  const cleanupSql = cleanupPlayerId
    ? `delete from qqt_private.players where player_id = '${cleanupPlayerId}'::uuid;`
    : null;
  fs.writeFileSync(`${out}/checks.json`, JSON.stringify({ url: base, remoteDatabaseE2E: true,
    cleanupPerformed: false, cleanupSql, evidence: evidence.concat(liveEvidence) }, null, 2) + '\n');
  fs.writeFileSync(`${out}/cleanup.sql`, `${cleanupSql || '-- desktop live test did not create a player'}\n`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
