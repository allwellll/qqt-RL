'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createServer } = require('../web/server');
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/qqt_round4_20261007/browser-local');
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = process.env.WEB_URL || `http://127.0.0.1:${server.address().port}/`, evidence = [];
  let browser;
  try {
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport, isMobile: viewport.width < 600, hasTouch: viewport.width < 600 });
      const page = await context.newPage(), errors = [], consoleErrors = [], unexpectedRemote = [], writes = [], profiles = [];
      const matches = new Map(), receipts = new Map();
      let registered = false, failResult = true, failProfile = false, failRead = false, upgraded = false;
      let holdProfile = false, releaseProfile, notifyProfileHeld, deferNextRead = false, oldRead;
      let row = { rank: 5, nickname: '旧榜单', victory_message: '旧宣言', best_win_duration_ms: 9000, player_ip: '31.*.*.6' };
      const recordRows = value => ({ leaderboard_contract_version: 1, rows: [{ rank: 1,
        record_id: '00000000-0000-4000-8000-000000000001', nickname: value.nickname,
        victory_message: value.victory_message, game_duration_ms: value.best_win_duration_ms || 9000,
        player_ip: value.player_ip || null, is_mine: false, is_latest: false }] });
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
      // A single deny-by-default router prevents ALL remote requests from escaping,
      // including new write endpoints and Edge fallback. Local assets remain real.
      await context.route('**/*', async route => {
        const request = route.request(), url = request.url();
        if (url.startsWith(base)) return route.continue();
        const endpoint = new URL(url).pathname;
        if (endpoint.endsWith('/qqt_get_profile')) return route.fulfill({ json: registered
          ? { profile_contract_version: 2, registered: true, nickname: row.nickname, victory_message: row.victory_message }
          : { profile_contract_version: 2, registered: false } });
        if (endpoint.endsWith('/qqt_my_win_leaderboard')) {
          if (deferNextRead) { deferNextRead = false; oldRead = route; return; }
          return route.fulfill({ status: failRead ? 503 : 200, json: failRead ? {} : recordRows(row) });
        }
        if (endpoint.endsWith('/qqt_submit_result') || endpoint.endsWith('/submit-result')) {
          const p = request.postDataJSON().p_payload; writes.push(p);
          if (failResult) return route.fulfill({ status: 422, json: { error: 'controlled result failure' } });
          await new Promise(resolve => setTimeout(resolve, 120));
          if (!registered) { registered = true; row = { ...row, nickname: p.nickname, victory_message: p.victory_message }; }
          matches.set(p.client_match_id, p); row = { ...row, rank: 1 };
          return route.fulfill({ json: { level: 2, points: 3, wins: 1, games: matches.size,
            client_match_id: p.client_match_id, match_upgraded: upgraded,
            match_rank: { rank: 1, total: 5, percentile: 80, comparison: 'player-best-v1', result: p.result,
              mode: p.mode, map_id: p.map_id, difficulty: p.difficulty, opponent: p.opponent, duration_ms: p.game_duration_ms } } });
        }
        if (endpoint.endsWith('/qqt_update_profile')) {
          const p = request.postDataJSON(); profiles.push(p);
          assert.equal(p.p_nickname, null); assert(matches.has(p.p_client_match_id));
          if (failProfile) return route.fulfill({ status: 422, json: { error: 'controlled profile failure' } });
          if (holdProfile) await new Promise(resolve => { releaseProfile = resolve; notifyProfileHeld?.(); });
          if (receipts.has(p.p_client_match_id)) assert.deepEqual(receipts.get(p.p_client_match_id), p);
          else { receipts.set(p.p_client_match_id, p); row = { ...row, victory_message: p.p_victory_message || row.victory_message }; }
          return route.fulfill({ json: { saved: true, profile_contract_version: 2, client_match_id: p.p_client_match_id,
            nickname: row.nickname, victory_message: row.victory_message, superseded: false } });
        }
        unexpectedRemote.push({ method: request.method(), endpoint });
        return route.abort();
      });
      await page.addInitScript(() => {
        window.clockOffset = 0; const now = Date.now; Date.now = () => now() + clockOffset;
        window.setInterval = cb => { window.appTick = cb; return 1; };
        window.requestAnimationFrame = cb => { window.appFrame = cb; return 1; };
      });
      const wait = fn => page.waitForFunction(fn, null, { polling: 50, timeout: 20000 });
      const render = () => page.evaluate(() => appFrame(performance.now()));
      const load = async () => {
        await page.waitForFunction(() => window.appFrame, null, { polling: 50, timeout: 60000 });
        await page.evaluate(() => {
          window.restartCount = 0;
          const instances = new WeakSet();
          const reset = QQT.Sim.prototype.reset;
          QQT.Sim.prototype.reset = function (...args) {
            if (!instances.has(this)) { instances.add(this); window.restartCount++; }
            window.appSim = this; return reset.apply(this, args);
          };
          const frame = QQT.Sim.prototype.frameStep;
          QQT.Sim.prototype.frameStep = function (...args) { window.appSim = this; return frame.apply(this, args); };
          appFrame(performance.now());
        });
        await wait(() => getComputedStyle(document.getElementById('loading')).opacity === '0');
        await wait(() => !document.getElementById('leaderboard-status').textContent.includes('无法连接'));
      };
      const action = id => viewport.width < 600 ? page.locator(id).tap() : page.locator(id).click();
      const finish = (result = 'draw') => page.evaluate(async result => {
        appSim.t = 119; appSim.maxSteps = 120;
        // Bun mode decides the timeout from base inventories, not surviving players.
        if (result !== 'draw') {
          const winner = result === 'win' ? appSim.team[0] : 1 - appSim.team[0];
          appSim.bunStored = [[0, 0], [0, 0]];
          appSim.bunStored[winner] = [1, 1];
        }
        clockOffset += 12000; await appTick(); appFrame(performance.now());
      }, result);
      const newGame = async trigger => {
        const before = await page.evaluate(() => { window.beforeRestart = appSim; return restartCount; });
        await trigger();
        await wait(() => { appFrame(performance.now()); return appSim !== beforeRestart && !appSim.done && document.getElementById('settlement').hidden; });
        assert.equal(await page.evaluate(() => restartCount), before + 1, 'exactly one real Sim reset per settlement action');
        await page.locator('#settlement-form').evaluate(e => e.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
        await page.locator('#settlement-close').evaluate(e => e.click());
        await page.waitForTimeout(100); await render();
        assert.equal(await page.evaluate(() => restartCount), before + 1, 'stale hidden actions cannot reset replacement match');
      };
      const nextMatch = async (result = 'draw') => {
        if (await page.evaluate(() => appSim.done)) await newGame(() => action('#settlement-close'));
        await finish(result);
      };
      const status = text => wait(() => { appFrame(performance.now()); return document.getElementById('settlement-status').textContent; }).then(async () => {
        await page.waitForFunction(text => { appFrame(performance.now()); return document.getElementById('settlement-status').textContent.includes(text); }, text, { polling: 50, timeout: 20000 });
      });
      const shot = name => page.screenshot({ path: path.join(out, `${viewport.width}-${name}.png`), fullPage: true });
      const layout = async () => {
        await render();
        const card = await page.locator('#settlement').boundingBox(), refresh = await page.locator('#leaderboard-retry').boundingBox();
        assert(card.x >= 0 && card.x + card.width <= viewport.width + 1);
        assert(card.x + card.width <= refresh.x || card.y + card.height <= refresh.y, 'card must not cover ranking refresh');
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'no horizontal overflow');
        const submit = await page.locator('#settlement-submit').boundingBox();
        assert(submit.y >= card.y && submit.y + submit.height <= card.y + card.height);
      };
      await page.goto(base); await load(); await finish('loss');
      const checkLoss = async name => {
        await render();
        assert.equal(await page.locator('#settlement-status').textContent(), '小伙子，再沉淀沉淀吧');
        assert.equal(await page.locator('#settlement-form').isVisible(), false);
        for (const id of ['#player-nickname', '#player-message', '#settlement-submit']) {
          assert.equal(await page.locator(id).isVisible(), false);
          assert(await page.locator(id).isDisabled());
        }
        assert.equal(await page.locator('#current-nickname').isVisible(), false);
        const counts = [writes.length, profiles.length];
        await page.locator('#settlement-form').evaluate(e => {
          e.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
        });
        await page.locator('#settlement-submit').evaluate(e => { e.click(); e.click(); });
        await page.evaluate(() => document.activeElement.blur());
        await page.keyboard.press('Enter'); await page.waitForTimeout(100);
        assert.deepEqual([writes.length, profiles.length], counts, 'loss cannot authorize a write via submit/Enter');
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        const card = await page.locator('#settlement').boundingBox(), refresh = await page.locator('#leaderboard-retry').boundingBox();
        assert(card.x >= 0 && card.x + card.width <= viewport.width + 1);
        assert(card.x + card.width <= refresh.x || card.y + card.height <= refresh.y);
        const close = await page.locator('#settlement-close').boundingBox();
        assert(close.y >= card.y && close.y + close.height <= card.y + card.height, 'loss X must fit inside the compact card');
        await shot(name);
      };
      await checkLoss('first-loss'); assert.equal(profiles.length, 0); assert.equal(writes.length, 0);
      await page.reload(); await load(); await checkLoss('first-loss-reloaded');
      await nextMatch('win');
      assert.equal(await page.locator('#settlement-title').textContent(), '胜利');
      await wait(() => !document.getElementById('player-nickname').disabled);
      assert(await page.locator('#player-nickname').isVisible());
      assert.match(await page.locator('#player-nickname').inputValue(), /^QQT玩家[A-HJKMNP-Z2-9]{3}$/);
      await page.locator('#player-nickname').fill('');
      assert.equal(await page.locator('#settlement button[type=submit]').count(), 1);
      await action('#settlement-submit'); assert.equal(writes.length, 0);
      assert.equal(await page.locator('label:has(#player-message)').textContent(), '胜利感言');
      assert.equal(await page.locator('thead .message').textContent(), '胜利感言');
      await page.locator('#player-nickname').fill('初次玩家'); await page.locator('#player-message').fill('首次胜利感言');
      await layout(); await shot('first-card');
      await page.reload(); await load(); await wait(() => !document.getElementById('settlement-submit').disabled);
      assert.equal(await page.locator('#player-message').inputValue(), '首次胜利感言');
      await action('#settlement-submit'); await status('提交失败'); assert.equal(writes.length, 1);
      assert(await page.evaluate(() => appSim.done === false && restartCount === 0), 'restored card result failure does not reset the current Sim');
      deferNextRead = true; await action('#leaderboard-retry'); failResult = false; holdProfile = true;
      const profileHeld = new Promise((resolve, reject) => {
        const deadline = setTimeout(() => reject(new Error('profile mock not reached')), 10000);
        notifyProfileHeld = () => { clearTimeout(deadline); resolve(); };
      });
      await page.locator('#settlement-submit').scrollIntoViewIfNeeded();
      const box = await page.locator('#settlement-submit').boundingBox(), x = box.x + box.width / 2, y = box.y + box.height / 2;
      if (viewport.width < 600) {
        const cdp = await context.newCDPSession(page);
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
        await page.waitForTimeout(350); await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }); await cdp.detach();
      } else { await page.mouse.move(x, y); await page.mouse.down(); await page.waitForTimeout(350); await page.mouse.up(); }
      await page.locator('#settlement-submit').evaluate(e => { e.click(); e.click(); });
      await page.keyboard.down('Enter'); await page.keyboard.down('Enter'); await page.keyboard.up('Enter');
      await wait(() => { appFrame(performance.now()); return document.querySelector('#leaderboard-list .rank')?.textContent === '1'; });
      assert.equal(writes.length, 2); assert.deepEqual(writes[0], writes[1]);
      assert(await page.locator('#settlement-close').isDisabled());
      await page.locator('#player-message').fill('下一局感言');
      await page.evaluate(() => { window.before = appSim; document.activeElement.blur(); window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyR' })); appFrame(performance.now()); });
      assert(await page.evaluate(() => appSim === before)); await profileHeld; assert(releaseProfile);
      holdProfile = false; await newGame(() => releaseProfile());
      assert.equal(await page.locator('#player-nickname').isVisible(), false);
      assert(await page.locator('#player-nickname').isDisabled());
      assert.equal(await page.locator('#leaderboard-list .player').textContent(), '初次玩家');
      assert.equal(await page.locator('#leaderboard-list .message').textContent(), '首次胜利感言');
      assert(oldRead); await oldRead.fulfill({ json: recordRows({ nickname: '过期榜单', victory_message: '过期宣言' }) }); oldRead = null;
      await page.waitForTimeout(100); assert.equal(await page.locator('#leaderboard-list .rank').textContent(), '1');
      await shot('first-success');
      await page.reload(); await load();
      await wait(() => document.querySelector('#leaderboard-list .message')?.textContent === '首次胜利感言');
      await finish(); await wait(() => !document.getElementById('settlement-submit').disabled);
      assert.equal(await page.locator('#player-nickname').isVisible(), false);
      assert.equal(await page.locator('#player-nickname').evaluate(e => e.required), false);
      assert.equal(await page.locator('#current-nickname').textContent(), '昵称：初次玩家');
      assert.equal(await page.locator('#player-message').inputValue(), '下一局感言');
      await layout(); await shot('returning-card');
      failProfile = true; await action('#settlement-submit'); await status('战绩已提交，感言提交失败');
      assert(await page.evaluate(() => appSim.done && restartCount === 0), 'profile failure does not restart');
      const count = writes.length, frozen = profiles.at(-1); failProfile = false;
      await page.reload(); await load(); await wait(() => !document.getElementById('settlement-submit').disabled);
      await page.locator('#player-message').fill('后续编辑'); failRead = true;
      await newGame(() => action('#settlement-submit'));
      assert((await page.locator('#leaderboard-status').textContent()).includes('排行榜刷新失败'));
      assert.equal(writes.length, count, 'reload profile retry does not submit result again');
      assert.deepEqual(profiles.at(-1), frozen, 'retry preserves profile intent');
      failRead = false; await action('#leaderboard-retry');
      await wait(() => document.querySelector('#leaderboard-list .message')?.textContent === '下一局感言');
      assert(writes.every(p => p.nickname === '初次玩家')); assert(profiles.every(p => p.p_nickname === null));
      // Ordinary and upgraded matches both update declarations; empty retains server value.
      await nextMatch(); await page.locator('#player-message').fill('');
      await newGame(() => action('#settlement-submit'));
      assert.equal(await page.locator('#leaderboard-list .message').textContent(), '下一局感言');
      upgraded = true; await nextMatch('win'); await page.locator('#player-message').fill('升级局即时感言');
      await newGame(() => action('#settlement-submit'));
      assert.equal(await page.locator('#leaderboard-list .message').textContent(), '升级局即时感言');
      await page.reload(); await load();
      await wait(() => document.querySelector('#leaderboard-list .message')?.textContent === '升级局即时感言');
      await finish(); await layout(); await shot('final-returning-card');
      const beforeLoss = [writes.length, profiles.length];
      await nextMatch('loss'); await checkLoss('returning-loss');
      await page.reload(); await load(); await checkLoss('returning-loss-reloaded');
      assert.deepEqual([writes.length, profiles.length], beforeLoss);
      await newGame(() => action('#settlement-close'));
      assert.deepEqual([writes.length, profiles.length], beforeLoss, 'loss X restarts without writing result or profile');
      await shot('loss-x-new-game');
      assert.equal(await page.locator('#player-nickname').isVisible(), false);
      assert(!/仅升级时更新|新资料将在升级时更新/.test(await page.locator('body').textContent()));
      assert.deepEqual(errors, []); assert.deepEqual(unexpectedRemote, []);
      assert(consoleErrors.every(e => /422|503/.test(e)), JSON.stringify(consoleErrors));
      evidence.push({ viewport, resultAttemptsMocked: writes.length, uniqueResults: matches.size, profileAttemptsMocked: profiles.length,
        firstNicknameRequired: true, returningNicknameHiddenDisabled: true, declarationEveryMatch: true,
        profileOnlyRetryAfterReload: true, immutableProfileRetry: true, oldLeaderboardReadRejected: true,
        serverMockPersistsAcrossReload: true, emptyRetention: true, noOverflowOrRefreshObstruction: true,
        firstAndReturningLossProfilesBlocked: true, lossRestoredAfterReload: true, winAndDrawProfilesPreserved: true,
        newWording: true, submitSuccessRestartsOnce: true, failedSubmitNeverRestarts: true, closeRestartsOnceWithoutWrites: true,
        noUnmockedRemoteRequests: true, errors, expectedConsoleErrors: consoleErrors });
      await context.close();
    }
    fs.writeFileSync(path.join(out, 'checks.json'), JSON.stringify({ browser: await browser.version(), url: base,
      kind: 'Real Chromium, Sim, Canvas and local assets; deny-by-default mocked remote requests; no production writes', evidence }, null, 2) + '\n');
    console.log('Round4 real Chromium desktop/mobile, first/returning profiles, every-match persistence/reload, only-profile retry, races, idempotency and layout passed');
  } finally { if (browser) await browser.close(); await new Promise(resolve => server.close(resolve)); }
})().catch(e => { console.error(e); process.exitCode = 1; });
