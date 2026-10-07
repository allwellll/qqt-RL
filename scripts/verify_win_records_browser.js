'use strict';
const assert = require('assert'), fs = require('fs'), path = require('path');
const { randomUUID } = require('crypto');
const { PGlite } = require('@electric-sql/pglite');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createServer } = require('../web/server');
const out = path.resolve('runs/qqt_win_records_20261008/browser-local');
const files = ['supabase/migrations/20261005140000_leaderboard.sql',
  'supabase/migrations/20261005173000_leaderboard_ip_and_best_win.sql',
  'supabase/manual_upgrade_20261006.sql', 'supabase/manual_profile_update_20261007.sql'];
const upgrade = 'supabase/manual_win_records_20261008.sql';
const fixture = (extra = {}) => ({ player_id: randomUUID(), player_secret: 'a'.repeat(64), client_match_id: randomUUID(),
  nickname: 'QQT玩家', victory_message: '历史当前感言', result: 'win', game_duration_ms: 4000, wall_duration_ms: 4000,
  client_total_ms: 4000, opponent: 'bun.coop_hunter', difficulty: 'hard', seed: 7, mode: '1v2', map_id: 'training806',
  completed_at: new Date().toISOString(), client_version: 'dev', ...extra });
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const server = createServer(); await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}/`, evidence = [];
  let browser;
  try {
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const db = new PGlite();
      const context = await browser.newContext({ viewport });
      try {
        await db.exec('create role anon; create role authenticated; create role service_role;');
        for (const file of files) await db.exec(fs.readFileSync(file, 'utf8'));
        const call = async (name, args = [], role = 'anon') => {
          await db.exec(`set role ${role}`);
          try { return (await db.query(`select public.${name}(${args.map((_, i) => `$${i + 1}`).join(',')}) result`, args)).rows[0].result; }
          finally { await db.exec('reset role'); }
        };
        // Seed only the in-memory DB, including one genuine legacy win before migration.
        await call('qqt_submit_result', [JSON.stringify(fixture({ nickname: '历史玩家' }))]);
        await db.exec(fs.readFileSync(upgrade, 'utf8'));
        await call('qqt_submit_result', [JSON.stringify(fixture({ nickname: 'QQT玩家', victory_message: '同昵称其他身份' }))]);
        const page = await context.newPage(), errors = [], unexpected = [], writes = [], profiles = [];
        let failProfile = false, deferRead = false, oldRead;
        // Serialize local SQL because SET ROLE is connection scoped. Every external request is intercepted.
        let sqlQueue = Promise.resolve();
        await context.route('**/*', async route => {
          const req = route.request(), url = req.url();
          if (url.startsWith(base)) return route.continue();
          const name = new URL(url).pathname.split('/').pop();
          if (!['qqt_get_profile', 'qqt_my_win_leaderboard', 'qqt_update_profile', 'qqt_submit_result', 'submit-result'].includes(name)) {
            unexpected.push(name); return route.abort();
          }
          const body = req.postDataJSON();
          sqlQueue = sqlQueue.then(async () => {
            try {
              let result;
              if (name === 'qqt_get_profile' || name === 'qqt_my_win_leaderboard')
                result = await call(name, [body.p_player_id, body.p_player_secret]);
              else if (name === 'qqt_update_profile') {
                profiles.push(body);
                if (failProfile) return route.fulfill({ status: 422, json: { error: 'controlled profile failure' } });
                result = await call(name, [body.p_player_id, body.p_player_secret, body.p_client_match_id, body.p_nickname, body.p_victory_message]);
              } else {
                const p = body.p_payload; writes.push(p);
                await db.query("update qqt_private.match_results set received_at=clock_timestamp()-interval '11 seconds' where player_id=$1", [p.player_id]);
                result = await call('qqt_submit_result', [JSON.stringify(p)]);
                await call('qqt_record_match_ip', [p.player_id, p.client_match_id, '123.45.67.89'], 'service_role');
              }
              if (name === 'qqt_my_win_leaderboard' && deferRead) { deferRead = false; oldRead = { route, result }; return; }
              await route.fulfill({ json: result });
            } catch (e) { errors.push(`local SQL: ${e.message}`); await route.fulfill({ status: 500, json: { error: 'local SQL failure' } }); }
          });
          await sqlQueue;
        });
        page.on('pageerror', e => errors.push(e.message));
        await page.addInitScript(() => {
          window.clockOffset = 0; const now = Date.now; Date.now = () => now() + clockOffset;
          window.setInterval = cb => { window.appTick = cb; return 1; };
          window.requestAnimationFrame = cb => { window.appFrame = cb; return 1; };
        });
        const wait = fn => page.waitForFunction(fn, null, { polling: 50, timeout: 60000 });
        const render = () => page.evaluate(() => appFrame(performance.now()));
        const load = async () => {
          await wait(() => window.appFrame);
          await page.evaluate(() => {
            const reset = QQT.Sim.prototype.reset;
            QQT.Sim.prototype.reset = function (...args) { window.appSim = this; return reset.apply(this, args); };
            const step = QQT.Sim.prototype.frameStep;
            QQT.Sim.prototype.frameStep = function (...args) { window.appSim = this; return step.apply(this, args); };
            appFrame(performance.now());
          });
          await wait(() => getComputedStyle(document.getElementById('loading')).opacity === '0');
          await wait(() => document.getElementById('leaderboard-status').textContent === '排行榜已更新');
        };
        const finish = async (result, steps) => {
          await page.evaluate(async ({ result, steps }) => {
            appSim.t = steps - 1; appSim.maxSteps = steps; appSim.bunStored = [[0, 0], [0, 0]];
            if (result !== 'draw') appSim.bunStored[result === 'win' ? appSim.team[0] : 1 - appSim.team[0]] = [1, 1];
            clockOffset += 12000; await appTick(); appFrame(performance.now());
          }, { result, steps });
          await wait(() => !document.getElementById('settlement').hidden);
        };
        const submit = async message => {
          await page.locator('#player-message').fill(message); await page.locator('#settlement-submit').click();
          await wait(() => { appFrame(performance.now()); return document.getElementById('settlement').hidden; });
        };
        const readRows = () => page.locator('#leaderboard-list tr:not(.leaderboard-ellipsis)').evaluateAll(rows => rows.map(r => ({
          rank: Number(r.querySelector('.rank').textContent), nickname: r.querySelector('.player').childNodes[0].textContent,
          message: r.querySelector('.message').textContent, mine: r.classList.contains('is-mine'), latest: r.classList.contains('is-latest') })));
        await page.goto(base); await load(); await finish('win', 120);
        const initial = await page.locator('#player-nickname').inputValue(); assert.match(initial, /^QQT玩家[A-HJKMNP-Z2-9]{3}$/);
        await page.reload(); await load(); assert.equal(await page.locator('#player-nickname').inputValue(), initial);
        await page.locator('#player-nickname').fill('QQT玩家'); await page.reload(); await load();
        assert.equal(await page.locator('#player-nickname').inputValue(), 'QQT玩家', 'deleted suffix stays deleted');
        await page.locator('#player-nickname').fill(''); await page.reload(); await load();
        assert.equal(await page.locator('#player-nickname').inputValue(), '', 'blank draft survives reload');
        await page.locator('#settlement-submit').click(); assert.equal(writes.length, 0);
        await page.locator('#player-nickname').fill('QQT玩家'); await submit('第一局感言');
        assert.equal(writes[0].nickname, 'QQT玩家', 'actual input saved without suffix');
        await page.reload(); await load();
        assert.equal((await readRows()).filter(r => r.mine).length, 1);
        deferRead = true; await page.locator('#leaderboard-retry').click();
        await finish('win', 100); failProfile = true;
        await page.locator('#player-message').fill('第二局感言'); await page.locator('#settlement-submit').click();
        await wait(() => { appFrame(performance.now()); return document.getElementById('settlement-status').textContent.includes('感言提交失败'); });
        const count = writes.length; assert(await page.evaluate(() => appSim.done));
        failProfile = false; await page.locator('#settlement-submit').click();
        await wait(() => { appFrame(performance.now()); return document.getElementById('settlement').hidden; });
        assert.equal(writes.length, count, 'only profile is retried');
        assert(oldRead); await oldRead.route.fulfill({ json: oldRead.result }); await page.waitForTimeout(100);
        assert.equal((await readRows()).filter(r => r.mine).length, 2, 'stale read cannot revert new rows');
        await finish('win', 140); await submit('');
        let rows = await readRows(); assert.deepEqual(rows.filter(r => r.mine).map(r => r.message), ['第二局感言', '第一局感言', '第二局感言']);
        assert.equal(rows.filter(r => r.latest).length, 1); assert.equal(rows.find(r => r.latest).rank, 5);
        assert(rows.some(r => r.nickname === 'QQT玩家' && !r.mine));
        assert.equal(await page.locator('.leaderboard-ellipsis').count(), 0);
        assert.equal(rows.find(r => r.nickname === '历史玩家').message, '—');
        assert.equal(await page.locator('#leaderboard-list tr.is-latest .ip').textContent(), '123.*.*.89');
        await render(); await page.screenshot({ path: path.join(out, `${viewport.width}-inside.png`), fullPage: true });
        await page.reload(); await load(); assert.equal((await readRows()).filter(r => r.mine).length, 3);
        for (let i = 0; i < 25; i++) await call('qqt_submit_result', [JSON.stringify(fixture({ nickname: `对手${i}`, game_duration_ms: 1000 + i * 10 }))]);
        await page.locator('#leaderboard-retry').click(); await wait(() => document.querySelectorAll('.leaderboard-ellipsis').length === 1);
        rows = await readRows(); assert.equal(rows.length, 21); assert.equal(rows.at(-1).rank, 30); assert(rows.at(-1).mine && rows.at(-1).latest);
        assert.equal(await page.locator('.leaderboard-ellipsis td').textContent(), '…');
        assert(await page.locator('#leaderboard-list .ip').evaluateAll(cells => cells.every(e => e.scrollWidth <= e.clientWidth)), 'masked IP fits its column');
        await render(); await page.screenshot({ path: path.join(out, `${viewport.width}-outside.png`), fullPage: true });
        const longest=(await call('qqt_win_leaderboard')).rows[0].record_id;
        await db.query('update qqt_private.win_records set nickname=$1,victory_message=$2,ip_display=$3 where record_id=$4',
          ['昵称'.repeat(12),'感言'.repeat(40),'ffff:*:*:ffff',longest]);
        await page.locator('#leaderboard-retry').click();
        await wait(() => document.querySelector('#leaderboard-list .ip').textContent === 'ffff:*:*:ffff');
        assert(await page.locator('#leaderboard-list td').evaluateAll(cells => cells.every(e => e.scrollWidth <= e.clientWidth)), 'max nickname/message/IPv6 fit each cell');
        await page.screenshot({ path: path.join(out, `${viewport.width}-longest.png`), fullPage: true });
        const beforeLoss = [writes.length, profiles.length]; await finish('loss', 120);
        assert.equal(await page.locator('#settlement-status').textContent(), '小伙子，再沉淀沉淀吧');
        assert(!await page.locator('#settlement-form').isVisible());
        await page.locator('#settlement-form').evaluate(e => e.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
        await page.waitForTimeout(100); assert.deepEqual([writes.length, profiles.length], beforeLoss);
        await page.locator('#settlement-close').click(); await render(); await finish('draw', 120); await submit('平局资料');
        assert.equal((await db.query('select count(*)::int n from qqt_private.win_records')).rows[0].n, 30, 'draw/loss never create win records');
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        assert.equal(await page.locator('thead th').count(), 5); assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
        evidence.push({ viewport, mockResultWrites: writes.length, mockProfileWrites: profiles.length, sqlRows: 30,
          editableDefaultAndActualSave: true, threeMatchSnapshots: true, sameNameIdentitySafe: true,
          top20InsideAndRank30Outside: true, reloadHighlightAndStaleProtection: true,
          profileOnlyRetry: true, lossAndDrawSafe: true, longestFieldsFit: true, noOverflow: true, errors, unexpectedRemote: unexpected });
      } finally { await context.close(); await db.close(); }
    }
    fs.writeFileSync(path.join(out, 'checks.json'), JSON.stringify({ browser: await browser.version(),
      kind: 'Real local Chromium/Sim/assets; all remote requests mock via local PGlite, zero production writes', evidence }, null, 2) + '\n');
    console.log('Win-record real Chromium + local SQL desktop/mobile passed');
  } finally { if (browser) await browser.close(); await new Promise(r => server.close(r)); }
})().catch(e => { console.error(e); process.exitCode = 1; });
