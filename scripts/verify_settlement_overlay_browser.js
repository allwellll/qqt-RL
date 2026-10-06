'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const base = process.env.WEB_URL || 'http://127.0.0.1:8080/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/settlement_overlay_20261006/browser');

async function main() {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] });
  const evidence = [];
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport, isMobile: viewport.width < 600, hasTouch: viewport.width < 600 });
      const errors = [], submissions = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
      await page.route('**/rest/v1/rpc/qqt_leaderboard', route => route.fulfill({
        status: 200, contentType: 'application/json', body: '[]',
      }));
      const submit = route => {
        const payload = route.request().postDataJSON().p_payload;
        submissions.push(payload);
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
          level: 1, points: submissions.length * 3, wins: submissions.filter(row => row.result === 'win').length,
          games: submissions.length,
          match_rank: {
            rank: 2, total: 5, percentile: 60, comparison: 'player-best-v1', result: payload.result,
            mode: payload.mode, map_id: payload.map_id, difficulty: payload.difficulty,
            opponent: payload.opponent, duration_ms: payload.game_duration_ms,
          },
        }) });
      };
      await page.route('**/rest/v1/rpc/qqt_submit_result', submit);
      await page.route('**/functions/v1/submit-result', submit);
      await page.addInitScript(() => {
        window.clockOffset = 0;
        const realNow = Date.now;
        Date.now = () => realNow() + window.clockOffset;
        window.setInterval = callback => { window.appTick = callback; return 1; };
        window.requestAnimationFrame = callback => { window.appFrame = callback; return 1; };
      });
      await page.goto(base);
      await page.waitForFunction(() => window.appFrame, null, { polling: 50, timeout: 60000 });
      await page.evaluate(() => {
        const reset = QQT.Sim.prototype.reset;
        const frame = QQT.Sim.prototype.frameStep;
        window.resetCount = 0;
        QQT.Sim.prototype.reset = function (...args) { window.resetCount++; return reset.apply(this, args); };
        QQT.Sim.prototype.frameStep = function (...args) { window.appSim = this; return frame.apply(this, args); };
        appFrame(performance.now());
      });
      await page.waitForSelector('.loading.done', { timeout: 60000 });
      await page.waitForFunction(() => getComputedStyle(document.getElementById('loading')).opacity === '0', null, { polling: 50 });
      // Ignore the initial map setup; count only restarts after the real page is ready.
      await page.evaluate(() => { window.resetCount = 0; });

      const geometry = await page.evaluate(() => {
        const stage = document.querySelector('.stage').getBoundingClientRect();
        const overlay = document.getElementById('settlement').getBoundingClientRect();
        const canvas = document.getElementById('game');
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        const colors = new Set();
        for (let i = 0; i < pixels.length; i += 400) colors.add(`${pixels[i]},${pixels[i + 1]},${pixels[i + 2]},${pixels[i + 3]}`);
        const style = getComputedStyle(document.getElementById('settlement'));
        return {
          stage: { x: stage.x, y: stage.y, width: stage.width, height: stage.height },
          overlay: { x: overlay.x, y: overlay.y, width: overlay.width, height: overlay.height },
          canvas: { width: canvas.getBoundingClientRect().width, height: canvas.getBoundingClientRect().height },
          canvasColors: colors.size, position: style.position, backgroundColor: style.backgroundColor,
          border: style.border, boxShadow: style.boxShadow,
        };
      });
      assert(geometry.canvasColors > 100, '终局画布必须有真实非空像素');
      assert.equal(geometry.position, 'absolute');
      assert(Math.abs(geometry.overlay.x - geometry.stage.x) < 1 && Math.abs(geometry.overlay.y - geometry.stage.y) < 1);
      assert.equal(geometry.backgroundColor, 'rgba(0, 0, 0, 0)');
      assert.match(geometry.border, /^0px none /);
      assert.equal(geometry.boxShadow, 'none');

      const outcomes = [];
      for (const [result, title] of [['win', '胜利'], ['loss', '失败'], ['draw', '平局']]) {
        await page.evaluate(async result => {
          window.appSim.t = 119; window.appSim.maxSteps = 120;
          window.appSim.bunStored = result === 'win' ? [[2, 1], [0, 1]] : result === 'loss' ? [[1, 0], [1, 2]] : [[1, 0], [0, 1]];
          window.clockOffset += 12000;
          await window.appTick();
          window.appFrame(performance.now() + 50);
        }, result);
        await page.waitForFunction(() => document.getElementById('settlement').hidden === false, null, { polling: 50 });
        await page.waitForFunction(() => {
          if (window.appFrame) window.appFrame(performance.now());
          return document.getElementById('settlement-rank').textContent.includes('第 2 名');
        }, null, { polling: 50 });
        const overlayGeometry = await page.evaluate(() => {
          const stage = document.querySelector('.stage').getBoundingClientRect();
          const overlay = document.getElementById('settlement').getBoundingClientRect();
          const canvas = document.getElementById('game').getBoundingClientRect();
          return { stage: { x: stage.x, y: stage.y, width: stage.width, height: stage.height },
            overlay: { x: overlay.x, y: overlay.y, width: overlay.width, height: overlay.height },
            canvas: { x: canvas.x, y: canvas.y, width: canvas.width, height: canvas.height } };
        });
        assert(overlayGeometry.overlay.width >= overlayGeometry.canvas.width && overlayGeometry.overlay.height >= overlayGeometry.canvas.height,
          'settlement overlay must cover the terminal canvas');
        assert(overlayGeometry.overlay.x <= overlayGeometry.canvas.x && overlayGeometry.overlay.y <= overlayGeometry.canvas.y);
        assert.equal(await page.locator('#settlement-title').textContent(), title);
        assert.equal(await page.locator('#settlement-time').textContent(), '本局耗时 12.0秒');
        assert.match(await page.locator('#settlement-rank').textContent(), /第 2 名 \/ 5 位玩家 · 超过 60\.00% 玩家/);
        assert.match(await page.locator('#play-again').textContent(), /再来一局.*R/);
        assert.equal(await page.locator('#settlement').evaluate(el => el.parentElement.className), 'stage');

        const beforeInput = await page.evaluate(() => window.resetCount);
        await page.locator('.profile-settings').evaluate(el => { el.open = true; });
        await page.locator('#player-nickname').focus();
        await page.keyboard.press('r');
        assert.equal(await page.evaluate(() => window.resetCount), beforeInput, '输入框聚焦时 R 不得重开');
        await page.locator('#player-nickname').evaluate(el => el.blur());
        await page.screenshot({ path: path.join(out, `${viewport.width}-${result}.png`), fullPage: true });

        const xss = await page.evaluate(() => {
          QQTLeaderboard.renderSettlement(document, { status: '<img src=x onerror=alert(1)>', settlement: {
            result: 'win', duration_ms: 12000, submitted: true,
            ranking: { rank: 2, total: 5, percentile: 60 },
          } });
          const status = document.getElementById('settlement-status');
          return { text: status.textContent, images: status.querySelectorAll('img').length };
        });
        assert.equal(xss.text, '<img src=x onerror=alert(1)>');
        assert.equal(xss.images, 0);
        await page.evaluate(() => QQTLeaderboard.renderSettlement(document, window.__lastSettlementState || { status: '', settlement: null }));
        await page.evaluate(() => document.getElementById('settlement').hidden = false);
        const beforeRestart = await page.evaluate(() => window.resetCount);
        await page.evaluate(() => {
          document.getElementById('play-again').click();
        });
        await page.evaluate(() => window.appFrame(performance.now() + 80));
        assert.equal(await page.evaluate(() => window.resetCount), beforeRestart + 2, '结算按钮安全重开一次');
        assert.equal(await page.locator('#settlement').isVisible(), false);
        assert.equal(await page.evaluate(() => window.appSim.done), false);
        outcomes.push({ result, title, rank: '2/5/60.00%', inputFocusBlocksR: true, oneRestart: true, xssTextContent: true });
      }
      const beforeButton = await page.evaluate(() => window.resetCount);
      await page.locator('#restart').click();
      await page.evaluate(() => window.appFrame(performance.now() + 100));
      assert.equal(await page.evaluate(() => window.resetCount), beforeButton + 2, '#restart 保持安全重开');
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), '页面不得横向溢出');
      assert.deepStrictEqual(errors, []);
      evidence.push({ viewport, geometry, outcomes, submissions: submissions.length, errors });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync(path.join(out, 'checks.json'), `${JSON.stringify({ url: base, evidence }, null, 2)}\n`);
  console.log('desktop/mobile settlement overlay, outcomes, rank, restart gate, XSS and canvas checks passed');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
