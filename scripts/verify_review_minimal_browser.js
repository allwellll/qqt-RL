'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createServer } = require('../web/server');
const P = require('../web/review_protocol');
const A = require('../web/review_advantage');
const catalog = require('../web/review-catalog.json');
const out = process.env.EVIDENCE_DIR || 'runs/review_minimal_browser';
const format = n => n === 0 ? '0' : Math.abs(n) < .001 ? Number(n.toPrecision(5)).toString() : n.toFixed(3);

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}/`;
  const verifier = fs.readFileSync('web/device_auth.js', 'utf8').match(/const VERIFIER = '([a-f0-9]+)'/)[1];
  let browser;
  const results = [];
  try {
    browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox', '--disable-gpu'] });
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const name = viewport.width > 600 ? 'desktop' : 'mobile';
      const context = await browser.newContext({ viewport, isMobile: name === 'mobile', hasTouch: name === 'mobile' });
      const errors = [], blocked = [], badHttp = [];
      // Only real local read requests may leave this browser context.
      await context.route('**/*', route => {
        const r = route.request();
        if (r.url().startsWith(base) && r.method() === 'GET') return route.continue();
        blocked.push({ method: r.method(), path: new URL(r.url()).pathname });
        return route.abort();
      });
      await context.addInitScript(rule => {
        localStorage.setItem('qqt.advanced.authorization', JSON.stringify({ schema: 'qqt.advanced/v1', rule, granted: true }));
        window.reviewProbe = { frames: [], cursor: null };
        let api, frame;
        const proto = CanvasRenderingContext2D.prototype;
        const fillText = proto.fillText, beginPath = proto.beginPath, moveTo = proto.moveTo, lineTo = proto.lineTo, stroke = proto.stroke;
        proto.fillText = function (value, x, y, ...rest) {
          if (frame && this.canvas.id === 'game') frame.labels.push({ value, x, y });
          return fillText.call(this, value, x, y, ...rest);
        };
        proto.beginPath = function () { this.reviewPath = []; return beginPath.call(this); };
        proto.moveTo = function (x, y) { this.reviewPath?.push([x, y]); return moveTo.call(this, x, y); };
        proto.lineTo = function (x, y) { this.reviewPath?.push([x, y]); return lineTo.call(this, x, y); };
        proto.stroke = function (...args) {
          if (this.canvas.id === 'reward-chart' && this.strokeStyle === '#e9f7ef') reviewProbe.cursor = this.reviewPath;
          return stroke.apply(this, args);
        };
        Object.defineProperty(window, 'QQTVisual', { get: () => api, set(v) {
          api = v;
          const create = v.createRenderer;
          v.createRenderer = (...args) => {
            const renderer = create(...args), render = renderer.render;
            renderer.render = (sim, now, motion) => {
              frame = { tick: sim.t, alive: Array.from(sim.alive), advantage: motion.playerAdvantages?.[0], labels: [] };
              const result = render(sim, now, motion);
              reviewProbe.frames.push(frame);
              if (reviewProbe.frames.length > 100) reviewProbe.frames.shift();
              frame = null;
              return result;
            };
            return renderer;
          };
        } });
      }, verifier);
      const page = await context.newPage();
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
      page.on('response', r => { if (r.status() >= 400) badHttp.push({ path: new URL(r.url()).pathname, status: r.status() }); });
      const ready = () => page.waitForFunction(() => document.getElementById('integrity').textContent === 'hash 已验证', null, { timeout: 120000 });
      const jump = tick => page.locator('#seek').evaluate((e, t) => { e.value = t; e.dispatchEvent(new Event('input')); }, tick);
      const row = catalog.replays[0], doc = JSON.parse(fs.readFileSync(path.join('web/review-replays', row.file)));
      const rewardTick = doc.rewards.find(r => Object.values(r[1]).reduce((s, v) => s + v, 0) !== 0)[0];
      await page.goto(`${base}review.html#replay=${row.id}&tick=${rewardTick}`);
      await ready();
      assert(await page.locator('#advanced-settings').isVisible());
      const checkLinked = async (pid = 0) => {
        const tick = Number(await page.locator('#seek').inputValue());
        const delta = doc.rewards.find(r => r[0] === tick)?.[pid + 1] || {};
        assert.equal(await page.locator('#reward-delta').textContent(), format(Object.values(delta).reduce((s, v) => s + v, 0)));
        assert.equal(await page.locator('#exact-time').textContent(), `tick ${tick} · ${P.time(tick)}`);
        const probe = await page.evaluate(() => ({ frame: reviewProbe.frames.at(-1), cursor: reviewProbe.cursor }));
        assert.equal(probe.frame.tick, tick);
        assert.equal(probe.frame.advantage.advantage, A.atTick(doc, tick).advantage);
        assert.equal(probe.frame.advantage.delta, A.atTick(doc, tick).delta);
        assert(Math.abs(probe.cursor[0][0] - tick / doc.actions.length * 560) < 1e-8, 'chart cursor follows the authoritative tick');
        for (const [pid, role] of ['模型 P0', '对手 P1'].entries()) if (probe.frame.alive[pid]) assert(probe.frame.labels.some(l => l.value === role && l.x >= 0 && l.x <= 900 && l.y >= 0 && l.y <= 810), role + ' rendered on canvas');
        assert(probe.frame.labels.some(l => l.value.includes('优势增量')));
        return { tick, reward: await page.locator('#reward-delta').textContent(), cursor: probe.cursor, labels: probe.frame.labels };
      };
      const samples = [await checkLinked()];
      await jump(rewardTick + 1); samples.push(await checkLinked());
      // Native slider pointer drag, followed by source-data and chart verification.
      const range = await page.locator('#seek').boundingBox();
      await page.mouse.move(range.x + range.width * (rewardTick + 1) / doc.actions.length, range.y + range.height / 2);
      await page.mouse.down();
      await page.mouse.move(range.x + range.width * .6, range.y + range.height / 2, { steps: 8 });
      await page.mouse.up();
      assert(Math.abs(Number(await page.locator('#seek').inputValue()) / doc.actions.length - .6) < .04);
      samples.push(await checkLinked());
      await page.locator('#reward-player').selectOption('1'); samples.push(await checkLinked(1));
      await page.locator('#reward-player').selectOption('0');
      await page.locator('#play').click();
      await page.waitForFunction(t => Number(document.getElementById('seek').value) > t, samples[2].tick);
      await page.locator('#play').click();
      assert.equal(await page.locator('#play-state').textContent(), '已暂停');
      samples.push(await checkLinked());
      const frozen = await page.locator('#seek').inputValue();
      await page.waitForTimeout(250);
      assert.equal(await page.locator('#seek').inputValue(), frozen);
      const chart = await page.locator('#reward-chart').boundingBox();
      await page.mouse.click(chart.x + chart.width * .25, chart.y + chart.height / 2);
      assert(Math.abs(Number(await page.locator('#seek').inputValue()) / doc.actions.length - .25) < .01);
      samples.push(await checkLinked());
      await jump(rewardTick);
      await page.reload(); await ready(); assert.equal(Number(await page.locator('#seek').inputValue()), rewardTick);
      const absent = ['#episode-warning', '.return-warning', '#state-details', '#components', '.timeline-layout', '.display-controls', '.dagger-training', '.critic-comparison', '#policy-decision', '#model-summary', '#annotations', '#reward-total'];
      for (const selector of absent) assert(await page.locator(selector).isHidden(), selector + ' removed from viewing layout');
      assert(!await page.locator('.diagnostic-warning:visible').count());
      assert(!(await page.locator('body').innerText()).includes('当前状态'));
      const layout = await page.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth, height: innerHeight, board: document.querySelector('.board').getBoundingClientRect().toJSON(), inspector: document.querySelector('.inspector').getBoundingClientRect().toJSON() }));
      assert(layout.scrollWidth <= viewport.width);
      if (name === 'desktop') assert(layout.inspector.x >= layout.board.right && layout.board.bottom < viewport.height);
      else assert(layout.inspector.y >= layout.board.bottom);
      await page.evaluate(() => scrollTo(0, 0));
      await page.screenshot({ path: path.join(out, `${name}.png`) });
      if (name === 'mobile') await page.screenshot({ path: path.join(out, 'mobile-full.png'), fullPage: true });
      await page.locator('.library summary').click();
      assert.equal(await page.locator('.replay-row').count(), catalog.replays.length + (catalog.evaluation?.replays.length || 0));
      const next = catalog.replays[1];
      await page.locator(`.replay-row[data-replay="${next.id}"]`).click(); await ready();
      assert((await page.locator('#selected-replay').textContent()).includes(String(next.seed)));
      assert.equal(await page.locator('.library').getAttribute('open'), null);
      // A rejected real historical deep link must show no stale reward or curve.
      await page.evaluate(() => { location.hash = 'replay=h8-v36-g32-carry_home-2026101785&tick=107'; });
      await page.waitForFunction(() => document.getElementById('integrity').textContent === '验证失败');
      assert.equal(await page.locator('#reward-delta').textContent(), '未记录');
      assert(await page.locator('#reward-chart').isHidden());
      assert((await page.locator('#reward-empty').innerText()).includes('无数据'));
      await page.evaluate(({ id, tick }) => { location.hash = `replay=${id}&tick=${tick}`; }, { id: row.id, tick: rewardTick });
      await ready(); samples.push(await checkLinked());
      assert(await page.locator('#reward-empty').isHidden());
      assert.deepEqual(errors, []); assert.deepEqual(badHttp, []); assert.deepEqual(blocked, []);
      results.push({ viewport, replay: row.id, sha256: row.sha256, samples, layout, native_drag: true, playback_pause: true, chart_seek: true, selector: true, reload: true, rejected_replay_no_stale_reward: true, errors, badHttp, blocked_requests: blocked });
      await context.close();
      console.log(`Review minimal ${viewport.width}x${viewport.height} GREEN: real replay, drag/play/pause, labels, reward/chart linkage, selector, reload, honest unavailable state`);
    }
    fs.writeFileSync(path.join(out, 'checks.json'), JSON.stringify({ passed: true, browser: browser.version(), results }, null, 2) + '\n');
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
