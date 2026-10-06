'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const base = process.env.WEB_URL || 'http://127.0.0.1:8080/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/half_tile_20261006/browser');
const launch = { headless: true, executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] };

async function main() {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch(launch);
  const evidence = [];
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport, isMobile: viewport.width < 600, hasTouch: viewport.width < 600 });
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('response', r => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
      await page.addInitScript(() => {
        window.setInterval = cb => { window.appTick = cb; return 1; };
        window.requestAnimationFrame = cb => { window.appFrame = cb; return 1; };
      });
      await page.goto(base);
      await page.waitForFunction(() => window.appFrame, null, { polling: 50, timeout: 60000 });
      await page.evaluate(() => {
        const frame = QQT.Sim.prototype.frameStep;
        QQT.Sim.prototype.frameStep = function (...args) { window.appSim = this; return frame.apply(this, args); };
        appFrame(performance.now());
      });
      await page.waitForSelector('.loading.done', { timeout: 60000 });
      const result = await page.evaluate(() => {
        const ensure = (condition, message) => { if (!condition) throw new Error(message); };
        const N = QQT.H * QQT.W;
        const idle = [QQT.MOVE_IDLE, 0, 0, 1];
        const cell = (r, c) => r * QQT.W + c;
        const cases = [];
        const hit = (sim, cells) => {
          const covered = new Uint8Array(N);
          for (const [r, c] of cells) covered[cell(r, c)] = 1;
          sim._resolveExplosions = () => ({ covered, triggered: new Uint8Array(N), sources: [] });
          sim.step([idle, idle]);
          return sim.trapped[0] > 0;
        };
        const scene = (y, x) => {
          const sim = window.appSim;
          sim.reset('open', { nativeItems: true, nativeTrap: true });
          for (const key of ['wall', 'brick', 'cover', 'bush', 'crate', 'fuse', 'blastLinger', 'horzLinger', 'vertLinger', 'fieldItem']) sim[key].fill(0);
          sim.pos.set([y, x, 12.5, 12.5]); sim.invuln.fill(0); sim.spawnProtection.fill(0);
          return sim;
        };
        for (const [axis, value, sides, center] of [
          ['x', 5.91, [[[5, 5]], [[5, 6]]], null], ['x', 5.90, [[[5, 5]], [[5, 6]]], null],
          ['x', 5.89, [[[5, 5]], [[5, 6]]], 0], ['x', 6.00, [[[5, 5]], [[5, 6]]], null],
          ['x', 6.09, [[[5, 5]], [[5, 6]]], null], ['x', 6.11, [[[5, 5]], [[5, 6]]], 1],
          ['y', 5.91, [[[5, 5]], [[6, 5]]], null], ['y', 5.90, [[[5, 5]], [[6, 5]]], null],
          ['y', 5.89, [[[5, 5]], [[6, 5]]], 0], ['y', 6.00, [[[5, 5]], [[6, 5]]], null],
          ['y', 6.09, [[[5, 5]], [[6, 5]]], null], ['y', 6.11, [[[5, 5]], [[6, 5]]], 1],
        ]) {
          for (let i = 0; i < sides.length; i++) {
            const sim = axis === 'x' ? scene(5.5, value) : scene(value, 5.5);
            const actual = hit(sim, sides[i]);
            const expected = center === i;
            ensure(actual === expected, `${axis}=${value} side=${i}: expected ${expected}, got ${actual}`);
            cases.push({ axis, value, side: i, expected, actual });
          }
        }
        for (const axis of [0, 1]) for (const sign of [-1, 1]) {
          const point = [5.5, 5.5]; point[axis] = 6 + sign * .10005;
          ensure(hit(scene(...point), [[Math.floor(point[0]), Math.floor(point[1])]]), 'just over 10% must hit');
        }
        ensure(hit(scene(5.5, 6.0), [[5, 5], [5, 6]]), 'same tick should hit');
        for (const [y, x] of [[6, 6], [6, 5], [5, 6], [5, 5]]) {
          const rows = y === 6 ? [5, 6] : [4, 5], cols = x === 6 ? [5, 6] : [4, 5];
          const cornerCells = rows.flatMap(r => cols.map(c => [r, c]));
          ensure(!hit(scene(y, x), cornerCells.slice(0, 3)), `corner ${y},${x} missing cell should survive`);
          ensure(hit(scene(y, x), cornerCells), `corner ${y},${x} full coverage should hit`);
        }
        const replay = scene(5.5, 5.9);
        const snap = JSON.parse(JSON.stringify(replay.snapshotReplay(null)));
        const restored = new QQT.Sim(99); restored.restoreReplay(snap);
        ensure(!hit(replay, [[5, 5]]), 'replay original should survive');
        ensure(!hit(restored, [[5, 5]]), 'replay restore should survive');
        return { cases, sameTick: true, cornerMissingCellSafe: true, replayStable: true };
      });
      await page.screenshot({ path: path.join(out, `${viewport.width}-half-tile.png`), fullPage: true });
      assert.deepStrictEqual(errors, []);
      evidence.push({ viewport, result, errors, overflow: await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth) });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(out, 'checks.json'), `${JSON.stringify({ url: base, evidence }, null, 2)}\n`);
  console.log('desktop/mobile half-tile threshold, four-way, same-tick, corner and replay browser checks passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
