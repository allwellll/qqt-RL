'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const base = process.env.WEB_URL || 'http://127.0.0.1:8080/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/banana_corner_20261006/browser');

async function main() {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] });
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
        const ensure = (ok, message) => { if (!ok) throw new Error(message); };
        const V = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        const opposite = [QQT.MOVE_DOWN, QQT.MOVE_UP, QQT.MOVE_RIGHT, QQT.MOVE_LEFT];
        const scene = (direction, side) => {
          const s = window.appSim;
          s.reset('open', { nativeItems: true, nativeTrap: true });
          for (const key of ['wall', 'brick', 'fuse', 'pushable', 'crate']) s[key].fill(0);
          const offset = side < 0 ? 5 : 35;
          s.pos[0] = direction < 2 ? 5.5 : 5 + offset / 40;
          s.pos[1] = direction < 2 ? 5 + offset / 40 : 5.5;
          const [dy, dx] = V[direction], aheadRow = 5 + dy, aheadCol = 5 + dx;
          const row = aheadRow + (direction < 2 ? 0 : side), col = aheadCol + (direction < 2 ? side : 0);
          s.wall[row * QQT.W + col] = 1;
          s._setMovementStatus(0, QQT.MOVE_STATUS_SLIDE, 0); s.slideDir[0] = direction;
          return s;
        };
        const cases = [];
        for (const direction of [0, 1, 2, 3]) for (const side of [-1, 1]) {
          const s = scene(direction, side); s.wall[5 * QQT.W + 5] = 0;
          for (let i = 0; i < 5; i++) s.frameStep(0, opposite[direction], .02);
          const perp = direction < 2 ? 1 : 0;
          ensure(Math.abs(s.pos[perp] * 40 - 220) <= 1, `alignment direction=${direction} side=${side}`);
          const axis = direction < 2 ? 0 : 1, before = s.pos[axis];
          for (let i = 0; i < 15; i++) s.frameStep(0, opposite[direction], .02);
          ensure((s.pos[axis] - before) * (V[direction][0] || V[direction][1]) > .5, `direction direction=${direction}`);
          ensure(s.movementStatus[0] === QQT.MOVE_STATUS_SLIDE, `status direction=${direction}`);
          cases.push({ direction, side, alignedPx: s.pos[perp] * 40, axis: s.pos[axis] });
        }
        for (const direction of [0, 1, 2, 3]) {
          const s = scene(direction, -1); s.wall[5 * QQT.W + 5] = 0;
          const [dy, dx] = V[direction], ar = 5 + dy, ac = 5 + dx;
          const closed = direction < 2 ? [[ar, ac - 1], [ar, ac + 1]] : [[ar - 1, ac], [ar + 1, ac]];
          closed.push([ar, ac]); for (const [r, c] of closed) s.wall[r * QQT.W + c] = 1;
          const before = Array.from(s.pos); s.frameStep(0, opposite[direction], .02);
          ensure(JSON.stringify(Array.from(s.pos)) === JSON.stringify(before), `closed direction=${direction}`);
        }
        for (const kind of ['wall', 'bomb']) {
          const s = scene(3, -1); s.wall[5 * QQT.W + 5] = 0;
          for (let i = 0; i < 20; i++) s.frameStep(0, opposite[3], .02);
          const alignedY = s.pos[0]; s[kind === 'wall' ? 'wall' : 'fuse'][5 * QQT.W + 9] = 30;
          for (let i = 0; i < 30; i++) s.frameStep(0, opposite[3], .02);
          ensure(s.pos[1] < 9 && s.pos[0] === alignedY, `dynamic ${kind}`);
        }
        const original = scene(3, 1); original.wall[5 * QQT.W + 5] = 0;
        for (const direction of [0, 1, 2, 3]) for (const side of [-1, 1]) for (const kind of ['wall', 'bomb']) {
          const s = scene(direction, side); s.wall.fill(0);
          const perp = direction < 2 ? 1 : 0;
          s.pos[perp] = 5 + (side < 0 ? 4 : 36) / 40;
          const [dy, dx] = V[direction]; s.wall[(5 + dy) * QQT.W + 5 + dx] = 1;
          const target = (5 + (direction < 2 ? 0 : side)) * QQT.W + 5 + (direction < 2 ? side : 0);
          s[kind === 'wall' ? 'wall' : 'fuse'][target] = kind === 'wall' ? 1 : 30;
          const before = Array.from(s.pos);
          const restored = new QQT.Sim(99); restored.restoreReplay(JSON.parse(JSON.stringify(s.snapshotReplay(null))));
          for (const sim of [s, restored]) {
            sim.frameStep(0, opposite[direction], .02);
            ensure(JSON.stringify(Array.from(sim.pos)) === JSON.stringify(before), `alignment-side ${kind} direction=${direction} side=${side}`);
          }
        }
        scene(3, 1).wall[5 * QQT.W + 5] = 0;
        for (let i = 0; i < 8; i++) original.frameStep(0, opposite[3], .02);
        const frame = JSON.parse(JSON.stringify(original.snapshotReplay(null)));
        const restored = new QQT.Sim(99); restored.restoreReplay(frame);
        ensure(JSON.stringify(Array.from(restored.pos)) === JSON.stringify(Array.from(original.pos)), 'replay position');
        ensure(restored.movementStatus[0] === original.movementStatus[0] && restored.slideDir[0] === original.slideDir[0], 'replay slide state');
        return { cases, closedDirections: 4, dynamic: ['wall', 'bomb'], replay: true };
      });
      await page.screenshot({ path: path.join(out, `${viewport.width}-banana-corner.png`), fullPage: true });
      assert.deepStrictEqual(errors, []);
      evidence.push({ viewport, result, errors, overflow: await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth) });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(out, 'checks.json'), `${JSON.stringify({ url: base, evidence }, null, 2)}\n`);
  console.log('desktop/mobile banana corner alignment, four-way, closed, dynamic obstacle and replay browser checks passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
