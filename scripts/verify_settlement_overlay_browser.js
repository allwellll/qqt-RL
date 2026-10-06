'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.WEB_URL || 'http://127.0.0.1:8091/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/qqt_main_20261006/terminal-local');
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const evidence = [];
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport });
      const errors = [], submissions = []; let withRank = true;
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/rest/v1/rpc/qqt_leaderboard', r => r.fulfill({ json: [] }));
      await page.route('**/build-info.json', r => r.fulfill({ json: { commit: 'a'.repeat(40) } }));
      const submit = r => {
        const p = r.request().postDataJSON().p_payload; submissions.push(p);
        return r.fulfill({ json: { level: 1, points: 3, wins: 1, games: submissions.length,
          ...(withRank ? { match_rank: { rank: 2, total: 5, percentile: 60, comparison: 'player-best-v1',
            result: p.result, mode: p.mode, map_id: p.map_id, difficulty: p.difficulty,
            opponent: p.opponent, duration_ms: p.game_duration_ms } } : {}) } });
      };
      await page.route('**/rest/v1/rpc/qqt_submit_result', submit);
      await page.route('**/functions/v1/submit-result', submit);
      await page.addInitScript(() => {
        window.clockOffset = 0; const now = Date.now; Date.now = () => now() + clockOffset;
        window.setInterval = callback => { window.appTick = callback; return 1; };
        window.requestAnimationFrame = callback => { window.appFrame = callback; return 1; };
        const key = Object.keys(localStorage).find(k => k.startsWith('qqt.')); // Fresh browser context only.
        if (key) localStorage.removeItem(key);
      });
      await page.goto(base);
      await page.waitForFunction(() => window.appFrame, null, { polling: 50, timeout: 60000 });
      await page.evaluate(() => {
        const frame = QQT.Sim.prototype.frameStep, reset = QQT.Sim.prototype.reset;
        window.resetCount = 0;
        QQT.Sim.prototype.reset = function(...args) { resetCount++; return reset.apply(this,args); };
        QQT.Sim.prototype.frameStep = function(...args) { window.appSim = this; return frame.apply(this,args); };
        const ctx = document.getElementById('game').getContext('2d'), fill = ctx.fillText;
        window.canvasText = [];
        ctx.fillText = function(text,x,y,...args) { canvasText.push({ text: String(text), x,y, font:this.font, color:this.fillStyle }); return fill.call(this,text,x,y,...args); };
        appFrame(performance.now());
      });
      await page.waitForFunction(() => getComputedStyle(document.getElementById('loading')).opacity === '0', null, { polling: 50 });
      assert.equal(await page.locator('#play-again').count(),0);
      const outcomes = [];
      for (const [result,title,delivery,ranked] of [
        ['win','胜利',false,true], ['loss','失败',false,true], ['draw','平局',false,false],
        ['win','胜利',true,true], ['loss','失败',true,false]
      ]) {
        withRank = ranked;
        await page.evaluate(async ({result,delivery}) => {
          appSim.t = 119; appSim.maxSteps = 120;
          appSim.bunStored = result === 'win' ? [[2,1],[0,1]] : result === 'loss' ? [[1,0],[1,2]] : [[1,0],[0,1]];
          clockOffset += 12000; await appTick();
          if (delivery) { appSim.t = 120; appSim.maxSteps = 2400; appSim.bunScore = result === 'win' ? [1,0] : [0,1]; }
          appFrame(performance.now());
        }, {result,delivery});
        await page.waitForFunction(() => { appFrame(performance.now()); return document.getElementById('settlement-status').textContent.includes('已更新'); }, null, { polling: 50 });
        const texts = await page.evaluate(() => { canvasText = []; appFrame(performance.now()); return canvasText; });
        assert.equal(texts.filter(row => row.text === title).length,1);
        assert(texts.some(row => row.text.includes(delivery ? result === 'loss' ? '运包失败' : '运包成功' : '时间到') && row.text.includes('包子')));
        assert(texts.some(row => row.text === '本局耗时 12.0秒'));
        assert(texts.some(row => row.text === (ranked ? '第 2 名 / 5 位玩家' : '排名待数据库升级')));
        assert(texts.some(row => row.text === '按 R 再来一局'));
        const titleRow = texts.find(row => row.text === title);
        const lines = texts.filter(row => row.x === titleRow.x && row.y >= titleRow.y);
        for (let i = 1; i < lines.length; i++) assert(lines[i].y > lines[i-1].y, 'Canvas text lines must have separate baselines');
        assert.equal(await page.locator('#settlement').evaluate(el => getComputedStyle(el).clipPath), 'inset(50%)');
        await page.screenshot({ path: path.join(out, `${viewport.width}-${result}-${delivery ? 'delivery' : 'timeout'}.png`), fullPage: true });
        // A real input focused in the game must still block R; upgraded profile tests exercise its own inputs.
        await page.locator('#replay-seek').evaluate(el=>{ el.closest('details').open=true; });
        await page.locator('#replay-seek').focus(); const before = await page.evaluate(() => resetCount);
        await page.keyboard.press('r'); assert.equal(await page.evaluate(() => resetCount),before);
        await page.locator('#replay-seek').evaluate(el=>el.blur());
        await page.evaluate(() => {
          window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR',repeat:true}));
        }); assert.equal(await page.evaluate(() => resetCount),before);
        await page.evaluate(() => {
          window.dispatchEvent(new KeyboardEvent('keyup',{code:'KeyR'}));
          window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR'}));
          window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR'}));
          window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR',repeat:true}));
          window.dispatchEvent(new KeyboardEvent('keyup',{code:'KeyR'}));
        });
        await page.waitForFunction(old=>resetCount===old+2,before,{polling:50});
        await page.evaluate(() => appFrame(performance.now()));
        assert.equal(await page.evaluate(() => appSim.done),false);
        assert.equal(await page.locator('#settlement').evaluate(el=>el.hidden),true);
        outcomes.push({ result, delivery, ranked, canvasLines:lines, focusAndRepeatGate:true });
      }
      assert.equal(submissions.length,5);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(errors,[]);
      evidence.push({viewport,outcomes,submissions:submissions.length,errors}); await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({url:base,kind:'real Chromium/Sim/Canvas; mocked settlement RPC',evidence},null,2)+'\n');
  console.log('Canvas terminal merge: desktop/mobile win/loss/draw, delivery/timeout, ranks and guarded R passed');
})().catch(error=>{ console.error(error); process.exitCode=1; });
