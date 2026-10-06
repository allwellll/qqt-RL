'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.WEB_URL || 'http://127.0.0.1:8091/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/qqt_round2_20261007/browser-local');
(async () => {
  fs.mkdirSync(out,{recursive:true});
  const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
  const evidence=[];
  try {
    for (const viewport of [{width:1440,height:1000},{width:390,height:844}]) {
      const context = await browser.newContext({viewport,isMobile:viewport.width<600,hasTouch:viewport.width<600});
      const page = await context.newPage(), errors=[], consoleErrors=[], writes=[], profileWrites=[];
      let fail=false, upgraded=false, submitting=false, hold=true, releaseSubmit;
      page.on('pageerror',e=>errors.push(e.message));
      page.on('console',msg=>{if(msg.type()==='error') consoleErrors.push(msg.text());});
      // Mock every write entry point in both local and Pages tests. Reads/assets are real.
      const submit = async route => {
        const p=route.request().postDataJSON().p_payload; writes.push(p);
        if(fail) return route.fulfill({status:422,json:{error:'controlled rejection'}});
        submitting=true;
        if (hold) await new Promise(resolve=>{releaseSubmit=resolve;});
        else await new Promise(resolve=>setTimeout(resolve,250));
        submitting=false;
        return route.fulfill({json:{level:upgraded?2:1,points:10,wins:1,games:writes.length,
          client_match_id:p.client_match_id,match_upgraded:upgraded,
          match_rank:{rank:2,total:5,percentile:60,comparison:'player-best-v1',result:p.result,
            mode:p.mode,map_id:p.map_id,difficulty:p.difficulty,opponent:p.opponent,duration_ms:p.game_duration_ms}}});
      };
      await page.route('**/functions/v1/submit-result',submit);
      await page.route('**/rest/v1/rpc/qqt_submit_result',submit);
      await page.route('**/rest/v1/rpc/qqt_update_profile', async route=>{
        const p=route.request().postDataJSON(); profileWrites.push(p);
        await new Promise(resolve=>setTimeout(resolve,200));
        await route.fulfill({json:{saved:true,nickname:p.p_nickname,victory_message:p.p_victory_message}});
      });
      await page.addInitScript(()=>{
        window.clockOffset=0; const now=Date.now; Date.now=()=>now()+clockOffset;
        window.setInterval=cb=>{window.appTick=cb;return 1;};
        window.requestAnimationFrame=cb=>{window.appFrame=cb;return 1;};
      });
      const load = async()=>{
        await page.waitForFunction(()=>window.appFrame,null,{polling:50,timeout:60000});
        await page.evaluate(()=>{
          const frame=QQT.Sim.prototype.frameStep;
          QQT.Sim.prototype.frameStep=function(...args){window.appSim=this;return frame.apply(this,args);};
          appFrame(performance.now());
        });
        await page.waitForFunction(()=>getComputedStyle(document.getElementById('loading')).opacity==='0',null,{polling:50});
      };
      const action = async id=>viewport.width<600?page.locator(id).tap():page.locator(id).click();
      const longPress = async id => {
        const target = page.locator(id); await target.scrollIntoViewIfNeeded();
        const r = await target.boundingBox(), x = r.x + r.width/2, y = r.y + r.height/2;
        if (viewport.width < 600) {
          const cdp = await context.newCDPSession(page);
          await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y}]});
          await page.waitForTimeout(350);
          await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]}); await cdp.detach();
        } else {
          await page.mouse.move(x,y); await page.mouse.down(); await page.waitForTimeout(350); await page.mouse.up();
        }
      };
      const render = ()=>page.evaluate(()=>appFrame(performance.now()));
      const waitState = fn=>page.waitForFunction(fn,null,{polling:50,timeout:15000});
      const unlock = async()=>{
        await action('#advanced-open'); await page.locator('#settings-password').fill('demaxiya');
        await page.locator('#settings-password').press('Enter');
        await page.waitForFunction(()=>!document.getElementById('advanced-settings').hidden,null,{polling:50});
      };
      await page.goto(base); await load();
      assert.equal(await page.locator('#advanced-settings').evaluate(e=>e.hidden),true);
      assert.equal(await page.locator('#restart').isVisible(),false);
      const canvasBefore=await page.locator('#game').boundingBox();
      await action('#advanced-open'); await page.locator('#settings-password').fill('bad'); await page.keyboard.press('Enter');
      assert.match(await page.locator('#password-status').textContent(),/口令不对/);
      assert.equal(await page.locator('#advanced-settings').evaluate(e=>e.hidden),true);
      await page.keyboard.press('Escape'); assert.equal(await page.locator('#password-dialog').evaluate(e=>e.open),false);
      assert.equal(await page.evaluate(()=>document.activeElement.id),'advanced-open');
      await unlock(); assert.equal(await page.locator('#game').boundingBox().then(r=>r.width),canvasBefore.width);
      for(const id of ['restart','sound-toggle','map-select','match-mode','team-mode','opponent','model-details','model-status','replay-select','replay-speed','replay-seek','status','pending-submit']) {
        assert.equal(await page.locator('#'+id).evaluate(e=>!!e.closest('#advanced-settings')),true);
      }
      await page.locator('#advanced-close').focus(); await page.keyboard.press('Escape');
      await action('#advanced-open'); assert.equal(await page.locator('#password-dialog').evaluate(e=>e.open),true);
      assert.equal(await page.locator('#settings-password').inputValue(),''); await page.keyboard.press('Escape');
      await unlock(); await page.reload(); await load();
      assert.equal(await page.locator('#advanced-settings').evaluate(e=>e.hidden),true,'reload locks');
      await action('#announcement-open');
      assert.equal(await page.locator('#announcement-dialog').evaluate(e=>e.open),true);
      const dialogBounds = await page.locator('#announcement-dialog').boundingBox();
      assert(dialogBounds.x >= 0 && dialogBounds.y >= 0 && dialogBounds.x + dialogBounds.width <= viewport.width && dialogBounds.y + dialogBounds.height <= viewport.height);
      assert.match(await page.locator('#announcement-dialog').textContent(),/顶尖玩家的水平差距/);
      assert.equal(await page.locator('#announcement-dialog a').getAttribute('href'),'mailto:1035628914@qq.com');
      for(let i=0;i<5;i++) await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(()=>!!document.activeElement.closest('#announcement-dialog')),true,'native dialog traps focus');
      await page.screenshot({path:path.join(out,`${viewport.width}-announcement.png`),fullPage:true});
      await page.keyboard.press('Escape'); assert.equal(await page.evaluate(()=>document.activeElement.id),'announcement-open');
      await action('#announcement-open'); await action('#announcement-dialog [data-dialog-close]');
      await action('#announcement-open'); await page.mouse.click(2,2);
      assert.equal(await page.locator('#announcement-dialog').evaluate(e=>e.open),false);
      // Regression through real Sim placement and human frame movement, not a visual-only status injection.
      const trap=await page.evaluate(()=>{
        const s=appSim, snap=s.snapshotReplay(), cell=4*QQT.W+5;
        for(const k of ['wall','brick','crate','fuse']) s[k].fill(0);
        s.invuln.fill(0); s.spawnProtection.fill(0);
        s.pos[0]=4.5;s.pos[1]=4.9;s.pos[2]=4.5;s.pos[3]=5.5;
        for(let p=2;p<s.nPlayers;p++){s.pos[p*2]=10.5;s.pos[p*2+1]=10.5;}
        s.itemSlots[1]=[{item:QQT.ITEM_SLOW_GLUE,count:1}];s._syncHeldItem(1);
        const actions=Array.from({length:s.nPlayers},()=>[QQT.MOVE_IDLE,0,0,1]);actions[1][2]=1;
        s.step(actions); const before={item:s.fieldItem[cell],armed:s.fieldArmed[cell],owner:s.fieldOwner[cell]};
        s.frameStep(0,QQT.MOVE_RIGHT,.05);const after={status:s.movementStatus[0],consumed:s.fieldItem[cell]===0};
        s.restoreReplay(snap); appFrame(performance.now()); return {before,after};
      });
      assert.equal(trap.before.armed,0);assert.equal(trap.before.owner,1);assert.equal(trap.after.status,await page.evaluate(()=>QQT.MOVE_STATUS_SLOW));assert(trap.after.consumed);
      const outcomes=[];
      for(const result of ['win','loss','draw']) {
        upgraded=result==='loss'; fail=result==='win';
        const before=writes.length;
        await page.evaluate(async result=>{
          appSim.t=119;appSim.maxSteps=120;
          appSim.bunStored=result==='win'?[[2,1],[0,1]]:result==='loss'?[[1,0],[1,2]]:[[1,0],[0,1]];
          clockOffset+=12000;await appTick();appFrame(performance.now());
        },result);
        assert.equal(writes.length,before,'terminal must not implicitly submit');
        assert.equal(await page.locator('#settlement button').count(),2);
        await action('#settlement-close');await render();
        assert.equal(await page.locator('#settlement').evaluate(e=>e.hidden),true);assert.equal(writes.length,before);
        await action('#game');await render();assert(await page.locator('#settlement').isVisible());
        if (result === 'draw') { await page.locator('#settlement-submit').focus(); await page.keyboard.down('Enter'); }
        else await longPress('#settlement-submit');
        await render();
        if(fail) {
          await waitState(()=>{appFrame(performance.now());return document.getElementById('settlement-submit').disabled===false;});
          assert.match(await page.locator('#settlement-status').textContent(),/可重试/);assert.equal(writes.length,before+1);
          fail=false;await action('#settlement-submit');await render();
        }
        await waitState(()=>{appFrame(performance.now());return document.getElementById('settlement-submit').textContent==='提交中…';});
        // Real held pointer/touch and repeated Enter share the same in-flight guard.
        await page.keyboard.down('Enter'); await page.keyboard.down('Enter'); await page.keyboard.up('Enter');
        await page.locator('#settlement-submit').evaluate(e=>{e.click();e.click();});
        await page.evaluate(()=>{window.oldSim=appSim;window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR'}));window.dispatchEvent(new KeyboardEvent('keyup',{code:'KeyR'}));appFrame(performance.now());});
        assert(submitting); assert.equal(await page.evaluate(()=>appSim===oldSim),true,'R must wait for submission');
        assert(releaseSubmit); releaseSubmit(); releaseSubmit=null;
        await waitState(()=>{appFrame(performance.now());return document.getElementById('settlement-submit').textContent==='提交成功';});
        assert.equal(writes.length,before+(result==='win'?2:1));
        if(result==='win') assert.deepEqual(writes[before],writes[before+1]);
        assert.match(await page.locator('#settlement-rank').textContent(),/第 2 名/);
        if(upgraded) {
          assert(await page.locator('#leaderboard-profile').isVisible());assert(await page.locator('#settlement-close').isDisabled());
          await page.screenshot({path:path.join(out,`${viewport.width}-upgrade.png`),fullPage:true});
          const formRect = await page.locator('#leaderboard-profile').boundingBox(), buttonRect = await page.locator('#settlement-submit').boundingBox();
          assert(formRect.y >= buttonRect.y + buttonRect.height, 'upgrade form must not cover settlement actions');
          await page.locator('#player-nickname').fill('练习玩家');await page.locator('#player-message').fill('继续练习');
          await page.keyboard.press('r');await render();assert.equal(await page.evaluate(()=>appSim===oldSim),true);
          await action('#profile-submit');await render();
          await page.evaluate(()=>{document.activeElement.blur();window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR'}));window.dispatchEvent(new KeyboardEvent('keyup',{code:'KeyR'}));});
          assert.equal(await page.evaluate(()=>appSim===oldSim),true,'R cannot interrupt profile request');
          await waitState(()=>document.getElementById('profile-status').textContent.includes('已提交'));
          await render(); assert.equal(await page.locator('#settlement-close').isDisabled(),false);
          await action('#settlement-close'); await render(); assert.equal(await page.locator('#leaderboard-profile').isVisible(),false);
          await action('#game'); await render();
          await page.locator('#player-message').fill('再练一局'); await render();
          assert.equal(await page.locator('#settlement-close').isDisabled(),true,'new edits protect unsaved form');
          await action('#profile-skip');await render();
        }
        await page.screenshot({path:path.join(out,`${viewport.width}-${result}.png`),fullPage:true});
        await page.locator('#settlement-submit').evaluate(e=>e.blur());
        await page.evaluate(()=>{
          window.oldSim=appSim;
          window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR',repeat:true}));
          appFrame(performance.now());
        }); assert.equal(await page.evaluate(()=>appSim===oldSim),true,'repeat alone must not restart');
        await page.evaluate(()=>{
          window.dispatchEvent(new KeyboardEvent('keyup',{code:'KeyR'}));
          window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR'}));
          window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR'}));
          window.dispatchEvent(new KeyboardEvent('keyup',{code:'KeyR'}));
        });
        await waitState(()=>{appFrame(performance.now());return appSim!==oldSim && !appSim.done;});
        outcomes.push({result,explicitSubmit:true,retry:result==='win',upgradeGuard:upgraded});
      }
      // An unsubmitted R restart survives reload and needs explicit authorization.
      await page.evaluate(async()=>{appSim.t=119;appSim.maxSteps=120;clockOffset+=12000;await appTick();appFrame(performance.now());});
      const draftWrites=writes.length;
      await page.keyboard.press('r');await render();
      await page.reload();await load();assert.equal(writes.length,draftWrites);
      hold=false; await unlock(); await action('#pending-submit');
      await waitState(()=>{appFrame(performance.now());return JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).queue.length===0;});
      assert.equal(writes.length,draftWrites+1);
      await action('#advanced-close');await render();
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      assert.deepEqual(errors,[]);
      // A mocked 422 is an expected console network error; other console errors fail.
      assert(consoleErrors.every(e=>e.includes('422')),JSON.stringify(consoleErrors));
      await page.screenshot({path:path.join(out,`${viewport.width}-final.png`),fullPage:true});
      evidence.push({viewport,trap,outcomes,writeRequestsMocked:writes.length,profileRequestsMocked:profileWrites.length,
        passwordAndRefresh:true,announcementFocusAndClose:true,persistedDraft:true,horizontalOverflow:false,errors,consoleErrors});
      await context.close();
    }
  } finally {await browser.close();}
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({url:base,kind:'real Chromium/Sim/Canvas/assets; real leaderboard reads; every result/profile write mocked',evidence},null,2)+'\n');
  console.log('Round 2 real Chromium desktop/mobile: all four requirements passed; remote writes mocked');
})().catch(e=>{console.error(e);process.exitCode=1;});
