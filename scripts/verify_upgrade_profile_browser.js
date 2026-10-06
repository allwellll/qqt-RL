'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base=process.env.WEB_URL || 'http://127.0.0.1:8091/';
const out=path.resolve(process.env.EVIDENCE_DIR || 'runs/qqt_main_20261006/upgrade-local');
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});const evidence=[];
  try {
    for (const viewport of [{width:1440,height:1000},{width:390,height:844}]) {
      const context=await browser.newContext({viewport}); const page=await context.newPage();
      const errors=[],submissions=[],profiles=[]; let upgraded=false, serverMessage='';
      page.on('pageerror',e=>errors.push(e.message));
      await page.route('**/build-info.json',r=>r.fulfill({json:{commit:'a'.repeat(40)}}));
      await page.route('**/rest/v1/rpc/qqt_leaderboard',r=>r.fulfill({json:profiles.length ? [{rank:1,nickname:profiles.at(-1).p_nickname,victory_message:serverMessage,player_ip:'123.*.*.89',best_win_duration_ms:12000}]:[]}));
      const submit=r=>{const p=r.request().postDataJSON().p_payload;submissions.push(p);return r.fulfill({json:{
        level:upgraded?2:1,points:upgraded?10:3,wins:1,games:submissions.length,client_match_id:p.client_match_id,match_upgraded:upgraded}});};
      await page.route('**/functions/v1/submit-result',submit);await page.route('**/rest/v1/rpc/qqt_submit_result',submit);
      await page.route('**/rest/v1/rpc/qqt_update_profile',async r=>{
        const p=r.request().postDataJSON();profiles.push(p);serverMessage=p.p_victory_message || serverMessage;
        await new Promise(resolve=>setTimeout(resolve,100));await r.fulfill({json:{saved:true,nickname:p.p_nickname,victory_message:serverMessage}});
      });
      await page.addInitScript(()=>{
        window.clockOffset=0;const now=Date.now;Date.now=()=>now()+clockOffset;
        window.setInterval=callback=>{window.appTick=callback;return 1;};
        window.requestAnimationFrame=callback=>{window.appFrame=callback;return 1;};
      });
      async function ready(){
        await page.waitForFunction(()=>window.appFrame,null,{polling:50,timeout:60000});
        await page.evaluate(()=>{
          const frame=QQT.Sim.prototype.frameStep;QQT.Sim.prototype.frameStep=function(...args){window.appSim=this;return frame.apply(this,args);};
          appFrame(performance.now());
        });
        await page.waitForFunction(()=>getComputedStyle(document.getElementById('loading')).opacity==='0',null,{polling:50});
      }
      await page.goto(base);await ready();
      assert.equal(await page.locator('aside #player-nickname, aside #player-message, aside #leaderboard-profile').count(),0);
      assert.equal(await page.locator('#leaderboard-profile').isVisible(),false);
      async function terminal(result,upgrade){
        upgraded=upgrade;
        await page.evaluate(async result=>{appSim.t=119;appSim.maxSteps=120;
          appSim.bunStored=result==='win'?[[2,1],[0,1]]:result==='loss'?[[1,0],[1,2]]:[[1,0],[0,1]];
          clockOffset+=12000;await appTick();appFrame(performance.now());},result);
        await page.waitForFunction(()=>{appFrame(performance.now());return document.getElementById('settlement-status').textContent.includes('已更新');},null,{polling:50});
        assert.equal(await page.locator('#leaderboard-profile').isVisible(),upgrade);
      }
      async function restart(){await page.locator('#game').click({position:{x:4,y:4}});await page.keyboard.press('r');
        await page.waitForFunction(()=>{appFrame(performance.now());return !appSim.done;},null,{polling:50});
        assert.equal(await page.locator('#leaderboard-profile').isVisible(),false);
      }
      await terminal('win',false);await restart();
      await terminal('win',true);
      assert.equal(await page.locator('#player-nickname').inputValue(),'');
      await page.locator('#profile-submit').click();assert.equal(profiles.length,0,'first name is required');
      await page.locator('#player-nickname').fill('首次昵称😀');await page.locator('#player-message').fill('首次宣言');
      await page.keyboard.press('r');assert.equal(await page.evaluate(()=>appSim.done),true,'focused profile blocks R');
      // Native DOM submit twice while the first request is in flight exercises the actual gate.
      await page.evaluate(()=>{const form=document.getElementById('leaderboard-profile');form.requestSubmit();form.requestSubmit();});
      await page.waitForFunction(()=>document.getElementById('profile-status').textContent.includes('已提交'),null,{polling:50});
      assert.equal(profiles.length,1);
      const geometry=await page.evaluate(()=>{
        const stage=document.querySelector('.stage').getBoundingClientRect(), form=document.getElementById('leaderboard-profile').getBoundingClientRect();
        return {stage:{top:stage.top,bottom:stage.bottom,left:stage.left,right:stage.right},form:{top:form.top,bottom:form.bottom,left:form.left,right:form.right}};
      });
      assert(geometry.form.bottom<=geometry.stage.bottom && geometry.form.left>=geometry.stage.left && geometry.form.right<=geometry.stage.right,`upgrade form must fit stage: ${JSON.stringify({viewport,geometry})}`);
      await page.screenshot({path:path.join(out,`${viewport.width}-first-upgrade.png`),fullPage:true});
      await page.locator('#player-message').fill('<img src=x onerror=alert(1)>');await page.locator('#profile-submit').click();
      await page.waitForFunction(()=>document.getElementById('profile-status').textContent.includes('尖括号'),null,{polling:50});assert.equal(profiles.length,1);
      await page.locator('#player-message').fill('新的宣言');await page.locator('#profile-submit').click();
      await page.waitForFunction(()=>document.getElementById('profile-status').textContent.includes('已提交'),null,{polling:50});assert.equal(profiles.length,2);
      await page.locator('#player-message').fill('');await page.locator('#profile-submit').click();
      await page.waitForFunction(()=>document.getElementById('profile-status').textContent.includes('已提交'),null,{polling:50});assert.equal(profiles.length,3);
      assert.equal(profiles.at(-1).p_victory_message,'');
      const retained=await page.evaluate(()=>JSON.parse(localStorage.getItem('qqt.leaderboard.v1')).victory_message);
      assert.equal(retained,'新的宣言');
      await restart();await terminal('loss',false);await restart();
      await terminal('loss',true);assert.equal(await page.locator('#player-nickname').inputValue(),'首次昵称😀');
      assert.equal(await page.locator('#player-message').inputValue(),'新的宣言');
      await page.locator('#profile-skip').click();assert.equal(await page.locator('#leaderboard-profile').isVisible(),false);await restart();
      await terminal('draw',true);assert.equal(await page.locator('#player-nickname').inputValue(),'首次昵称😀');
      await page.screenshot({path:path.join(out,`${viewport.width}-draw-upgrade.png`),fullPage:true});await restart();
      const preReloadCount=profiles.length;await page.reload();await ready();
      assert.equal(await page.locator('#leaderboard-profile').isVisible(),false,'refresh never restores old upgrade form');
      await terminal('win',true);assert.equal(await page.locator('#player-nickname').inputValue(),'首次昵称😀');
      assert.equal(await page.locator('#player-message').inputValue(),'新的宣言');assert.equal(profiles.length,preReloadCount);await restart();
      // Real WebAudio buffer loading and the actual app tick, release action and mute checkbox.
      await page.reload();await page.waitForFunction(()=>window.appFrame,null,{polling:50,timeout:60000});await ready();
      await page.evaluate(()=>{
        window.soundStarts=0;const start=AudioBufferSourceNode.prototype.start;
        AudioBufferSourceNode.prototype.start=function(...args){soundStarts++;return start.apply(this,args);};
        appSim.wall.fill(0);appSim.brick.fill(0);appSim.crate.fill(0);appSim.fuse.fill(0);
        appSim.pos[0]=5.5;appSim.pos[1]=5.5;appSim.pos[2]=1.5;appSim.pos[3]=1.5;appSim.pos[4]=1.5;appSim.pos[5]=2.5;
        appSim._addItemToSlots(0,1,3);
        QQTBunCoopHunterBot.BunCoopHunterBot.prototype.act=function(){return {move:0,ability:0};};
      });
      // Wait for the original audio fetch/decode (assets are actual WAVs, never mocked).
      await page.waitForTimeout(500);await page.locator('#game').click({position:{x:4,y:4}});
      await page.keyboard.press('1');await page.evaluate(async()=>{await appTick();appFrame(performance.now());});
      const firstRelease=await page.evaluate(()=>({starts:soundStarts,count:appSim.itemSlots[0][0].count}));
      assert.equal(firstRelease.count,2);assert.equal(firstRelease.starts,1);
      await page.keyboard.press('1');await page.evaluate(async()=>{await appTick();});assert.equal(await page.evaluate(()=>soundStarts),1,'blocked duplicate release silent');
      await page.locator('#sound-toggle').uncheck();
      await page.evaluate(()=>{appSim.pos[1]=6.5;});await page.locator('#game').click({position:{x:4,y:4}});await page.keyboard.press('1');
      await page.evaluate(async()=>{await appTick();});assert.equal(await page.evaluate(()=>soundStarts),1,'muted successful release creates no source');
      assert.equal(await page.evaluate(()=>appSim.itemSlots[0][0].count),1);
      const writesBeforeSpectating=submissions.length;
      await page.evaluate(()=>{document.getElementById('match-mode').value='model-vs-rule';appSim.done=true;appSim.winner=0;appFrame(performance.now());});
      assert.equal(await page.locator('#leaderboard-profile').isVisible(),false);assert.equal(submissions.length,writesBeforeSpectating,'spectator never submits');
      await page.reload();await ready();
      await page.locator('#replay-select').evaluate(el=>{el.closest('details').open=true;});
      await page.waitForFunction(()=>document.querySelectorAll('#replay-select option').length>1,null,{polling:50});
      const replayValue=await page.locator('#replay-select option').nth(1).getAttribute('value');
      await page.locator('#replay-select').selectOption(replayValue);
      await page.waitForFunction(()=>document.getElementById('replay-toggle').textContent==='暂停',null,{polling:50,timeout:60000});
      await page.evaluate(async()=>{await appTick();appFrame(performance.now());});
      assert.equal(await page.locator('#leaderboard-profile').isVisible(),false);assert.equal(submissions.length,writesBeforeSpectating,'real replay never submits');
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);
      evidence.push({viewport,geometry,results:['win','loss','draw'],profileWrites:profiles.length,matchWrites:submissions.length,
        cacheRefresh:true,emptyPreserved:true,doubleSubmitGate:true,skipAndR:true,watchReplayExcluded:true,releaseSourceStarts:firstRelease.starts,mutedRelease:true,errors});
      await context.close();
    }
  }finally{await browser.close();}
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({url:base,kind:'real Chromium/Sim/WebAudio assets; explicitly mocked database RPC',evidence},null,2)+'\n');
  console.log('Desktop/mobile upgrade-only profiles, cache/refresh, declaration, safe R, duplicate submissions and real release audio passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
