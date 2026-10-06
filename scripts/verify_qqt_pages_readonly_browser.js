'use strict';
const assert=require('assert'),fs=require('fs'),path=require('path');
const {execFileSync}=require('child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base=process.env.WEB_URL || 'https://allwellll.github.io/qqt-RL/';
const out=path.resolve(process.env.EVIDENCE_DIR || 'runs/qqt_main_20261006/pages-readonly');
const head=execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch({headless:true,args:['--no-sandbox']});const evidence=[];
 try{
  for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
   const context=await browser.newContext({viewport});const page=await context.newPage(),errors=[],writes=[];
   page.on('pageerror',e=>errors.push(e.message));
   page.on('request',r=>{if(r.method()==='POST' && /qqt_submit_result|submit-result|qqt_update_profile/.test(r.url())) writes.push(new URL(r.url()).pathname);});
   // Freeze gameplay ticks so this read-only live check can never complete/submit a match.
   await page.addInitScript(()=>{window.setInterval=()=>1;});
   await page.goto(base);await page.waitForFunction(()=>document.getElementById('loading').classList.contains('done') && getComputedStyle(document.getElementById('loading')).opacity==='0',null,{polling:50,timeout:60000});
   await page.waitForFunction(()=>document.getElementById('leaderboard-status').textContent.includes('已更新'),null,{polling:50,timeout:30000});
   assert.equal(await page.locator('aside #leaderboard-profile, aside #player-nickname, aside #player-message, #play-again').count(),0);
   assert.equal(await page.locator('#leaderboard-profile').isVisible(),false);
   const checked=await page.evaluate(async()=>{
    const info=await(await fetch('build-info.json',{cache:'no-store'})).json();
    const masked=Array.from(document.querySelectorAll('.leaderboard-entry .ip')).every(el=>el.textContent==='—' || /^\d{1,3}\.\*\.\*\.\d{1,3}$/.test(el.textContent) || /^[0-9a-f]{1,4}:\*:\*:[0-9a-f]{1,4}$/i.test(el.textContent));
    return {commit:info.commit,columns:document.querySelectorAll('thead th').length,rows:document.querySelectorAll('.leaderboard-entry').length,masked,overflow:document.documentElement.scrollWidth>innerWidth};
   });assert.equal(checked.commit,head);assert.equal(checked.columns,5);assert.equal(checked.masked,true);assert.equal(checked.overflow,false);assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);
   await page.screenshot({path:path.join(out,`${viewport.width}.png`),fullPage:true});evidence.push({viewport,...checked,errors,writes});await context.close();
  }
 }finally{await browser.close();}
 fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({url:base,head,networkMock:false,gameTicksFrozen:true,evidence},null,2)+'\n');console.log('Live Pages desktop/mobile nonmock leaderboard read, masked-only display, current commit and no writes passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
