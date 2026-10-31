'use strict';
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright'),fs=require('fs');
const out='runs/bomb_threat_v1/iter2/zero_bomb_audit';
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),page=await browser.newPage({viewport:{width:1400,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',e=>{if(e.type()==='error')errors.push(e.text());});
 await page.route('**/*',r=>{
  const u=new URL(r.request().url());
  if(!['localhost','127.0.0.1'].includes(u.hostname)||!['GET','HEAD'].includes(r.request().method()))return r.abort();
  return r.continue();
 });
 await page.goto((process.env.ZERO_BOMB_REVIEW_URL||'http://localhost:8109/')+out+'/index.html');
 await page.waitForFunction(()=>window.reviewAudit?.verified,{timeout:60000});const rows=[];
 for(const key of await page.locator('#episode option').evaluateAll(options=>options.map(x=>x.value))){
  await page.selectOption('#episode',key);await page.waitForFunction(k=>window.reviewAudit?.key===k,key,{timeout:60000});
  await page.locator('#play').click();await page.waitForFunction(()=>window.reviewAudit.tick>=10,{timeout:10000});await page.locator('#play').click();
  rows.push(await page.evaluate(()=>({key:reviewAudit.key,verified:reviewAudit.verified,tick:reviewAudit.tick,frames:reviewAudit.frames,short_playback_fps:reviewAudit.fps})));
  await page.locator('#seek').evaluate(e=>{e.value=e.max;e.dispatchEvent(new Event('input'));});
  const reached=await page.evaluate(()=>reviewAudit.tick===Number(document.getElementById('seek').max));if(!reached)throw Error('terminal seek failed');
 }
 await page.screenshot({path:out+'/browser.png',fullPage:true});
 fs.writeFileSync(out+'/BROWSER.json',JSON.stringify({status:errors.length?'FAIL':'PASS',rows,errors,
  scope:'all complete Replay hashes verified; per-game play>=10ticks plus terminal seek; FPS short sample, not sustained benchmark',database_writes_blocked:true},null,2));
 await browser.close();console.log(JSON.stringify({games:rows.length,errors}));if(errors.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
