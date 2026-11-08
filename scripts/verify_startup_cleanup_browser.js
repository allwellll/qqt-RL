'use strict';
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
const {createServer}=require('../web/server');
(async()=>{
 fs.mkdirSync('.hermes',{recursive:true});const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const base=process.env.STARTUP_PUBLIC_URL||`http://127.0.0.1:${server.address().port}/`,publicCheck=!!process.env.STARTUP_PUBLIC_URL;
 const opts={headless:true,executablePath:chromium.executablePath(),args:['--no-sandbox','--disable-gpu']};const raw=process.env.HTTPS_PROXY||process.env.https_proxy;if(raw){const u=new URL(raw);opts.proxy={server:u.protocol+'//'+u.host,bypass:'127.0.0.1,localhost'};}
 const browser=await chromium.launch(opts),results=[];
 try{for(const width of [1280,390]){
  const context=await browser.newContext({viewport:{width,height:900},hasTouch:width===390,isMobile:width===390});
  if(!publicCheck)await context.route('http://127.0.0.1:*/**',async r=>{const x=await fetch(r.request().url());await r.fulfill({status:x.status,headers:Object.fromEntries(x.headers),body:Buffer.from(await x.arrayBuffer())});});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{let visual;Object.defineProperty(window,'QQTVisual',{get:()=>visual,set:v=>{visual=v;const create=v.createRenderer;v.createRenderer=(...a)=>{const r=create(...a),render=r.render;r.render=(s,...a)=>{window.startupSim=s;return render(s,...a);};return r;};}});});
  for(const visit of ['cold','warm']){
   await page.goto(base+'?startup-check='+Date.now(),{waitUntil:'domcontentloaded',timeout:60000});
   await page.waitForFunction(()=>document.querySelector('#loading')?.classList.contains('done'),null,{timeout:180000});
   const data=await page.evaluate(()=>{
    const marks=performance.getEntriesByType('mark').map(e=>({name:e.name,ms:e.startTime})),last=name=>marks.filter(e=>e.name===name).at(-1)?.ms;
    return {marks,totalMs:last('qqt:first-frame'),readyToFrameMs:last('qqt:first-frame')-last('qqt:model-ready'),resetMs:last('qqt:reset-ready')-last('qqt:reset-start'),replayReadyAtFirstFrame:!!last('qqt:replays-ready'),ruleset:QQTRulesets.current,nPlayers:startupSim.nPlayers,modelOptions:[...document.querySelector('#opponent').options].map(o=>o.value),scripts:[...document.scripts].map(s=>s.getAttribute('src')).filter(Boolean)};
   });
   assert(data.readyToFrameMs<1000);assert.equal(data.ruleset,'bun-native-v4-facing');assert.equal(data.nPlayers,2);assert.equal(data.modelOptions.length,4);assert(data.modelOptions.every(s=>s.startsWith('model:')));
   assert(!data.scripts.some(s=>/sim-stock|sim-bun-native-v[23]|hunter|bun_rule_bot/.test(s)));
   for(const id of ['ruleset-select','team-mode','match-mode','model-opponent','map-select','physics-revision','item-supply-status','boundary-recovery-status','bubble-passage-legend','kill-mode-status','model-specialty','model-details'])assert.equal(await page.locator('#'+id).count(),0,id+' removed');
   assert.equal(await page.locator('#boundary-recovery').isChecked(),true);assert.equal(await page.locator('#boundary-recovery').isVisible(),false);
   const movement=await page.evaluate(()=>Array.from(startupSim.pos.slice(0,2)));await page.keyboard.down('ArrowRight');await page.waitForTimeout(80);await page.keyboard.up('ArrowRight');
   if(process.env.EXPECTED_SHA)assert.equal(await page.evaluate(async()=>(await(await fetch('build-info.json?fresh='+Date.now(),{cache:'no-store'})).json()).commit),process.env.EXPECTED_SHA);
   await page.evaluate(()=>document.querySelector('#advanced-settings').hidden=false);
   const descriptions=await page.locator('aside').evaluate(e=>{const clone=e.cloneNode(true);for(const s of clone.querySelectorAll('select'))s.remove();return clone.textContent;});assert(!/队友|救援|猎手|可互救|前端辅助|穿泡说明|旧规则可|观战/.test(descriptions));
   for(const mode of ['carry','kill']){await page.selectOption('#game-mode',mode);await page.waitForFunction(m=>startupSim.killMode===(m==='kill'),mode);assert.equal(await page.evaluate(()=>startupSim.nPlayers),2);}
   await page.screenshot({path:`.hermes/startup-cleanup-${publicCheck?'public':'local'}-${width}-${visit}.png`});
   assert.deepEqual(errors,[]);results.push({width,visit,...data,inputPressed:true,modeChecks:['carry','kill'],errors});console.log(JSON.stringify({width,visit,totalMs:data.totalMs,readyToFrameMs:data.readyToFrameMs,resetMs:data.resetMs,replayReadyAtFirstFrame:data.replayReadyAtFirstFrame}));
  }
  const review=await context.newPage();review.on('pageerror',e=>errors.push(e.message));await review.goto(base+'review.html',{waitUntil:'domcontentloaded',timeout:60000});await review.waitForFunction(()=>Number(document.querySelector('#count')?.textContent)>0,null,{timeout:30000});assert(await review.locator('canvas').count()>0);assert.deepEqual(errors,[]);await context.close();
 }
 fs.writeFileSync(`.hermes/startup-cleanup-${publicCheck?'public':'local'}.json`,JSON.stringify({status:'PASS',sha:process.env.EXPECTED_SHA||null,scope:'Chromium desktop and mobile viewport; cold new context, warm same-context revisit; not physical iOS',results},null,2)+'\n');
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
