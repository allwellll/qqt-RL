'use strict';
// Real worker/model/network in both seats/modes; never submits player results.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.WEB_URL||'http://localhost:8123/';
const out=process.env.EVIDENCE_FILE||'runs/model-temperature-browser.json';
const seed=2026629001;
fs.mkdirSync(path.dirname(out),{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
 const results=[];
 let activePage=null;
 const heartbeat=setInterval(async()=>{try{if(activePage)console.log(JSON.stringify(await activePage.evaluate(()=>({stage:'browser-progress',tick:window.__gameSim?.t,done:window.__gameSim?.done,status:document.querySelector('#model-status')?.textContent,mode:document.querySelector('#match-mode')?.value}))));}catch{}},15000);heartbeat.unref();
 try{
  for(const mobile of [false,true]){
   const context=await browser.newContext({viewport:mobile?{width:390,height:844}:{width:1440,height:1000},isMobile:mobile,hasTouch:mobile});
   const writes=[],errors=[],responses=[];
   await context.route('**/*',route=>{
    const r=route.request(),p=new URL(r.url()).pathname;
    if(['GET','HEAD'].includes(r.method())||(r.method()==='POST'&&/\/rpc\/qqt_(leaderboard|get_profile|my_win_leaderboard|my_win_leaderboard_ruleset|ruleset_contract)$/.test(p)))return route.continue();
    writes.push({method:r.method(),path:p});return route.fulfill({status:200,contentType:'application/json',body:'{}'});
   });
   const page=await context.newPage();activePage=page;page.on('pageerror',e=>errors.push(e.message));
   page.on('response',r=>{if(r.url().includes('/models/'))responses.push({path:new URL(r.url()).pathname,status:r.status()});});
   await page.addInitScript(()=>{
    const interval=window.setInterval;
    window.setInterval=(fn,ms,...args)=>ms===100?(window.__gameTick=fn,1):interval(fn,ms,...args);
   });
   await page.goto(base+'?seed='+seed,{waitUntil:'domcontentloaded',timeout:120000});
   await page.waitForFunction(()=>window.__gameTick&&document.querySelector('#model-options option'),null,{timeout:120000});
   const identity=await page.evaluate(async()=>{
    const defaults=await(await fetch('model_inference_defaults.json',{cache:'no-store'})).json();
    const manifest=await(await fetch('models.json',{cache:'no-store'})).json();
    const row=manifest.models.find(r=>r.id===defaults.default_model_id);
    const digest=async file=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await(await fetch(file,{cache:'no-store'})).arrayBuffer())),x=>x.toString(16).padStart(2,'0')).join('');
    return {defaults,row,simulator_sha256:await digest('sim-bun-native-v2.js'),sampling_sha256:await digest('model_sampling_v1.js')};
   });
   assert(identity.row);if(process.env.MODEL_ID)assert.equal(identity.row.id,process.env.MODEL_ID);
   if(process.env.MODEL_SHA256)assert.equal(identity.row.sha256,process.env.MODEL_SHA256);
   if(process.env.ONNX_SHA256)assert.equal(identity.row.onnx.sha256,process.env.ONNX_SHA256);
   await page.evaluate(()=>{
    window.__modelActions=[];
    const load=QQTModelWorker.load;
    QQTModelWorker.load=async(...args)=>{
     const m=await load(...args);window.__model=m;window.__loadedRow=args[0];
     const act=m.act;m.act=async(...a)=>{
      const r=await act.apply(m,a);if(!r.every(Number.isFinite))throw Error('nonfinite action');
      window.__modelActions.push({seat:a[1],tick:a[0].t,action:r,decision:m.lastDecision});return r;
     };return m;
    };
    const reset=QQT.Sim.prototype.reset;QQT.Sim.prototype.reset=function(...a){window.__gameSim=this;return reset.apply(this,a);};
   });
   await page.evaluate(id=>{const e=document.getElementById('opponent');e.value='model:'+id;e.dispatchEvent(new Event('change'));},identity.row.id);
   await page.waitForFunction(()=>document.querySelector('#model-status').textContent.startsWith('已加载：')||document.querySelector('#model-status').textContent.startsWith('加载失败：'),null,{timeout:20*60*1000});
   assert((await page.textContent('#model-status')).startsWith('已加载：'),await page.textContent('#model-status'));
   for(const d of await page.locator('dialog[open] [data-dialog-close]').all())await d.click();
   const humanPositions=[];
   for(const key of ['ArrowRight','ArrowDown','ArrowLeft','ArrowUp']){
    if(mobile){
     const button=page.locator(`[data-control-key="${key}"]`),rect=await button.boundingBox();assert(rect);
     const cdp=await context.newCDPSession(page);
     await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:rect.x+rect.width/2,y:rect.y+rect.height/2}]});
     await page.waitForTimeout(250);await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await cdp.detach();
    }else{await page.keyboard.down(key);await page.waitForTimeout(250);await page.keyboard.up(key);}
    humanPositions.push(await page.evaluate(()=>Array.from(__gameSim.pos.slice(0,2))));
   }
   assert(humanPositions.some((a,i)=>i&&a.some((v,j)=>Math.abs(v-humanPositions[0][j])>.001)),'real human input must move P0');
   if(mobile)await page.locator('[data-control-key="Space"]').tap();
   else{await page.keyboard.press('Space');}
   await page.evaluate(()=>__gameTick());
   assert(await page.evaluate(()=>__gameSim.liveBombs(0)>0),'human bomb input must reach native physics');
   await page.screenshot({path:out+`-${mobile?'mobile':'desktop'}-human-controls.png`,fullPage:true});
   const modes=[];
   for(const mode of ['human-vs-opponent','model-vs-rule']){
    await page.evaluate(async mode=>{const e=document.getElementById('match-mode');e.value=mode;e.dispatchEvent(new Event('change'));},mode);
    await page.waitForTimeout(150);
    await page.evaluate(async()=>{window.__modelActions=[];for(let i=0;i<2400&&!__gameSim.done;i++)await __gameTick();});
    const r=await page.evaluate(()=>({ticks:__gameSim.t,done:__gameSim.done,winner:__gameSim.winner,nativeItems:__gameSim.nativeItems,nativeTrap:__gameSim.nativeTrap,players:__gameSim.nPlayers,ruleset:QQTRulesets.current,backend:__model.backend,greedy:__model.greedy,temperature:__model.temperature,profile:__model.samplingProfile,model_sha256:__loadedRow.sha256,onnx_sha256:__loadedRow.onnx.sha256,checkpoint_sha256:__model.meta.checkpoint_sha256,actions:__modelActions,touch_hidden:document.querySelector('#touch-controls').hidden}));
    assert(r.done&&r.ticks<=2400);assert(r.nativeItems&&r.nativeTrap);assert.equal(r.players,2);assert.equal(r.ruleset,'bun-native-v2');assert.equal(r.backend,'worker-wasm');
    assert(!r.greedy);assert.equal(r.temperature,Number(process.env.TEMPERATURE||.3));
    assert.equal(r.model_sha256,identity.row.sha256);assert.equal(r.onnx_sha256,identity.row.onnx.sha256);
    assert(r.actions.length&&r.actions.every(a=>a.seat===1));
    assert(r.actions.every(a=>!a.decision||(a.decision.move_mask[a.decision.move]&&a.decision.ability_mask[a.decision.ability])));
    modes.push({mode,...r});await page.screenshot({path:out+`-${mobile?'mobile':'desktop'}-${mode}.png`,fullPage:true});
   }
   // A fresh fixed seed must reproduce the complete observation-mode action sequence.
   const old=modes[1];await page.evaluate(()=>document.getElementById('restart').click());await page.waitForTimeout(150);
   const repeated=await page.evaluate(async()=>{window.__modelActions=[];for(let i=0;i<2400&&!__gameSim.done;i++)await __gameTick();return {actions:__modelActions,ticks:__gameSim.t,winner:__gameSim.winner};});
   assert.deepEqual(repeated.actions,old.actions);assert.equal(repeated.ticks,old.ticks);assert.equal(repeated.winner,old.winner);
   let build=null;if(process.env.EXPECTED_HEAD){build=await page.evaluate(async()=>await(await fetch('build-info.json?verify='+Date.now(),{cache:'no-store'})).json());assert.equal(build.commit,process.env.EXPECTED_HEAD);}
   assert.deepEqual(errors,[]);results.push({mobile,identity,modes,humanPositions,seed_replay_exact:true,build,errors,writes_blocked:writes,modelResponses:responses});await context.close();
  }
  assert.deepEqual(results[0].modes[1].actions,results[1].modes[1].actions,'same seed spectator actions agree across viewports');
  fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify({status:'PASS',live_network:true,model_cache_mock:false,cross_viewport_seed_exact:true,results},null,2)+'\n');
  console.log(JSON.stringify({status:'PASS',viewports:results.length,modes:results.map(r=>r.modes.map(x=>({mode:x.mode,ticks:x.ticks,winner:x.winner,temperature:x.temperature,actions:x.actions.length}))),seed_replay_exact:true}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
