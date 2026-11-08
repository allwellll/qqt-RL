'use strict';
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
const {createServer}=require('../web/server');
(async()=>{
 fs.mkdirSync('.hermes',{recursive:true});const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const publicCheck=!!process.env.PREVIEW_PUBLIC_URL,base=process.env.PREVIEW_PUBLIC_URL||`http://127.0.0.1:${server.address().port}/`;
 const opts={headless:true,executablePath:chromium.executablePath(),args:['--no-sandbox','--disable-gpu']};
 const raw=process.env.HTTPS_PROXY||process.env.https_proxy;if(raw){const u=new URL(raw);opts.proxy={server:u.protocol+'//'+u.host,bypass:'127.0.0.1,localhost'};}
 const browser=await chromium.launch(opts),results=[];
 try{for(const width of [1280,390]){
  const context=await browser.newContext({viewport:{width,height:900},hasTouch:width===390,isMobile:width===390});
  if(!publicCheck)await context.route('http://127.0.0.1:*/**',async r=>{const x=await fetch(r.request().url());await r.fulfill({status:x.status,headers:Object.fromEntries(x.headers),body:Buffer.from(await x.arrayBuffer())});});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{
   window.previewLabels=[];const text=CanvasRenderingContext2D.prototype.fillText;
   CanvasRenderingContext2D.prototype.fillText=function(value,...a){if(String(value).includes('秒复活'))previewLabels.push({text:String(value),x:a[0],y:a[1]});return text.call(this,value,...a);};
   let visual;Object.defineProperty(window,'QQTVisual',{get:()=>visual,set:v=>{visual=v;const create=v.createRenderer;v.createRenderer=(...args)=>{const r=create(...args),render=r.render;r.render=(s,...a)=>{window.previewSim=s;if(!s.__previewObserved){s.__previewObserved=true;const step=s.step;s.step=function(...a){const was=this.alive[0],info=step.apply(this,a);if(!was&&this.alive[0])window.arrival={point:Array.from(this.pos.slice(0,2)),tick:this.t};return info;};}return render(s,...a);};return r;};}});
  });
  await page.goto(base+'?seed=20261108&preview='+Date.now(),{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.previewSim?.t>0,{timeout:60000});
  if(process.env.EXPECTED_SHA)assert.equal(await page.evaluate(async()=>(await(await fetch('build-info.json?fresh='+Date.now(),{cache:'no-store'})).json()).commit),process.env.EXPECTED_SHA);
  const before=await page.evaluate(()=>{const s=previewSim;for(const k of ['fuse','wall','brick','cover','pushable','crate','fieldItem','blastLinger','brickLinger'])s[k].fill(0);s.pendingCrateType.fill(-1);s.owner.fill(-1);s.bunLoose.fill(0);s.airdropFalls=[];s.invuln.fill(1000);s.spawnProtection.fill(0);s.bombsCap[1]=0;s.spdG[1]=0;previewLabels=[];window.arrival=null;s._killPlayer(0);s._killPlayer(1);return{point:s.killRespawnPos[0].slice(),ticks:s.killRespawn[0],markers:QQTVisual.respawnMarkers(s,0),spectator:QQTVisual.respawnMarkers(s,-1)};});
  assert.equal(before.ticks,30);assert.deepEqual(before.markers.map(m=>m.pid),[0]);assert.deepEqual(before.spectator,[]);
  await page.waitForTimeout(200);const visible=await page.evaluate(()=>previewLabels.slice());assert(visible.length>0);assert(visible.every(x=>x.text.startsWith('自己')));assert(visible.every(x=>x.y<780));
  await page.screenshot({path:`.hermes/self-respawn-${publicCheck?'public':'local'}-${width}.png`});
  await page.evaluate(()=>{const p=previewSim.killRespawnPos[0];previewSim.blastLinger[Math.floor(p[0])*15+Math.floor(p[1])]=8;});
  await page.waitForFunction(old=>previewSim.killRespawnPos[0].some((x,i)=>x!==old[i]),before.point,{timeout:3000});
  const rerouted=await page.evaluate(()=>({point:previewSim.killRespawnPos[0].slice(),markers:QQTVisual.respawnMarkers(previewSim,0)}));assert.notDeepEqual(rerouted.point,before.point);
  await page.waitForFunction(()=>window.arrival,{timeout:6000});const after=await page.evaluate(()=>({arrival,point:previewSim.killRespawnPos[0].slice(),markers:QQTVisual.respawnMarkers(previewSim,0),status:document.querySelector('#status').textContent}));
  assert.deepEqual(after.arrival.point,after.point);assert.deepEqual(after.markers,[]);assert(!/killRespawnPos|bunSpawnPos|spawn_pos|respawn_pos/.test(after.status));assert.deepEqual(errors,[]);
  results.push({width,before,visible:visible.slice(0,2),rerouted,after:{arrival:after.arrival,point:after.point,markers:after.markers},errors});await context.close();console.log('self respawn PASS',width);
 }
 fs.writeFileSync(`.hermes/self-respawn-${publicCheck?'public':'local'}.json`,JSON.stringify({status:'PASS',sha:process.env.EXPECTED_SHA||null,scope:'actual Chromium desktop/mobile viewport; injected deaths and blast; not physical iOS',results},null,2)+'\n');
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
