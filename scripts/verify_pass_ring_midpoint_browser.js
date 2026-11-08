'use strict';
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');
const {createServer}=require('../web/server');
(async()=>{
 fs.mkdirSync('.hermes',{recursive:true});const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const base=process.env.RING_PUBLIC_URL||`http://127.0.0.1:${server.address().port}/`,publicCheck=!!process.env.RING_PUBLIC_URL;
 const opts={headless:true,executablePath:chromium.executablePath(),args:['--no-sandbox','--disable-gpu']};const raw=process.env.HTTPS_PROXY||process.env.https_proxy;if(raw){const u=new URL(raw);opts.proxy={server:u.protocol+'//'+u.host,bypass:'127.0.0.1,localhost'};}
 const browser=await chromium.launch(opts);
 try{
  const context=await browser.newContext({viewport:{width:900,height:900}});
  if(!publicCheck)await context.route('http://127.0.0.1:*/**',async r=>{const x=await fetch(r.request().url());await r.fulfill({status:x.status,headers:Object.fromEntries(x.headers),body:Buffer.from(await x.arrayBuffer())});});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  // Isolated real renderer/physics acceptance; no extra model download/evaluation.
  await context.route('**/__ring_acceptance',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><canvas id="game" width="900" height="880"></canvas><script src="sim-bun-native-v4-facing.js"></script><script src="visual_renderer.js"></script>'}));
  await page.goto(base+'__ring_acceptance',{waitUntil:'load'});
  if(process.env.EXPECTED_SHA)assert.equal(await page.evaluate(async()=>(await(await fetch('build-info.json?fresh='+Date.now(),{cache:'no-store'})).json()).commit),process.env.EXPECTED_SHA);
  const result=await page.evaluate(async()=>{
   const level=(await(await fetch('assets/maps/levels.json')).json()).find(x=>x.qqt_id===806);
   const assets=await QQTVisual.loadAssets(level),canvas=document.querySelector('#game'),renderer=QQTVisual.createRenderer(canvas,level,assets);
   const arc=CanvasRenderingContext2D.prototype.arc,arcs=[];
   CanvasRenderingContext2D.prototype.arc=function(x,y,r,start,end,...a){if(r===27&&this.lineWidth===5)arcs.push({x,y,progress:(end-start)/(2*Math.PI)});return arc.call(this,x,y,r,start,end,...a);};
   function scene(){const s=new QQT.Sim(77);s.reset('open',{nativeItems:true,nativeTrap:true});for(const k of ['wall','brick','cover','crate','fuse','pushable','fieldItem'])s[k].fill(0);s.pos.set([5.5,4.5,10.5,10.5]);s.invuln.fill(100);s.spawnProtection.fill(0);s.bombsCap[0]=5;s.spdG[0]=80/120;s.lastMoveDir[0]=QQT.MOVE_RIGHT;s.fuse[81]=30;s.owner[81]=1;s.frameStep(0,QQT.MOVE_IDLE,.016);return s;}
   function advance(s,ms){const target=s.nativeMove[0].touchStart+ms;while(s.nativeMove[0].clock<target-1e-6)s.frameStep(0,QQT.MOVE_IDLE,Math.min(.016,(target-s.nativeMove[0].clock)/1000));}
   const s=scene(),stages=[];
   for(const ms of [500,500.5,549,550,575,599]){
    advance(s,ms);arcs.length=0;renderer.render(s,performance.now(),{humanPid:0,prevPos:s.pos,curPos:s.pos,lastTickT:performance.now(),tickMs:100});
    stages.push({ms,hint:QQTVisual.bubblePassageHints(s,0)[0],drawn:arcs.at(-1)});
   }
   const physics=scene();advance(physics,500.5);const hint=QQTVisual.bubblePassageHints(physics,0)[0];const info=physics.step([[4,1,0,1],[4,0,0,1]]);const active=physics.nativeMove[0].passActive;
   for(let i=0;i<40;i++)physics.frameStep(0,QQT.MOVE_RIGHT,.05);
   return {stages,firstValidWindow:{hint,placed:info.placed[0],active,finalPosition:Array.from(physics.pos.slice(0,2))}};
  });
  for(const row of result.stages){assert(Math.abs(row.hint.progress-Math.min(1,row.ms/550))<1e-10);assert(Math.abs(row.drawn.progress-row.hint.progress)<1e-10);assert.equal(row.hint.ready,row.ms>500);}
  assert(result.firstValidWindow.placed&&result.firstValidWindow.active);assert(result.firstValidWindow.finalPosition[1]>=7);assert.deepEqual(errors,[]);
  await page.screenshot({path:`.hermes/pass-ring-midpoint-${publicCheck?'public':'local'}.png`});
  fs.writeFileSync(`.hermes/pass-ring-midpoint-${publicCheck?'public':'local'}.json`,JSON.stringify({status:'PASS',sha:process.env.EXPECTED_SHA||null,result,errors},null,2)+'\n');console.log(JSON.stringify({status:'PASS',sha:process.env.EXPECTED_SHA||null,stages:result.stages.map(x=>({ms:x.ms,progress:x.hint.progress,ready:x.hint.ready,drawn:x.drawn.progress})),firstValidWindow:result.firstValidWindow}));
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
