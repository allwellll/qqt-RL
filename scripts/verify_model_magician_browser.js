'use strict';
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
const {createServer}=require('../web/server');
(async()=>{
 const fixtureOnly=process.env.MAGICIAN_FIXTURE_ONLY==='1';
 const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const base=fixtureOnly?`http://127.0.0.1:${server.address().port}/`:'https://allwellll.github.io/qqt-RL/';
 const opts={headless:true,executablePath:chromium.executablePath(),args:['--no-sandbox','--disable-gpu']};
 const raw=process.env.HTTPS_PROXY||process.env.https_proxy;if(raw){const u=new URL(raw);opts.proxy={server:u.protocol+'//'+u.host,bypass:'localhost,127.0.0.1'};}
 const browser=await chromium.launch(opts);
 try{
  const context=await browser.newContext({viewport:{width:1280,height:900},hasTouch:true}),page=await context.newPage(),errors=[];
  if(fixtureOnly){
   await context.route('http://127.0.0.1:*/**',async route=>{const r=await fetch(route.request().url());await route.fulfill({status:r.status,headers:Object.fromEntries(r.headers),body:Buffer.from(await r.arrayBuffer())});});
   await context.route('**/releases/download/**',r=>r.abort());
  }
  page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{let visual;Object.defineProperty(window,'QQTVisual',{get:()=>visual,set:v=>{
   visual=v;const load=v.loadAssets,create=v.createRenderer;
   v.loadAssets=async(...args)=>{window.actorLevel=args[0];const a=await load(...args);window.actorAssets=a;return a;};
   window.createActorRenderer=create;
   v.createRenderer=(...args)=>{const r=create(...args),render=r.render;r.render=(s,t,m)=>{window.actorLive={sim:s,motion:m};return render(s,t,m);};return r;};
  }});});
  await page.goto(base+'?magician-check='+Date.now(),{waitUntil:'domcontentloaded',timeout:60000});
  await page.waitForFunction(()=>window.actorAssets?.characters.magician,null,{timeout:60000});
  if(!fixtureOnly)await page.waitForFunction(()=>window.actorLive?.sim.t>0&&document.querySelector('#loading').classList.contains('done'),null,{timeout:600000});
  let sha=null;if(process.env.EXPECTED_SHA){sha=await page.evaluate(async()=>(await(await fetch('build-info.json?fresh='+Date.now(),{cache:'no-store'})).json()).commit);assert.equal(sha,process.env.EXPECTED_SHA);}
  const fixtures=await page.evaluate(()=>{
   const rows=[],Q=QQT,A=actorAssets;
   const canvas=document.createElement('canvas');canvas.width=900;canvas.height=880;const ctx=canvas.getContext('2d'),draw=ctx.drawImage.bind(ctx);
   const tags=new Map();for(const key of ['maomao','pipi','magician'])for(const action of ['stand','walk'])A.characters[key][action].forEach((frames,row)=>frames.forEach((im,i)=>tags.set(im,`${key}:${action}:${row}:${i}`)));
   for(const key of ['trap','pop','protection'])A.effects[key].frames.forEach(im=>tags.set(im,key));
   let hits=[];ctx.drawImage=(im,...args)=>{if(tags.has(im))hits.push(tags.get(im));return draw(im,...args);};
   for(const killMode of [true,false]){
    const sim=new Q.Sim(20261108);sim.reset(actorLevel,{nativeItems:true,nativeTrap:true,killMode,teams:[0,1]});
    sim.pos.set([5.5,3.5,5.5,11.5]);sim.cover.fill(0);sim.invuln.fill(0);sim.spawnProtection.fill(0);
    const renderer=createActorRenderer(canvas,actorLevel,A),motion={characters:['maomao','magician'],teamRings:true,humanPid:0,prevPos:sim.pos,curPos:sim.pos,lastTickT:0,tickMs:100,intents:[Q.MOVE_IDLE,Q.MOVE_IDLE]};
    let now=1000;const directions=[];
    const render=()=>{hits=[];renderer.render(sim,now,motion);now+=200;return hits.slice();};
    for(const [move,row] of [[Q.MOVE_DOWN,0],[Q.MOVE_LEFT,1],[Q.MOVE_RIGHT,2],[Q.MOVE_UP,3]]){
     motion.intents[1]=move;sim.lastMoveDir[1]=move;const walk=render();motion.intents[1]=Q.MOVE_IDLE;const stand=render();
     directions.push({move,row,walk:walk.some(x=>x.startsWith(`magician:walk:${row}:`)),stand:stand.some(x=>x.startsWith(`magician:stand:${row}:`))});
    }
    motion.characters[0]='pipi';const humanPipi=render().some(x=>x.startsWith('pipi:'));motion.characters[0]='maomao';const humanMaomao=render().some(x=>x.startsWith('maomao:'));
    sim.trapped[1]=1;const trap=render().includes('trap');sim._trapStep(new Uint8Array(2));const deathHits=render(),dead=!sim.alive[1],hidden=!deathHits.some(x=>x.startsWith('magician:')),pop=deathHits.includes('pop');
    sim.lastDied.fill(false);
    for(let i=0;i<(killMode?30:sim.bunRespawnTicks);i++)killMode?sim._killRespawnStep():sim._bunRespawnStep();
    motion.prevPos=motion.curPos=sim.pos;const respawn=!!sim.alive[1]&&render().some(x=>x.startsWith('magician:'));
    rows.push({killMode,directions,humanPipi,humanMaomao,trap,dead,hidden,pop,respawn});
   }
   return rows;
  });
  console.log('render fixtures',JSON.stringify(fixtures));
  for(const f of fixtures){assert(f.directions.every(x=>x.walk&&x.stand));for(const field of ['humanPipi','humanMaomao','trap','dead','hidden','pop','respawn'])assert(f[field],`${f.killMode}:${field}`);}
  fs.mkdirSync('.hermes',{recursive:true});const screenshots=[];
  if(!fixtureOnly){
   assert.deepEqual(await page.locator('#character option').evaluateAll(rows=>rows.map(o=>o.value)),['pipi','maomao']);
   for(const mode of ['carry','kill']){
    await page.selectOption('#game-mode',mode);await page.waitForFunction(m=>window.actorLive.sim.killMode===(m==='kill'),mode);
    const live=await page.evaluate(()=>({characters:actorLive.motion.characters,teamRings:actorLive.motion.teamRings}));assert.deepEqual(live.characters,['maomao','magician']);assert(live.teamRings);
   }
   for(const [width,label]of [[1280,'desktop'],[390,'mobile']]){await page.setViewportSize({width,height:900});await page.waitForTimeout(150);const path=`.hermes/model-magician-public-${label}.png`;await page.screenshot({path});screenshots.push(path);}
   await page.selectOption('#character','pipi');await page.waitForFunction(()=>actorLive.motion.characters[0]==='pipi');assert.equal(await page.evaluate(()=>actorLive.motion.characters[1]),'magician');
  }
  assert.deepEqual(errors,[]);const result={status:'PASS',sha,fixtureOnly,fixtures,screenshots,errors,scope:'real Chromium; mobile viewport on the same running page; controlled native rendering/lifecycle fixtures, not physical iOS'};
  fs.writeFileSync(`.hermes/model-magician-${fixtureOnly?'fixtures':'public'}.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
