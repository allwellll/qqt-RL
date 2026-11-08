'use strict';
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
const {createServer}=require('../web/server');
(async()=>{
 fs.mkdirSync('.hermes',{recursive:true});const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const base=process.env.ITEMS_PUBLIC_URL||`http://127.0.0.1:${server.address().port}/`,publicCheck=!!process.env.ITEMS_PUBLIC_URL;
 const opts={headless:true,executablePath:chromium.executablePath(),args:['--no-sandbox','--disable-gpu']};const raw=process.env.HTTPS_PROXY||process.env.https_proxy;if(raw){const u=new URL(raw);opts.proxy={server:u.protocol+'//'+u.host,bypass:'127.0.0.1,localhost'};}
 const browser=await chromium.launch(opts),results=[];
 try{for(const width of [1280,390]){
  const context=await browser.newContext({viewport:{width,height:900},hasTouch:width===390,isMobile:width===390});
  if(!publicCheck)await context.route('http://127.0.0.1:*/**',async r=>{const x=await fetch(r.request().url());await r.fulfill({status:x.status,headers:Object.fromEntries(x.headers),body:Buffer.from(await x.arrayBuffer())});});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{window.openings=[];let visual;Object.defineProperty(window,'QQTVisual',{get:()=>visual,set:v=>{visual=v;const create=v.createRenderer;v.createRenderer=(...a)=>{const r=create(...a),render=r.render;r.render=(s,...a)=>{window.itemsSim=s;if(!s.__openingCaptured){s.__openingCaptured=true;openings.push({seed:s.seed,kill:s.killMode,report:s.killOpeningItems,pos:Array.from(s.pos),crate:Array.from(s.crate),types:Array.from(s.crateType)});}return render(s,...a);};return r;};}});});
  await page.goto(base+'?seed=20261108&opening-items='+Date.now(),{waitUntil:'domcontentloaded',timeout:60000});await page.waitForFunction(()=>window.itemsSim?.t>0,null,{timeout:180000});
  if(process.env.EXPECTED_SHA)assert.equal(await page.evaluate(async()=>(await(await fetch('build-info.json?fresh='+Date.now(),{cache:'no-store'})).json()).commit),process.env.EXPECTED_SHA);
  const opening=await page.evaluate(()=>openings[0]);assert.equal(opening.report.actual,24);assert.equal(opening.crate.reduce((n,x)=>n+x,0),24);
  const counts=[0,0,0];for(const {cell,type}of opening.report.placements){counts[type]++;const mirror=Math.floor(cell/15)*15+14-cell%15;assert(opening.crate[mirror]);assert.equal(opening.types[mirror],type);}assert.deepEqual(counts,[8,8,8]);
  await page.screenshot({path:`.hermes/kill-opening-items-${publicCheck?'public':'local'}-${width}.png`});
  const pickups=await page.evaluate(()=>{
    const s=itemsSim,results=[];s.invuln.fill(1000);s.bombsCap[1]=0;s.spdG[1]=0;
    for(const type of [0,1,2]){
      const record=s.killOpeningItems.placements.find(x=>x.type===type&&s.crate[x.cell]);
      const field=['bombsCap','blastCap','spdG'][type],before=s[field][0];s.pos[0]=Math.floor(record.cell/15)+.5;s.pos[1]=record.cell%15+.5;
      const info=s.step([[4,0,0,1],[4,0,0,1]]);results.push({type,field,before,after:s[field][0],removed:!s.crate[record.cell],growth:info.growth?.[0]});
    }return results;
  });assert(pickups.every(x=>x.after>x.before&&x.removed));
  await page.selectOption('#game-mode','carry');await page.waitForFunction(()=>itemsSim.isBun&&!itemsSim.killMode);
  const carry=await page.evaluate(async()=>{const l=(await(await fetch('assets/maps/levels.json')).json()).find(x=>x.qqt_id===806),s=new QQT.Sim(20261108);s.reset(l,{nativeItems:true,nativeTrap:true,killMode:false,teams:[0,1],bananaSlideSpeedPx:480});const actual=openings.find(x=>!x.kill);return {hasOpening:!!itemsSim.killOpeningItems,sameCrates:JSON.stringify(actual.crate)===JSON.stringify(Array.from(s.crate)),sameTypes:JSON.stringify(actual.types)===JSON.stringify(Array.from(s.crateType))};});assert.deepEqual(carry,{hasOpening:false,sameCrates:true,sameTypes:true});
  assert.deepEqual(errors,[]);results.push({width,opening,counts,pickups,carry,errors});console.log(JSON.stringify({width,actual:opening.report.actual,counts,pickups,carry}));await context.close();
 }
 fs.writeFileSync(`.hermes/kill-opening-items-${publicCheck?'public':'local'}.json`,JSON.stringify({status:'PASS',sha:process.env.EXPECTED_SHA||null,scope:'real Chromium app; desktop/mobile viewport; test-only positioning for actual native pickup, not physical iOS',results},null,2)+'\n');
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
