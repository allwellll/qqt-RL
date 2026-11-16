'use strict';
// CDP mobile Chromium, not physical iOS. Supabase requests are intercepted.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const staged=process.env.STICK_STAGED==='1',base='https://allwellll.github.io/qqt-RL/';
 const opts={headless:true,executablePath:chromium.executablePath(),args:['--no-sandbox','--disable-gpu']};
 const proxy=process.env.HTTPS_PROXY||process.env.https_proxy;if(proxy){const u=new URL(proxy);opts.proxy={server:u.protocol+'//'+u.host};}
 const browser=await chromium.launch(opts);
 try{
  const context=await browser.newContext({viewport:{width:390,height:900},hasTouch:true,isMobile:true}),page=await context.newPage(),errors=[];
  await context.route('**/*.supabase.co/**',r=>r.fulfill({json:[]}));
  if(staged)for(const file of ['controls.js','app.js','style.css'])await context.route(base+file,r=>r.fulfill({path:'web/'+file,contentType:file.endsWith('css')?'text/css':'application/javascript'}));
  page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{
   const raf=requestAnimationFrame.bind(window),paused=[];window.tapPause=false;
   window.requestAnimationFrame=callback=>raf(t=>{if(tapPause)paused.push(callback);else callback(t);});
   window.resumeTapFrames=()=>{tapPause=false;for(const cb of paused.splice(0))requestAnimationFrame(cb);};
   window.addEventListener('pointerdown',e=>{if(e.target.closest?.('[data-floating-direction]'))window.lastStickPointer=e.pointerId;},true);
   window.stickFrames=[];window.stickPresses=[];window.stickSteps=[];window.stickPending=0;window.stickDelay=0;
   let controls,visual,worker;
   Object.defineProperty(window,'QQTControls',{get:()=>controls,set:v=>{controls=v;const bind=v.bindTouchControls;v.bindTouchControls=(root,held,press)=>{
    window.stickHeld=held;const clear=bind(root,held,k=>{stickPresses.push(k);press(k);}),choose=clear.moveFor;
    clear.moveFor=(s,pid,now)=>{const start=performance.now(),mv=choose(s,pid,now);stickFrames.push({now,cost:performance.now()-start,move:mv,pos:Array.from(s.pos.slice(0,2)),pending:stickPending});if(stickFrames.length>500)stickFrames.shift();return mv;};return clear;
   };}});
   Object.defineProperty(window,'QQTVisual',{get:()=>visual,set:v=>{visual=v;const create=v.createRenderer;v.createRenderer=(...args)=>{const r=create(...args),render=r.render;r.render=(s,...a)=>{window.stickSim=s;if(!s.__stickProbe){s.__stickProbe=true;const step=s.step;s.step=function(actions){const result=step.call(this,actions);if(result.placed[0]||result.itemReleased[0])stickSteps.push({placed:result.placed[0],item:result.itemReleased[0]});return result;};}return render(s,...a);};return r;};}});
   Object.defineProperty(window,'QQTModelWorker',{get:()=>worker,set:v=>{worker=v;const load=v.load;v.load=async(...args)=>{const m=await load(...args),forward=m.forward;m.forward=async(...a)=>{stickPending++;try{if(stickDelay)await new Promise(r=>setTimeout(r,stickDelay));return await forward(...a);}finally{stickPending--;}};return m;};}});
  });
  await page.goto(base+'?radial-stick='+Date.now(),{waitUntil:'domcontentloaded',timeout:60000});
  const watch=setInterval(async()=>{try{console.log('loading',await page.locator('#loading').innerText());}catch{}},30000);watch.unref();
  await page.waitForFunction(()=>window.stickSim?.t>0&&document.querySelector('#loading').classList.contains('done'),null,{timeout:600000});
  clearInterval(watch);
  if(process.env.EXPECTED_SHA)assert.equal(await page.evaluate(async()=>(await(await fetch('build-info.json?fresh='+Date.now(),{cache:'no-store'})).json()).commit),process.env.EXPECTED_SHA);
  const cdp=await context.newCDPSession(page),touch=async(type,points=[])=>{await cdp.send('Input.dispatchTouchEvent',{type,touchPoints:points});};
  const state=()=>page.evaluate(()=>{const z=document.querySelector('[data-floating-direction]');return {direction:z.dataset.direction,secondary:z.dataset.secondary,resolved:z.dataset.resolved,active:z.dataset.active,held:[...stickHeld],pos:Array.from(stickSim.pos.slice(0,2)),presses:stickPresses.slice(),steps:stickSteps.slice(),slots:stickSim.itemSlots[0].map(x=>({...x}))};});
  const fixture=async()=>{await page.evaluate(()=>{dispatchEvent(new Event('blur'));const s=stickSim;for(const k of ['wall','brick','cover','pushable','crate','fuse','fieldItem','blastLinger','brickLinger'])s[k].fill(0);s.owner.fill(-1);s.done=false;s.alive.fill(1);s.trapped.fill(0);s.invuln.fill(1000);s.spawnProtection.fill(0);s.movementStatus.fill(0);s.bunCarried.fill(-1);s.pos.set([6.5,4.5,11.5,13.5]);s.nativeMove=null;s.spdG[0]=1.3;s.spdG[1]=0;s.bombsCap[0]=10;s.bombsCap[1]=0;s.itemSlots[0]=[{item:1,count:3},{item:2,count:4},{item:1,count:5}];s._syncHeldItem(0);stickPresses=[];stickSteps=[];stickFrames=[];});await page.waitForTimeout(40);};
  const layout=await page.evaluate(()=>{const rect=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};};return {zone:rect(document.querySelector('[data-floating-direction]')),dial:rect(document.querySelector('.touch-stick')),bomb:rect(document.querySelector('.touch-bomb-hit')),items:[...document.querySelectorAll('[data-touch-item]')].map(rect)};});
  const start={id:1,x:layout.zone.x+layout.zone.w*.5,y:layout.zone.y+layout.zone.h*.5},right={...start,x:start.x+30};
  const tapCases=[];
  for(const [dx,dy,mv,name]of [[0,-30,0,'ArrowUp'],[0,30,1,'ArrowDown'],[-30,0,2,'ArrowLeft'],[30,0,3,'ArrowRight']]){
   await fixture();const p={...start,x:start.x+dx,y:start.y+dy};await touch('touchStart',[p]);
   assert.equal((await state()).direction,name,'pointerdown immediately selects a visible sector');
   await page.waitForTimeout(60);const moving=await state();await touch('touchEnd');await page.waitForTimeout(40);const stopped=await state();await page.waitForTimeout(50);
   assert.deepEqual((await state()).pos,stopped.pos,'release does not leave continuous movement');
   assert((mv<2?moving.pos[0]!==6.5:moving.pos[1]!==4.5),'held sector actually moves');tapCases.push(name);
  }
  await fixture();const beforeShort=(await state()).pos;await page.evaluate(()=>{tapPause=true;});await page.waitForTimeout(40);
  await touch('touchStart',[right]);assert.equal((await state()).direction,'ArrowRight');await touch('touchEnd');
  assert.deepEqual((await state()).pos,beforeShort,'down/up occurs with game rAF paused');await page.evaluate(()=>resumeTapFrames());await page.waitForTimeout(50);
  const shortTap=await state();assert(shortTap.pos[1]>beforeShort[1]&&shortTap.pos[1]-beforeShort[1]<.15,'quick tap commits only a bounded human frame');await page.waitForTimeout(80);assert.deepEqual((await state()).pos,shortTap.pos,'quick tap cannot auto-walk');
  await fixture();await touch('touchStart',[{...start,x:layout.zone.x+2}]);assert.equal((await state()).direction,'','edge guard rejects accidental contact');await touch('touchEnd');
  await fixture();await touch('touchStart',[start]);assert.equal((await state()).direction,'');
  await touch('touchMove',[{...start,x:start.x+3}]);assert.equal((await state()).direction,'');
  await touch('touchMove',[right]);assert.equal((await state()).direction,'ArrowRight');
  await touch('touchMove',[{...right,y:right.y+4}]);assert.equal((await state()).direction,'ArrowRight','radial jitter regression');
  await touch('touchMove',[start]);assert.equal((await state()).direction,'','centre stops instead of reversing');
  await touch('touchMove',[{...start,x:start.x-30}]);assert.equal((await state()).direction,'ArrowLeft','opposite radial input is immediate');
  await touch('touchEnd');
  await fixture();await touch('touchStart',[start]);await touch('touchMove',[{...start,x:start.x+80}]);
  const followed=await page.locator('.touch-stick').boundingBox();assert(Math.abs(followed.x+followed.width/2-(start.x+80-followed.width/2))<2,'overflow moves the real radial centre');await touch('touchEnd');
  await fixture();await page.evaluate(()=>{const s=stickSim;s.pos[1]=6.2;for(let c=0;c<15;c++)if(c!==6)s.wall[5*15+c]=1;});
  await touch('touchStart',[start]);await touch('touchMove',[right]);await page.waitForTimeout(20);await touch('touchMove',[{...right,y:right.y-18}]);
  await page.waitForTimeout(120);const corner=await state();assert.equal(corner.secondary,'ArrowUp');assert(corner.pos[0]<6.5,'diagonal input actually turns through native junction');await touch('touchEnd');
  await fixture();await page.evaluate(()=>{stickDelay=300;});
  await touch('touchStart',[start]);await touch('touchMove',[right]);await page.waitForTimeout(750);
  const responsiveness=await page.evaluate(()=>{const f=stickFrames,delta=f.slice(1).map((x,i)=>x.now-f[i].now).sort((a,b)=>a-b),cost=f.map(x=>x.cost).sort((a,b)=>a-b),pending=f.filter(x=>x.pending);return {frames:f.length,pendingFrames:pending.length,pendingTravel:pending.length?pending.at(-1).pos[1]-pending[0].pos[1]:0,p95FrameMs:delta[Math.floor(delta.length*.95)],p95InputMs:cost[Math.floor(cost.length*.95)]};});
  console.log('responsiveness',JSON.stringify(responsiveness));
  assert(responsiveness.pendingFrames>=8&&responsiveness.pendingTravel>0,'rAF keeps human moving during delayed model inference');
  const b=layout.bomb,bomb={id:2,x:b.x+b.w/2,y:b.y+b.h/2};await touch('touchStart',[right,bomb]);await page.waitForTimeout(700);
  assert.equal((await state()).direction,'ArrowRight');assert((await state()).steps.some(x=>x.placed),'real bomb placement with a second finger');
  await touch('touchEnd',[right]);await touch('touchEnd');await page.evaluate(()=>{stickDelay=0;});
  const itemCases=[];for(let i=0;i<3;i++){
   await fixture();await touch('touchStart',[start]);await touch('touchMove',[right]);
   const r=layout.items[i],item={id:2,x:r.x+r.w/2,y:r.y+r.h/2};await touch('touchStart',[right,item]);await page.waitForTimeout(220);
   const used=await state();assert.deepEqual(used.presses,['Digit'+(i+1)]);assert.equal(used.slots[i].count,[3,4,5][i]-1);for(let j=0;j<3;j++)if(j!==i)assert.equal(used.slots[j].count,[3,4,5][j]);assert(used.steps.some(x=>x.item));assert.equal(used.direction,'ArrowRight');itemCases.push(i+1);await touch('touchCancel');
  }
  for(const cleanup of ['pointercancel','blur','lostpointercapture','resize']){
   await fixture();await touch('touchStart',[start]);await touch('touchMove',[{...right,y:right.y-18}]);
   if(cleanup==='pointercancel')await touch('touchCancel');else {await page.evaluate(name=>{if(name==='lostpointercapture'){const z=document.querySelector('[data-floating-direction]');z.releasePointerCapture(window.lastStickPointer);}else dispatchEvent(new Event(name));},cleanup);}
   if(cleanup==='lostpointercapture')await touch('touchMove',[{...right,y:right.y-17}]);
   const clean=await state();assert.equal(clean.direction,'');assert.equal(clean.secondary,'');assert(!clean.held.some(k=>k.startsWith('touch:')));if(cleanup!=='pointercancel')await touch('touchEnd');
  }
  await fixture();await page.keyboard.down('Numpad5');assert.equal(await page.evaluate(()=>QQTControls.moveForHeld(stickHeld)),0);await page.keyboard.up('Numpad5');assert.deepEqual((await state()).presses,[]);
  await page.evaluate(()=>{const e=document.createElement('input');e.id='stick-typing';document.body.append(e);});await page.locator('#stick-typing').fill('abc');await page.keyboard.press('ArrowRight');assert.equal(await page.evaluate(()=>QQTControls.moveForHeld(stickHeld)),4);
  await page.locator('#stick-typing').evaluate(e=>e.remove());await page.evaluate(()=>scrollTo(0,0));await page.waitForTimeout(50);
  await fixture();await touch('touchStart',[start]);await touch('touchMove',[right]);await page.waitForTimeout(100);
  assert.equal(await page.evaluate(()=>getSelection().toString()),'');assert(await page.locator('[data-floating-direction]').evaluate(e=>!e.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true}))));
  assert.equal((await state()).direction,'ArrowRight');
  fs.mkdirSync('.hermes',{recursive:true});await page.screenshot({path:`.hermes/mobile-stick-${staged?'staged':'public'}.png`});await touch('touchEnd');
  assert.deepEqual(errors,[]);const result={status:'PASS',sha:process.env.EXPECTED_SHA||null,staged,layout,tapCases,shortTap,responsiveness,corner,itemCases,cleanup:true,typing:true,numpad:true,errors,scope:'390x900 mobile Chromium/CDP; controlled live native scene; artificial 300ms inference delay; no physical iPhone claim; Supabase intercepted'};
  fs.writeFileSync(`.hermes/mobile-stick-${staged?'staged':'public'}.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
