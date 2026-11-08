'use strict';
// Minimal real Chromium/CDP touch acceptance; not a physical iOS/Safari test.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs'), assert = require('node:assert/strict');
const {createServer} = require('../web/server');
(async () => {
  fs.mkdirSync('.hermes', {recursive:true});
  const server = createServer(); await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = process.env.MOBILE_PUBLIC_URL || `http://127.0.0.1:${server.address().port}/`;
  const tag = process.env.MOBILE_PUBLIC_URL ? 'public' : 'local';
  const opts = {headless:true, executablePath:chromium.executablePath(), args:['--no-sandbox', '--disable-gpu']};
  const proxy = process.env.HTTPS_PROXY || process.env.https_proxy;
  if (proxy) { const u = new URL(proxy); opts.proxy = {server:u.protocol+'//'+u.host, bypass:'127.0.0.1,localhost'}; }
  let browser;
  const results = [];
  try {
    browser = await chromium.launch(opts);
    for (const viewport of [{width:390,height:900}, {width:844,height:390}]) {
      const context = await browser.newContext({viewport, hasTouch:true, isMobile:true});
      const page = await context.newPage(), errors = [];
      page.on('pageerror', e => errors.push(e.message));
      if (tag === 'local') await context.route('http://127.0.0.1:*/**', async r => {
        const x = await fetch(r.request().url());
        await r.fulfill({status:x.status, headers:Object.fromEntries(x.headers), body:Buffer.from(await x.arrayBuffer())});
      });
      await page.addInitScript(() => {
        window.touchPresses = []; window.touchEvents = []; window.pointerTrace=[]; for(const name of ['pointerdown','pointerup','pointermove','lostpointercapture','gotpointercapture','pointercancel'])window.addEventListener(name,e=>pointerTrace.push({name,id:e.pointerId,target:e.target.className}),true);
        let controls, visual;
        Object.defineProperty(window, 'QQTControls', {get:()=>controls, set:v=>{
          controls=v; const bind=v.bindTouchControls;
          v.bindTouchControls=(root,held,press)=>{window.touchHeld=held; root.querySelector('[data-floating-direction]').addEventListener('pointerdown',e=>window.touchDirectionPointer=e.pointerId); return bind(root,held,key=>{touchPresses.push(key);press(key);});};
        }});
        Object.defineProperty(window, 'QQTVisual', {get:()=>visual, set:v=>{
          visual=v;const create=v.createRenderer;
          v.createRenderer=(...args)=>{const renderer=create(...args), render=renderer.render;
            renderer.render=(s,...a)=>{window.touchSim=s;if(!s.__touchObserved){s.__touchObserved=true;const step=s.step;
              s.step=function(actions){const info=step.call(this,actions);if(info.placed[0]||info.itemReleased[0])touchEvents.push({placed:info.placed[0],released:info.itemReleased[0],action:actions[0].slice()});return info;};}
              return render(s,...a);};return renderer;};
        }});
      });
      await page.goto(base+'?touch-tolerance='+Date.now(), {waitUntil:'domcontentloaded'});
      await page.waitForFunction(()=>window.touchSim && document.querySelector('#loading').classList.contains('done'), {timeout:60000});
      if (process.env.EXPECTED_SHA) assert.equal(await page.evaluate(async()=>(await(await fetch('build-info.json?fresh='+Date.now(),{cache:'no-store'})).json()).commit),process.env.EXPECTED_SHA);
      const cdp = await context.newCDPSession(page);
      const touch = async (type, points=[]) => {await cdp.send('Input.dispatchTouchEvent', {type,touchPoints:points}); await page.waitForTimeout(35);};
      const state = () => page.evaluate(()=>({held:[...touchHeld],move:QQTControls.moveForHeld(touchHeld),pos:Array.from(touchSim.pos.slice(0,2)),presses:touchPresses.slice(),events:touchEvents.slice(),slots:touchSim.itemSlots[0].map(x=>({...x})),direction:document.querySelector('[data-floating-direction]').dataset.direction}));
      const fixture = async (slots=[{item:1,count:3},{item:2,count:5},{item:1,count:7}]) => {
        await page.evaluate(slots=>{
          window.dispatchEvent(new Event('blur'));
          const s=touchSim;
          for(const k of ['wall','brick','cover','pushable','crate','fuse','fieldItem','blastLinger','brickLinger'])s[k].fill(0);
          s.owner.fill(-1);s.done=false;s.winner=null;s.isBun=false;s.alive.fill(1);s.trapped.fill(0);s.invuln.fill(1000);s.spawnProtection.fill(0);s.bunCarried.fill(-1);s.movementStatus.fill(0);
          s.pos.set([5.5,2.5,11.5,13.5]);s.spdG[0]=1.3;s.spdG[1]=0;s.bombsCap[0]=10;s.blastCap[0]=1;s.nativeMove=[];s.itemSlots[0]=slots;s._syncHeldItem(0);touchPresses=[];touchEvents=[];
        },slots);
        await page.waitForTimeout(80);
      };
      await fixture();
      const layout = await page.evaluate(()=>{
        const rect = e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};};
        const zone=document.querySelector('[data-floating-direction]'),bomb=document.querySelector('.touch-bomb-hit');
        return {zone:rect(zone),bombHit:rect(bomb),bomb:rect(bomb.querySelector('button')),items:[...document.querySelectorAll('[data-touch-item]')].map(rect),overflow:document.documentElement.scrollWidth>innerWidth,deadzone:document.querySelector('.touch-stick').getBoundingClientRect().width/2*10/48};
      });
      assert(!layout.overflow); assert(layout.bomb.width>=88);assert(Math.abs(layout.bombHit.width/layout.bomb.width-1.25)<.01);
      assert(layout.items.every(r=>r.width>=48&&r.x>=0&&r.x+r.width<=viewport.width));
      assert.equal(new Set(layout.items.map(r=>r.y)).size,1,'three slots share one row');
      assert(layout.bombHit.y+layout.bombHit.height<layout.items[0].y,'item row is outside extended bomb target');
      assert(Math.abs(layout.zone.width/viewport.width-.45)<.01);assert(Math.abs(layout.zone.height/viewport.height-.6)<.01);
      const start={x:layout.zone.x+layout.zone.width*.3,y:layout.zone.y+layout.zone.height*.6,id:1};
      const moved={...start,x:start.x+32};
      await touch('touchStart',[start]);assert.equal((await state()).move,4,'off-center down defines a new center');
      await touch('touchMove',[{...start,x:start.x+8}]);assert.equal((await state()).move,4,'deadzone');
      await touch('touchMove',[moved]);assert.equal((await state()).move,3);
      await touch('touchMove',[{...start,x:start.x+20,y:start.y+25}]);assert.equal((await state()).move,3,'51-degree diagonal keeps right');
      await touch('touchMove',[{...start,x:start.x+10,y:start.y+30}]);assert.equal((await state()).move,1,'72-degree diagonal switches down');
      await touch('touchEnd'); await touch('touchStart',[start]); const resumed=await state(); assert.equal(resumed.move,1,'short tap resumes down');
      await touch('touchEnd');await page.waitForTimeout(300);await touch('touchStart',[start]);assert.equal((await state()).move,4,'late tap idles');
      const outside={...start,x:layout.zone.x+layout.zone.width+80};
      await touch('touchMove',[outside]);const beforeHold=await state();assert.equal(beforeHold.move,3);
      await page.waitForTimeout(850);const longHold=await state();assert.equal(longHold.move,3);assert(longHold.pos[1]>beforeHold.pos[1]);assert.deepEqual(longHold.presses,[],'captured movement does not hit action keys');
      assert.equal(await page.evaluate(()=>getSelection().toString()),'');
      assert(await page.locator('[data-floating-direction]').evaluate(e=>!e.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true}))));
      await page.screenshot({path:`.hermes/touch-tolerance-${tag}-${viewport.width}.png`});
      await touch('touchCancel');assert.equal((await state()).move,4);
      await fixture();await touch('touchStart',[start]);await touch('touchMove',[moved]);
      // Hit the enlarged ring, outside the visible bubble circle.
      const bomb={x:layout.bombHit.x+2,y:layout.bombHit.y+layout.bombHit.height/2,id:2};
      await touch('touchStart',[moved,bomb]);await page.waitForTimeout(180);const multiBomb=await state();
      assert.equal(multiBomb.move,3);assert.equal(multiBomb.presses.filter(k=>k==='Space').length,1);assert(multiBomb.events.some(e=>e.placed));
      await touch('touchEnd',[bomb]);assert.equal((await state()).move,3);await touch('touchEnd');
      const itemCases=[];
      for(let i=0;i<3;i++){
        await fixture();await touch('touchStart',[start]);await touch('touchMove',[moved]);
        const r=layout.items[i],item={x:r.x+r.width/2,y:r.y+r.height/2,id:2};
        await touch('touchStart',[moved,item]);await page.waitForTimeout(170);const used=await state();
        assert.deepEqual(used.presses,['Digit'+(i+1)]);assert.equal(used.slots[i].count,[3,5,7][i]-1);
        for(let j=0;j<3;j++)if(i!==j)assert.equal(used.slots[j].count,[3,5,7][j]);
        assert.equal(used.events.filter(e=>e.released).length,1);assert(!used.events.some(e=>e.placed));
        await touch('touchEnd',[item]);assert.equal((await state()).move,3);await touch('touchEnd');itemCases.push(used);
      }
      await fixture([]);const r=layout.items[0];await touch('touchStart',[{x:r.x+r.width/2,y:r.y+r.height/2,id:1}]);await touch('touchEnd');assert.deepEqual((await state()).presses,[],'empty slot never dispatches an item');
      const seven=Array.from({length:7},(_,i)=>({item:i%2+1,count:i+2}));await fixture(seven);
      await page.locator('#touch-item-page').tap();await page.locator('#touch-item-page').tap();
      assert.equal(await page.locator('[data-touch-item="0"]').getAttribute('data-control-key'),'Digit7');
      await page.locator('[data-touch-item="0"]').tap();await page.waitForTimeout(160);const seventh=await state();assert.equal(seventh.slots[6].count,7);assert.deepEqual(seventh.slots.slice(0,6),seven.slice(0,6));
      for(const cleanup of ['lostpointercapture','blur','visibilitychange']){
        await fixture();await touch('touchStart',[start]);await touch('touchMove',[moved]);assert.equal((await state()).move,3);
        await page.evaluate(name=>{
          if(name==='lostpointercapture')document.querySelector('[data-floating-direction]').releasePointerCapture(window.touchDirectionPointer);
          else if(name==='blur')window.dispatchEvent(new Event(name));
          else {Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event(name));delete document.hidden;}
        },cleanup);
        // Chromium flushes the pending lost-capture notification on the next pointer event.
        if(cleanup==='lostpointercapture')await touch('touchMove',[{...moved,x:moved.x+1}]);
        assert.equal((await state()).move,4,cleanup+' clears direction');await touch('touchEnd');
      }
      await fixture();await page.keyboard.down('ArrowRight');await touch('touchStart',[start]);await touch('touchMove',[moved]);await touch('touchEnd');assert.equal((await state()).move,3,'touch release preserves held keyboard');await page.keyboard.up('ArrowRight');assert.equal((await state()).move,4);
      for(const [code,key] of [['Numpad5','Clear'],['Numpad1','End'],['Numpad2','ArrowDown'],['Numpad3','PageDown']]){
        await cdp.send('Input.dispatchKeyEvent',{type:'keyDown',code,key,location:3});assert.notEqual((await state()).move,4);await cdp.send('Input.dispatchKeyEvent',{type:'keyUp',code,key,location:3});assert.equal((await state()).move,4);
      }
      assert.deepEqual((await state()).slots,[{item:1,count:3},{item:2,count:5},{item:1,count:7}]);
      await page.evaluate(()=>{const e=document.createElement('input');e.id='touch-typing-probe';document.body.append(e);});
      await page.locator('#touch-typing-probe').fill('昵称');await page.locator('#touch-typing-probe').focus();await page.keyboard.type('abc');await page.keyboard.press('ArrowRight');assert.equal((await state()).move,4);assert.equal(await page.locator('#touch-typing-probe').inputValue(),'昵称abc');
      assert.deepEqual(errors,[]);results.push({viewport,layout,resumed,longHold,multiBomb,itemCases,seventh,cleanup:true,keyboardIsolation:true,typing:true,errors});
      await context.close();console.log('touch tolerance PASS',viewport.width,viewport.height);
    }
    fs.writeFileSync(`.hermes/touch-tolerance-${tag}.json`,JSON.stringify({status:'PASS',scope:'Chromium CDP touch, not physical iOS',sha:process.env.EXPECTED_SHA||null,results},null,2)+'\n');
  } finally {if(browser)await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
