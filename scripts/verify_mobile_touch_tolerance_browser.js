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
    for (const viewport of [{width:390,height:900}, {width:844,height:390}, {width:568,height:320}]) {
      const context = await browser.newContext({viewport, hasTouch:true, isMobile:true});
      const page = await context.newPage(), errors = [];
      page.on('pageerror', e => errors.push(e.message));
      await context.route('**/*.supabase.co/**', r => r.fulfill({json:[]}));
      if (tag === 'local') await context.route('http://127.0.0.1:*/**', async r => {
        const x = await fetch(r.request().url());
        await r.fulfill({status:x.status, headers:Object.fromEntries(x.headers), body:Buffer.from(await x.arrayBuffer())});
      });
      await page.addInitScript(() => {
        window.touchPresses = []; window.touchEvents = []; window.pointerTrace=[]; for(const name of ['pointerdown','pointerup','pointermove','lostpointercapture','gotpointercapture','pointercancel'])window.addEventListener(name,e=>pointerTrace.push({name,id:e.pointerId,target:e.target.className,x:e.clientX,y:e.clientY,time:e.timeStamp}),true);
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
      await page.waitForFunction(()=>window.touchSim && document.querySelector('#loading').classList.contains('done'), null, {timeout:600000});
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
        return {canvas:rect(document.querySelector('#game')),toolbar:rect(document.querySelector('.game-toolbar')),zone:rect(zone),bombHit:rect(bomb),bomb:rect(bomb.querySelector('button')),items:[...document.querySelectorAll('[data-touch-item]')].map(rect),overflow:document.documentElement.scrollWidth>innerWidth,deadzone:document.querySelector('.touch-stick').getBoundingClientRect().width*.09};
      });
      assert(!layout.overflow); assert(layout.bomb.width>=88);assert(Math.abs(layout.bombHit.width/layout.bomb.width-1.25)<.01);
      assert(layout.items.every(r=>r.width>=48&&r.x>=0&&r.x+r.width<=viewport.width));
      assert.equal(new Set(layout.items.map(r=>r.y)).size,1,'three slots share one row');
      assert(layout.bombHit.y+layout.bombHit.height<layout.items[0].y,'item row is outside extended bomb target');
      const overlaps=(a,b)=>a.x<b.x+b.width && a.x+a.width>b.x && a.y<b.y+b.height && a.y+a.height>b.y;
      assert(Math.abs(layout.canvas.x+layout.canvas.width/2-viewport.width/2)<1,'board centered horizontally');
      for(const r of [layout.zone,layout.bombHit,...layout.items,layout.toolbar]) {
        assert(!overlaps(layout.canvas,r),'controls do not cover the board');
        assert(r.y>=0 && r.y+r.height<=viewport.height,'controls fit the screen');
      }
      assert(!overlaps(layout.toolbar,layout.bombHit),'toolbar does not cover action targets');
      const start={x:layout.zone.x+layout.zone.width*.5,y:layout.zone.y+layout.zone.height*.5,id:1};
      const moved={...start,x:start.x+32};
      await touch('touchStart',[start]);assert.equal((await state()).move,4,'centre down starts a neutral floating drag');
      await touch('touchMove',[{...start,x:start.x+3}]);assert.equal((await state()).move,4,'3px stays below activation distance');
      await touch('touchMove',[{...start,x:start.x+layout.deadzone+1}]);assert.equal((await state()).move,3,'18% radial deadzone activates');
      await touch('touchMove',[moved]);assert.equal((await state()).move,3);
      const low={...start,y:start.y+40};
      await touch('touchMove',[low]);assert.equal((await state()).move,1,'radial down turns down');
      await touch('touchMove',[{...low,x:low.x+2}]);assert.equal((await state()).move,1,'2px sideways jitter retains down');
      await touch('touchMove',[moved]);assert.equal((await state()).move,3,'radial right turns right');
      await touch('touchMove',[{...start,x:start.x-32}]);assert.equal((await state()).move,2,'radial reversal turns left');
      await touch('touchMove',[{...start,y:start.y-40}]);assert.equal((await state()).move,0,'radial up turns up');
      await page.waitForTimeout(100);assert.equal((await state()).move,0,'stationary finger keeps current direction');
      await touch('touchEnd'); await touch('touchStart',[start]); const resumed=await state(); assert.equal(resumed.move,4,'short tap starts idle without restoring the previous direction');
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
      // Exercise browser-generated rapid double taps on both board and screen actions.
      for(let i=0;i<6;i++) await page.touchscreen.tap(layout.canvas.x+layout.canvas.width/2,layout.canvas.y+layout.canvas.height/2);
      assert.equal(await page.evaluate(()=>visualViewport.scale),1,'rapid taps do not zoom the viewport');
      const cancellation=await page.evaluate(()=>{
        const stage=document.querySelector('.stage'),canvas=document.querySelector('#game');
        const probe=(target,type)=>{
          const e=new Event(type,{bubbles:true,cancelable:true});
          Object.defineProperties(e,{changedTouches:{value:[{identifier:99,target,clientX:20,clientY:80}]},touches:{value:type==='touchend'?[]:[{identifier:99,target,clientX:20,clientY:80}]}});
          target.dispatchEvent(e);return e.defaultPrevented;
        };
        const result={};
        for(const [name,target] of [['blank',stage],['board',canvas]])result[name]=['touchstart','touchmove','touchend'].map(type=>probe(target,type));
        result.safariPinch=['gesturestart','gesturechange','gestureend'].map(type=>probe(stage,type));
        result.form=probe(document.querySelector('#player-nickname'),'touchstart');
        result.aside=probe(document.querySelector('aside'),'touchmove');
        return result;
      });
      assert.deepEqual(cancellation,{blank:[true,true,true],board:[true,true,true],safariPinch:[true,true,true],form:false,aside:false},'non-passive guards cancel Safari gestures only in gameplay');
      await page.locator('#game-fullscreen').tap();
      await page.waitForFunction(()=>document.querySelector('#game-fullscreen').getAttribute('aria-pressed')==='true');
      const nativeFullscreen=await page.evaluate(()=>document.fullscreenElement===document.querySelector('.stage'));
      if(nativeFullscreen) {
        await page.evaluate(()=>document.exitFullscreen());
        await page.waitForFunction(()=>!document.fullscreenElement);
        assert(await page.locator('.stage').evaluate(e=>e.classList.contains('game-immersive')),'browser escape retains immersion and scroll lock');
      }
      await page.locator('#game-fullscreen').tap();
      await page.waitForFunction(()=>!document.querySelector('.stage').classList.contains('game-immersive'));
      assert.equal(await page.evaluate(()=>document.body.classList.contains('game-immersive')),false);
      // iOS/unavailable and permission-denied fullscreen must both leave a usable exit.
      for(const denied of [false,true]) {
        await page.evaluate(denied=>{const stage=document.querySelector('.stage');stage.webkitRequestFullscreen=undefined;stage.requestFullscreen=denied?()=>Promise.reject(Error('Denied')):undefined;},denied);
        await page.locator('#game-fullscreen').tap();
        await page.waitForFunction(()=>document.querySelector('.stage').classList.contains('game-immersive'));
        assert(await page.locator('#game-screen-status').textContent().then(s=>s.includes('沉浸模式')));
        const screen=await page.locator('.stage').boundingBox();assert.equal(screen.width,viewport.width);assert.equal(screen.height,viewport.height);
        assert.equal(await page.evaluate(()=>getComputedStyle(document.body).position),'fixed','body scroll locked');
        assert.equal(await page.evaluate(()=>getComputedStyle(document.documentElement).overflowY),'hidden','root scroll locked');
        const beforeScroll=await page.evaluate(()=>scrollY);
        // Swipe both directions across empty stage space, then pinch and rapidly tap.
        const blank={id:7,x:8,y:75};
        for(const down of [true,false]) {
          await touch('touchStart',[{...blank,y:down?75:200}]);
          for(let i=1;i<=5;i++)await touch('touchMove',[{...blank,y:down?75+i*25:200-i*25}]);
          await touch('touchEnd');
        }
        await touch('touchStart',[{id:7,x:viewport.width/2-15,y:80},{id:8,x:viewport.width/2+15,y:80}]);
        await touch('touchMove',[{id:7,x:viewport.width/2-70,y:110},{id:8,x:viewport.width/2+70,y:110}]);await touch('touchEnd');
        for(let i=0;i<8;i++)await page.touchscreen.tap(viewport.width/2,80);
        assert.equal(await page.evaluate(()=>visualViewport.scale),1,'swipes, pinch and repeated taps cannot zoom');
        assert.equal(await page.evaluate(()=>scrollY),beforeScroll,'swipes cannot scroll the underlying page');
        assert(await page.locator('.stage').evaluate(e=>e.classList.contains('game-immersive')),'swipes cannot leave immersion');
        // End the match, then restart while still immersed; hidden pad must not hide toolbar.
        await page.evaluate(()=>{window.finishedTouchSim=touchSim;touchSim.done=true;touchSim.winner=null;});
        await page.waitForFunction(()=>document.querySelector('#touch-controls').hidden);
        await page.locator('#touch-restart').tap();
        await page.waitForFunction(()=>touchSim!==finishedTouchSim && !touchSim.done && !document.querySelector('#touch-controls').hidden);
        assert(await page.locator('.stage').evaluate(e=>e.classList.contains('game-immersive')),'restart preserves immersion');
        await page.locator('#game-fullscreen').tap();
        assert.equal(await page.locator('#game-fullscreen').getAttribute('aria-pressed'),'false');
      }
      // Exercise a rotation during immersion; pointer state clears and the board stays centered.
      await fixture();await touch('touchStart',[start]);await touch('touchMove',[moved]);
      await page.locator('#game-fullscreen').evaluate(e=>e.click());
      await page.setViewportSize({width:viewport.height,height:viewport.width});
      await page.waitForFunction(()=>QQTControls.moveForHeld(touchHeld)===4);
      assert.equal((await state()).move,4,'rotation releases joystick');
      await touch('touchEnd');
      const rotated=await page.locator('#game').boundingBox();
      assert(Math.abs(rotated.x+rotated.width/2-viewport.height/2)<1,'rotation centers board');
      await page.locator('#game-fullscreen').tap();
      await page.setViewportSize(viewport);
      await page.evaluate(()=>{const e=document.createElement('input');e.id='touch-typing-probe';document.body.append(e);});
      await page.locator('#touch-typing-probe').fill('昵称');await page.locator('#touch-typing-probe').focus();await page.keyboard.type('abc');await page.keyboard.press('ArrowRight');assert.equal((await state()).move,4);assert.equal(await page.locator('#touch-typing-probe').inputValue(),'昵称abc');
      assert.deepEqual(errors,[]);results.push({viewport,layout,resumed,longHold,multiBomb,itemCases,seventh,cleanup:true,keyboardIsolation:true,typing:true,nativeFullscreen,immersiveFallback:true,finishedRestart:true,rapidTapScale:1,rotation:true,cancellation,swipePinchLock:true,errors});
      await context.close();console.log('touch tolerance PASS',viewport.width,viewport.height);
    }
    fs.writeFileSync(`.hermes/touch-tolerance-${tag}.json`,JSON.stringify({status:'PASS',scope:'Chromium CDP touch, not physical iOS',sha:process.env.EXPECTED_SHA||null,results},null,2)+'\n');
  } finally {if(browser)await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
