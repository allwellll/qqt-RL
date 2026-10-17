'use strict';
const fs=require('fs'),assert=require('assert');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.WEB_URL||'http://127.0.0.1:8097/',out=process.env.EVIDENCE_DIR||'runs/replay_review_20261017/browser-a1';
(async()=>{
  fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const evidence=[];
  try{
    for(const mobile of [false,true]){
      const context=await browser.newContext({viewport:mobile?{width:390,height:844}:{width:1600,height:1100},deviceScaleFactor:1,isMobile:mobile,hasTouch:mobile});
      const page=await context.newPage(),errors=[],http=[];
      page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)http.push({path:new URL(r.url()).pathname,status:r.status()});});
      await page.addInitScript(()=>{
        window.overlayFrames=[];let active=null,api;
        const fill=CanvasRenderingContext2D.prototype.fillText;
        CanvasRenderingContext2D.prototype.fillText=function(value,x,y,...args){if(active&&this.canvas.id==='game')active.labels.push({value,x,y,font:this.font});return fill.call(this,value,x,y,...args);};
        Object.defineProperty(window,'QQTVisual',{get:()=>api,set(v){api=v;const create=v.createRenderer;v.createRenderer=(...args)=>{const renderer=create(...args),render=renderer.render;renderer.render=(sim,now,motion)=>{
          active={tick:sim.t,alpha:motion.fraction,visualPos:motion.visualPos.slice(),authority:Array.from(sim.pos),smooth:motion.smooth.slice(),labels:[],advantage:motion.playerAdvantages?.[0]||null};
          const result=render(sim,now,motion);overlayFrames.push(active);if(overlayFrames.length>2000)overlayFrames.shift();active=null;return result;
        };return renderer;};}});
      });
      await page.goto(base+'review.html');await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});
      await page.locator('#advanced-open').click();await page.locator('#settings-password').fill(process.env.QQT_SETTINGS_PASSWORD);await page.locator('#password-form button[type=submit]').click();await page.locator('#advanced-settings').waitFor({state:'visible'});
      const catalog=await page.evaluate(async()=>await(await fetch('review-catalog.json')).json());
      const layout=await page.evaluate(()=>{const c=document.querySelector('.center'),children=[...c.children],board=c.querySelector('.board'),play=c.querySelector('.playback-region'),time=c.querySelector('.timeline-head');return {direct:board.nextElementSibling===play&&play.nextElementSibling===time&&time.nextElementSibling.classList.contains('timeline-layout'),boardBottom:board.getBoundingClientRect().bottom,playTop:play.getBoundingClientRect().top,overflow:document.documentElement.scrollWidth>innerWidth+1};});
      assert(layout.direct,'canvas, playback, timeline must be consecutive');assert(Math.abs(layout.playTop-layout.boardBottom)<4);assert(!layout.overflow);
      for(const scene of catalog.scenes){
        await page.locator('#scene-filter').selectOption(scene.id);assert.equal(await page.locator('.replay-row').count(),2);
        const row=catalog.replays.find(r=>r.scene===scene.id);await page.locator(`[data-replay="${row.id}"]`).click();await page.waitForFunction(id=>document.getElementById('locator').value.startsWith(id+'@'),row.id);
        const doc=await page.evaluate(async file=>await(await fetch('review-replays/'+file)).json(),row.file);
        const tick=Math.min(50,doc.actions.length),idx=doc.advantages.ticks.indexOf(tick),expected=Object.fromEntries(['taken_q','mean_q','advantage','delta'].map(k=>[k,doc.advantages[k][idx]===null?null:doc.advantages[k][idx]*doc.advantages.scale]));
        await page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input'));},tick);
        const frame=await page.evaluate(()=>overlayFrames.at(-1));assert.deepEqual(frame.advantage,expected);assert.equal(frame.tick,tick);assert.equal(frame.alpha,0);
        const fmt=v=>v===null?'未记录':(v>=0?'+':'')+v.toFixed(3);
        assert(frame.labels.some(l=>l.value===`优势增量 ΔA ${fmt(expected.delta)}`));assert(frame.labels.some(l=>l.value===`A ${fmt(expected.advantage)}`));
        const label=frame.labels.find(l=>l.value.startsWith('优势增量')),cssScale=await page.locator('#game').evaluate(c=>c.getBoundingClientRect().width/c.width);
        assert(Number(label.font.match(/([\d.]+)px/)[1])*cssScale>=10,'head advantage remains readable on mobile');
        assert.equal(frame.labels.filter(l=>l.value.startsWith('优势增量')).length,1,'never attribute model advantage to opponent');
        assert(frame.labels.some(l=>l.value==='模型 P0')&&frame.labels.some(l=>l.value==='对手 P1'));
        const details=await page.locator('#advantage-details dd').allTextContents();
        for(const [i,k]of ['taken_q','mean_q','advantage','delta'].entries())assert.equal(details[i+1],Math.abs(expected[k])<1e-8?'0':expected[k].toFixed(3));
        const mark=page.locator(`#advantage-delta-track .mark[data-tick="${tick}"]`);assert.equal(Number(await mark.getAttribute('data-value')),expected.delta);
        const clickable=page.locator('#advantage-delta-track .dense-label').nth(1),jumpTick=Number(await clickable.getAttribute('data-tick'));
        await clickable.click();assert.equal(Number(await page.locator('#seek').inputValue()),jumpTick);
        await page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input'));},tick);
        await page.locator('#step-next').click();assert.equal(Number(await page.locator('#seek').inputValue()),tick+1);assert.equal((await page.evaluate(()=>overlayFrames.at(-1))).alpha,0);
        await page.locator('#step-back').click();
        await page.locator('#advantage-overlay').uncheck();assert.equal(await page.evaluate(()=>overlayFrames.at(-1).labels.filter(l=>l.value.startsWith('优势增量')).length),0);await page.locator('#advantage-overlay').check();
        await page.evaluate(()=>overlayFrames.length=0);await page.locator('#play').click();await page.waitForTimeout(1400);await page.locator('#play').click();
        const frames=await page.evaluate(()=>overlayFrames);assert(frames.length>25);
        assert(new Set(frames.map(f=>f.advantage?.delta)).size>2,'real delta changes with actions');
        assert(frames.every(f=>{const a=f.advantage;return !a||!f.labels.some(l=>l.value==='模型 P0')||f.labels.some(l=>l.value===`优势增量 ΔA ${a.delta===null?'未记录':(a.delta>=0?'+':'')+a.delta.toFixed(3)}`);}));
        const groups=new Map();for(const f of frames){if(!groups.has(f.tick))groups.set(f.tick,[]);groups.get(f.tick).push(f);}
        const follows=[...groups.values()].some(group=>{const points=group.filter(f=>f.smooth[0]&&f.labels.some(l=>l.value.startsWith('优势增量')));return points.length>=3&&new Set(points.map(f=>f.labels.find(l=>l.value.startsWith('优势增量')).x+','+f.labels.find(l=>l.value.startsWith('优势增量')).y)).size>=3;});
        // Scenarios can be blocked at the sampled instant; deterministic renderer fixture below proves float tracking.
        const current=Number(await page.locator('#seek').inputValue());assert(Number.isInteger(current));
        await page.screenshot({path:`${out}/${mobile?'mobile':'desktop'}-${scene.id}.png`,fullPage:true});
        await page.locator('#game').screenshot({path:`${out}/${mobile?'mobile':'desktop'}-${scene.id}-canvas.png`});
        await page.reload();await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});assert(await page.locator('#advanced-settings').isVisible());assert.equal(Number(await page.locator('#seek').inputValue()),current);assert.equal((await page.evaluate(()=>overlayFrames.at(-1))).tick,current);
        evidence.push({mobile,scene:scene.id,replay_id:row.id,compared_tick:tick,expected,frames:frames.length,overlay_follows_live_float:follows,layout,remembered_authorization:true,deep_link:true});
      }
      const floating=await page.evaluate(async()=>{
        const c=await(await fetch('review-catalog.json')).json(),row=c.replays.find(r=>r.scene==='native'),d=await(await fetch('review-replays/'+row.file)).json();
        const level=(await(await fetch('assets/maps/levels.json')).json())[0],p=new QQTReview.Player(d,level),canvas=document.createElement('canvas');canvas.id='game';canvas.width=900;canvas.height=810;
        const renderer=QQTVisual.createRenderer(canvas,level,await QQTVisual.loadAssets(level));
        while(p.sim.t<d.actions.length){const before=QQTReview.state(p.sim),tick=p.sim.t,row=d.actions[tick];p.step();
          if(!QQTReviewView.canInterpolate(before,p.sim,0,d.events,row[0]))continue;
          const start=new QQTReview.Player(d,level);start.seek(tick);const coords=[];
          for(const alpha of [.1,.25,.4,.55,.7,.85]){const m=QQTReviewView.motion(start.sim,p.sim,alpha,true,d.events,[row[0],row[2]]);m.playerAdvantages=[QQTReviewAdvantage.atTick(d,tick),null];renderer.render(start.sim,tick*100+alpha*100,m);const f=overlayFrames.at(-1),label=f.labels.find(l=>l.value.startsWith('优势增量'));if(label)coords.push({x:label.x,y:label.y,visual:m.visualPos.slice()});}
          const axis=Math.abs(p.sim.pos[1]-before.pos[1])>1e-9?'x':'y',values=coords.map(c=>c[axis]),direction=Math.sign(values.at(-1)-values[0]);
          if(coords.length===6&&direction&&values.every((v,i)=>!i||(v-values[i-1])*direction>0))return {tick,axis,coords,strictly_monotonic:true};
        }return null;
      });assert(floating,'advantage head label follows multiple strictly monotonic floating screen coordinates');evidence.push({mobile,deterministic_float_overlay:floating});
      await page.locator('#scene-filter').selectOption('');await page.locator('#show-history').check();assert.equal(await page.locator('.replay-row').count(),catalog.replays.length+catalog.history_replays.length);
      assert.deepEqual(errors,[]);assert.deepEqual(http,[]);await context.close();
    }
    fs.writeFileSync(out+'/checks.json',JSON.stringify({passed:true,evidence,console_errors:0,http_errors:0},null,2));console.log(JSON.stringify(evidence));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
