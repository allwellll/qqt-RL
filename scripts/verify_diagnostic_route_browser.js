'use strict';
// A conspicuously labelled fixture is intercepted only within this browser context.
// Neither the catalog nor replay files on disk/the public site are changed.
const fs=require('fs'),assert=require('assert'),zlib=require('zlib');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const P=require('../web/review_protocol'),R=require('../web/review_route'),E=require('./export_diagnostic_route');
const fixture=require('../tests/fixtures/route_replay');
const base=process.env.WEB_URL||'http://127.0.0.1:8097/',out=process.env.EVIDENCE_DIR||'runs/replay_review_20261017/browser-route';
(async()=>{
  assert(process.env.QQT_SETTINGS_PASSWORD,'private runner required');fs.mkdirSync(out,{recursive:true});
  const {replay,source}=fixture(),converted=await E.convert(replay,source,structuredClone(source)),d=converted.document;
  const bytes=Buffer.from(JSON.stringify(d)+'\n'),catalog=JSON.parse(fs.readFileSync('web/review-catalog.json'));
  const row=catalog.replays.find(r=>r.id===d.meta.id);row.sha256=await P.sha256(bytes);row.bytes=bytes.length;row.gzip_bytes=zlib.gzipSync(bytes,{level:9}).length;
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),evidence=[];
  try{for(const mobile of [false,true]){
    row.sha256=await P.sha256(bytes);row.bytes=bytes.length;row.gzip_bytes=zlib.gzipSync(bytes,{level:9}).length;
    const context=await browser.newContext({viewport:mobile?{width:390,height:844}:{width:1600,height:1100},isMobile:mobile,hasTouch:mobile});
    const page=await context.newPage(),errors=[],http=[],requests=[];
    page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
    page.on('response',r=>{if(r.status()>=400)http.push(new URL(r.url()).pathname);});page.on('request',r=>{if(r.url().includes('/review-replays/'))requests.push(new URL(r.url()).pathname);});
    await page.addInitScript(()=>{
      window.routeFrames=[];let api,frame;
      const fill=CanvasRenderingContext2D.prototype.fillText;
      CanvasRenderingContext2D.prototype.fillText=function(value,x,y,...args){if(frame&&this.canvas.id==='game')frame.labels.push({value,x,y});return fill.call(this,value,x,y,...args);};
      Object.defineProperty(window,'QQTVisual',{get:()=>api,set(v){api=v;const create=v.createRenderer;v.createRenderer=(...args)=>{const renderer=create(...args),render=renderer.render;renderer.render=(sim,now,motion)=>{
        frame={tick:sim.t,alpha:motion.fraction,route:motion.playerRoutes?.[0],visual:motion.visualPos.slice(),authority:Array.from(sim.pos),labels:[]};
        const result=render(sim,now,motion);routeFrames.push(frame);if(routeFrames.length>2000)routeFrames.shift();frame=null;return result;
      };return renderer;};}});
    });
    await page.goto(base+'review.html');await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});
    await page.locator('#advanced-open').click();await page.locator('#settings-password').fill(process.env.QQT_SETTINGS_PASSWORD);await page.locator('#password-form button[type=submit]').click();await page.locator('#advanced-settings').waitFor({state:'visible'});
    assert.equal(await page.locator('#route-potential-track .mark').count(),0);assert((await page.locator('#route-status').textContent()).includes('未记录'));assert(await page.locator('#advantage-mode [value="shaped"]').evaluate(e=>e.disabled));
    await page.route('**/review-catalog.json',r=>r.fulfill({json:catalog}));
    await page.route('**/review-replays/'+row.file,r=>r.fulfill({contentType:'application/json',body:bytes}));
    await page.reload();await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});assert(await page.locator('#advanced-settings').isVisible());
    const jump=async tick=>{await page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input'));},tick);};
    const checks=[];
    for(const tick of [1,2,3,4,d.actions.length]){
      await jump(tick);const r=R.atTick(d,tick),frame=await page.evaluate(()=>routeFrames.at(-1));assert.equal(frame.tick,tick);assert.equal(frame.alpha,0);
      assert.equal(frame.route.text,R.overlay(d,tick).text);assert(frame.labels.some(l=>l.value===frame.route.text));assert.equal(frame.labels.filter(l=>l.value.startsWith('路线ΔΦ')).length,1);
      assert(frame.labels.some(l=>l.value==='模型 P0')&&frame.labels.some(l=>l.value==='对手 P1'));
      const details=await page.locator('#route-details').textContent();assert(details.includes(R.signed(r.route_potential_delta)));assert(details.includes(R.signed(r.route_shaped_reward)));
      assert((await page.locator('#shaped-status').textContent()).includes('未记录'));assert.equal(await page.locator('#shaped-advantage-track .mark').count(),0);
      if(tick===4){assert(details.includes('不可达（INF）'));assert(details.includes('D · 入笼交付'));}
      checks.push({tick,delta:r.route_potential_delta,route_reward:r.route_shaped_reward});
    }
    await page.locator('#route-potential-track .mark[data-tick="2"]').click();assert.equal(Number(await page.locator('#seek').inputValue()),2);
    assert((await page.locator('#route-status').textContent()).includes('测试 fixture'));
    const rewardTick=source.events.find(e=>e.tick>4&&e.tick<d.actions.length)?.tick;
    if(rewardTick){const tooltip=await page.locator(`#route-reward-track .mark[data-tick="${rewardTick}"]`).getAttribute('title');assert(tooltip.includes('Reward')&&tooltip.includes('同 tick 事件共用一次'));}
    await jump(1);await page.locator('#step-next').click();assert.equal((await page.evaluate(()=>routeFrames.at(-1))).tick,2);
    await page.locator('#route-overlay').uncheck();assert.equal(await page.evaluate(()=>routeFrames.at(-1).labels.filter(l=>l.value.startsWith('路线ΔΦ')).length),0);await page.locator('#route-overlay').check();
    await jump(1);await page.evaluate(()=>routeFrames.length=0);await page.locator('#play').click();await page.waitForTimeout(700);await page.locator('#play').click();
    const frames=await page.evaluate(()=>routeFrames);assert(frames.length>12);assert(frames.every(f=>f.route.text===R.overlay(d,f.tick).text),'route values must use authoritative integer tick');
    const layout=await page.evaluate(()=>{const board=document.querySelector('.board'),play=board.nextElementSibling,time=play.nextElementSibling;return {direct:play.classList.contains('playback-region')&&time.classList.contains('timeline-head')&&time.nextElementSibling.classList.contains('timeline-layout'),scrollWidth:document.documentElement.scrollWidth,innerWidth:innerWidth,scrollX:scrollX,bodyWidth:document.body.scrollWidth,overflow:document.documentElement.scrollWidth>innerWidth+1,offenders:[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1).slice(0,12).map(e=>({id:e.id,tag:e.tagName,cls:e.className,right:e.getBoundingClientRect().right,parent:e.parentElement?.id,overflow:getComputedStyle(e.parentElement).overflow}))};});assert(layout.direct);if(layout.overflow){console.log(JSON.stringify({mobile,layout}));await page.screenshot({path:out+'/overflow.png',fullPage:true});}assert(!layout.overflow);
    await jump(4);await page.screenshot({path:`${out}/${mobile?'mobile':'desktop'}-fixture.png`,fullPage:true});await page.locator('#game').screenshot({path:`${out}/${mobile?'mobile':'desktop'}-fixture-canvas.png`});
    await page.reload();await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});assert.equal(Number(await page.locator('#seek').inputValue()),4);
    // Independently exercise real shaped-critic presentation with a synthetic mathematical fixture.
    const shaped=structuredClone(source);shaped.meta.shaped_critic={schema:'qqt.route-shaped-a1/v1',trained_on_route_reward:true,checkpoint_sha256:d.meta.checkpoint_sha256,critic_sha256:'b'.repeat(64),reward_ledger_sha256:source.meta.reward_ledger_sha256,counterfactual_sha256:'c'.repeat(64),action_scope:'joint_move_ability',legal_actions:'synthetic legal joint mask'};
    shaped.shaped_results=[1,2].map(tick=>({tick,taken:d.actions[tick-1].slice(0,2),candidates:[{action:d.actions[tick-1].slice(0,2),legal:true,probability:.5,reward:tick,nextValue:0,terminal:false},{action:[(d.actions[tick-1][0]+1)%5,0],legal:true,probability:.5,reward:0,nextValue:0,terminal:false}]}));
    const sd=(await E.convert(replay,shaped,structuredClone(shaped))).document,sb=Buffer.from(JSON.stringify(sd)+'\n');row.bytes=sb.length;row.sha256=await P.sha256(sb);
    await page.unroute('**/review-replays/'+row.file);await page.route('**/review-replays/'+row.file,r=>r.fulfill({contentType:'application/json',body:sb}));
    await page.reload();await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});await jump(2);
    await page.locator('#advantage-mode').selectOption('shaped');assert(await page.evaluate(()=>routeFrames.at(-1).labels.some(l=>l.value.startsWith('塑形优势增量'))));
    assert((await page.locator('#advantage-details').textContent()).includes('策略加权一步优势'));assert.equal(Number(await page.locator('#shaped-delta-track .mark[data-tick="2"]').getAttribute('data-value')),.5);
    // Values stay at tick; their anchors follow production floating visual positions.
    const anchors=await page.evaluate(async file=>{
      const c=await(await fetch('review-replays/'+file)).json(),level=(await(await fetch('assets/maps/levels.json')).json()).find(l=>l.qqt_id===806),p=new QQTReview.Player(c,level);
      while(p.sim.t<c.actions.length){const start=p.sim.t,before=QQTReview.state(p.sim),a=c.actions[start];p.step();if(QQTReviewRoute.atTick(c,start).route_potential_delta===null||!QQTReviewView.canInterpolate(before,p.sim,0,c.events,a[0]))continue;
        const from=new QQTReview.Player(c,level);from.seek(start);const canvas=document.createElement('canvas');canvas.id='game';canvas.width=900;canvas.height=810;const renderer=QQTVisual.createRenderer(canvas,level,await QQTVisual.loadAssets(level)),points=[];
        for(const alpha of [.1,.25,.4,.55,.7,.85]){const motion=QQTReviewView.motion(from.sim,p.sim,alpha,true,c.events,[a[0],a[2]]);motion.playerRoutes=[QQTReviewRoute.overlay(c,start),null];motion.playerAdvantages=[QQTReviewAdvantage.atTick(c,start),null];renderer.render(from.sim,start*100+alpha*100,motion);const label=routeFrames.at(-1).labels.find(l=>l.value.startsWith('路线ΔΦ'));points.push({...label,alpha,tick:start});}
        const axis=Math.abs(p.sim.pos[1]-before.pos[1])>1e-9?'x':'y',direction=Math.sign(points.at(-1)[axis]-points[0][axis]);return {axis,direction,points};
      }throw Error('no safe move');
    },row.file);
    assert(anchors.direction&&anchors.points.every((p,i)=>!i||(p[anchors.axis]-anchors.points[i-1][anchors.axis])*anchors.direction>0));assert.equal(new Set(anchors.points.map(p=>p.value)).size,1);assert(!anchors.points[0].value.includes('未记录'));
    assert.equal(new Set(requests).size,1,'fixture remains lazy: only selected replay requested');assert.deepEqual(errors,[]);assert.deepEqual(http,[]);
    evidence.push({kind:'synthetic-fixture-not-training-evidence',mobile,checks,frames:frames.length,layout,anchors,errors,http,lazy_requests:new Set(requests).size});await context.close();
  }}finally{await browser.close();}
  fs.writeFileSync(out+'/results.json',JSON.stringify(evidence,null,2)+'\n');console.log('Route browser GREEN: desktop/mobile fixture, integer tick, negative/INF/terminal, click/reward, original vs shaped, float anchors, auth/deep link, lazy load, zero errors');
})().catch(e=>{console.error(e);process.exitCode=1;});
