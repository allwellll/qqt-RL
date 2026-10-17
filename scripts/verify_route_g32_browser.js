'use strict';
// Real public artifact only: no request interception, no synthetic trajectories.
const fs=require('fs'),assert=require('node:assert/strict'),P=require('../web/review_protocol'),R=require('../web/review_route');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.WEB_URL||'http://127.0.0.1:8097/',out=process.env.EVIDENCE_DIR||'runs/replay_review_20261017/browser-route-real';
(async()=>{
  fs.mkdirSync(out,{recursive:true});assert(process.env.QQT_SETTINGS_PASSWORD);
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),results=[];
  try{for(const mobile of [false,true]){
    const context=await browser.newContext({viewport:mobile?{width:390,height:844}:{width:1600,height:1100},isMobile:mobile,hasTouch:mobile}),page=await context.newPage(),errors=[],http=[],requests=[];
    page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)http.push({path:new URL(r.url()).pathname,status:r.status()});});page.on('request',r=>{if(r.url().includes('/review-replays/'))requests.push(new URL(r.url()).pathname);});
    await page.addInitScript(()=>{
      window.routeFrames=[];let api,frame;const fill=CanvasRenderingContext2D.prototype.fillText;
      CanvasRenderingContext2D.prototype.fillText=function(value,x,y,...rest){if(frame&&this.canvas.id==='game')frame.labels.push({value,x,y});return fill.call(this,value,x,y,...rest);};
      Object.defineProperty(window,'QQTVisual',{get:()=>api,set(v){api=v;const create=v.createRenderer;v.createRenderer=(...args)=>{const renderer=create(...args),render=renderer.render;renderer.render=(sim,now,motion)=>{frame={tick:sim.t,alpha:motion.fraction,alive:Array.from(sim.alive),visual:motion.visualPos.slice(),authority:Array.from(sim.pos),smooth:motion.smooth,route:motion.playerRoutes?.[0],labels:[]};const result=render(sim,now,motion);routeFrames.push(frame);if(routeFrames.length>3000)routeFrames.shift();frame=null;return result;};return renderer;};}});
    });
    await page.goto(base+'review.html');await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});
    await page.locator('#advanced-open').click();await page.locator('#settings-password').fill(process.env.QQT_SETTINGS_PASSWORD);await page.locator('#password-form button[type=submit]').click();await page.locator('#advanced-settings').waitFor({state:'visible'});
    const catalog=await page.evaluate(async()=>await(await fetch('review-catalog.json')).json()),row=catalog.replays[0];assert(row.diagnostic);
    const d=await page.evaluate(async file=>await(await fetch('review-replays/'+file)).json(),row.file);assert.equal(d.meta.checkpoint_sha256,'e141311268c6eaf419f1f651252bb27d46c106aeb407230c20eed4c8897a34e2');
    assert.equal(await page.locator('#episode-warning').textContent(),'诊断场景，不代表自然完整对局成功 · 预置掉包 / assist-off / 固定向上对手');assert(await page.locator('#route-experiment').isVisible());
    const experiment=await page.locator('#route-experiment').textContent();for(const v of ['67,108,864','7/16','6/16','2/16','0/16','12→9/64','9→1/64','不建议'])assert(experiment.includes(v));
    const jump=async tick=>page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input'));},tick);
    const checked=[];
    for(const tick of [1,2,100,101,500,1199,1200]){
      await jump(tick);const r=R.atTick(d,tick),frame=await page.evaluate(()=>routeFrames.at(-1));assert.equal(frame.tick,tick);assert.equal(frame.alpha,0);assert.equal(frame.route.text,R.overlay(d,tick).text);
      assert(frame.labels.some(l=>l.value===frame.route.text));assert(frame.labels.some(l=>l.value==='模型 P0'));if(frame.alive[1])assert(frame.labels.some(l=>l.value==='对手 P1'));assert.equal(frame.labels.filter(l=>l.value.startsWith('路线ΔΦ')).length,1);
      assert.equal(await page.locator('#exact-time').textContent(),`tick ${tick} · ${P.time(tick)}`);assert.equal(await page.locator('#reward-delta').textContent(),'未记录');assert.equal(await page.locator('#reward-total').textContent(),'未记录');
      assert((await page.locator('#route-details').textContent()).includes(R.signed(r.route_potential_delta)));assert((await page.locator('#route-details').textContent()).includes(R.signed(r.route_shaped_reward)));assert((await page.locator('#shaped-status').textContent()).includes('未记录'));assert((await page.locator('#advantage-status').textContent()).includes('未记录'));
      checked.push({tick,route:r});
    }
    assert.equal(await page.locator('#shaped-advantage-track .mark').count(),0);assert.equal(await page.locator('#advantage-track .mark').count(),0);assert(await page.locator('#advantage-mode [value="shaped"]').evaluate(e=>e.disabled));assert.equal(await page.locator('#route-potential-track .mark').count(),1200);
    const clickable=page.locator('#route-potential-track .dense-label').first(),clickTick=Number(await clickable.getAttribute('data-tick'));await clickable.click();assert.equal(Number(await page.locator('#seek').inputValue()),clickTick);await page.locator('#step-next').click();assert.equal(Number(await page.locator('#seek').inputValue()),clickTick+1);
    await jump(120);await page.evaluate(()=>routeFrames.length=0);await page.locator('#play').click();await page.waitForTimeout(1800);await page.locator('#play').click();
    const frames=await page.evaluate(()=>routeFrames);assert(frames.length>35);assert(frames.every(f=>f.route.text===R.overlay(d,f.tick).text));
    const groups=new Map();for(const f of frames){if(!groups.has(f.tick))groups.set(f.tick,[]);groups.get(f.tick).push(f);}
    const monotonic=[0,1].map(pid=>[...groups.values()].some(group=>{const g=group.filter(f=>f.smooth[pid]);if(g.length<3)return false;const axis=Math.abs(g.at(-1).visual[pid*2]-g[0].visual[pid*2])>1e-8?pid*2:pid*2+1,values=g.map(f=>f.visual[axis]),sign=Math.sign(values.at(-1)-values[0]);return sign&&values.every((v,i)=>!i||(v-values[i-1])*sign>0);}));assert(monotonic[0],'real moving model produces intermediate screen positions');
    const tick=Number(await page.locator('#seek').inputValue());assert(Number.isInteger(tick));assert((await page.locator('#refresh-status').textContent()).includes('α'));
    const layout=await page.evaluate(()=>{const b=document.querySelector('.board'),p=b.nextElementSibling;return {direct:p.classList.contains('playback-region')&&p.nextElementSibling.classList.contains('timeline-head')&&p.nextElementSibling.nextElementSibling.classList.contains('timeline-layout'),overflow:document.documentElement.scrollWidth>innerWidth+1};});assert(layout.direct);assert(!layout.overflow);
    await jump(101);await page.locator('#game').screenshot({path:out+`/${mobile?'mobile':'desktop'}-canvas.png`});await page.screenshot({path:out+`/${mobile?'mobile':'desktop'}.png`,fullPage:true});await page.reload();await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});assert(await page.locator('#advanced-settings').isVisible());assert.equal(Number(await page.locator('#seek').inputValue()),101);
    assert.equal(new Set(requests).size,1,'selected replay only, no bundled history');assert.deepEqual(errors,[]);assert.deepEqual(http,[]);results.push({mobile,replay:row.id,sha256:row.sha256,checks:checked,frames:frames.length,monotonic,layout,remembered_auth:true,deep_link:true,errors,http});await context.close();
  }}finally{await browser.close();}
  fs.writeFileSync(out+'/checks.json',JSON.stringify({passed:true,kind:'real-g32-production-trace',results},null,2)+'\n');console.log('Real g32 browser GREEN: desktop/mobile, honest diagnostic scope, route tick values, null rewards/A, real monotonic motion, integer seek/pause, deep link/auth, lazy load, zero errors');
})().catch(e=>{console.error(e);process.exitCode=1;});
