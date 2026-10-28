'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {createServer}=require('../web/server');
const catalog=require('../web/review-catalog.json');
const out=process.env.EVIDENCE_DIR||'reports/review_best_d_20261028/browser';
const format=n=>n===0?'0':Math.abs(n)<.001?Number(n.toPrecision(5)).toString():n.toFixed(3);
(async()=>{
  fs.mkdirSync(out,{recursive:true});const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const base=`http://127.0.0.1:${server.address().port}/`,results=[];
  const verifier=fs.readFileSync('web/device_auth.js','utf8').match(/const VERIFIER = '([a-f0-9]+)'/)[1];let browser;
  try{
    browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox','--disable-gpu']});
    for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
      const name=viewport.width>600?'desktop':'mobile',ctx=await browser.newContext({viewport}),errors=[],blocked=[],http=[];
      await ctx.route('**/*',route=>{const r=route.request();if(r.url().startsWith(base)&&r.method()==='GET')return route.continue();blocked.push({method:r.method(),path:new URL(r.url()).pathname});return route.abort();});
      await ctx.addInitScript(rule=>{
        localStorage.setItem('qqt.advanced.authorization',JSON.stringify({schema:'qqt.advanced/v1',rule,granted:true}));
        window.dProbe={renders:[],cursor:null,curves:0};let api;
        const p=CanvasRenderingContext2D.prototype,begin=p.beginPath,move=p.moveTo,line=p.lineTo,stroke=p.stroke;
        p.beginPath=function(){this.dPath=[];return begin.call(this);};p.moveTo=function(x,y){this.dPath?.push([x,y]);return move.call(this,x,y);};p.lineTo=function(x,y){this.dPath?.push([x,y]);return line.call(this,x,y);};
        p.stroke=function(...args){if(this.canvas.id==='reward-chart'&&this.strokeStyle==='#e9f7ef')dProbe.cursor=this.dPath;if(!this.canvas.id&&this.strokeStyle==='#74cdb2'&&this.dPath?.length>2){dProbe.curves++;dProbe.curvePoints=this.dPath.length;}return stroke.apply(this,args);};
        Object.defineProperty(window,'QQTVisual',{get:()=>api,set(v){api=v;const create=v.createRenderer;v.createRenderer=(...args)=>{const renderer=create(...args),render=renderer.render;renderer.render=(s,now,motion)=>{dProbe.renders.push({tick:s.t,characters:motion.characters,palettes:motion.playerPalettes,labels:motion.playerLabels,advantages:motion.playerAdvantages});if(dProbe.renders.length>20)dProbe.renders.shift();return render(s,now,motion);};return renderer;};}});
      },verifier);
      const page=await ctx.newPage(),modelRequests=[];
      page.on('request',r=>{const pathname=new URL(r.url()).pathname;if(pathname.endsWith('/best-d-20261028/model.json'))modelRequests.push(pathname);});
      page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)http.push({path:new URL(r.url()).pathname,status:r.status()});});
      const ready=()=>page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:120000});
      await page.goto(base+'review.html');await ready();assert((await page.locator('#selected-replay').textContent()).includes(String(catalog.replays[0].seed)));
      assert.equal(await page.locator('#model-filter').inputValue(),'');
      await page.locator('.library summary').click();await page.locator('#model-filter').selectOption(catalog.evaluation.models[0].id);await ready();
      assert.equal(await page.locator('.replay-row').count(),15);assert((await page.locator('#selected-replay').textContent()).includes('u720'));
      await page.screenshot({path:path.join(out,`${name}-selector.png`),fullPage:name==='mobile'});
      const checked=[];
      for(const scene of ['open','native','carry_home']){
        const row=catalog.evaluation.replays.find(r=>r.scene===scene&&(scene!=='open'||r.opponent==='bun.hunter/normal')),doc=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));
        await page.evaluate(id=>{location.hash=`replay=${id}&tick=0`;},row.id);await ready();
        await page.waitForFunction(seed=>document.getElementById('selected-replay').textContent.includes(String(seed)),row.seed);
        const tick=doc.rewards.find(r=>Object.values(r[1]).some(v=>v!==0))?.[0]||Math.min(50,doc.actions.length);
        await page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input'));},tick);
        const parts=doc.rewards.find(r=>r[0]===tick)?.[1]||{},reward=format(Object.values(parts).reduce((n,v)=>n+v,0));
        assert.equal(await page.locator('#reward-delta').textContent(),reward);
        assert((await page.locator('#exact-time').textContent()).includes(`第 ${tick} 帧`));
        assert((await page.locator('.reward-total > span').textContent()).includes('离线复算'));
        assert(await page.locator('#reward-chart').isVisible());assert(await page.locator('#reward-empty').isHidden());
        const probe=await page.evaluate(()=>({frame:dProbe.renders.at(-1),cursor:dProbe.cursor,curves:dProbe.curves,curvePoints:dProbe.curvePoints}));
        assert.deepEqual(probe.frame.characters,['maomao','maomao']);assert.deepEqual(probe.frame.palettes,[null,'blue']);assert.equal(probe.frame.advantages[0].advantage,null);assert.equal(probe.frame.advantages[0].delta,null);
        assert(Math.abs(probe.cursor[0][0]-tick/doc.actions.length*560)<1e-8);assert(probe.curves>0);assert.equal(probe.curvePoints,doc.actions.length+1);
        await page.locator('#play').click();await page.waitForFunction(t=>Number(document.getElementById('seek').value)>t,tick);await page.locator('#play').click();
        const stopped=await page.locator('#seek').inputValue();await page.waitForTimeout(250);assert.equal(await page.locator('#seek').inputValue(),stopped);
        assert.equal(await page.locator('#play-state').textContent(),'已暂停');
        await page.evaluate(()=>{document.querySelector('.library').open=false;scrollTo(0,0);});
        assert(await page.locator('#policy-decision').isHidden());assert(await page.locator('.critic-comparison').isHidden());
        const layout=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));assert(layout.scrollWidth<=viewport.width);
        await page.screenshot({path:path.join(out,`${name}-${scene}.png`),fullPage:name==='mobile'});
        checked.push({id:row.id,sha256:row.sha256,tick,reward,probe,playback_pause:true,layout});
      }
      assert.deepEqual(errors,[]);assert.deepEqual(blocked,[]);assert.deepEqual(http,[]);assert.deepEqual(modelRequests,[]);
      results.push({viewport,default_preserved:true,model_selector:true,checked,errors,blocked,http,model_asset_requests:modelRequests});await ctx.close();
      console.log(`D real Chromium ${name} GREEN: selector / three scenes / reward / chart / playback / missing A1 / appearance`);
    }
    fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({passed:true,browser:browser.version(),results},null,2)+'\n');
  }finally{if(browser)await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e.message);process.exitCode=1;});
