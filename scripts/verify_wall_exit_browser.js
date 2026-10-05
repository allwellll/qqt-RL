'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.WEB_URL || 'http://127.0.0.1:8080/';
const out = path.resolve(process.env.EVIDENCE_DIR || 'runs/wall_exit_20261005/browser');
const directions = ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'];
const vectors = [[-1,0],[1,0],[0,-1],[0,1]];
const launch = { headless: true, args: ['--no-sandbox'] };
if(process.env.CHROMIUM_PATH) launch.executablePath=process.env.CHROMIUM_PATH;
async function open(browser,viewport,baseline=false) {
  const page=await browser.newPage({viewport,isMobile:viewport.width<600,hasTouch:viewport.width<600}), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
  if(baseline)await page.route('**/sim.js',route=>route.fulfill({contentType:'application/javascript',
    body:execFileSync('git',['show','61f131c:web/sim.js'],{encoding:'utf8'})}));
  await page.addInitScript(()=>{
    window.setInterval=cb=>{window.appTick=cb;return 1;};
    window.realRAF=window.requestAnimationFrame;
    window.requestAnimationFrame=cb=>{window.appFrame=cb;return 1;};
    Date.now=()=>19;
  });
  await page.goto(base);
  try { await page.waitForFunction(()=>window.appFrame, null, { polling: 50 }); }
  catch(e) { throw new Error(JSON.stringify({errors,loading:await page.locator('#loading').textContent().catch(()=>null),reason:e.message})); }
  await page.evaluate(()=>{
    const step=QQT.Sim.prototype.frameStep;
    QQT.Sim.prototype.frameStep=function(...args){window.appSim=this;return step.apply(this,args);};
    window.frameNow=performance.now();appFrame(frameNow);
  });
  await page.waitForFunction(()=>document.querySelector('.loading.done'), null, { polling: 50 });
  return {page,errors};
}
async function screenshot(page,file){
  await page.evaluate(()=>{window.requestAnimationFrame=realRAF;});
  try{await page.screenshot({path:file,fullPage:true});}
  finally{await page.evaluate(()=>{window.requestAnimationFrame=cb=>{window.appFrame=cb;return 1;};});}
}
async function advance(page,key,frames=60,target=null) {
  await page.keyboard.down(key);
  const result=await page.evaluate(({frames,target,key})=>{
    let reached=false;
    for(let n=0;n<frames;n++){
      appFrame(frameNow+=20);
      if(target && appSim.centerCell(0).join(',')===target.join(',')){
        const axis=key==='ArrowUp'||key==='ArrowDown'?0:1;
        if(Math.abs(appSim.pos[axis]-(target[axis]+.5))<.025){reached=true;break;}
      }
    }
    return {pos:Array.from(appSim.pos.slice(0,2)),cell:appSim.centerCell(0),reached};
  },{frames,target,key});
  await page.keyboard.up(key);
  return result;
}
async function setup(page,move,cell=80,offset=null,kind=null,real=false){
  return page.evaluate(async({move,cell,offset,kind,real})=>{
    const level=(await fetch('assets/maps/levels.json').then(r=>r.json())).find(l=>l.qqt_id===806);
    const s=appSim;s.reset(level,{nativeItems:true,nativeTrap:true,teams:[0,1,1],bananaSlideSpeedPx:480});
    const row=Math.floor(cell/QQT.W),col=cell%QQT.W,[dy,dx]=[[-1,0],[1,0],[0,-1],[0,1]][move];
    if(!real){
      s.wall.fill(0);s.brick.fill(0);
      for(let r=row-1;r<=row+1;r++)for(let c=col-1;c<=col+1;c++)s.wall[r*QQT.W+c]=1;
      s.wall[(row+dy)*QQT.W+col+dx]=0;
    }
    s.fuse.fill(0);s.spdG[0]=1;
    s.pos[0]=row+(offset?offset[0]:20)/40;s.pos[1]=col+(offset?offset[1]:20)/40;
    const target=(row+dy)*QQT.W+col+dx;
    if(kind==='bomb')s.fuse[target]=30;
    else if(kind==='debris'){s.brick[target]=1;s.brickLinger[target]=2;}
    else if(kind)s[kind][target]=1;
    appFrame(frameNow+=20);
    return {start:Array.from(s.pos.slice(0,2)),target:[row+dy,col+dx]};
  },{move,cell,offset,kind,real});
}
(async()=>{
  fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch(launch),results=[];
  try{
    const old=await open(browser,{width:1440,height:1000},true), baseline=[];
    for(let move=0;move<4;move++){
      const offset=move===0?[34,20]:move===1?[1,20]:move===2?[20,34]:[20,1];
      const fixture=await setup(old.page,move,80,offset);
      const result=await advance(old.page,directions[move],60,fixture.target);
      assert(!result.reached,'reproduce all four old wall-exit failures');baseline.push({move,...fixture,...result});
    }
    assert.deepStrictEqual(old.errors,[]);results.push({baseline});await old.page.close();
    for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
      const {page,errors}=await open(browser,viewport), exits=[],blocked=[];
      const climbed=await page.evaluate(()=>{
        const s=appSim;s.wall.fill(0);s.brick.fill(0);s.fuse.fill(0);s.pos[0]=4.5;s.pos[1]=3.5;
        s.wall[3*QQT.W+4]=1;s.fuse[4*QQT.W+4]=30;s.owner[4*QQT.W+4]=1;s.spdG[0]=80/120;
        const advance=(move,count)=>{for(let n=0;n<count;n++)s.frameStep(0,move,.02);};
        advance(3,26);advance(0,2);
        const info=s.step([[4,1,0,1],[4,0,0,1],[4,0,0,1]]);
        advance(0,11);advance(3,13);
        if(!info.placed[0]||s.centerCell(0).join(',')!=='3,4')throw new Error('actual wall entry failed');
        s.fuse.fill(0);s.frameStep(0,4,1);s._nativeState(0).passActive=false;
        window.climbedFrame=s.snapshotReplay();return {pos:Array.from(s.pos.slice(0,2)),wallCell:s.centerCell(0)};
      });
      for(let move=0;move<4;move++){
        await page.evaluate(()=>{appSim.restoreReplay(climbedFrame);appFrame(frameNow+=20);});
        const [dy,dx]=vectors[move],result=await advance(page,directions[move],60,[3+dy,4+dx]);
        assert(result.reached,'actual wall-pass actor exits every direction');exits.push({actualClimb:true,move,...result});
      }
      const mapCases=await page.evaluate(async()=>{
        const level=(await fetch('assets/maps/levels.json').then(r=>r.json())).find(l=>l.qqt_id===806),cases=[];
        for(let move=0;move<4;move++){
          const seen=new Set(),[dy,dx]=[[-1,0],[1,0],[0,-1],[0,1]][move];
          for(let cell=0;cell<QQT.N;cell++)if(level.wall[cell]){
            const row=Math.floor(cell/QQT.W),col=cell%QQT.W,tr=row+dy,tc=col+dx,target=tr*QQT.W+tc;
            if(tr<0||tr>=QQT.H||tc<0||tc>=QQT.W||level.wall[target]||level.brick[target])continue;
            const type=level.layers[0][cell]||level.layers[1][cell];
            if(!seen.has(type)){cases.push({move,cell,type});seen.add(type);}
          }
        }
        return cases;
      });
      for(let move=0;move<4;move++)for(const offset of [[1,1],[39,39],[20,20]]){
        const fixture=await setup(page,move,80,offset),result=await advance(page,directions[move],60,fixture.target);
        assert(result.reached);assert.equal(result.pos[move<2?1:0],fixture.start[move<2?1:0]);
        exits.push({move,offset,...result});
      }
      for(const entry of mapCases){
        const fixture=await setup(page,entry.move,entry.cell,[39,1],null,true);
        await screenshot(page,`${out}/${viewport.width}-${entry.move}-${entry.type}-before.png`);
        const result=await advance(page,directions[entry.move],60,fixture.target);
        assert(result.reached,JSON.stringify(entry));
        await screenshot(page,`${out}/${viewport.width}-${entry.move}-${entry.type}-after.png`);
        exits.push({...entry,...result});
      }
      for(let move=0;move<4;move++)for(const kind of ['wall','brick','bomb','pushable','debris']){
        const fixture=await setup(page,move,80,[20,20],kind),result=await advance(page,directions[move]);
        assert.deepStrictEqual(result.pos,fixture.start,`no movement into ${kind}`);blocked.push({move,kind,...result});
      }
      const layout=await page.evaluate(()=>{
        const c=document.querySelector('#game'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data,colors=new Set();
        for(let i=0;i<d.length;i+=400)colors.add(`${d[i]},${d[i+1]},${d[i+2]}`);
        return {colors:colors.size,overflow:document.documentElement.scrollWidth>innerWidth};
      });
      assert(layout.colors>100&&!layout.overflow);assert.deepStrictEqual(errors,[]);
      results.push({viewport,climbed,exits,blocked,layout,errors});await page.close();
      console.log(`${viewport.width}: ${exits.length} real keyboard exits and ${blocked.length} blocked targets passed`);
    }
    fs.writeFileSync(`${out}/checks.json`,JSON.stringify({url:base,results},null,2)+'\n');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
