'use strict';
const fs=require('fs'),assert=require('assert');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.WEB_URL||'http://127.0.0.1:8097/';
const out=process.env.EVIDENCE_DIR||'runs/replay_review_20261017/motion-local';
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:Number(process.env.VIEWPORT_WIDTH||1440),height:Number(process.env.VIEWPORT_HEIGHT||1050)}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.addInitScript(()=>{
      window.motionSamples=[];
      const sprites=new WeakMap(),draw=CanvasRenderingContext2D.prototype.drawImage;
      let frame=null,visual;
      CanvasRenderingContext2D.prototype.drawImage=function(img,...args){
        const pid=sprites.get(img);
        if(pid!==undefined){
          if(this.canvas.id!=='game')sprites.set(this.canvas,1);
          else if(frame){const transform=this.getTransform();frame.sprites.push({pid,x:args[0],y:args[1],anchorX:args[0]+img.width/2,anchorY:args[1]+img.height,screenX:(args[0]+img.width/2+transform.e)*frame.scaleX,screenY:(args[1]+img.height+transform.f)*frame.scaleY});}
        }
        return draw.call(this,img,...args);
      };
      Object.defineProperty(window,'QQTVisual',{get:()=>visual,set(api){
        visual=api;const create=api.createRenderer;
        api.createRenderer=(canvas,level,assets)=>{
          for(const rows of Object.values(assets.characters.maomao))for(const row of rows)for(const img of row)sprites.set(img,0);
          const renderer=create(canvas,level,assets),render=renderer.render;
          renderer.render=(sim,now,motion)=>{
            const rect=canvas.getBoundingClientRect();
            frame={tick:sim.t,alpha:motion.fraction,now,wallTime:performance.now(),pos:Array.from(sim.pos),next:motion.curPos.slice(),smooth:motion.smooth.slice(),sprites:[],scaleX:rect.width/canvas.width,scaleY:rect.height/canvas.height};
            const result=render(sim,now,motion);motionSamples.push(frame);frame=null;return result;
          };return renderer;
        };
      }});
    });
    await page.goto(base+'review.html'+(process.env.REPLAY_ID?'#replay='+encodeURIComponent(process.env.REPLAY_ID)+'&tick=0':''));
    await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});
    await page.locator('#advanced-open').click();
    if(await page.locator('#password-dialog').isVisible()){
      assert(process.env.QQT_SETTINGS_PASSWORD);await page.locator('#settings-password').fill(process.env.QQT_SETTINGS_PASSWORD);
      await page.locator('#password-form button[type=submit]').click();
    }
    await page.locator('#advanced-settings').waitFor({state:'visible'});
    const fixture=await page.evaluate(async()=>{
      const catalog=await(await fetch('review-catalog.json')).json(),doc=await(await fetch('review-replays/'+catalog.replays[0].file)).json();
      const level=(await(await fetch('assets/maps/levels.json')).json()).find(l=>l.qqt_id===806),assets=await QQTVisual.loadAssets(level);
      const canvas=document.createElement('canvas');canvas.id='game';canvas.width=900;canvas.height=810;
      const renderer=QQTVisual.createRenderer(canvas,level,assets),p=new QQTReview.Player(doc,level),found=[];
      while(p.sim.t<doc.actions.length&&found.filter(Boolean).length<2){
        const before=QQTReview.state(p.sim),tick=p.sim.t;p.step();const row=doc.actions[tick],intents=[row[0],row[2]];
        for(let pid=0;pid<2;pid++)if(!found[pid]&&QQTReviewView.canInterpolate(before,p.sim,pid,doc.events,intents[pid])){
          const from=new QQTReview.Player(doc,level);from.seek(tick);const authority=JSON.stringify(QQTReview.state(from.sim));
          const coords=[];for(const alpha of [0,.13,.27,.41,.59,.73,.87,1]){
            const motion=QQTReviewView.motion(from.sim,p.sim,alpha,true,doc.events,intents);
            renderer.render(from.sim,tick*100+alpha*100,motion);
            const s=motionSamples.at(-1).sprites.find(s=>s.pid===pid);if(s)coords.push([s.anchorX,s.anchorY]);
          }
          const axis=Math.abs(p.sim.pos[pid*2]-before.pos[pid*2])>1e-8?1:0;
          found[pid]={tick,coords,axis,authorityUnchanged:authority===JSON.stringify(QQTReview.state(from.sim))};
        }
      }return found;
    });
    fs.writeFileSync(out+'/fixture.json',JSON.stringify(fixture,null,2));
    for(const f of fixture){assert(f&&f.coords.length===8);const values=f.coords.map(c=>c[f.axis]),sign=Math.sign(values.at(-1)-values[0]);assert(values.every((v,i)=>!i||(v-values[i-1])*sign>0),'actual production drawImage anchors must be strictly monotonic');assert(f.authorityUnchanged);if(process.env.EXPECT_SMOOTH)assert(values.some(v=>!Number.isInteger(v)),'RED: integer renderer discards float screen positions');}
    await page.evaluate(()=>{motionSamples.length=0;});await page.locator('#play').click();
    await page.waitForTimeout(Number(process.env.SAMPLE_MS||16000));await page.locator('#play').click();
    const samples=await page.evaluate(()=>motionSamples),groups=new Map();
    for(const f of samples)for(const s of f.sprites){const key=f.tick+':'+s.pid;if(!groups.has(key))groups.set(key,[]);groups.get(key).push({...s,alpha:f.alpha,smooth:f.smooth[s.pid]});}
    const moving=[...groups].map(([key,rows])=>({key,rows,distinct:new Set(rows.map(r=>r.x+','+r.y)).size})).filter(g=>g.rows.length>=3);
    const summary={frames:samples.length,errors,fixture,scale:samples[0]&&[samples[0].scaleX,samples[0].scaleY],players:[0,1].map(pid=>({pid,groups:moving.filter(g=>g.key.endsWith(':'+pid)).length,intermediate:moving.filter(g=>g.key.endsWith(':'+pid)&&g.distinct>=3).length,stationary:moving.filter(g=>g.key.endsWith(':'+pid)&&g.distinct===1).length,rounded:moving.filter(g=>g.key.endsWith(':'+pid)).flatMap(g=>g.rows).every(r=>Number.isInteger(r.x)&&Number.isInteger(r.y))}))};
    if(process.env.EXPECT_SMOOTH)assert(summary.players.every(p=>!p.rounded&&p.intermediate>0));
    fs.writeFileSync(out+'/samples.json',JSON.stringify(samples));fs.writeFileSync(out+'/summary.json',JSON.stringify(summary,null,2));
    await page.screenshot({path:out+'/desktop.png',fullPage:true});console.log(JSON.stringify(summary));assert.equal(errors.length,0);
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
