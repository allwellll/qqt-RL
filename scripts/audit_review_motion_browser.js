'use strict';
const fs=require('fs'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.WEB_URL||'http://127.0.0.1:8097/',out=process.env.EVIDENCE_DIR||'runs/replay_review_20261017/motion-route-audit';
(async()=>{
  fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),results=[];
  try{for(const mobile of [false,true])for(const smooth of [true,false]){
    const key=`${mobile?'mobile':'desktop'}-${smooth?'smooth':'discrete'}`,folder=out+'/'+key;fs.mkdirSync(folder,{recursive:true});
    const context=await browser.newContext({viewport:mobile?{width:390,height:844}:{width:1600,height:1100},isMobile:mobile,hasTouch:mobile,...(process.env.RECORD_VIDEO?{recordVideo:{dir:folder,size:mobile?{width:390,height:844}:{width:1600,height:1100}}}:{})});
    const page=await context.newPage(),errors=[],http=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)http.push({path:new URL(r.url()).pathname,status:r.status()});});
    await page.addInitScript(()=>{
      window.motionAudit=[];let api,frame;const sprites=new WeakMap(),draw=CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage=function(img,...args){const pid=sprites.get(img);if(pid!==undefined){if(this.canvas.id!=='game')sprites.set(this.canvas,1);else if(frame){const m=this.getTransform(),x=args[0]+img.width/2,y=args[1]+img.height;frame.sprites.push({pid,x,y,screenX:(x+m.e)*frame.scaleX,screenY:(y+m.f)*frame.scaleY,expectedX:frame.visual[pid*2+1]*60,expectedY:Math.min(api.playerVisualY(frame.visual[pid*2],img.height),780-img.height)+img.height});}}return draw.call(this,img,...args);};
      Object.defineProperty(window,'QQTVisual',{get:()=>api,set(v){api=v;const create=v.createRenderer;v.createRenderer=(canvas,level,assets)=>{for(const rows of Object.values(assets.characters.maomao))for(const row of rows)for(const img of row)sprites.set(img,0);const renderer=create(canvas,level,assets),render=renderer.render;renderer.render=(sim,now,motion)=>{const rect=canvas.getBoundingClientRect();frame={tick:sim.t,wall:performance.now(),now,playing:document.getElementById('play-state').textContent==='播放中',nextTick:motion.nextTick,reasons:motion.snapReasons,alpha:motion.fraction,visual:motion.visualPos.slice(),authority:Array.from(sim.pos),next:motion.curPos.slice(),intents:motion.intents?.slice(),smooth:motion.smooth.slice(),alive:Array.from(sim.alive),carried:Array.from(sim.bunCarried),trapped:Array.from(sim.trapped),sprites:[],scaleX:rect.width/canvas.width,scaleY:rect.height/canvas.height};const result=render(sim,now,motion);motionAudit.push(frame);frame=null;return result;};return renderer;};}});
    });
    await page.goto(base+'review.html');await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:90000});await page.locator('#advanced-open').click();await page.locator('#settings-password').fill(process.env.QQT_SETTINGS_PASSWORD);await page.locator('#password-form button[type=submit]').click();await page.locator('#advanced-settings').waitFor({state:'visible'});
    const d=await page.evaluate(async()=>{const c=await(await fetch('review-catalog.json')).json();return await(await fetch('review-replays/'+c.replays.find(r=>r.diagnostic).file)).json();});
    const probes=await page.evaluate(async doc=>{
      const level=(await(await fetch('assets/maps/levels.json')).json()).find(l=>l.qqt_id===806),assets=await QQTVisual.loadAssets(level),canvas=document.createElement('canvas');canvas.id='game';canvas.width=900;canvas.height=810;
      const renderer=QQTVisual.createRenderer(canvas,level,assets),result=[];
      for(const [pid,tick]of [[0,2],[0,20],[0,36],[0,60],[1,101]]){
        const p=new QQTReview.Player(doc,level),n=new QQTReview.Player(doc,level);p.seek(tick);n.seek(tick+1);const before=JSON.stringify(QQTReview.state(p.sim)),coords=[],action=doc.actions[tick],intents=[action[0],action[2]];
        for(const alpha of [0,.13,.27,.41,.59,.73,.87,1]){const motion=QQTReviewView.motion(p.sim,n.sim,alpha,true,doc.events,intents);motion.intents=intents;renderer.render(p.sim,tick*100+alpha*100,motion);const sprite=motionAudit.at(-1).sprites.find(s=>s.pid===pid);if(!sprite)throw Error('production sprite occluded at probe '+pid+':'+tick+':'+alpha);coords.push([sprite.x,sprite.y]);}
        result.push({pid,tick,axis:Math.abs(n.sim.pos[pid*2]-p.sim.pos[pid*2])>1e-8?1:0,coords,authority_unchanged:before===JSON.stringify(QQTReview.state(p.sim))});
      }return result;
    },d);
    for(const p of probes){const values=p.coords.map(c=>c[p.axis]),sign=Math.sign(values.at(-1)-values[0]);assert(sign&&values.every((v,i)=>!i||(v-values[i-1])*sign>0),'actual production renderer must receive intermediate coordinates, including bomb tick');assert(values.some(v=>!Number.isInteger(v)));assert(p.authority_unchanged);}
    assert.equal(d.meta.seed,2026101732);await page.locator('#smooth').setChecked(smooth);await page.locator('#speed').selectOption(process.env.SAMPLE_SPEED||'4');
    await page.evaluate(()=>motionAudit.length=0);await page.locator('#play').click();
    const end=Number(process.env.END_TICK||1200);await page.waitForFunction(t=>Number(document.getElementById('seek').value)>=t,end,{timeout:180000});if(end<1200)await page.locator('#play').click();
    const samples=await page.evaluate(()=>motionAudit),summary={key,probes,frames:samples.length,first_tick:samples[0]?.tick,last_tick:samples.at(-1)?.tick,max_frame_screen_distance:[0,0],max_continuous_frame_screen_distance:[0,0],max_wall_gap_ms:0,non_float_override:[],phase_resets:[],scheduler_gaps:[],snaps:[],anomalies:[],safe_intermediate_groups:[0,0],css_scales:[...new Set(samples.map(f=>f.scaleX.toFixed(6)+','+f.scaleY.toFixed(6)))],errors,http};
    const groups=new Map();let previous;
    for(const f of samples){
      if(previous){summary.max_wall_gap_ms=Math.max(summary.max_wall_gap_ms,f.wall-previous.wall);if(f.playing&&previous.playing&&f.tick===previous.tick&&f.alpha+1e-6<previous.alpha)summary.phase_resets.push(f.tick);if(f.tick-previous.tick>1)summary.scheduler_gaps.push({tick:f.tick,from:previous.tick,wall_ms:f.wall-previous.wall});}
      if(f.tick<d.actions.length&&f.nextTick!==f.tick+1)summary.anomalies.push({tick:f.tick,kind:'missing_one_tick_preview'});
      for(const sprite of f.sprites){const pid=sprite.pid;if(Math.abs(sprite.x-sprite.expectedX)>1e-6||Math.abs(sprite.y-sprite.expectedY)>1e-6)summary.non_float_override.push({tick:f.tick,pid});
        const groupKey=f.tick+':'+pid;if(!groups.has(groupKey))groups.set(groupKey,[]);groups.get(groupKey).push({...sprite,smooth:f.smooth[pid],alpha:f.alpha});
        const prior=previous?.sprites.find(s=>s.pid===pid);if(prior){const distance=Math.hypot(sprite.screenX-prior.screenX,sprite.screenY-prior.screenY);summary.max_frame_screen_distance[pid]=Math.max(summary.max_frame_screen_distance[pid],distance);
          if(previous.smooth[pid]&&f.smooth[pid])summary.max_continuous_frame_screen_distance[pid]=Math.max(summary.max_continuous_frame_screen_distance[pid],distance);
          if(distance>5){const events=d.events.filter(e=>e.player===pid&&e.tick>previous.tick&&e.tick<=f.tick).map(e=>e.kind),classification=!f.playing?'pause_snap':!smooth?'smoothing_disabled':f.tick-previous.tick>1?'scheduler_gap':previous.smooth[pid]?'safe_continuous':previous.reasons?.[pid]||events.join('+')||'unclassified';summary.snaps.push({tick:f.tick,from:previous.tick,pid,distance,classification,events,wall_ms:f.wall-previous.wall});}
          if(smooth&&f.playing&&f.tick===previous.tick&&previous.smooth[pid]&&!f.smooth[pid])summary.anomalies.push({tick:f.tick,pid,kind:'gate_changed_inside_tick'});
        }
      }previous=f;
    }
    for(const [key,points]of groups){const pid=Number(key.split(':')[1]),values=points.filter(p=>p.smooth);if(values.length<3)continue;const axis=Math.abs(values.at(-1).x-values[0].x)>1e-6?'x':'y',direction=Math.sign(values.at(-1)[axis]-values[0][axis]);if(direction&&values.every((p,i)=>!i||(p[axis]-values[i-1][axis])*direction>0))summary.safe_intermediate_groups[pid]++;}
    summary.anomalies.push(...summary.phase_resets.map(tick=>({tick,kind:'alpha_reset'})),...summary.non_float_override.map(v=>({...v,kind:'float_override_lost'})));
    summary.actual_fps=samples.length*1000/(samples.at(-1).wall-samples[0].wall);summary.status=await page.locator('#refresh-status').textContent();
    const paused=[];for(const tick of [79,80,104,115,116,500]){await page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input'));},tick);await page.locator('#step-next').click();await page.locator('#step-back').click();const f=await page.evaluate(()=>motionAudit.at(-1));assert.equal(f.tick,tick);assert.equal(f.alpha,0);assert.deepEqual(f.visual,f.authority);assert(f.sprites.every(s=>Math.abs(s.x-s.expectedX)<1e-6&&Math.abs(s.y-s.expectedY)<1e-6));paused.push({tick,alpha:f.alpha});}summary.pause_seek_step=paused;
    await page.screenshot({path:folder+'/page.png',fullPage:true});await page.locator('#game').screenshot({path:folder+'/canvas.png'});fs.writeFileSync(folder+'/frames.json',JSON.stringify(samples));fs.writeFileSync(folder+'/summary.json',JSON.stringify(summary,null,2)+'\n');results.push(summary);console.log(JSON.stringify({...summary,snaps:summary.snaps.slice(0,12)}));assert.deepEqual(summary.anomalies,[]);assert.deepEqual(errors,[]);assert.deepEqual(http,[]);await context.close();
  }}finally{await browser.close();}
  fs.writeFileSync(out+'/summary.json',JSON.stringify(results,null,2)+'\n');
})().catch(e=>{console.error(e);process.exitCode=1;});
