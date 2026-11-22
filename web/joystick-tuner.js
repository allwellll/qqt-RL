'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const {trajectory, ...defaults} = QQTControls.TRAJECTORY_TOUCH_PROFILE;
  const controls={deadzone:$('deadzone'),hysteresis:$('hysteresis'),buffer:$('buffer'),diagMin:$('diagMin'),diagMax:$('diagMax')};
  const labels={deadzone:'deadzoneRatio',hysteresis:'hysteresisDeg',buffer:'bufferMs',diagMin:'diagonalMin',diagMax:'diagonalMax'};
  const read=()=>({...Object.fromEntries(Object.entries(labels).map(([id,key])=>[key,+controls[id].value])),diagonalMode:$('diagonalMode').value});
  const save=c=>localStorage.setItem('qqt.joystick.tuner.v2',JSON.stringify(c));
  const load=()=>{try{return {...defaults,...JSON.parse(localStorage.getItem('qqt.joystick.tuner.v2')||'{}')}}catch(_){return {...defaults}}};
  const updateLabels=()=>{$('deadzoneValue').textContent=(+controls.deadzone.value*100).toFixed(0)+'%';$('hysteresisValue').textContent=controls.hysteresis.value+'°';$('bufferValue').textContent=controls.buffer.value+'ms';$('diagMinValue').textContent=controls.diagMin.value+'°';$('diagMaxValue').textContent=controls.diagMax.value+'°'};
  const apply=c=>{for(const [id,key] of Object.entries(labels))controls[id].value=c[key];$('diagonalMode').value=c.diagonalMode||'preserve';updateLabels();save(c)};
  let config=load();
  const engine=globalThis.QQTBunNativeV5FacingWindow||globalThis.QQT;
  const {H, W, NATIVE_CELL_PX, NATIVE_HALF_PX} = engine;
  const createSim = () => {
    const s = new engine.Sim(3);
    s.reset('open', {nativeItems:true, nativeTrap:true});
    for (const key of ['wall','brick','cover','crate','fuse','pushable','fieldItem','blastLinger']) s[key].fill(0);
    for (let row=0; row<H; row++) for (let col=0; col<W; col++) {
      const i = row*W + col;
      if (row===0 || row===H-1 || col===0 || col===W-1) s.wall[i] = 1;
      else if (row>=2 && row<11 && col>=3 && col<13 &&
          ((row%2===0 && col%3!==0) || (row===6 && col>5 && col<10))) s.brick[i] = 1;
    }
    for (const [row,col] of [[3,5],[3,9],[9,7],[9,11]]) s.wall[row*W+col] = 1;
    s.fuse[4*W+9] = 18;
    s.pos[0] = 10.5;
    s.pos[1] = 2.5;
    s.invuln.fill(100);
    s.spdG[0] = 2/3;
    return s;
  };
  let sim=createSim();const zone=$('direction-zone'),held=new Set();let bind,listenerScope;const angleSamples=[];let lastAngleSample=0;const angleCanvas=$('angle-chart'),angleCtx=angleCanvas.getContext('2d');
  const drawAngle=now=>{const rect=angleCanvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));if(angleCanvas.width!==w||angleCanvas.height!==h){angleCanvas.width=w;angleCanvas.height=h}angleCtx.setTransform(dpr,0,0,dpr,0,0);const cw=rect.width,ch=rect.height;angleCtx.clearRect(0,0,cw,ch);const pad={l:34,r:8,t:9,b:20},pw=cw-pad.l-pad.r,ph=ch-pad.t-pad.b;angleCtx.font='10px system-ui';angleCtx.textAlign='right';for(const deg of [-180,-90,0,90,180]){const y=pad.t+(180-deg)/360*ph;angleCtx.strokeStyle=deg===0?'#47666b':'#263c42';angleCtx.lineWidth=1;angleCtx.beginPath();angleCtx.moveTo(pad.l,y);angleCtx.lineTo(cw-pad.r,y);angleCtx.stroke();angleCtx.fillStyle='#99afb4';angleCtx.fillText(deg+'°',pad.l-5,y+3)}for(const [deg,label] of [[0,'右'],[90,'上'],[-90,'下'],[180,'左']]){const y=pad.t+(180-deg)/360*ph;angleCtx.fillStyle='#d6e4e5';angleCtx.textAlign='left';angleCtx.fillText(label,cw-pad.r-17,y-3)}const start=now-8000;while(angleSamples.length&&angleSamples[0].t<start)angleSamples.shift();const x=t=>pad.l+(t-start)/8000*pw,y=a=>pad.t+(180-a)/360*ph;angleCtx.strokeStyle='#64e0bc';angleCtx.lineWidth=2;angleCtx.beginPath();let pen=false;for(const p of angleSamples){if(p.a==null){pen=false;continue}if(!pen){angleCtx.moveTo(x(p.t),y(p.a));pen=true}else angleCtx.lineTo(x(p.t),y(p.a))}angleCtx.stroke();const current=angleSamples.at(-1);if(current&&current.a!=null){angleCtx.fillStyle='#ffd166';angleCtx.beginPath();angleCtx.arc(x(current.t),y(current.a),3,0,Math.PI*2);angleCtx.fill()}};
  const sampleAngle=now=>{if(now-lastAngleSample<40)return;lastAngleSample=now;const active=zone.dataset.active==='true';const raw=active?latestSegment?.angle:null;angleSamples.push({t:now,a:Number.isFinite(raw)?((raw+180)%360+360)%360-180:null});drawAngle(now)};window.addEventListener('resize',()=>drawAngle(performance.now()),{passive:true});

  const overlay=$('stick-angle-overlay'), overlayCtx=overlay.getContext('2d'), trail=[];
  let latestSegment=null;
  const collectSegment=segment=>{
    latestSegment=segment;
    if(!segment){trail.length=0;angleSamples.push({t:performance.now(),a:null});return;}
    if(!trail.length)trail.push(segment.from);
    trail.push(segment.to);
    if(trail.length>160)trail.shift();
  };
  function drawStickAngle(){
    const rect=zone.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
    const w=Math.round(rect.width*dpr), h=Math.round(rect.height*dpr);
    if(overlay.width!==w||overlay.height!==h){overlay.width=w;overlay.height=h;}
    const ctx=overlayCtx;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,rect.width,rect.height);
    const label=$('stick-angle-label');
    if(zone.dataset.active!=='true'||!latestSegment){
      label.textContent='滑动手指以查看轨迹角和移动方向';
      return;
    }
    const {from,to,angle,turn,direction}=latestSegment;
    ctx.strokeStyle='#71caff';ctx.lineWidth=2;ctx.beginPath();
    trail.forEach((p,i)=>i?ctx.lineTo(p.x-rect.left,p.y-rect.top):ctx.moveTo(p.x-rect.left,p.y-rect.top));
    ctx.stroke();
    ctx.strokeStyle='#64e0bc';ctx.lineWidth=4;ctx.beginPath();
    ctx.moveTo(from.x-rect.left,from.y-rect.top);ctx.lineTo(to.x-rect.left,to.y-rect.top);ctx.stroke();
    const turnText=turn==null?'—':`${Math.round(turn)}°`;
    label.textContent=`轨迹 ${Math.round(angle)}° · 转角 ${turnText} · 控制方向 ${keyName(direction)}`;
  }
  const render=()=>{const canvas=$('board'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,cw=w/W,ch=h/H;ctx.clearRect(0,0,w,h);ctx.fillStyle='#152b30';ctx.fillRect(0,0,w,h);for(let r=0;r<H;r++)for(let c=0;c<W;c++){const x=c*cw,y=r*ch;ctx.fillStyle=(r+c)%2?'#1b3739':'#193235';ctx.fillRect(x,y,cw,ch);ctx.strokeStyle='#25484a';ctx.strokeRect(x+.5,y+.5,cw-1,ch-1);const i=r*W+c;if(sim.wall[i]){ctx.fillStyle='#315862';ctx.fillRect(x,y,cw,ch);ctx.fillStyle='#4c7880';ctx.fillRect(x+6,y+6,cw-12,5);ctx.fillStyle='#203e45';ctx.fillRect(x+6,y+ch-11,cw-12,5)}else if(sim.brick[i]){ctx.fillStyle='#8a5540';ctx.fillRect(x,y,cw,ch);ctx.strokeStyle='#c47a55';ctx.beginPath();ctx.moveTo(x+4,y+ch*.52);ctx.lineTo(x+cw-4,y+ch*.52);ctx.moveTo(x+cw*.5,y+5);ctx.lineTo(x+cw*.5,y+ch*.5);ctx.stroke()}if(sim.fuse[i]>0){ctx.fillStyle='#edbd58';ctx.beginPath();ctx.arc(x+cw/2,y+ch/2,Math.min(cw,ch)*.2,0,Math.PI*2);ctx.fill();ctx.fillStyle='#fff0a0';ctx.fillRect(x+cw*.67,y+ch*.15,4,10)}}const px=sim.pos[1]*cw,py=sim.pos[0]*ch;ctx.fillStyle='#0a1719aa';ctx.beginPath();ctx.ellipse(px,py+14,16,6,0,0,Math.PI*2);ctx.fill();ctx.fillStyle='#64e0bc';ctx.beginPath();ctx.arc(px,py,15,0,Math.PI*2);ctx.fill();ctx.fillStyle='#e8ffff';ctx.beginPath();ctx.arc(px-5,py-4,3,0,Math.PI*2);ctx.arc(px+5,py-4,3,0,Math.PI*2);ctx.fill();ctx.fillStyle='#193b40';ctx.fillRect(px-9,py+5,18,4);const half=NATIVE_HALF_PX/NATIVE_CELL_PX;ctx.strokeStyle='#a0ffdc';ctx.lineWidth=1;ctx.strokeRect((sim.pos[1]-half)*cw,(sim.pos[0]-half)*ch,2*half*cw,2*half*ch);$('position').textContent=`${sim.pos[1].toFixed(2)}, ${sim.pos[0].toFixed(2)}`};
  const bindControls=()=>{if(bind){bind();listenerScope.abort();}held.clear();listenerScope=new AbortController();bind=QQTControls.bindTouchControls(zone,held,()=>{},{...config,trajectory,signal:listenerScope.signal,onTrajectorySegment:collectSegment})};const restart=()=>{sim=createSim();render()};const keyName=k=>({ArrowUp:'上',ArrowDown:'下',ArrowLeft:'左',ArrowRight:'右'})[k]||'无';const log=s=>{$('log').textContent=(new Date().toLocaleTimeString()+' '+s+'\n'+$('log').textContent).slice(0,4000)};
  function update(){const now=performance.now();if(zone.dataset.active==='true'){$('angle').textContent=$('angle-current').textContent=latestSegment?zone.dataset.rawAngle+'°':'—';$('radius').textContent=latestSegment?zone.dataset.rawRadius+' px':'—';for(const [id,val] of [['primary',zone.dataset.direction],['secondary',zone.dataset.secondary],['resolved',zone.dataset.resolved]])$(id).textContent=keyName(val)}else{for(const id of ['angle','angle-current','radius','primary','secondary','resolved'])$(id).textContent='—';}sampleAngle(now);const move=bind.moveFor(sim,0,now);sim.frameStep(0,move,bind.frameSeconds(1/60));drawStickAngle();if(move!==4){const d=Object.entries(QQTControls.MOVEMENT_DIRECTIONS).find(([,v])=>v===move)?.[0];$('resolved').textContent=keyName(d);if(d!==$('resolved').dataset.last){log('控制方向 '+keyName(d));$('resolved').dataset.last=d}}render();requestAnimationFrame(update)}
  for(const input of Object.values(controls))input.addEventListener('input',()=>{config=read();updateLabels();save(config);bindControls();restart()});$('diagonalMode').addEventListener('change',()=>{config=read();save(config);bindControls();restart()});$('restart').addEventListener('click',restart);$('reset').addEventListener('click',()=>{apply(defaults);config=read();bindControls();restart();$('status').textContent='已恢复默认'});$('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(JSON.stringify(read(),null,2));$('status').textContent='配置已复制'}catch(_){$('status').textContent='请使用下载配置'}});$('download').addEventListener('click',()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(read(),null,2)+'\n'],{type:'application/json'}));a.download='qqt-joystick-profile.json';a.click();$('status').textContent='配置已下载'});
  apply(config);bindControls();restart();log('训练场就绪');requestAnimationFrame(update);
})();
