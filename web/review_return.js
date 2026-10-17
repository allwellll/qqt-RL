'use strict';
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.QQTReviewReturn=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const MOVES=['向上','向下','向左','向右','停留'],DIRS=[[-1,0],[1,0],[0,-1],[0,1]],SCHEMA='qqt.return-diagnostic/v1';
  const near=(a,b)=>a&&b&&a.every((v,i)=>Math.abs(v-b[i])<1e-6);
  function distances(s,pid=0){
    const d=Array(195).fill(32767),[r,c]=s.bunBases[pid];
    for(let y=r;y<r+3;y++)for(let x=c;x<c+3;x++)if(!s.wall[y*15+x])d[y*15+x]=0;
    for(let pass=0;pass<195;pass++){let changed=false;for(let i=0;i<195;i++)if(!s.wall[i])for(const [dy,dx]of DIRS){const y=Math.floor(i/15)+dy,x=i%15+dx;if(y<0||y>=13||x<0||x>=15)continue;const j=y*15+x;if(s.wall[j])continue;const v=d[j]+1+3*Number(Boolean(s.brick[j]));if(v<d[i]){d[i]=v;changed=true;}}if(!changed)break;}return d;
  }
  function path(s,d,pos=Array.from(s.pos).slice(0,2)){
    let y=Math.floor(pos[0]),x=Math.floor(pos[1]);const cells=[[y,x]],first=[];
    for(let n=0;n<195&&d[y*15+x]>0&&d[y*15+x]<32767;n++){
      const next=DIRS.map(([dy,dx],move)=>({y:y+dy,x:x+dx,move})).filter(v=>v.y>=0&&v.y<13&&v.x>=0&&v.x<15&&!s.wall[v.y*15+v.x]&&d[v.y*15+v.x]+1+3*Number(Boolean(s.brick[v.y*15+v.x]))===d[y*15+x]);
      if(!next.length)break;if(n===0)first.push(...next.map(v=>v.move));({y,x}=next[0]);cells.push([y,x]);
    }return {first,cells};
  }
  function classify(points,i){
    const p=points[i];if(!p||p.phase!=='C_RETURN'||!p.alive||!p.carrying||!Number.isFinite(p.distance))return {kind:null,axis:null};
    const window=points.slice(Math.max(0,i-9),i+1);
    if(window.length<10||window.some(v=>v.phase!==p.phase||v.target!==p.target||!v.alive||!v.carrying||v.distance!==p.distance))return {kind:null,axis:null};
    const periodic=window.every((v,j)=>j<2||near(v.pos,window[j-2].pos)),a=window[0].pos,b=window[1].pos;
    const dy=Math.abs(a[0]-b[0]),dx=Math.abs(a[1]-b[1]);
    if(periodic&&dy+dx>.02&&Math.min(dy,dx)<1e-6)return {kind:'return_oscillation',axis:dy>dx?'vertical':'horizontal'};
    if(window.every(v=>near(v.pos,a)))return {kind:'return_stagnation',axis:null};
    return {kind:null,axis:null};
  }
  function analyze(doc,level,P){
    if(!doc.route)return {points:[],intervals:[]};
    const player=new P.Player(doc,level),points=[],intervals=[];let topology='',d,active=null;
    for(let t=0;t<=doc.actions.length;t++){
      if(t)player.step();const s=player.sim,rIndex=doc.route.ticks.indexOf(t),r=doc.route;
      const key=Array.from(s.wall).join('')+Array.from(s.brick).join('');if(key!==topology){d=distances(s);topology=key;}
      const p={tick:t,pos:Array.from(s.pos).slice(0,2),phase:rIndex<0?null:r.route_phase[rIndex],target:rIndex<0?null:r.target[rIndex],distance:rIndex<0?null:r.distance_after[rIndex],alive:Boolean(s.alive[0]),carrying:s.bunCarried[0]===1,action:t?doc.actions[t-1].slice(0,2):null,path_first_moves:path(s,d).first};
      const prior=points[t-1],before=rIndex<0?null:r.distance_before[rIndex];
      p.no_progress_ticks=p.phase==='C_RETURN'&&p.alive&&p.carrying&&Number.isFinite(p.distance)&&p.distance>=before?(prior?.phase===p.phase&&prior.target===p.target?(prior.no_progress_ticks||0)+1:1):0;
      points.push(p);Object.assign(p,classify(points,t));if(p.no_progress_ticks<20)p.kind=null;
      if(p.kind){if(!active||active.kind!==p.kind||active.end!==t-1){active={kind:p.kind,axis:p.axis,start:t,end:t,observed_start:Math.max(1,t-9),distance:p.distance};intervals.push(active);}else active.end=t;}else active=null;
    }return {points,intervals};
  }
  function describe(p){
    if(!p||p.phase!=='C_RETURN')return '未采集返程路线，不能判断有效靠近';
    const progress=`当前持敌包返己笼 · 连续 ${p.no_progress_ticks} tick 路线距离未下降 · 距离 ${p.distance??'未记录'}`;
    return progress+(p.kind==='return_oscillation'?` · ${p.axis==='vertical'?'上下':'左右'}往返 · 返程振荡失败`:p.kind==='return_stagnation'?' · 位置不变 · 返程停滞失败':p.no_progress_ticks?' · 诊断目标已确定，尚未观察到距离下降':' · 本 tick 路线距离下降');
  }
  function comparison(original,shaped){return original?.advantage==null||shaped?.advantage==null?null:shaped.advantage-original.advantage;}
  const temporalDelta=(previous,current)=>previous==null||current==null?null:current-previous;
  function validatePair(original,shaped){if(original.role!=='original'||shaped.role!=='shaped')throw Error('critic role');if(original.critic===shaped.critic||original.checkpoint===shaped.checkpoint)throw Error('distinct critic/checkpoint required');}
  function validateAudit(a,doc,replayHash){
    const fail=()=>{throw Error('return audit binding / schema');};
    if(a?.schema!==SCHEMA||a.replay_id!==doc.meta.id||a.replay_sha256!==replayHash||a.policy_sha256!==doc.meta.model_sha256||a.checkpoint_sha256!==doc.meta.checkpoint_sha256||a.simulator_sha256!==doc.meta.simulator_sha256||!Array.isArray(a.rows)||!Array.isArray(a.intervals))fail();
    let prior=0;for(const r of a.rows){if(!Number.isInteger(r.tick)||r.tick<=prior||r.tick>doc.actions.length||JSON.stringify(r.taken)!==JSON.stringify(doc.actions[r.tick-1].slice(0,2))||!Array.isArray(r.move_probabilities)||r.move_probabilities.length!==5||r.move_probabilities.some(v=>!Number.isFinite(v)||v<0||v>1)||Math.abs(r.move_probabilities.reduce((n,v)=>n+v,0)-1)>1e-6||!Array.isArray(r.candidates)||r.candidates.some(c=>!Number.isFinite(c.probability)||c.q_original!==null||c.q_shaped!==null)||r.original_a1!==null||r.shaped_a1!==null||r.cross_critic_delta!==null)fail();prior=r.tick;}
    for(const r of a.rows){
      if(!Array.isArray(r.move_mask)||r.move_mask.length!==5||r.move_mask.some(v=>v!==0&&v!==1)||!Array.isArray(r.ability_mask)||r.ability_mask.length!==3||r.ability_mask.some(v=>v!==0&&v!==1)||!Array.isArray(r.ability_probabilities)||r.ability_probabilities.length!==3||r.ability_probabilities.some(v=>!Number.isFinite(v)||v<0||v>1)||Math.abs(r.ability_probabilities.reduce((n,v)=>n+v,0)-1)>1e-6)fail();
      if(r.move_probabilities.some((v,i)=>!r.move_mask[i]&&v!==0)||r.ability_probabilities.some((v,i)=>!r.ability_mask[i]&&v!==0)||r.candidates.length!==r.move_mask.reduce((n,v)=>n+v,0)*r.ability_mask.reduce((n,v)=>n+v,0))fail();
      const seen=new Set();for(const c of r.candidates){const key=JSON.stringify(c.action);if(seen.has(key)||!Array.isArray(c.action)||c.action.length!==2||!r.move_mask[c.action[0]]||!r.ability_mask[c.action[1]]||Math.abs(c.probability-r.move_probabilities[c.action[0]]*r.ability_probabilities[c.action[1]])>3e-9||!/^[a-f0-9]{64}$/.test(c.state_sha256)||!Array.isArray(c.position)||c.position.length!==2||c.position.some(v=>!Number.isFinite(v)))fail();seen.add(key);}
    }
    if(a.advantage_audit.status!=='unavailable'||a.advantage_audit.original.role!=='original'||a.advantage_audit.shaped.role!=='shaped'||!a.advantage_audit.missing_links.length)fail();
    for(const role of ['original','shaped'])for(const key of ['checkpoint','critic','state'])if(!/^[a-f0-9]{64}$/.test(a.advantage_audit[role][key]))fail();
    if(a.advantage_audit.shaped.checkpoint!==doc.meta.checkpoint_sha256||a.advantage_audit.shaped.critic!==doc.meta.critic_sha256)fail();
    validatePair(a.advantage_audit.original,a.advantage_audit.shaped);return a;
  }
  return {SCHEMA,MOVES,DIRS,distances,path,classify,analyze,describe,comparison,temporalDelta,validatePair,validateAudit};
});
