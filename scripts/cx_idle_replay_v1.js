'use strict';
// Offline conservative current-hazard certificate; no online policy or reward changes.
const fs=require('node:fs'),assert=require('node:assert/strict');
const Q=require('./native_eval_simulator_v1'),F=require('./cc_strong_metrics_fountain_v1');
const N=195,W=15,H=13,INF=1e6,DIRS=[[-1,0],[1,0],[0,-1],[0,1]];
const neighbors=i=>DIRS.map(([r,c])=>[Math.floor(i/W)+r,i%W+c]).filter(([r,c])=>r>=0&&r<H&&c>=0&&c<W).map(([r,c])=>r*W+c);
function hazards(s){
 const mask=new Uint8Array(N),owners=[new Uint8Array(N),new Uint8Array(N)];
 for(let i=0;i<N;i++){
  if(s.blastLinger[i]>0||s.fieldItem[i]||s.lastReplayCovered?.[i])mask[i]=1;
  if(s.fuse[i]<=0)continue;
  const p=s.owner[i],blast=s.bombBlast[i]>0?s.bombBlast[i]:Q.CFG.blast;
  mask[i]=1;if(p>=0&&p<2)owners[p][i]=1;
  // Ignore destructible barriers AND other bombs. Superset of all current bomb rays even after chains destroy bricks.
  for(const [dr,dc] of DIRS)for(let k=1;k<=blast;k++){
   const r=Math.floor(i/W)+dr*k,c=i%W+dc*k;if(r<0||r>=H||c<0||c>=W)break;
   const j=r*W+c;if(s.wall[j])break;mask[j]=1;if(p>=0&&p<2)owners[p][j]=1;
  }
 }
 return {mask,owners};
}
function targets(s,p,blocked){
 const out=[];const enemy=1-p,[er,ec]=s.centerCell(enemy);
 for(let i=0;i<N;i++){
  if(blocked[i])continue;let yes=false;const r=Math.floor(i/W),c=i%W;
  for(const [dr,dc] of DIRS)for(let k=1;k<=s.blastCap[p];k++){
   const y=r+dr*k,x=c+dc*k;if(y<0||y>=H||x<0||x>=W)break;
   const j=y*W+x;if(s.wall[j])break;
   if(s.brick[j]||(s.alive[enemy]&&s.trapped[enemy]===0&&y===er&&x===ec)){yes=true;break;}
   if(s.fuse[j]>1)break;
  }
  if(yes)out.push(i);
 }
 return out;
}
function field(goals,blocked){const d=new Int16Array(N).fill(30000),q=[];for(const i of goals){d[i]=0;q.push(i);}
 for(let k=0;k<q.length;k++)for(const j of neighbors(q[k]))if(!blocked[j]&&d[j]>d[q[k]]+1){d[j]=d[q[k]]+1;q.push(j);}return d;}
function potential(y,x,d,blocked){
 const r=Math.floor(y),c=Math.floor(x),at=r*W+c;if(d[at]>=30000)return INF;
 let best=d[at]+Math.abs(y-r-.5)+Math.abs(x-c-.5);
 for(const j of neighbors(at))if(!blocked[j]&&d[j]<30000)best=Math.min(best,d[j]+Math.abs(y-Math.floor(j/W)-.5)+Math.abs(x-j%W-.5));return best;
}
function sweptSafe(y,x,ny,nx,mask){const rad=Q.CFG.radius;
 for(let r=Math.max(0,Math.floor(Math.min(y,ny)-rad));r<=Math.min(H-1,Math.floor(Math.max(y,ny)+rad));r++)
  for(let c=Math.max(0,Math.floor(Math.min(x,nx)-rad));c<=Math.min(W-1,Math.floor(Math.max(x,nx)+rad));c++)if(mask[r*W+c])return false;
 return true;
}
function classify(s,p,ability){
 const live=s.liveBombs(p),left=s.bombsCap[p]-live;
 if(!s.alive[p]||s.trapped[p]>0)return {kind:'d',reason:!s.alive[p]?'dead_respawn':'controlled'};
 if(ability!==0)return {kind:'d',reason:'active_ability_while_stationary'};
 if(live>=s.bombsCap[p])return {kind:'a',reason:'capacity_full_wait'};
 if(s.bunCarried[p]>=0)return {kind:'d',reason:'carrying_protected'};
 if(s.movementStatus[p]!==0)return {kind:'d',reason:'movement_status_uncertain'};
 const blocked=new Uint8Array(N);for(let i=0;i<N;i++)blocked[i]=Number(s.wall[i]||s.brick[i]||s.fuse[i]>1||s.pushable[i]);
 const goals=targets(s,p,blocked),d=field(goals,blocked),y=s.pos[2*p],x=s.pos[2*p+1],at=Math.floor(y)*W+Math.floor(x);
 if(!goals.length||d[at]>=30000)return {kind:'d',reason:'no_reachable_enemy_or_brick_target'};
 if(d[at]===0)return {kind:'d',reason:'already_at_attack_position_no_advance_proof'};
 const h=hazards(s),safeHere=sweptSafe(y,x,y,x,h.mask),before=potential(y,x,d,blocked),moves=[];
 for(let a=0;a<4;a++){
  // _steer is the production pure model movement adapter; never call whole alternative step on live replay.
  const [ny,nx]=s._steer(y,x,a,blocked,Q.CFG.stepLen*s.spdG[p]*s.playerMoveScale(p),p);
  if(Math.abs(ny-y)+Math.abs(nx-x)<1e-8)continue;
  if(potential(ny,nx,d,blocked)>=before-1e-5)continue;
  moves.push({action:a,ny,nx,safe:safeHere&&sweptSafe(y,x,ny,nx,h.mask)});
 }
 const advance=moves.find(m=>m.safe);
 if(left>0&&advance)return {kind:'c',reason:'certified_current_hazard_free_advance',action:advance.action,target_distance:d[at],left};
 if(safeHere&&moves.length&&moves.every(m=>!m.safe))return {kind:'b',reason:'safe_wait_progress_routes_cross_existing_hazard'};
 return {kind:'d',reason:safeHere?'no_strict_progress_action':'current_hazard_exposure_uncertain'};
}
function empty(){return {ticks:0,alive:0,actionable:0,stationary:0,stationary_alive:0,stationary_actionable:0,idle_action:0,max_stationary:0,run:0,max_stationary_alive:0,alive_run:0,max_stationary_actionable:0,actionable_run:0,max_c_run:0,c_run:0,
 categories:{a:0,b:0,c:0,d:0},reasons:{},capacity_hist:{},blast_hist:{},c_capacity_hist:{},c_blast_hist:{},pickups:[0,0,0],capacity_ticks:0,blast_ticks:0,examples:[],segments:[]};}
let current=null;
function install(){F.install();const step=Q.Sim.prototype.step;
 Q.Sim.prototype.step=function(actions){
  if(!this.__ftn)return step.call(this,actions);
  if(!current)current={players:[empty(),empty()]};
  const pre=this.pos.slice(),wasAlive=this.alive.slice(),wasTrap=this.trapped.slice(),cap=this.bombsCap.slice(),blast=this.blastCap.slice(),speed=this.spdG.slice();
  const oldBlocked=new Uint8Array(N);for(let i=0;i<N;i++)oldBlocked[i]=Number(this.wall[i]||this.brick[i]||this.fuse[i]>1);
  const classifications=[0,1].map(p=>{
   const a=actions[p][0],y=pre[2*p],x=pre[2*p+1];
   const next=a===4||!wasAlive[p]||wasTrap[p]>0?[y,x]:this._steer(y,x,a,oldBlocked,Q.CFG.stepLen*this.spdG[p]*this.playerMoveScale(p),p);
   if(Math.abs(next[0]-y)+Math.abs(next[1]-x)>=1e-8)return {kind:'d',reason:'unexpected_native_stationary_unclassified'};
   return classify(this,p,current.actions[this.t][2*p+1]);
  });
  const info=step.call(this,actions);
  for(let p=0;p<2;p++){
   const m=current.players[p],stationary=Math.abs(this.pos[2*p]-pre[2*p])+Math.abs(this.pos[2*p+1]-pre[2*p+1])<1e-8;
   m.ticks++;m.alive+=Number(wasAlive[p]);m.actionable+=Number(wasAlive[p]&&!wasTrap[p]);m.idle_action+=Number(actions[p][0]===4);
   if(wasAlive[p]){m.capacity_ticks+=cap[p];m.blast_ticks+=blast[p];}
   m.pickups[0]+=Math.max(0,this.bombsCap[p]-cap[p]);m.pickups[1]+=Math.max(0,this.blastCap[p]-blast[p]);m.pickups[2]+=Math.max(0,this.spdG[p]-speed[p]);
   m.alive_run=stationary&&wasAlive[p]?m.alive_run+1:0;m.max_stationary_alive=Math.max(m.max_stationary_alive,m.alive_run);
   m.actionable_run=stationary&&wasAlive[p]&&!wasTrap[p]?m.actionable_run+1:0;m.max_stationary_actionable=Math.max(m.max_stationary_actionable,m.actionable_run);
   if(!stationary){if(m.run>=20)m.segments.push({start:this.t-m.run-1,end:this.t-1,length:m.run});m.run=0;m.c_run=0;continue;}
   m.stationary++;m.stationary_alive+=Number(wasAlive[p]);m.stationary_actionable+=Number(wasAlive[p]&&!wasTrap[p]);m.run++;m.max_stationary=Math.max(m.run,m.max_stationary);
   const r=classifications[p];
   if(!m.category_segments)m.category_segments=[];
   const seg=m.category_segments[m.category_segments.length-1],tick=this.t-1;
   if(seg&&seg.kind===r.kind&&seg.end===tick)seg.end=tick+1;
   else m.category_segments.push({start:tick,end:tick+1,kind:r.kind});
   m.categories[r.kind]++;m.reasons[r.reason]=(m.reasons[r.reason]||0)+1;
   m.capacity_hist[cap[p]]=(m.capacity_hist[cap[p]]||0)+1;m.blast_hist[blast[p]]=(m.blast_hist[blast[p]]||0)+1;
   if(r.kind==='c'){
    m.c_run++;m.max_c_run=Math.max(m.max_c_run,m.c_run);m.c_capacity_hist[cap[p]]=(m.c_capacity_hist[cap[p]]||0)+1;m.c_blast_hist[blast[p]]=(m.c_blast_hist[blast[p]]||0)+1;
    if(m.examples.length<5)m.examples.push({tick:this.t-1,...r,capacity:cap[p],blast:blast[p]});
   }else m.c_run=0;
  }
  return info;
 };
}
function replay(file){const d=JSON.parse(fs.readFileSync(file));current={players:[empty(),empty()],actions:d.actions};const result=F.replay(file);
 for(const m of current.players){if(m.run>=20)m.segments.push({start:d.actions.length-m.run,end:d.actions.length,length:m.run});delete m.run;delete m.c_run;delete m.alive_run;delete m.actionable_run;assert.equal(Object.values(m.categories).reduce((a,b)=>a+b,0),m.stationary);}
 result.idle=current.players;delete result.timeline;return result;}
function main(){const cfg=JSON.parse(fs.readFileSync(process.argv[2]));install();const rows=[];
 for(const file of cfg.files){try{rows.push(replay(file));}catch(e){rows.push({file,error:String(e.stack)});}}
 fs.writeFileSync(cfg.output+'.tmp',rows.map(r=>JSON.stringify(r)).join('\n')+'\n');fs.renameSync(cfg.output+'.tmp',cfg.output);
 console.log(JSON.stringify({files:rows.length,errors:rows.filter(r=>r.error).length}));assert(!rows.some(r=>r.error));}
if(require.main===module)main();module.exports={classify,hazards,targets,field,potential,sweptSafe};
