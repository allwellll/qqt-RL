'use strict';
// Passive production-Sim diagnostics. Event observations are never called Q or
// causal treatment effects. No action, RNG, rule, or observation mutation.
const N=195,W=15,INF=1e6;
const cell=(s,p)=>Math.floor(s.pos[2*p])*W+Math.floor(s.pos[2*p+1]);
function neighbors(i){const r=Math.floor(i/W),c=i%W;return [r>0?i-W:-1,r<12?i+W:-1,c>0?i-1:-1,c<14?i+1:-1].filter(x=>x>=0);}
function field(s,goals,weighted=false){
 const d=Array(N).fill(INF),bricks=Array(N).fill(INF),q=[];
 for(const i of goals)if(!s.wall[i]&&!s.pushable[i]){d[i]=0;bricks[i]=0;q.push(i);}
 while(q.length){q.sort((a,b)=>d[b]-d[a]);const i=q.pop();
  for(const k of neighbors(i)){if(s.wall[k]||s.pushable[k]||(!weighted&&s.brick[k]))continue;
   const nd=d[i]+1+(weighted&&s.brick[i]?8:0);
   if(nd<d[k]){d[k]=nd;bricks[k]=bricks[i]+Number(Boolean(s.brick[i]));q.push(k);}
  }
 }
 return {distance:d,bricks};
}
function cage(s,t){const [r,c]=s.bunBases[t],out=[];for(let a=r;a<r+3;a++)for(let b=c;b<c+3;b++)out.push(a*W+b);return out;}
function route(s){const goals=cage(s,1),start=cage(s,0),f=field(s,goals),w=field(s,goals,true);
 const min=(x)=>Math.min(...start.map(i=>x[i]));const d=min(f.distance),weighted=min(w.distance);
 // Separate lexicographic search for minimum brick count, not brick count on
 // the 1+8*brick path (which can differ).
 const b=Array(N).fill(INF),q=[...goals];for(const i of goals)b[i]=0;
 while(q.length){const i=q.shift();for(const k of neighbors(i))if(!s.wall[k]&&!s.pushable[k]){const v=b[i]+Number(Boolean(s.brick[i]));if(v<b[k]){b[k]=v;q.push(k);}}}
 return {bfs_distance:d>=INF?null:d,minimum_bricks:min(b)>=INF?null:min(b),weighted_distance:weighted>=INF?null:weighted};
}
function init(s){
 if(s.__r3)return s.__r3;
 return s.__r3={version:1,pickups:[Array(5).fill(0),Array(5).fill(0)],releases:[[0,0],[0,0]],
  contacts:[],slides:[],pending_slides:[null,null],bombs:new Map(),last_placement:[-1000,-1000],batch:[0,0],
  live_hist:[[],[]],batch_hist:[[],[]],alive_ticks:[0,0],zero_bomb_ticks:[0,0],chains:[],
  route_initial:route(s),route_final:null,path_bricks:[0,0],path_brick_cells:[[],[]]};
}
function exportMetrics(s){const x=init(s);return {version:x.version,pickups:x.pickups,releases:x.releases,
 contacts:x.contacts,slides:x.slides,unfinished_slides:x.pending_slides.map(Boolean),
 live_hist:x.live_hist,batch_hist:x.batch_hist,alive_ticks:x.alive_ticks,zero_bomb_ticks:x.zero_bomb_ticks,
 chains:x.chains,route_initial:x.route_initial,route_final:route(s),path_bricks:x.path_bricks,path_brick_cells:x.path_brick_cells,
 attributes:[0,1].map(p=>[s.bombsCap[p],s.blastCap[p],s.spdG[p]]),
 limitations:['glue subsequent outcomes observational, not counterfactual causal benefit',
 'banana time saved is same traversed polyline at native carry speed estimate, not matched action replay',
 'banana home progress must be positive; sideways/backtracking included in actual slide cost',
 'chain batches are verified triggering-ray edges; independent simultaneous explosions excluded',
 'single map806 birth geometries; production Sim.step only']};}
function install(Q){
 const P=Q.Sim.prototype;
 for(const name of ['_collectCrate','_placeHeldItem','_contactFieldItem','_resolveExplosions','step'])if(typeof P[name]!=='function')throw Error('missing '+name);
 const collect=P._collectCrate;P._collectCrate=function(p,i){const type=this.crateType[i],r=collect.call(this,p,i);if(this.__btIdentity&&r&&type>=0&&type<5)init(this).pickups[p][type]++;return r;};
 const place=P._placeHeldItem;P._placeHeldItem=function(p,...args){const type=this.heldItem[p],r=place.call(this,p,...args);if(this.__btIdentity&&r&&type>0)init(this).releases[p][type-1]++;return r;};
 const contact=P._contactFieldItem;P._contactFieldItem=function(p,move,i=null){const at=i??cell(this,p),type=this.fieldItem[at],owner=this.fieldOwner[at],x=this.__btIdentity?init(this):null;
  const before=x?{tick:this.t,player:p,owner,type,cell:at,carried:this.bunCarried[p],pos:Array.from(this.pos.slice(2*p,2*p+2)),
   distance:field(this,cage(this,this.team[p])).distance[cell(this,p)],enemy_distance:field(this,[cell(this,1-p)]).distance[cell(this,p)],
   armed:Boolean(this.fieldArmed[at]),kills:(this.__bt?.kills||[]).length,stock:this.__anti?.stock_pickups?.slice()||[0,0]}:null;
  const r=contact.call(this,p,move,i);
  if(x&&type&&this.fieldItem[at]===0){
   const goals=owner>=0?cage(this,this.team[owner]):[];for(let z=0;z<N;z++)if(this.bunLoose[z*2]||this.bunLoose[z*2+1])goals.push(z);
   const f=field(this,goals),old=x.previous_cells?.[p]??cell(this,p),now=cell(this,p);
   before.toward_package_geodesic=f.distance[now]<INF&&f.distance[now]<f.distance[old];
   const chase=field(this,[cell(this,1-p)]);
   before.chase_geometry_proxy=before.enemy_distance<=4&&chase.distance[now]<chase.distance[old];
   before.critical_path=before.carried>=0||before.toward_package_geodesic||before.chase_geometry_proxy;
   x.contacts.push(before);if(type===1&&owner===p&&before.carried>=0){
    if(x.pending_slides[p]){x.pending_slides[p].end_reason='retrigger';x.slides.push(x.pending_slides[p]);}
    x.pending_slides[p]={...before,start_tick:this.t,start_pos:before.pos,path_cells:0,last_pos:before.pos,end_tick:null,saved_ticks_estimate:null};}}
  return r;
 };
 const explosions=P._resolveExplosions;P._resolveExplosions=function(...args){
  if(!this.__btIdentity)return explosions.apply(this,args);
  const x=init(this),fuse=Array.from(this.fuse),owners=Array.from(this.owner),brick=Array.from(this.brick),r=explosions.apply(this,args);
  const sources=r.sources||[],edges=[];
  for(const src of sources)for(const target of sources)if(src.cell!==target.cell&&fuse[target.cell]>0&&src.covered[target.cell])edges.push([src.cell,target.cell]);
  // Count only chains reachable from a natural timer root through actual
  // pre-explosion rays. Batch IDs were recorded at real placement.
  for(let p=0;p<2;p++){
   const roots=sources.filter(a=>owners[a.cell]===p&&fuse[a.cell]===0),seen=new Set(roots.map(a=>a.cell));
   let changed=true;while(changed){changed=false;for(const [a,b] of edges)if(seen.has(a)&&owners[b]===p&&!seen.has(b)){seen.add(b);changed=true;}}
   const chained=[...seen].filter(i=>fuse[i]>0),batches=[...new Set([...seen].map(i=>x.bombs.get(i)?.batch).filter(i=>i!==undefined))];
   if(chained.length)x.chains.push({tick:this.t,player:p,roots:roots.map(a=>a.cell),triggered:chained,batches,batch_count:batches.length,edges:edges.filter(([a,b])=>seen.has(a)&&seen.has(b))});
   // Both distance fields must certify a brick lies on a shortest weighted
   // two-cage route before this explosion. Shared-cause bricks remain separate.
   const a=field(this,cage(this,0),true),b=field(this,cage(this,1),true),d=Math.min(...cage(this,0).map(i=>b.distance[i]));
   for(let i=0;i<N;i++)if(brick[i]&&d<INF&&a.distance[i]+b.distance[i]+8===d){
    const causes=sources.filter(z=>z.covered[i]);if(causes.some(z=>(z.causeMask??(1<<z.owner))===(1<<p))&&!causes.some(z=>((z.causeMask??(1<<z.owner))&(1<<(1-p)))!==0)){
     if(!x.path_brick_cells[p].includes(i)){x.path_brick_cells[p].push(i);x.path_bricks[p]++;}}}
  }
  for(const src of sources)x.bombs.delete(src.cell);
  return r;
 };
 const step=P.step;P.step=function(actions){
  if(!this.__btIdentity)return step.call(this,actions);
  const x=init(this),before=Array.from(this.fuse),wasTrapped=Array.from(this.trapped);x.previous_cells=[cell(this,0),cell(this,1)];const r=step.call(this,actions);
  for(let p=0;p<2;p++){
   if(r.placed[p])for(let i=0;i<N;i++)if(this.fuse[i]>0&&this.owner[i]===p&&before[i]===0){if(this.t-x.last_placement[p]>=8)x.batch[p]++;x.last_placement[p]=this.t;x.bombs.set(i,{batch:x.batch[p],tick:this.t,owner:p});}
   if(this.alive[p]){const ids=[];for(let i=0;i<N;i++)if(this.fuse[i]>0&&this.owner[i]===p)ids.push(i);x.alive_ticks[p]++;x.zero_bomb_ticks[p]+=Number(!ids.length);x.live_hist[p].push(ids.length);x.batch_hist[p].push(new Set(ids.map(i=>x.bombs.get(i)?.batch).filter(i=>i!==undefined)).size);}
   const slide=x.pending_slides[p];if(slide){const pos=Array.from(this.pos.slice(2*p,2*p+2));slide.path_cells+=Math.hypot(pos[0]-slide.last_pos[0],pos[1]-slide.last_pos[1]);slide.last_pos=pos;
    if(this.movementStatus[p]!==2||this.done||!this.alive[p]){slide.end_tick=this.t;slide.end_reason=this.done?'terminal':!this.alive[p]?'death':'completed';
     const end=field(this,cage(this,this.team[p])).distance[cell(this,p)];slide.home_progress=slide.distance<INF&&end<INF?slide.distance-end:null;
     slide.saved_ticks_estimate=slide.home_progress!==null&&slide.home_progress>0?Math.max(0,slide.path_cells/(72/40/10)-(this.t-slide.start_tick)):0;
     slide.blast_within30=false;x.slides.push(slide);x.pending_slides[p]=null;}}
   for(const sl of x.slides)if(sl.player===p&&this.t<=((sl.end_tick??this.t)+30)&&(r.died?.[p]||(!wasTrapped[p]&&this.trapped[p]>0)))sl.blast_within30=true;
   for(const c of x.contacts)if(c.type===2&&c.owner>=0&&c.owner!==c.player&&this.t-c.tick<=50){
    c.followup_kill ||= (this.__bt?.kills||[]).slice(c.kills).some(k=>k.attacker===c.owner);
    c.followup_pickup ||= (this.__anti?.stock_pickups?.[c.owner]??0)>c.stock[c.owner];c.observation_ticks=this.t-c.tick;}
  }
  if(this.__bt&&this.done)this.__bt.r3=exportMetrics(this);
  return r;
 };
}
module.exports={install,field,route,exportMetrics};
