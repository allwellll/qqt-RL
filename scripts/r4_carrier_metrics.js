'use strict';
// Versioned passive measurements for STEER_2225. No action/RNG/rule mutation.
const {field}=require('./r3_items_metrics');
const W=15,N=195,INF=1e6;
const cell=(s,p)=>Math.floor(s.pos[p*2])*W+Math.floor(s.pos[p*2+1]);
const neighbors=i=>[i>=W?i-W:-1,i%W<14?i+1:-1,i<N-W?i+W:-1,i%W>0?i-1:-1].filter(z=>z>=0);
function cage(s,t){const [r,c]=s.bunBases[t],v=[];for(let a=r;a<r+3;a++)for(let b=c;b<c+3;b++)v.push(a*W+b);return v;}
function route(s,p,cache={}){
 const signature=Array.from({length:N},(_,i)=>Number(Boolean(s.wall[i]||s.brick[i]||s.pushable[i]))).join('')+':'+s.team[p];
 if(cache.signature!==signature){cache.signature=signature;cache.distance=field(s,cage(s,s.team[p])).distance;}
 const d=cache.distance,out=[],start=cell(s,p);if(d[start]>=INF)return null;
 let at=start;out.push(at);
 while(d[at]>0){const next=neighbors(at).find(z=>d[z]===d[at]-1);if(next===undefined)throw Error('invalid shortest path');at=next;out.push(at);}
 return out;
}
function nearPath(at,path){return path.some(z=>Math.abs(Math.floor(at/W)-Math.floor(z/W))+Math.abs(at%W-z%W)<=3);}
function init(s){return s.__r4Carrier ||= {version:2,carrier_ticks:[0,0],own_actionable_ticks:[0,0],
 enemy_free_ticks:[0,0],reachable_path_ticks:[0,0],interception_ticks:[0,0],free_interception_ticks:[0,0],
 exposed_carrier_lives:[0,0],carrier_deaths:[0,0],carrier_causal_kills:[0,0],enemy_deliveries:[0,0],
 own_deliveries:[0,0],stock_pickups:[0,0],completed_antifarm_cycles:[0,0],
 life:[0,0],seen_lives:[new Set(),new Set()],cycle:[null,null],cache:[{},{}],events:[]};}
function beforeStep(s){
 const x=init(s),snapshot={carried:s.bunCarried.slice(),alive:s.alive.slice()};
 for(let p=0;p<2;p++){
  const q=1-p;if(!s.alive[q]||s.bunCarried[q]!==s.team[p])continue;
  x.carrier_ticks[p]++;
  if(!x.seen_lives[p].has(x.life[q])){x.seen_lives[p].add(x.life[q]);x.exposed_carrier_lives[p]++;}
  const active=s.alive[p]&&s.trapped[p]===0;if(!active)continue;
  x.own_actionable_ticks[p]++;const free=s.trapped[q]===0;if(free)x.enemy_free_ticks[p]++;
  const path=route(s,q,x.cache[q]);if(path){x.reachable_path_ticks[p]++;
   if(nearPath(cell(s,p),path)){x.interception_ticks[p]++;if(free)x.free_interception_ticks[p]++;}}
 }
 return snapshot;
}
function afterStep(s,b,r){
 const x=init(s);
 for(let p=0;p<2;p++){
  const q=1-p;
  if(r.died?.[q]&&b.carried[q]===s.team[p]){
   x.carrier_deaths[p]++;
   const killed=Boolean(r.causalKill?.[p]);
   if(killed){x.carrier_causal_kills[p]++;if(x.cycle[p])x.cycle[p].killed=true;}
   x.events.push({tick:s.t,type:'carrier_death',learner:p,victim:q,causal_kill:killed,origin:b.carried[q]});
  }
 }
 for(let q=0;q<2;q++)if(r.died?.[q])x.life[q]++;
}
function stockUpdate(s,invoke){
 const x=init(s),old=s.bunStored.map(a=>a.slice()),carried=s.bunCarried.slice(),result=invoke();
 for(let p=0;p<2;p++){
  const q=1-p,t=s.team[p],e=s.team[q];
  if(old[t][t]>s.bunStored[t][t]&&s.bunCarried[q]===t){
   x.stock_pickups[p]++;x.cycle[p]={pickup_tick:s.t,killed:false};
   x.events.push({tick:s.t,type:'own_stock_taken',learner:p,actor:q,origin:t});
  }
  if(s.bunStored[e][t]>old[e][t]&&carried[q]===t){
   x.enemy_deliveries[p]+=s.bunStored[e][t]-old[e][t];x.cycle[p]=null;
   x.events.push({tick:s.t,type:'enemy_delivery',learner:p,origin:t});
  }
  if(s.bunStored[t][e]>old[t][e]&&carried[p]===e){x.own_deliveries[p]+=s.bunStored[t][e]-old[t][e];}
  if(s.bunStored[t][t]>old[t][t]&&carried[p]===t){
   if(x.cycle[p]?.killed){x.completed_antifarm_cycles[p]++;
    x.events.push({tick:s.t,type:'antifarm_cycle_completed',learner:p,pickup_tick:x.cycle[p].pickup_tick});}
   x.cycle[p]=null;
  }
 }
 return result;
}
function exportMetrics(s){const {life,seen_lives,cycle,cache,...out}=init(s);return {...out,
 unresolved_cycles:cycle.map(Boolean),
 definitions:{interception:'pre-action own alive/uncontrolled within Manhattan3 of canonical unobstructed shortest enemy-home path; tie order up/right/down/left; geometry, not intent',
  interception_denominator:'own actionable carrier ticks; no-path ticks remain in denominator, and reachable ticks are reported separately',
  killed_rate:'causal killed own-origin carriers / enemy lives with observed own-origin possession',
  cycles:'observed original own stock take -> causal kill of that carrier -> native return own-origin package to own stock',
  exposure:'pre-first-terminal only; one sample per physical tick'},
 limitations:['geometric interception does not prove a useful block','same life can contain multiple possessions; life denominator is deduplicated',
 'deaths use pre-action possession; pickup-and-death within the same tick is not inferred','no native JAX reward ledger in production JS']};}
function install(Q){
 const P=Q.Sim.prototype,step=P.step,update=P._bunUpdate;
 if(step.__r4Carrier)throw Error('duplicate carrier metric installation');
 P._bunUpdate=function(...args){return this.__btIdentity?stockUpdate(this,()=>update.apply(this,args)):update.apply(this,args);};
 function wrapped(actions){
  if(!this.__btIdentity||this.done)return step.call(this,actions);
  const before=beforeStep(this),r=step.call(this,actions);afterStep(this,before,r);
  if(this.done&&this.__bt)this.__bt.r4_carrier=exportMetrics(this);return r;
 }
 wrapped.__r4Carrier=true;P.step=wrapped;
}
module.exports={install,route,nearPath,init,beforeStep,afterStep,stockUpdate,exportMetrics};
