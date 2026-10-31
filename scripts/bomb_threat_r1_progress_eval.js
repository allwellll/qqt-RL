'use strict';
// Passive per-bomb lifetime attribution. No model, RNG, actions or Sim edits.
const fs=require('node:fs'),Q=require('../web/sim-stock-window-v1'),live=new Set();
const step=Q.Sim.prototype.step,resolve=Q.Sim.prototype._resolveExplosions,grow=Q.Sim.prototype._grow;
const low=[2,2,1.3],high=[10,8,2.4],fraction=a=>a.map((v,k)=>Math.max(0,Math.min(1,(v-low[k])/(high[k]-low[k]))));
Q.Sim.prototype._grow=function(p,...args){
 const before=[this.bombsCap[p],this.blastCap[p],this.spdG[p]];
 const result=grow.call(this,p,...args),after=[this.bombsCap[p],this.blastCap[p],this.spdG[p]];
 if(this.__r1){const x=this.__r1,f=fraction(after),types=after.map((v,k)=>v>before[k]+1e-6?k:null).filter(v=>v!==null);
  const delta=f.map((v,k)=>types.includes(k)?Math.max(0,v-x.highwater[p][k]):0);
  x.highwater[p]=x.highwater[p].map((v,k)=>Math.max(v,types.includes(k)?f[k]:v));
  if(types.length)x.itemEvents.push({seat:p,tick:this.t,types,before,after,progress:delta,rawCandidate:.08*delta.reduce((a,b)=>a+b,0),bombs100:0,self100:0,void30:false});
 }return result;
};
function ray(sim,cell){const mark=new Uint8Array(195);mark[cell]=1;const blast=new Int16Array(sim.bombBlast);return sim._rays(mark,blast,sim.brick).covered;}
Q.Sim.prototype._resolveExplosions=function(...args){
 const x=resolve.apply(this,args);
 if(this.__r1)this.__r1.lastSources=x.sources.map(s=>{
   let brick=false;for(let c=0;c<195;c++)if(s.covered[c]&&this.brick[c])brick=true;
   const enemy=1-s.owner,e=Math.floor(this.pos[2*enemy])*15+Math.floor(this.pos[2*enemy+1]);
   return {cell:s.cell,owner:s.owner,brick,threat:Boolean(s.covered[e]&&this.alive[enemy])};
 });
 return x;
};
Q.Sim.prototype.step=function(actions){
 if(!this.__btIdentity)return step.call(this,actions);
 if(!this.__r1){this.__r1={...this.__btIdentity,ticks:0,close:0,closeEligible:0,phaseTicks:{combat:0,window:0,carrier:0,uncontrolled:0},bombs:new Map(),finished:[],itemEvents:[],highwater:[0,1].map(p=>fraction([this.bombsCap[p],this.blastCap[p],this.spdG[p]])),stats:[],kills:[0,0],self:[0,0]};live.add(this);}
 const startCells=[0,1].map(p=>{const [r,c]=this.centerCell(p);return this._actionCell(actions[p][4],r*15+c);});
 const x=this.__r1,beforeFuse=Array.from(this.fuse),beforeOwner=Array.from(this.owner),beforeCrate=Array.from(this.crate),
       stats=[0,1].map(p=>[this.bombsCap[p],this.blastCap[p],this.spdG[p]]);
 const oldRays=new Map();for(let c=0;c<195;c++)if(beforeFuse[c]>0)oldRays.set(c,ray(this,c));
 const info=step.call(this,actions);x.ticks++;
 const learner=x.seat,enemy=1-learner,canAct=this.alive[learner]&&this.trapped[learner]===0;
 const phase=!canAct?'uncontrolled':this.bunCarried[learner]>=0?'carrier':!this.alive[enemy]||this.trapped[enemy]>0?'window':'combat';
 x.phaseTicks[phase]++;
 const dist=Math.abs(this.pos[0]-this.pos[2])+Math.abs(this.pos[1]-this.pos[3]);
 x.close+=Number(dist<=3&&canAct&&this.alive[enemy]);x.closeEligible+=Number(canAct&&this.alive[enemy]);
 for(let p=0;p<2;p++){
   x.kills[p]+=Number(info.causalKill?.[p]||0);x.self[p]+=Number(info.ownBombDefeat?.[p]||0);
   if(info.placed[p]){
     const cell=startCells[p];
     const connected=[...oldRays].some(([c,r])=>c!==cell&&beforeOwner[c]===p&&r[cell]);
     const b={owner:p,cell,tick:this.t,threat:false,connected,brick:false};x.bombs.set(cell,b);
     const covered=ray(this,cell),enemy=1-p,e=Math.floor(this.pos[2*enemy])*15+Math.floor(this.pos[2*enemy+1]);
     b.threat=Boolean(covered[e]&&this.alive[enemy]);
   }
   for(const item of x.itemEvents)if(item.seat===p&&this.t-item.tick<=30&&(info.ownBombDefeat?.[p]||!this.alive[p]||this.trapped[p]>0||this.done))item.void30=true;
   for(const item of x.itemEvents)if(item.seat===p&&this.t>item.tick&&this.t-item.tick<=100){item.bombs100+=Number(info.placed[p]);item.self100+=Number(info.ownBombDefeat?.[p]||0);}
 }
 for(const source of x.lastSources||[]){const b=x.bombs.get(source.cell);if(b&&b.owner===source.owner){b.brick||=source.brick;b.threat||=source.threat;}}
 for(const [cell,b] of x.bombs){
   // Update threat while the source remains live. Lifetime classification is
   // finalized only after resolution; terminal live bombs are UNKNOWN.
   if(this.fuse[cell]>0){const covered=ray(this,cell),p=1-b.owner,e=Math.floor(this.pos[2*p])*15+Math.floor(this.pos[2*p+1]);b.threat||=Boolean(covered[e]&&this.alive[p]);}
   else{x.finished.push(b);x.bombs.delete(cell);}
 }
 x.done=this.done;x.winner=this.winner;
 return info;
};
function collect(){return {version:'r1-passive-bomb-lifetime-items-v3',source_sha256:require('node:crypto').createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),engine:'production_JS',teacher:false,
 limitations:['threat means real blast-ray coverage over enemy cell; safe-exit reduction without direct cell coverage not measured',
 'brick source attributed to own physical bomb ray; overlapping sources not a unique causal removal proof',
 'postitem self-harm is temporal association; no claim item caused death',
 'topology paid-route traversal requires native reward provenance, unavailable in JS'],
 episodes:[...live].map(s=>{const x=s.__r1,p=x.seat,done=x.finished.filter(b=>b.owner===p),pending=[...x.bombs.values()].filter(b=>b.owner===p);return {
 seed:x.seed,seat:p,ticks:x.ticks,done:x.done,winner:x.winner,
 attribute_final:[s.bombsCap[p],s.blastCap[p],s.spdG[p]],attribute_full:fraction([s.bombsCap[p],s.blastCap[p],s.spdG[p]]).map(v=>Number(v>=1-1e-5)),
 attribute_highwater:x.highwater[p],item_detour_causal_deaths:null,item_detour_reason:'unavailable: observational pickup/death is not causal route attribution',
 learner:{placed:done.length+pending.length,resolved_bombs:done.length,unknown_terminal_bombs:pending.length,
 empty_bombs:done.filter(b=>!b.threat&&!b.connected&&!b.brick).length,kills:x.kills[p],self_harm:x.self[p],
 close_ticks:x.close,close_tick_denominator:x.closeEligible,phase_ticks:x.phaseTicks,draw:Number(x.winner===null||x.winner===-1),
 bombs_per_kill:x.kills[p]?(done.length+pending.length)/x.kills[p]:null},
 bomb_events:done,pending_bombs:pending,item_events:x.itemEvents.filter(e=>e.seat===p).map(e=>({...e,observed_ticks:Math.min(100,s.t-e.tick),full100:s.t-e.tick>=100}))};})};}
process.on('exit',code=>{if(code===0&&live.size){const f=process.argv[4]+'.r1_metrics.json';fs.writeFileSync(f+'.tmp',JSON.stringify(collect()));fs.renameSync(f+'.tmp',f);}});
require('./bomb_threat_eval');
