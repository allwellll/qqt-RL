'use strict';
// Passive, per-instance observer. All state lives outside Sim; no RNG or action calls.
const fs=require('node:fs');
const observers=new WeakMap(),installed=new WeakSet();
const copy=x=>Array.isArray(x)||ArrayBuffer.isView(x)?Array.from(x):null;
function source(row,n){const a=copy(row);return a&&a.length===n?a.map(Boolean):null;}
function sourceKind(a,p){if(a===null)return 'missing';const owners=a.flatMap((v,i)=>v?[i]:[]);return owners.length===0?'none':owners.length>1?'shared':owners[0]===p?'unique_self':'unique_other';}
function snap(s){return {alive:copy(s.alive),trapped:copy(s.trapped),carry:copy(s.bunCarried),stock:s.bunStored.map(copy)};}
function newCustody(x,p,origin,tick,pickup){return {id:`P${p}-C${++x.ids[p]}`,player:p,origin,start_tick:tick,source:pickup?.source||'preexisting_or_transfer_unknown',pickup_window:pickup?.window??null,ticks:0,actionable_ticks:0};}
function attach(s,{learner}){
 if(observers.has(s))throw Error('source observer already attached');
 const n=s.nPlayers;if(!Number.isInteger(learner)||learner<0||learner>=n)throw Error('invalid learner identity');const x={learner,n,ids:Array(n).fill(0),active:Array(n).fill(null),custody:Array(n).fill(null),pending:Array(n).fill(null),controls:[],deaths:[],custodies:[],rescues:[],native:Array(n).fill(0),ticks:0,denominators:Array.from({length:n},()=>({alive_ticks:0,actionable_ticks:0,carry_ticks:0,carry_actionable_ticks:0,controls:0,deaths:0,source_rows_missing:0,native_ownBombDefeat_recorded_ticks:0,native_ownBombDefeat_missing_ticks:0})),before:null,trap:false,finalizers:[]};
 for(let p=0;p<n;p++)if(s.bunCarried[p]>=0)x.custody[p]=newCustody(x,p,s.bunCarried[p],s.t,null);
 observers.set(s,x);return x;
}
function install(Q){
 const P=Q.Sim.prototype;if(installed.has(P))return;installed.add(P);
 const reach=41/Number(fs.readFileSync(require.resolve('../web/sim-bun-native-v2'),'utf8').match(/const NATIVE_CELL_PX = (\d+)/)[1]);
 const step=P.step,trap=P._trapStep,kill=P._killPlayer,update=P._bunUpdate;
 P._trapStep=function(...args){const x=observers.get(this);if(!x)return trap.apply(this,args);x.trap=true;try{return trap.apply(this,args);}finally{x.trap=false;}};
 P._killPlayer=function(p,...args){const x=observers.get(this);if(x&&x.trap){
  let q=null;for(let i=0;i<this.nPlayers;i++)if(i!==p&&this.alive[i]&&this.trapped[i]===0&&this.team[i]!==this.team[p]&&Math.abs(this.pos[2*i]-this.pos[2*p])<reach&&Math.abs(this.pos[2*i+1]-this.pos[2*p+1])<reach){q=i;break;}
  x.finalizers.push({victim:p,kind:q!==null?'enemy_active_pop':this.trapped[p]<=0?'expiry':'unknown',finalizer:q,trap_remaining_at_kill:this.trapped[p]});
 }return kill.call(this,p,...args);};
 P._bunUpdate=function(...args){const x=observers.get(this);if(!x)return update.apply(this,args);const b=snap(this),r=update.apply(this,args);
  for(let p=0;p<x.n;p++)if(b.carry[p]<0&&this.bunCarried[p]>=0){const o=this.bunCarried[p],others=b.alive.map((_,q)=>q).filter(q=>this.team[q]!==this.team[p]);x.pending[p]={source:b.stock[o]?.[o]>this.bunStored[o]?.[o]?'native_stock':'native_loose_or_contact',window:others.some(q=>!b.alive[q]||b.trapped[q]>0)};}
  return r;
 };
 P.step=function(actions){const x=observers.get(this);if(!x||this.done)return step.call(this,actions);const b=snap(this);x.before=b;x.finalizers=[];x.ticks++;
  for(let p=0;p<x.n;p++){const d=x.denominators[p],actionable=!!b.alive[p]&&b.trapped[p]===0;d.alive_ticks+=Number(!!b.alive[p]);d.actionable_ticks+=Number(actionable);d.carry_ticks+=Number(b.carry[p]>=0);d.carry_actionable_ticks+=Number(actionable&&b.carry[p]>=0);const c=x.custody[p];if(c){c.ticks++;c.actionable_ticks+=Number(actionable);}}
  const r=step.call(this,actions);
  for(let p=0;p<x.n;p++){
   const d=x.denominators[p],c=x.custody[p];if(r.ownBombDefeat?.[p]===undefined||r.ownBombDefeat?.[p]===null)d.native_ownBombDefeat_missing_ticks++;else{d.native_ownBombDefeat_recorded_ticks++;x.native[p]+=Number(r.ownBombDefeat[p]);}
   if(b.trapped[p]===0&&this.trapped[p]>0){const causal=source(r.causalDamageSource?.[p],x.n),physical=source(r.physicalDamageSource?.[p],x.n),e={id:`control-${x.controls.length}`,tick:this.t,victim:p,carried_at_initial_control:b.carry[p],custody_id:c?.id||null,causal_source:causal,physical_source:physical,causal_kind:sourceKind(causal,p),physical_kind:sourceKind(physical,p)};x.controls.push(e);x.active[p]=e;d.controls++;d.source_rows_missing+=Number(causal===null||physical===null);}
   if(r.died?.[p]){const a=x.active[p],f=x.finalizers.find(v=>v.victim===p)||{victim:p,kind:'unknown',finalizer:null};x.deaths.push({tick:this.t,victim:p,carried_at_step_start:b.carry[p],custody_id:c?.id||null,initial_control:a||null,uninterrupted_control:!!a,strict_same_custody:!!a&&a.custody_id!==null&&a.custody_id===c?.id,finalizer:f,native_ownBombDefeat:r.ownBombDefeat?.[p]??null});x.active[p]=null;d.deaths++;}
   else if(b.trapped[p]>0&&this.trapped[p]===0){if(x.active[p])x.rescues.push({tick:this.t,victim:p,initial_control_id:x.active[p].id});x.active[p]=null;}
   if(c&&(this.bunCarried[p]!==c.origin||r.died?.[p]||this.done)){const delivered=this.bunStored[this.team[p]]?.[c.origin]>b.stock[this.team[p]]?.[c.origin];c.end_tick=this.t;c.outcome=delivered?'same_custody_delivery':r.died?.[p]?'death_drop':this.bunCarried[p]!==c.origin&&this.trapped[p]>0?'controlled_drop':this.done&&this.bunCarried[p]===c.origin?(this.t>=this.maxSteps?'timeout_still_carrying':'other_terminal_still_carrying'):'drop_transfer_unknown';x.custodies.push(c);x.custody[p]=null;}
   if(this.bunCarried[p]>=0&&!x.custody[p]&&(!this.done||!c)){const next=newCustody(x,p,this.bunCarried[p],this.t,x.pending[p]);if(this.done){next.end_tick=this.t;next.outcome=this.t>=this.maxSteps?'timeout_still_carrying':'other_terminal_still_carrying';x.custodies.push(next);}else x.custody[p]=next;}x.pending[p]=null;
  }
  return r;
 };
}
function exportMetrics(s){const x=observers.get(s);if(!x)throw Error('source observer absent');const players=[];
 for(let p=0;p<x.n;p++){const controls=x.controls.filter(e=>e.victim===p),deaths=x.deaths.filter(e=>e.victim===p),count=(rows,pred)=>rows.filter(pred).length;
  const out={player:p,...x.denominators[p],native_ownBombDefeat:x.native[p],native_ownBombDefeat_not_total_self_harm:true,unique_causal_self_controls:count(controls,e=>e.causal_kind==='unique_self'),unique_physical_self_controls:count(controls,e=>e.physical_kind==='unique_self'),control_source_coverage:Object.fromEntries(['causal','physical'].map(k=>[k,Object.fromEntries(['missing','none','shared','unique_self','unique_other'].map(v=>[v,count(controls,e=>e[k+'_kind']===v)]))]))};
  for(const type of ['causal','physical']){const rows=deaths.filter(e=>e.initial_control?.[type+'_kind']==='unique_self');out[`unique_${type}_self_control_deaths`]=rows.length;out[`${type}_self_death_finalizers`]=Object.fromEntries(['expiry','enemy_active_pop','unknown'].map(k=>[k,count(rows,e=>e.finalizer.kind===k)]));}
  out.strict_carrier_by_source=Object.fromEntries(['causal','physical'].map(type=>[type,{control_by_other:count(controls,e=>e.carried_at_initial_control>=0&&e.custody_id!==null&&e[type+'_kind']==='unique_other'),kill_by_other:count(deaths,e=>e.strict_same_custody&&e.initial_control.carried_at_initial_control>=0&&e.initial_control[type+'_kind']==='unique_other')}]));out.strict_carrier_control_by_other=out.strict_carrier_by_source.causal.control_by_other;out.strict_carrier_kill_by_other=out.strict_carrier_by_source.causal.kill_by_other;players.push(out);
 }
 return {version:'g7-source-metrics-v1',learner:x.learner,ticks:x.ticks,complete:!!s.done,players,controls:x.controls,deaths:x.deaths,custodies:x.custodies,open_custodies:x.custody.filter(Boolean),censored_controls:x.active.filter(Boolean),rescues:x.rescues,missing_data:{reward:'unrecorded',advantage:'unrecorded',death_without_initial_control:x.deaths.filter(d=>!d.initial_control).length,unknown_finalizer:x.deaths.filter(d=>d.finalizer.kind==='unknown').length},definitions:{strict_carrier_kill:'unique causal other source at initial control while carrying, uninterrupted control, same custody through death; final pop actor separate',custody:'mutually exclusive first-terminal outcomes; one origin may recur only as a new custody id',self_harm:'native death-tick ownBombDefeat and physical/causal initial-self-control deaths are separate, overlapping columns'},limitations:['pickup native_loose_or_contact does not uniquely distinguish loose pickup and contact transfer','same-tick drop/reacquire with unchanged origin may be unobservable','source masks describe native causal/physical attribution, not learned policy intent','first terminal only; censored active controls are not counted as deaths']};
}
module.exports={install,attach,exportMetrics,sourceKind};
