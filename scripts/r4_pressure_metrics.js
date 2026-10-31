'use strict';
// Passive full-possession denominators, with unknown reasons retained explicitly.
function install(Q){
 const P=Q.Sim.prototype,step=P.step,place=P._placeHeldItem;
 P._placeHeldItem=function(p,...args){const type=this.heldItem[p],r=place.call(this,p,...args);
  if(r&&this.__pressure?.active[p])this.__pressure.active[p].releases.push({tick:this.t,type,cell:[Math.floor(this.pos[2*p]),Math.floor(this.pos[2*p+1])]});return r;};
 P.step=function(actions){
  if(!this.__btIdentity)return step.call(this,actions);
  const x=this.__pressure ||= {active:[null,null],possessions:[[],[]],ticks:[0,0]},
   before={carried:this.bunCarried.slice(),alive:this.alive.slice(),stock:this.bunStored.map(r=>r.slice())};
  for(let p=0;p<2;p++)if(before.carried[p]>=0&&!x.active[p])
   x.active[p]={player:p,origin:before.carried[p],start_tick:this.t,source:'observed_pre_action',releases:[]};
  const r=step.call(this,actions);
  for(let p=0;p<2;p++){
   if(before.carried[p]>=0)x.ticks[p]++;
   if(before.carried[p]<0&&this.bunCarried[p]>=0)x.active[p]={player:p,origin:this.bunCarried[p],start_tick:this.t,source:'native_acquisition',releases:[]};
   const a=x.active[p];if(!a)continue;
   if(this.bunCarried[p]!==a.origin||!this.alive[p]||this.done){
    const delivered=this.bunStored[this.team[p]][a.origin]>before.stock[this.team[p]][a.origin];
    Object.assign(a,{end_tick:this.t,carry_ticks:this.t-a.start_tick,delivered,
     reason:delivered?'delivery':r.died?.[p]?'death':this.done?'terminal_or_timeout':this.trapped[p]>0?'native_control_drop':'drop_or_transfer_unknown'});
    x.possessions[p].push(a);x.active[p]=null;
   }
  }
  if(this.done&&this.__bt){
   const exportAfterAllObservers=()=>{
   const r3=this.__bt.r3||{},kills=this.__bt.kills||[];
   for(const arr of x.possessions)for(const a of arr){
    a.self_banana_contacts=(r3.contacts||[]).filter(e=>e.player===a.player&&e.owner===a.player&&e.type===1&&e.tick>=a.start_tick&&e.tick<=a.end_tick);
    a.complete_self_slides=(r3.slides||[]).filter(e=>e.player===a.player&&e.start_tick>=a.start_tick&&e.end_tick<=a.end_tick);
    a.glue_enemy_contacts=(r3.contacts||[]).filter(e=>e.owner===a.player&&e.player!==a.player&&e.type===2&&e.tick>=a.start_tick&&e.tick<=a.end_tick);
    a.causal_carrier_death=kills.some(k=>k.attacker!==a.player&&k.victim_carried===a.origin&&k.tick===a.end_tick);
    a.used_item=a.releases.length>0||a.self_banana_contacts.length>0;
   }
   return {version:1,ticks:x.ticks,possessions:x.possessions,unfinished:x.active.map(Boolean),
    opponent:process.env.R4_PRESSURE_OPPONENT||'unspecified',
    definitions:{acquisitions:'native carry transition by origin and player; genuine stock/loose source retained',
     success:'same possession delivered native stock increment / acquisitions, by item-use and physical seat',
     death:'ControlFunnel causal kill, matched physical death tick and package origin',
     banana:'own placement contact plus complete slide trajectory; saved ticks is same-polyline native carry-speed estimate'},
    missing_data:['matched no-item counterfactual delivery/tick benefit','exact causal glue route-delay branch'],
    limitations:['item use comparison is observational state-selection, not treatment effect','unresolved death/drop/transfer reasons remain explicit','Sim.step only, teacher disabled']};
   };
   Object.defineProperty(this.__bt,'r4_pressure',{enumerable:true,configurable:true,get:exportAfterAllObservers});
  }return r;
 };
}
module.exports={install};
