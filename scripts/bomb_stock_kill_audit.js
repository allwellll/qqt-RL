'use strict';
const fs=require('node:fs'),path=require('node:path');
const simPath=require.resolve('../web/sim');
const Q=require('../web/review-sim/stock-legacy-20261028');
require.cache[simPath]={id:simPath,filename:simPath,loaded:true,exports:Q};
const live=new Set(),original=Q.Sim.prototype.step;
Q.Sim.prototype.step=function(actions){
 const before={pos:Array.from(this.pos),carried:Array.from(this.bunCarried),alive:Array.from(this.alive),trapped:Array.from(this.trapped)};
 const info=original.call(this,actions);
 const x=this.__audit||(this.__audit={ticks:0,requests:[0,0],placed:[0,0],kills:[]});live.add(this);x.ticks++;
 for(let p=0;p<2;p++){
  x.requests[p]+=Number(actions[p][1]===1);x.placed[p]+=Number(info.placed[p]);
  if(info.died[1-p]){
   const e=1-p,[r,c]=this.centerCell(e);
   x.kills.push({tick:this.t,attacker:p,victim:e,causal:!!info.causalKill?.[p],pos_before:before.pos.slice(2*e,2*e+2),pos_after:Array.from(this.pos).slice(2*e,2*e+2),in_attacker_cage:this._bunBaseTeam(r,c)===this.team[p],carried_before:before.carried[e],trapped_before:before.trapped[e],stock_take_same_tick:before.carried[e]<0&&this.bunCarried[e]>=0});
  }
 }
 return info;
};
const [model,out,regions]=process.argv.slice(2);
require('./expert_validation_js_eval').build('greedy').run(model,out,regions).then(()=>{
 fs.writeFileSync(out+'.kill_audit.json',JSON.stringify({engine:'production_JS_legacy_stock',teacher:false,episodes:[...live].filter(s=>s.__audit.ticks>10).map(s=>({...s.__audit,winner:s.winner,done:s.done}))}));
}).catch(e=>{console.error(e.stack);process.exitCode=1;});
