'use strict';
// Passive native control funnel; no action or simulator mutation.
const Q=require('../web/sim.js'),inference=require('./je05_web_hunter.js');
const fs=require('fs');
const {ControlFunnel}=require('./je10_control_funnel');
const records=[],actual=new WeakSet();
const lagged=Q.TransformerModel.prototype._getLaggedInputs;
Q.TransformerModel.prototype._getLaggedInputs=function(sim,...rest){actual.add(sim);return lagged.call(this,sim,...rest);};
const original=Q.Sim.prototype.step;
Q.Sim.prototype.step=function(actions){
  if(!actual.has(this))return original.call(this,actions);
  if(!this._je10){this._je10=new ControlFunnel(0);records.push(this._je10.metrics);}
  const before={trapped:this.trapped.slice(),alive:this.alive.slice()};
  const info=original.call(this,actions);this._je10.observe(before,this,info);return info;
};
async function main(){
  const options=require('./eval_web_hunter.js').parseArgs(process.argv.slice(2));
  await inference.run(options);
  const result=JSON.parse(fs.readFileSync(options.out));
  if(records.length!==result.episodes.length)throw Error('passive main-simulator episode alignment');
  result.episodes.forEach((e,i)=>{e.je10_control=records[i];if(!result.protocol.native_trap&&e.je10_control.surviving_kill!==e.counts.surviving_causal_kill)throw Error('passive kill ledger parity');});
  result.je10_control_protocol={native_trap:result.protocol.native_trap,native_items:result.protocol.native_items,control_available:result.protocol.native_trap,limitation:'Frozen fast Hunter uses non-native-trap combat; native control chain is measured separately in declared route/native-full tasks.'};
  result.je10_control_scope='native JS passive event ledger v2; original unique control attribution persists to death; native causalKill also separate; active_contact_pop excludes expiry-tick ambiguity; final_pop includes timeout, no reward inference';
  fs.writeFileSync(options.out+'.tmp',JSON.stringify(result));fs.renameSync(options.out+'.tmp',options.out);
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
