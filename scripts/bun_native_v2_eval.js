'use strict';
// Process-local rule selection and passive instrumentation around the frozen
// production evaluator. No teacher, action or observation intervention.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
const rule=process.env.BUN_V2_RULE,tier=process.env.BUN_V2_TIER;
if(!['stock-window-v1','bun-native-v2'].includes(rule))throw Error('explicit rule required');
if(!['hard','normal','easy'].includes(tier))throw Error('explicit difficulty required');
const selected=path.resolve(rule==='bun-native-v2'?'web/sim-bun-native-v2.js':'web/sim-stock-window-v1.js');
const Q=require(selected),load=Module._load;
Module._load=function(request,parent,...args){
 const resolved=Module._resolveFilename(request,parent);
 if([path.resolve('web/sim.js'),path.resolve('web/sim-stock-window-v1.js')].includes(resolved))return Q;
 return load.call(this,request,parent,...args);
};
const {observe,stockUpdate}=require('./bun_native_v2_metrics');
const step=Q.Sim.prototype.step,update=Q.Sim.prototype._bunUpdate;
Q.Sim.prototype._bunUpdate=function(...args){return stockUpdate(this,()=>update.apply(this,args));};
Q.Sim.prototype.step=function(actions){return observe(this,actions,()=>step.call(this,actions));};
if(process.env.BUN_V2_OPPONENT==='bun.coop_hunter'){
 const authority=require('./eval_web_hunter'),bots=require('../web/bot_contract'),coop=require('../web/bun_coop_hunter_bot');
 authority.createHunter=difficulty=>{const r=bots.createDefaultRegistry({coopHunter:coop});
  return {adapter:r.create('bun.coop_hunter',{difficulty}),spec:r.list().find(s=>s.id==='bun.coop_hunter'),config:{difficulty}};};
}
const file=path.resolve('scripts/bomb_threat_eval.js');
let source=fs.readFileSync(file,'utf8');
function replace(a,b){if(source.split(a).length!==2)throw Error('source anchor '+a);source=source.replace(a,b);}
replace("hash(fs.readFileSync('web/sim-stock-window-v1.js'))",`hash(fs.readFileSync(${JSON.stringify(selected)}))`);
replace("require('./expert_validation_js_eval').build(mode)",`require('./expert_validation_js_eval').build(mode,${JSON.stringify(tier)})`);
replace('victim_pos:before.pos.slice(2*enemy,2*enemy+2)',`victim_pos:before.pos.slice(2*enemy,2*enemy+2),within3_attacker_home:require('./bun_native_v2_metrics').nearPosition(this.bunBases[this.team[p]],before.pos.slice(2*enemy,2*enemy+2))`);
source=source.replaceAll("'stock-window-v1'",JSON.stringify(rule));
const m=new Module(file,module);m.filename=file;m.paths=module.paths;m._compile(source,file);
