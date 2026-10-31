'use strict';
// Actual production Sim and Hunter, teacher off. Passive sidecar only.
const fs=require('node:fs'),Q=require(process.env.QQT_STOCK_RULE==='stock-window-v1'?'../web/sim-stock-window-v1.js':'../web/sim');
const crypto=require('node:crypto'),hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const simulatorHash=hash(fs.readFileSync('web/sim-stock-window-v1.js'));
const {ControlFunnel}=require('./je10_control_funnel');
const [mode,model,out,regions,spawnFile]=process.argv.slice(2);
const spawn=JSON.parse(fs.readFileSync(spawnFile)),originalReset=Q.Sim.prototype.reset,originalStep=Q.Sim.prototype.step,live=new Set();
if(spawn.spawn_side_version){if(spawn.spawn_side_version!==require('./bun_spawn_side').VERSION)throw Error('unknown spawn contract');spawn.spawn_cells.forEach(require('./bun_spawn_side').validate);}
Q.Sim.prototype.reset=function(level,opts){
 const r=originalReset.call(this,level,opts);
 const index=((this.seed??0)-spawn.seed+spawn.spawn_cells.length)%spawn.spawn_cells.length;
 // Constructor does not retain seed consistently, assigned by the caller below.
 return r;
};
// build() compiles route evaluator; use a local explicit setup extension below.
let resetIndex=0;
Q.Sim.prototype.step=function(actions){
 // Review verification runs a second Sim. Only the actual evaluation birth
 // is marked; its replay must never be counted as a second game.
 if(!this.__btIdentity)return originalStep.call(this,actions);
 const before={pos:Array.from(this.pos),carried:Array.from(this.bunCarried),alive:Array.from(this.alive),trapped:Array.from(this.trapped)};
 if(!this.__bt){
  this.__bt={...this.__btIdentity,ticks:0,requests:[0,0],placed:[0,0],deaths:[0,0],self_deaths:[0,0],causal_kills:[0,0],cage_kills:[0,0],kills:[]};
  this.__btControl=[new ControlFunnel(0),new ControlFunnel(1)];live.add(this);
 }
 const x=this.__bt,info=originalStep.call(this,actions);x.ticks++;
 for(let p=0;p<2;p++){
  x.requests[p]+=Number(actions[p][1]===1);x.placed[p]+=Number(info.placed[p]);x.deaths[p]+=Number(info.died[p]);
  x.self_deaths[p]+=Number(info.ownBombDefeat?.[p]||0);
  const previous=this.__btControl[p].metrics.causal_kill;
  this.__btControl[p].observe(before,this,info);
  if(this.__btControl[p].metrics.causal_kill>previous){
   const enemy=1-p,r=Math.floor(before.pos[2*enemy]),c=Math.floor(before.pos[2*enemy+1]),inCage=this._bunBaseTeam(r,c)===this.team[p];
   x.causal_kills[p]++;x.cage_kills[p]+=Number(inCage);
   x.kills.push({tick:this.t,attacker:p,in_attacker_cage:inCage,victim_carried:before.carried[enemy],victim_pos:before.pos.slice(2*enemy,2*enemy+2)});
  }
 }
 x.done=this.done;x.winner=this.winner;x.stored=this.bunStored.map(r=>r.slice());
 return info;
};
// Extend the already validated production evaluator at its precise natural reset
// anchor. No teacher/actions/physics changes, only frozen held-out birth cells.
const Module=require('node:module'),path=require('node:path');
const route=path.resolve('scripts/dagger_rl_route.js'),src=fs.readFileSync(route,'utf8');
const {withoutDiagnosticReview}=require('./bomb_threat_eval_adapter');
const anchor="if(task==='natural_full2400')return s;";
if(src.split(anchor).length!==2)throw Error('natural spawn anchor');
const replacement=`if(task==='natural_full2400'){const doc=JSON.parse(fs.readFileSync(${JSON.stringify(spawnFile)})),pair=doc.spawn_cells[(seed-doc.seed)%doc.spawn_cells.length];if(!pair)throw Error('missing frozen spawn');for(let p=0;p<2;p++){s.pos[p*2]=pair[p][0]+.5;s.pos[p*2+1]=pair[p][1]+.5;}s.bunSpawnPos=pair.map(x=>[x[0]+.5,x[1]+.5]);s.__btIdentity={seed,seat:team};s._gen++;return s;}`;
const sideReplacement=spawn.spawn_side_version?replacement.replace('s.__btIdentity={seed,seat:team};','s.__btIdentity={seed,seat:team,spawn_side_version:doc.spawn_side_version,spawn_cells:pair,respawn_cells:s.bunSpawnPos.map(x=>x.map(v=>Math.floor(v)))};'):replacement;
// The common adapter reads the route source; intercept just that immutable read.
const read=fs.readFileSync;fs.readFileSync=function(file,...args){if(path.resolve(String(file))===route&&args[0]==='utf8')return withoutDiagnosticReview(src.replace(anchor,sideReplacement));return read.call(this,file,...args);};
const evaluator=require('./expert_validation_js_eval').build(mode);fs.readFileSync=read;
evaluator.run(model,out,regions).then(()=>{
 const episodes=[...live].filter(s=>s.__bt?.ticks>=10).map(s=>s.__bt);
 const reviewStatus='disabled_for_custom_birth_evaluation_only; official Review remains strict';
 const doc=JSON.parse(fs.readFileSync(out));doc.review=reviewStatus;doc.rule_version='stock-window-v1';doc.simulator_sha256=simulatorHash;
 fs.writeFileSync(out+'.tmp',JSON.stringify(doc));fs.renameSync(out+'.tmp',out);
 fs.writeFileSync(out+'.outcome.json.tmp',JSON.stringify({measurement_version:2,rule_version:'stock-window-v1',simulator_sha256:simulatorHash,engine:'production_JS',mode,teacher:false,diagnostic_review:reviewStatus,episodes,missing_metrics:['native_training_reward_fraction','native_brick_reward_events']}));
 fs.renameSync(out+'.outcome.json.tmp',out+'.outcome.json');
}).catch(e=>{console.error(e.stack);process.exitCode=1;});
