'use strict';
// Actual browser model encoding/sampling and real registered browser bots; P1 only.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),assert=require('node:assert/strict');
const crypto=require('node:crypto'),hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const Q=require('../web/sim-bun-native-v2'),Sample=require('./model_sampling_v1'),base=require('./interceptor_primary_inference_v1');
const Bots=require('../web/bot_contract'),Hunter=require('../web/bun_hunter_bot'),Coop=require('../web/bun_coop_hunter_bot'),Rule=require('../web/bun_rule_bot');
const D=require('./cx_web_idle_adapter_v1');
const cfg=JSON.parse(fs.readFileSync(process.argv[2]));
for(const [f,h] of Object.entries(cfg.source_sha256))assert.equal(hash(fs.readFileSync(f)),h,f);
async function loadModel(file){
 if(!file.startsWith('@bot/')){const x=await base.loadModel(file);Sample.attach(x.model,{inference:{greedy:cfg.greedy,temperature:cfg.temperature,seed:cfg.sampling_seed}});return x;}
 const [id,difficulty]=file.slice(5).split('@');
 const registry=Bots.createDefaultRegistry({BunRuleTacticalBot:Rule.BunRuleTacticalBot,hunter:Hunter,coopHunter:Coop});
 const bot=registry.create(id,difficulty?{difficulty}:{});let episode=null;
 const observation=s=>({schema:'qqt.bot.observation/v1',tick:s.t,state:Rule.stateFromSim(s),legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}});
 const model={act:async(s,p,rng)=>{
  if(episode!==s){episode=s;bot.reset({schema:'qqt.bot.context/v1',episode_id:'screen-'+s.seed,seed:s.seed>>>0,ruleset:'bun',max_ticks:s.maxSteps,metadata:{}});}
  const a=await bot.act(observation(s),p,rng);model.lastDecision={move:a.move,ability:a.ability};
  return [a.move,Number(a.ability===1),Number(a.ability===2),0,-1,-1,a.itemSlot??0];
 },observe:(info,s,p)=>{if(bot.observe_transition)bot.observe_transition(info,observation(s),p);}};
 return {model,meta:{ability_actions:3},sha:hash(file)};
}
let source=fs.readFileSync(path.join(__dirname,'resume_gpu67_eval_v1.js'),'utf8');
source=source.replace("const {loadModel}=require('./interceptor_primary_inference_v1');",'const loadModel=globalThis.__cxWebLoad;');
source=source.replace('   const info=s.step(joint);','   const info=s.step(joint);right.model.observe(info,s,q);');
source=source.replace('actions.push([dec[0].move,dec[0].ability,dec[1].move,dec[1].ability]);','actions.push([dec[0].move,dec[0].ability,dec[1].move,dec[1].ability,joint[0][6]??0,joint[1][6]??0]);');
source=source.replace('fresh.step([S.engineAction(a[0],a[1]),S.engineAction(a[2],a[3])]);','const joint=[S.engineAction(a[0],a[1]),S.engineAction(a[2],a[3])];joint[0][6]=a[4];joint[1][6]=a[5];fresh.step(joint);');
source=source.replace("schema:'resume-gpu67-native-replay-v1'","schema:'cx-web-native-replay-slots-v1'");
source=source.replace("module.exports={setup,threat,exits};","module.exports={setup,threat,exits,main};");
const m=new Module(path.join(__dirname,'cx_web_temperature_private_v1.js'),module);m.filename=path.join(__dirname,'cx_web_temperature_private_v1.js');m.paths=module.paths;
globalThis.__cxWebLoad=loadModel;m._compile(source,m.filename);delete globalThis.__cxWebLoad;
async function main(){
 assert(cfg.games.every(g=>g.seat===1),'model must be P1');
 await m.exports.main();D.install();
 const complete=JSON.parse(fs.readFileSync(cfg.output+'/COMPLETE.json'));
 for(const r of complete.results){
  const dir=cfg.output+`/match-${r.seed}-P1-${r.mode}-${r.opponent_mode}`;
  const diagnostics=D.replay(dir+'/review.json');assert(diagnostics.r3_chain_check);
  r.idle=diagnostics.idle;r.fountain=diagnostics.players;r.temperature=cfg.temperature;r.protocol_sha256=cfg.protocol_sha256;r.bot=cfg.bot;r.web_model_sampling_sha256=cfg.source_sha256['scripts/model_sampling_v1.js'];
  fs.writeFileSync(dir+'/RESULT.json',JSON.stringify(r));
 }
 fs.writeFileSync(cfg.output+'/COMPLETE.json',JSON.stringify(complete));
}
if(require.main===module)main().catch(e=>{console.error(e.stack);process.exitCode=1;});
module.exports={loadModel};
