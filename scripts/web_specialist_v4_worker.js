'use strict';
// One CPU-only compatibility process uses the production v4 engine. Archived sources stay intact.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const cfg=JSON.parse(fs.readFileSync(process.argv[2]));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
assert.equal(process.env.CUDA_VISIBLE_DEVICES,'');
for(const [f,h] of Object.entries(cfg.source_sha256))assert.equal(hash(f),h,f);
const Q=require(path.resolve(cfg.engine)),engineSHA=hash(cfg.engine);
assert.equal(Q.RULE_VERSION,'bun-native-v4-facing');
// All legacy encoders/observers in this private process receive the same production engine.
const old=require.resolve('../web/sim-bun-native-v2');
require.cache[old]={id:old,filename:old,loaded:true,exports:Q};
const Sample=require(path.resolve(cfg.production+'/web/model_sampling_v1.js'));
const Bots=require(path.resolve(cfg.production+'/web/bot_contract.js'));
const Hunter=require(path.resolve(cfg.production+'/web/bun_hunter_bot.js'));
const Coop=require(path.resolve(cfg.production+'/web/bun_coop_hunter_bot.js'));
const Rule=require(path.resolve(cfg.production+'/web/bun_rule_bot.js'));
const ort=require(process.env.QQT_ORT_NODE_MODULE || 'onnxruntime-node');
const base={loadModel:async file=>{
 const bytes=fs.readFileSync(file),doc=JSON.parse(bytes),graph=cfg.asset_graphs[file];assert.equal(hash(graph),doc.meta.onnx_sha256);
 const session=await ort.InferenceSession.create(graph,{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
 const model=new Q.TransformerModel(doc),reference=model.forward.bind(model);let verified=false;
 model.forward=async(obs,state)=>{const r=await session.run({obs:new ort.Tensor('float32',obs,[1,25,13,15]),state:new ort.Tensor('float32',Float32Array.from(state),[1,37])});
  const v={move:r.move.data,bomb:r.bomb.data,value:r.value.data[0]};if(!verified){const e=reference(obs,state);model.forwardMaxError=Math.max(...['move','bomb'].flatMap(k=>Array.from(v[k],(x,i)=>Math.abs(x-e[k][i]))));assert(model.forwardMaxError<.002);verified=true;}return v;};
 return {model,meta:doc.meta,sha:crypto.createHash('sha256').update(bytes).digest('hex')};
}};
function compile(name,source){const m=new Module(path.join(__dirname,name),module);m.filename=path.join(__dirname,name);m.paths=module.paths;m._compile(source,m.filename);return m.exports;}
function once(src,a,b){assert.equal(src.split(a).length-1,1,a);return src.replace(a,b);}
function setup(game,level){
 const s=new Q.Sim(game.seed);s.reset(level,{nativeTrap:true,nativeItems:true});s.maxSteps=game.horizon||2400;
 // Browser screens use native reset spawn. Artificial short duels have explicit task state.
 if(game.specialist){
  const r=game.specialist;
  s.crate.fill(0);s.crateType.fill(-1);s.superCrate.fill(0);s.crateCount.fill(0);
  s.fieldItem.fill(0);s.cover.fill(0);s.bunLoose.fill(0);s.bunStored=[[0,0],[0,0]];s.bunInitial=[0,0];s.bunCarried.fill(-1);
  s.bunRespawn.fill(0);s.pendingCrateType.fill(-1);s.airdropFalls=[];s._pendingDeathDrops=[];
  s.pos.set(r.spawn.flatMap(([y,x])=>[y+.5,x+.5]));s.spawnCells=r.spawn.map(x=>x.slice());
  s.bunSpawnPos=r.spawn.map(([y,x])=>[y+.5,x+.5]);
  for(let p=0;p<2;p++){
   [s.bombsCap[p],s.blastCap[p],s.spdG[p]]=r.stats[p];
   [s.loBombs[p],s.loBlast[p],s.loSpeed[p]]=r.stats[p];
   s.itemSlots[p]=r.slots[p].flatMap((item,i)=>item?[{item,count:r.counts[p][i]}]:[]);
   s.lastMoveDir[p]=r.heading[p];s.slideDir[p]=r.heading[p];
  }
  s.fuse.set(r.fuse.flat());s.bombBlast.set(r.blast.flat());
  for(let i=0;i<195;i++)s.owner[i]=s.fuse[i]>0?1-game.seat:-1;
  s.itemsEnabled=true;s.crateRate=0;s._applyItemSupply();
  assert.equal(s.itemSupply.added,15);assert.equal(s.bunStored.flat().reduce((a,b)=>a+b,0),0);
  const step=s.step;
  s.step=function(actions){const info=step.call(this,actions);
   if(info.died.some(Boolean)){
    this.done=true;const dead=Array.from(info.died);this.winner=dead[0]&&dead[1]?null:dead[0]?1:0;
   }return info;};
 }
 assert([0,1].every(p=>s.alive[p]));return s;
}
globalThis.__itemsSetup=setup;
// Offline diagnostics replay the same six-column action log and exact engine hash.
let fountain=fs.readFileSync('scripts/cc_strong_metrics_fountain_v1.js','utf8');
fountain=once(fountain,"const { setup } = require('./resume_gpu67_eval_v1');",'const setup=globalThis.__itemsSetup;');
fountain=once(fountain,"const SIM_SHA = Q.SIMULATOR_SHA256;",`const SIM_SHA = '${engineSHA}';`);
fountain=once(fountain,'const act = row => [S.engineAction(row[0], row[1]), S.engineAction(row[2], row[3])];','const act = row => [0,1].map(p=>{const a=S.engineAction(row[2*p],row[2*p+1]);a[6]=row[4+p]||0;return a;});');
globalThis.__itemsFountain=compile('cx_items_fountain_private_v1.js',fountain);
let idle=fs.readFileSync('scripts/cx_idle_replay_v1.js','utf8');
idle=once(idle,"F=require('./cc_strong_metrics_fountain_v1')",'F=globalThis.__itemsFountain');
const D=compile('cx_items_idle_private_v1.js',idle+'\nmodule.exports.install=install;module.exports.replay=replay;');
let loadCount=0;
async function loadModel(file){
 const opponent=loadCount++>0;
 if(!file.startsWith('@bot/')){
  const x=await base.loadModel(file);
  Sample.attach(x.model,{inference:{greedy:opponent?false:cfg.greedy,temperature:.3,seed:cfg.sampling_seed}});
  x.model.observe=()=>{};return x;
 }
 const [id,difficulty]=file.slice(5).split('@');
 const registry=Bots.createDefaultRegistry({BunRuleTacticalBot:Rule.BunRuleTacticalBot,hunter:Hunter,coopHunter:Coop});
 const bot=registry.create(id,difficulty?{difficulty}:{});let episode=null;
 const observation=s=>({schema:'qqt.bot.observation/v1',tick:s.t,state:Rule.stateFromSim(s),legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}});
 const model={act:async(s,p,rng)=>{
  if(episode!==s){episode=s;bot.reset({schema:'qqt.bot.context/v1',episode_id:'items-screen-'+s.seed,seed:s.seed>>>0,ruleset:'bun',max_ticks:s.maxSteps,metadata:{}});}
  const a=await bot.act(observation(s),p,rng);model.lastDecision={move:a.move,ability:a.ability};
  return [a.move,Number(a.ability===1),Number(a.ability===2),0,-1,-1,a.itemSlot??0];
 },observe:(info,s,p)=>{if(bot.observe_transition)bot.observe_transition(info,observation(s),p);}};
 return {model,meta:{ability_actions:3},sha:crypto.createHash('sha256').update(file).digest('hex')};
}
globalThis.__itemsLoad=loadModel;
let src=fs.readFileSync('scripts/resume_gpu67_eval_v1.js','utf8');
src=once(src,"const {loadModel}=require('./interceptor_primary_inference_v1');",'const loadModel=globalThis.__itemsLoad;');
const start=src.indexOf('function setup(game,level){'),end=src.indexOf('\nfunction threat(',start);
assert(start>=0&&end>start);src=src.slice(0,start)+'const setup=globalThis.__itemsSetup;'+src.slice(end);
src=src.replaceAll("fileHash('web/sim-bun-native-v2.js')",`'${engineSHA}'`);
src=once(src,"rule:'bun-native-v2'","rule:'bun-native-v4-facing'");
src=once(src,'   const info=s.step(joint);','   const info=s.step(joint);right.model.observe(info,s,q);');
src=once(src,'actions.push([dec[0].move,dec[0].ability,dec[1].move,dec[1].ability]);','actions.push([dec[0].move,dec[0].ability,dec[1].move,dec[1].ability,joint[0][6]??0,joint[1][6]??0]);');
src=once(src,'fresh.step([S.engineAction(a[0],a[1]),S.engineAction(a[2],a[3])]);','const joint=[S.engineAction(a[0],a[1]),S.engineAction(a[2],a[3])];joint[0][6]=a[4];joint[1][6]=a[5];fresh.step(joint);');
// The independent production sampler owns greedy/tau settings for each policy.
src=once(src,"left.model.greedy=game.mode==='greedy';right.model.greedy=game.opponent_mode==='greedy';","left.model.greedy=cfg.greedy;right.model.greedy=false;");
src=once(src,"module.exports={setup,threat,exits};","module.exports={setup,threat,exits,main};");
const worker=compile('cx_items_matches_private_v1.js',src);
async function main(){
 await worker.main();D.install();const complete=JSON.parse(fs.readFileSync(cfg.output+'/COMPLETE.json'));
 for(const r of complete.results){
  const dir=cfg.output+`/match-${r.seed}-P${r.seat}-${r.mode}-${r.opponent_mode}`;
  const diagnostics=D.replay(dir+'/review.json');assert(diagnostics.r3_chain_check);
  r.idle=diagnostics.idle;r.fountain=diagnostics.players;r.temperature=cfg.greedy?null:.3;
  r.protocol_sha256=cfg.protocol_sha256;r.bot=cfg.bot||null;r.ruleset=Q.RULE_VERSION;
  r.simulator_sha256=engineSHA;r.split=cfg.split;r.specialist=!!cfg.games[0].specialist;
  r.extra_replay_hash_checks=diagnostics.hash_checks_verified;
  fs.writeFileSync(dir+'/RESULT.json',JSON.stringify(r));
 }
 fs.writeFileSync(cfg.output+'/COMPLETE.json',JSON.stringify(complete));
 console.log(JSON.stringify({status:'PASS_ITEMS_PRODUCTION_REPLAY',games:complete.results.length}));
}
if(require.main===module)main().catch(e=>{console.error(e.stack);process.exitCode=1;});
module.exports={setup};
