'use strict';
// Bounded natural bun-native-v2 games. One common protocol, independent RNG,
// explicit legacy/7-action adapters, and complete replay verification.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const Q=require('../web/sim-bun-native-v2'),R=require('../web/r3_model_inputs'),A=require('../web/five-item-slots-model-v1'),S=require('../web/five-item-slots-v1'),P=require('../web/five-item-slots-review-v1');
const Hunter=require('../web/bun_hunter_bot'),Bots=require('../web/bot_contract');
const {ControlFunnel}=require('./je10_control_funnel');
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
function rng(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
R.installSim(Q);
async function main(){
 const cfg=JSON.parse(fs.readFileSync(process.argv[2])),raw=fs.readFileSync(cfg.model_json),doc=JSON.parse(raw),m=doc.meta;
 assert.equal(hash(raw),cfg.model_sha256);assert.equal(hash(fs.readFileSync(m.onnx_path)),m.onnx_sha256);
 const ort=require(path.resolve(process.env.ONNX_RUNTIME_DIR)),session=await ort.InferenceSession.create(m.onnx_path,{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
 const referenceModel=new Q.TransformerModel(doc),model=new Q.ORTTransformerModel({meta:m},null);
 if(m.ability_actions===7)A.attach(model,m);else R.attach(model);model.greedy=true;
 const reference=referenceModel.forward.bind(referenceModel);let checked=0,maxError=0;const samples=[];
 model.forward=async(obs,state)=>{
  const out=await session.run({obs:new ort.Tensor('float32',Float32Array.from(obs),[1,...m.obs_shape]),state:new ort.Tensor('float32',Float32Array.from(state),[1,state.length])});
  const y={move:Array.from(out.move.data),bomb:Array.from(out.bomb.data),value:out.value.data[0]};
  if(checked<4){const z=reference(obs,state),error=Math.max(...['move','bomb'].flatMap(k=>y[k].map((v,i)=>Math.abs(v-z[k][i]))),Math.abs(y.value-z.value));assert(error<.002);maxError=Math.max(maxError,error);samples.push({obs:Array.from(obs),state:Array.from(state),js:{move:Array.from(z.move),bomb:Array.from(z.bomb),value:z.value},onnx:y});checked++;}
  for(const tensor of Object.values(out))tensor.dispose();return y;
 };
 const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806),results=[];
 for(const g of cfg.games){
  const id=`${cfg.label}-${cfg.tier}-${g.seed}-P${g.seat}`,file=cfg.output+'/'+id+'.json';
  if(fs.existsSync(file+'.result')){const r=JSON.parse(fs.readFileSync(file+'.result'));assert.equal(r.model_sha256,cfg.model_sha256);assert.equal(hash(fs.readFileSync(file)),r.replay_sha256);results.push(r);continue;}
  const sim=P.setup({seed:g.seed,max_ticks:2400,frozen_birth:{cells:g.spawn}},level),p=g.seat,q=1-p,actionRng=rng(g.seed^0x515154),botRng=rng(g.seed^0xB07);
  const bot=Bots.createDefaultRegistry({hunter:Hunter}).create('bun.hunter',{difficulty:cfg.tier});bot.reset({seed:g.seed^0xB07});
  const causal=new ControlFunnel(p),actions=[],checks=[[0,hash(JSON.stringify(P.state(sim)))]],metrics={bombs:0,enemy_bombs:0,deaths:0,self_deaths:0,causal_kills:0,deliveries:0,carrying_deaths:0,uses:[0,0,0,0,0]};
  while(!sim.done){
   const before={trapped:Array.from(sim.trapped)},carried=sim.bunCarried[p],stored=sim.bunStored[p].slice(),own=await model.act(sim,p,actionRng),enemy=bot.act({state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim}},q,botRng);
   const ability=own[2]?2+Math.max(0,own[6]||0):own[1]?1:0,ea=enemy.ability>=2?2+(enemy.itemSlot||0):enemy.ability;
   const row=[];row[2*p]=own[0];row[2*p+1]=ability;row[2*q]=enemy.move;row[2*q+1]=ea;actions.push(row);
   const joint=[];joint[p]=own;joint[q]=S.engineAction(enemy.move,ea);const info=sim.step(joint);
   metrics.bombs+=Number(info.placed[p]);metrics.enemy_bombs+=Number(info.placed[q]);metrics.deaths+=Number(info.died[p]);metrics.self_deaths+=Number(info.ownBombDefeat[p]);
   causal.observe(before,sim,info);metrics.causal_kills=causal.metrics.causal_kill;
   if(carried>=0){metrics.deliveries+=Math.max(0,sim.bunStored[p][carried]-stored[carried]);metrics.carrying_deaths+=Number(info.died[p]);}
   if(ability>=2)metrics.uses[ability-2]++;
   if(sim.t%200===0||sim.done)checks.push([sim.t,hash(JSON.stringify(P.state(sim)))]);
  }
  const meta={id,seed:g.seed,tick_ms:100,max_ticks:2400,scene:'native',native_rules:true,action_schema:'move-v1+five-item-slots-v1',ability_actions:7,rule_version:'bun-native-v2',model:cfg.label,checkpoint_sha256:cfg.policy_sha256,model_sha256:cfg.model_sha256,simulator_sha256:hash(fs.readFileSync('web/sim-bun-native-v2.js')),opponent_sha256:hash(fs.readFileSync('web/bun_hunter_bot.js')),map_sha256:await P.hash(level),policy:'greedy',teacher:false,learner_seat:p,opponent:`production JS Hunter/${cfg.tier}`,frozen_birth:{schema:'qqt.frozen-birth/v1',cells:g.spawn},reward_status:'unrecorded',training_reward_status:'unrecorded'};
  const replay={schema:P.SCHEMA,meta,actions,checks,events:[],rewards:[],summary:{ticks:sim.t,done:sim.done,winner:sim.winner,reward:[null,null],failures:[]}};
  meta.replay_identity=await P.hash(P.identity(replay));await new P.Player(replay,level).verify();
  fs.writeFileSync(file,JSON.stringify(replay));const r={id,seed:g.seed,seat:p,tier:cfg.tier,model:cfg.label,model_sha256:cfg.model_sha256,winner:sim.winner,ticks:sim.t,...metrics,replay_verified:true,replay_sha256:hash(fs.readFileSync(file))};
  fs.writeFileSync(file+'.result',JSON.stringify(r));results.push(r);console.log(JSON.stringify({id,winner:sim.winner,ticks:sim.t}));
 }
 await session.release();fs.writeFileSync(cfg.output+'/COMPLETE.json',JSON.stringify({status:'PASS',results,maxError,samples}));
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
