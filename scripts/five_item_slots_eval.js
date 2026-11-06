'use strict';
// Complete production-JS games; independent action RNG, no teacher/takeover.
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const Q=require('../web/sim-bun-native-v2'),S=require('../web/five-item-slots-v1'),A=require('../web/five-item-slots-model-v1'),P=require('../web/five-item-slots-review-v1');
const {loadModel}=require('./five_item_slots_inference');
const Hunter=require('../web/bun_hunter_bot'),Bots=require('../web/bot_contract');
const {snapshot,FunnelTracker}=require('./r4_iteration_metrics_20261101_funnel_helper');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
function rng(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
function passive(){
 // An independent copy corrects release type attribution for explicit slot3+
 // without editing the historical three-action instrumentation.
 const Module=require('node:module'),path=require('node:path'),file=path.resolve('scripts/r3_items_metrics.js');
 let src=fs.readFileSync(file,'utf8');const old='const type=this.heldItem[p],r=place.call(this,p,...args);';
 assert.equal(src.split(old).length,2);src=src.replace(old,'const slot=Number.isInteger(args[1])&&args[1]>=0?args[1]:0;const type=this.itemSlots[p][slot]?.item||0,r=place.call(this,p,...args);');
 const m=new Module(file,module);m.filename=file;m.paths=module.paths;m._compile(src,file);m.exports.install(Q);return m.exports;
}
async function main(){
 const cfg=JSON.parse(fs.readFileSync(process.argv[2])),loaded=await loadModel(cfg.model_json),R3=passive(),results=[];
 const tier=cfg.opponent_tier||'hard';
 assert(['easy','normal','hard'].includes(tier),`invalid opponent tier: ${tier}`);
 const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
 for(const game of cfg.games){
  const existing=cfg.output+`/five-slots-${game.seed}-${game.seat}-${game.mode}/RESULT.json`;
  if(fs.existsSync(existing)){const value=JSON.parse(fs.readFileSync(existing));assert.equal(value.model_json_sha256,loaded.sha);assert.equal(hash(fs.readFileSync(existing.replace('RESULT.json','review.json'))),value.review_sha256);assert(value.complete&&value.review_verified);results.push(value);console.log(JSON.stringify({status:'REUSED_COMPLETE_GAME',seed:game.seed,seat:game.seat,mode:game.mode}));continue;}
  const s=P.setup({seed:game.seed,max_ticks:2400,frozen_birth:{cells:game.spawn}},level);s.__btIdentity=true;
  const p=game.seat,q=1-p,model=loaded.model;model.greedy=game.mode==='greedy';
  const bot=Bots.createDefaultRegistry({hunter:Hunter}).create('bun.hunter',{difficulty:tier});bot.reset({seed:game.seed^0xB07});
  const actionRng=rng(game.action_seed),botRng=rng(game.seed^0xB07),funnel=new FunnelTracker(p);
  const checks=[[0,hash(JSON.stringify(P.state(s)))]],actions=[],decisions=[],placed=[0,0],deaths=[0,0],slotByPhase=Array.from({length:3},()=>Array.from({length:5},()=>({held:0,legal:0,probability_sum:0,requested:0,actual:0,type:{}})));
  let actionable=0,firstOpportunity=null;
  const oldPlace=s._placeHeldItem.bind(s);s._placeHeldItem=function(player,cell,index){const selected=Number.isInteger(index)&&index>=0?index:0;const item=this.itemSlots[player][selected]?.item||0;const r=oldPlace(player,cell,index);if(r&&player===p){const z=slotByPhase[this.__slotPhase][selected];z.actual++;z.type[item]=(z.type[item]||0)+1;}return r;};
  while(!s.done){
   const before=snapshot(s),phase=s.bunCarried[p]>=0?2:!s.alive[q]||s.trapped[q]>0?1:0;s.__slotPhase=phase;
   const own=await model.act(s,p,actionRng),d=structuredClone(model.lastDecision),slots=s.itemSlots[p];
   if(model.lastReference){const top=(v,m)=>Array.from(v,(x,i)=>m[i]?x:-Infinity).reduce((b,x,i,a)=>x>a[b]?i:b,0);assert.equal(top(model.lastReference.move,d.move_mask),top(d.move_probability,d.move_mask));assert.equal(top(model.lastReference.bomb,d.ability_mask),top(d.ability_probability,d.ability_mask));}
   if(s.alive[p]&&!s.trapped[p])actionable++;
   for(let i=0;i<5;i++){const z=slotByPhase[phase][i];z.held+=Number((slots[i]?.count||0)>0);z.legal+=d.ability_mask[i+2];z.probability_sum+=d.ability_probability[i+2];z.requested+=Number(d.ability===i+2);}
   if(firstOpportunity===null&&d.ability_mask.slice(2).some(Boolean))firstOpportunity=s.t;
   const enemy=bot.act({state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},q,botRng);
   const ab=enemy.ability>=2?2+(enemy.itemSlot||0):enemy.ability;
   assert(Number.isInteger(ab)&&ab>=0&&ab<=6,'native bot requested a slot outside this seven-action replay ABI; never clamp/fallback');
   const joint=[];joint[p]=own;joint[q]=S.engineAction(enemy.move,ab);
   const row=[];row[2*p]=d.move;row[2*p+1]=d.ability;row[2*q]=enemy.move;row[2*q+1]=ab;actions.push(row);decisions.push({tick:s.t,...d});
   const info=s.step(joint);funnel.observe(before,snapshot(s),info,s.t);
   for(let k=0;k<2;k++){placed[k]+=Number(info.placed[k]);deaths[k]+=Number(info.died[k]);}
   if(s.t%200===0||s.done)checks.push([s.t,hash(JSON.stringify(P.state(s)))]);
  }
  const meta={id:`five-slots-${game.seed}-${p}-${game.mode}`,seed:game.seed,tick_ms:100,max_ticks:2400,scene:'native',native_rules:true,
   action_schema:'move-v1+five-item-slots-v1',ability_actions:7,rule_version:'bun-native-v2',model:'five-slot technical acceptance',checkpoint:cfg.checkpoint,
   checkpoint_sha256:cfg.policy_sha256,model_sha256:loaded.sha,simulator_sha256:hash(fs.readFileSync('web/sim-bun-native-v2.js')),opponent_sha256:hash(fs.readFileSync('web/bun_hunter_bot.js')),map_sha256:await P.hash(level),
   policy:game.mode,teacher:false,learner_seat:p,opponent:`production JS Hunter/${tier}`,opponent_tier:tier,frozen_birth:{schema:'qqt.frozen-birth/v1',cells:game.spawn},
   reward_status:'unrecorded',training_reward_status:'unrecorded',recorded_at:new Date().toISOString()};
  const doc={schema:P.SCHEMA,meta,actions,checks,events:[],rewards:[],summary:{ticks:s.t,done:s.done,winner:s.winner,reward:[null,null],failures:[]}};
  meta.replay_identity=await P.hash(P.identity(doc));P.validate(doc);const player=new P.Player(doc,level);await player.verify();
  const branches=[];
  if(firstOpportunity!==null){
   player.seek(firstOpportunity);player.checkpoint();const ownActions=A.jointActions(player.sim,p),opp=P.unpack(actions[firstOpportunity])[q],before=await P.hash(P.state(player.sim));
   for(const action of ownActions){player.restore(firstOpportunity);assert.equal(await P.hash(P.state(player.sim)),before);const joint=[];joint[p]=action.engine_action;joint[q]=opp;const info=player.sim.step(joint);branches.push({move:action.move,ability:action.ability,requested_slot:action.ability>=2?action.ability-2:null,after_hash:await P.hash(P.state(player.sim)),placed:info.placed[p],death:info.died[p]});}
  }
  const dir=cfg.output+'/'+meta.id;fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(dir+'/review.json',JSON.stringify(doc));fs.writeFileSync(dir+'/DECISIONS.json',JSON.stringify(decisions));
  const result={status:'PASS',seed:game.seed,seat:p,mode:game.mode,ticks:s.t,complete:s.done,winner:s.winner,teacher:false,placed,deaths,
   slot_by_phase:slotByPhase,phase_order:['combat','native_window','carrying'],own_alive_actionable_ticks:actionable,funnel:funnel.finish(s.t,s.maxSteps),r3:R3.exportMetrics(s),
   counterfactual:{same_snapshot_and_rng:true,first_opportunity_tick:firstOpportunity,branches,scope:'one-step physical alternatives; not Q and not full causal net benefit'},
   review_verified:true,review_sha256:hash(fs.readFileSync(dir+'/review.json')),model_json_sha256:loaded.sha,onnx_json_checked_inputs:loaded.model.verifiedInputs,onnx_json_max_error:loaded.model.forwardMaxError,limitations:['four technical games are not evidence of learned useful slot choice','banana benefit remains observational polyline estimate','glue followup is observational, no causal claim']};
  fs.writeFileSync(dir+'/RESULT.json',JSON.stringify(result));results.push(result);console.log(JSON.stringify({status:'COMPLETE_GAME',id:meta.id,ticks:s.t,winner:s.winner,firstOpportunity}));
 }
 fs.writeFileSync(cfg.output+'/COMPLETE.json',JSON.stringify({status:'PASS',episodes:results.length,independent_environment_seeds:new Set(results.map(x=>x.seed)).size,paired_inference_modes:true,results,onnx_json_max_error:loaded.model.forwardMaxError},null,2));await loaded.session.release();
}
if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});
