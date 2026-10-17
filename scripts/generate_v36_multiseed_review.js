'use strict';
// CPU-only fixed matrix; observed route diagnostics are never training telemetry.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict'),zlib=require('node:zlib');
const Q=require('../web/sim'),P=require('../web/review_protocol'),R=require('../web/review_route'),A=require('../web/review_advantage');
const authority=require('./eval_web_hunter'),G=require('./generate_diagnostic_replays');
const {BunRuleTacticalBot,stateFromSim}=require('../web/bun_rule_bot');
const ROOT=path.resolve(__dirname,'..');
const RUN='runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36_attempt02';
const CHECKPOINT='e2dd2cb9890abf5753b34adcdb14abc93b3f977229c29326fd2033e584a73945';
const STATE='6bc988b647d4e0fad8dcef66dd47bbc8d3b4ea7dd75c3a8b2530c9b01fba55bd';
const MATRIX=Object.freeze(['native','native','open','open','carry_home','carry_home','tactical_v2','tactical_v2'].map((scene,i)=>Object.freeze({seed:2026101780+i,scene,opponent:scene==='tactical_v2'?'bun.tactical_v2':'bun.hunter/normal',max_ticks:scene==='carry_home'?300:2400})));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const hash=v=>sha(JSON.stringify(v));
const fileSha=f=>sha(fs.readFileSync(f));
const idFor=m=>`h8-v36-g32-${m.scene}-${m.seed}`;
const BLOCKER='同 checkpoint 只有 PBRS Critic，没有独立原始任务 Critic；自然 Web 状态没有完整训练 reward ledger / 隐藏 aux 映射，不能计算严格原始/塑形 Q1/A1/ΔA。未找到 v36 历史训练 trajectory；不能用 v35 轨迹或新评估局冒充。';
const ROUTE_LIMIT='观察状态重建的连续路线诊断，不是训练 episode telemetry；仅持敌包返程/原生死亡或控制窗口库存可观察目标，无法证明归因的散落包目标为缺测；拓扑改变的训练 safety certificate 未恢复时该步 Phi/PBRS 缺测并重置诊断 ledger。';
function readAudit(){return JSON.parse(fs.readFileSync('runs/replay_review_20261017/v36-audit.json'));}

async function loadActor(){
  const modelDir=path.join(RUN,'authority_je09_v1/bun392_bridge_pfsp8_g000032/model');
  const bytes=fs.readFileSync(path.join(modelDir,'model.json')),doc=JSON.parse(bytes),onnx=fs.readFileSync(path.join(modelDir,'model.onnx'));
  assert.equal(doc.meta.checkpoint_sha256,CHECKPOINT);assert.equal(sha(onnx),doc.meta.onnx_sha256);
  const ort=require(process.env.ORT_NODE_MODULE||'onnxruntime-node');
  const session=await ort.InferenceSession.create(path.join(modelDir,'model.onnx'),{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
  const model=new Q.TransformerModel(doc);model.greedy=true;model.inferEvery=1;
  let parity;
  return {sha:sha(bytes),onnx_sha:sha(onnx),getParity:()=>parity,close:()=>session.release(),async act(s,rng){
    const {obs,state}=model._getLaggedInputs(s,0);
    const result=await session.run({obs:new ort.Tensor('float32',obs,[1,24,13,15]),state:new ort.Tensor('float32',Float32Array.from(state),[1,24])});
    const logits={move:Array.from(result.move.data),ability:Array.from(result.bomb.data)};
    if(parity===undefined){const ref=model.forward(obs,state);parity=Math.max(...logits.move.map((v,i)=>Math.abs(v-ref.move[i])),...logits.ability.map((v,i)=>Math.abs(v-ref.bomb[i])));assert(parity<.002);}
    const mask=s.legalMask(),ability=model._abilityMask(s,mask,0);
    const soft=(x,m)=>{const peak=Math.max(...x.filter((_,i)=>m[i])),p=x.map((v,i)=>m[i]?Math.exp(v-peak):0),sum=p.reduce((a,b)=>a+b,0);return p.map(v=>v/sum);};
    const pm=soft(logits.move,mask.mm[0]),pa=soft(logits.ability,ability);
    const action=[model._sampleMasked(logits.move,mask.mm[0],rng),model._sampleMasked(logits.ability,ability,rng)];
    return {action,move_mask:Array.from(mask.mm[0],Boolean),ability_mask:Array.from(ability,Boolean),move_probability:pm,ability_probability:pa,legal_count:pm.filter(v=>v>0).length*pa.filter(v=>v>0).length};
  }};
}

// Production metric: reverse shortest paths, 1+3 weight on entered bricks.
function weightedMap(s,base){
  const d=Array(195).fill(32767),seen=Array(195).fill(false);
  for(let r=base[0];r<base[0]+3;r++)for(let c=base[1];c<base[1]+3;c++)if(!s.wall[r*15+c])d[r*15+c]=0;
  for(let n=0;n<195;n++){
    let i=-1;for(let j=0;j<195;j++)if(!seen[j]&&(i<0||d[j]<d[i]))i=j;
    if(i<0||d[i]===32767)break;seen[i]=true;const r=Math.floor(i/15),c=i%15;
    for(const [dr,dc] of Q.DIRS.slice(0,4)){const rr=r+dr,cc=c+dc,j=rr*15+cc;if(rr>=0&&rr<13&&cc>=0&&cc<15&&!s.wall[j])d[j]=Math.min(d[j],d[i]+1+3*Number(Boolean(s.brick[i])));}
  }
  return d;
}
function objective(s,p=0){
  const enemy=1-p,carry=s.alive[p]&&s.bunCarried[p]===enemy;
  const stock=s.alive[p]&&s.bunCarried[p]<0&&(s.bunRespawn[enemy]>0||s.trapped[enemy]>0)&&s.bunStored[enemy][enemy]>0;
  const home=carry&&s._bunBaseTeam(...s.centerCell(p))===p;
  if(!carry&&!stock)return {phase:'NONE',identity:null,target:null,distance:null,base:0};
  const phase=home?'D_DELIVERY':carry?'C_RETURN':'B_STOCK',goal=carry?p:enemy,grid=weightedMap(s,s.bunBases[goal]),pos=[s.pos[p*2],s.pos[p*2+1]],r=Math.floor(pos[0]),c=Math.floor(pos[1]);
  let distance=32767;
  for(const [dr,dc] of [[0,0],...Q.DIRS.slice(0,4)]){const rr=r+dr,cc=c+dc,j=rr*15+cc;if(rr<0||rr>=13||cc<0||cc>=15||s.wall[j]||grid[j]>=32767)continue;const local=(Math.abs(pos[0]-(rr+.5))+Math.abs(pos[1]-(cc+.5)))*(dr||dc?1+3*Number(Boolean(s.brick[j])):1);distance=Math.min(distance,local+grid[j]);}
  return {phase,identity:`${phase}:${carry?enemy+1:s.bunStored[enemy][enemy]}`,target:{type:carry?'home_cage':'enemy_cage',provenance:carry?'held_enemy_bun':'enemy_stock',id:'target-'+hash({team:p,goal,phase})},distance:distance===32767?'INF':distance,base:distance===32767?-1:-Math.min(distance,64)/64};
}
function routeStep(before,after,oldObj,newObj,ledger){
  const changed=before.wall.some((v,i)=>v!==after.wall[i])||before.brick.some((v,i)=>v!==after.brick[i]);
  // This excludes unproved topology safety and loose-drop credit, never fabricates it.
  const loose=before.bunLoose.some((v,i)=>i%2===1&&v>0)||after.bunLoose.some((v,i)=>i%2===1&&v>0);
  const missing=(changed&&(oldObj.phase!=='NONE'||newObj.phase!=='NONE'))||(loose&&oldObj.phase==='NONE');
  if(missing){ledger.phi=0;ledger.identity=null;return {phase:oldObj.phase,target:oldObj.target,distance_before:oldObj.distance,distance_after:newObj.distance,phi_before:null,phi_after:null,potential_delta:null,shaped_reward:null,terminal:after.done,missing_reason:changed?'unverified_topology_safety':'unverified_drop_provenance'};}
  const active=oldObj.phase!=='NONE',same=active&&ledger.identity===oldObj.identity;
  const offset=same?ledger.offset:ledger.phi-oldObj.base;
  const pb=active?oldObj.base+offset:ledger.phi;
  const sameNext=newObj.phase!=='NONE'&&newObj.identity===oldObj.identity;
  const pa=after.done?0:sameNext?newObj.base+offset:pb;
  const delta=.995*pa-pb;
  ledger.phi=pa;ledger.offset=after.done?0:offset;ledger.identity=after.done?null:oldObj.identity;
  return {phase:oldObj.phase,target:oldObj.target,distance_before:oldObj.distance,distance_after:newObj.distance,phi_before:pb,phi_after:pa,potential_delta:delta,shaped_reward:.05*delta,terminal:after.done,phase_transition:oldObj.identity!==newObj.identity,invalidated:active&&newObj.phase==='NONE'&&!after.done,terminal_correction:after.done&&pb!==0,missing_reason:null};
}
function routeColumns(rows,meta){
  const out={schema:R.SCHEMA,scale:1e-6,targets:[]};for(const k of R.COLUMNS)out[k]=[];
  const targets=new Map();
  for(const r of rows){
    let target=null;if(r.target){if(!targets.has(r.target.id)){targets.set(r.target.id,out.targets.length);out.targets.push(r.target);}target=targets.get(r.target.id);}
    for(const k of R.COLUMNS){let v=null;
      if(k==='ticks')v=r.tick;else if(k==='route_phase')v=r.phase;else if(k==='target')v=target;
      else if(k==='route_potential_delta')v=r.potential_delta;else if(k==='route_shaped_reward')v=r.shaped_reward;
      else if(Object.hasOwn(r,k))v=r[k];
      if(R.VALUES.includes(k)&&v!==null)v=Math.round(v/1e-6);out[k].push(v);
    }
  }
  const measured=rows.filter(r=>r.phi_before!==null).length;
  meta.route={schema:R.SCHEMA,kind:'production',status:'recorded',gamma:.995,lambda:.05,checkpoint_sha256:CHECKPOINT,policy_sha256:meta.model_sha256,simulator_sha256:meta.simulator_sha256,reward_ledger_sha256:meta.source_hashes['scripts/generate_v36_multiseed_review.js'],route_implementation_sha256:meta.source_hashes['scripts/je09_route_pbrs.py'],source_sha256:hash(rows),recomputed_sha256:hash(rows),source_run:'stage_h8_route_pbrs_continuous_v36_attempt02',generated_at:'2026-10-17T15:50:00Z',distance_metric:'production_weighted_grid_v1',shaped_critic:null,measured_count:measured,missing_count:rows.length-measured,shaped_measured_count:0,shaped_missing_count:rows.length,semantics:'diagnostic-post-target-uncorrected/v1',limitations:ROUTE_LIMIT};
  return out;
}
async function episode(m,actor,audit){
  const level=authority.loadLevel(),scene=m.scene==='tactical_v2'?'native':m.scene;
  const meta={id:idFor(m),episode_id:idFor(m),seed:m.seed,scene,variant:scene==='open'?'open':'native',tick_ms:100,max_ticks:m.max_ticks,model:'H8 v36 连续路线 PBRS g32 raw Actor',checkpoint:'H8/v36/attempt02/g32/raw policy.pt',checkpoint_sha256:CHECKPOINT,model_sha256:actor.sha,onnx_sha256:actor.onnx_sha,simulator_sha256:fileSha('web/sim.js'),opponent:m.opponent,opponent_sha256:fileSha(m.opponent==='bun.tactical_v2'?'web/bun_rule_bot.js':'web/bun_hunter_bot.js'),map:'806',map_sha256:await P.hash(level),policy:'greedy',backend:'onnxruntime-node/cpu/fp32',diagnostic:true,scene_name:m.scene==='carry_home'?'预置持包返程诊断':m.scene==='tactical_v2'?'Tactical Arena 完整规则终局':m.scene==='open'?'开放抢包对战':'原生抢包对战',scene_rules:m.scene==='carry_home'?'je05 carry_home：预置持敌包、清砖、1HP；300 tick；非自然抢包':m.scene==='tactical_v2'?'Tactical Arena 清砖、无道具、默认 Sim 规则；2400 tick 完整终局':m.scene==='open'?'清可破坏砖；原生糖泡/道具；2400 tick':'原生砖块/糖泡/道具；2400 tick',reward_status:'unrecorded',training_reward_status:'unrecorded',advantages_status:'blocked',advantage_status:'无法严格复算',teacher:{eligible:null,query:null,accept:null,reject:null},batch_id:'v36-fixed-eight-20261017',training_audit:audit,
    source_hashes:Object.fromEntries(['web/sim.js','web/bun_hunter_bot.js','web/bun_rule_bot.js','web/bot_contract.js','web/tactical_arena.js','scripts/generate_v36_multiseed_review.js','scripts/je09_route_pbrs.py'].map(f=>[f,fileSha(f)]))};
  if(m.scene==='tactical_v2'){meta.arena='tactical_v2';meta.native_rules=false;meta.arena_map_sha256=await P.hash(require('../web/tactical_arena').buildTacticalArena(level));}
  const s=P.replaySetup(meta,level),rng=Q.mulberry32(m.seed^0x515154),tactical=m.opponent==='bun.tactical_v2'?new BunRuleTacticalBot():null,hunter=tactical?null:authority.createHunter('normal');
  hunter?.adapter.reset({schema:'qqt.bot.context/v1',episode_id:meta.id,seed:m.seed,ruleset:'bun',max_ticks:m.max_ticks,metadata:{}});
  const d={schema:P.SCHEMA,meta,actions:[],events:[],rewards:[],checks:[[0,await P.hash(P.state(s))]],diagnostic_ticks:[]};
  const track={idle:0,reverse:0,distance:G.distance(s,0)},ledger={phi:0,offset:0,identity:null},routes=[];
  let draws=0;const simRng=s.rng;s.rng=()=>{draws++;return simRng();};
  while(!s.done){
    const before=P.state(s),oldObj=objective(s),decision=await actor.act(s,rng),raw=tactical?.act(s,1),op=raw?{move:raw[0],ability:raw[2]?2:raw[1]?1:0}:hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},1,rng);
    const row=[...decision.action,op.move,op.ability],drawBefore=draws;d.actions.push(row);
    const info=s.step(P.unpack(row));tactical?.observeTransition(info,stateFromSim(s),1);
    const after=P.state(s),events=G.events(before,s,info,track),r={tick:s.t,...routeStep(before,after,oldObj,objective(s),ledger)};
    d.events.push(...events);routes.push(r);const stateHash=await P.hash(after);d.checks.push([s.t,stateHash]);
    d.diagnostic_ticks.push({tick:s.t,state_before_sha256:d.checks.at(-2)[1],state_after_sha256:stateHash,rng_draws_before:drawBefore,rng_draws_after:draws,decision,events,native_info_sha256:hash(info),reward_components:null,training_reward:null,raw_a1:null,shaped_a1:null,cross_critic_delta_a:null,raw_time_delta_a:null,shaped_time_delta_a:null,route:r});
    if(s.t>m.max_ticks)throw Error('terminal not reached');
  }
  const failures=['enemy_drop_unpicked','window_closed_empty','return_deviation','cage_no_delivery','self_bomb','stagnation'].filter(k=>d.events.some(e=>e.player===0&&e.kind===k));
  if(s.winner===1)failures.push('match_loss');if(!d.events.some(e=>e.kind==='delivery'&&e.player===0))failures.push('no_delivery');
  d.summary={ticks:s.t,done:s.done,winner:s.winner,score:s.bunScore.slice(),stored:s.bunStored.map(x=>x.slice()),reward:[null,null],failures};
  meta.advantage_audit={status:'blocked',definition:'Q1=r+gamma*(terminal?0:V(next)); A1=Q1(taken)-sum(pi_legal*Q1)',reason:BLOCKER,checkpoint_sha256:CHECKPOINT,training_state_sha256:STATE,policy_sha256:actor.sha,shaped_critic_sha256:audit.critic_sha256,raw_task_critic_sha256:null,raw_task_a1_status:'blocked',shaped_a1_status:'blocked',measured_count:0,missing_count:s.t};
  meta.advantage={...A.missing({...meta,gamma:.995}),reason:BLOCKER};d.advantages=A.empty();
  d.route=routeColumns(routes,meta);meta.diagnostic_ticks_sha256=hash(d.diagnostic_ticks);meta.replay_identity=await P.hash(P.identity(d));P.validate(d);await new P.Player(d,level).verify();return d;
}
async function verifyDocument(d,m){
  P.validate(d);assert.equal(d.meta.checkpoint_sha256,CHECKPOINT);assert.equal(d.meta.seed,m.seed);assert.equal(d.meta.opponent,m.opponent);assert.equal(d.checks.length,d.actions.length+1);assert.equal(d.diagnostic_ticks.length,d.actions.length);assert.equal(hash(d.diagnostic_ticks),d.meta.diagnostic_ticks_sha256);
  const player=new P.Player(d,authority.loadLevel());await player.verify();player.reset();
  for(let i=0;i<d.actions.length;i++){const tick=d.diagnostic_ticks[i];assert.equal(tick.state_before_sha256,await P.hash(P.state(player.sim)));player.step();assert.equal(tick.state_after_sha256,await P.hash(P.state(player.sim)));assert.equal(tick.raw_a1,null);assert.equal(tick.shaped_a1,null);assert.equal(tick.reward_components,null);}
}
async function verifyBatch(catalog){
  const rows=catalog.replays.filter(r=>r.batch_id==='v36-fixed-eight-20261017');assert.equal(rows.length,8);assert.deepEqual(rows.map(r=>r.seed),MATRIX.map(m=>m.seed));
  for(const m of MATRIX){const row=rows.find(r=>r.id===idFor(m));assert(row);const bytes=fs.readFileSync(path.join(ROOT,'web/review-replays',row.file));assert.equal(sha(bytes),row.sha256);assert.equal(bytes.length,row.bytes);await verifyDocument(JSON.parse(bytes),m);}
}
async function main(){
  assert.equal(process.env.CUDA_VISIBLE_DEVICES,'');const audit=readAudit();assert.equal(audit.policy_sha256,CHECKPOINT);assert.equal(audit.state_sha256,STATE);const actor=await loadActor();
  fs.mkdirSync('runs/replay_review_20261017/multiseed',{recursive:true});
  const created=[];
  try{for(const m of MATRIX){const d=await episode(m,actor,audit),privatePath=`runs/replay_review_20261017/multiseed/${idFor(m)}.json`;fs.writeFileSync(privatePath,JSON.stringify(d));console.log(JSON.stringify({seed:m.seed,scene:m.scene,...d.summary}));created.push(d);}
    // Entire matrix is re-inferred, including real opponent decisions; retain all outcomes.
    for(let i=0;i<MATRIX.length;i++){const rerun=await episode(MATRIX[i],actor,audit);assert.deepEqual(rerun,created[i],'full policy/opponent/route/hash rerun');console.log('EXACT',MATRIX[i].seed);}
    const cp='web/review-catalog.json',catalog=JSON.parse(fs.readFileSync(cp)),rows=[];
    for(const d of created){const file=d.meta.id+'.json',bytes=Buffer.from(JSON.stringify(d));fs.writeFileSync('web/review-replays/'+file,bytes);rows.push({id:d.meta.id,file,seed:d.meta.seed,scene:d.meta.scene,variant:d.meta.variant,scene_name:d.meta.scene_name,scene_rules:d.meta.scene_rules,model:d.meta.model,opponent:d.meta.opponent,checkpoint_sha256:CHECKPOINT,map:'806',map_sha256:d.meta.map_sha256,diagnostic:true,batch_id:d.meta.batch_id,summary:d.summary,sha256:sha(bytes),bytes:bytes.length,gzip_bytes:zlib.gzipSync(bytes,{level:9}).length});}
    catalog.replays=[...catalog.replays.filter(r=>!MATRIX.some(m=>r.id===idFor(m))),...rows];catalog.v36_fixed_eight={schema:'qqt.fixed-seed-batch/v1',seeds:MATRIX,selection:'all eight declared seeds retained; no post-hoc selection or replacement',predeclared_contract:'reports/v36_multiseed_review_20261017.md',checkpoint_sha256:CHECKPOINT,policy_sha256:actor.sha,onnx_js_max_error:actor.getParity(),full_policy_opponent_reruns:8,strict_advantage_coverage:0,training_trajectory_status:'blocked: no preserved v36 historical rollout',limitations:BLOCKER};catalog.screened_episodes=catalog.replays.length;catalog.raw_bytes=catalog.replays.reduce((n,r)=>n+r.bytes,0);catalog.gzip_bytes=catalog.replays.reduce((n,r)=>n+r.gzip_bytes,0);await verifyBatch(catalog);fs.writeFileSync(cp,JSON.stringify(catalog,null,2)+'\n');
  }finally{await actor.close();}
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
module.exports={MATRIX,verifyBatch,episode,loadActor,weightedMap,objective,routeStep,idFor};
