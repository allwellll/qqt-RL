'use strict';
// Read-only source audit; only the compact replay and audit receipt are published.
const fs=require('fs'),path=require('path'),assert=require('node:assert/strict'),crypto=require('crypto'),zlib=require('zlib');
const Q=require('../web/review-sim/route-v35'),P=require('../web/review_protocol'),R=require('../web/review_route'),A=require('../web/review_advantage');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const EXPECTED={
  'actor_model.json':'7bdfea4b1f8d3cc05a931cf2760ac8d5edc64494f49d0050dcf7b2079f336a2c',
  'actor.onnx':'8590e94d3c2123945b493b8c792b6a8b0a76b4d4ad91eebf99164b5232ace484',
  'actor_policy.pt':'e141311268c6eaf419f1f651252bb27d46c106aeb407230c20eed4c8897a34e2',
  'critic_params.npz':'3174b62f39ea9d56f206c917d2107a8f5e31bda2fe693e88630f72715c0604d8',
  'training_config.json':'84592654e26735440a664afa3f85e85b496b07d733864b94d86b2b4352fd1e93',
  'trajectory.json':'a98bcdaee0e3c5ca73049f0bae42fcfd571da083737a929aba00c3c4673d388c',
  'recalculation_proof.json':'c84fbe93dda39c68c2b28cd099baf3d1569eccd428c5894d4b054fd2803de187'};
function snapshot(s){return {carry:s.bunCarried.slice(),loose:Array.from(s.bunLoose),stock:s.bunStored.map(a=>a.slice()),pos:Array.from(s.pos),alive:s.alive.slice(),trap:s.trapped.slice(),brick:Array.from(s.brick)};}
// Independent shortest paths by repeated relaxation, rather than source Dijkstra.
function distance(s,goals,pos){
  const d=Array(195).fill(32767);for(const g of goals)if(!s.wall[g])d[g]=0;
  for(let pass=0;pass<195;pass++){
    let changed=false;
    for(let i=0;i<195;i++)if(!s.wall[i])for(const j of [i-15,i+15,...(i%15?[i-1]:[]),...(i%15<14?[i+1]:[])]){
      if(j<0||j>=195||s.wall[j])continue;
      const v=d[j]+1+3*Number(Boolean(s.brick[j]));if(v<d[i]){d[i]=v;changed=true;}
    }
    if(!changed)break;
  }
  return d[Math.floor(pos[0])*15+Math.floor(pos[1])];
}
async function main(){
  assert.equal(process.env.CUDA_VISIBLE_DEVICES,'','explicit CPU isolation');
  const [source,output='web']=process.argv.slice(2);assert(source,'usage: SOURCE_PACKAGE [WEB_OUTPUT]');
  const hashes={};for(const [name,expected]of Object.entries(EXPECTED)){const b=fs.readFileSync(path.join(source,name));assert.equal(sha(b),expected,name);hashes[name]={sha256:expected,bytes:b.length};}
  const trace=JSON.parse(fs.readFileSync(path.join(source,'trajectory.json'))),modelDoc=JSON.parse(fs.readFileSync(path.join(source,'actor_model.json')));
  assert.equal(trace.simulator_sha256,sha(fs.readFileSync('web/review-sim/route-v35.js')));assert.equal(trace.model_sha256,EXPECTED['actor_model.json']);assert.equal(trace.manifest_sha256,sha(JSON.stringify(trace.manifest)));assert.equal(trace.manifest.assist,false);assert.equal(trace.manifest.seed,2026101732);assert.equal(trace.trajectory.length,1200);
  const config=JSON.parse(fs.readFileSync(path.join(source,'training_config.json')));
  assert.equal(config.gamma,.995);
  const ort=require(process.env.ORT_NODE_MODULE||'onnxruntime-node'),session=await ort.InferenceSession.create(path.join(source,'actor.onnx'),{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
  const model=new Q.TransformerModel(modelDoc);model.greedy=true;model.inferEvery=1;
  const receiptPath=path.join(output,'review-evidence/route-v35-g32.json');
  const level=require('./eval_web_hunter').loadLevel(),generated=process.env.ROUTE_GENERATED_AT||(fs.existsSync(receiptPath)?JSON.parse(fs.readFileSync(receiptPath)).generated_at:new Date().toISOString());
  const meta={id:'h8-route-v35-g32-drop-return-2026101732',episode_id:'route-v35-g32-drop-return-2026101732',seed:2026101732,tick_ms:100,max_ticks:1200,
    scene:'route_drop_return',scene_name:'掉包返程诊断（非自然完整对局）',scene_rules:'生产诊断初态：清砖、1HP、敌持己包且 trap=1；assist-off；120秒；不是自然击杀证据',diagnostic:true,
    model:'H8 v35 路线 PBRS g32 raw',checkpoint:'H8/v35/g32/policy.pt',checkpoint_sha256:EXPECTED['actor_policy.pt'],model_sha256:EXPECTED['actor_model.json'],critic_sha256:EXPECTED['critic_params.npz'],onnx_sha256:EXPECTED['actor.onnx'],
    simulator_sha256:trace.simulator_sha256,opponent:'固定向上 / 不放泡 / 不用道具 [0,0,0]',opponent_sha256:sha('[0,0,0]'),map:'806',map_sha256:await P.hash(level),
    policy:'greedy',backend:'onnxruntime-node/cpu/fp32',assist:false,reward_status:'unrecorded',training_reward_status:'unrecorded',reward_source:'轨迹未采集原环境/训练 reward ledger；路线辅助 reward 单独记录，禁止相加冒充训练总 Reward',teacher:{eligible:null,query:null,accept:null,reject:null},
    evidence:'review-evidence/route-v35-g32.json'};
  const s=P.replaySetup(meta,level),rng=Q.mulberry32(meta.seed^0x515154),d={schema:P.SCHEMA,meta,actions:[],rewards:[],events:[],checks:[[0,await P.hash(P.state(s))]]};
  const route={schema:R.SCHEMA,scale:1e-6,targets:[]};for(const k of R.COLUMNS)route[k]=[];
  const targetMap=new Map();let target=null,credit=-1,objective=null,offset=0,priorPhi=0,maxResidual=0,parity=0,priorPhase=null,reverse=0;
  const recalculated=[];
  for(const row of trace.trajectory){
    const before=snapshot(s),phases=[P.phase(s,0),P.phase(s,1)];assert.deepEqual(before,row.state_before,'before '+row.tick);
    const input=model._getLaggedInputs(s,0);input.state=s.encodeStateJAX(0);input.state[9]+=s.trapped[0]/60;input.state[10]+=s.trapped[1]/60;
    const logits=await session.run({obs:new ort.Tensor('float32',input.obs,[1,24,13,15]),state:new ort.Tensor('float32',Float32Array.from(input.state),[1,24])});
    if(s.t===0){const ref=model.forward(input.obs,input.state);parity=Math.max(...['move','bomb'].flatMap(k=>Array.from(logits[k].data,(v,i)=>Math.abs(v-ref[k][i]))));assert(parity<.002);}
    const masks=s.legalMask(),raw=model._decodeAbility(model._sampleMasked(logits.move.data,masks.mm[0],rng),model._sampleMasked(logits.bomb.data,model._abilityMask(s,masks,0),rng));
    assert.deepEqual(raw.map(Number),row.action,'policy re-inference '+row.tick);
    const action=[row.action[0],row.action[2]?2:row.action[1]?1:0,0,0];d.actions.push(action);
    const info=s.step(P.unpack(action));assert.deepEqual(snapshot(s),row.state_after,'production state '+row.tick);
    assert.deepEqual({died:Array.from(info.died||[]),mutual_death:Boolean(info.mutualDeath),pickup:s.bunCarried[0]!==before.carry[0]},row.native_events,'native event '+row.tick);
    if(info.died[1]&&before.carry[1]===1){const cell=Math.floor(s.pos[2])*15+Math.floor(s.pos[3]);if(s.bunLoose[cell*2+1]>before.loose[cell*2+1]){target=cell;credit=0;}}
    if(target!==null&&s.bunLoose[target*2+1]===0)credit=-1;
    const phase=s.bunCarried[0]===1?3:target!==null&&credit===0&&s.bunLoose[target*2+1]>0?2:0,id=phase===3?2:target??-1;
    const goals=[];if(phase===3){const b=s.bunBases[0];for(let r=b[0];r<b[0]+3;r++)for(let c=b[1];c<b[1]+3;c++)goals.push(r*15+c);}else if(phase===2)goals.push(target);
    const dist=phase?distance(s,goals,s.pos):32767,distBefore=phase?distance({...s,brick:before.brick},goals,before.pos):32767;
    if(phase===3&&before.carry[0]===1){if(dist>distBefore)reverse++;else if(dist<distBefore)reverse=0;}else reverse=0;
    if(reverse===3&&dist>distBefore)d.events.push({tick:s.t,kind:'return_deviation',player:0});
    const base=phase?(dist<32767?-Math.min(dist,64)/64:-1):0,key=phase+':'+id;
    if(phase&&objective!==key){offset=priorPhi-base;objective=key;}
    const phi=phase?base+offset:priorPhi,delta=.995*phi-priorPhi,reward=.05*delta;
    assert.equal(phase,row.route_phase);assert.equal(id,row.target_identity);assert.equal(dist,row.distance);
    for(const [a,b]of [[priorPhi,row.phi_before],[phi,row.phi_after],[delta,row.potential_delta],[reward,row.shaped_reward]]){maxResidual=Math.max(maxResidual,Math.abs(a-b));assert(Math.abs(a-b)<1e-12);}
    for(let p=0;p<2;p++){
      const emit=kind=>d.events.push({tick:s.t,kind,player:p});
      if(phases[p]!==P.phase(s,p))d.events.push({tick:s.t,kind:'phase',player:p,from:phases[p],to:P.phase(s,p)});
      if(info.placed[p])emit('bomb');if(info.died[p])emit('death');if(!before.alive[p]&&s.alive[p])emit('revive');
      if(!before.trap[p]&&s.trapped[p])emit('control');if(before.trap[p]&&!s.trapped[p]&&s.alive[p])emit('control_end');
      if(before.carry[p]>=0&&info.died[p])emit('drop');if(before.carry[p]<0&&s.bunCarried[p]>=0)emit('pickup');if(s.bunStored[p][1-p]>before.stock[p][1-p])emit('delivery');
    }
    let ti=null;if(phase){const targetKey=phase+':'+id;if(!targetMap.has(targetKey)){targetMap.set(targetKey,route.targets.length);route.targets.push({type:phase===3?'home_cage':'loose_bun',provenance:phase===3?'held_enemy_bun':'enemy_carried_drop',id:'target-'+sha(meta.id+':'+targetKey)});}ti=targetMap.get(targetKey);}
    const values={ticks:s.t,route_phase:phase===3?'C_RETURN':phase===2?'A_ENEMY_DROP':'NONE',target:ti,distance_before:phase?(distBefore===32767?'INF':distBefore):null,distance_after:phase?(dist===32767?'INF':dist):null,
      phi_before:priorPhi,phi_after:phi,route_potential_delta:delta,route_shaped_reward:reward,phase_transition:priorPhase===null?null:phase!==priorPhase,terminal:s.done,terminal_correction:s.done?false:null};
    for(const k of R.COLUMNS)route[k].push(values[k]===undefined?null:R.VALUES.includes(k)?Math.round(values[k]/route.scale):values[k]);
    recalculated.push([s.t,phase,id,dist,priorPhi,phi,delta,reward]);priorPhi=phi;priorPhase=phase;
    if(s.t%200===0||s.done)d.checks.push([s.t,await P.hash(P.state(s))]);
  }
  await session.release();assert(s.done);assert.equal(s.t,1200);
  d.route=route;meta.route={schema:R.SCHEMA,kind:'production',status:'recorded',gamma:.995,lambda:.05,checkpoint_sha256:meta.checkpoint_sha256,policy_sha256:meta.model_sha256,simulator_sha256:meta.simulator_sha256,
    reward_ledger_sha256:sha(fs.readFileSync('scripts/je09_route_pbrs_replay.js')),route_implementation_sha256:sha(fs.readFileSync('scripts/je09_route_pbrs_replay.js')),source_sha256:EXPECTED['trajectory.json'],recomputed_sha256:sha(JSON.stringify(recalculated)),source_run:'stage_h8_route_pbrs_v35',generated_at:generated,distance_metric:'production_weighted_grid_v1',shaped_critic:null,measured_count:1200,missing_count:0,shaped_measured_count:0,shaped_missing_count:1200,
    semantics:'diagnostic-post-target-uncorrected/v1',limitations:'诊断导出器按转移后目标分类；前后路程按该同一目标独立重算。敌方 trap 因果归属预置，非自然击杀。末 tick Phi 未归零，非训练 PBRS 账本；拓扑/安全/失效未采集。Critic 参数已交接，但本轨迹缺训练 reward ledger 及同口径反事实结果，不能计算两类 A1。'};
  meta.advantage={...A.missing({...meta,gamma:.995}),reason:'本实验 Critic 已由塑形回报训练，不能冒充原始任务 Critic；轨迹缺训练 reward ledger 与严格联合动作反事实结果'};d.advantages=A.empty();
  meta.selection_rule='retain supplied predeclared seed, no selection; return_deviation = three increasing weighted grid distances while continuously carrying, reset on progress';
  d.summary={ticks:s.t,done:s.done,winner:s.winner,score:s.bunScore.slice(),stored:s.bunStored.map(a=>a.slice()),reward:[null,null],failures:d.events.some(e=>e.kind==='return_deviation')?['return_deviation']:[],route_reward:recalculated.reduce((n,r)=>n+r[7],0)};
  meta.replay_identity=await P.hash(P.identity(d));P.validate(d);await new P.Player(d,level).verify();
  const bytes=Buffer.from(JSON.stringify(d)),file=meta.id+'.json',receipt={schema:'qqt.route-g32-audit/v1',source_commit:'6358fa192c847016bf1745247efae2027804f549',generated_at:generated,files:hashes,simulator_sha256:meta.simulator_sha256,manifest_sha256:trace.manifest_sha256,checkpoint_state_sha256:'cdc1e497c0fa83d3ba2f9a5986e365d3a3953ea63cbacd106016829ffdd98f12',ticks:1200,actions_reinferred:1200,state_transitions_verified:1200,route_recalculated:1200,max_residual:maxResidual,onnx_js_max_error:parity,recomputed_sha256:meta.route.recomputed_sha256,final_state_sha256:d.checks.at(-1)[1],route_reward:d.summary.route_reward,final_phi:priorPhi,native_reward:null,training_total_reward:null,original_advantage:null,shaped_advantage:null,limitations:meta.route.limitations};
  fs.mkdirSync(path.join(output,'review-replays'),{recursive:true});fs.mkdirSync(path.join(output,'review-evidence'),{recursive:true});fs.writeFileSync(path.join(output,'review-replays',file),bytes);fs.writeFileSync(path.join(output,'review-evidence/route-v35-g32.json'),JSON.stringify(receipt,null,2)+'\n');
  const catalogPath=path.join(output,'review-catalog.json'),catalog=JSON.parse(fs.readFileSync(catalogPath));catalog.replays=catalog.replays.filter(r=>r.id!==meta.id);catalog.replays.unshift({...Object.fromEntries(['id','seed','scene','scene_name','scene_rules','model','opponent','checkpoint_sha256','critic_sha256','map','map_sha256','diagnostic'].map(k=>[k,meta[k]])),file,summary:d.summary,sha256:sha(bytes),bytes:bytes.length,gzip_bytes:zlib.gzipSync(bytes).length});
  if(!catalog.scenes.some(s=>s.id===meta.scene))catalog.scenes.unshift({id:meta.scene,name:meta.scene_name,source:'je08 production drop_return diagnostic; not a natural full match',base_map:806});
  catalog.replays[0].gzip_bytes=zlib.gzipSync(bytes,{level:9}).length;
  catalog.model='H8 v35 g32 路线诊断 / H8 v34 g32 原始任务优势';catalog.screened_episodes=7;catalog.raw_bytes=catalog.replays.reduce((n,r)=>n+r.bytes,0);catalog.gzip_bytes=catalog.replays.reduce((n,r)=>n+r.gzip_bytes,0);fs.writeFileSync(catalogPath,JSON.stringify(catalog,null,2)+'\n');console.log(JSON.stringify({ticks:1200,actions:1200,max_residual:maxResidual,replay_sha256:sha(bytes),bytes:bytes.length,gzip_bytes:zlib.gzipSync(bytes,{level:9}).length,route_reward:d.summary.route_reward}));
}
if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});
module.exports={EXPECTED,distance,snapshot};
