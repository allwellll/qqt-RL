'use strict';
// Fixed CPU/WASM evaluation on the production Web simulator; every seed survives.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const Q=require('../web/sim'),P=require('../web/review_protocol'),A=require('../web/review_advantage');
const H=require('./eval_web_hunter'),G=require('./generate_diagnostic_replays');
const {instrument}=require('./diagnostic_counterfactual');
const {loadOptionalModel,RECEIPT}=require('./best_d_model_input');
const E='web/review-evidence/best-d-20261028',PRIVATE=process.env.BEST_D_PRIVATE||'.hermes/best-d-private';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
async function main(){
  assert.equal(process.env.CUDA_VISIBLE_DEVICES,'','explicit CPU isolation required');
  const matrix=JSON.parse(fs.readFileSync(`${E}/matrix.json`)),receipt=JSON.parse(fs.readFileSync(RECEIPT));
  assert.equal(receipt.policy_sha256,matrix.checkpoint_sha256);
  const local=loadOptionalModel({receipt});
  const actorProof=JSON.parse(fs.readFileSync(`${E}/actor-parity.json`));
  assert.equal(receipt.model_sha256,actorProof.model_sha256);
  const source=process.env.BEST_D_ONNX;assert(source,'BEST_D_ONNX must name the verified same-checkpoint ONNX');
  const sourceModel=JSON.parse(fs.readFileSync(path.join(path.dirname(source),'model.json')));
  assert.equal(sourceModel.meta.checkpoint_sha256,matrix.checkpoint_sha256);
  const doc=local?.document||sourceModel;
  if(local){
    assert.equal(doc.meta.checkpoint_sha256,matrix.checkpoint_sha256);
    assert.equal(doc.flat,sourceModel.flat);assert.deepEqual(doc.tensors,sourceModel.tensors);
    console.log('Local export / ONNX companion tensors exact.');
  }
  assert.equal(sha(fs.readFileSync(source)),sourceModel.meta.onnx_sha256);
  assert.equal(sourceModel.meta.onnx_sha256,actorProof.wasm_graph_sha256);
  if(process.argv.includes('--check-inputs')){
    console.log('Generation source identities checked; local export tensors '+(local?'verified against companion':'SKIP')+'.');
    return;
  }
  const ort=require('onnxruntime-web');ort.env.wasm.numThreads=1;
  const session=await ort.InferenceSession.create(fs.readFileSync(source),{executionProviders:['wasm']});
  const actor=new Q.TransformerModel(doc);actor.greedy=true;actor.inferEvery=1;
  const level=H.loadLevel(),samples=[],summaries=[];
  fs.mkdirSync(PRIVATE,{recursive:true});
  for(const item of matrix.episodes){
    const {seed,scene,difficulty,max_ticks}=item,id=`d-slot05-u720-${scene}-${difficulty}-${seed}`;
    const meta={id,episode_id:`offline-${id}`,seed,scene,tick_ms:100,max_ticks,model_id:matrix.model_id,
      model:'D · ITER-PBT-V1 slot05 u720',checkpoint:matrix.checkpoint,checkpoint_sha256:matrix.checkpoint_sha256,
      model_sha256:receipt.model_sha256,opponent:`bun.hunter/${difficulty}`,simulator_sha256:sha(fs.readFileSync('web/sim.js')),
      opponent_sha256:sha(fs.readFileSync('web/bun_hunter_bot.js')),map:'806',map_sha256:await P.hash(level),
      scene_name:scene==='open'?`清砖对拼 · ${({hard:'困难',normal:'普通',easy:'简单'})[difficulty]}猎手`:scene==='native'?'原生整局抢包对战':'持包返程',
      scene_rules:scene==='open'?'806清砖；双方1血；300 tick；固定近距离出生；无抢包终局；原生对拼规则':scene==='native'?'原生806带砖、糖泡、道具；2400 tick；普通猎手；无预置携包':'806清砖；双方1血；预置模型携敌包；300 tick；普通猎手',
      native_rules:scene!=='open',...(scene==='open'?{evaluation_setup:'best_d_clear_duel_v1'}:{}),
      policy:'greedy',backend:'onnxruntime-web/wasm/cpu/single-thread',infer_every:1,assist:false,
      diagnostic:true,batch_id:'best-d-20261028',training_reward_status:'unrecorded',
      selection_rule:matrix.selection,matrix_sha256:sha(fs.readFileSync(`${E}/matrix.json`)),
      teacher:{eligible:null,query:null,accept:null,reject:null}};
    meta.advantage=A.missing({...meta,gamma:.995});meta.advantage.reason='未计算同状态全部合法联合动作的Reward/Critic反事实；raw A1、shaped A1、ΔA均未记录，Value/Logit不作为优势。';
    const sim=P.replaySetup(meta,level),hunter=H.createHunter(difficulty),rng=Q.mulberry32(seed^0x515154),step=instrument(sim);
    hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:meta.episode_id,seed,ruleset:'bun',max_ticks,metadata:{}});
    const replay={schema:P.SCHEMA,meta,actions:[],rewards:[],events:[],advantages:A.empty(),checks:[[0,await P.hash(P.state(sim))]]};
    const track={idle:0,reverse:0,distance:G.distance(sim,0)},totals={death:[0,0],delivery:[0,0],safe_kill:[0,0]};
    const stream=fs.createWriteStream(path.join(PRIVATE,`${id}.trace.jsonl`));
    const write=async v=>{if(!stream.write(JSON.stringify(v)+'\n'))await new Promise(r=>stream.once('drain',r));};
    await write({state:P.state(sim),legal:sim.legalMask()});
    while(!sim.done){
      const before=P.state(sim),{obs,state}=actor._getLaggedInputs(sim,0);
      const result=await session.run({obs:new ort.Tensor('float32',obs,[1,24,13,15]),state:new ort.Tensor('float32',Float32Array.from(state),[1,24])});
      const move=Array.from(result.move.data),bomb=Array.from(result.bomb.data);assert([...move,...bomb].every(Number.isFinite));
      const masks=sim.legalMask(),am=actor._abilityMask(sim,masks,0);
      const action=[actor._sampleMasked(move,masks.mm[0],rng),actor._sampleMasked(bomb,am,rng)];
      if([0,1,50,199].includes(sim.t)){
        const ref=actor.forward(obs,state),error=Math.max(...move.map((v,i)=>Math.abs(v-ref.move[i])),...bomb.map((v,i)=>Math.abs(v-ref.bomb[i])),Math.abs(result.value.data[0]-ref.value));
        assert(error<.002,`JS/WASM parity ${error}`);
        assert.deepEqual(action,[actor._sampleMasked(ref.move,masks.mm[0],rng),actor._sampleMasked(ref.bomb,am,rng)]);
        samples.push({id,tick:sim.t,obs:Array.from(obs),state:Array.from(state),wasm:{move,bomb,value:result.value.data[0]},js:{move:Array.from(ref.move),bomb:Array.from(ref.bomb),value:ref.value},max_error:error,action,move_mask:Array.from(masks.mm[0]),ability_mask:Array.from(am)});
      }
      const bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:sim.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim}},1,rng),row=[...action,bot.move,bot.ability];
      const entry=step(row);replay.actions.push(row);replay.events.push(...G.events(before,sim,entry.info,track));await write(entry);
      for(let p=0;p<2;p++){
        totals.death[p]+=Number(entry.info.died[p]);totals.delivery[p]+=replay.events.filter(e=>e.tick===sim.t&&e.kind==='delivery'&&e.player===p).length;
        const victim=1-p,info=entry.info;
        totals.safe_kill[p]+=Number(info.died[victim]&&!info.mutualDeath&&[info.physicalDamageSource[victim],info.causalDamageSource[victim]].some(src=>src[p]&&src.filter(Boolean).length===1));
      }
      if(sim.t%200===0||sim.done)replay.checks.push([sim.t,await P.hash(P.state(sim))]);
    }
    await new Promise(r=>stream.end(r));
    const failures=['enemy_drop_unpicked','window_closed_empty','return_deviation','cage_no_delivery','self_bomb','stagnation'].filter(k=>replay.events.some(e=>e.player===0&&e.kind===k));
    if(sim.winner===1)failures.push('match_loss');if(!totals.delivery[0]&&scene!=='open')failures.push('no_delivery');
    // Clear duel outcome uses safe kills/deaths, separate from the simulator's bun score.
    const outcome=scene==='open'?(totals.death[0]?'loss':totals.safe_kill[0]?'win':'draw'):(sim.winner===0?'win':sim.winner===1?'loss':'draw');
    if(scene==='open'&&outcome==='loss'&&!failures.includes('match_loss'))failures.push('match_loss');
    replay.summary={ticks:sim.t,done:sim.done,winner:sim.winner,score:sim.bunScore.slice(),stored:sim.bunStored.map(a=>a.slice()),reward:[0,0],failures,outcome,...totals};
    meta.replay_identity=await P.hash(P.identity(replay));P.validate(replay);
    fs.writeFileSync(path.join(PRIVATE,`${id}.json`),JSON.stringify(replay));
    summaries.push({id,...item,...replay.summary});console.log(id,outcome,'death',totals.death[0],'delivery',totals.delivery[0]);
  }
  await session.release();
  fs.writeFileSync(path.join(PRIVATE,'parity-inputs.json'),JSON.stringify(samples));
  fs.writeFileSync(path.join(PRIVATE,'screen.json'),JSON.stringify(summaries,null,2));
}
if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});
