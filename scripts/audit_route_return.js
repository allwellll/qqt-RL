'use strict';
// Offline CPU audit. No checkpoint tensors or full simulator states are published.
const fs=require('fs'),path=require('path'),assert=require('node:assert/strict'),crypto=require('crypto'),zlib=require('zlib');
const {execFileSync}=require('child_process'),SOURCE_COMMIT='6358fa192c847016bf1745247efae2027804f549';
const P=require('../web/review_protocol'),F=require('../web/review_return'),Q=require('../web/review-sim/route-v35'),{EXPECTED}=require('./import_route_pbrs_g32');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex'),round=v=>Math.round(v*1e9)/1e9;
function softmax(logits,mask){const max=Math.max(...Array.from(logits).filter((_,i)=>mask[i])),p=Array.from(logits,(v,i)=>mask[i]?Math.exp(v-max):0),sum=p.reduce((n,v)=>n+v,0);return p.map(v=>v/sum);}
function clone(player){const fresh=P.replaySetup(player.document.meta,player.level);for(let i=0;i<player.draws;i++)fresh.rng();const s=Object.assign(Object.create(Object.getPrototypeOf(player.sim)),structuredClone(Object.fromEntries(Object.entries(player.sim).filter(([,v])=>typeof v!=='function'))));s.rng=fresh.rng;return s;}
async function main(){
  assert.equal(process.env.CUDA_VISIBLE_DEVICES,'','CPU isolation required');const source=process.argv[2];assert(source,'training source required');
  const pkg=path.join(source,'reports/je09_bun_20261017/route_pbrs_replay_g32'),file='web/review-replays/h8-route-v35-g32-drop-return-2026101732.json',bytes=fs.readFileSync(file),doc=JSON.parse(bytes),level=require('./eval_web_hunter').loadLevel();
  for(const name of ['actor_model.json','actor.onnx','training_config.json','trajectory.json'])assert.equal(sha(fs.readFileSync(path.join(pkg,name))),EXPECTED[name]);
  const sourceTrace=JSON.parse(fs.readFileSync(path.join(pkg,'trajectory.json'))),seal=JSON.parse(fs.readFileSync(path.join(source,'runs/je09_bun_20261017/stage_h8_route_pbrs_v35/SOURCE_SEAL.json'))),sourceHashes={};
  const sealedSource=name=>execFileSync('git',['-C',source,'show',SOURCE_COMMIT+':'+name]);
  for(const name of ['scripts/je09_stage_h5c_ops.py','scripts/je09_stage_h2_ops.py','scripts/je09_stage_h_reward.py','scripts/je09_stage_f_reward.py','scripts/je09_bun_ops.py','scripts/je09_bun_scenes.py','scripts/je09_route_pbrs.py','scripts/je09_stage_h8_pbrs_ops.py']){sourceHashes[name]=sha(sealedSource(name));assert.equal(sourceHashes[name],seal.sources[name],name+' sealed source');}
  for(const name of ['scripts/je09_route_pbrs_replay.js','scripts/je08_web_diagnostics.js'])sourceHashes[name]=sha(sealedSource(name));
  const ort=require(process.env.ORT_NODE_MODULE||'onnxruntime-node'),session=await ort.InferenceSession.create(path.join(pkg,'actor.onnx'),{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
  const model=new Q.TransformerModel(JSON.parse(fs.readFileSync(path.join(pkg,'actor_model.json'))));model.greedy=true;model.inferEvery=1;
  const player=new P.Player(doc,level),analysis=F.analyze(doc,level,P),rows=[],hash=crypto.createHash('sha256');let nativeFirst,allActions=0,branchCount=0,maxProbabilityError=0;
  for(let tick=1;tick<=doc.actions.length;tick++){
    const s=player.sim,input=model._getLaggedInputs(s,0);input.state=s.encodeStateJAX(0);input.state[9]+=s.trapped[0]/60;input.state[10]+=s.trapped[1]/60;
    const outputs=await session.run({obs:new ort.Tensor('float32',input.obs,[1,24,13,15]),state:new ort.Tensor('float32',Float32Array.from(input.state),[1,24])}),mask=s.legalMask(),am=model._abilityMask(s,mask,0),pm=softmax(outputs.move.data,mask.mm[0]),pa=softmax(outputs.bomb.data,am);
    const taken=[pm.indexOf(Math.max(...pm)),pa.indexOf(Math.max(...pa))];assert.deepEqual(taken,doc.actions[tick-1].slice(0,2));allActions++;
    const before=Array.from(s.pos).slice(0,2),grid=F.distances(s),route=F.path(s,grid),candidates=[];
    if(tick>=147&&tick<=207){
      const reference=model.forward(input.obs,input.state),refMove=softmax(reference.move,mask.mm[0]),refAbility=softmax(reference.bomb,am);
      maxProbabilityError=Math.max(maxProbabilityError,...pm.map((v,i)=>Math.abs(v-refMove[i])),...pa.map((v,i)=>Math.abs(v-refAbility[i])));assert(maxProbabilityError<2e-5,'independent JS/ONNX actor probability parity');
      for(let move=0;move<5;move++)for(let ability=0;ability<am.length;ability++)if(mask.mm[0][move]&&am[ability]){
        const branch=clone(player),info=branch.step(P.unpack([move,ability,...doc.actions[tick-1].slice(2)])),state=P.state(branch),stateHash=await P.hash(state),position=Array.from(branch.pos).slice(0,2),distance=F.distances(branch)[Math.floor(position[0])*15+Math.floor(position[1])];
        let collisionCorrection=false;if(move<4){const blocked=Uint8Array.from(s.wall,(v,i)=>Boolean(v||s.brick[i]||s.fuse[i])),step=Q.CFG.stepLen*s.spdG[0]*s.playerMoveScale(0),ideal=before.map((v,i)=>v+F.DIRS[move][i]*step),direct=s._tryMove(before[0],before[1],move,blocked,step);collisionCorrection=ideal.some((v,i)=>Math.abs(v-direct[i])>1e-6)||direct.some((v,i)=>Math.abs(v-position[i])>1e-6);}
        const value={action:[move,ability],probability:round(pm[move]*pa[ability]),position:position.map(round),distance,collision_correction:collisionCorrection,events:doc.events.filter(e=>e.tick===tick).map(e=>e.kind),q_original:null,q_shaped:null,state_sha256:stateHash};candidates.push(value);branchCount++;
        hash.update(JSON.stringify([tick,move,ability,pm[move]*pa[ability],stateHash,info]));
        if(move===taken[0]&&ability===taken[1]){const original=clone(player);original.step(P.unpack(doc.actions[tick-1]));assert.deepEqual(P.state(original),state,'same RNG taken branch');}
      }
    }
    player.step();if(tick===1)nativeFirst={causalDamageSource:player.info.causalDamageSource,creditedKill:player.info.creditedKill,causalKill:player.info.causalKill};
    assert.deepEqual(Array.from(player.sim.pos),sourceTrace.trajectory[tick-1].state_after.pos,'source trajectory positions');
    if(candidates.length){const selected=candidates.find(c=>JSON.stringify(c.action)===JSON.stringify(taken));assert.equal(selected.state_sha256,await P.hash(P.state(player.sim)));rows.push({tick,taken,position_before:before.map(round),position_after:Array.from(player.sim.pos).slice(0,2).map(round),cell:player.sim.centerCell(0),distance_before:doc.route.distance_before[tick-1],distance_after:doc.route.distance_after[tick-1],path_first_moves:route.first,path:route.cells,move_mask:mask.mm[0],ability_mask:am,move_probabilities:pm.map(round),ability_probabilities:pa.map(round),candidates,events:doc.events.filter(e=>e.tick===tick),phi:doc.route.phi_after[tick-1]*doc.route.scale,route_reward:doc.route.route_shaped_reward[tick-1]*doc.route.scale,original_a1:null,shaped_a1:null,cross_critic_delta:null});}
  }
  await session.release();
  const original=JSON.parse(fs.readFileSync('web/review-evidence/h8-g32-critic.json')),shaped=JSON.parse(fs.readFileSync('web/review-evidence/route-v35-critic.json'));
  // Audit evidence of the missing bridge, rather than silently assigning training-only state.
  const missingLinks=[
    '诊断仅在外部 Tracker 中预置 trapCause；原生 Sim tick1 的 causalDamageSource、creditedKill 均为 false。训练 scene9 则在原生状态预置 trapped_source/trapped_causal。两者因果 ledger 不等价，不能把 Tracker credit 偷换为训练 native info。',
    '完整训练 reward 依赖 scene、terminal/settlement、drop nonce/credit/cycle_paid、transport/bridge、carry_items、potential/routing 等 aux；源诊断导出器从未初始化或采集这些训练 aux，也未声明 JS→训练状态适配版本。COMMITTED 的批训练 aux 不属于本 seed 诊断 episode。',
    '训练 apply_reward 在阶段边界与 terminal 归零并做 topology/safety 核验；公开诊断按转移后目标重定基、末 tick Phi=.421875 未归零。诊断 route_shaped_reward 不是实际训练 PBRS reward，不能直接 env+它计算塑形 Q。',
    '现有 diagnostic_reward.py 是旧封存 ledger 的观测转移适配器，未验证本 v35 诊断的 scene9/因果初态/隐藏 aux 和全部反事实分支，不能直接更换 Critic 后宣称本实验真实 A1。'
  ];
  const witness=new P.Player(doc,level);witness.seek(176);const down=[];for(let i=0;i<2;i++){witness.sim.step(P.unpack([1,0,0,0]));const pos=Array.from(witness.sim.pos).slice(0,2);down.push({tick:177+i,position:pos.map(round),distance:F.distances(witness.sim)[Math.floor(pos[0])*15+Math.floor(pos[1])]});}
  const result={schema:F.SCHEMA,replay_id:doc.meta.id,replay_sha256:sha(bytes),policy_sha256:doc.meta.model_sha256,simulator_sha256:doc.meta.simulator_sha256,checkpoint_sha256:doc.meta.checkpoint_sha256,source_hashes:sourceHashes,window:{first:147,last:207,selection:'user-reported tick177 ±30 inclusive, retained without outcome selection'},cpu_only:true,actor_actions_verified:allActions,counterfactual_branches_verified:branchCount,counterfactual_sha256:hash.digest('hex'),gamma:.995,coordinate_order:'[row/y, col/x], zero based',home_cage:{top_left:[1,4],extent:[3,3],player:0},intervals:analysis.intervals,rows,native_first_tick:nativeFirst,down_two_steps:{scope:'production JS two-step position witness, fixed opponent; not a Q or advantage',steps:down},advantage_audit:{status:'unavailable',original:{role:'original',checkpoint:original.checkpoint_sha256,critic:original.critic_sha256,state:original.state_sha256},shaped:{role:'shaped',checkpoint:shaped.policy_sha256,critic:shaped.critic_sha256,state:shaped.state_sha256},policy_comparison:'both critics would use the same v35 actor legal joint policy distribution at the same state; not v34 actor probabilities',definition:'Q1=r+gamma*V(next); A1=Q_taken-sum(pi_legal*Q1); cross_critic_delta=A1_shaped-A1_original, distinct from temporal A1(t)-A1(t-1)',missing_links:missingLinks,required_artifacts:['versioned full JS-native-event → frozen training state/info adapter with exact causal/scene binding','declared episode initial training aux and all 1200 taken-ledger transitions','cloned ledger/RNG legal joint branches and same-generation critic parity for both roles','independent counterfactual reward/Q/A reproduction with terminal and phase handling']}};
  result.independent_js_probability_max_error=maxProbabilityError;F.validateAudit(result,doc,sha(bytes));
  const output='web/review-evidence/route-g32-return.json',data=Buffer.from(JSON.stringify(result));fs.writeFileSync(output,data);
  const cp='web/review-catalog.json',catalog=JSON.parse(fs.readFileSync(cp)),row=catalog.replays.find(r=>r.id===doc.meta.id);row.diagnostic_audit={file:'review-evidence/route-g32-return.json',schema:F.SCHEMA,sha256:sha(data),bytes:data.length,gzip_bytes:zlib.gzipSync(data,{level:9}).length};fs.writeFileSync(cp,JSON.stringify(catalog,null,2)+'\n');
  console.log(JSON.stringify({actor_actions_verified:allActions,rows:rows.length,branches:branchCount,intervals:result.intervals,bytes:data.length,gzip_bytes:row.diagnostic_audit.gzip_bytes,sha256:sha(data),tick177:rows.find(r=>r.tick===177)},null,2));
}
if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});
module.exports={softmax,clone};
