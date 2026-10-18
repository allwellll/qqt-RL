'use strict';
// Offline CPU publication. SOURCE is read-only; only this checkout receives output.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('node:assert/strict'),zlib=require('zlib');
const Q=require('../web/sim'),P=require('../web/review_protocol'),M=require('./je10_review_math');
const {softmax}=require('./diagnostic_counterfactual'),G=require('./generate_diagnostic_replays');
const TASKS=['real_drop','native_death_stock','native_sugar_stock','natural_full2400'];
const BATCH='je10-best-completed-20261018',MODEL='Bun392 JE10 v38 r01 g8';
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const hash=x=>sha(JSON.stringify(x));
const rawParts=(b,s,info)=>({transport_capture_observed:s.bunStored[0][1]>b.bunStored[0][1]&&!info.died[0]&&!info.mutualDeath?24:0,death:info.died[0]?-12:0,own_bomb_defeat:info.ownBombDefeat?.[0]?-16:0,mutual_death:info.mutualDeath?-12:0,carrier_drop:info.died[0]&&b.bunCarried[0]>=0?-.5:0});
const sum=o=>Object.values(o).reduce((a,b)=>a+b,0);
function routeStep(obj,next,ledger,terminal){
  let phiBefore=ledger.phi,phiAfter=ledger.phi,offset=ledger.offset,identity=ledger.identity;
  if(obj){if(identity!==JSON.stringify(obj.target))offset=ledger.phi-obj.phi;
    phiBefore=obj.phi+offset;
    phiAfter=terminal?0:next&&JSON.stringify(next.target)===JSON.stringify(obj.target)?next.phi+offset:phiBefore;
    identity=JSON.stringify(obj.target);
  }
  return {phi_before:phiBefore,phi_after:phiAfter,reward:obj?.5*(.995*phiAfter-phiBefore):0,ledger:{phi:phiAfter,offset,identity}};
}
function clone(player){
  const s=P.replaySetup(player.document.meta,player.level);for(let i=0;i<player.draws;i++)s.rng();
  const rng=s.rng;Object.assign(s,structuredClone(Object.fromEntries(Object.entries(player.sim).filter(([,v])=>typeof v!=='function'))));s.rng=rng;return s;
}
async function main(){
  assert.equal(process.env.CUDA_VISIBLE_DEVICES,'','explicit CPU only');
  const source=process.argv[2],selection=JSON.parse(fs.readFileSync('runs/je10_release/selection.json')),chosen=selection.selected;
  const base=path.join(source,'runs/je10_bun_20261018',chosen.round),evalDir=path.join(base,'authority_je10',`g${String(chosen.generation).padStart(6,'0')}`),cp=path.join(base,'formal/bun392_bridge_pfsp8',`gen${String(chosen.generation).padStart(6,'0')}`);
  const seal=JSON.parse(fs.readFileSync(path.join(cp,'COMMITTED.json')));
  assert.equal(sha(fs.readFileSync(path.join(cp,'policy.pt'))),chosen.checkpoint_sha256);
  const stateDigest=crypto.createHash('sha256');for await(const chunk of fs.createReadStream(path.join(cp,'state.pkl')))stateDigest.update(chunk);assert.equal(stateDigest.digest('hex'),seal.state_sha256);
  const modelBytes=fs.readFileSync(path.join(evalDir,'model/model.json')),modelDoc=JSON.parse(modelBytes),onnxBytes=fs.readFileSync(path.join(evalDir,'model/model.onnx'));
  assert.equal(modelDoc.meta.checkpoint_sha256,chosen.checkpoint_sha256);assert.equal(sha(onnxBytes),modelDoc.meta.onnx_sha256);
  const ort=require(process.env.ORT_NODE_MODULE||'onnxruntime-node'),session=await ort.InferenceSession.create(onnxBytes,{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
  const actor=new Q.TransformerModel(modelDoc);actor.greedy=true;actor.inferEvery=1;
  fs.mkdirSync('web/models',{recursive:true});
  const stem='bun392-je10-v38-r01-g8',modelFile=`${stem}-${sha(modelBytes)}.json`,onnxFile=`${stem}-${sha(onnxBytes)}.onnx`;
  fs.writeFileSync('web/models/'+modelFile,modelBytes);fs.writeFileSync('web/models/'+onnxFile,onnxBytes);
  const reference=actor.forward(new Float32Array(24*13*15),new Float32Array(24));
  const release='web-bun392-je10-v38-r01-g8-20261018',url=`https://github.com/allwellll/qqt-RL/releases/download/${release}/`;
  const row={id:stem,display_name:MODEL,file:modelFile,url:url+modelFile,sha256:sha(modelBytes),bytes:modelBytes.length,candidate:'je10_v38_r01_attempt04_g8_raw',cycle:8,source_sha256:chosen.checkpoint_sha256,strategy:'raw',score:selection.candidates[0].balanced_score,
    description:'已完成固定协议的平衡选择：natural full交付6/16、七类路线62/112、Tactical12/64；Hunter hard0/64，困难战斗仍薄弱。诊断不代表综合晋升。',score_protocol:selection.rule,
    onnx:{file:onnxFile,url:url+onnxFile,sha256:sha(onnxBytes),bytes:onnxBytes.length,meta:{...modelDoc.meta,state_dim:24,attention_heads:4,generation:8},reference_zero:{move:Array.from(reference.move),bomb:Array.from(reference.bomb),value:reference.value}}};
  delete row.onnx.meta.onnx_path;
  const manifest=JSON.parse(fs.readFileSync('web/models.json'));manifest.models=[row,...manifest.models.filter(x=>x.id!==stem)];fs.writeFileSync('web/models.json',JSON.stringify(manifest,null,2)+'\n');
  const level=require('./eval_web_hunter').loadLevel(),catalog=JSON.parse(fs.readFileSync('web/review-catalog.json')),rows=[];
  for(const task of TASKS){
    const original=fs.readFileSync(path.join(evalDir,`REVIEW_FIXED_${task}.json`)),d=JSON.parse(original),originalHash=sha(original);
    assert.equal(d.meta.model_sha256,sha(modelBytes));assert.equal(d.meta.checkpoint_sha256,chosen.checkpoint_sha256);assert.equal(d.meta.seed,2026101800);
    const originalTicks=d.diagnostic_ticks;const originalPlayer=new P.Player(d,level);await originalPlayer.verify();
    const player=new P.Player(d,level),tracker=new M.Tracker(0);if(task==='real_drop')tracker.trapCause[1][0]=true;
    let ledger={phi:0,offset:0,identity:null},last=null;const diagnostics=[],checks=[[0,hash(P.state(player.sim))]],events=[],track={idle:0,reverse:0,distance:G.distance(player.sim,0)};
    for(let i=0;i<d.actions.length;i++){
      const s=player.sim,b=P.state(s),snapshot=M.snapshot(s),obj=M.objective(s,0,tracker),input=actor._getLaggedInputs(s,0);
      const res=await session.run({obs:new ort.Tensor('float32',input.obs,[1,24,13,15]),state:new ort.Tensor('float32',Float32Array.from(input.state),[1,24])});
      const logits={move:Array.from(res.move.data),ability:Array.from(res.bomb.data)},mask=s.legalMask(),am=actor._abilityMask(s,mask,0),pm=softmax(logits.move,mask.mm[0]),pa=softmax(logits.ability,am);
      const taken=[actor._sampleMasked(logits.move,mask.mm[0],Q.mulberry32(1)),actor._sampleMasked(logits.ability,am,Q.mulberry32(1))];
      assert.deepEqual(taken,d.actions[i].slice(0,2),'same checkpoint action re-inference');
      const candidates=[];let actual;
      for(let move=0;move<5;move++)for(let ability=0;ability<3;ability++){
        if(!mask.mm[0][move]||!am[ability])continue;
        const branch=clone(player),tr=Object.assign(Object.create(M.Tracker.prototype),structuredClone(tracker));
        const info=branch.step(P.unpack([move,ability,...d.actions[i].slice(2)]));tr.observe(branch,snapshot,info);
        const parts=rawParts(b,branch,info),causal=b.trapped[1]===0&&branch.trapped[1]>0&&info.causalDamageSource?.[1]?.[0]&&!info.causalDamageSource?.[1]?.[1];
        const route=routeStep(obj,M.objective(branch,0,tr),ledger,branch.done),raw=sum(parts),shaped=raw+(causal?4:0)+route.reward;
        const c={action:[move,ability],probability:pm[move]*pa[ability],raw_reward:raw,shaped_reward:shaped,next_state_sha256:hash(P.state(branch)),terminal:branch.done};candidates.push(c);
        if(move===taken[0]&&ability===taken[1])actual={route,parts,control:causal?4:0,info,state:P.state(branch)};
      }
      const mass=candidates.reduce((n,c)=>n+c.probability,0);const chosenC=candidates.find(c=>c.action[0]===taken[0]&&c.action[1]===taken[1]);
      const approx={};for(const role of ['raw','shaped']){approx[role+'_q1_taken']=chosenC[role+'_reward'];approx[role+'_q1_mean']=candidates.reduce((n,c)=>n+c.probability*c[role+'_reward'],0)/mass;approx[role+'_a1']=approx[role+'_q1_taken']-approx[role+'_q1_mean'];}
      approx.delta_a=approx.shaped_a1-approx.raw_a1;for(const key of ['raw_a1','shaped_a1','delta_a'])approx[key+'_increment']=last?approx[key]-last[key]:null;last=approx;
      player.step();const after=P.state(player.sim);assert.deepEqual(after,actual.state,'taken clone exact state');
      assert.equal(hash(after),originalTicks[i].state_after_sha256);tracker.observe(player.sim,snapshot,player.info);ledger=actual.route.ledger;
      const ev=G.events(b,player.sim,player.info,track);events.push(...ev);checks.push([i+1,hash(after)]);
      diagnostics.push({...originalTicks[i],events:ev,decision:{action:taken,move_mask:Array.from(mask.mm[0]),ability_mask:Array.from(am),move_probability:pm,ability_probability:pa,legal_count:candidates.length},training_reward:null,raw_a1:null,shaped_a1:null,approximate_a1:approx,counterfactuals:candidates,route_diagnostic:{phase:obj?.phase||'COMBAT',target:obj?.target||null,phi_before:actual.route.phi_before,phi_after:actual.route.phi_after},approximate_reward_components:{...actual.parts,control_entry:actual.control,route_observable_reconstruction:actual.route.reward}});
    }
    d.diagnostic_ticks=diagnostics;d.checks=checks;d.events=events;
    Object.assign(d.meta,{model:MODEL,map:'806',batch_id:BATCH,task,scene_name:task==='real_drop'?'真实掉包追取及返程诊断':task==='natural_full2400'?'原生自然完整对局':task==='native_death_stock'?'死亡窗口库存取包及返程诊断':'糖泡窗口库存取包及返程诊断',scene_rules:task==='natural_full2400'?'生产原生砖块/糖泡/道具；2400 tick上限；完整终局':'固定合法 reset fixture，1HP；生产原生规则；1200 tick上限；非自然初始化',onnx_sha256:sha(onnxBytes),original_replay_sha256:originalHash,training_state_sha256:seal.state_sha256,recording_scope:'fixed asynchronous evaluation diagnostic; not historical training rollout',
      approximate_a1:{schema:'qqt.je10-approx-a1/v1',status:'offline_approximation',future_value:0,gamma:.995,definition:'Q1_raw=r_raw+gamma*0; Q1_shaped=r_raw+control_entry+route_observable_reconstruction+gamma*0; A1=Q1(taken)-sum(pi_legal*Q1); delta_a=A1_shaped-A1_raw; increment=x[t]-x[t-1]',reward_definition:'r_raw=24*observed_capture-12*death-16*own_bomb_defeat-12*mutual_death-.5*carrier_death; control_entry=4 if observed exclusive learner-caused control; route=.5*(.995*Phi_after-Phi_before) when objective active',limitations:'零后续值 V=0 的当前可观测任务奖励子集与控制/路线辅助项；capture即时计入，未复原settlement延迟、cap/clawback、avoidable danger与其他hidden aux，不能视为完整当前或训练Reward；不是训练原始/塑形 Critic，不是长期动作优势。全部合法 move×ability，固定对手实际动作，同状态/随机流克隆；同checkpoint masked softmax乘积作基线，greedy执行。训练hidden aux/完整reward未记录，严格raw/shaped A1未记录。',reward_source_hashes:Object.fromEntries(['scripts/je05_bun_transport.py','scripts/je04_bun_ops.py','scripts/je10_control_credit.py','scripts/je10_ops.py'].map(f=>[f,sha(fs.readFileSync(path.join(source,f)))])),source_sha256:sha(fs.readFileSync(__filename)),route_source_sha256:sha(fs.readFileSync('scripts/je10_review_math.js')),measured_count:diagnostics.length}});
    // The production evaluation has no policy-history rollout; keep strict fields absent/null.
    d.meta.replay_identity=await P.hash(P.identity(d));P.validate(d);await new P.Player(d,level).verify();
    const bytes=Buffer.from(JSON.stringify(d)),file=d.meta.id+'.json';fs.writeFileSync('web/review-replays/'+file,bytes);
    rows.push({id:d.meta.id,file,seed:d.meta.seed,scene:'native',task,scene_name:d.meta.scene_name,scene_rules:d.meta.scene_rules,model:MODEL,opponent:d.meta.opponent,checkpoint_sha256:chosen.checkpoint_sha256,model_sha256:sha(modelBytes),onnx_sha256:sha(onnxBytes),map:'806',map_sha256:d.meta.map_sha256,diagnostic:true,batch_id:BATCH,summary:d.summary,sha256:sha(bytes),bytes:bytes.length,gzip_bytes:zlib.gzipSync(bytes).length});
    console.log(task,d.summary.ticks,'all selected actions and cloned states exact; approximate A1 100%');
  }
  catalog.model=MODEL+'；保留历史 Review 对照';catalog.replays=[...rows,...catalog.replays.filter(r=>r.batch_id!==BATCH)];catalog.raw_bytes=catalog.replays.reduce((n,r)=>n+r.bytes,0);catalog.gzip_bytes=catalog.replays.reduce((n,r)=>n+r.gzip_bytes,0);
  catalog.je10_selection={...selection,source:'same frozen asynchronous protocol; 37 completed points at release snapshot'};
  fs.writeFileSync('web/review-catalog.json',JSON.stringify(catalog,null,2)+'\n');await session.release();
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
module.exports={routeStep,rawParts};
