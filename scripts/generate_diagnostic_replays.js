'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const Q=require('../web/sim'),P=require('../web/review_protocol'),authority=require('./eval_web_hunter');
const {BunRuleTacticalBot,stateFromSim}=require('../web/bun_rule_bot');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const input=process.env.REPLAY_INPUT||'runs/replay_review_20261017/inputs',output=process.env.REPLAY_EPISODES||'runs/replay_review_20261017/episodes';
const stage=process.env.REPLAY_STAGE||'h7';
const level=authority.loadLevel();
function distance(s,team) {
  const d=new Int16Array(195).fill(32767),queue=[],b=s.bunBases[team];
  for(let r=b[0];r<b[0]+3;r++)for(let c=b[1];c<b[1]+3;c++){const i=r*15+c;if(!s.wall[i]&&!s.brick[i]){d[i]=0;queue.push(i);}}
  for(let q=0;q<queue.length;q++){const i=queue[q],r=Math.floor(i/15),c=i%15;for(const [dr,dc] of Q.DIRS.slice(0,4)){const rr=r+dr,cc=c+dc,j=rr*15+cc;if(rr>=0&&rr<13&&cc>=0&&cc<15&&!s.wall[j]&&!s.brick[j]&&d[j]>d[i]+1){d[j]=d[i]+1;queue.push(j);}}}
  const [r,c]=s.centerCell(team);return d[r*15+c];
}
function events(before,s,info,track) {
  const out=[],t=s.t,emit=(kind,player,extra={})=>out.push({tick:t,kind,player,...extra});
  for(let p=0;p<2;p++){
    const enemy=1-p,cell=s.centerCell(p),i=cell[0]*15+cell[1];
    if(info.placed[p])emit('bomb',p);
    if(info.itemReleased[p])emit('use_item',p);
    if(info.died[p]){emit('death',p);if(info.ownBombDefeat[p])emit('self_bomb',p);}
    if(!before.alive[p]&&s.alive[p])emit('revive',p);
    if(!before.trapped[p]&&s.trapped[p])emit('control',p);
    if(before.trapped[p]&&!s.trapped[p]&&s.alive[p])emit('control_end',p);
    if(before.bunCarried[p]>=0&&info.died[p]){
      const origin=before.bunCarried[p],loose=s.bunLoose[i*2+origin]>before.bunLoose[i*2+origin];
      emit(loose?'drop':'native_transfer',p,{origin,cell:i});
      if(p===1&&loose){track.drop={tick:t,origin,cell:i,picked:false};}
    }
    if(before.bunCarried[p]<0&&s.bunCarried[p]>=0){
      const origin=s.bunCarried[p],fromLoose=before.bunLoose[i*2+origin]>s.bunLoose[i*2+origin];
      emit(fromLoose?'pickup':'stock_theft',p,{origin,cell:i});
      if(p===0&&track.drop&&track.drop.origin===origin)track.drop.picked=true;
    }
    if(s.bunStored[p][enemy]>before.bunStored[p][enemy])emit('delivery',p,{origin:enemy});
    const oldPhase=P.phase(before,p),newPhase=P.phase(s,p);if(oldPhase!==newPhase)emit('phase',p,{from:oldPhase,to:newPhase});
  }
  if(track.drop&&!track.drop.picked&&t===track.drop.tick+120)emit('enemy_drop_unpicked',0,{drop_tick:track.drop.tick});
  if(before.pos[0]===s.pos[0]&&before.pos[1]===s.pos[1]&&s.alive[0]&&!s.trapped[0])track.idle++;else track.idle=0;
  if(track.idle===100)emit('stagnation',0);
  if(before.bunCarried[0]===1&&s.bunCarried[0]===1){const d=distance(s,0);if(d>track.distance)track.reverse++;else if(d<track.distance)track.reverse=0;}else track.reverse=0;
  if(track.reverse===3)emit('return_deviation',0);
  track.distance=distance(s,0);
  if(before.bunCarried[0]===1&&s.bunCarried[0]===1&&s._bunBaseTeam(...s.centerCell(0))===0&&!s.trapped[0])emit('cage_no_delivery',0);
  if((!before.alive[1]||before.trapped[1])&&s.alive[1]&&!s.trapped[1]&&s.bunCarried[0]<0)emit('window_closed_empty',0);
  return out;
}
async function load() {
  const proof=JSON.parse(fs.readFileSync(path.join(input,'proof.json'))),bytes=fs.readFileSync(path.join(input,'model.json')),doc=JSON.parse(bytes);
  assert.equal(sha(bytes),proof.model_sha256);assert.equal(sha(fs.readFileSync(path.join(input,'model.onnx'))),proof.onnx_sha256);
  const ort=require(process.env.ORT_NODE_MODULE || 'onnxruntime-node');
  const session=await ort.InferenceSession.create(path.join(input,'model.onnx'),{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
  const model=new Q.TransformerModel(doc);model.greedy=true;model.inferEvery=1;
  let parity,lastDecision;
  async function act(s,rng){
    const {obs,state}=model._getLaggedInputs(s,0);
    // Historical H7 re-inference preserves its original input bug; new batches use production encoding.
    if(stage==='h7'){state[9]+=s.trapped[0]/60;state[10]+=s.trapped[1]/60;}
    const result=await session.run({obs:new ort.Tensor('float32',obs,[1,24,13,15]),state:new ort.Tensor('float32',Float32Array.from(state),[1,24])});
    const logits={move:result.move.data,bomb:result.bomb.data};
    assert([...logits.move,...logits.bomb].every(Number.isFinite));
    if(parity===undefined){const ref=model.forward(obs,state);parity=Math.max(...['move','bomb'].flatMap(k=>Array.from(logits[k],(v,i)=>Math.abs(v-ref[k][i]))));assert(parity<.002,`ONNX parity ${parity}`);}
    const mask=s.legalMask(),ability=model._abilityMask(s,mask,0);
    lastDecision={move:Array.from(logits.move),ability:Array.from(logits.bomb),move_mask:Array.from(mask.mm[0]),ability_mask:Array.from(ability)};
    return [model._sampleMasked(logits.move,mask.mm[0],rng),model._sampleMasked(logits.bomb,ability,rng)];
  }
  return {proof,act,session,getParity:()=>parity,getDecision:()=>lastDecision};
}
async function episode(seed,variant,loaded,verify=false,opponent='normal',scene=variant==='open'?'open':'native') {
  const meta={id:`${stage}-g32-${scene}-${opponent==='normal'?'':opponent+'-'}${seed}`,episode_id:`offline-${stage}-g32-${opponent==='tactical'?'tactical-v2':'hunter-'+opponent}-${seed}`,seed,variant,tick_ms:100,max_ticks:scene==='carry_home'?300:2400,
    model:stage==='h8'?'H8 v34 g32 raw':'H7 v33 g32 raw',checkpoint:`${stage.toUpperCase()}/g32/policy.pt`,checkpoint_sha256:loaded.proof.checkpoint_sha256,
    model_sha256:loaded.proof.model_sha256,onnx_sha256:loaded.proof.onnx_sha256,opponent:opponent==='tactical'?'bun.tactical/v2':`bun.hunter/${opponent}`,
    simulator_sha256:sha(fs.readFileSync('web/sim.js')),opponent_sha256:sha(fs.readFileSync(opponent==='tactical'?'web/bun_rule_bot.js':'web/bun_hunter_bot.js')),
    source_hashes:Object.fromEntries(['web/sim.js','web/bun_hunter_bot.js','web/bun_rule_bot.js','web/bot_contract.js','scripts/generate_diagnostic_replays.js'].map(n=>[n,sha(fs.readFileSync(n))])),
    map:'806',map_sha256:await P.hash(level),scene,scene_name:scene==='native'?'原生抢包对战':scene==='open'?'开放抢包对战':'持包返程任务',scene_rules:scene==='native'?'原生砖块、糖泡、道具；240秒':scene==='open'?'清除可破坏砖块；原生糖泡与道具；240秒':'je05 carry_home：清砖、1HP、持敌包、返程出生；30秒',scene_sha256:sha(Buffer.from(JSON.stringify({map:await P.hash(level),scene}))),policy:'greedy',backend:'onnxruntime-node/cpu/fp32',teacher:{eligible:null,query:null,accept:null,reject:null}};
  const s=P.setup(meta,level),hunter=opponent==='tactical'?null:authority.createHunter(opponent),tactical=opponent==='tactical'?new BunRuleTacticalBot():null,rng=Q.mulberry32(seed^0x515154);
  let bunBefore,drops,explosions;
  const updateBun=s._bunUpdate.bind(s),dropBun=s._bunDrop.bind(s),resolve=s._resolveExplosions.bind(s);
  s._bunUpdate=()=>{bunBefore=P.state(s);return updateBun();};
  s._bunDrop=p=>{if(s.bunCarried[p]>=0)drops.push({player:p,origin:s.bunCarried[p],cell:s.centerCell(p)});return dropBun(p);};
  s._resolveExplosions=(...args)=>{const result=resolve(...args);explosions=result.sources.map(x=>({...x,covered:Array.from(x.covered),horzCovered:Array.from(x.horzCovered),vertCovered:Array.from(x.vertCovered)}));return result;};
  hunter?.adapter.reset({schema:'qqt.bot.context/v1',episode_id:meta.episode_id,seed,ruleset:'bun',max_ticks:meta.max_ticks,metadata:{}});
  const d={schema:P.SCHEMA,meta,actions:[],rewards:[],events:[],checks:[[0,await P.hash(P.state(s))]]};
  const trace=verify?null:fs.createWriteStream(path.join(output,`${meta.id}.trace.jsonl`));
  const track={idle:0,reverse:0,distance:distance(s,0)};
  if(trace)trace.write(JSON.stringify({state:P.state(s),legal:s.legalMask()})+'\n');
  while(!s.done){
    drops=[];bunBefore=null;explosions=[];
    const before=P.state(s),a=await loaded.act(s,rng),rawBot=tactical?.act(s,1),bot=rawBot?{move:rawBot[0],ability:rawBot[2]?2:rawBot[1]?1:0}:hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},1,rng);
    const row=[a[0],a[1],bot.move,bot.ability];d.actions.push(row);
    const info=s.step(P.unpack(row));d.events.push(...events(before,s,info,track));
    tactical?.observeTransition(info,stateFromSim(s),1);
    if(s.t%200===0||s.done)d.checks.push([s.t,await P.hash(P.state(s))]);
    if(trace)trace.write(JSON.stringify({state:P.state(s),info,legal:s.legalMask(),actions:row,bun_before:bunBefore,drops,explosions})+'\n');
    if(s.t>meta.max_ticks)throw Error('terminal not reached');
  }
  if(trace)await new Promise(resolve=>trace.end(resolve));
  const failures=['enemy_drop_unpicked','window_closed_empty','return_deviation','cage_no_delivery','self_bomb','stagnation'].filter(k=>d.events.some(e=>e.player===0&&e.kind===k));
  d.summary={ticks:s.t,done:s.done,winner:s.winner,score:s.bunScore.slice(),stored:s.bunStored.map(a=>a.slice()),reward:[0,0],failures};
  meta.replay_identity=await P.hash({meta:{...meta,replay_identity:undefined},actions:d.actions});
  return d;
}
async function run(){
  assert.equal(process.env.CUDA_VISIBLE_DEVICES,'','explicit CPU isolation required');
  fs.mkdirSync(output,{recursive:true});const loaded=await load(),summaries=[];
  const multi=process.argv.includes('--multi-scene');
  const count=multi?6:12;
  for(let index=0;index<count;index++){
    if(multi){
      const scene=['native','open','carry_home'][Math.floor(index/2)],seed=2026101900+index;
      const file=path.join(output,`${stage}-g32-${scene}-${seed}.json`);
      if(fs.existsSync(file)){summaries.push({id:JSON.parse(fs.readFileSync(file)).meta.id,seed,scene});continue;}
      const d=await episode(seed,scene==='open'?'open':'native',loaded,false,'normal',scene);fs.writeFileSync(file,JSON.stringify(d));summaries.push({id:d.meta.id,seed,scene,...d.summary});console.log(JSON.stringify(summaries.at(-1)));continue;
    }
    const supplement=process.argv.includes('--supplement'),seed=(stage==='h8'?(supplement?2026101760:2026101740):(supplement?2026101720:2026101700))+index,variant=supplement?'native':index<6?'native':'open',opponent=supplement?(index<6?'hard':'tactical'):'normal',file=path.join(output,`${stage}-g32-${variant}-${opponent==='normal'?'':opponent+'-'}${seed}.json`);
    if(fs.existsSync(file)){const d=JSON.parse(fs.readFileSync(file));assert.equal(d.meta.checkpoint_sha256,loaded.proof.checkpoint_sha256);summaries.push({id:d.meta.id,seed,variant,...d.summary});continue;}
    const d=await episode(seed,variant,loaded,false,opponent);fs.writeFileSync(file,JSON.stringify(d));summaries.push({id:d.meta.id,seed,variant,...d.summary});console.log(JSON.stringify(summaries.at(-1)));
  }
  fs.writeFileSync(path.join(output,process.argv.includes('--supplement')?'screen-supplement.json':'screen.json'),JSON.stringify({schema:'qqt.diagnostic-screen/v1',checkpoint:loaded.proof.checkpoint_sha256,parity_max_error:loaded.getParity(),episodes:summaries},null,2));
  await loaded.session.release();
}
if(require.main===module)run().catch(e=>{console.error(e);process.exitCode=1;});
module.exports={load,episode,events,distance};
