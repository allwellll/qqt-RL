'use strict';
const fs=require('fs');
const path=require('path');
const Module=require('module');
const crypto=require('crypto');
const assert=require('node:assert/strict');
const QQT=require('../web/sim.js');
const root=path.resolve(__dirname,'..');
const authorityFile=path.join(__dirname,'eval_web_hunter.js');
const residualFile=path.join(__dirname,'eval_web_hunter_residual.js');
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');

// Only await inference. Physical steps, Bot calls, seeds and outcome attribution
// remain the exact frozen authority; asynchronous inference does not advance time.
function asynchronousAuthority(){
  let code=fs.readFileSync(authorityFile,'utf8');
  assert.equal((code.match(/const raw = model.act\(sim, 0, rng\);/g)||[]).length,1);
  code=code.replace('function runEpisode(', 'async function runEpisode(')
    .replace('const raw = model.act(sim, 0, rng);','const raw = await model.act(sim, 0, rng);')
    .replace('const tracer = attackTrace ? new AttackTracer(sim) : null;',
      'const tracer = new AttackTracer(sim); const funnel={valid:0,approach:0,contact:0,legal_attack:0,request_bomb:0,placed_bomb:0,kill:0};')
    .replace('const tick = sim.t;', 'const tick = sim.t; const oldDistance=Math.abs(sim.pos[0]-sim.pos[2])+Math.abs(sim.pos[1]-sim.pos[3]); const legal=sim.legalMask(); funnel.valid+=Number(sim.alive[0]); funnel.contact+=Number(sim.alive[0]&&sim.alive[1]&&oldDistance<=3); funnel.legal_attack+=Number(sim.alive[0]&&legal.bm[0][1]);')
    .replace('counts.bombs += info.placed[0] ? 1 : 0;', 'counts.bombs += info.placed[0] ? 1 : 0; funnel.request_bomb+=Number(raw[1]===1); funnel.placed_bomb+=Number(info.placed[0]); funnel.approach+=Number(Math.abs(sim.pos[0]-sim.pos[2])+Math.abs(sim.pos[1]-sim.pos[3])<oldDistance); funnel.kill+=Number(info.causalKill[0]||info.creditedKill[0]);')
    .replace('return record;', 'record.funnel=funnel; return record;');
  const m=new Module(authorityFile,module);m.filename=authorityFile;m.paths=module.paths;
  m._compile(code,authorityFile);return m.exports;
}

async function loadModel(file){
  const bytes=fs.readFileSync(file),doc=JSON.parse(bytes);
  let model;
  if(doc.meta.arch==='transformer'){
    model=new QQT.TransformerModel(doc);
    assert.deepEqual(doc.meta.obs_shape,[24,13,15]);
    assert.ok(!doc.meta.input_encoding,'native Bun only');
    const ort=require(path.join(root,'runs/je03_original_nonbun_20261011/baseline_v1/deps/node_modules/onnxruntime-node'));
    const onnx=doc.meta.onnx_path;assert.ok(onnx,'explicit native ONNX required');
    const session=await ort.InferenceSession.create(onnx,{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
    const reference=model.forward.bind(model);let verified=false;
    model.forward=async(obs,state)=>{
      const r=await session.run({obs:new ort.Tensor('float32',obs,[1,...model.obsShape]),
        state:new ort.Tensor('float32',Float32Array.from(state),[1,state.length])});
      const result={move:r.move.data,bomb:r.bomb.data,value:r.value.data[0]};
      if(!verified){
        const expected=reference(obs,state);
        const diffs=['move','bomb'].flatMap(k=>Array.from(result[k],(x,i)=>Math.abs(x-expected[k][i])));
        model.forwardMaxError=Math.max(...diffs);assert.ok(model.forwardMaxError<0.002,`ONNX parity ${model.forwardMaxError}`);
        verified=true;
      }
      return result;
    };
    model.act=async(sim,pid,rng)=>{
      const input=model._getLaggedInputs(sim,pid),mask=sim.legalMask();
      if(doc.meta.bun_state_encoding==='bun24-trap-timer-v2'){input.state=require('./je05_bun_encoding.js').encodeState(sim,pid);}
      const logits=await model.forward(input.obs,input.state);
      assert.ok([...logits.move,...logits.bomb].every(Number.isFinite));
      return model._decodeAbility(model._sampleMasked(logits.move,mask.mm[pid],rng),
        model._sampleMasked(logits.bomb,model._abilityMask(sim,mask,pid),rng));
    };
    model.backendSha=hash(fs.readFileSync(onnx));
  }else if(doc.meta.arch==='mlp'){
    model=new QQT.MLPModel(doc);
    const act=model.act.bind(model);
    model.act=(sim,pid,rng)=>{
      // Native legacy encoder supports 13 columns. This is an observation-only
      // transfer adapter; no walls, spawn coordinates or simulation are changed.
      const old=sim.oldMode;sim.oldMode=model.obsShape[2]===13;
      try{return act(sim,pid,rng);}finally{sim.oldMode=old;}
    };
  }else throw Error(`unsupported native model ${doc.meta.arch}`);
  model.greedy=true;model.inferEvery=1;
  return {model,sha:hash(bytes),meta:doc.meta};
}

async function run(options){
  const authority=asynchronousAuthority();authority.loadModel=loadModel;
  let code=fs.readFileSync(residualFile,'utf8');
  code=code.replace('function settle(', 'async function settle(')
    .replace('const raw = model.act(sim, 0, rng);','const raw = await model.act(sim, 0, rng);')
    .replace('function runShard(', 'async function runShard(')
    .replace('const loaded = authority.loadModel(', 'const loaded = await authority.loadModel(')
    .replace('const record = authority.runEpisode(', 'const record = await authority.runEpisode(')
    .replace('const residual = settle(', 'const residual = await settle(')
    .replace('wrapper.act = (current, player, random) => {','wrapper.act = async (current, player, random) => {');
  const m=new Module(residualFile,module);m.filename=residualFile;m.paths=module.paths;
  const normalRequire=m.require.bind(m);
  m.require=name=>name==='./eval_web_hunter.js'?authority:normalRequire(name);
  m._compile(code,residualFile);
  const result=await m.exports.runShard(options);
  result.inference_protocol='original_architecture_cpu_v1'; result.passive_funnel=true;
  result.original_source_hashes={primary:hash(fs.readFileSync(authorityFile)),residual:hash(fs.readFileSync(residualFile)),adapter:hash(fs.readFileSync(__filename))};
  fs.writeFileSync(options.out+'.tmp',JSON.stringify(result));fs.renameSync(options.out+'.tmp',options.out);
  console.log(`${path.basename(options.modelPath)} ${options.difficulty} [${options.start},${options.end}) ${result.elapsed_seconds.toFixed(3)}s`);
}

if(require.main===module)run(require('./eval_web_hunter.js').parseArgs(process.argv.slice(2))).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={asynchronousAuthority,loadModel,run};
