'use strict';
// Genuine fixed first-game states, both seats; no synthetic carry fixtures.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const B=process.env.V2_SELECTION_OUTPUT||'runs/v2_fresh_selection_20261031',O=B+'/publication',W=path.resolve('../cx_publish_r1_1805_20261030'),Q=require(W+'/web/sim-bun-native-v2'),I=require(W+'/web/r3_model_inputs');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
(async()=>{
 const row=JSON.parse(fs.readFileSync(O+'/CATALOG_ROW.json')),selection=JSON.parse(fs.readFileSync(B+'/RESULTS.json')).selection,label=selection.winner;
 assert(selection.eligible_for_conditional_publish);const raw=fs.readFileSync(O+'/'+row.file),doc=JSON.parse(raw);assert.equal(sha(raw),row.sha256);assert(!doc.meta.onnx_path);
 const ort=require('../runs/je03_original_nonbun_20261011/baseline_v1/deps/node_modules/onnxruntime-node'),graph=O+'/'+row.onnx.file;
 assert.equal(sha(fs.readFileSync(graph)),row.onnx.sha256);const session=await ort.InferenceSession.create(graph,{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
 const model=I.attach(new Q.TransformerModel(doc));model.greedy=true;I.installSim(Q);
 const infer=async(obs,state)=>{const r=await session.run({obs:new ort.Tensor('float32',obs,[1,...doc.meta.obs_shape]),state:new ort.Tensor('float32',state,[1,doc.meta.state_dim])});return {move:Array.from(r.move.data),bomb:Array.from(r.bomb.data),value:r.value.data[0]};};
 const zero={obs:new Float32Array(doc.meta.obs_shape.reduce((a,b)=>a*b,1)),state:new Float32Array(doc.meta.state_dim)},rzero=model.forward(zero.obs,zero.state),ozero=await infer(zero.obs,zero.state);
 const plain=r=>({move:Array.from(r.move),bomb:Array.from(r.bomb),value:r.value});const reference_zero=plain(rzero);for(const k of ['move','bomb'])assert(Math.max(...ozero[k].map((v,i)=>Math.abs(v-rzero[k][i])))<.002);assert(Math.abs(ozero.value-rzero.value)<.002);
 const seed=JSON.parse(fs.readFileSync(B+'/PROTOCOL.json')).conditions.hunter_hard.seed,tracePath=`${B}/shards/hunter_hard/${label}/fixed-natural_full2400-${seed}.trace.json`,trace=JSON.parse(fs.readFileSync(tracePath));
 const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806),spawn=JSON.parse(fs.readFileSync(B+'/hard_spawns.json')).spawn_cells[0],s=new Q.Sim(seed);s.reset(level,{nativeItems:true,nativeTrap:true});for(let p=0;p<2;p++){s.pos[2*p]=spawn[p][0]+.5;s.pos[2*p+1]=spawn[p][1]+.5;}s.bunSpawnPos=spawn.map(a=>a.map(x=>x+.5));s._gen++;
 const samples=[];
 for(let t=0;t<=360&&!s.done;t++){
  if([0,40,120,240,360].includes(t))for(const seat of [0,1]){
   const {obs,state}=model._getLaggedInputs(s,seat),expected=model.forward(obs,state),actual=await infer(obs,Float32Array.from(state));
   const masks=s.legalMask(),legal={move:masks.mm[seat],bomb:model._abilityMask(s,masks,seat)};
   const argmax=(a,m)=>Array.from(a).reduce((best,v,i)=>m[i]&&(best<0||v>a[best])?i:best,-1),errors={};
   for(const k of ['move','bomb']){errors[k]=Math.max(...actual[k].map((v,i)=>Math.abs(v-expected[k][i])));assert(errors[k]<.002);assert.equal(argmax(actual[k],legal[k]),argmax(expected[k],legal[k]));}
   errors.value=Math.abs(actual.value-expected.value);assert(errors.value<.002);
   samples.push({sample:t,seat,obs:Array.from(obs),state:Array.from(state),masks:legal,js:plain(expected),onnx:actual,errors,masked_greedy_exact:true});
  }
  const a=trace.actions[t];if(a)s.step([[a[0],Number(a[1]===1),Number(a[1]===2)],[a[2],Number(a[3]===1),Number(a[3]===2)]]);
 }
 await session.release();fs.writeFileSync(O+'/JS_ONNX.json',JSON.stringify({passed:true,model_sha256:row.sha256,onnx_sha256:row.onnx.sha256,reference_zero,samples,scope:'genuine fixed first-game states, both seats, fp32; no training-bf16 identity claim'}));
 row.onnx.reference_zero=reference_zero;fs.writeFileSync(O+'/CATALOG_ROW.json',JSON.stringify(row,null,2)+'\n');console.log('frozen winner JS/ONNX both-seat states PASS',samples.length);
})().catch(e=>{console.error(e);process.exit(1)});
