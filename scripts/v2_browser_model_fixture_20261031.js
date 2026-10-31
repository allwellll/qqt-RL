'use strict';
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const B='runs/v2_fresh_selection_20261031',W=require('node:path').resolve('../cx_publish_r1_1805_20261030'),Q=require(W+'/web/sim-bun-native-v2'),I=require(W+'/web/r3_model_inputs');
const m=JSON.parse(fs.readFileSync(B+'/MODELS.json')).r3_a_latest,doc=JSON.parse(fs.readFileSync(m.model_json));
const ort=require('../runs/je03_original_nonbun_20261011/baseline_v1/deps/node_modules/onnxruntime-node');
(async()=>{
 const session=await ort.InferenceSession.create(m.onnx_path,{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
 const infer=async i=>{const r=await session.run({obs:new ort.Tensor('float32',i.obs,[1,25,13,15]),state:new ort.Tensor('float32',i.state,[1,37])});return {move:Array.from(r.move.data),bomb:Array.from(r.bomb.data),value:r.value.data[0]};};
 const zero={obs:new Float32Array(25*195),state:new Float32Array(37)},reference=await infer(zero),fixtures=[];
 const trace=JSON.parse(fs.readFileSync(B+'/shards/hunter_hard/r3_a_latest/fixed-natural_full2400-2026314000.trace.json'));
 const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806),spawn=JSON.parse(fs.readFileSync(B+'/hard_spawns.json')).spawn_cells[0];
 I.installSim(Q);const s=new Q.Sim(2026314000);s.reset(level,{nativeItems:true,nativeTrap:true});for(let p=0;p<2;p++){s.pos[2*p]=spawn[p][0]+.5;s.pos[2*p+1]=spawn[p][1]+.5;}s.bunSpawnPos=spawn.map(a=>a.map(x=>x+.5));s._gen++;
 for(let t=0;t<=240&&!s.done;t++){
  if(t%120===0){const i=I.inputs(s,1);fixtures.push({tick:t,seat:1,obs:Array.from(i.obs),state:Array.from(i.state),expected:await infer(i)});}
  const a=trace.actions[t];if(a)s.step([[a[0],Number(a[1]===1),Number(a[1]===2)],[a[2],Number(a[3]===1),Number(a[3]===2)]]);
 }
 const meta={...doc.meta};delete meta.onnx_path;
 const row={id:'r3-compatibility-fixture',file:'r3.json',bytes:fs.statSync(m.model_json).size,sha256:m.model_sha256,
 onnx:{file:'r3.onnx',sha256:m.onnx_sha256,bytes:fs.statSync(m.onnx_path).size,meta,reference_zero:reference}};
 fs.writeFileSync(B+'/R3_BROWSER_FIXTURES.json',JSON.stringify({row,fixtures,files:{'r3.json':m.model_json,'r3.onnx':m.onnx_path}}));
 console.log('R3 P1 browser fixtures ready',fixtures.length);
})().catch(e=>{console.error(e);process.exitCode=1});
