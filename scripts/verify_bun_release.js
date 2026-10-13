'use strict';
// Compare the actual browser JSON with its native fp32 ONNX graph on both seats.
const fs = require('fs'), assert = require('node:assert/strict'), crypto = require('crypto');
const QQT = require('../web/sim.js');
const authority = require('./eval_web_hunter.js');
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
async function verify(modelFile, graphFile, output) {
  const bytes = fs.readFileSync(modelFile), doc = JSON.parse(bytes);
  assert.equal(doc.meta.rule, 'bun'); assert.equal(doc.meta.embed, 392); assert.equal(doc.meta.depth, 4);
  assert.deepEqual(doc.meta.obs_shape, [24,13,15]); assert.equal(doc.meta.state_dim, 24);
  assert.equal(doc.meta.move_actions, 5); assert.equal(doc.meta.ability_actions, 3);
  assert.ok(!doc.meta.input_encoding && !doc.meta.onnx_path, 'published JSON must be native and portable');
  const ort = require(process.env.QQT_ONNX_RUNTIME || 'onnxruntime-node');
  const session = await ort.InferenceSession.create(graphFile, {executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
  const model = new QQT.TransformerModel(doc), rows = [];
  try {
    for (let i = 0; i < 4; i++) {
      const sim = new QQT.Sim(3923072+i);
      sim.reset(authority.loadLevel(), {nativeItems:true,nativeTrap:true,teams:[0,1]});
      for (let t = 0; t < i*7; t++) sim.step([[i%4,0,0],[(i+1)%4,0,0]]);
      for (let seat = 0; seat < 2; seat++) {
        if(i===1)sim.trapped[1]=30;
        if(i===2){sim.alive[1]=false;sim.bunRespawn[1]=40;}
        if(i===3){sim.bunCarried[0]=1;sim.bunStored[1][1]=0;}
        const obs=sim.encodeObsJAX(seat,24), state=Float32Array.from(sim.encodeStateJAX(seat,doc.meta.bun_state_encoding));
        assert.equal(obs.length,24*13*15); assert.equal(state.length,24);
        const expected=model.forward(obs,state);
        const actual=await session.run({obs:new ort.Tensor('float32',obs,[1,24,13,15]),state:new ort.Tensor('float32',state,[1,24])});
        const errors={};
        for (const k of ['move','bomb']) {
          errors[k]=Math.max(...Array.from(actual[k].data,(x,j)=>Math.abs(x-expected[k][j])));
          assert.ok(errors[k]<0.002 && Array.from(actual[k].data).every(Number.isFinite));
        }
        errors.value=Math.abs(actual.value.data[0]-expected.value);
        assert.ok(errors.value<0.002 && Number.isFinite(actual.value.data[0]));
        rows.push({sample:i,seat,max_errors:errors,obs:Array.from(obs),state:Array.from(state),
          js:{move:Array.from(expected.move),bomb:Array.from(expected.bomb),value:expected.value},
          onnx:{move:Array.from(actual.move.data),bomb:Array.from(actual.bomb.data),value:actual.value.data[0]},
          masks:{move:sim.legalMask().mm[seat],bomb:model._abilityMask(sim,sim.legalMask(),seat)}});
      }
    }
  } finally { await session.release(); }
  const result={passed:true,native_bun:true,cuda_visible_devices:process.env.CUDA_VISIBLE_DEVICES,
    model_sha256:sha(bytes),onnx_sha256:sha(fs.readFileSync(graphFile)),samples:rows};
  fs.writeFileSync(output+'.tmp',JSON.stringify(result,null,2)+'\n');fs.renameSync(output+'.tmp',output);
  console.log(JSON.stringify(result));return result;
}
if(require.main===module) verify(...process.argv.slice(2)).catch(e=>{console.error(e.message);process.exitCode=1;});
module.exports={verify};
