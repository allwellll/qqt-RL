'use strict';
// All frozen evaluation physics are delegated unchanged; only seed identity differs.
const fs=require('fs'),assert=require('node:assert/strict');
const namespace=process.env.JE09_NAMESPACE||'runs/je09_bun_20261015/stage_e_long_return_coverage_pfsp8_v10';
process.env.JE09_NAMESPACE=namespace;
const fixed=require('./je09_web_stage_d.js'),short=require('./je09_web_stage_b.js'),stageA=require('./je09_web_stage_a.js');
const dec=JSON.parse(fs.readFileSync(namespace+'/ROUND_DECLARATION.json'));
short.manifest.seed=dec.seeds.stage_b_main;short.manifest.holdout_seed=dec.seeds.stage_b_holdout;short.manifest.experiment_id=dec.experiment_id;
stageA.manifest.seed=dec.seeds.stage_a_main;stageA.manifest.holdout_seed=dec.seeds.stage_a_holdout;
async function run(model,out,mode){
 if(mode.startsWith('stage_a_'))return stageA.run(model,out,mode.endsWith('holdout')?'holdout':'evaluation');
 if(mode.startsWith('stage_b_'))return short.run(model,out,mode.endsWith('holdout')?'holdout':'evaluation');
 return fixed.evaluate(model,out,mode);
}
if(require.main===module)run(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={run};
