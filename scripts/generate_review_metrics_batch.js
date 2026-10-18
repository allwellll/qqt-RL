'use strict';
// Frozen seeds, no outcome filtering. All computation uses the CPU Actor.
const fs=require('fs'),P=require('../web/review_protocol'),G=require('./generate_diagnostic_replays');
const MATRIX=[{seed:2026101810,scene:'native'},{seed:2026101811,scene:'native'},{seed:2026101812,scene:'open'},{seed:2026101813,scene:'open'},{seed:2026101814,scene:'carry_home'},{seed:2026101815,scene:'carry_home'}],SEEDS=MATRIX.map(r=>r.seed);
async function main(){const model=await G.load(),out=process.env.REPLAY_EPISODES;
 for(const {seed,scene} of MATRIX){if(process.argv.includes('--extension')&&scene==='native')continue;const d=await G.episode(seed,scene==='open'?'open':'native',model,false,'normal',scene);
  Object.assign(d.meta,{id:`je10-g8-metrics-${scene==='native'?'natural':scene}-${seed}`,episode_id:`offline-je10-g8-${scene}-${seed}`,model:'Bun392 JE10 v38 r01 g8',checkpoint:'JE10/v38/r01/g8/policy.pt',batch_id:'je10-critic-metrics-20261018',task:scene==='carry_home'?'carry_home':scene==='open'?'open_full2400':'natural_full2400',diagnostic:true,training_reward_status:'unrecorded',scene_name:scene==='native'?'原生自然完整对局':scene==='open'?'开放布局完整对局':'预置持包返程诊断',scene_rules:scene==='native'?'生产原生806砖块、糖泡、道具；2400 tick上限；真实Hunter normal；无预置携包/控制/掉包；assist-off':scene==='open'?'生产806清砖变体；原生糖泡/道具；2400 tick；完整对局；无预置携包；assist-off':'生产carry_home初始化；1HP/清砖/预置持敌包；300 tick；诊断场景，不代表自然完整对局成功；assist-off',recording_origin:'best_model_replacement',selection_rule:'predeclared seeds 2026101810..2026101815; all retained; no outcome selection'});
  d.meta.replay_identity=await P.hash(P.identity(d));fs.writeFileSync(`${out}/${d.meta.id}.json`,JSON.stringify(d));console.log(JSON.stringify({id:d.meta.id,seed,...d.summary}));
 }
 await model.session.release();}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
module.exports={SEEDS,MATRIX};
