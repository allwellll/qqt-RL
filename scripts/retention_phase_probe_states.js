'use strict';
// Existing immutable Review chain only. No new episodes or favourable seeds.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const P=require('../web/review_protocol.js'),authority=require('./eval_web_hunter.js');
const phase=require('../web/retention_phase_conditioning.js');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
async function extract(folder){
  const tasks=['native_death_stock','native_sugar_stock','real_drop','carry_near','carry_mid','carry_far','recovery_threat','natural_full2400'];
  const rows=[],sources=[];
  for(const task of tasks){
    const file=path.join(folder,`REVIEW_FIXED_${task}.json`),bytes=fs.readFileSync(file),doc=JSON.parse(bytes);
    const player=new P.Player(doc,authority.loadLevel());await player.verify();
    const team=doc.meta.learner_team ?? doc.meta.player ?? 0;
    sources.push({task,sha256:hash(bytes),seed:doc.meta.seed,artificial_carry:task.startsWith('carry_')});
    while(player.sim.t<doc.actions.length){
      const sim=player.sim;
      if(sim.t%20===0){
        const obs=Array.from(sim.encodeObsJAX(team,24));
        const gv=Array.from(sim.encodeStateJAX(team,'bun24-trap-timer-v2'));
        rows.push({obs,gv,phase:phase.features(obs,gv).indexOf(1),task,tick:sim.t,team,artificial_carry:task.startsWith('carry_')});
      }
      player.step();
    }
  }
  return {schema:'phase5-fixed-review-probe/v1',rows,sources,controls_training:false,diagnostic_only:true};
}
module.exports={extract};
if(require.main===module)extract(process.argv[2]).then(result=>fs.writeFileSync(process.argv[3],JSON.stringify(result))).catch(error=>{console.error(error);process.exit(1)});
