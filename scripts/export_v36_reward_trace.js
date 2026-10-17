'use strict';
// Private full states are emitted only below ignored runs/, never into web/.
const fs=require('node:fs'),assert=require('node:assert/strict');
const P=require('../web/review_protocol'),G=require('./generate_v36_multiseed_review');
const {instrument}=require('./diagnostic_counterfactual');
(async()=>{
  fs.mkdirSync('runs/replay_review_20261017/reward-traces',{recursive:true});
  for(const m of G.MATRIX){
    const id=G.idFor(m),d=JSON.parse(fs.readFileSync('web/review-replays/'+id+'.json'));
    const player=new P.Player(d,require('./eval_web_hunter').loadLevel());await player.verify();
    const write=fs.createWriteStream('runs/replay_review_20261017/reward-traces/'+id+'.jsonl');
    write.write(JSON.stringify({state:P.state(player.sim),legal:player.sim.legalMask()})+'\n');
    const step=instrument(player.sim);
    for(let i=0;i<d.actions.length;i++){
      const entry=step(d.actions[i]);assert.equal(await P.hash(entry.state),d.checks[i+1][1]);
      if(!write.write(JSON.stringify(entry)+'\n'))await new Promise(r=>write.once('drain',r));
    }
    await new Promise(r=>write.end(r));console.log('TRACE',m.seed,d.actions.length);
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
