'use strict';
// Historical compatibility fixtures remain verified independently of public admission.
// The current catalog is exhaustively gated by test_review_admission.js.
const assert=require('node:assert/strict'),fs=require('fs'),crypto=require('crypto');
const P=require('./review_protocol');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const catalog=require('../reports/review_catalog_before_gate_20261018.json'),model=require('./models.json').models.find(row=>row.id==='bun392-je10-v38-r01-g8');
const tasks=['real_drop','native_death_stock','native_sugar_stock','natural_full2400'];
async function main(){
  assert.equal(model.id,'bun392-je10-v38-r01-g8','historical JE10 must stay selectable');
  assert.equal(catalog.replays[0].batch_id,'je10-best-completed-20261018');
  for(const [i,task] of tasks.entries()){
    const row=catalog.replays[i],bytes=fs.readFileSync('web/review-replays/'+row.file),d=JSON.parse(bytes);
    assert.equal(row.task,task);assert.equal(row.seed,2026101800);assert.equal(sha(bytes),row.sha256);assert.equal(bytes.length,row.bytes);
    for(const key of ['id','file','scene','scene_name','scene_rules','model','opponent','checkpoint_sha256','map_sha256','summary'])assert(row[key]!==undefined,key);
    assert.equal(row.checkpoint_sha256,model.source_sha256);assert.equal(d.meta.model_sha256,model.sha256);assert.equal(d.meta.onnx_sha256,model.onnx.sha256);
    assert.equal(d.meta.checkpoint_sha256,model.source_sha256);P.validate(d);
    assert.equal(d.diagnostic_ticks.length,d.actions.length);assert.equal(d.checks.length,d.actions.length+1);
    const a=d.meta.approximate_a1;assert.equal(a.status,'offline_approximation');assert.equal(a.future_value,0);assert(a.definition.includes('sum(pi'));
    for(const [j,t] of d.diagnostic_ticks.entries()){
      assert.equal(t.tick,j+1);assert.equal(t.raw_a1,null);assert.equal(t.shaped_a1,null);
      for(const key of ['raw_a1','shaped_a1','delta_a'])assert(Number.isFinite(t.approximate_a1[key]));
      assert(Math.abs(t.approximate_a1.delta_a-(t.approximate_a1.shaped_a1-t.approximate_a1.raw_a1))<1e-10);
      for(const key of ['raw_a1','shaped_a1','delta_a'])assert.equal(t.approximate_a1[key+'_increment'],j?t.approximate_a1[key]-d.diagnostic_ticks[j-1].approximate_a1[key]:null);
      const c=t.counterfactuals,mass=c.reduce((n,v)=>n+v.probability,0);assert(Math.abs(mass-1)<1e-6);
      const taken=c.find(v=>v.action[0]===d.actions[j][0]&&v.action[1]===d.actions[j][1]);assert(taken);
      for(const role of ['raw','shaped'])assert(Math.abs(t.approximate_a1[role+'_a1']-(taken[role+'_reward']-c.reduce((n,v)=>n+v.probability*v[role+'_reward'],0)/mass))<1e-10);
    }
    for(const mutate of [x=>x.diagnostic_ticks[0].approximate_a1.raw_a1=100,x=>x.diagnostic_ticks[0].approximate_a1.shaped_q1_mean=NaN,x=>x.diagnostic_ticks[1].approximate_a1.delta_a_increment=9,x=>x.diagnostic_ticks[0].counterfactuals.pop(),x=>x.diagnostic_ticks[0].raw_a1=1,x=>x.meta.approximate_a1.future_value=1,x=>x.diagnostic_ticks[0].counterfactuals[0].probability+=.1]){const bad=structuredClone(d);mutate(bad);assert.throws(()=>P.validate(bad),/approximate A1/);}
    const player=new P.Player(d,require('../scripts/eval_web_hunter').loadLevel());await player.verify();
    if(d.meta.native_fixture){const bad=structuredClone(d);bad.meta.native_fixture.frame.pos[0]+=.1;await assert.rejects(new P.Player(bad,player.level).verify(),/fixture frame hash/);}
    for(let t=1;t<=d.actions.length;t++){player.seek(t-1);assert.equal(await P.hash(P.state(player.sim)),d.diagnostic_ticks[t-1].state_before_sha256);player.step();assert.equal(await P.hash(P.state(player.sim)),d.diagnostic_ticks[t-1].state_after_sha256);}
  }
  const ui=fs.readFileSync('web/review.js','utf8');for(const k of ['raw A1','shaped A1','ΔA','raw_a1_increment','shaped_a1_increment','delta_a_increment'])assert(ui.includes(k),'Review visible '+k);
  console.log('JE10 model/replay identity, all-action replay, policy-centered approximate A1 contract GREEN');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
