'use strict';
// Independent all-branch verification: consume the published probabilities, never regenerate episodes.
const fs=require('fs'),crypto=require('crypto'),assert=require('node:assert/strict');
const P=require('../web/review_protocol'),Q=require('../web/sim'),M=require('./je10_review_math');
const level=require('./eval_web_hunter').loadLevel();
const sha=x=>crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');
const rows=require('../web/review-catalog.json').replays.filter(r=>r.batch_id==='je10-best-completed-20261018');
(async()=>{
 let count=0,actions=0;
 for(const row of rows){
  const d=JSON.parse(fs.readFileSync('web/review-replays/'+row.file)),p=new P.Player(d,level),tracker=new M.Tracker(0);if(row.task==='real_drop')tracker.trapCause[1][0]=true;
  let phi=0,offset=0,identity=null;
  for(const t of d.diagnostic_ticks){
   const b=P.state(p.sim),prior=M.snapshot(p.sim),obj=M.objective(p.sim,0,tracker),taken=d.actions[p.sim.t];let actual;
   assert.equal(sha(b),t.state_before_sha256);
   for(const c of t.counterfactuals){
    const fresh=P.replaySetup(d.meta,level);for(let i=0;i<p.draws;i++)fresh.rng();
    const random=fresh.rng;Object.assign(fresh,structuredClone(Object.fromEntries(Object.entries(p.sim).filter(([,v])=>typeof v!=='function'))));fresh.rng=random;
    const tr=Object.assign(Object.create(M.Tracker.prototype),structuredClone(tracker));const info=fresh.step(P.unpack([...c.action,...taken.slice(2)]));tr.observe(fresh,prior,info);
    assert.equal(sha(P.state(fresh)),c.next_state_sha256);assert.equal(fresh.done,c.terminal);
    const capture=fresh.bunStored[0][1]>b.bunStored[0][1]&&!info.died[0]&&!info.mutualDeath;
    const raw=24*Number(capture)-12*Number(info.died[0])-16*Number(!!info.ownBombDefeat?.[0])-12*Number(!!info.mutualDeath)-.5*Number(info.died[0]&&b.bunCarried[0]>=0);
    const entry=b.trapped[1]===0&&fresh.trapped[1]>0&&info.causalDamageSource?.[1]?.[0]&&!info.causalDamageSource?.[1]?.[1];
    let off=offset,pb=phi,pa=phi,key=identity,route=0;
    if(obj){key=JSON.stringify(obj.target);if(identity!==key)off=phi-obj.phi;pb=obj.phi+off;const next=M.objective(fresh,0,tr);pa=fresh.done?0:next&&JSON.stringify(next.target)===key?next.phi+off:pb;route=.5*(.995*pa-pb);}
    assert.equal(raw,c.raw_reward);assert.equal(raw+4*Number(!!entry)+route,c.shaped_reward);
    if(c.action[0]===taken[0]&&c.action[1]===taken[1])actual={off,pb,pa,key};count++;
   }
   assert.equal(actual.pb,t.route_diagnostic.phi_before);assert.equal(actual.pa,t.route_diagnostic.phi_after);
   phi=actual.pa;offset=actual.off;identity=actual.key;p.step();tracker.observe(p.sim,prior,p.info);assert.equal(sha(P.state(p.sim)),t.state_after_sha256);actions++;
  }
  console.log(row.task,'independent every-branch state/reward/route GREEN');
 }
 console.log(JSON.stringify({actions,counterfactual_branches:count,passed:true}));
})().catch(e=>{console.error(e);process.exitCode=1;});
