'use strict';
const assert = require('assert');
const QQT = require('./sim'), Coop = require('./bun_coop_hunter_bot');
function scene(seed = 2026100801) {
  const sim = new QQT.Sim(seed);
  sim.reset('open', { nativeItems: true, nativeTrap: true, teams: [0, 1] });
  for (const key of ['wall', 'brick', 'crate', 'fuse', 'blastLinger']) sim[key].fill(0);
  sim.isBun = true; sim.bunStored = [[0, 0], [0, 0]]; sim.bunInitial = [Infinity, Infinity];
  sim.maxSteps = 200; sim.invuln.fill(0); sim.spdG.fill(1); sim.blastCap.fill(3); sim.bombsCap.fill(1);
  // Foe crosses the future firing lane then stops at the far end of the corridor.
  sim.pos.set([9.5, 10.5, 6.5, 8.5]);
  sim.wall[5 * QQT.W + 10] = 1;
  return sim;
}
function observed(sim, b) {
  b.analyzeSim(sim, 1);
  sim.step(Array.from({length:sim.nPlayers},(_,p)=>[p===0?0:4,0,0,0]));
  return b.analyzeSim(sim, 1);
}
const make = () => new Coop.BunCoopHunterBot({ difficulty: 'hard', seed: 2026100801 });
module.exports = { scene, observed, make };
if (require.main === module) {
  const sim = scene(), b = make(), d = observed(sim, b);
  assert.equal(d.reason, 'bomb_direct', 'RED: attack the arriving player while it is still off the firing ray');
  assert(d.directThreat, 'RED: active attack must carry a timed target/body/path proof, not generic direction');
  assert.equal(d.directThreat.target, 0);
  assert(d.directThreat.targetPath.length > 6, 'predict to actual fuse, beyond old six-tick route');
  assert.equal(d.directThreat.guaranteed, false, 'a constant observed path is conditional, not all-action reachability');
  assert.equal(d.directThreat.expectedExplosionTick, 32);
  assert.equal(d.directThreat.predictedHitTick, 32);
  const openState=b.decisionState, openGeometry=b.geometry(openState);
  b.lastGoalMode='OPEN_ROUTE';
  assert(b.directBomb(openState,openGeometry,1,new Float64Array(openGeometry.N)),
    'RED: a proven safe direct threat outranks untargeted wall opening');
  const proofFor = (sim, moving = false) => {
    const bot=make();bot.analyzeSim(sim,1);const st=bot.decisionState,g=bot.geometry(st),cell=st.players[1].cell;
    if(moving)bot.enemyTrends.set(0,{confidence:true,vy:-0.3,vx:0});
    return bot.directThreat(st,g,1,cell,bot.predict(st,g,[],()=>true),bot.predict(st,g,[{cell,e:31,blast:st.players[1].blast}],()=>true));
  };
  const departingItem=scene();departingItem.pos[0]=9.05;
  departingItem.fieldItem[9*QQT.W+10]=QQT.ITEM_BANANA;
  departingItem.fieldArmed[9*QQT.W+10]=1;
  assert.equal(proofFor(departingItem,true),null,'RED: field contact at movement start invalidates the path even when ending in another tile');
  for(const boundary of ['hard-wall','soft-brick','range','invulnerable','half','banana','glue-expiry','field-item','early-chain']){
    const s=scene();s.pos.set([6.5,10.5,6.5,8.5]);
    if(boundary==='hard-wall')s.wall[6*QQT.W+9]=1;
    if(boundary==='soft-brick')s.brick[6*QQT.W+9]=1;
    if(boundary==='range')s.blastCap[1]=1;
    if(boundary==='invulnerable')s.invuln[0]=40;
    if(boundary==='half')s.pos[0]=6.95;
    if(boundary==='banana'){s.movementStatus[0]=QQT.MOVE_STATUS_SLIDE;s.slideDir[0]=3;}
    if(boundary==='glue-expiry'){s.movementStatus[0]=QQT.MOVE_STATUS_SLOW;s.movementStatusTicks[0]=2;}
    if(boundary==='field-item'){s.fieldItem[6*QQT.W+10]=QQT.ITEM_BANANA;s.fieldArmed[6*QQT.W+10]=1;}
    if(boundary==='early-chain'){
      const cell=6*QQT.W+7;s.fuse[cell]=5;s.owner[cell]=0;s.bombBlast[cell]=3;s.invuln[0]=6;
    }
    assert.equal(proofFor(s),null,`${boundary}: no false timed hit claim`);
  }
  const slow=scene();slow.pos.set([6.5,10.5,6.5,8.5]);slow.movementStatus[0]=QQT.MOVE_STATUS_SLOW;slow.movementStatusTicks[0]=40;
  assert(proofFor(slow),'known long-lived slow state can be forecast');
  const crossed=scene();crossed.pos.set([6.95,10.5,6.5,8.5]);
  const other=7*QQT.W+12;crossed.fuse[other]=31;crossed.owner[other]=0;crossed.bombBlast[other]=3;
  assert.equal(proofFor(crossed),null,
    'a different source completing half-body union must not be called this bubble alone hitting the target');
  // The existing same-line kill remains higher-value than opening a wall. The
  // target really tries to leave through the mouth now occupied by the bubble.
  const enclosed=scene(), killer=make();enclosed.pos.set([6.5,10.5,6.5,8.5]);
  for(const [y,x] of [[5,9],[7,9],[5,10],[7,10],[6,11]])enclosed.wall[y*QQT.W+x]=1;
  const kill=killer.analyzeSim(enclosed,1);
  assert.equal(kill.reason,'bomb_kill');assert(kill.directThreat);
  assert.deepEqual(kill.directThreat.safeConstantMoves,[]);
  const {trace}=require('../scripts/eval_staggered_attack');
  const trapped=trace(enclosed,[null,killer],1,(s,bs,ds)=>{
    const a=killer.act(s,1);ds.set(1,killer.lastDecision);return [[2,0,0,0],a];
  },80);
  assert.equal(trapped.stats.enemyTraps,1);assert.equal(trapped.stats.selfTraps,0);
  // A real early chain with no immunity reports its actual earlier hit tick.
  const early=scene();early.pos.set([6.5,10.5,6.5,8.5]);
  const anchor=6*QQT.W+6;early.fuse[anchor]=5;early.owner[anchor]=0;early.bombBlast[anchor]=2;
  const chainProof=proofFor(early);assert(chainProof);assert.equal(chainProof.predictedHitTick,5);
  assert.equal(chainProof.expectedExplosionTick,5);
  for(const boundary of ['self-corner','stationary-ally','rescue','carrier','enemy-counterbubble']){
    const s=scene(), bot=make();
    if(boundary==='self-corner'){
      s.wall.fill(1);for(const [y,x] of [[6,8],[6,9],[6,10],[7,10],[8,10],[9,10]])s.wall[y*QQT.W+x]=0;
    }
    if(['stationary-ally','rescue','carrier'].includes(boundary)){
      s.reset('open',{nativeItems:true,nativeTrap:true,teams:[0,1,1]});
      for(const k of ['wall','brick','crate','fuse','blastLinger'])s[k].fill(0);
      s.pos.set([9.5,10.5,6.5,8.5,6.5,9.5]);s.wall[5*QQT.W+10]=1;
      s.invuln.fill(0);s.blastCap.fill(3);s.bombsCap.fill(1);s.bunStored=[[0,0],[0,0]];s.bunInitial=[Infinity,Infinity];
      if(boundary==='rescue'){s.pos[4]=7.5;s.pos[5]=8.5;s.trapped[2]=50;}
      if(boundary==='carrier')s.bunCarried[2]=0;
    }
    if(boundary==='enemy-counterbubble'){
      const c=6*QQT.W+9;s.fuse[c]=1;s.owner[c]=0;s.bombBlast[c]=5;
    }
    const d=observed(s,bot);
    assert.notEqual(d.reason,'bomb_direct',`${boundary}: direct opportunity cannot override safety/duties`);
  }
  const {runDirect,SEEDS,OUTLET_COLUMNS}=require('../scripts/eval_direct_threat');
  assert.equal(runDirect(SEEDS[0],true,Coop,2).stats.enemyTraps,1,
    'RED: two-bubble follow-up must cover the redirected target body, not chain empty pressure');
  for(const seed of SEEDS){
    const joint=runDirect(seed,true,Coop,2,OUTLET_COLUMNS[SEEDS.indexOf(seed)]);
    assert.equal(joint.stats.enemyTraps,1);assert(joint.stats.multiBubblePairs>0);
    const first=joint.bubbles[0], second=joint.bubbles.find(b=>b.actualHit);
    assert(first.forcedReroute && !first.actualHit);
    assert(second.directThreat && second.placedTick<first.actualExplosionTick);
    assert(second.actualExplosionTick>first.actualExplosionTick);
    assert.equal(joint.stats.selfTraps+joint.stats.friendlyTraps+joint.stats.selfConfinedTicks+joint.stats.teammateBlockedTicks,0);
    assert(second.damageSafetyMarginTicks>=3);
    const hit=runDirect(seed,false),reroute=runDirect(seed,true);
    assert.equal(hit.stats.enemyTraps,1);assert.equal(reroute.stats.enemyTraps,1);
    for(const row of [hit,reroute]){
      assert.equal(row.stats.selfTraps+row.stats.friendlyTraps+row.stats.selfConfinedTicks,0);
      for(const bubble of row.bubbles.filter(b=>b.directThreat)){
        assert.equal(bubble.expectedExplosionTick,bubble.actualExplosionTick);
        assert(bubble.damageSafetyMarginTicks>=3);
        assert(bubble.actualHit||bubble.forcedReroute,'explicit target proof has actual tactical effect');
      }
    }
    assert(reroute.movementChanges.length===1);
  }
  assert.deepEqual(runDirect(SEEDS[0],true),runDirect(SEEDS[0],true),'replay is deterministic');
  console.log('Direct threat: timed off-ray attack, physical hit/reroute/full-body follow-up, blockers/status/half/chain guards, team duties, insurance and deterministic Sim passed');
}
