'use strict';
const assert = require('assert'), fs = require('fs'), crypto = require('crypto');
const QQT = require('../web/sim'), Hunter = require('../web/bun_hunter_bot'), Coop = require('../web/bun_coop_hunter_bot');
const { referenceBot } = require('./eval_bot_cooperation');
const { scene } = require('../web/test_bot_direct_threat');
const { trace } = require('./eval_staggered_attack');
const SEEDS = [2026100801, 2026100802, 2026100803, 2026100804];
const BASELINE = 'ecaf59423782ff511e04b2e25543f99e2eece4eb';
const OUTLET_COLUMNS = [11, 12, 13, 11];
function runDirect(seed, responsive, api = Coop, capacity = 1, outletColumn = null) {
  const sim = scene(seed), bot = new api.BunCoopHunterBot({ difficulty: 'hard', seed });
  sim.bombsCap.fill(capacity);
  if (outletColumn != null) {
    sim.wall[8 * QQT.W + outletColumn] = 1;
    sim.wall[9 * QQT.W + outletColumn] = 1;
  }
  const controller = new Coop.BunCoopHunterBot({ difficulty: 'hard', seed });
  const oracle = new Coop.BunCoopHunterBot({ difficulty: 'hard', seed });
  oracle.analyze = function (state, pid) {
    this.decisionState = state; this.observeEnemyTrends(state,pid);
    return { action:[4,0], reason:'observer', mode:'OBSERVE' };
  };
  let changed = false, evade = null;
  const proofEvents = [], movementChanges = [];
  const result = trace(sim, [null, bot], 1, (s, bots, decisions) => {
    let targetMove = 0;
    // Independently react to any visible danger: choose a safe constant route.
    // The attacker sees only past movement, never this controller's future input.
    if (responsive && s.fuse.some(f => f > 0) && !changed) {
      controller.analyzeSim(s, 0);
      const st = controller.decisionState, g = controller.geometry(st), p = controller.predict(st, g, [], () => true);
      let threateningSources = [];
      const moveHit = move => {
        let y = st.players[0].y, x = st.players[0].x;
        for (let t = 1; t <= p.T; t++) {
          const next = st.threatPosition(0, y, x, move, p, t); if (!next) return null;
          [y, x] = next;
          if (t > st.players[0].invuln && st.threatHit(0,y,x,p.impact.subarray(t*g.N,(t+1)*g.N))) {
            if(move===0) threateningSources=p.sources.filter(source=>source.tick===t).filter(source=>{
              const covered=new Uint8Array(g.N);for(const c of source.covered)covered[c]=1;
              return st.threatHit(0,y,x,covered);
            }).map(source=>source.cell);
            return true;
          }
        }
        return false;
      };
      if (moveHit(0) === true) {
        evade = [3,2,1,4].find(a => moveHit(a) === false);
        if (evade != null) { changed = true; movementChanges.push({ tick: s.t, from: 0, to: evade,
          threateningSources, reason: 'visible-bomb-invalidates-intended-route' }); }
      }
    }
    if (changed) targetMove = evade;
    const action = bot.act(s, 1); decisions.set(1, bot.lastDecision);
    if (s.t === 0) action[1] = 0; // predeclared one-tick observation warmup for both APIs
    if (s.t >= 90) { s.bombsCap.fill(0); action[1] = 0; }
    oracle.analyzeSim(s,1);
    if (action[1]) {
      const st=oracle.decisionState,g=oracle.geometry(st),cell=st.players[1].cell;
      st.bombs=Hunter.hunterStateFromSim(s).bombs;
      const before=oracle.predict(st,g,[],()=>true),after=oracle.predict(st,g,[{cell,e:31,blast:st.players[1].blast}],()=>true);
      const proof=oracle.directThreat(st,g,1,cell,before,after);
      if(proof)proofEvents.push({tick:s.t,cell,...proof,
        offRayAtPlacement:!oracle.blastCells(st,g,cell,st.players[1].blast).has(st.players[proof.target].cell)});
    }
    return [[targetMove,0,0,0], action];
  }, 125);
  for (const b of result.bubbles) {
    const proof = proofEvents.find(p => p.tick + 1 === b.placedTick && p.cell === b.cell);
    if (proof) {
      b.directThreat = proof;
      b.actualTargetPath = result.frames.filter(f => f.tick >= b.placedTick && f.tick <= b.actualExplosionTick)
        .map(f => ({ tick:f.tick, y:f.after[0], x:f.after[1], move:f.actions[0][0] }));
      const frame=result.frames.find(f=>f.tick===b.actualExplosionTick);
      const ownBodyHit=frame && frame.contacts[proof.target].every(c=>b.covered.includes(c));
      b.actualHit = !!ownBodyHit && (b.hitEnemies || []).includes(proof.target);
      b.forcedReroute = movementChanges.some(c => c.tick >= b.placedTick && c.tick < b.actualExplosionTick && c.threateningSources.includes(b.cell));
    }
  }
  return { seed, responsive, capacity, outletColumn, ...result, proofEvents, movementChanges };
}
function summary(rows) {
  const bubbles=rows.flatMap(r=>r.bubbles.filter(b=>b.owner===1)), proofs=bubbles.filter(b=>b.directThreat);
  return { episodes:rows.length, bombs:bubbles.length, targetProofs:proofs.length,
    offRayPredictiveAttempts:proofs.filter(b=>b.directThreat.offRayAtPlacement).length,
    effectiveOffRayAttempts:proofs.filter(b=>b.directThreat.offRayAtPlacement&&(b.actualHit||b.forcedReroute)).length,
    meanFirstThreatPlacementTick:rows.reduce((n,r)=>n+Math.min(...r.bubbles.filter(b=>b.directThreat).map(b=>b.placedTick)),0)/rows.length,
    meanFirstActualHitTick:rows.reduce((n,r)=>n+Math.min(...r.bubbles.filter(b=>b.actualHit).map(b=>b.actualExplosionTick)),0)/rows.length,
    missedActualHits:bubbles.filter(b=>(b.hitEnemies||[]).includes(0)&&!b.directThreat).length,
    targetProportion:proofs.length/Math.max(1,bubbles.length), conditionalPredictedHits:proofs.length,
    actualHits:rows.reduce((n,r)=>n+r.stats.pressureContacts,0), actualTraps:rows.reduce((n,r)=>n+r.stats.enemyTraps,0),
    forcedReroutes:rows.reduce((n,r)=>n+r.movementChanges.length,0), effectiveProofs:proofs.filter(b=>b.actualHit||b.forcedReroute).length,
    predictionTruePositives:proofs.filter(b=>b.actualHit).length, conditionalMisses:proofs.filter(b=>!b.actualHit).length,
    conditionalHitPrecision:proofs.filter(b=>b.actualHit).length/Math.max(1,proofs.length),
    guaranteedHitClaims:0, ...Object.fromEntries(['selfTraps','friendlyTraps','selfConfinedTicks','teammateBlockedTicks','overlapTicks'].map(k=>[k,rows.reduce((n,r)=>n+r.stats[k],0)])) };
}
if (require.main === module) {
  const reference = referenceBot(BASELINE), baseline = reference.api;
  const before = SEEDS.flatMap(seed=>[false,true].map(responsive=>runDirect(seed,responsive,baseline)));
  const after = SEEDS.flatMap(seed=>[false,true].map(responsive=>runDirect(seed,responsive,Coop)));
  const jointBefore=SEEDS.map((seed,i)=>runDirect(seed,true,baseline,2,OUTLET_COLUMNS[i]));
  const jointAfter=SEEDS.map((seed,i)=>runDirect(seed,true,Coop,2,OUTLET_COLUMNS[i]));
  const record={ baseline:BASELINE, baselineBotHash:reference.sha256,
    candidateHashes:Object.fromEntries(['bun_hunter_bot.js','bun_coop_hunter_bot.js'].map(file=>[file,
      crypto.createHash('sha256').update(fs.readFileSync(`web/${file}`)).digest('hex')])),
    seeds:SEEDS, protocol:{ capacity:1, fuse:31, insurance:3, observationWarmup:1, activeUntil:90,
    targetPolicies:['continue-observed-up','visible-bomb-safe-constant-reroute'], controlledFixture:true, futureInputRead:false },
    before:{summary:summary(before),rows:before},after:{summary:summary(after),rows:after},
    joint:{protocol:{capacity:2,outletColumns:OUTLET_COLUMNS,selection:'explicit controlled outlet fixtures; no natural win-rate claim'},
      before:{summary:summary(jointBefore),rows:jointBefore},after:{summary:summary(jointAfter),rows:jointAfter}} };
  assert(record.after.summary.effectiveOffRayAttempts>record.before.summary.effectiveOffRayAttempts);
  assert(record.after.summary.meanFirstThreatPlacementTick<record.before.summary.meanFirstThreatPlacementTick);
  for(const key of ['selfTraps','friendlyTraps','selfConfinedTicks','teammateBlockedTicks','overlapTicks']) {
    assert(record.after.summary[key]<=record.before.summary[key],`safety regression: ${key}`);
    assert(record.joint.after.summary[key]<=record.joint.before.summary[key],`joint safety regression: ${key}`);
  }
  for(const [oldRows,newRows] of [[before,after],[jointBefore,jointAfter]]) for(let i=0;i<oldRows.length;i++) {
    for(const key of ['selfTraps','friendlyTraps','selfConfinedTicks','teammateBlockedTicks','overlapTicks','ownBubbleBlockedTicks'])
      assert(newRows[i].stats[key]<=oldRows[i].stats[key],`per-row safety regression ${i}: ${key}`);
    assert(newRows[i].stats.enemyTraps>=oldRows[i].stats.enemyTraps);
    for(const bubble of newRows[i].bubbles.filter(b=>b.directThreat)) {
      assert.equal(bubble.expectedExplosionTick,bubble.actualExplosionTick);
      assert(bubble.damageSafetyMarginTicks>=3);
      assert(bubble.actualHit||bubble.forcedReroute);
    }
  }
  fs.writeFileSync('runs/qqt_direct_threat_20261008/comparison.json',JSON.stringify(record,null,2)+'\n');
  console.log(JSON.stringify({before:record.before.summary,after:record.after.summary,
    jointBefore:record.joint.before.summary,jointAfter:record.joint.after.summary},null,2));
}
module.exports={runDirect,SEEDS,BASELINE,OUTLET_COLUMNS,summary};
