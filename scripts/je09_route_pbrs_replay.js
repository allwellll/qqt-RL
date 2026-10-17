'use strict';
// Deterministic assist-off production simulator trace for the H8 g32 handoff.
const fs = require('fs');
const crypto = require('crypto');
const QQT = require('../web/sim.js');
const authority = require('./eval_web_hunter.js');
const inference = require('./je05_web_hunter.js');
const diagnostics = require('./je08_web_diagnostics.js');

const INF = 32767;
function sha(data) { return crypto.createHash('sha256').update(data).digest('hex'); }
function traceSnapshot(s) {
  return {tick:s.t, pos:Array.from(s.pos), alive:Array.from(s.alive), trapped:Array.from(s.trapped),
    bun_carried:Array.from(s.bunCarried), bun_stored:s.bunStored.map(x=>Array.from(x)),
    bun_loose:Array.from(s.bunLoose), wall:Array.from(s.wall), brick:Array.from(s.brick)};
}
function targetMap(s, target) {
  if (target == null || target < 0) return null;
  return diagnostics.distance(s.wall, s.brick, [target]);
}
async function main() {
  const [modelPath, outPath] = process.argv.slice(2);
  if (!modelPath || !outPath) throw new Error('usage: node scripts/je09_route_pbrs_replay.js MODEL.json OUT.json');
  if (process.env.CUDA_VISIBLE_DEVICES) throw new Error('replay must run CPU-only');
  const seed = 2026101732, team = 0, enemy = 1, ticks = 1200;
  const model = await inference.loadModel(modelPath);
  const s = diagnostics.setup('drop_return', seed, team, null, ticks);
  const tracker = new diagnostics.Tracker(team); tracker.trapCause[enemy][team] = true;
  const rng = QQT.mulberry32(seed ^ 0x515154);
  const rows = []; let objective = null, offset = 0, priorPhi = 0;
  const goalsFor = (base) => { const a=[]; for(let r=base[0];r<base[0]+3;r++)for(let c=base[1];c<base[1]+3;c++)a.push(r*QQT.W+c); return a; };
  while (!s.done && s.t < ticks) {
    const before = diagnostics.snapshot(s), raw = await model.model.act(s, team, rng);
    const actions=[]; actions[team]=[raw[0],Number(raw[1]),Number(raw[2]||0)]; actions[enemy]=[0,0,0];
    const info=s.step(actions); tracker.observe(s, before, info);
    const target=tracker.target[enemy], targetId=target==null?-1:target;
    const carry=s.bunCarried[team]===enemy, drop=target!=null&&tracker.credit[enemy]===team&&s.bunLoose[target*2+enemy]>0;
    const phase=carry?3:(drop?2:0), id=phase===3?enemy+1:targetId;
    let dist=INF;
    if(phase===3) dist=diagnostics.distance(s.wall,s.brick,goalsFor(s.bunBases[team]))[Math.floor(s.pos[team*2])*QQT.W+Math.floor(s.pos[team*2+1])];
    else if(phase===2) { const m=targetMap(s,target); if(m)dist=m[Math.floor(s.pos[team*2])*QQT.W+Math.floor(s.pos[team*2+1])]; }
    const base=phase===0?0:(dist<INF?-Math.min(dist,64)/64:-1);
    const key=`${phase}:${id}`;
    if(phase&&objective!==key){offset=priorPhi-base; objective=key;}
    const phi=phase?base+offset:priorPhi;
    const delta=.995*phi-priorPhi, shaped=.05*delta;
    const after=diagnostics.snapshot(s);
    rows.push({tick:s.t, action:[...actions[team]], state_before:before, state_after:after,
      route_phase:phase, target_identity:id, target_provenance:phase===2?'enemy_origin_drop':(phase===3?'enemy_bun':'none'),
      distance:dist, phi_before:priorPhi, phi_after:phi, potential_delta:delta, lambda:.05, shaped_reward:shaped,
      native_events:{died:Array.from(info.died||[]), mutual_death:Boolean(info.mutualDeath), pickup:s.bunCarried[team]!==before.carry[team]}});
    priorPhi=phi;
  }
  const manifest={schema:'je09.route-pbrs-replay/v1',checkpoint_generation:32,seed,team,opponent:'stationary native simulator opponent',assist:false,
    ticks:rows.length,seed_protocol:'fixed declared seed; production web/sim.js; model greedy inference',source_files:['web/sim.js','scripts/je08_web_diagnostics.js','scripts/je05_web_hunter.js']};
  fs.writeFileSync(outPath+'.tmp',JSON.stringify({manifest,manifest_sha256:sha(JSON.stringify(manifest)),model_sha256:model.sha,
    simulator_sha256:sha(fs.readFileSync('web/sim.js')),trajectory:rows})); fs.renameSync(outPath+'.tmp',outPath);
}
main().catch(e=>{console.error(e);process.exitCode=1;});
