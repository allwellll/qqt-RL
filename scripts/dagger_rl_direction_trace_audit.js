'use strict';
// Read-only native one-step audit of fixed geometric disagreements, never Q.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const assert=require('node:assert/strict');
const P=require('../web/review_protocol.js'),A=require('./eval_web_hunter.js');
const D=require('./dagger_rl_route.js'),T=require('./je08_web_diagnostics.js');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const read=p=>JSON.parse(fs.readFileSync(p));
function write(p,x){fs.writeFileSync(p+'.tmp',JSON.stringify(x,null,2)+'\n');fs.renameSync(p+'.tmp',p);}
function goals(s,target){
  if(target.type==='real_loose')return [target.cell];
  assert(['home_cage','enemy_cage'].includes(target.type));
  const [r,c]=s.bunBases[target.team],out=[];
  for(let y=r;y<r+3;y++)for(let x=c;x<c+3;x++)out.push(y*15+x);
  return out;
}
async function audit(folder){
  const replayPath=path.join(folder,'REVIEW_FIXED_NATIVE.json');
  const tracePath=path.join(folder,fs.readdirSync(folder).find(x=>/^fixed-.*\.trace\.json$/.test(x)));
  const replay=read(replayPath),trace=read(tracePath),first=trace.first_deviation;
  const out={replay_sha256:sha(fs.readFileSync(replayPath)),trace_sha256:sha(fs.readFileSync(tracePath)),seed:replay.meta.seed,first};
  const player=new P.Player(replay,A.loadLevel());let replaySteps=0;
  const replayStep=player.step.bind(player);player.step=()=>{const stepped=replayStep();replaySteps+=Number(stepped);return stepped;};
  await player.verify();out.full_review_verified=true;
  if(!first){out.scope='No geometric disagreement; no phase/return success implied.';out.prefix_replay_step_calls=replaySteps;return out;}
  const tick=trace.diagnostic_ticks.find(t=>t.state_before_sha256===first.state_hash);
  assert(tick);const s=player.seek(first.tick),team=0;
  // All declared dev fixed game0 traces use P0; do not infer another seat.
  assert.equal(replay.meta.seed,2026102480);assert.equal(sha(JSON.stringify(P.state(s))),first.state_hash);
  const blocked=Array.from(s.wall,(v,i)=>Number(Boolean(v||s.brick[i]||s.pushable?.[i])));
  const field=T.distance(blocked,new Uint8Array(195),goals(s,first.target));
  const distance=s=>D.continuous([s.pos[team*2],s.pos[team*2+1]],field,blocked);
  out.distance_before=distance(s);assert(Math.abs(out.distance_before-first.distance)<1e-5);
  out.before={pos:Array.from(s.pos),movementStatus:s.movementStatus.slice(),slideDir:s.slideDir.slice(),legal_moves:s.legalMask().mm[team]};
  const branches=[];
  for(let move=0;move<5;move++){
    const state=player.seek(first.tick);assert.equal(sha(JSON.stringify(P.state(state))),first.state_hash);
    const row=replay.actions[first.tick].slice();row[team*2]=move;
    const actions=[[row[0],Number(row[1]===1),Number(row[1]===2)],[row[2],Number(row[3]===1),Number(row[3]===2)]];
    state.step(actions);
    if(move===first.action[0])assert.equal(sha(JSON.stringify(P.state(state))),tick.state_after_sha256);
    branches.push({move,pre_legal:!!out.before.legal_moves[move],distance_after:distance(state),pos_after:Array.from(state.pos),alive:state.alive[team],trapped:state.trapped[team],factual:move===first.action[0]});
  }
  const legal=branches.filter(x=>x.pre_legal),best=Math.min(...legal.map(x=>x.distance_after));
  out.native_one_step_best=legal.filter(x=>Math.abs(x.distance_after-best)<1e-5).map(x=>x.move);
  out.factual_one_step_best=out.native_one_step_best.includes(first.action[0]);
  out.geometric_oracle_overlaps_one_step_best=first.oracle.some(x=>out.native_one_step_best.includes(x));
  out.branches=branches;out.factual_next_hash_exact=true;out.diagnostic_native_step_calls=5;
  out.prefix_replay_step_calls=replaySteps;
  out.scope='Same original pre-state, RNG and other action factors; native next positions on the frozen pre-state target field. One-step direction contrasts, not Q/GAE, future safety or a full teacher correctness proof. Prefix replay counted separately from adopted training.';
  return out;
}
async function run(base){
  assert(base);const protocol=read(path.join(base,'PROTOCOL.json'));
  const cps=['no_update/u000',...protocol.paired_seeds.flatMap(s=>protocol.arms.flatMap(a=>[16,32].map(u=>`${a}_s${s}/u${String(u).padStart(3,'0')}`)))];
  const target=path.join(base,'FIXED_DEV_DIRECTION_AUDIT_v2.json');
  const result=fs.existsSync(target)?read(target):{schema:'fixed_dev_native_direction_audit/v2',expected:cps.length,checkpoints:{},controls_training:false,adopted_environment_steps:0,optimizer_steps:0};
  for(const cp of cps){const folder=path.join(base,'formal',cp,'retention');
    if(!fs.existsSync(path.join(folder,'natural_full64.json')))continue;
    if(!result.checkpoints[cp]){result.checkpoints[cp]=await audit(folder);write(target,result);}
  }
  result.completed=Object.keys(result.checkpoints).length;result.passed=result.completed===result.expected;
  result.native_branch_steps=Object.values(result.checkpoints).reduce((n,x)=>n+(x.diagnostic_native_step_calls||0),0);
  result.prefix_replay_steps=Object.values(result.checkpoints).reduce((n,x)=>n+x.prefix_replay_step_calls,0);
  result.prefix_scope='Original Review action-prefix deterministic replays, not new independent episodes or optimized transitions.';
  write(target,result);console.log(JSON.stringify({completed:result.completed,expected:result.expected,native_branch_steps:result.native_branch_steps,one_step_best_among_flagged:Object.values(result.checkpoints).filter(x=>x.factual_one_step_best).length}));
}
if(require.main===module)run(process.argv[2]).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={audit,run};
