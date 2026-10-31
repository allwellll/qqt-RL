'use strict';
// Passive full-game observer. It leaves production actions, state and RNG intact.
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const base='runs/bomb_threat_v1/iter2/a3sr/round15/diagnostic';
const manifest=JSON.parse(fs.readFileSync(base+'/MANIFEST.json'));
const id=Number(process.argv[2]),entry=manifest.games.find(e=>e.id===id);
assert(entry);const destination=`${base}/game_${String(id).padStart(3,'0')}`;
fs.mkdirSync(destination,{recursive:true});
const Q=require('../web/sim-stock-window-v1');
const {bfs}=require('./bomb_threat_low_bomb_metrics');
const original=Q.Sim.prototype.step,trace=[];
let sim,initial,firstEnemyBomb=null,firstDeath=null,firstMove=null;
const moveCounts=[0,0,0,0,0],abilityCounts=[0,0,0],first100Move=[0,0,0,0,0],first100Ability=[0,0,0];
const cells=new Set(),earlyCells=new Set();let maxSpawnDistance=0,pathLength=0,moving=0,legalTicks=0,probSum=0;
let first100Moving=0,first100Legal=0,first100Prob=0,first100Trapped=0,first100Dead=0;
const p=entry.seat,cell=(s,p)=>{const [r,c]=s.centerCell(p);return r*15+c;};
Q.Sim.prototype.step=function(actions){
 if(this.__btIdentity?.seed!==entry.expected.seed)return original.call(this,actions);
 const s=this;sim=s;assert.equal(s.__btIdentity.seat,p);
 const pos=[s.pos[2*p],s.pos[2*p+1]],enemy=[s.pos[2*(1-p)],s.pos[2*(1-p)+1]],tick=s.t;
 if(!initial){
  const field=bfs({...s,crate:null},cell(s,p)),free=bfs({...s,crate:null},cell(s,p),true);
  initial={positions:Array.from(s.pos),team:Array.from(s.team),alive:Array.from(s.alive),
   trapped:Array.from(s.trapped),geometry_id:entry.geometry_id,birth_cells:entry.birth_cells,
   connected_cells:Array.from(field).filter(x=>x>=0).length,
   enemy_distance:field[cell(s,1-p)],enemy_distance_ignore_bricks:free[cell(s,1-p)],
   spawn_walkable:!s.wall[cell(s,p)]&&!s.brick[cell(s,p)],
   cell_layout_sha256:hash(JSON.stringify({wall:Array.from(s.wall),brick:Array.from(s.brick)}))};
  assert.deepEqual(Array.from(s.pos),entry.birth_cells.flatMap(x=>x.map(v=>v+.5)));
 }
 const a=actions[p],ability=a[2]?2:Number(a[1]),dec=s._daggerItemDecision?.[p];
 assert(dec&&dec.probabilities,'production masked probability telemetry missing');
 const alive=Boolean(s.alive[p]),trapped=s.trapped[p],legal=alive&&!trapped&&Boolean(dec.legal[1]);
 const probability=Number(dec.probabilities[1]);
 moveCounts[a[0]]++;abilityCounts[ability]++;legalTicks+=legal;probSum+=legal?probability:0;
 cells.add(cell(s,p));if(tick<100)earlyCells.add(cell(s,p));
 if(tick<100){first100Move[a[0]]++;first100Ability[ability]++;first100Legal+=legal;first100Prob+=legal?probability:0;first100Trapped+=alive&&trapped>0;first100Dead+=!alive;}
 const info=original.call(s,actions),delta=Math.abs(pos[0]-s.pos[2*p])+Math.abs(pos[1]-s.pos[2*p+1]);
 if(delta>1e-7){moving++;if(tick<100)first100Moving++;if(firstMove===null)firstMove=s.t;}
 pathLength+=delta;
 maxSpawnDistance=Math.max(maxSpawnDistance,Math.abs(s.pos[2*p]-initial.positions[2*p])+Math.abs(s.pos[2*p+1]-initial.positions[2*p+1]));
 if(info.placed[1-p]&&firstEnemyBomb===null)firstEnemyBomb=s.t;
 if(info.died[p]&&firstDeath===null)firstDeath=s.t;
 trace.push([tick,pos,enemy,a[0],ability,Number(alive),trapped,Number(info.placed[p]),Number(info.placed[1-p]),Number(info.died[p]),Number(legal),probability,Array.from(dec.legal)]);
 return info;
};
process.env.QQT_STOCK_RULE='stock-window-v1';process.env.BT_FRESH_TIER=entry.tier;
Object.assign(process.env,{JE10_ROUTE_TASK:'natural_full2400',JE10_ROUTE_GAMES:String(entry.games),
 JE10_ROUTE_START:String(entry.episode_index),JE10_ROUTE_END:String(entry.episode_index+1),JE10_ROUTE_SEED_START:String(entry.seed_start)});
process.argv=[process.argv[0],process.argv[1],'greedy',manifest.model.model_json,destination+'/evaluation.json',manifest.regions,entry.spawn_manifest];
assert.equal(hash(fs.readFileSync(manifest.model.model_json)),manifest.model.model_sha256);
require('./bomb_threat_fresh256_eval');
process.on('beforeExit',()=>{
 if(!sim||sim.__round15Recorded)return;sim.__round15Recorded=true;
 const side=JSON.parse(fs.readFileSync(destination+'/evaluation.json.outcome.json'));
 assert.equal(side.episodes.length,1);const actual=side.episodes[0];
 const fields=['seed','seat','ticks','done','winner','placed','requests','deaths','causal_kills','kills','stored'];
 const comparison=Object.fromEntries(fields.map(k=>[k,JSON.stringify(actual[k])===JSON.stringify(entry.expected[k])]));
 assert(Object.values(comparison).every(Boolean),'passive replay differs from frozen outcome');
 const record={status:'MATCH',id,seed:entry.expected.seed,seat:p,group:entry.group,geometry_id:entry.geometry_id,
  expected:entry.expected,comparison,initial,opponent_first_bomb_tick:firstEnemyBomb,own_first_death_tick:firstDeath,
  first_move_tick:firstMove,whole_game_never_moved:moving===0,spawn_local_roaming:maxSpawnDistance<=2,
  stationary_position_fraction:1-moving/trace.length,unique_cells:cells.size,first100_unique_cells:earlyCells.size,
  max_spawn_manhattan_displacement:maxSpawnDistance,path_length:pathLength,
  move_counts:moveCounts,ability_counts:abilityCounts,
  first100:{ticks:Math.min(trace.length,100),move_counts:first100Move,ability_counts:first100Ability,
   actual_move_ticks:first100Moving,legal_bomb_ticks:first100Legal,mean_legal_bomb_probability:first100Legal?first100Prob/first100Legal:null,
   trapped_ticks:first100Trapped,dead_ticks:first100Dead},
  legal_bomb_ticks:legalTicks,mean_legal_bomb_probability:legalTicks?probSum/legalTicks:null,
  early_suppression_observed:firstDeath!==null&&firstDeath<=100||first100Trapped>0,
  interpretation:'position geometry and timing are observations; causality of threat avoidance is not identified',
  model_sha256:manifest.model.model_sha256,teacher:false,engine:'production_JS',tier:entry.tier};
 fs.writeFileSync(destination+'/TRACE.json',JSON.stringify({columns:['tick','position','enemy_position','move','ability','alive','native_trapped','placed','enemy_placed','died','bomb_legal','masked_bomb_probability','ability_mask'],rows:trace}));
 fs.writeFileSync(destination+'/METRICS.json',JSON.stringify(record,null,2));
 console.log(JSON.stringify({id,status:'MATCH',ticks:trace.length,firstEnemyBomb,firstDeath,unique_cells:cells.size}));
});
