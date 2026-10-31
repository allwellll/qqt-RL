'use strict';
const fs=require('fs'),crypto=require('crypto'),assert=require('assert');
process.env.QQT_STOCK_RULE='stock-window-v1';process.env.BT_FRESH_TIER='hard';
const B='runs/bomb_threat_v1/low_bomb_diag_u800',manifest=JSON.parse(fs.readFileSync(B+'/MANIFEST.json'));
const index=Number(process.argv[2]),entry=manifest.games.find(r=>r.index===index),p=entry.seat;
const Q=require('../web/sim-stock-window-v1'),{bfs,rayThreat,rejectionSnapshot,classify}=require('./bomb_threat_low_bomb_metrics'),{abilityRank,probabilities}=require('./bomb_threat_iter2_actions');
const original=Q.Sim.prototype.step,trace=[],snapshots=[];
const m={index,seed:entry.seed,seat:p,group:entry.group,ticks:0,alive_ticks:0,active_both_ticks:0,dead_ticks:0,trapped_ticks:0,first_death_tick:null,
 connected_active_ticks:0,brick_only_ticks:0,disconnected_hard_ticks:0,dynamic_disconnected_ticks:0,near5_active_ticks:0,near5_alive_ticks:0,
 own_half_ticks:0,own_cage_near_ticks:0,carry_enemy_ticks:0,carry_own_ticks:0,return_progress_ticks:0,
 move_counts:[0,0,0,0,0],ability_counts:[0,0,0],actual_move_ticks:0,stationary_position_ticks:0,
 bomb_requests:0,bombs_placed:0,rejected_requests:0,rejection_reason_counts:{},unexplained_rejections:0,
 bomb_legal_ticks:0,legal_bomb_probability_sum:0,legal_rank_counts:[0,0,0],opportunity_ticks:0,opportunity_probability_sum:0,opportunity_rank_counts:[0,0,0],rank2_mid_opportunity_ticks:0,
 invulnerable_geometric_threat_ticks:0,distance_histogram:{'0-5':0,'6-10':0,'11-20':0,'21+':0,disconnected:0,opponent_uncontrolled:0},
 observations:'passive same actual forward; masked ability softmax; current-grid BFS and direct-ray opportunity, not Q or kill prediction'};
const columns=['tick','alive','trapped','enemy_alive','enemy_trapped','grid_distance','ignore_bricks_distance','dynamic_distance','own_half','own_cage_near','carried_origin','move','ability','actual_placed','request_rejected','bomb_legal','bomb_masked_probability','bomb_legal_rank','current_ray_threat','enemy_invuln','ability_logits','ability_mask','ability_probabilities','bomb_unmasked_probability','position','enemy_position'];
let sref;
function cell(s,p){const [r,c]=s.centerCell(p);return r*15+c;}
function distances(s){const start=cell(s,p),target=cell(s,1-p);return {start,target,field:bfs(s,start),noBrick:bfs(s,start,true),dyn:bfs(s,start,false,true)};}
function snapshot(s){const d=distances(s);snapshots.push({tick:s.t,learner_cell:d.start,enemy_cell:d.target,bfs_distance:d.field[d.target],ignore_bricks_distance:d.noBrick[d.target],dynamic_distance:d.dyn[d.target],brick_only_connection:d.field[d.target]<0&&d.noBrick[d.target]>=0,alive:Array.from(s.alive),trapped:Array.from(s.trapped)});}
Q.Sim.prototype.step=function(actions){
 if(this.__btIdentity?.seed!==entry.seed)return original.call(this,actions);
 const s=this;sref=s;if(s.t%200===0)snapshot(s);
 const d=distances(s),dist=d.field[d.target],bd=d.noBrick[d.target],dyn=d.dyn[d.target],tick=s.t;
 const alive=Boolean(s.alive[p]),trapped=s.trapped[p],enemyAlive=Boolean(s.alive[1-p]),enemyTrapped=s.trapped[1-p],both=alive&&!trapped&&enemyAlive&&!enemyTrapped;
 const dec=s._daggerItemDecision?.[p];assert(dec,'production forward telemetry required');
 const rank=abilityRank(dec.logits,dec.legal),unmasked=probabilities(dec.logits,dec.logits.map(()=>1));
 const a=actions[p],ability=a[2]?2:a[1]?1:0,ray=rayThreat(s,p),enemyInvuln=s.invuln[1-p],legal=alive&&!trapped&&Boolean(dec.legal[1]),opportunity=legal&&both&&ray&&enemyInvuln<=0;
 const pos=[s.pos[p*2],s.pos[p*2+1]],epos=[s.pos[(1-p)*2],s.pos[(1-p)*2+1]],own=s.bunBases[s.team[p]].map(v=>v+1),enemy=s.bunBases[1-s.team[p]].map(v=>v+1);
 const md=(a,b)=>Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1]),half=md(pos,own)<=md(pos,enemy),near=md(pos,own)<=3;
 const carried=s.bunCarried[p],homeCells=[];for(let y=own[0]-1;y<=own[0]+1;y++)for(let x=own[1]-1;x<=own[1]+1;x++)if(y>=0&&y<13&&x>=0&&x<15)homeCells.push(y*15+x);
 const homeDistance=f=>{const dd=homeCells.map(i=>f[i]).filter(v=>v>=0);return dd.length?Math.min(...dd):-1;},homeBefore=homeDistance(d.field);
 const reject=rejectionSnapshot(s,actions,p);
 m.ticks++;m.alive_ticks+=alive;m.dead_ticks+=!alive;m.trapped_ticks+=alive&&trapped>0;m.active_both_ticks+=both;
 m.move_counts[a[0]]++;m.ability_counts[ability]++;m.own_half_ticks+=alive&&half;m.own_cage_near_ticks+=alive&&near;
 m.carry_enemy_ticks+=alive&&carried===1-s.team[p];m.carry_own_ticks+=alive&&carried===s.team[p];
 if(both){m.connected_active_ticks+=dist>=0;m.brick_only_ticks+=dist<0&&bd>=0;m.disconnected_hard_ticks+=bd<0;m.dynamic_disconnected_ticks+=dyn<0;
  m.near5_active_ticks+=dist>=0&&dist<=5;m.distance_histogram[dist<0?'disconnected':dist<=5?'0-5':dist<=10?'6-10':dist<=20?'11-20':'21+']++;
 }else m.distance_histogram.opponent_uncontrolled++;
 m.near5_alive_ticks+=alive&&dist>=0&&dist<=5;
 if(legal){m.bomb_legal_ticks++;m.legal_bomb_probability_sum+=rank.probability;m.legal_rank_counts[rank.rank-1]++;}
 if(opportunity){m.opportunity_ticks++;m.opportunity_probability_sum+=rank.probability;m.opportunity_rank_counts[rank.rank-1]++;m.rank2_mid_opportunity_ticks+=rank.rank===2&&rank.probability>=.2&&rank.probability<=.5;}
 m.invulnerable_geometric_threat_ticks+=legal&&both&&ray&&enemyInvuln>0;
 const info=original.call(s,actions),placed=Boolean(info.placed[p]);m.bomb_requests+=reject.requested;m.bombs_placed+=placed;
 const delta=md(pos,[s.pos[p*2],s.pos[p*2+1]]);m.actual_move_ticks+=delta>1e-7;m.stationary_position_ticks+=delta<=1e-7;
 if(info.died[p]&&m.first_death_tick===null)m.first_death_tick=s.t;
 if(reject.requested&&!placed){m.rejected_requests++;if(!reject.why.length&&p===1&&info.placed[0]&&reject.cell===cell(s,0))reject.why.push('same_cell_player0_placed_first');
  if(!reject.why.length){m.unexplained_rejections++;reject.why.push('unknown');}for(const why of reject.why)m.rejection_reason_counts[why]=(m.rejection_reason_counts[why]||0)+1;
 }
 if(alive&&s.alive[p]&&carried===1-s.team[p]&&s.bunCarried[p]===carried){const after=bfs(s,cell(s,p)),hd=homeDistance(after);m.return_progress_ticks+=homeBefore>=0&&hd>=0&&hd<homeBefore;}
 trace.push([tick,Number(alive),trapped,Number(enemyAlive),enemyTrapped,dist,bd,dyn,Number(half),Number(near),carried,a[0],ability,Number(placed),Number(reject.requested&&!placed),Number(legal),rank.probability,rank.rank,Number(ray),enemyInvuln,dec.logits,dec.legal,rank.probabilities,unmasked[1],pos,epos]);
 return info;
};
process.env.BT_REPLAY_BASE=B;require('./bomb_threat_best_replay_record');
process.on('beforeExit',()=>{
 if(m.status)return;snapshot(sref);m.status='RECORDED_PENDING_MATCH';m.category=classify(m);
 m.mean_bomb_probability_legal=m.bomb_legal_ticks?m.legal_bomb_probability_sum/m.bomb_legal_ticks:null;
 m.mean_bomb_probability_threat=m.opportunity_ticks?m.opportunity_probability_sum/m.opportunity_ticks:null;
 m.fractions={own_half:m.own_half_ticks/m.ticks,own_half_alive:m.own_half_ticks/m.alive_ticks,own_cage_near:m.own_cage_near_ticks/m.ticks,own_cage_near_alive:m.own_cage_near_ticks/m.alive_ticks,carry_enemy:m.carry_enemy_ticks/m.ticks,stationary_position:m.stationary_position_ticks/m.ticks};
 const out=`${B}/game_${String(index).padStart(3,'0')}`;fs.writeFileSync(out+'/TRACE.json',JSON.stringify({columns,rows:trace}));
 fs.writeFileSync(out+'/METRICS.json',JSON.stringify({...m,snapshots},null,2));console.log(JSON.stringify({index,category:m.category,bombs:m.bombs_placed,opportunities:m.opportunity_ticks}));
});
