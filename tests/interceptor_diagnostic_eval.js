'use strict';
// Passive instrumentation of the opt-in interceptor. No extra RNG calls or
// changed actions. Full joint-action hashes must match the frozen baseline.
const Q=require('../web/sim-bun-native-v2'),I=require('../web/bun_interceptor_bot'),H=require('../web/bun_hunter_bot');
const Original=I.BunInterceptorBot,INF=1e6;
function eligibility(state,g,pid,plan){
 const me=state.players[pid],foe=state.players[1-pid],out={};
 if(!plan)return {plan_absent:me.carrying>=0?'bot_carrying':!foe.alive?'carrier_dead':foe.trapped>0?'carrier_controlled':'no_home_path'};
 out.common_cells=plan.common.length;out.near_cells=plan.near.length;
 out.candidates=plan.candidates.length;
 if(plan.best)return {...out,best:plan.best};
 const own=I.distances(g,[me.cell]);let noHit=0,unreachable=0,late=0,passed=0,valid=0;
 for(let c=0;c<g.N;c++){
  if(!g.open[c]||g.bombAt[c]>=0)continue;
  const ray=I.footprint(state,g,c,me.blast),hit=plan.near.filter(x=>ray[x]);
  if(!hit.length){noHit++;continue;}if(own[c]>=INF){unreachable++;continue;}
  const choke=hit.filter(x=>plan.common.includes(x)),impact=(choke.length?choke:hit).sort((a,b)=>plan.from[a]-plan.from[b]||a-b)[0];
  const eta=plan.foe.slideTicks+Math.max(Original.prototype.firstTicks.call(this,g,plan.foe,impact,false),plan.from[impact]*this.moveTicks(plan.foe,false));
  const arrival=own[c]===0?0:Math.max(this.firstTicks(g,me,c,false),own[c]*this.moveTicks(me,false));
  const detour=I.distances(g,plan.goals,ray)[plan.foe.cell]-plan.short;
  if(arrival+H.NEW_BOMB_TICK>eta+H.FLAME_TICKS-1&&detour<4){late++;continue;}
  if(eta+H.FLAME_TICKS<arrival){passed++;continue;}valid++;
 }
 return {...out,reject_no_hit:noHit,reject_unreachable:unreachable,reject_time:late,reject_passed:passed,valid};
}
class DiagnosticBot extends Original{
 act(sim,pid=1){
  this.row={tick:sim.t,pid,carrier:1-pid,bot_carrying:sim.bunCarried[pid],carrier_origin:sim.bunCarried[1-pid],
   bot_alive:!!sim.alive[pid],bot_trapped:sim.trapped[pid],carrier_alive:!!sim.alive[1-pid],carrier_trapped:sim.trapped[1-pid],
   bot_pos:Array.from(sim.pos.slice(2*pid,2*pid+2)),carrier_pos:Array.from(sim.pos.slice(2*(1-pid),2*(1-pid)+2))};
  sim.__interceptorDiagBot=this;const a=super.act(sim,pid);this.row.mode=this.lastDecision.mode;this.row.reason=this.lastDecision.reason;
  this.row.action=a;return a;
 }
 chooseGoal(...args){const goal=super.chooseGoal(...args),[state,g,pid]=args,plan=this.interception;
  this.row.planner=eligibility.call(this,state,g,pid,plan);
  this.row.goal=goal.seeds.map(x=>x.cell);this.row.goal_source=plan?(Math.abs(state.players[pid].row-state.players[1-pid].row)+Math.abs(state.players[pid].col-state.players[1-pid].col)<=2?'near_attackSeeds':plan.best?(plan.best.common?'common_choke':'near_route'):'fallback_attackSeeds'):'parent_goal';
  return goal;
 }
 considerBomb(...args){
  const [state,g,pid]=args,me=state.players[pid],plan=this.interception;
  const b=this.row.bomb={called:true};
  if(plan){
   b.inventory=me.bombsLeft>0;b.live_limit=me.liveBombs<this.cfg.maxLiveBombs;b.cell_free=g.bombAt[me.cell]<0;
   const ray=I.footprint(state,g,me.cell,me.blast),detour=I.distances(g,plan.goals,ray)[plan.foe.cell]-plan.short;
   const hits=plan.near.map(c=>({cell:c,eta:plan.foe.slideTicks+Math.max(this.firstTicks(g,plan.foe,c,false),plan.from[c]*this.moveTicks(plan.foe,false))})).filter(x=>ray[x.cell]&&Math.abs(x.eta-H.NEW_BOMB_TICK)<=this.moveTicks(plan.foe,false)+H.FLAME_TICKS);
   b.detour=detour;b.time_hits=hits;b.at_target=plan.best?.plant===me.cell;b.wait=!!(b.at_target&&plan.best.plantAt>1);
   b.proposed=b.inventory&&b.live_limit&&b.cell_free&&!b.wait&&(hits.length>0||detour>=4);
  }
  this.bombContext={pid,ownEscape:null,robust:[],allied:[]};
  const result=super.considerBomb(...args),ctx=this.bombContext;this.bombContext=null;
  b.result=result?.reason||null;b.own_escape=ctx.ownEscape;b.robust=ctx.robust;b.allied=ctx.allied;
  return result;
 }
 escape(g,pred,player,...rest){const result=super.escape(g,pred,player,...rest),c=this.bombContext;
  if(c&&player===this.currentPlayer&&c.ownEscape===null)c.ownEscape=result.surv;
  return result;
 }
 analyze(state,pid){this.currentPlayer=state.players[pid];return super.analyze(state,pid);}
 hypoSurvivors(...args){const r=super.hypoSurvivors(...args);if(this.bombContext)this.bombContext.robust.push(r);return r;}
}
I.BunInterceptorBot=DiagnosticBot;
const P=Q.Sim.prototype,step=P.step,explode=P._resolveExplosions;
P._resolveExplosions=function(...args){const r=explode.apply(this,args),d=this.__interceptorDiag;
 if(d)for(const src of r.sources||[]){const event=d.pending.find(x=>x.cell===src.cell&&x.explosion_tick===null);
  if(event){const p=event.carrier,pos=Array.from(this.pos.slice(2*p,2*p+2));event.explosion_tick=this.t;
   event.carrier_pos_at_explosion=pos;event.carrier_distance_at_explosion=Math.abs(pos[0]-(Math.floor(src.cell/15)+.5))+Math.abs(pos[1]-(src.cell%15+.5));
   event.ray_hits_carrier_cell=!!src.covered[Math.floor(pos[0])*15+Math.floor(pos[1])];event.carrier_still_same_possession=this.bunCarried[p]===event.origin;}}
 return r;};
P.step=function(actions){
 if(!this.__btIdentity)return step.call(this,actions);
 const d=this.__interceptorDiag ||= {rows:[],placements:[],pending:[]},bot=this.__interceptorDiagBot,row=bot?.row;
 const r=step.call(this,actions);
 if(row&&row.tick===this.t-1&&row.carrier_origin===(this.team[row.pid]??row.pid)){
  d.rows.push(row);
  if(r.placed[row.pid]){
   const cell=Math.floor(row.bot_pos[0])*15+Math.floor(row.bot_pos[1]),best=row.planner?.best;
   const hits=row.bomb?.time_hits||[],impact=best?.plant===cell?{cell:best.impact,eta:best.carrierETA}:hits.slice().sort((a,b)=>Math.abs(a.eta-H.NEW_BOMB_TICK)-Math.abs(b.eta-H.NEW_BOMB_TICK)||a.cell-b.cell)[0];
   const e={tick:this.t,cell,carrier:row.carrier,origin:row.carrier_origin,reason:row.reason,mode:row.mode,
    predicted_from_tick:row.tick,impact:impact?.cell??null,predicted_arrival_relative:impact?.eta??null,
    actual_arrival_tick:null,arrival_censor:null,explosion_tick:null};d.placements.push(e);d.pending.push(e);
  }
 }
 for(const e of d.pending){const p=e.carrier,c=Math.floor(this.pos[2*p])*15+Math.floor(this.pos[2*p+1]);
  if(e.actual_arrival_tick===null&&!e.arrival_censor&&e.impact!==null){
   if(this.bunCarried[p]!==e.origin||!this.alive[p])e.arrival_censor='possession_ended';
   else if(c===e.impact){e.actual_arrival_tick=this.t;e.arrival_error_ticks=this.t-(e.predicted_from_tick+e.predicted_arrival_relative);}
   else if(this.done)e.arrival_censor='terminal';
  }
 }
 if(this.done&&this.__bt)this.__bt.interceptor_diagnostic={rows:d.rows,placements:d.placements,
  definitions:{exposure:'pre-action enemy carries bot-origin package; all states including bot dead/controlled/carrying retained',
   arrival:'first actual cell occupancy after placement in the same possession; absent visits censored',
   detonation_distance:'Manhattan from actual carrier position after movement to bomb cell center at actual resolution',
   escape:'first own escape in considerBomb; robust intersection must be computed with own_escape, not nonzero robust alone'},
  limitations:['arrival uses current-speed route ETA, not a route oracle','bot action intent is distinct from geometric interception','no action/RNG/physics mutation']};
 return r;
};
module.exports={Original,DiagnosticBot,eligibility};
if(require.main===module)require('../scripts/r4_interceptor_eval');
