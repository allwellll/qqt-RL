// 实验中 / EXPERIMENTAL: opt-in only; not the default web or training path.
/* Opt-in interception opponent. Existing Hunter and simulator remain intact. */
'use strict';
(function(root,factory){
  const api=factory(typeof module==='object'&&module.exports?require('./bun_hunter_bot'):root.QQTBunHunterBot);
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.QQTBunInterceptorBot=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(H){
  const INF=1000000,DIRS=[[-1,0],[1,0],[0,-1],[0,1]],FUSE=H.NEW_BOMB_TICK;
  function distances(g,starts,blocked){
    const d=new Int32Array(g.N).fill(INF),queue=new Int32Array(g.N);let h=0,t=0;
    for(const s of starts)if(s>=0&&s<g.N&&g.open[s]&&!blocked?.[s]&&d[s]){d[s]=0;queue[t++]=s;}
    while(h<t){const c=queue[h++];for(let a=0;a<4;a++){
      const q=g.nb[c*4+a];if(q>=0&&g.open[q]&&!blocked?.[q]&&d[q]===INF){d[q]=d[c]+1;queue[t++]=q;}
    }}return d;
  }
  function footprint(state,g,c,blast){
    const blocked=new Uint8Array(g.N);blocked[c]=1;
    for(let a=0;a<4;a++){let q=c;for(let k=1;k<=blast;k++){
      q=g.nb[q*4+a];if(q<0||state.wall[q])break;blocked[q]=1;
      if(state.brick[q]||g.bombAt[q]>=0)break;
    }}return blocked;
  }
  class BunInterceptorBot extends H.BunHunterBot{
    constructor(options={}){
      const difficulty=options.difficulty||'hard_intercept';
      if(difficulty!=='hard_intercept')throw Error('interceptor requires hard_intercept');
      super({...options,difficulty:'hard'});this.difficulty=difficulty;
    }
    act(sim,pid=1){
      const state=H.hunterStateFromSim(sim);
      // All of these are public physical fields. No policy/critic/oracle data.
      state.players.forEach((p,i)=>Object.assign(p,{
        movementStatus:Number(sim.movementStatus?.[i]||0),slideDir:Number(sim.slideDir?.[i]??1),
        postSlideSpeed:sim.nativeItems?.valueOf()? .6 : Number(sim.spdG[i])*(p.carrying>=0?Number(sim.bunCarrySpeedScale||.5):1)
      }));
      const d=this.analyze(state,pid);return [d.action[0],d.action[1],0];
    }
    foeOf(state,pid){
      const team=state.players[pid].team??pid;
      const carrier=this.enemiesOf(state,pid).find(q=>state.players[q].alive&&state.players[q].carrying===team);
      if(carrier!==undefined){state._foe||={};state._foe[pid]=carrier;return state.players[carrier];}
      return super.foeOf(state,pid);
    }
    projectedCarrier(g,foe){
      let cell=foe.cell,slideTicks=0;
      if(foe.movementStatus===2&&foe.slideDir<4){
        let next;while((next=g.nb[cell*4+foe.slideDir])>=0&&g.open[next]&&g.bombAt[next]<0)cell=next;
        const distance=Math.abs(Math.floor(cell/g.W)+.5-foe.y)+Math.abs(cell%g.W+.5-foe.x);
        slideTicks=Math.ceil(distance/Math.max(.001,.3*foe.speed));
      }
      return {...foe,cell,row:Math.floor(cell/g.W),col:cell%g.W,
        y:cell===foe.cell?foe.y:Math.floor(cell/g.W)+.5,x:cell===foe.cell?foe.x:cell%g.W+.5,
        speed:foe.movementStatus===2?foe.postSlideSpeed||foe.speed:foe.speed,slideTicks};
    }
    routePlan(state,g,pid){
      const me=state.players[pid],foe=this.foeOf(state,pid),team=me.team??pid;
      if(!foe.alive||foe.trapped>0||foe.carrying!==team||me.carrying>=0)return null;
      const f=this.projectedCarrier(g,foe),goals=this.baseCells(state,g,foe.team??(1-team));
      const home=distances(g,goals),from=distances(g,[f.cell]),own=distances(g,[me.cell]),short=home[f.cell];
      if(short>=INF)return null;
      const near=[],common=[];
      for(let c=0;c<g.N;c++)if(from[c]>0&&from[c]+home[c]<=short+2){
        near.push(c);const block=new Uint8Array(g.N);block[c]=1;
        if(distances(g,goals,block)[f.cell]>short+2)common.push(c);
      }
      const candidates=[];
      for(let c=0;c<g.N;c++){
        if(!g.open[c]||g.bombAt[c]>=0||own[c]>=INF)continue;
        const ray=footprint(state,g,c,me.blast),hit=near.filter(x=>ray[x]);
        if(!hit.length)continue;
        const choke=hit.filter(x=>common.includes(x));
        const impact=(choke.length?choke:hit).sort((a,b)=>from[a]-from[b]||a-b)[0];
        const carrierETA=f.slideTicks+(from[impact]===0?0:this.firstTicks(g,f,impact,false));
        // firstTicks is Manhattan: use graph length for all later corridor bends.
        const eta=f.slideTicks+Math.max(carrierETA-f.slideTicks,from[impact]*this.moveTicks(f,false));
        const arrive=own[c]===0?0:Math.max(this.firstTicks(g,me,c,false),own[c]*this.moveTicks(me,false));
        const detour=distances(g,goals,ray)[f.cell]-short;
        const timed=arrive+FUSE<=eta+H.FLAME_TICKS-1;
        if(!timed&&detour<4)continue;
        // Late unavoidable choke routes can still be obstructed immediately;
        // do not chase a target whose carrier has already passed its ray.
        if(eta+H.FLAME_TICKS<arrive)continue;
        candidates.push({plant:c,impact,arrival:arrive,carrierETA:eta,
          plantAt:Math.max(arrive,eta-FUSE),detour,timed,common:choke.length>0,
          score:(choke.length?0:10000)+Math.max(arrive,eta-FUSE)+own[c]});
      }
      candidates.sort((a,b)=>a.score-b.score||a.plant-b.plant);
      return {foe:f,goals,short,common,near,from,candidates,best:candidates[0]||null};
    }
    chooseGoal(state,g,pid,pred,perceive){
      this.interception=this.routePlan(state,g,pid);
      const plan=this.interception,me=state.players[pid],foe=this.foeOf(state,pid);
      if(plan){
        if(Math.abs(me.row-foe.row)+Math.abs(me.col-foe.col)<=2)
          return {mode:'INTERCEPT',seeds:super.attackSeeds(state,g,pid,pred,perceive),threat:false};
        if(plan.best)return {mode:'INTERCEPT',seeds:[{cell:plan.best.plant,cost:0}],threat:false,noDig:true};
        return {mode:'INTERCEPT',seeds:super.attackSeeds(state,g,pid,pred,perceive),threat:false};
      }
      return super.chooseGoal(state,g,pid,pred,perceive);
    }
    considerBomb(state,g,pid,pred,perceive,field){
      const plan=this.interception,me=state.players[pid];
      if(plan&&me.bombsLeft>0&&me.liveBombs<this.cfg.maxLiveBombs&&g.bombAt[me.cell]<0){
        const ray=footprint(state,g,me.cell,me.blast),home=distances(g,plan.goals,ray);
        const detour=home[plan.foe.cell]-plan.short;
        const atBlast=plan.near.filter(c=>{
          const t=plan.foe.slideTicks+Math.max(this.firstTicks(g,plan.foe,c,false),
            plan.from[c]*this.moveTicks(plan.foe,false));
          return ray[c]&&Math.abs(t-FUSE)<=this.moveTicks(plan.foe,false)+H.FLAME_TICKS;
        });
        const atTarget=plan.best?.plant===me.cell;
        // Deliberate wait aligns the fuse. Recompute every tick, including glue
        // slowdown and banana collision endpoint; no stale stored travel time.
        if(atTarget&&plan.best.plantAt>1)return null;
        if(atBlast.length||detour>=4){
          const mine={cell:me.cell,e:FUSE,blast:me.blast},withBomb=this.predict(state,g,[mine],perceive);
          const escape=this.escape(g,withBomb,me,false);let safe=escape.surv;
          if(this.cfg.robust)safe&=this.hypoSurvivors(state,g,pid,perceive,[mine]);
          for(const q of this.alliesOf(state,pid))if(state.players[q].alive&&!this.escape(g,withBomb,state.players[q],true).surv)safe=0;
          if(safe){const move=this.pickMove(g,me,safe,escape,field,true,this.dangerCells(withBomb,g.N));
            this.lastInterceptionPlacement={tick:state.tick,cell:me.cell,detour,timed:atBlast.length>0};
            return {move,reason:atBlast.length?'bomb_intercept_timed':'bomb_intercept_route'};
          }
        }
      }
      return super.considerBomb(state,g,pid,pred,perceive,field);
    }
    pickMove(g,me,acts,escape,field,threatened,danger){
      if(!threatened&&(acts&16)&&this.interception?.best?.plant===me.cell&&this.interception.best.plantAt>1)return 4;
      return super.pickMove(g,me,acts,escape,field,threatened,danger);
    }
  }
  return {BunInterceptorBot,distances,footprint,DIFFICULTIES:{hard_intercept:{...H.DIFFICULTIES.hard}},VERSION:'bun-interceptor-route-v1'};
});
