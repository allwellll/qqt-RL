'use strict';
// Independent bounded-plan variant. Frozen complete-v3 and production unchanged.
const V3=require('./interceptor_complete_v3');
const VERSION='bun-interceptor-complete-v4-bounded-progress';
const NO_PROGRESS_TICKS=31;

function homeDistances(state,team) {
  // Fixed wall-only graph: breaking bricks cannot masquerade as carrier motion.
  const {height:H,width:W,wall}=state,N=H*W;
  const d=new Array(N).fill(1000000000),queue=[],base=state.bunBases[team];
  for(let r=base[0];r<base[0]+3;r++)for(let c=base[1];c<base[1]+3;c++){
    const cell=r*W+c;if(r>=0&&r<H&&c>=0&&c<W&&!wall[cell]){d[cell]=0;queue.push(cell);}
  }
  for(let i=0;i<queue.length;i++){
    const a=queue[i],r=Math.floor(a/W),c=a%W;
    for(const [dr,dc] of [[-1,0],[1,0],[0,-1],[0,1]]){
      const rr=r+dr,cc=c+dc;if(rr<0||rr>=H||cc<0||cc>=W)continue;
      const b=rr*W+cc;if(!wall[b]&&d[b]>d[a]+1){d[b]=d[a]+1;queue.push(b);}
    }
  }
  return d;
}

class BunInterceptorBot extends V3.BunInterceptorBot {
  reset(seed) {super.reset(seed);this.noProgress=null;}
  updateNoProgress(state,pid) {
    const tick=state.tick;
    if(!Number.isInteger(tick)||tick<0)throw Error('complete-v4 requires actual nonnegative integer simulation tick');
    const me=state.players[pid],team=me.team??pid;
    const carrier=state.players.findIndex((p,q)=>q!==pid&&(p.team??q)!==team&&p.alive&&p.trapped<=0&&p.carrying===team);
    if(carrier<0||me.carrying>=0||!me.alive||me.trapped>0){this.noProgress=null;return null;}
    const foe=state.players[carrier];
    const cell=Math.max(0,Math.min(state.height-1,Math.floor(foe.y)))*state.width+
      Math.max(0,Math.min(state.width-1,Math.floor(foe.x)));
    let m=this.noProgress;
    if(!m||m.carrier!==carrier||tick<m.lastTick){
      const distance=homeDistances(state,foe.team??carrier);
      m={carrier,lastTick:tick,lastCell:cell,bestDistance:distance[cell],lastProgressTick:tick,blocked:false,distance};
      this.noProgress=m;return m;
    }
    if(tick===m.lastTick)return m;
    const progress=cell!==m.lastCell&&m.distance[cell]<m.bestDistance;
    if(progress){m.bestDistance=m.distance[cell];m.lastProgressTick=tick;}
    m.lastCell=cell;m.lastTick=tick;m.blocked=tick-m.lastProgressTick>=NO_PROGRESS_TICKS;
    return m;
  }
  analyze(state,pid) {
    const m=this.updateNoProgress(state,pid),decision=super.analyze(state,pid);
    return {...decision,no_progress_blocked:!!m?.blocked,
      no_progress_ticks:m?state.tick-m.lastProgressTick:0,
      carrier_best_distance:m?.bestDistance??-1};
  }
  routePlan(state,g,pid) {
    if(this.noProgress?.blocked)return null;
    return super.routePlan(state,g,pid);
  }
}
module.exports={...V3,BunInterceptorBot,VERSION,NO_PROGRESS_TICKS,homeDistances};
