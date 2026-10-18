'use strict';
// Observable JE10 route diagnostics copied from the frozen evaluator, independent of training.
const QQT=require('../web/sim'),Q=QQT;
const DIRS=[[-1,0],[1,0],[0,-1],[0,1],[0,0]],INF=32767;
function gridDistance(wall,brick,goals,h=QQT.H,w=QQT.W){
  const d=Array(h*w).fill(INF),visited=Array(h*w).fill(false);
  for(const i of goals)if(!wall[i])d[i]=0;
  for(let step=0;step<h*w;step++){
    let at=-1;for(let i=0;i<h*w;i++)if(!visited[i]&&(at<0||d[i]<d[at]))at=i;
    if(at<0||d[at]>=INF)break;visited[at]=true;
    const r=Math.floor(at/w),c=at%w,cost=1+3*Number(Boolean(brick[at]));
    for(const [dr,dc]of [[-1,0],[1,0],[0,-1],[0,1]]){
      const rr=r+dr,cc=c+dc;if(rr<0||rr>=h||cc<0||cc>=w||wall[rr*w+cc])continue;
      const k=rr*w+cc;d[k]=Math.min(d[k],d[at]+cost);
    }
  }
  return d;
}
function snapshot(s){return {carry:s.bunCarried.slice(),loose:Array.from(s.bunLoose),stock:s.bunStored.map(x=>x.slice()),
  pos:Array.from(s.pos),alive:s.alive.slice(),trap:s.trapped.slice(),brick:Array.from(s.brick)};}
class Tracker {
  constructor(team=0){this.team=team;this.nonce=[0,0];this.target=[null,null];this.credit=[-1,-1];this.trapCause=[[false,false],[false,false]];
    this.dropTick=null;this.pickupTick=null;this.captureTick=null;this.arrivalTick=null;this.net=0;this.positive=0;this.negative=0;this.invalid=0;
    this.eligibleDrops=0;this.ownPickup=0;this.phaseEnds=0;}
  observe(s,b,info){
    const team=this.team,enemy=1-team;
    for(let victim=0;victim<2;victim++){
      if(b.trap[victim]===0&&s.trapped[victim]>0)this.trapCause[victim]=info.causalDamageSource[victim].map(Boolean);
      if(info.died[victim]){
        const [r,c]=s.centerCell(victim),cell=r*QQT.W+c,origin=b.carry[victim];
        const cause=b.trap[victim]>0?this.trapCause[victim]:info.causalDamageSource[victim];
        if(origin>=0&&s.bunLoose[cell*2+origin]>b.loose[cell*2+origin]){
          this.nonce[origin]++;this.target[origin]=cell;
          this.credit[origin]=cause[1-victim]&&!cause[victim]&&s.alive[1-victim]&&!info.mutualDeath?1-victim:-1;
          if(origin===enemy&&this.credit[origin]===team){this.dropTick=s.t;this.eligibleDrops++;}
        }
        this.trapCause[victim]=[false,false];
      }
    }
    const acquired=b.carry[team]<0&&s.bunCarried[team]>=0,origin=s.bunCarried[team];
    const target=this.target[enemy],at=s.centerCell(team),index=at[0]*QQT.W+at[1];
    if(target!==null&&this.credit[enemy]===team&&b.loose[target*2+enemy]>0){
      if(index===target&&this.arrivalTick===null)this.arrivalTick=s.t;
      if(acquired&&origin===enemy&&index===target&&s.bunLoose[target*2+enemy]<b.loose[target*2+enemy]&&this.pickupTick===null)this.pickupTick=s.t;
    }
    if(acquired&&origin===team)this.ownPickup++;
    if(acquired&&origin===enemy&&this.pickupTick===null)this.invalid++;
    const captured=b.carry[team]===enemy&&s.bunStored[team][enemy]>b.stock[team][enemy];
    if(captured&&this.captureTick===null)this.captureTick=s.t;
    if(b.carry[team]===enemy&&s.bunCarried[team]!==enemy)this.phaseEnds++;
    const changed=b.brick.some((v,i)=>Boolean(v)!==Boolean(s.brick[i]));
    if(b.carry[team]===enemy&&s.bunCarried[team]===enemy&&s.alive[team]&&!info.died[team]&&!info.mutualDeath&&!changed){
      const base=s.bunBases[team],goals=[];
      for(let r=base[0];r<base[0]+3;r++)for(let c=base[1];c<base[1]+3;c++)goals.push(r*QQT.W+c);
      const map=gridDistance(s.wall,s.brick,goals),old=Math.floor(b.pos[team*2])*QQT.W+Math.floor(b.pos[team*2+1]);
      if(map[old]<INF&&map[index]<INF){const delta=map[old]-map[index];this.net+=delta;this.positive+=Math.max(0,delta);this.negative+=Math.min(0,delta);}
    }
    if(target!==null&&s.bunLoose[target*2+enemy]===0)this.credit[enemy]=-1;
  }
  result(){return {eligible_drop_events:this.eligibleDrops,first_pickup:this.pickupTick!==null,
    arrival_ticks:this.arrivalTick===null||this.dropTick===null?null:this.arrivalTick-this.dropTick,
    pickup_ticks:this.pickupTick===null||this.dropTick===null?null:this.pickupTick-this.dropTick,
    capture:this.captureTick!==null,capture_tick:this.captureTick,carry_net_topology:this.net,
    carry_positive_topology:this.positive,carry_negative_topology:this.negative,
    invalid_enemy_pickup:this.invalid,own_pickup:this.ownPickup,phase_ends:this.phaseEnds};}
}
function cage(s,t){const b=s.bunBases[t],a=[];for(let r=b[0];r<b[0]+3;r++)for(let c=b[1];c<b[1]+3;c++)a.push(r*15+c);return a;}
function continuous(pos,d,blocked){let best=INF;const r=Math.floor(pos[0]),c=Math.floor(pos[1]);
  for(const[dr,dc]of DIRS){const rr=r+dr,cc=c+dc,i=rr*15+cc;if(rr<0||rr>=13||cc<0||cc>=15||blocked[i]||d[i]>=INF)continue;best=Math.min(best,d[i]+Math.abs(pos[0]-rr-.5)+Math.abs(pos[1]-cc-.5));}return best;}
function objective(s,team,tracker){
  const enemy=1-team,carry=s.bunCarried[team]===enemy,window=!s.alive[enemy]||s.trapped[enemy]>0;
  const foeDistance=Math.abs(s.pos[team*2]-s.pos[enemy*2])+Math.abs(s.pos[team*2+1]-s.pos[enemy*2+1]);
  if(!s.alive[team]||s.trapped[team]>0||(!window&&(!carry||foeDistance<=4)))return null;
  let goals,phase,target;
  if(carry){goals=cage(s,team);phase='C_RETURN';target={type:'home_cage',team};}
  else if(window&&tracker.target[enemy]!==null&&s.bunLoose[tracker.target[enemy]*2+enemy]>0&&tracker.credit[enemy]===team){goals=[tracker.target[enemy]];phase='A_ENEMY_DROP';target={type:'real_loose',origin:enemy,cell:goals[0],nonce:tracker.nonce[enemy]};}
  else if(window&&s.bunStored[enemy][enemy]>0){goals=cage(s,enemy);phase='B_STOCK';target={type:'enemy_cage',team:enemy};}
  else return null;
  const blocked=Array.from(s.wall,(v,i)=>Number(Boolean(v||s.brick[i]||s.pushable?.[i]))),d=gridDistance(blocked,new Uint8Array(195),goals);
  const pos=[s.pos[team*2],s.pos[team*2+1]],distance=continuous(pos,d,blocked),mask=s.legalMask().mm[team];
  const gain=DIRS.map(([dr,dc],move)=>{const rr=Math.floor(pos[0])+dr,cc=Math.floor(pos[1])+dc,i=rr*15+cc;
    if(!mask[move]||rr<0||rr>=13||cc<0||cc>=15||blocked[i]||s.fuse[i]>0||s.blastLinger[i]>0)return -INF;
    return distance-continuous([pos[0]+.18*dr,pos[1]+.18*dc],d,blocked);});
  const best=Math.max(...gain),oracle=gain.map((v,i)=>v>1e-5&&v>=best-1e-5?i:-1).filter(i=>i>=0);
  if(d[Math.floor(pos[0])*15+Math.floor(pos[1])]===0&&mask[4])oracle.push(4);
  return {phase,target,distance,oracle,phi:distance>=INF?-1:-Math.min(distance,64)/64};
}

module.exports={Tracker,snapshot,objective};
