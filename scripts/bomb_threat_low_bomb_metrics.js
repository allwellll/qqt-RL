'use strict';
const D=[[-1,0],[1,0],[0,-1],[0,1]],W=15,N=195;
function bfs(s,start,ignoreBricks=false,dynamic=false){
 const dist=new Int16Array(N).fill(-1),q=new Int16Array(N);dist[start]=0;q[0]=start;let end=1;
 for(let i=0;i<end;i++){const x=q[i],r=Math.floor(x/W),c=x%W;for(const[dr,dc]of D){const y=r+dr,z=c+dc,j=y*W+z;
  if(y<0||y>=13||z<0||z>=W||dist[j]>=0||s.wall[j]||(!ignoreBricks&&s.brick[j])||s.pushable?.[j]||s.crate?.[j]||(dynamic&&s.fuse[j]>0))continue;
  dist[j]=dist[x]+1;q[end++]=j;
 }}return dist;
}
function rayThreat(s,p){
 const [r,c]=s.centerCell(p),[er,ec]=s.centerCell(1-p),d=Math.abs(r-er)+Math.abs(c-ec);
 if((r!==er&&c!==ec)||d>s.blastCap[p])return false;if(d===0)return true;
 const dr=Math.sign(er-r),dc=Math.sign(ec-c);
 for(let k=1;k<=d;k++){const i=(r+dr*k)*W+c+dc*k;
  if(s.wall[i])return false;if(k===d)return true;
  if(s.brick[i]||s.fuse[i]>0||s.pushable?.[i]||s.crate?.[i])return false;
 }return true;
}
function rejectionSnapshot(s,actions,p){
 const [r,c]=s.centerCell(p),i=s._actionCell(actions[p][4],r*15+c),why=[];let live=0;
 for(let j=0;j<N;j++)if(s.owner[j]===p&&s.fuse[j]>1)live++;
 if(!s.alive[p])why.push('dead');if(s.trapped[p]>0)why.push('native_sugar');if(s.fuse[i]>1)why.push('existing_bomb_after_decrement');
 if(s.brick[i])why.push('brick');if(s.isBun&&s.bunCarried[p]>=0)why.push('carrying');
 // P0 may place first during same step; owner counts itself only, same-cell collision separately checked after step.
 if(live>=s.bombsCap[p])why.push('capacity_after_decrement');
 return {requested:actions[p][1]===1,cell:i,why,live_after_decrement:live};
}
function classify(m){
 if(m.group!=='low')return '对照';
 if((m.first_death_tick!==null&&m.first_death_tick<=400)||(m.dead_ticks+m.trapped_ticks)/m.ticks>=.25)return '被对手压制';
 if(m.active_both_ticks&&m.brick_only_ticks/m.active_both_ticks>=.5)return '碰不到敌人';
 if(m.rank2_mid_opportunity_ticks>=10&&m.rank2_mid_opportunity_ticks/Math.max(1,m.opportunity_ticks)>=.5)return '去打了但argmax选不到放泡';
 if(m.active_both_ticks&&m.connected_active_ticks/m.active_both_ticks>=.5&&m.near5_active_ticks/m.active_both_ticks<.1&&m.own_half_ticks/m.ticks>=.6)return '能碰到但躲着不打';
 return '其他';
}
module.exports={bfs,rayThreat,rejectionSnapshot,classify};
