'use strict';
const {bfs}=require('./bomb_threat_low_bomb_metrics');
function kinematics(points){let path=0;for(let i=1;i<points.length;i++)path+=Math.hypot(points[i][0]-points[i-1][0],points[i][1]-points[i-1][1]);const net=Math.hypot(points.at(-1)[0]-points[0][0],points.at(-1)[1]-points[0][1]);return {path,net,ratio:path>1e-6?net/path:null,oscillation:path>1e-6&&net/path<.1};}
function reversals(actions,controlled){const events=[];for(let i=1;i<actions.length;i++)if(controlled[i]&&controlled[i-1]&&((actions[i]===2&&actions[i-1]===3)||(actions[i]===3&&actions[i-1]===2)))events.push(i);return {events,intervals:events.slice(1).map((x,i)=>x-events[i])};}
function movementField(s,start,ignoreBricks=false,dynamic=false){
 // Sim.step blocks wall/brick/live-bomb. A crate is a walk-over pickup,
 // although pushing a BOX into an occupied crate cell is forbidden.
 return bfs({...s,crate:null},start,ignoreBricks,dynamic);
}
function homeDistance(s,start,targets,ignoreBricks=false,dynamic=false){const f=movementField(s,start,ignoreBricks,dynamic),v=targets.map(i=>f[i]).filter(x=>x>=0);return v.length?Math.min(...v):-1;}
function category(w){if(!w.eligible)return '其他';if(w.carry&&w.net_home_progress<.1&&!w.bfs_progress)return '持包停滞';if(w.path>1e-6&&w.ratio<.1)return '原地振荡';if(w.path>=1&&w.ratio>=.1&&w.legal>0&&w.prob<.1)return '正常游走，只是概率低';return '其他';}
module.exports={kinematics,reversals,homeDistance,category,movementField};
