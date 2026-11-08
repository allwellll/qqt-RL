'use strict';
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.QQTKillOpeningItems=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const OPENING_ITEM_COUNT=24;
  const W=15,H=13;
  function distances(sim,spawn){
    const result=new Int16Array(W*H).fill(-1),queue=[spawn[0]*W+spawn[1]];
    result[queue[0]]=0;
    for(let head=0;head<queue.length;head++){
      const cell=queue[head],row=Math.floor(cell/W),col=cell%W;
      for(const [dr,dc] of [[-1,0],[1,0],[0,-1],[0,1]]){
        const y=row+dr,x=col+dc,i=y*W+x;
        if(y<0||y>=H||x<0||x>=W||result[i]>=0||sim.wall[i]||sim.brick[i])continue;
        result[i]=result[cell]+1;queue.push(i);
      }
    }
    return result;
  }
  function apply(sim){
    if(!sim.killMode)return null;
    if(sim.killOpeningItems)return sim.killOpeningItems;
    const first=sim.centerCell(0),mirrorSpawn=[first[0],W-1-first[1]];
    // 806 has four declared mirrored birth pairs. Independent side sampling
    // can otherwise make equal nearest distances mathematically impossible.
    if(!sim._nativeStaticBlocked(...mirrorSpawn)){
      sim.pos[2]=mirrorSpawn[0]+.5;sim.pos[3]=mirrorSpawn[1]+.5;
      sim.spawnCells[1]=mirrorSpawn;
    }
    const spawns=[sim.centerCell(0),sim.centerCell(1)],distance=spawns.map(p=>distances(sim,p));
    const allowed=new Set(sim._itemSupplyCells().filter(cell=>spawns.every(([r,c])=>
      Math.max(Math.abs(Math.floor(cell/W)-r),Math.abs(cell%W-c))>1)));
    const pairs=[];
    for(const cell of allowed){
      const mirror=Math.floor(cell/W)*W+(W-1-cell%W);
      if(cell>=mirror||!allowed.has(mirror))continue;
      const a=Math.min(distance[0][cell],distance[0][mirror]),b=Math.min(distance[1][cell],distance[1][mirror]);
      // Actual 806 spawns are not mirror positions. Equal shortest path per
      // pair guarantees equal nearest distance for the union and each type.
      if(distance.every(d=>d[cell]>=0&&d[mirror]>=0)&&a===b)pairs.push([cell,mirror]);
    }
    const placements=[],count=Math.min(Math.floor(OPENING_ITEM_COUNT/2),pairs.length);
    for(let n=0;n<count;n++){
      const pair=pairs.splice(Math.floor(sim.rng()*pairs.length),1)[0],type=n%3;
      for(const cell of pair){sim.crate[cell]=1;sim.crateType[cell]=type;sim.superCrate[cell]=0;placements.push({cell,type});}
    }
    return sim.killOpeningItems={requested:OPENING_ITEM_COUNT,actual:placements.length,
      shortage:OPENING_ITEM_COUNT-placements.length,placements};
  }
  return {OPENING_ITEM_COUNT,apply};
});
