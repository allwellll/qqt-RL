// 实验中 / EXPERIMENTAL: opt-in only; not the default web or training path.
// Independent experimental kill arena. Production sim/v2 remain unchanged.
(function(root){
 'use strict';
 const base=typeof module==='object'?require('./sim-bun-native-v2'):root.QQT;
 const VERSION='kill-arena-sugar-respawn-v1',H=13,W=15,N=195;
 function spawnOrder(seed,t,p,count){return Math.imul((seed^t^Math.imul(p,0x9e3779b9))>>>0,0x85ebca6b)>>>0;}
 function safeSpawns(s,p){
  const q=1-p,[er,ec]=s.centerCell(q),danger=s.dangerMap();
  return s.arenaSpawns.map((a,i)=>({a,i})).filter(({a:[r,c]})=>{
   const at=r*W+c;return !s.wall[at]&&!s.brick[at]&&!s.pushable[at]&&!s.fuse[at]&&!s.blastLinger[at]&&danger[at]===0&&
    (!s.alive[q]||Math.abs(r-er)+Math.abs(c-ec)>=3);
  });
 }
 class Sim extends base.Sim{
  resetArena(level,options={}){
   const copy=JSON.parse(JSON.stringify(level));delete copy.bun;delete copy.native_rule;delete copy.game_mode;
   // Keep every native spawn available, with identical map sanitation in JAX.
   for(const [r,c] of copy.spawns){copy.wall[r*W+c]=0;copy.brick[r*W+c]=0;}
   if(options.clearBricks)copy.brick.fill(0);
   if(options.itemsDisabled){copy.no_items=true;copy.item_field=0;copy.initial_crates=[];}
   super.reset(copy,{nativeItems:true,nativeTrap:true,playerModes:['new','new']});
   this.isBun=false;this.bunBases=[];this.bunStored=[[0,0],[0,0]];this.bunLoose.fill(0);this.bunCarried=[-1,-1];
   this.maxSteps=1200;this.arenaSpawns=copy.spawns.map(x=>x.slice());
   this.arenaMapId=level.source_id??level.id;this.arenaMapName=level.name;
   const st=level.initial_stats;this.arenaInitial=[st.bombs,st.blast,st.speed];
   for(let p=0;p<2;p++){
    this.bombsCap[p]=this.loBombs[p]=st.bombs;this.blastCap[p]=this.loBlast[p]=st.blast;
    this.spdG[p]=this.loSpeed[p]=st.speed;this.hp[p]=1;this.invuln[p]=10;this.spawnProtection[p]=10;
   }
   if(options.spawnPair){
    if(JSON.stringify(options.spawnPair[0])===JSON.stringify(options.spawnPair[1]))throw Error('duplicate spawns');
    for(let p=0;p<2;p++)for(let k=0;k<2;k++)this.pos[p*2+k]=options.spawnPair[p][k]+.5;
   }
   this.arenaKills=[0,0];this.arenaDeaths=[0,0];this.arenaSelf=[0,0];
   this.arenaLifeAge=[0,0];this.arenaLifeSpawn=[this.centerCell(0),this.centerCell(1)];
   this.arenaTrapSources=[0,0];this.arenaEvents=[];this.arenaLastExplosion=null;
   this.arenaItemsDisabled=Boolean(options.itemsDisabled);this._gen++;return this;
  }
  _resolveExplosions(...args){const r=super._resolveExplosions(...args);this.arenaLastExplosion=r;return r;}
  _trapStep(newly){
   if(this.arenaKills)for(let p=0;p<2;p++)if(newly[p]){
    const cells=this._explosionContactCells(p);let mask=0;
    for(const s of this.arenaLastExplosion?.sources||[])if(cells.some(i=>s.covered[i]))mask|=s.causeMask??(1<<s.owner);
    this.arenaTrapSources[p]=mask;
   }
   return super._trapStep(newly);
  }
  _killPlayer(p){
   if(!this.arenaKills)return super._killPlayer(p);
   if(!this.alive[p])return;
   const q=1-p,near=this.alive[q]&&this.trapped[q]===0&&
    Math.abs(this.pos[2*p]-this.pos[2*q])<41/40&&Math.abs(this.pos[2*p+1]-this.pos[2*q+1])<41/40;
   const mask=near?1<<q:this.arenaTrapSources[p];
   const enemy=Boolean(mask&(1<<q)),self=Boolean(mask&(1<<p))&&!enemy;
   const [r,c]=this.centerCell(p),[sr,sc]=this.arenaLifeSpawn[p];
   const protectedKill=this.invuln[p]>0||this.arenaLifeAge[p]<15||Math.abs(r-sr)+Math.abs(c-sc)<=2;
   this.arenaDeaths[p]++;if(enemy)this.arenaKills[q]++;if(self)this.arenaSelf[p]++;
   this.arenaEvents.push({tick:this.t,type:'death',victim:p,attacker:enemy?q:self?p:-1,self,enemy,
    reward_eligible:!protectedKill,life_age:this.arenaLifeAge[p],early_30:this.arenaLifeAge[p]<30,
    source_mask:mask,spawn_distance:Math.abs(r-sr)+Math.abs(c-sc)});
   return super._killPlayer(p);
  }
  _arenaRespawn(){
   for(let p=0;p<2;p++)if(!this.alive[p]){
    const valid=safeSpawns(this,p),n=this.arenaSpawns.length;
    if(!valid.length)continue;
    const start=spawnOrder(this.seed,this.t,p,n)%n;
    valid.sort((a,b)=>(a.i-start+n)%n-(b.i-start+n)%n);const at=valid[0].a;
    this.pos[p*2]=at[0]+.5;this.pos[p*2+1]=at[1]+.5;this.hp[p]=1;this.alive[p]=true;
    this.invuln[p]=10;this.spawnProtection[p]=10;this.trapped[p]=0;this.arenaLifeAge[p]=0;
    this.arenaLifeSpawn[p]=at.slice();this.arenaTrapSources[p]=0;
    this.bombsCap[p]=this.loBombs[p];this.blastCap[p]=this.loBlast[p];this.spdG[p]=this.loSpeed[p];
    this.itemSlots[p]=[];this._syncHeldItem(p);this._clearMovementStatus(p);
    if(this.nativeMove?.[p]){this.nativeMove[p].wallExit=null;this.nativeMove[p].remainder=0;}
    this.arenaEvents.push({tick:this.t,type:'respawn',player:p,cell:at.slice(),invuln_ticks:10});
   }
  }
  step(actions){
   if(!this.arenaKills)return super.step(actions);
   if(this.done)return {died:[false,false],placed:[false,false],arena_done:true};
   this._arenaRespawn();this.arenaLifeAge=this.arenaLifeAge.map((age,p)=>age+Number(this.alive[p]));
   const oldEvents=this.arenaEvents.length,r=super.step(actions);
   this.done=this.t>=1200;this.winner=this.done?(this.arenaKills[0]===this.arenaKills[1]?null:this.arenaKills[0]>this.arenaKills[1]?0:1):null;
   // Disable random graveyard/airdrop only for declared deterministic parity.
   if(this.arenaItemsDisabled){this.crate.fill(0);this.pendingCrateType.fill(-1);this.graveyard=[];this._airdropQueue=[];this.airdropFalls=[];this._pendingDeathDrops=[];}
   r.arena_events=this.arenaEvents.slice(oldEvents);r.arena_kills=this.arenaKills.slice();r.arena_done=this.done;
   return r;
  }
 }
 const api={...base,Sim,RULE_VERSION:VERSION,safeSpawns,spawnOrder};
 if(typeof module==='object')module.exports=api;else root.QQTKillArena=api;
})(typeof globalThis!=='undefined'?globalThis:this);
