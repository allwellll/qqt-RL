'use strict';
// Actual production stock-window Sim.step and hard Hunter, no policy/teacher.
const fs=require('fs'),crypto=require('crypto'),assert=require('assert');
const Q=require('../web/sim-stock-window-v1'),authority=require('./eval_web_hunter');
const base='runs/bomb_threat_v1/iter2/a3sr',file=base+'/JS_FIXTURES.json';
const fixtures=JSON.parse(fs.readFileSync(file)).episodes;
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function write(p,x){fs.writeFileSync(p+'.tmp',JSON.stringify(x,null,2));fs.renameSync(p+'.tmp',p);}
const rows=[];const began=Date.now();
for(const fixture of fixtures){
 const out=base+`/js_${String(rows.length).padStart(4,'0')}.json`;
 let row;
 if(fs.existsSync(out)){row=JSON.parse(fs.readFileSync(out));assert(row.seed===fixture.seed);}
 else{
  const s=new Q.Sim(fixture.seed);s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
  s.brick.set(fixture.brick);s.pos.set(fixture.cells.flatMap(p=>p.map(x=>x+.5)));s.bunSpawnPos=fixture.cells.map(p=>p.map(x=>x+.5));
  assert(Array.from(s.bunCarried).every(x=>x===-1));
  const hunters=[authority.createHunter('hard'),authority.createHunter('hard')];
  const rng=[Q.mulberry32(fixture.seed^0x515154),Q.mulberry32(fixture.seed^0x7a11c0de)];
  hunters.forEach((h,p)=>h.adapter.reset({schema:'qqt.bot.context/v1',episode_id:`a3sr-${fixture.seed}-${p}`,seed:fixture.seed^p,ruleset:'bun',max_ticks:200,metadata:{}}));
  const own=[0,0],death=[0,0],placed=[0,0],firstSelf=[null,null];let ticks=0;
  for(let t=0;t<200;t++){
   const ac=hunters.map((h,p)=>h.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},p,rng[p]));
   const joint=ac.map(a=>[a.move,Number(a.ability===1),Number(a.ability===2)]);
   const info=s.step(joint);ticks++;
   for(let p=0;p<2;p++){own[p]+=Number(info.ownBombDefeat[p]);death[p]+=Number(info.died[p]);placed[p]+=Number(info.placed[p]);if(info.ownBombDefeat[p]&&firstSelf[p]===null)firstSelf[p]=s.t;}
   if(s.done)break;
  }
  row={seed:fixture.seed,spawn_mode:fixture.spawn_mode,ticks,self_defeats:own,deaths:death,bombs_placed:placed,first_self_defeat:firstSelf,terminal:Boolean(s.done),engine:'production stock-window-v1 Sim.step',opponents:'hard Hunter vs hard Hunter'};
  write(out,row);
 }
 rows.push(row);
 if(rows.length%25===0)write(base+'/JS_AUDIT_PROGRESS.json',{status:'RUNNING',games:rows.length,target:1000,elapsed_seconds:(Date.now()-began)/1000});
}
const groups={};
for(const mode of [0,1]){
 const x=rows.filter(r=>r.spawn_mode===mode);
 groups[mode]={games:x.length,player_episodes:2*x.length,self_defeat_episodes:x.filter(r=>r.self_defeats.some(v=>v>0)).length,self_defeat_player_episodes:x.reduce((n,r)=>n+r.self_defeats.filter(v=>v>0).length,0),self_defeat_events:x.reduce((n,r)=>n+r.self_defeats.reduce((a,b)=>a+b,0),0),ticks:x.reduce((n,r)=>n+r.ticks,0)};
 groups[mode].self_defeat_game_fraction=groups[mode].self_defeat_episodes/x.length;
}
const result={status:'PASS',games:rows.length,window_ticks:200,scope:'first200ticks mechanism safety diagnostic, not full-game capability',fixture_sha256:sha(file),source_sha256:{script:sha(__filename),sim:sha('web/sim-stock-window-v1.js'),hunter:sha('scripts/eval_web_hunter.js')},groups,elapsed_seconds:(Date.now()-began)/1000};
write(base+'/JS_AUDIT.json',result);
console.log(JSON.stringify(result));
