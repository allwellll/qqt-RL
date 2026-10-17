'use strict';
const assert=require('assert'),fs=require('fs');
const V=require('./review_presentation'),P=require('./review_protocol'),visual=require('./visual_renderer');
(async()=>{
  const catalog=JSON.parse(fs.readFileSync('web/review-catalog.json'));
  const keys=new Set();
  for(const row of catalog.replays){
    const doc=JSON.parse(fs.readFileSync(`web/review-replays/${row.file}`));
    for(const key of V.componentNames(doc)){
      keys.add(key);assert(V.reward(key).known,`missing reward translation: ${key}`);
      assert(/[\u4e00-\u9fff]/.test(V.reward(key).name));assert(V.reward(key).help.includes(key));
    }
  }
  assert.equal(keys.size,19);
  assert.equal(V.reward('unknown_reward').name,'未翻译（unknown_reward）');
  assert(!V.reward('__proto__').known);
  assert(V.componentNames({meta:{},rewards:[[1,{}, {opponent_only:1}]]}).includes('opponent_only'));
  const state=()=>({t:1,done:false,pos:[5.5,5.5,8.5,8.5],alive:[true,true],trapped:[0,0],
    bunCarried:[-1,-1],heldItem:[0,0],hp:[1,1],movementStatus:[0,0],slideDir:[4,4],spdG:[1,1],
    spawnProtection:[0,0],invuln:[0,0],bunRespawn:[0,0],wall:new Array(195).fill(0),
    brick:new Array(195).fill(0),pushable:new Array(195).fill(0),fuse:new Array(195).fill(0),
    fieldItem:new Array(195).fill(0),crate:new Array(195).fill(0)});
  const before=state(),after=state();after.t=2;after.pos[1]=5.8;
  const original=JSON.stringify(before);
  assert(V.canInterpolate(before,after,0));
  const motion=V.motion(before,after,.5,true);
  assert.equal(motion.prevPos[1]+(motion.curPos[1]-motion.prevPos[1])*motion.fraction,5.65);
  assert.equal(JSON.stringify(before),original,'presentation must not mutate authoritative state');
  for(const pid of [0,1]){
    const next=structuredClone(before);next.t++;next.pos[pid*2+1]+=.3;
    const coordinates=[0,.13,.27,.41,.59,.73,.87,1].map(alpha=>V.motion(before,next,alpha,true,[],[3,3]).visualPos[pid*2+1]*visual.CELL);
    assert(coordinates.every((v,i)=>!i||v>coordinates[i-1]),'strict monotonic intermediate screen anchors');
    assert(coordinates.some(v=>!Number.isInteger(v)),'float override must reach production draw coordinates');
    assert.deepEqual(V.motion(before,next,.7,false).visualPos,before.pos);
    assert(!V.canInterpolate(before,next,pid,[],0),'collision side steering must snap');
    next.pos[pid*2+1]=before.pos[pid*2+1]+.1;assert(!V.canInterpolate(before,next,pid,[],3),'collision shortened step must snap');
  }
  assert.deepEqual(V.motion(before,after,.5,false).curPos,before.pos);
  assert.deepEqual(motion.playerLabels,['模型 P0','对手 P1']);
  assert.equal(motion.characters[0],'maomao');assert.equal(motion.playerPalettes[1],'blue');
  for(const mutate of [s=>s.alive[0]=false,s=>s.trapped[0]=1,s=>s.bunCarried[0]=1,
    s=>s.heldItem[0]=1,s=>s.hp[0]=0,s=>s.movementStatus[0]=1,s=>s.spawnProtection[0]=1,
    s=>s.pos[1]=7,s=>s.pos[1]=before.pos[1]+V.MAX_DISTANCE+.001,s=>s.t=3,
    s=>s.done=true,s=>s.brick[5*15+5]=1,s=>s.fuse[5*15+5]=30]){
    const bad=structuredClone(after);mutate(bad);assert(!V.canInterpolate(before,bad,0));
  }
  const dead=structuredClone(before);dead.alive[0]=false;assert(!V.canInterpolate(dead,after,0));
  for(const kind of ['death','revive','control','control_end','drop','pickup','native_transfer','delivery','bomb']) {
    assert(!V.canInterpolate(before,after,0,[{tick:2,player:0,kind}]));
    assert.deepEqual(V.motion(before,after,.75,true,[{tick:2,player:0,kind}]).visualPos,before.pos);
  }
  const wall=structuredClone(before),wallNext=structuredClone(after);wall.wall[80]=wallNext.wall[80]=1;
  assert(!V.canInterpolate(wall,wallNext,0),'reject swept solid box');
  const blue=visual.bluePaletteColor(220,20,30);assert(blue[2]>blue[0]*3);
  assert.deepEqual(visual.bluePaletteColor(240,205,180),[240,205,180],'skin tones unchanged');
  assert.deepEqual(visual.bluePaletteColor(255,255,255),[255,255,255]);
  const doc=JSON.parse(fs.readFileSync(`web/review-replays/${catalog.replays[0].file}`));
  const player=new P.Player(doc,require('../scripts/eval_web_hunter').loadLevel());let continuous=0,breaks=0;
  while(player.sim.t<doc.actions.length){const from=P.state(player.sim);player.step();const to=P.state(player.sim);
    for(let p=0;p<2;p++)V.canInterpolate(from,to,p,doc.events)?continuous++:breaks++;
  }
  assert(continuous>20);assert(breaks>0);
  assert.equal(await P.hash(P.state(player.sim)),doc.checks.at(-1)[1],'presentation leaves terminal hash intact');
  console.log(`Review presentation GREEN: ${keys.size} translations, palette, discontinuities, ${continuous} safe / ${breaks} snapped real transitions, unchanged terminal hash`);
})().catch(e=>{console.error(e);process.exitCode=1;});
