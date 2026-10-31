'use strict';
// Read-only causal-rule inspection at first and each newly visited Hunter cell.
// Pure Hunter prediction calls only; never calls its RNG or changes a policy.
const fs=require('fs'),P=require('../web/review_protocol'),A=require('./eval_web_hunter');
const H=require('../web/bun_hunter_bot'),root='runs/bomb_threat_v1/iter2/zero_bomb_audit';
const rows=[];
for(const name of fs.readdirSync(root)){
 const file=`${root}/${name}/review.json`;if(!fs.existsSync(file)||name.startsWith('A2'))continue;
 const d=JSON.parse(fs.readFileSync(file)),player=new P.Player(d,A.loadLevel()),bot=new H.BunHunterBot({difficulty:'normal'}),samples=[],seen=new Set(),maskCounts=[0,0];
 for(let t=0;t<d.actions.length;t++){
  const s=player.sim,mask=s.legalMask();for(let p=0;p<2;p++)maskCounts[p]+=Number(mask.bm[p][1]);
  const key=s.centerCell(1).join(',')+':'+Array.from(s.brick).join('');
  if(!seen.has(key)){
   seen.add(key);const state=H.hunterStateFromSim(s),pid=1,g=bot.geometry(state),me=state.players[pid],foe=bot.foeOf(state,pid);
   const perceive=b=>b.owner===pid||b.fuse<=29,pred=bot.predict(state,g,[],perceive),goal=bot.chooseGoal(state,g,pid,pred,perceive),field=bot.fieldForGoal(state,g,goal,me,pred,foe);
   const dig=bot.digTarget(state,g,me,field,pred),near=foe.alive&&!(foe.trapped>0)&&Math.abs(me.row-foe.row)+Math.abs(me.col-foe.col)<=bot.cfg.attackRadius&&foe.invuln<H.NEW_BOMB_TICK;
   const predB=bot.predict(state,g,[{cell:me.cell,e:H.NEW_BOMB_TICK,blast:me.blast}],perceive),esc=bot.escape(g,predB,me,false);
   samples.push({tick:t,cell:me.cell,goal:goal.mode,dig_target:dig,foe_near:near,escape_survivor_mask:esc.surv,
    bomb_placement_allowed:me.carrying<0&&me.bombsLeft>0&&g.bombAt[me.cell]<0&&!state.brick[me.cell],
    first_rejection:me.carrying>=0?'carrying':me.bombsLeft<=0?'capacity':g.bombAt[me.cell]>=0?'existing_bomb':state.brick[me.cell]?'brick':dig<0&&!near?'no_dig_or_attack_goal':!esc.surv?'no_surviving_escape':'later_conditions_not_classified'});
  }player.step();
 }
 rows.push({name,legal_bomb_ticks:maskCounts,sample_scope:'first visit per Hunter cell and brick layout; all ticks legal-mask counted',samples});
}
fs.writeFileSync(`${root}/GEOMETRY.json`,JSON.stringify({rows},null,2));
console.log(JSON.stringify(rows.map(r=>({name:r.name,legal_bomb_ticks:r.legal_bomb_ticks,samples:r.samples}))));
