'use strict';
// Read-only replay of the fixed JS natural protocol. Never supplies actions.
const fs=require('fs'),crypto=require('crypto');
const Q=require('../web/sim'),P=require('../web/review_protocol');
const authority=require('./eval_web_hunter'),inference=require('./je05_web_hunter');
const {setup,objective}=require('./je10_route_funnel'),topology=require('./je08_web_diagnostics');
const sha=x=>crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');
async function audit(modelFile,regions,outFile,begin='0',end='32'){
  const loaded=await inference.loadModel(modelFile),tables=JSON.parse(fs.readFileSync(regions)),episodes=[];
  for(let game=Number(begin);game<Number(end);game++){
    const seed=2026102100+game,team=game%2,enemy=1-team;
    const s=setup('natural_full2400',seed,team,tables),tracker=new topology.Tracker(team);
    const rng=Q.mulberry32(seed^0x515154),hunter=authority.createHunter('normal');
    hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:`je10-natural_full2400-${game}`,seed,ruleset:'bun',max_ticks:s.maxSteps,metadata:{}});
    const acquisitions=[],history=[],windowEdges=[],actions=[],deliveries=[];
    let bunStage=null,currentAcquisition=null;
    const nativeBunUpdate=s._bunUpdate;
    s._bunUpdate=function(...args){
      bunStage={alive:Array.from(this.alive),trapped:Array.from(this.trapped),respawn:Array.from(this.bunRespawn),
        carried:Array.from(this.bunCarried),stock:this.bunStored.map(x=>Array.from(x)),loose:Array.from(this.bunLoose)};
      return nativeBunUpdate.apply(this,args);
    };
    let lastWindow=null,lastWindowEnd=null,delivery=0,pickup=0,freeStockApproachTicks=0;
    while(!s.done&&s.t<s.maxSteps){
      const before=P.state(s),b=topology.snapshot(s),tick=s.t;
      const window=!s.alive[enemy]?'DEAD':s.trapped[enemy]>0?'NATIVE_SUGAR':'FREE_ALIVE';
      if(window!==lastWindow){windowEdges.push({tick,from:lastWindow,to:window});if(window==='FREE_ALIVE'&&lastWindow!==null)lastWindowEnd=tick;lastWindow=window;}
      const obj=objective(s,team,tracker);
      const a=await loaded.model.act(s,team,rng),bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,rng);
      const joint=[];joint[team]=[a[0],Number(a[1]),Number(a[2]||0)];joint[enemy]=[bot.move,Number(bot.ability===1),Number(bot.ability===2)];
      const cage=s.bunBases[enemy],pos=[s.pos[2*team],s.pos[2*team+1]],nextDistance=Math.abs(pos[0]-cage[0]-1.5)+Math.abs(pos[1]-cage[1]-1.5);
      const row={tick,window,enemy_alive:s.alive[enemy],enemy_trapped:s.trapped[enemy],enemy_respawn:s.bunRespawn[enemy],pos,enemy_pos:[s.pos[2*enemy],s.pos[2*enemy+1]],carried:s.bunCarried[team],stock:s.bunStored[enemy][enemy],phase:obj?.phase||'COMBAT',target:obj?.target||null,action:joint[team],cage_l1:nextDistance};
      const info=s.step(joint);tracker.observe(s,b,info);
      row.cage_l1_after=Math.abs(s.pos[2*team]-cage[0]-1.5)+Math.abs(s.pos[2*team+1]-cage[1]-1.5);
      if(window==='FREE_ALIVE'&&b.carry[team]<0&&row.cage_l1_after<row.cage_l1-1e-5)freeStockApproachTicks++;
      actions.push([joint[0][0],joint[0][2]?2:joint[0][1]?1:0,joint[1][0],joint[1][2]?2:joint[1][1]?1:0]);
      const picked=b.carry[team]<0&&s.bunCarried[team]===enemy;
      const delivered=s.bunStored[team][enemy]>b.stock[team][enemy];delivery+=Number(delivered);pickup+=Number(picked);
      if(picked){
        const stock=b.stock[enemy][enemy]>s.bunStored[enemy][enemy];
        const looseBefore=before.bunLoose?.reduce((n,v)=>n+v,0),looseAfter=s.bunLoose.reduce((n,v)=>n+v,0);
        acquisitions.push({tick_before:tick,tick_after:s.t,source:stock?'STOCK':looseAfter<looseBefore?'LOOSE':'CONTACT_OR_OTHER',window,
          native_interaction_window:!bunStage.alive[enemy]?'DEAD':bunStage.trapped[enemy]>0?'NATIVE_SUGAR':'FREE_ALIVE',native_bun_stage:bunStage,
          last_window_end:lastWindowEnd,ticks_since_window_end:lastWindowEnd===null?null:tick-lastWindowEnd,
          state_before_sha256:sha(before),state_after_sha256:sha(P.state(s)),state_before:before,state_after:P.state(s),
          info,pre_window:[...history.slice(-64),row],
          geometric_approach_last16:history.slice(-15).concat([row]).filter(x=>x.cage_l1_after<x.cage_l1-1e-5).length,
          note:'Geometric approach is observable; intentionality is not inferred from cage contact alone.'});
        currentAcquisition=acquisitions.length-1;
      }
      if(delivered)deliveries.push({tick:s.t,acquisition_index:currentAcquisition});
      if(s.bunCarried[team]!==enemy)currentAcquisition=null;
      history.push(row);if(history.length>64)history.shift();
    }
    episodes.push({game,seed,team,ticks:s.t,pickup,delivery,window_edges:windowEdges,free_stock_geometric_approach_ticks:freeStockApproachTicks,
      acquisitions,deliveries,actions_sha256:sha(actions),final_state_sha256:sha(P.state(s))});
  }
  fs.writeFileSync(outFile,JSON.stringify({schema:'qqt.stock-window-audit/v1',bot:'JS bun.hunter/normal',model_json_sha256:loaded.sha,episode_range:[Number(begin),Number(end)],assist:false,episodes}));
}
if(require.main===module)audit(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={audit};
