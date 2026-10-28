'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const Q=require('../web/sim.js'),P=require('../web/review_protocol.js');
const authority=require('./eval_web_hunter.js'),inference=require('./je05_web_hunter.js');
const topology=require('./je08_web_diagnostics.js'),residual=require('./eval_web_hunter_residual.js');
const {ControlFunnel}=require('./je10_control_funnel');
const {ItemMetrics}=require('./dagger_rl_metrics');
const {StockAcquisitionFunnel}=require('./stock_acquisition_funnel');
const DIRS=[[-1,0],[1,0],[0,-1],[0,1],[0,0]],INF=32767;
const GAMES=Number(process.env.JE10_ROUTE_GAMES||16),SEED_START=Number(process.env.JE10_ROUTE_SEED_START||2026101800);
if(!Number.isInteger(GAMES)||GAMES<1||!Number.isInteger(SEED_START))throw Error('invalid route evaluation protocol');
const GAME_START=Number(process.env.JE10_ROUTE_START||0),GAME_END=Number(process.env.JE10_ROUTE_END||GAMES);
if(!Number.isInteger(GAME_START)||!Number.isInteger(GAME_END)||GAME_START<0||GAME_END>GAMES||GAME_START>=GAME_END)throw Error('invalid route shard range');
const TASKS=['native_death_stock','native_sugar_stock','real_drop','carry_near','carry_mid','carry_far','recovery_threat','natural_full2400'];
const sha=v=>crypto.createHash('sha256').update(v).digest('hex');
function cage(s,t){const b=s.bunBases[t],a=[];for(let r=b[0];r<b[0]+3;r++)for(let c=b[1];c<b[1]+3;c++)a.push(r*15+c);return a;}
function setup(task,seed,team,tables){
  if(['carry_near','carry_mid','carry_far'].includes(task))return topology.setup(task,seed,team,tables[1],1200);
  const s=new Q.Sim(seed);s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});s.maxSteps=task==='natural_full2400'?2400:1200;
  if(task==='natural_full2400')return s;
  s.initialHp=1;s.hp=[1,1];s.invuln=[0,0];
  if(seed%2===1)s.brick.fill(0);
  // Frozen reset fixtures create legal windows, never post-reset edits.
  if(task==='real_drop')return topology.setup('drop_return',seed,team,null,1200);
  const enemy=1-team,base=s.bunBases[enemy],goals=cage(s,enemy);
  const blocked=Array.from(s.wall,(v,i)=>Number(Boolean(v||s.brick[i]))),d=topology.distance(blocked,new Uint8Array(195),goals);
  const wanted=[2,8,16][seed%3],cells=d.map((v,i)=>v>=wanted&&v<wanted+3&&!blocked[i]?i:-1).filter(i=>i>=0);
  const reachable=d.map((v,i)=>v>0&&v<INF&&!blocked[i]?i:-1).filter(i=>i>=0);
  if(!reachable.length)throw Error('stock fixture has no legal noncage reachable start');
  const closest=reachable.reduce((best,i)=>Math.min(best,Math.abs(d[i]-wanted)),INF);
  const candidates=cells.length?cells:reachable.filter(i=>Math.abs(d[i]-wanted)===closest);
  const at=candidates[seed%candidates.length];s.pos[team*2]=Math.floor(at/15)+.5;s.pos[team*2+1]=at%15+.5;
  if(task==='native_sugar_stock'||task==='recovery_threat')s.trapped[enemy]=task==='recovery_threat'?8:60;
  else{s.alive[enemy]=false;s.hp[enemy]=0;s.bunRespawn[enemy]=100;}
  s.bunSpawnPos=[[s.pos[0],s.pos[1]],[s.pos[2],s.pos[3]]];s._gen++;return s;
}
function continuous(pos,d,blocked){let best=INF;const r=Math.floor(pos[0]),c=Math.floor(pos[1]);
  for(const[dr,dc]of DIRS){const rr=r+dr,cc=c+dc,i=rr*15+cc;if(rr<0||rr>=13||cc<0||cc>=15||blocked[i]||d[i]>=INF)continue;best=Math.min(best,d[i]+Math.abs(pos[0]-rr-.5)+Math.abs(pos[1]-cc-.5));}return best;}
function objective(s,team,tracker){
  const enemy=1-team,carry=s.bunCarried[team]===enemy,window=!s.alive[enemy]||s.trapped[enemy]>0;
  const foeDistance=Math.abs(s.pos[team*2]-s.pos[enemy*2])+Math.abs(s.pos[team*2+1]-s.pos[enemy*2+1]);
  if(!s.alive[team]||s.trapped[team]>0||(!window&&!carry))return null;
  let goals,phase,target;
  if(carry){goals=cage(s,team);phase='C_RETURN';target={type:'home_cage',team};}
  else if(window&&tracker.target[enemy]!==null&&s.bunLoose[tracker.target[enemy]*2+enemy]>0&&tracker.credit[enemy]===team){goals=[tracker.target[enemy]];phase='A_ENEMY_DROP';target={type:'real_loose',origin:enemy,cell:goals[0],nonce:tracker.nonce[enemy]};}
  else if(window&&s.bunStored[enemy][enemy]>0){goals=cage(s,enemy);phase='B_STOCK';target={type:'enemy_cage',team:enemy};}
  else return null;
  const blocked=Array.from(s.wall,(v,i)=>Number(Boolean(v||s.brick[i]||s.pushable?.[i]))),d=topology.distance(blocked,new Uint8Array(195),goals);
  const pos=[s.pos[team*2],s.pos[team*2+1]],distance=continuous(pos,d,blocked),mask=s.legalMask().mm[team];
  const gain=DIRS.map(([dr,dc],move)=>{const rr=Math.floor(pos[0])+dr,cc=Math.floor(pos[1])+dc,i=rr*15+cc;
    if(!mask[move]||rr<0||rr>=13||cc<0||cc>=15||blocked[i]||s.fuse[i]>0||s.blastLinger[i]>0)return -INF;
    return distance-continuous([pos[0]+.18*dr,pos[1]+.18*dc],d,blocked);});
  const best=Math.max(...gain),oracle=gain.map((v,i)=>v>1e-5&&v>=best-1e-5?i:-1).filter(i=>i>=0);
  if(d[Math.floor(pos[0])*15+Math.floor(pos[1])]===0&&mask[4])oracle.push(4);
  return {phase,target,distance,oracle,phi:distance>=INF?-1:-Math.min(distance,64)/64};
}
async function run(modelFile,outFile,regions){
  const loaded=await inference.loadModel(modelFile),tables=JSON.parse(fs.readFileSync(regions)),results={},out=path.dirname(outFile);

  // Observe the existing forward/act calls; no extra inference or RNG draws.
  const originalForward=loaded.model.forward.bind(loaded.model);let lastLogits=null;
  loaded.model.forward=async(...args)=>{const z=await originalForward(...args);lastLogits=z;return z;};
  const originalAct=loaded.model.act.bind(loaded.model);
  loaded.model.act=async(s,pid,rng)=>{const result=await originalAct(s,pid,rng),mask=loaded.model._abilityMask(s,s.legalMask(),pid),z=Array.from(lastLogits.bomb),mx=Math.max(...z.filter((_,i)=>mask[i]));
    const w=z.map((v,i)=>mask[i]?Math.exp(v-mx):0),den=w.reduce((a,b)=>a+b,0),probabilities=w.map(v=>v/den);let greedy=0;for(let i=1;i<3;i++)if(mask[i]&&(!mask[greedy]||z[i]>z[greedy]))greedy=i;
    s._daggerItemDecision=s._daggerItemDecision||{};s._daggerItemDecision[pid]={legal:Array.from(mask),logits:z,probabilities,greedy};return result;};
  const selected=process.env.JE10_ROUTE_TASK?TASKS.filter(t=>t===process.env.JE10_ROUTE_TASK):TASKS;
  if(!selected.length)throw Error('unknown fixed route task');
  for(const task of selected){const episodes=[];
    for(let game=GAME_START;game<GAME_END;game++){
      const seed=SEED_START+game,team=game%2,enemy=1-team,s=setup(task,seed,team,tables),initial=P.state(s),tracker=new topology.Tracker(team);
      if(task==='real_drop')tracker.trapCause[enemy][team]=true;
      const rng=Q.mulberry32(seed^0x515154),hunter=authority.createHunter('normal');
      hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:`je10-${task}-${game}`,seed,ruleset:'bun',max_ticks:s.maxSteps,metadata:{}});
      const stageCounts={},firstByPhase={},itemMetrics=new ItemMetrics(s,team);
      const m={window_events:Number(!s.alive[enemy]||s.trapped[enemy]>0),target_events:0,first_oracle_actions:0,first_oracle_matches:0,window_ticks:0,target_ticks:0,oracle_opportunities:0,oracle_matches:0,first_oracle_match:0,progress_ticks:0,net_progress:0,pickup:0,return_started:0,delivery:0,recovery_switch_opportunities:0,recovery_switches:0,death:0,self_death:0,first_control:0,attributed_control:0,final_pop:0,surviving_kill:0};
      const control=new ControlFunnel(team);
      const acquisitionFunnel=new StockAcquisitionFunnel(team);
      let nativeBunStage=null;
      const nativeBunUpdate=s._bunUpdate;
      s._bunUpdate=function(...args){
        nativeBunStage={alive:Array.from(this.alive),trapped:Array.from(this.trapped),
          carried:Array.from(this.bunCarried),stock:this.bunStored.map(row=>Array.from(row))};
        return nativeBunUpdate.apply(this,args);
      };
      const ticks=[],actions=[],checks=[[0,sha(JSON.stringify(initial))]],events=[];
      let first=null,seenOracle=false,previousWindow=false,previousObjective=null,decisionTarget=null,phaseNeedsFirst=false,phi=0,offset=0;
      const limit=s.maxSteps;
      while(!s.done&&s.t<limit){
        const b=topology.snapshot(s),window=!s.alive[enemy]||s.trapped[enemy]>0,obj=objective(s,team,tracker),beforeState=game===0?P.state(s):null;
        const a=await loaded.model.act(s,team,rng),bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,rng);
        const joint=[];joint[team]=[a[0],Number(a[1]),Number(a[2]||0)];joint[enemy]=[bot.move,Number(bot.ability===1),Number(bot.ability===2)];
        m.window_ticks+=Number(window);m.target_ticks+=Number(!!obj);
        if(window&&!previousWindow&&s.t>0)m.window_events++;
        const decisionKey=obj?JSON.stringify([obj.phase,obj.target]):null;
        if(decisionKey!==decisionTarget){phaseNeedsFirst=!!obj;if(obj)m.target_events++;decisionTarget=decisionKey;}
        if(obj){const st=stageCounts[obj.phase]||(stageCounts[obj.phase]={target_ticks:0,oracle_opportunities:0,oracle_matches:0,first_actions:0,first_matches:0,progress_ticks:0});st.target_ticks++;
          if(obj.oracle.length){const match=obj.oracle.includes(a[0]);st.oracle_opportunities++;st.oracle_matches+=Number(match);
            if(phaseNeedsFirst){st.first_actions++;st.first_matches+=Number(match);m.first_oracle_actions++;m.first_oracle_matches+=Number(match);phaseNeedsFirst=false;}
            if(!match&&!firstByPhase[obj.phase])firstByPhase[obj.phase]={tick:s.t,action:joint[team],oracle:obj.oracle,phase:obj.phase,target:obj.target,distance:obj.distance,state_hash:sha(JSON.stringify(beforeState||P.state(s)))};
          }
        }

        if(obj?.oracle.length){m.oracle_opportunities++;const match=obj.oracle.includes(a[0]);m.oracle_matches+=Number(match);if(!seenOracle){m.first_oracle_match=Number(match);seenOracle=true;}if(!match&&first===null)first={tick:s.t,action:joint[team],oracle:obj.oracle,phase:obj.phase,target:obj.target,distance:obj.distance,state_hash:sha(JSON.stringify(beforeState||P.state(s)))};}
        if(previousWindow&&!window){m.recovery_switch_opportunities++;m.recovery_switches+=Number(!obj);}
        itemMetrics.before(joint[team]);const info=s.step(joint);itemMetrics.after();control.observe({trapped:b.trap,alive:b.alive},s,info);tracker.observe(s,b,info);const next=objective(s,team,tracker);
        acquisitionFunnel.observe(b,s,s.t,nativeBunStage);
        let route=0;
        if(obj){if(previousObjective!==JSON.stringify(obj.target)){offset=phi-obj.phi;}const pb=obj.phi+offset;const pa=s.done?0:next&&JSON.stringify(next.target)===JSON.stringify(obj.target)?next.phi+offset:pb;route=.5*(.995*pa-pb);phi=pa;previousObjective=JSON.stringify(obj.target);}
        const gain=obj&&next&&JSON.stringify(obj.target)===JSON.stringify(next.target)?obj.distance-next.distance:0;
        m.progress_ticks+=Number(gain>1e-5);m.net_progress+=gain;if(obj)stageCounts[obj.phase].progress_ticks+=Number(gain>1e-5);
        const picked=b.carry[team]<0&&s.bunCarried[team]===enemy;
        m.pickup+=Number(picked);m.return_started+=Number(picked&&next?.phase==='C_RETURN');
        m.delivery+=Number(s.bunStored[team][enemy]>b.stock[team][enemy]);
        m.death+=Number(info.died[team]);m.self_death+=Number(info.ownBombDefeat?.[team]);
        const entry=b.trap[enemy]===0&&s.trapped[enemy]>0;
        m.first_control+=Number(entry);const causal=entry&&info.causalDamageSource?.[enemy]?.[team]&&!info.causalDamageSource?.[enemy]?.[enemy];m.attributed_control+=Number(causal);
        m.final_pop+=Number(b.trap[enemy]>0&&info.died[enemy]);m.surviving_kill=control.metrics.surviving_kill;
        if(game===0){
          const row=[joint[0][0],joint[0][2]?2:joint[0][1]?1:0,joint[1][0],joint[1][2]?2:joint[1][1]?1:0];actions.push(row);
          if(entry||info.died.some(Boolean)||picked||m.delivery)events.push({tick:s.t,kind:entry?'control_entry':picked?'pickup':info.died.some(Boolean)?'death':'delivery',player:team});
          ticks.push({tick:s.t,state_hash:sha(JSON.stringify(P.state(s))),action:row,events:{control_entry:entry,death:info.died,pickup:picked},phase:obj?.phase||'COMBAT',target:obj?.target||null,oracle:obj?.oracle||[],distance:obj?.distance??null,phi,
            state_before_sha256:sha(JSON.stringify(beforeState)),state_after_sha256:sha(JSON.stringify(P.state(s))),control_event_id:control.event,control_owner:control.owner,approximate_reward_components:{control_entry:causal?4:0,route_observable_reconstruction:route},
            approximation:'observable JS reconstruction lacks checkpointed training aux/safety certificates; not recorded training reward or counterfactual Q'});
          if(s.t%200===0||s.done)checks.push([s.t,sha(JSON.stringify(P.state(s)))]);
        }
        previousWindow=window;
      }
      const ep={...control.metrics,game,seed,team,geometry:initial.brick.some(Boolean)?'native_bricks':'open_bricks',ticks:s.t,...m,...acquisitionFunnel.metrics,first_deviation:first,first_deviations_by_phase:firstByPhase,stage_funnel:stageCounts,item_metrics:itemMetrics.rows,oracle_agreement:m.oracle_matches/Math.max(1,m.oracle_opportunities),progress_per_target_tick:m.progress_ticks/Math.max(1,m.target_ticks),final_alive:s.alive[team],fixture:task!=='natural_full2400'};episodes.push(ep);
      if(game===0){
        let gae=0;for(let i=ticks.length-1;i>=0;i--){const r=ticks[i].approximate_reward_components;gae=r.control_entry+r.route_observable_reconstruction+.995*.95*gae;ticks[i].approximate_reward_gae=gae;}
        fs.writeFileSync(path.join(out,`fixed-${task}-${seed}.trace.json`),JSON.stringify({task,seed,initial,actions,diagnostic_ticks:ticks,first_deviation:first}));
        {
          const level=authority.loadLevel(),modelDoc=JSON.parse(fs.readFileSync(modelFile));
          const meta={id:`je10-${modelDoc.meta.checkpoint_sha256.slice(0,10)}-${task}-${seed}`,seed,tick_ms:100,max_ticks:limit,scene:'native',policy:'greedy',diagnostic:true,
            checkpoint_sha256:modelDoc.meta.checkpoint_sha256,model_sha256:loaded.sha,simulator_sha256:sha(fs.readFileSync('web/sim.js')),opponent_sha256:sha(fs.readFileSync('web/bun_hunter_bot.js')),map_sha256:await P.hash(level),reward_status:'unrecorded',training_reward_status:'unrecorded',opponent:'bun.hunter/normal'};
          if(task!=='natural_full2400')meta.native_fixture={schema:'qqt.native-reset-fixture/v1',task,initial_hp:1,frame:initial,frame_sha256:await P.hash(initial)};
          const replay={schema:P.SCHEMA,meta,actions,rewards:[],events,checks,diagnostic_ticks:ticks,summary:{ticks:s.t,done:s.done,winner:s.winner,reward:[null,null],failures:first?['route_deviation']:[]}};
          meta.replay_identity=await P.hash(P.identity(replay));P.validate(replay);await new P.Player(replay,level).verify();
          fs.writeFileSync(path.join(out,`REVIEW_FIXED_${task}.json`),JSON.stringify(replay));
          if(task==='natural_full2400')fs.writeFileSync(path.join(out,'REVIEW_FIXED_NATIVE.json'),JSON.stringify(replay));
        }
      }
    }
    const sums=Object.fromEntries(Object.keys(episodes[0]).filter(k=>typeof episodes[0][k]==='number'&&!['game','seed','team'].includes(k)).map(k=>[k,episodes.reduce((a,e)=>a+e[k],0)]));
    const stage_funnel={};for(const ep of episodes)for(const [phase,counts]of Object.entries(ep.stage_funnel)){const out=stage_funnel[phase]||(stage_funnel[phase]={});for(const [key,val]of Object.entries(counts))out[key]=(out[key]||0)+val;}
    const acquisition_scope=['carry_near','carry_mid','carry_far'].includes(task)?'diagnostic_only_synthetic_carrier_reset_no_real_pickup':task==='natural_full2400'?'native_full_actual_acquisition_chain':'controlled_fixture_real_native_pickup_not_proof_of_inbound_from_home';
    results[task]={acquisition_scope,training_eligible:false,proves_post_pickup_unreachability:false,games:GAME_END-GAME_START,...sums,stage_funnel,pickup_episodes:episodes.filter(e=>e.pickup>0).length,delivery_episodes:episodes.filter(e=>e.delivery>0).length,oracle_opportunity_episodes:episodes.filter(e=>e.oracle_opportunities>0).length,first_oracle_match_episodes:episodes.filter(e=>e.first_oracle_match>0).length,episodes};
    fs.writeFileSync(outFile+'.partial.json',JSON.stringify({results}));
  }
  fs.writeFileSync(outFile+'.tmp',JSON.stringify({passed:true,model_json_sha256:loaded.sha,assist:false,seeds:Array.from({length:GAME_END-GAME_START},(_,i)=>SEED_START+GAME_START+i),results,
    oracle_scope:'observable legal continuous direction; production training also uses temporal safety verifier; report differences explicitly',review:'REVIEW_FIXED_<task>.json uses sole production Review schema, declared reset fixture and exact replay/hash verification; reward-GAE is approximate'}));fs.renameSync(outFile+'.tmp',outFile);
}
if(require.main===module)run(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={continuous,objective,setup,TASKS};
