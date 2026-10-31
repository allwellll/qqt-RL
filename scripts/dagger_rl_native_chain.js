'use strict';
// Bounded mechanism branches only. Never used as training or policy-Q evidence.
const fs=require('fs'),path=require('path'),v8=require('v8'),crypto=require('crypto');
const assert=require('node:assert/strict'),cp=require('child_process'),readline=require('readline');
const bank=require('./dagger_rl_js_bank'),inference=require('./je05_web_hunter');
const topology=require('./je08_web_diagnostics'),route=require('./dagger_rl_route');
const OPP=[1,0,3,2],hash=x=>crypto.createHash('sha256').update(v8.serialize(x)).digest('hex');

function publicRequest(s,m,model){
  const p=m.p,e=1-p,order=[p,e],grid=x=>Array.from(x),pair=x=>order.map(i=>x[i]);
  const mask=s.legalMask(),ability=order.map(i=>Array.from(model._abilityMask(s,mask,i)));
  const b=s.bunBases[p],goals=[];
  for(let r=b[0];r<b[0]+3;r++)for(let c=b[1];c<b[1]+3;c++)goals.push(r*15+c);
  const blocked=Array.from(s.wall,(v,i)=>Number(Boolean(v||s.brick[i])));
  return {state:{pos:order.map(i=>[s.pos[i*2],s.pos[i*2+1]]),wall:grid(s.wall),brick:grid(s.brick),
    pushable:grid(s.pushable),blast_linger:grid(s.blastLinger),fuse:grid(s.fuse),alive:pair(s.alive),
    bun_respawn:pair(s.bunRespawn),bun_spawn_pos:pair(s.bunSpawnPos),trapped:pair(s.trapped),
    field_item:grid(s.fieldItem),field_owner:Array.from(s.fieldOwner,x=>x<0?-1:x===p?0:1),field_armed:grid(s.fieldArmed),
    bun_carried:pair(s.bunCarried).map(x=>x<0?-1:x===p?0:1),move_status:pair(s.movementStatus),held_item:pair(s.heldItem)},
    distance:topology.distance(blocked,new Uint8Array(195),goals),real_custody:s.bunCarried[p]===e,
    terminal:s.done,move_mask:order.map(i=>Array.from(mask.mm[i])),ability_mask:ability};
}

function oracleService(logFile){
  const stderr=fs.openSync(logFile,'a');
  const child=cp.spawn('.venv/bin/python',['-m','scripts.dagger_rl_native_teacher'],{
    env:{...process.env,CUDA_VISIBLE_DEVICES:'',JAX_PLATFORMS:'cpu',JAX_PLATFORM_NAME:'cpu',OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1'},
    stdio:['pipe','pipe',stderr]});fs.closeSync(stderr);
  const lines=readline.createInterface({input:child.stdout});let waiting=null;
  lines.on('line',line=>{if(!waiting)return;const callback=waiting;waiting=null;
    try{const r=JSON.parse(line);if(r.error)throw Error(r.error);callback.resolve(r);}catch(e){callback.reject(e);}});
  child.on('exit',code=>{if(waiting){waiting.reject(Error(`oracle exit ${code}`));waiting=null;}});
  return {query:payload=>new Promise((resolve,reject)=>{assert.equal(waiting,null);waiting={resolve,reject};child.stdin.write(JSON.stringify(payload)+'\n');}),
    close:()=>{child.stdin.end();lines.close();}};
}

function accepted(response,action){return !!response.mask[action[0]]?.[action[2]?2:action[1]?1:0];}
function pickPlan(response){
  const acceptedPlans=response.plans.filter(x=>x.valid&&response.mask[x.first]?.[response.diagnostic[1]?0:2]);
  if(!acceptedPlans.length)return null;
  const best=Math.max(...acceptedPlans.map(x=>x.gain));
  return {...acceptedPlans.find(x=>x.gain>=best-1e-5),field:response.field,release:!response.diagnostic[1]};
}

async function run(bankFile,modelFile,outFile){
  const record=v8.deserialize(fs.readFileSync(bankFile));
  assert.ok(record.next_step_parity&&record.causal_proof.actual_acquisition&&!record.causal_proof.manual_carry_edit);
  const loaded=await inference.loadModel(modelFile),model=loaded.model;
  assert.equal(model.greedy,true);assert.equal(model.inferEvery,1);
  let lastLogits=null;const forward=model.forward.bind(model);
  model.forward=async(...args)=>{const z=await forward(...args);lastLogits=z;return z;};
  const oracle=oracleService(outFile+'.oracle.log'),results=[];
  try{
    const initial=bank.restoreInstrumented(record.snapshot);
    const firstTeacher=await oracle.query(publicRequest(initial.sim,initial.metrics,model));
    const plan=pickPlan(firstTeacher);
    for(const arm of ['committed_plan','requery_teacher','student']){
      const {sim:s,metrics:m}=bank.restoreInstrumented(record.snapshot),p=m.p,e=1-p;
      const rng=bank.statefulRng(0);rng.restore(record.policy_rng_after_decisions);
      const origin=m.distance(),stock=s.bunStored[p][e],trace=[];
      let contact=!!m.ownedSlide,completed=false,firstDivergence=null,death=false,drop=false,delivery=false,stoppedReason=null;
      if(arm==='committed_plan'&&!plan){results.push({arm,abstained:true,reason:'no accepted original forecast plan at this real student state',executed_physics_steps:0});continue;}
      for(let tick=0;tick<128&&!s.done;tick++){
        const observation=publicRequest(s,m,model),teacher=await oracle.query(observation);
        const student=await model.act(s,p,rng);let action=Array.from(student);
        if(arm==='requery_teacher'){
          const choices=[];teacher.mask.forEach((row,move)=>row.forEach((yes,ability)=>{if(yes)choices.push([move,Number(ability===1),Number(ability===2)]);}));
          if(choices.length)action=choices[0];
          else{
            // This is explicitly the native observable route fallback, not the
            // JAX temporal-safety verifier. No source state enters optimization.
            const objective=route.objective(s,p,{target:[null,null],credit:[null,null]});
            if(objective?.oracle.length)action=[objective.oracle[0],0,0];
            else action=[4,0,0];
          }
        }else if(arm==='committed_plan'){
          const cell=plan.field[0]*15+plan.field[1],armed=Boolean(s.fieldArmed[cell]);
          action=[!armed&&!contact?OPP[plan.direction]:plan.direction,0,Number(tick===0&&plan.release)];
        }
        const mask=model._abilityMask(s,s.legalMask(),p),ability=action[2]?2:action[1]?1:0;
        const requestedAction=Array.from(action);
        if(!s.legalMask().mm[p][action[0]]&&s.movementStatus[p]===2)action[0]=4;
        if(!s.legalMask().mm[p][action[0]]||!mask[ability]){
          assert.equal(arm,'committed_plan','student/requeried teacher selected illegal action');
          stoppedReason='original forecast requests an illegal native continuation';break;
        }
        const scores=Array.from(lastLogits.bomb),mx=Math.max(...scores.filter((_,i)=>mask[i]));
        const weights=scores.map((x,i)=>mask[i]?Math.exp(x-mx):0),den=weights.reduce((a,b)=>a+b,0);
        const before=bank.captureInstrumented(s,m),priorSlide=s.movementStatus[p]===2;
        const row={tick:s.t,state_hash:hash(before.fields),rng:before.rng,action,requested_action:requestedAction,student_action:student,
          item_teacher_mask:teacher.mask,item_teacher_diagnostic:teacher.diagnostic,
          teacher_plans:teacher.plans,student_in_item_set:accepted(teacher,student),
          branch_in_item_set:accepted(teacher,action),release_logits:scores,release_probability:weights[2]/den,
          actual_field_slots:s.itemSlots[p].map(x=>x?{...x}:null),distance:m.distance()};
        if(!firstDivergence&&student.some((x,i)=>x!==action[i]))firstDivergence={tick:s.t,student,branch:action};
        m.before(action);const joint=[];joint[p]=action;joint[e]=[4,0,0];
        const info=s.step(joint);m.after();
        const ownContacts=m.contacts.filter(x=>x.player===p&&x.owner===p&&x.item===1);
        contact||=ownContacts.length>0;completed||=contact&&priorSlide&&s.movementStatus[p]!==2;
        delivery=s.bunStored[p][e]>stock;
        death||=!s.alive[p]||s.trapped[p]>0;
        drop||=s.bunCarried[p]!==e&&!delivery;
        Object.assign(row,{after_state_hash:hash(bank.captureInstrumented(s,m).fields),after_distance:m.distance(),
          contacts:m.contacts,owned_banana_contact:ownContacts,sliding:s.movementStatus[p]===2,
          death,drop,delivery,events:{died:info.died,placed:info.placed}});trace.push(row);
        if(completed||death||drop||delivery)break;
      }
      const steps=trace.length,net=origin-m.distance()-.18*steps;
      results.push({arm,abstained:false,source_acquisition:record.causal_proof.source,
        executed_physics_steps:steps,owned_banana_contact:contact,full_slide_completed:completed,
        net_home_distance_after_total_walking_cost:net,beneficial_completed:completed&&!death&&!drop&&net>.5,
        death_or_native_trap:death,drop,delivery,stopped_reason:stoppedReason,censored:steps===128&&!completed&&!death&&!drop&&!delivery,
        first_student_vs_branch_divergence:firstDivergence,final_item_metrics:m.rows,trace});
    }
    const result={passed:true,source_bank:path.relative(process.cwd(),bankFile),source_bank_sha256:crypto.createHash('sha256').update(fs.readFileSync(bankFile)).digest('hex'),
      model_json_sha256:loaded.sha,seed:record.seed,tick:record.tick,category:record.category,
      source_proof:record.causal_proof,real_native_next_step_parity:record.next_step_parity,
      original_item_forecast_at_source:firstTeacher,committed_plan:plan,results,
      executed_branch_physics_steps:results.reduce((n,r)=>n+r.executed_physics_steps,0),optimizer_steps:0,
      scope:'mechanism only; enemy-wait/native recovery timers; original JAX item forecast adapted to native masks; observable JS route fallback, not JAX temporal-safety parity; not Q or independent capability'};
    fs.writeFileSync(outFile+'.tmp',JSON.stringify(result));fs.renameSync(outFile+'.tmp',outFile);
  }finally{oracle.close();}
}
if(require.main===module)run(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
module.exports={publicRequest,pickPlan,accepted};
