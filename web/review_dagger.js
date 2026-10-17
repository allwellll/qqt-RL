'use strict';
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.QQTReviewDagger=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const SCHEMA='qqt.dagger-quality/v1';
  const FRAGMENTS=Object.freeze({COMBAT_CONTROL:'战斗控制',A_LOOSE_RECOVERY:'A 散包追拾（来源未区分）',A_ENEMY_DROP_RECOVERY:'A 真实敌源掉包追拾',
    B_CAGE_STOCK:'B 库存冲笼',CARRY_RETURN:'持包返程',CAGE_DELIVERY:'入笼交付'});
  const FIELDS=Object.freeze({candidate:'候选片段',explicit_localization:'明确定位',
    teacher_proposed:'教师提议',strict_accepted:'严格接受',actor_query:'Actor 梯度查询',
    actor_gradient:'进入 Actor 梯度',teacher_acceptance:'教师接受率',supervision_coverage:'最终监督覆盖',
    noneligible_gradient_pollution:'非 eligible 梯度污染',legal_action_rate:'教师动作合法率',
    counterfactual_survival:'短反事实存活率',task_progress:'短反事实任务进展',
    drop_expert_consistency:'掉包专家一致率',return_expert_consistency:'返程专家一致率',
    stock_theft:'库存盗取',enemy_carried_drop:'敌携包归因掉落',enemy_origin_pickup:'敌源拾取',
    enemy_path_query:'敌源路径查询',own_recovery_pickup:'己包回收拾取',target_a_query:'A 课程 Actor 查询',
    native_loose_pickup:'原生散包拾取',strict_verifier_reject:'严格校验拒标',
    strict_verifier_reject_upper_bound:'严格拒标数量上界',AMBIGUOUS:'定位歧义',UNREACHABLE:'无法到达',
    DOOMED:'无安全动作',UNCLEAR_PROVENANCE:'来源不明',FUTURE_DEPENDENT:'依赖未来状态',
    native_loose_pickup_missing:'缺少原生散包拾取',enemy_origin_pickup_missing:'缺少敌源拾取',
    enemy_path_query_missing:'缺少敌源路径查询'});
  const label=key=>`${Object.hasOwn(FIELDS,key)?FIELDS[key]:Object.hasOwn(FRAGMENTS,key)?FRAGMENTS[key]:'未翻译'} · ${key}`;
  const value=(v,rate=false)=>v==null?'未记录':typeof v==='boolean'?(v?'是':'否'):
    rate?`${(v*100).toFixed(2)}%`:typeof v==='number'?v.toLocaleString('en-US'):String(v);
  function validateEvents(events,length){
    if(events===undefined)return;
    const fail=()=>{throw Error('dagger_events schema');};
    if(!Array.isArray(events)||events.length>2*(length+1))fail();
    const allowed=['tick','player','fragment','candidate','explicit_localization','teacher_proposed',
      'strict_accepted','actor_query','actor_gradient','rejection_reasons','action_set'];
    let previous=-1;
    for(const e of events){
      if(!e||Array.isArray(e)||Object.keys(e).some(k=>!allowed.includes(k))||
        !Number.isInteger(e.tick)||e.tick<0||e.tick>length||![0,1].includes(e.player))fail();
      const order=e.tick*2+e.player;if(order<=previous)fail();previous=order;
      if(e.fragment!==undefined&&e.fragment!==null&&!Object.hasOwn(FRAGMENTS,e.fragment))fail();
      for(const k of ['candidate','explicit_localization','teacher_proposed','strict_accepted','actor_query','actor_gradient']){
        if(e[k]!==undefined&&e[k]!==null&&typeof e[k]!=='boolean')fail();
      }
      if(e.strict_accepted===true&&(e.teacher_proposed===false||e.candidate===false))fail();
      if(e.actor_gradient===true&&(e.actor_query===false||e.strict_accepted===false||e.teacher_proposed===false||e.candidate===false))fail();
      if(e.rejection_reasons!==undefined&&e.rejection_reasons!==null&&(!Array.isArray(e.rejection_reasons)||
        e.rejection_reasons.length>16||new Set(e.rejection_reasons).size!==e.rejection_reasons.length||
        e.rejection_reasons.some(k=>typeof k!=='string'||!/^[A-Za-z][A-Za-z0-9_]{0,79}$/.test(k))))fail();
      if(e.strict_accepted===true&&e.rejection_reasons?.length)fail();
      if(e.action_set!==undefined&&e.action_set!==null){
        if(!Array.isArray(e.action_set)||!e.action_set.length||e.action_set.length>15)fail();
        const seen=new Set();for(const a of e.action_set){
          if(!Array.isArray(a)||a.length!==2||!Number.isInteger(a[0])||a[0]<0||a[0]>4||
            !Number.isInteger(a[1])||a[1]<0||a[1]>2||seen.has(a.join(',')))fail();seen.add(a.join(','));
        }
      }
      if(!allowed.slice(2).some(k=>e[k]!==undefined))fail();
    }
  }
  // Sparse records are point observations, never carried forward or inferred.
  const atTick=(doc,tick)=>doc.dagger_events?.filter(e=>e.tick===tick)||[];
  const episodeStatus=doc=>doc.dagger_events===undefined?'本局逐tick DAgger：未记录':
    `本局逐tick DAgger：${doc.dagger_events.length} 条稀疏记录（其他 tick 未记录）`;
  function episodeCounts(doc){
    return Object.fromEntries(['candidate','explicit_localization','teacher_proposed','strict_accepted','actor_query','actor_gradient'].map(key=>{
      const rows=(doc.dagger_events||[]).filter(e=>typeof e[key]==='boolean');
      return [key,rows.length?{recorded:rows.length,positive:rows.filter(e=>e[key]).length}:null];
    }));
  }
  function renderTrack(target,doc,onSeek,time){
    target.replaceChildren();const document=target.ownerDocument;
    for(const e of doc.dagger_events||[]){
      const b=document.createElement('button');b.className='mark';b.style.left=`${e.tick/doc.actions.length*100}%`;
      b.style.background=e.actor_gradient===true?'#9ac9e6':e.strict_accepted===false?'#e78f94':'#a9a3d6';
      b.title=`tick ${e.tick} · ${time(e.tick)} · ${e.player===0?'模型':'对手'} · ${e.fragment?label(e.fragment):'片段类别：未记录'} · `+
        ['candidate','explicit_localization','teacher_proposed','strict_accepted','actor_query','actor_gradient'].map(k=>`${label(k)} ${value(e[k])}`).join('；');
      b.title+=`；拒标原因 · rejection_reasons ${e.rejection_reasons==null?'未记录':e.rejection_reasons.map(label).join(' / ')||'无'}；动作集合 · action_set ${e.action_set==null?'未记录':JSON.stringify(e.action_set)}`;
      b.setAttribute('aria-label',b.title);b.onclick=event=>{event.stopPropagation();onSeek(e.tick);};target.append(b);
    }
    if(!doc.dagger_events?.length){const empty=document.createElement('span');empty.className='dagger-empty';
      empty.textContent=doc.dagger_events===undefined?'未记录':'无稀疏记录';target.append(empty);}
  }
  function validateQuality(d){
    if(d?.schema!==SCHEMA||!d.source_run||!['run','smoke','historical'].includes(d.scope)||
      !Array.isArray(d.sources)||!d.sources.length||!d.funnel||!d.fragments||!d.teacher_quality||!d.refusal_reasons)throw Error('DAgger quality schema');
    for(const s of d.sources)if(!/^[\w.-]+\.json$/.test(s.file)||!/^[a-f0-9]{64}$/.test(s.sha256))throw Error('DAgger source hash');
    for(const group of [d.funnel,d.teacher_quality,d.refusal_reasons,...Object.values(d.fragments)]){
      for(const v of Object.values(group))if(v!==null&&(typeof v!=='number'||!Number.isFinite(v)||v<0))throw Error('DAgger metric');
    }
    return d;
  }
  return {SCHEMA,FRAGMENTS,FIELDS,label,value,validateEvents,atTick,episodeStatus,episodeCounts,renderTrack,validateQuality};
});
