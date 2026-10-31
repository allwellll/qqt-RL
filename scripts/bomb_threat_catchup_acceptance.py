"""Passive operational acceptance from actual updates and immutable game markers."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.combat_mix_owner import read
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'


def continuation_evidence(run):
    """Recognize both actual worker ABIs; do not invent legacy receipts."""
    source=read(run/'SOURCE_RESTORE.json',{});technical=read(run/'TECHNICAL_CONTINUATION.json',{})
    if source.get('passed') and technical.get('passed'):
        return dict(abi='legacy-prewarm-v1',source_restore=source,prewarm=technical.get('prewarm'))
    restored=read(run/'RESTORE.json',{});smoke=read(run/'SMOKE.json',{})
    required=('passed','fresh_process','next_step_exact','Adam_exact','full_env_aux_RNG_exact','ring_exact')
    if all(restored.get(k) for k in required) and smoke.get('passed') and smoke.get('actor_critic_nonzero'):
        checkpoint=smoke.get('checkpoint',{})
        if checkpoint.get('checkpoint')!=restored.get('checkpoint'):return None
        return dict(abi='atomic-smoke-restore-v2',source_restore=restored,
                    prewarm=None,smoke_checkpoint=checkpoint,formal_after_update=checkpoint['update'])
    return None


def refresh():
    frozen=read(BASE/'CPU_CATCHUP_TARGETS.json',{});accepted={};pending=[]
    for slot,target in frozen.get('targets',{}).items():
        out=BASE/'eval'/Path(target['request']).stem
        marker=read(out/'NATURAL_COMPLETE.json',{});gate=read(out/'GATE_METRICS.json',{})
        if not marker or not gate or gate.get('source_natural_sha256')!=sha(out/'NATURAL_COMPLETE.json'):
            pending.append(slot);continue
        assert marker['checkpoint']==target['checkpoint'] and marker['seed']==2026122900
        assert marker['split']=='dev' and marker['rule_version']=='stock-window-v1'
        assert all(v['games']==64 for v in marker['natural_js_full2400'].values())
        assert set(marker['natural_js_full2400'])=={'greedy','categorical'}
        for name,digest in marker['files'].items():assert sha(out/name)==digest
        launch=read(out/'EVAL_LAUNCH.json',{})
        accepted[slot]=dict(**target,natural_sha256=sha(out/'NATURAL_COMPLETE.json'),
            gate_sha256=sha(out/'GATE_METRICS.json'),games=128,
            seconds_to_natural=marker['unix']-launch.get('unix',marker['unix']),
            completed_unix=marker['unix'],pbt_eligible=gate['pbt_selection_eligible'],best_eligible=gate['best_eligible'])
    auth=read(BASE/'CATCHUP_0445_AUTHORIZATION.json',{})
    ok=bool(frozen.get('targets')) and not pending
    result=dict(status='PASS' if ok else 'IN_PROGRESS',completed=accepted,pending=pending,
        target_deadline_unix=auth.get('target_latest_deadline_unix'),
        within_two_hour_target=all(v['completed_unix']<=auth['target_latest_deadline_unix'] for v in accepted.values()) if ok else None,
        cpu_status=read(BASE/'CPU_EVAL_STATUS.json',{}),old_evidence_not_rerun=True,
        verification='immutable shards SHA, marker seed/rules, two modes x64, gate source hash',unix=time.time())
    previous=read(BASE/'CPU_CATCHUP_ACCEPTANCE.json',{})
    atomic_json(BASE/'CPU_CATCHUP_ACCEPTANCE.json',result)
    if ok and previous.get('status')!='PASS':
        from scripts.iter_pbt_owner import notify
        notify('BOMB_THREAT_CATCHUP_COMPLETE','BOMB-THREAT首批7个slot最新点双64完整局及GATE已SHA验收；12路继续追赶新点，训练独立。',receipt='runs/bomb_threat_v1/CPU_CATCHUP_ACCEPTANCE.json')
    rows={};unavailable={}
    previous_chunk_result=read(BASE/'CHUNK320_ACCEPTANCE.json',{})
    previous_chunks=previous_chunk_result.get('accepted_slots',{})
    for gpu,entry in read(BASE/'SLOTS.json',{}).items():
        if previous_chunks.get(gpu,{}).get('run')==entry['run']:
            rows[gpu]=previous_chunks[gpu];continue
        run=BASE/'formal'/entry['run'];proto=read(BASE/'protocols'/(entry['run']+'.json'),{})
        if proto.get('chunk_updates',0)<320:continue
        evidence=continuation_evidence(run)
        if evidence is None:
            unavailable[gpu]=dict(run=entry['run'],reason='no complete recognized continuation receipt pair')
            continue
        cps=sorted(run.glob('u*/COMMITTED.json'))
        for file in cps:
            cm=read(file);m=cm.get('metrics',{})
            if m.get('fresh_parent') or not m.get('finite'):continue
            if cm['update']<=evidence.get('formal_after_update',-1):continue
            assert all(m['update_l2'][head]>0 for head in ('actor','critic'))
            assert cm['update']>proto['chunk_start_update']
            from scripts.checkpoint_policy_artifact import verify_policy
            verify_policy(file.parent)
            rows[gpu]=dict(run=entry['run'],first_committed=cm['checkpoint'],update=cm['update'],
                state_sha256=cm['state_sha256'],policy_sha256=cm['policy_sha256'],metrics=m,
                **evidence,chunk_updates=proto['chunk_updates'])
            break
    atomic_json(BASE/'CHUNK320_ACCEPTANCE.json',dict(status='PASS' if rows else 'AWAITING_FIRST_UPDATE',
        accepted_slots=rows,unavailable_slots=unavailable,
        scope='earliest retained real formal atomic checkpoint recognized by continuation ABI; may follow pruned earlier points; smoke/prewarm excluded',unix=time.time()))
    if rows and previous_chunk_result.get('status')!='PASS':
        from scripts.iter_pbt_owner import notify
        first=next(iter(rows.values()))
        notify('BOMB_THREAT_CHUNK320_FIRST_'+first['run'],
            f"BOMB-THREAT 320更新chunk正式首原子更新已验收：{first['run']} u{first['update']}，finite、Actor/Critic非零、完整源恢复，冻结训练继续。",receipt='runs/bomb_threat_v1/CHUNK320_ACCEPTANCE.json')
    return result

if __name__=='__main__':print(json.dumps(refresh(),indent=2))
