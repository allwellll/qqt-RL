"""Freeze H5 and one independent replication before any GPU updates."""
import json
import hashlib
import subprocess
from pathlib import Path
NAMESPACE='runs/je09_bun_20261016/stage_h5c_native_course_dagger_pfsp8_v26'
REPLICA='runs/je09_bun_20261016/stage_h5cr_native_course_dagger_replica_pfsp8_v27'


def declare(namespace=NAMESPACE,replica=False):
    root=Path(namespace);assert not root.exists(),'fresh namespace required'
    d=json.loads(Path('reports/je09_bun_20261015/stage_h3_declaration_v17_20261016.json').read_text())
    d.update(experiment_id='JE09-STAGE-H5'+('R' if replica else '')+'-STAGE-SWITCH-DAGGER-20261016-'+('v27' if replica else 'v26'),
        namespace=namespace,parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        parent='runs/je09_bun_20261016/stage_h3_natural_combat_coverage_pfsp8_v17/formal/bun392_bridge_pfsp8/gen000024',
        parent_generation=24,seed=3401000000 if replica else 3301000000)
    # Disjoint derived identities, with a separately keyed technical stream.
    names=list(d['seeds'])
    reserved=set()
    for path in Path('runs').glob('je09_bun_*/*/ROUND_DECLARATION.json'):
        prior=json.loads(path.read_text());reserved.update(prior.get('seeds',{}).values())
        if 'seed' in prior:reserved.add(prior['seed'])
    identities={};seeds={}
    for name in names:
        identity=d['experiment_id']+':'+name;nonce=0
        while True:
            value=int.from_bytes(hashlib.sha256((identity+':'+str(nonce)).encode()).digest()[:4],'big')
            if value not in reserved:break
            nonce+=1
        seeds[name]=value;identities[name]=dict(identity=identity,nonce=nonce);reserved.add(value)
    d['seeds']=seeds;d['seed']=seeds['train'];d['seed_derivation']=identities
    assert len(set(d['seeds'].values()))==len(d['seeds'])
    d['change']='Only add iterative learner-visited native stage-switch action-set oracle supervision and bounded dataset aggregation; no reward/network/pool/lesson/evaluation changes.'
    d['parent_preservation']='H3g24 Hunter37 hard5 Tactical10 controlled14/12 carry16 versus H3g32 35/4/8/13,12; H4g4 36/4/17/11,10; H4Rg16 39/1/15/11,14. Prefer hard combat plus balanced controlled retention; protocols share definitions but derived seeds differ, so no paired significance claimed.'
    d['dagger']=dict(coefficient=.05,temperature=1.,natural_scene=8,
        rollout='current learner actual actions; new policy resamples every generation; no oracle takeover',
        oracle='attributed native sugar/death: A actual credited loose target; B enemy stock cage; after actual pickup same-trajectory return120',
        exclude='free alive enemy while empty; glue/banana; terminal/dead/trapped learner; danger/unreachable; no future information',
        capacity_per_device=2048,new_per_generation_per_device=256,
        replay_per_microbatch=64,current_history_ratio=[.5,.5],
        shortage='fallback to available half; padded invalid rows have exact zero gradient',
        aggregation='bounded FIFO across updated learner rounds; round provenance and full buffer checkpointed',
        retention='unchanged combat .60, bridge .25, natural .10, other .05; unchanged C/E replay, D chase, F mode, G combat, H items and parent KL')
    d['dagger_opponent_course']=dict(weak_episode_fraction=[.5,.35,.2,.1],generations_per_level=8,
        split_courier_guard=[.5,.5],carry_move_duty=[0,.15,.35,.6],guard_move_duty=[0,.1,.25,.5],
        bomb_period=[64,32,16,8],live_bomb_action_capacity=[1,1,2,2],escape_response=[0,.25,.5,1],
        approach_dig_period=4,approach_move_duty=1.,
        native_only='Only opponent legal move/ability decisions; no state modifications, teleport, ownership injection, forced drop/death.',
        origin_semantics='Opponent taking learner stock carries learner-origin bun: its recovery is A-own-recovery, not enemy acquisition/capture. Both origins logged separately.',
        strong='At least nominal50% natural-full episodes retain original responsive/g0/stage1/PFSP mix; all nonfull lessons unchanged; observed ticks reported.',
        evaluation='Unchanged strong assist-off protocols; course never enters evaluation.')
    d['change']+=' Sampling amendment: native weak courier/guard opponent ladder inside DAgger, annealed across32generations; no stat mutation.'
    d['reason']='Read-only native same-state oracle proves reachable own sugar/death windows with cage stock, but learner often fails Objective Rush. Strict loose-first full metric is not all-path inventory acquisition.'
    d['terminal_relay']=dict(successor=REPLICA if not replica else None,
        policy='one independent fixed32 same-parent DAgger replication; unique owner directly starts it after terminal validation; no capability gates',
        acceptance='successor actual finite nonzero eight-device formal update plus smoke/fresh recovery; marker alone insufficient')
    d['technical_shape']=dict(devices=8,envs_per_device=256,steps=1024,fresh_ticks=2097152)
    root.mkdir(parents=True)
    text=json.dumps(d,indent=2)+'\n';(root/'ROUND_DECLARATION.json').write_text(text)
    Path('reports/je09_bun_20261015/'+('stage_h5cr_v27' if replica else 'stage_h5c_v26')+'_declaration_20261016.json').write_text(text)


if __name__=='__main__':
    declare();declare(REPLICA,True)
