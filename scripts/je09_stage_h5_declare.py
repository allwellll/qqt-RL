"""Freeze H5 and one independent replication before any GPU updates."""
import json
import hashlib
import subprocess
from pathlib import Path
NAMESPACE='runs/je09_bun_20261016/stage_h5_stage_switch_dagger_pfsp8_v24'
REPLICA='runs/je09_bun_20261016/stage_h5r_stage_switch_dagger_replica_pfsp8_v25'


def declare(namespace=NAMESPACE,replica=False):
    root=Path(namespace);assert not root.exists(),'fresh namespace required'
    d=json.loads(Path('reports/je09_bun_20261015/stage_h3_declaration_v17_20261016.json').read_text())
    d.update(experiment_id='JE09-STAGE-H5'+('R' if replica else '')+'-STAGE-SWITCH-DAGGER-20261016-'+('v25' if replica else 'v24'),
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
    d['reason']='Read-only native same-state oracle proves reachable own sugar/death windows with cage stock, but learner often fails Objective Rush. Strict loose-first full metric is not all-path inventory acquisition.'
    d['terminal_relay']=dict(successor=REPLICA if not replica else None,
        policy='one independent fixed32 same-parent DAgger replication; unique owner directly starts it after terminal validation; no capability gates',
        acceptance='successor actual finite nonzero eight-device formal update plus smoke/fresh recovery; marker alone insufficient')
    d['technical_shape']=dict(devices=8,envs_per_device=256,steps=1024,fresh_ticks=2097152)
    root.mkdir(parents=True)
    text=json.dumps(d,indent=2)+'\n';(root/'ROUND_DECLARATION.json').write_text(text)
    Path('reports/je09_bun_20261015/'+('stage_h5r_v25' if replica else 'stage_h5_v24')+'_declaration_20261016.json').write_text(text)


if __name__=='__main__':
    declare();declare(REPLICA,True)
