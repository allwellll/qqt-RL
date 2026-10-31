"""CPU producers for the three remaining campaign tasks; never starts training.

Bank states are reconstructed from real natural actions/RNG, not manufactured
carriers. Missing strata and unexecuted training are explicitly not acceptance.
"""
import json
import os
from pathlib import Path
import pickle
import sys
import time
from collections import Counter

from scripts.population_run_state import atomic_json, sha

BASE = Path('runs/authorized_campaign_20261026')


def read(path):
    return json.loads(Path(path).read_text())


def raw_source(path):
    return dict(path=str(path), sha256=sha(path))


def source_split(index):
    # All checkpoints and action modes for a source seed remain together.
    return ('train', 'train', 'train', 'dev', 'holdout')[index % 5]


def execution_job(job, protocol):
    job = dict(job)
    job.update(args=['-m', 'scripts.campaign_remaining_cpu', job['protocol']],
               ready_files=[protocol['source_checkpoint_protocol']] + [
                   str(BASE/'native_fullgame_v1'/m/'COMPLETE.json') for m in protocol['model_selection']],
               readiness='producer inputs; final products are never prerequisites',
               execution_revision='real_cpu_producer_v1',
               frozen_files={job['protocol']: sha(job['protocol']),
                             __file_relative(): sha(__file_relative())})
    return job


def __file_relative():
    return 'scripts/campaign_remaining_cpu.py'


def linked_funnel(games):
    result = {}
    for kind in ('death', 'native_syrup'):
        windows = [(g, w) for g in games for w in g['window_funnel']['windows'] if w['kind'] == kind]
        linked = [(g, e) for g in games for e in g['carry_episodes']['episodes']
                  if (e.get('stock_window_link') or {}).get('window_kind') == kind
                  or any((e.get('stock_window_link') or {}).get('window_id') == w['window_id']
                         for w in g['window_funnel']['windows'] if w['kind'] == kind)]
        result[kind] = dict(windows=len(windows),
                            windows_with_pickup=sum(bool(w['stock_pickup_ticks']) for _, w in windows),
                            attributed_windows=dict(Counter(w.get('cause_attribution', 'unknown') for _, w in windows)),
                            same_custody_acquisitions=len(linked),
                            same_custody_deliveries=sum(e['outcome'] == 'delivered' for _, e in linked),
                            custody_outcomes=dict(Counter(e['outcome'] for _, e in linked)))
    return result


def funnel(protocol, out, protocol_path):
    from scripts.passive_evaluation_metrics import summarize_cohort, coverage_matrix
    sources, rows = [], []
    for model in protocol['model_selection']:
        source = BASE / 'native_fullgame_v1' / model / 'SUMMARY.json'
        summary = read(source); sources.append(raw_source(source))
        for row in summary['rows']:
            games = [ep['passive_metrics'] for ep in row['episodes']]
            rows.append(dict(engine='native JAX JE07', opponent='JAX rule3', model=model,
                             action_mode=row['action_mode'], games=len(games),
                             funnel=linked_funnel(games), pooled=summarize_cohort(games)))
    # ABI2 is a separate, matched corpus; never add its episodes to ABI1 counts.
    abi = []
    for model in protocol['model_selection']:
        source = BASE / 'funnel_abi_v2' / model / 'SUMMARY.json'
        if not source.exists():
            abi.append(dict(model=model, available=False)); continue
        sources.append(raw_source(source))
        for row in read(source)['rows']:
            fields = [e['passive_abi_v2'] for e in row['episodes']]
            keys = ('own_alive_ticks', 'own_actionable_ticks', 'preterminal_alive_ticks', 'settlement_tail_alive_ticks')
            abi.append(dict(model=model, action_mode=row['action_mode'], games=len(fields),
                            available=True, **{k: sum(e[k] for e in fields) for k in keys}))
    js_source = BASE / 'production_js_matrix_v1/SUMMARY.json'
    sources.append(raw_source(js_source)); js = []
    jsrows = read(js_source)['rows']
    for model in protocol['model_selection']:
        for mode in protocol['budget']['action_modes']:
            episodes = [r['episode'] for r in jsrows if r['model'] == model and r['action_mode'] == mode]
            fields = ('window_stock_pickup', 'window_stock_delivery', 'stock_pickup_dead',
                      'stock_pickup_native_sugar', 'stock_pickup_free_alive', 'nonstock_pickup',
                      'nonstock_delivery', 'death', 'delivery')
            js.append(dict(engine='production JS', opponent='Hunter (matrix frozen configuration)',
                           model=model, action_mode=mode, games=len(episodes),
                           counts={k: sum(e.get(k, 0) for e in episodes) for k in fields},
                           reason_unknown='No event-supported caught-by-pursuer attribution; no cross-engine pooling'))
    atomic_json(out/'cross_engine_funnel_summary.json', dict(
        scientific_acceptance=True, acceptance_scope='complete available-trace funnel accounting, not all missing instrumentation',
        raw_sources=sources, native=rows, production_js=js, native_abi2_separate_corpus=abi,
        missing=['ABI2 alive/actionable for R2', 'JS terminal/truncation ABI equivalence',
                 'event-supported pursuer attribution'], coverage=coverage_matrix()))
    item_source = BASE/'side_cx_results/itemcausal/COMPLETE.json'
    items = dict(raw_sources=sources, scientific_acceptance=True,
                 acceptance_scope='available placement/hit accounting, complete-chain gaps disclosed',
                 phases=[dict(model=r['model'], action_mode=r['action_mode'], phases=r['pooled']['phases']) for r in rows],
                 unavailable=['Exact natural banana own-placement/self-slide-to-delivery completed chain and counterfactual net benefit',
                              'Natural glue full pursuit-delay net benefit'],
                 banana_primary_semantics='carrying self-slide acceleration and reduced delivery ticks; enemy slip is secondary')
    if item_source.exists():
        items['raw_sources'].append(raw_source(item_source))
        items['separate_causal_branch_diagnostic']=str(item_source)
        items['branch_caveat']='Round1 enemy-banana branches do not answer primary self-acceleration semantics; round2 delegated, not rerun'
    atomic_json(out/'complete_item_chain_outcomes.json', items)
    atomic_json(out/'PROGRESS.json', dict(completed_games=sum(r['games'] for r in rows)+sum(r['games'] for r in js), unix=time.time()))
    atomic_json(out/'COMPLETE.json', dict(passed=True, protocol_sha256=sha(protocol_path),
        execution_scope='full available-corpus aggregation; missing metrics explicitly retained',
        scientific_acceptance=False, missing_data=items['unavailable'],
        evidence={x:sha(out/x) for x in protocol['required_evidence']}, unix=time.time()))


def freeze_bank_sources(protocol, out):
    """Freeze the first 96 complete games per model/mode, before seeing pickups."""
    target=out/'SOURCE_MANIFEST.json'
    if target.exists():return read(target)
    selected=[]; sources=[]
    per_mode=protocol['budget']['bank_maximum_natural_episodes']//(len(protocol['model_selection'])*len(protocol['budget']['action_modes']))
    for model in protocol['model_selection']:
        path=BASE/'native_fullgame_v1'/model/'SUMMARY.json';sources.append(raw_source(path))
        for row in read(path)['rows']:
            if row['action_mode'] not in protocol['budget']['action_modes']:continue
            for ep in sorted(row['episodes'], key=lambda e:e['episode_index'])[:per_mode]:
                if ep['artificial_truncation']:raise ValueError('Incomplete source game')
                selected.append(dict(model=model, mode=row['action_mode'], episode=ep,
                                     split=source_split(ep['episode_index'])))
    value=dict(protocol_sha256=sha(out_protocol(protocol)), sources=sources, episodes=selected,
               selection='first fixed episode indices; include every no-pickup game; replay actions/RNG, no teacher',
               source_game_isolation='same source seed across model/mode assigned to same split',
               no_new_policy_rollouts=True, states_are_newly_restored_from_full_causal_prefix=True,
               frozen_unix=time.time())
    atomic_json(target,value);return value


def out_protocol(protocol):
    return BASE/'protocols/natural_bank_coverage_v1.json'


def bank(protocol, out, protocol_path):
    import numpy as np
    import jax
    import jax.numpy as j
    from scripts import expert_reward_alignment_fullmatch as F
    from scripts.terminal_checkpoint_restore import digest
    from scripts.carrier_mechanism_stats import validate_bank_record
    F.T.configure();F.provenance.install_provenance()
    assert jax.default_backend()=='cpu' and not os.environ.get('CUDA_VISIBLE_DEVICES')
    manifest=freeze_bank_sources(protocol,out)
    groups={}
    for row in manifest['episodes']:
        groups.setdefault((row['model'],row['mode'],row['episode']['provenance']['trace']),[]).append(row)

    def actions(st, ax, key, own):
        nk, _, _, ko, ke=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,5))(key),0,1)
        mm,bm=jax.vmap(F.E.env.legal_mask)(st)
        eb=bm[:,1].at[:,1:].set(bm[:,1,1:]&~ax['terminal'][:,None])
        def rule(s,m,b,k):
            return F.E.bun_jax_bots.rule_bot_actions(jax.tree.map(lambda a:a[None],s),j.array([1]),m[None],b[None],k,j.array([3]))[0]
        enemy=jax.vmap(rule)(st,mm[:,1],eb,ko)
        return nk,ke,j.stack((own,enemy),1)

    def one(st,ax,key,own):
        nk,ke,act=actions(st,ax,key,own)
        af,na,r,d,info,parts=F.T.transition(st,ax,act,ke,arm='control')
        return af,na,nk,r,d,info,act,ke
    one_jit=jax.jit(one)

    def chunk(carry, own):
        def step(c, move):
            st,ax,key,ended,got,pre,post,pa,aa,pk,ak,actsave,eksave=c
            af,na,nk,r,d,info,act,ke=one(st,ax,key,move)
            hit=F.E.causal_pickup(st,af,info)&~got&~ended
            select=F.select_tree
            pre=select(hit,st,pre);post=select(hit,af,post)
            pa=select(hit,ax,pa);aa=select(hit,na,aa)
            pk=select(hit,key,pk);ak=select(hit,nk,ak)
            actsave=select(hit,act,actsave);eksave=select(hit,ke,eksave)
            telemetry=j.stack((st.t,st.alive[:,1],st.trapped[:,1],st.bun_carried[:,0]),-1)
            af=select(ended,st,af);na=select(ended,ax,na);nk=j.where(ended[:,None],key,nk)
            return (af,na,nk,ended|d,got|hit,pre,post,pa,aa,pk,ak,actsave,eksave),(
                j.where(ended,0.,r[:,0]),telemetry,~ended)
        return jax.lax.scan(step,carry,own)
    scan=jax.jit(chunk)
    start=time.time();allrecords=[];raw=[];completed=0
    for number,((model,mode,tracepath),rows) in enumerate(groups.items()):
        target=out/'batches'/f'{model}_{mode}_{rows[0]["episode"]["episode_index"]:05d}.pkl'
        receipt=target.with_suffix('.json')
        if receipt.exists():
            saved=read(receipt);assert sha(target)==saved['sha256'];allrecords.extend(saved['records']);completed+=len(rows);continue
        episodes=[r['episode'] for r in rows]
        assert sha(tracepath)==episodes[0]['provenance']['trace_sha256']
        raw.append(raw_source(tracepath))
        with np.load(tracepath,allow_pickle=False) as z:trace={k:z[k] for k in z.files}
        names=trace['telemetry_names'].tolist();cols=[e['provenance']['telemetry_column'] for e in episodes]
        own=trace['telemetry'][:,cols][:,:,[names.index('own_move'),names.index('own_ability')]].astype(np.int32)
        F.SEED_START=episodes[0]['seed']-episodes[0]['episode_index']
        st,ax,key=F.initial_batch(j.array([e['episode_index'] for e in episodes],j.int32));n=len(episodes)
        carry=(st,ax,key,j.zeros(n,bool),j.zeros(n,bool),st,st,ax,ax,key,key,j.zeros((n,2,2),j.int32),key)
        outputs=[]
        for offset in range(0,len(own),64):
            carry,output=scan(carry,j.asarray(own[offset:offset+64]));jax.block_until_ready(carry[0].t)
            outputs.append(jax.device_get(output))
            atomic_json(out/'PROGRESS.json',dict(stage='replay_real_causal_source',source_batch=number,
                completed_games=completed,ticks=offset+len(own[offset:offset+64]),acquisitions_so_far=len(allrecords),
                cpu_only=True,unix=time.time()))
        rewards=np.concatenate([o[0] for o in outputs]);tele=np.concatenate([o[1] for o in outputs]);active=np.concatenate([o[2] for o in outputs])
        host=jax.device_get(carry);items=[];records=[]
        take=lambda tree,i:jax.tree.map(lambda a:np.asarray(a)[i:i+1],tree)
        for i,(ep,row) in enumerate(zip(episodes,rows)):
            valid=int(active[:,i].sum())
            if valid!=ep['valid_transitions'] or not np.allclose(rewards[:valid,i],trace['legacy_reward'][:valid,cols[i]],atol=1e-4,rtol=1e-5):
                raise ValueError('Full source reward/termination replay mismatch')
            expected=trace['telemetry'][:valid,cols[i]][:,[names.index(x) for x in ('tick','enemy_alive_before','enemy_trapped_before','own_carried_before')]]
            if not np.array_equal(tele[:valid,i],expected):raise ValueError('Full source state telemetry mismatch')
            if not host[4][i]:continue
            before,after=take(host[5],i),take(host[6],i);pa,aa=take(host[7],i),take(host[8],i)
            pk,ak=take(host[9],i),take(host[10],i);tick=int(before.t[0]);transition_index=int(np.flatnonzero(trace['telemetry'][:,cols[i],names.index('tick')]==tick)[0])
            source_id=f'{model}:{mode}:{ep["seed"]}'
            rec=dict(before_carried=int(before.bun_carried[0,0]),after_carried=int(after.bun_carried[0,0]),
                learner_alive_before=bool(before.alive[0,0]),learner_alive_after=bool(after.alive[0,0]),
                stock_steal=bool(before.bun_stored[0,1,1]>after.bun_stored[0,1,1]),
                enemy_alive=bool(before.alive[0,1]),enemy_trapped=int(before.trapped[0,1]),
                source_episode=source_id,source_seed=ep['seed'],source_policy=model,action_mode=mode,
                split=row['split'],layout='native layout0',map_id=806,seat=0,event_tick=tick,
                provenance='actual_env_step',distance_manhattan=float(np.abs(after.pos[0,0]-(np.asarray(F.E.env._BUN_BASES[0])+1)).sum()),
                window='death' if not before.alive[0,1] else 'native_syrup',
                pre_hash=digest(before),post_hash=digest(after),reward_aux_hash=digest(aa),source_trace=tracepath,
                source_trace_sha256=ep['provenance']['trace_sha256'],full_source_reward_and_telemetry_replay=True)
            validate_bank_record(rec)
            item=dict(record=rec,before=before,states=after,preaux=pa,aux=aa,prekey=pk,keys=ak,
                      acquisition_actions=take(host[11],i),acquisition_envkeys=take(host[12],i),
                      source_prefix_own_actions=own[:transition_index+1,i],source_episode=ep,
                      initial_seed_start=F.SEED_START)
            # Disk serialization roundtrip includes all environment, reward aux and RNG.
            restored=pickle.loads(pickle.dumps(item,pickle.HIGHEST_PROTOCOL))
            if transition_index+1<len(own):
                nextown=j.asarray(own[transition_index+1,i:i+1])
                expected_next=one_jit(after,aa,ak,nextown)
                restored_next=one_jit(restored['states'],restored['aux'],restored['keys'],nextown)
                jax.block_until_ready(restored_next[0].t)
                rec['next_step_parity']=digest(jax.device_get(expected_next))==digest(jax.device_get(restored_next))
                if not rec['next_step_parity']:raise ValueError('Complete state roundtrip next-step mismatch')
            else:rec['next_step_parity']=None
            items.append(item);records.append(rec)
        target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp')
        with tmp.open('wb') as stream:pickle.dump(items,stream,pickle.HIGHEST_PROTOCOL);stream.flush();os.fsync(stream.fileno())
        tmp.replace(target)
        atomic_json(receipt,dict(sha256=sha(target),records=records,source_games=len(episodes),
                    source_trace=tracepath,source_trace_sha256=sha(tracepath),validated_full_games=True,unix=time.time()))
        allrecords.extend(records);completed+=len(rows)
        print(json.dumps(dict(completed_games=completed,causal_states=len(allrecords),batch=str(receipt),elapsed=time.time()-start)),flush=True)
    files=[raw_source(p) for p in sorted((out/'batches').glob('*.pkl'))]
    result=dict(scientific_acceptance=bool(allrecords),raw_sources=manifest['sources']+files,
        collection_method='new full causal-prefix restoration of frozen genuine natural games; not an audit-only inventory',
        source_games=completed,no_acquisition_games=completed-len(allrecords),states=len(allrecords),records=allrecords,
        source_game_splits=dict(Counter(r['split'] for r in allrecords)),
        coverage=dict(map_ids=[806],seats=[0],windows=dict(Counter(r['window'] for r in allrecords)),
            distances=dict(Counter('near' if r['distance_manhattan']<6 else 'mid' if r['distance_manhattan']<12 else 'far' for r in allrecords))),
        missing=['cross-map and P1 natural sources not in frozen source corpus; never filled synthetically',
                 'pickle next-step parity verified in same process; fresh-process parity not yet asserted'],
        scientific_scope='available real natural acquisition bank frozen with provenance; not cross-map generalization',
        elapsed_seconds=time.time()-start,protocol_sha256=sha(protocol_path))
    atomic_json(out/'expanded_natural_bank_frozen.json',result)
    atomic_json(out/'COMPLETE.json',dict(passed=bool(allrecords),scientific_acceptance=False,
        protocol_sha256=sha(protocol_path),evidence_sha256=sha(out/'expanded_natural_bank_frozen.json'),
        scope=result['scientific_scope'],coverage_shortfall=result['missing'],unix=time.time()))


def ratio(protocol, out, protocol_path):
    source=BASE/'natural_bank_coverage_v1/expanded_natural_bank_frozen.json'
    bank=read(source)
    result=dict(passed=False,scientific_acceptance=False,raw_sources=[raw_source(source)],
        bank_states=bank['states'],
        reason='Frozen text specifies six GPU training runs but registered job is CPU acceptance only; latest user explicitly prohibits new GPU successors',
        technical_findings=['combat_item_engine.fresh and dagger_rl_r4_train.install_provenance hardcode .25 in actual reset/provenance',
                            'No validated .125 reset/provenance wrapper exists; cannot equate a sampling audit to training A/B',
                            'paired_seeds 2026103151..53 do not match source mapping keys 2026102951..53'],
        original_protocol_unchanged=True,gpu_started=False,protocol_sha256=sha(protocol_path),unix=time.time())
    atomic_json(out/'bank_ratio_technical_acceptance.json',result)
    atomic_json(out/'BLOCKED.json',result)
    print(json.dumps(result),flush=True)
    return 78


def main():
    path=Path(sys.argv[1]);protocol=read(path);out=Path(protocol['namespace']);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'EXECUTION_STARTED.json',dict(protocol_sha256=sha(path),cpu_only=True,
        producer=True,unix=time.time(),scope='existing frozen evidence processing; no GPU training'))
    code={'bank_coverage':bank,'funnel_acceptance':funnel,'bank_ratio':ratio}[protocol['kind']](protocol,out,path)
    return code or 0


if __name__=='__main__':
    raise SystemExit(main())
