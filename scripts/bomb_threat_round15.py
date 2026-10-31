"""Round15 donor admission and one bounded, atomic slot07 A3SR handoff."""
import json
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'
OUT = BASE / 'iter2/a3sr/round15'
AUTH = OUT / 'AUTHORIZATION.json'
INSTRUCTION = '.hermes/STEER_fresh_p1zero_slot07_20261030_0610.md'
CORRECTION = '.hermes/STEER_fix_20261030_0615.md'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def admission(rows):
    assert len(rows) == len({r['seed'] for r in rows}) == 64
    assert all(r['done'] and r['seat'] in (0, 1) for r in rows)
    wins = sum(r['winner'] == r['seat'] for r in rows)
    losses = sum(r['winner'] == 1-r['seat'] for r in rows)
    zero = sum(r['placed'][r['seat']] == 0 and r['placed'][1-r['seat']] > 0 for r in rows)
    return dict(win=wins, draw=64-wins-losses, loss=losses, win_minus_loss=wins-losses,
                unilateral_zero=zero, eligible=wins-losses >= 25 and zero <= 4)


def pin(checkpoints, reason):
    path = BASE / 'STATE_RETENTION_PINS.json'
    doc = read(path, dict(checkpoints=[]))
    known = {r['checkpoint'] for r in doc['checkpoints']}
    for cp in checkpoints:
        if cp not in known:
            doc['checkpoints'].append(dict(checkpoint=cp, reason=reason))
            known.add(cp)
    atomic_json(path, doc)


def prepare():
    if AUTH.exists():
        return read(AUTH)
    slots = read(BASE/'SLOTS.json')
    donor_run = slots['5']['run']
    proto = read(BASE/'protocols'/f'{donor_run}.json')
    assert proto['arms'][donor_run]['iter2_arm'] == 'A3SR'
    # Pin all remaining cadence points before their checkpoint writes, not
    # after evaluation has completed and a cleanup could already have run.
    candidates = [str((BASE/'formal'/donor_run/f'u{u:06d}').relative_to(ROOT))
                  for u in range(1080, 1281, 40)]
    pin(candidates, 'round15 admission candidate; protected before evaluation')
    boundary_proto = read(BASE/'protocols'/f"{slots['7']['run']}.json")
    boundary_end = boundary_proto['chunk_start_update']+boundary_proto['chunk_updates']
    pin([str((BASE/'formal'/slots['7']['run']/f'u{boundary_end:06d}').relative_to(ROOT))],
        'round15 slot07 old-line final complete state; protect before switch')
    value = dict(status='AUTHORIZED_WAITING_ELIGIBLE_DONOR', instruction=INSTRUCTION,
        instruction_sha256=sha(ROOT/INSTRUCTION), correction=CORRECTION,
        correction_sha256=sha(ROOT/CORRECTION), donor_run=donor_run,
        slot07_boundary_run=slots['7']['run'], candidates=candidates,
        target_slot=7, target_seed=2026112907, collector_seed=2026112905,
        chunk_updates=320, end_policy='one fixed320-update chunk then await explicit extension',
        budget_interpretation='new slot07 chunk keeps existing320-update scheduler budget; slot05 remains capped1280',
        copied=['Actor','Critic','Adam','env','aux','ring','ring_cursor','fixed_actor','counters'],
        rng='exact donor next-step parity first, then independent slot07 seed rekey',
        critic_warmup_updates=0, no_promotion=True, no_publication=True, unix=time.time())
    atomic_json(AUTH, value)
    atomic_json(OUT/'BUDGET_AMENDMENT01.json', dict(status='FROZEN',
        reason='instruction authorizes a new chunk but specifies no changed length; retain320 updates',
        chunk_updates=320, target_slot=7, other_slots_unchanged=True, unix=time.time()))
    return value


def select_donor(auth):
    if auth.get('donor') and (OUT/'SLOT07_HANDOFF.json').exists():
        return auth['donor']
    evidence = []
    for cpname in reversed(auth['candidates']):
        cp = ROOT/cpname
        if not (cp/'state.pkl').exists():
            continue
        cm = read(cp/'COMMITTED.json')
        out = BASE/'eval'/f"{auth['donor_run']}_{cp.name}"
        paths = sorted(out.glob('natural_greedy_*.json.outcome.json'))
        if len(paths) != 8:
            continue
        rows = []
        for path in paths:
            side = read(path)
            assert side['rule_version'] == 'stock-window-v1' and not side['teacher']
            primary = read(Path(str(path).removesuffix('.outcome.json')))
            assert primary['model_json_sha256'] == sha(out/'model/model.json') and not primary['assist']
            episodes = primary['results']['natural_full2400']['episodes']
            assert len(episodes) == len(side['episodes']) == 8
            for r, e in zip(side['episodes'], episodes):
                assert (r['seed'], r['seat'], r['ticks']) == (e['seed'], e['team'], e['ticks'])
            rows.extend(side['episodes'])
        assert {r['seed'] for r in rows} == set(range(2026122900, 2026122964))
        score = admission(rows)
        evidence.append(dict(checkpoint=cpname, **score))
        atomic_json(OUT/'donor_admissions'/f'{cp.name}.json',dict(checkpoint=cpname, **score,
            files={str(p.relative_to(ROOT)):sha(p) for p in paths}))
        atomic_json(OUT/'DONOR_ADMISSION.json', dict(candidates=evidence, unix=time.time()))
        if not score['eligible']:
            continue
        if auth.get('donor', {}).get('checkpoint') == cpname:
            return auth['donor']
        for f, key in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:
            assert sha(cp/f) == cm[key]
        assert cm['metrics']['finite'] and min(cm['metrics']['update_l2'].values()) > 0
        prior=auth.get('donor')
        auth.update(status='AUTHORIZED_DONOR_FROZEN', donor=cm, donor_dev=score,
            donor_selection='latest complete eligible before handoff; no replacement after handoff',
            donor_evidence={str(p.relative_to(ROOT)):sha(p) for p in paths}, unix=time.time())
        if prior:auth.setdefault('superseded_provisional_donors', []).append(prior)
        atomic_json(AUTH, auth)
        atomic_json(BASE/'notifications/ROUND15_DONOR_READY.json', auth)
        from scripts.iter_pbt_owner import notify
        notify('ROUND15_DONOR_READY', f"round15 donor u{cm['update']}已pin并核验SHA；dev胜负差{score['win_minus_loss']:+d}、单方零泡{score['unilateral_zero']}，等待slot07当前chunk原子终态。")
        return cm
    return None


def make_protocol(auth, chunk):
    source = auth['donor']
    donor_run = Path(source['checkpoint']).parent.name
    template = read(BASE/'protocols'/f'{donor_run}.json')
    recipe = template['arms'][donor_run]
    assert recipe['iter2_arm'] == 'A3SR'
    name = f'bt_slot07_c{chunk:06d}'
    path = BASE/'protocols'/f'{name}.json'
    if path.exists():
        existing = read(path)
        assert existing['continuation_source'] == source and existing['arms'][name] == recipe
        return path
    assert all(sha(ROOT/f) == h for f, h in template['source_sha256'].items())
    proto = dict(template)
    if 'round14_authorization' in proto:
        proto['ancestor_round14_authorization'] = proto.pop('round14_authorization')
    proto.update(namespace_id='A3SR-round15-slot07', arms={name:recipe},
        sources={str(auth['target_seed']):source}, continuation_source=source,
        chunk_start_update=source['update'], chunk_updates=auth['chunk_updates'],
        environment_steps_per_run=(source['update']+auth['chunk_updates'])*262144,
        slot=7, chunk=chunk, worker_module='scripts.bomb_threat_a3sr_resume_train',
        source_protocol=str((BASE/'protocols'/f'{donor_run}.json').relative_to(ROOT)),
        source_protocol_sha256=sha(BASE/'protocols'/f'{donor_run}.json'),
        round15_authorization=str(AUTH.relative_to(ROOT)),
        round14_collector_seed=auth['collector_seed'], round14_rekey=True,
        initialization='complete donor state parity then slot07 RNG rekey; no warmup or reward reset',
        pbt_lineage=dict(donor=source['checkpoint'], donor_update=source['update'],
            state_sha256=source['state_sha256'], policy_sha256=source['policy_sha256'],
            target_slot=7, target_seed=auth['target_seed'],
            rng=auth['rng'], copied=auth['copied'], critic_warmup_updates=0,
            formal_updates_discarded=0))
    atomic_json(path, proto)
    return path


def apply_switch(slot, entry, done, chunk):
    auth = read(AUTH, {})
    if slot != 7 or entry.get('round15_applied') or entry['run'] != auth.get('slot07_boundary_run'):
        return None
    source = select_donor(auth)
    if not source:
        atomic_json(OUT/'SLOT07_WAIT.json', dict(status='WAITING_ELIGIBLE_DONOR',
            old_last_atomic=done, no_subthreshold_copy=True, unix=time.time()))
        return None
    pin([done['checkpoint']], 'round15 replaced slot07 final complete state')
    for r in (done, source):
        for f,k in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:
            assert sha(ROOT/r['checkpoint']/f) == r[k]
    path = make_protocol(read(AUTH), chunk)
    receipt = dict(status='PROTOCOL_READY', protocol=str(path.relative_to(ROOT)),
        protocol_sha256=sha(path), source=source, old_last_atomic=done,
        old_state_retained=True, training_worker_signals=0, unix=time.time())
    atomic_json(OUT/'SLOT07_HANDOFF.json', receipt)
    atomic_json(BASE/'formal'/entry['run']/'ROUND15_SUCCESSOR.json', receipt)
    return dict(entry, run=path.stem, chunk=chunk, source=source, iter2_arm='A3SR',
        role='A3SR_round15_PBT_copy', round15_applied=True, round14_applied=True,
        round15_end_update=source['update']+auth['chunk_updates'], supersedes=entry['run'])


def terminal(entry, done):
    if not entry.get('round15_applied'):
        return False
    assert done['update'] == entry['round15_end_update']
    path = OUT/'SLOT07_TERMINAL.json'
    if not path.exists():
        value = dict(status='FIXED320_COMPLETE', run=entry['run'], last_atomic=done,
            next='await explicit extension; other authorized slots unchanged', unix=time.time())
        atomic_json(path, value)
        atomic_json(BASE/'notifications/ROUND15_SLOT07_TERMINAL.json', value)
        from scripts.iter_pbt_owner import notify
        notify('ROUND15_SLOT07_TERMINAL', f"round15 slot07 {entry['run']} u{done['update']}固定320更新完成并原子保存；未晋升，其他训练继续。")
    return True


def refresh():
    auth = read(AUTH, {})
    if not auth:
        return
    select_donor(auth)
    handoff = read(OUT/'SLOT07_HANDOFF.json', {})
    target = OUT/'SLOT07_FIRST_FORMAL.json'
    if not handoff or target.exists():
        return
    out = BASE/'formal'/Path(handoff['protocol']).stem
    smoke, restore = read(out/'SMOKE.json', {}), read(out/'RESTORE.json', {})
    if not (smoke.get('passed') and restore.get('passed')):
        return
    rows = [json.loads(l) for l in (out/'updates.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['new_update'] > handoff['source']['update']+2]
    if not rows:
        return
    r = rows[0]
    assert r['finite'] and min(r['update_l2'].values()) > 0
    value = dict(status='PASS', run=out.name, donor_update=handoff['source']['update'],
        first_post_restore_update=r['new_update'], update_l2=r['update_l2'], finite=True,
        environment_steps=r['environment_steps'], env_per_second=262144/r['cycle_seconds'],
        smoke=smoke, restore=restore, source_handoff=read(out/'SOURCE_HANDOFF.json'), unix=time.time())
    atomic_json(target, value)
    atomic_json(BASE/'notifications/ROUND15_SLOT07_FIRST_FORMAL.json', value)
    from scripts.iter_pbt_owner import notify
    notify('ROUND15_SLOT07_FIRST_FORMAL', f"round15 slot07从u{value['donor_update']}复制，正式u{r['new_update']} finite、Actor/Critic非零，{value['env_per_second']:.0f}env/s；完整恢复通过，未晋升。")


if __name__ == '__main__':
    prepare()
    refresh()
