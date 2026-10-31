"""Frozen u4002 evidence reuse, bounded pressure addendum and exact release."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import inspect
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha

B=Path('runs/r3_resume_publish_2110_20261031/selection')
OLD=Path('runs/v2_ownside_selection_20261031')
R3=Path('runs/r3_items_route_20261030')
W=Path('../cx_publish_r1_1805_20261030')
def read(p):return json.loads(Path(p).read_text())

def freeze():
    B.mkdir(parents=True,exist_ok=True)
    if (B/'PROTOCOL.json').exists():return read(B/'PROTOCOL.json')
    p=read(OLD/'PROTOCOL.json')
    models={'baseline':read(OLD/'MODELS.json')['r3_a_u3040']}
    for arm in ('a','b'):
        folder=R3/'evaluation_ownside_v1'/f'r3_{arm}_ownside_c000001_fresh'
        identity=read(folder/'model/IDENTITY.json');cm=identity['source']
        assert cm['update']==4002
        doc=read(identity['model_json']);meta=doc['meta']
        models[f'r3_{arm}_u4002']=dict(checkpoint=cm['checkpoint'],policy_sha256=cm['policy_sha256'],
            model_json=identity['model_json'],model_sha256=identity['model_sha256'],
            onnx_path=meta['onnx_path'],onnx_sha256=meta['onnx_sha256'],update=4002,r3=True)
    models['baseline'].update(r3=True,update=3040)
    for m in models.values():
        assert sha(Path(m['checkpoint'])/'policy.pt')==m['policy_sha256']
        assert sha(Path(m['model_json']))==m['model_sha256']
        assert sha(Path(m['onnx_path']))==m['onnx_sha256']
    p.update(id='R3-A4002-FROZEN-RELEASE-2110-20261031',models=models,candidate_order=['r3_a_u4002','r3_b_u4002'],
        budget=2816,primary_games=2304,pressure_games=512,pressure_labels=['baseline','r3_a_u4002'],
        reuse_games=2560,new_games=256,authorization='.hermes/STEER_2110_publish_a4002_fill_gpu.md',
        selection_note='existing frozen natural3x256 per model; pressure is separate, no new promotion gate',
        freeze_after_observation=True,selection_rule_unchanged=True,unix=time.time())
    atomic_json(B/'PROTOCOL.json',p);atomic_json(B/'MODELS.json',models)
    return p

def install_reused(p,ev):
    for label in p['models']:
        for c,v in p['conditions'].items():
            if c=='coop_hard' and label not in p['pressure_labels']:continue
            if label=='baseline':source=OLD/'shards'/c/'r3_a_u3040'
            elif c=='coop_hard':continue
            else:source=R3/'evaluation_ownside_v1'/f"r3_{label[3]}_ownside_c000001_fresh"/c.removeprefix('hunter_')/'greedy'
            dest=B/'shards'/c/label;dest.mkdir(parents=True,exist_ok=True)
            for start in range(0,256,8):
                f=source/f'shard_{start:03d}.json';proof=read(str(f)+'.ACCEPTED.json')
                assert all(sha(Path(a))==h for a,h in proof['files'].items())
                for suffix in ('','.outcome.json'):
                    target=dest/(f.name+suffix)
                    if not target.exists():shutil.copy2(str(f)+suffix,target)
                    assert sha(target)==sha(Path(str(f)+suffix))
                j=dict(label=label,condition=c,start=start,**v)
                rows=ev.validate(j,p['models'][label]);assert len(rows)==8
                atomic_json(str(dest/f.name)+'.ACCEPTED.json',dict(status='PASS',job=j,
                    source=str(f),files={str(dest/(f.name+s)):sha(dest/(f.name+s)) for s in ('','.outcome.json')},
                    original_proof_sha256=sha(Path(str(f)+'.ACCEPTED.json')),protocol_sha256=sha(B/'PROTOCOL.json')))
            trace=source/f"fixed-natural_full2400-{v['seed']}.trace.json"
            if trace.exists() and not (dest/trace.name).exists():shutil.copy2(trace,dest/trace.name)

def evaluate():
    from scripts import v2_fresh_selection_20261031 as ev
    p=freeze();ev.OUT=B
    original_jobs=ev.jobs
    ev.jobs=lambda protocol:[j for j in original_jobs(protocol) if j['condition']!='coop_hard' or j['label'] in protocol['pressure_labels']]
    install_reused(p,ev)
    if not (B/'RESULTS_PRIMARY.json').exists():
        partial=ev.aggregate(p,p['models'],False)
        selection=ev.select(partial['metrics'],p)
        assert selection['winner']=='r3_a_u4002' and selection['eligible_for_conditional_publish']
        expected={'r3_a_u4002':(338,353),'r3_b_u4002':(316,None),'baseline':(320,345)}
        for k,(p1,p0) in expected.items():
            actual=selection['comparison'][k]
            assert actual['P1_sum']==p1
            if p0 is not None:assert actual['P0_sum']==p0
        atomic_json(B/'RESULTS_PRIMARY.json',dict(status='PASS',games=2304,selection=selection,files=partial['files'],unix=time.time()))
    todo=[j for j in ev.jobs(p) if not Path(str(ev.path(j))+'.ACCEPTED.json').exists()]
    atomic_json(B/'EXECUTION_STATUS.json',dict(status='PRESSURE_RUNNING',reused_games=2560,new_games_remaining=len(todo)*8,workers=8,unix=time.time()))
    def collect(pair):
        import subprocess
        j,core=pair
        # Child processes inherit an exact, bounded single-core affinity.
        import threading
        os.sched_setaffinity(threading.get_native_id(),{core})
        return ev.collect(j,p['models'][j['label']])
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(collect,[(j,72+i%8) for i,j in enumerate(todo)]))
    result=ev.aggregate(p,p['models'],True)
    atomic_json(B/'COMPLETE.json',dict(status='COMPLETE',games=result['games'],results_sha256=sha(B/'RESULTS.json'),selection=result['selection'],unix=time.time()))
    return result

def prepare():
    from scripts import v2_release_prepare_20261031 as exp
    exp.B=B;exp.O=B/'publication';exp.W=W;exp.O.mkdir(exist_ok=True)
    # The exact old export contract is retained; only its historical6-model
    # count assertion is parameterized to this declared2816-game matrix.
    source=inspect.getsource(exp.main)
    anchor="done['games']==6144";assert source.count(anchor)==1
    namespace=dict(vars(exp));exec(compile(source.replace(anchor,"done['games']==2816"),'<u4002-export-contract>','exec'),namespace)
    namespace['main']()
    (B/'hard_spawns.json').write_bytes(Path(read(B/'PROTOCOL.json')['conditions']['hunter_hard']['spawn_manifest']).read_bytes())
    from scripts import v2_web_release_20261031 as release
    release.B=B;release.O=exp.O;release.W=W
    os.environ['V2_SELECTION_OUTPUT']=str(B)
    release.run(['node','scripts/verify_v2_release_export_20261031.js'],'JS_ONNX.log')
    release.run([__import__('sys').executable,'-m','scripts.v2_release_fp32_audit_20261031'],'PYTHON_PARITY.log')
    row=read(exp.O/'CATALOG_ROW.json')
    old=read(W/'web/models.json')
    if old['models'][0]['id']!=row['id']:
        assert old['models'][0]['source_sha256']==read(B/'MODELS.json')['baseline']['policy_sha256']
        atomic_json(exp.O/'ONLINE_CATALOG_BEFORE.json',old)
        atomic_json(W/'web/models.json',dict(old,models=[row,*old['models']]))
    for asset in (row,row['onnx']):
        dest=W/'web/models'/asset['file'];dest.parent.mkdir(exist_ok=True)
        if not dest.exists():dest.symlink_to((exp.O/asset['file']).resolve())
        assert sha(dest)==asset['sha256']
    import re
    f=W/'web/test_model_catalog.js';s=f.read_text()
    for key,value in [('id',row['id']),('source_sha256',row['source_sha256']),('score',row['score'])]:
        s=re.sub(r'assert\.strictEqual\(current\.'+key+r',[^;]+;',f'assert.strictEqual(current.{key},{json.dumps(value)});',s)
    f.write_text(s)
    f=W/'web/test_http.js';s=f.read_text();catalog=read(W/'web/models.json')
    s=re.sub(r'assert\.strictEqual\(manifest\.models\.length,\s*\d+\);',f'assert.strictEqual(manifest.models.length, {len(catalog["models"])});',s)
    s=re.sub(r'assert\.strictEqual\(manifest\.models\[0\]\.id,\s*[^;]+;',f'assert.strictEqual(manifest.models[0].id, {json.dumps(row["id"])});',s);f.write_text(s)
    replay=read('runs/new_model_replays_20261031/RECEIPT.json');assert replay['status']=='COMPLETED'
    atomic_json(exp.O/'REPLAY.json',dict(base='runs/new_model_replays_20261031',html='20261031-new-r3-sp-models-review.html',url=replay['url'],html_sha256=replay['html_sha256']))
    atomic_json(B/'PUBLICATION_PROTOCOL.json',dict(status='AUTHORIZED',allowed_files=['web/models.json','web/test_model_catalog.js','web/test_http.js'],ruleset_unchanged=True,authorization='.hermes/STEER_2110_publish_a4002_fill_gpu.md'))
    return release

def main():
    B.mkdir(parents=True,exist_ok=True);guard=lock(B/'OWNER.lock')
    if (B/'publication/DONE.json').exists():return
    evaluate();release=prepare()
    if not release.preflight():return
    # Preserve the earlier u3040 release report and do not relabel its6144
    # games as this run's denominator. Reuse the verified transport verbatim.
    source=inspect.getsource(release.release)
    source=source.replace('reports/publish_best_bun_native_v2_20261031.md','reports/publish_r3_a_u4002_20261031.md')
    source=source.replace('完整6144局比较见本轮本侧出生复核报告','本轮2816完整局（2304三档主矩阵、512协作hard压力；新增256、复用2560）见恢复发布报告')
    source=source.replace('Publish paired-validated bun-native-v2 default with separate ruleset boards','Publish paired-validated R3-a u4002 default; bun-native-v2 unchanged')
    namespace=dict(vars(release));exec(compile(source,'<u4002-release-contract>','exec'),namespace)
    namespace['release']()

if __name__=='__main__':
    try:main()
    except Exception as error:
        B.mkdir(parents=True,exist_ok=True)
        atomic_json(B/'BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason=str(error),training_signals=0,unix=time.time()))
        raise
