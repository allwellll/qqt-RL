"""Prepare the authorized release and fixed replay selection, without publishing."""
import copy,json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1');O=B/'publish_a3sr_u1280_20261030';R=B/'replay_a3sr_u1280'

def main():
    C=B/'fresh256_a3sr_vs_u800_20261030';proto=json.loads((C/'PROTOCOL.json').read_text());result=json.loads((C/'RESULTS.json').read_text())
    assert json.loads((C/'COMPLETE.json').read_text())['result_sha256']==sha(C/'RESULTS.json')
    assert json.loads((C/'PROGRESS.json').read_text())['completed_shards']==192
    role='a3sr_u1280';cp=B/'formal/bt_slot05_c000028/u001280';cm=json.loads((cp/'COMMITTED.json').read_text())
    source=proto['models'][role];assert source['checkpoint']==str(cp) and sha(cp/'policy.pt')==cm['policy_sha256']==source['policy_sha256']
    metrics=result['metrics'];p1={t:metrics[t][role]['by_seat']['P1']['win_minus_loss'] for t in proto['tiers']};old={t:metrics[t]['u800']['by_seat']['P1']['win_minus_loss'] for t in proto['tiers']}
    assert sum(p1.values())>sum(old.values()) and all(p1[t]>=old[t]-5 for t in p1)
    original=Path(source['model_json']);doc=json.loads(original.read_text());graph=original.with_name('model.onnx')
    assert sha(original)==source['model_sha256'];assert sha(graph)==doc['meta']['onnx_sha256']
    assert doc['meta']['checkpoint_sha256']==cm['policy_sha256']
    doc['meta'].pop('onnx_path',None)
    doc['meta'].update(name='Bun392 A3SR u1280',display_name='Bun392 A3SR u1280',source=str(cp/'policy.pt'),
        committed_sha256=sha(cp/'COMMITTED.json'),strategy='raw',state_dim=24,attention_heads=4,update=1280,
        generated_at='2026-10-30T00:00:00Z',rule_version='stock-window-v1')
    data=(json.dumps(doc,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode();import hashlib
    jhash=hashlib.sha256(data).hexdigest();ghash=sha(graph)
    name=f'bun392-a3sr-u1280-{jhash}.json';gname=f'bun392-a3sr-u1280-{ghash}.onnx'
    (O/name).write_bytes(data);(O/gname).write_bytes(graph.read_bytes())
    tag='web-bun392-a3sr-u1280-20261030';url=f'https://github.com/allwellll/qqt-RL/releases/download/{tag}/'
    games={t:{r:{k:metrics[t][r][k] for k in ('win','draw','loss','win_minus_loss')} for r in (role,'u800')} for t in metrics}
    summary='；'.join(f"{t} {games[t][role]['win']}/{games[t][role]['draw']}/{games[t][role]['loss']}" for t in ('hard','normal','easy'))
    row=dict(id='bun392-a3sr-u1280',display_name='Bun392 A3SR u1280（fresh256 P1：hard +13 / normal +26 / easy +52）',
        file=name,sha256=jhash,bytes=len(data),url=url+name,candidate='bomb_threat_v1_slot05_c000028_u001280_raw',cycle=1280,
        source_sha256=cm['policy_sha256'],strategy='raw',score=metrics['hard'][role]['win']/256,
        description=f'网页真人坐P0，模型坐P1；stock-window-v1，保持argmax。fresh256 P1胜-负：hard +13、normal +26、easy +52，合计+91；u800同seed对照0/+6/+11，合计+17。两座位合计胜/平/负（每档256局）：{summary}。对手为猎手（bun.hunter），不是协作猎手。已知短板：左右振荡、持包被砖困；部分击杀/安全指标未全面不差，不宣称普遍提升。网页fp32与训练bf16存在数值/动作差异。',
        score_protocol='production JS bun.hunter; stock-window-v1; argmax; teacher off; fixed paired 256 games per tier, P0/P1 128 each; hard/normal/easy seeds2026142900/2026143900/2026144900; score=hard wins/256, not comparable with historical scores',
        onnx=dict(file=gname,sha256=ghash,bytes=graph.stat().st_size,url=url+gname,meta=doc['meta']))
    # reference_zero is filled by the actual JS verifier, not guessed.
    atomic_json(O/'CATALOG_ROW.json',row)
    atomic_json(O/'manifest.json',dict(schema='qqt.bun-release/v1',source_ref=str(cp),policy_sha256=cm['policy_sha256'],committed_sha256=sha(cp/'COMMITTED.json'),
        rule_version='stock-window-v1',strategy='raw',generated_at='2026-10-30T00:00:00Z',tag=tag,
        assets=[dict(file=name,sha256=jhash,bytes=len(data)),dict(file=gname,sha256=ghash,bytes=graph.stat().st_size)],
        raw_tensors_unchanged=doc['flat']==json.loads(original.read_text())['flat'],metrics=games,P1=p1,u800_P1=old))
    (O/'SHA256SUMS').write_text(f'{jhash}  {name}\n{ghash}  {gname}\n{sha(O/"manifest.json")}  manifest.json\n')
    # Selection uses complete frozen hard outcomes in index order, never rollout searching.
    rows=[];primary={};files={}
    for i in range(0,256,8):
        p=C/'hard'/role/f'greedy_{i:03d}.json';q=Path(str(p)+'.outcome.json')
        a=json.loads(p.read_text());d=json.loads(q.read_text());rows+=d['episodes']
        primary.update({e['seed']:e for e in a['results']['natural_full2400']['episodes']});files[str(q)]=sha(q)
    rows.sort(key=lambda x:x['seed']);assert len(rows)==256
    selected=[];missing=[]
    def add(r,label):
        e=copy.deepcopy(r);e['primary_delivery']=primary[e['seed']]['delivery']
        selected.append(dict(index=e['seed']-proto['tiers']['hard']['seed'],seed=e['seed'],seat=e['seat'],rule=label,expected=e))
    for seat in (0,1):
        for r in [x for x in rows if x['seat']==seat][:4]:add(r,f'顺序前4局P{seat}；与旧u800页01–08同seed')
    selected.sort(key=lambda x:x['index'])
    for seat in (0,1):
        wins=[x for x in rows if x['seat']==seat and x['winner']==seat][:3]
        for r in wins:add(r,f'P{seat}按index最先3个胜局')
        losses=[x for x in rows if x['seat']==seat and x['winner'] not in (None,-1,seat)][:1]
        for r in losses:add(r,f'P{seat}按index最先1个负局')
        if not losses:missing.append(f'P{seat}负局无')
    zero=[r for r in rows if r['seat']==1 and r['placed'][1]==0 and r['placed'][0]>0][:2]
    for r in zero:add(r,'P1按index最先2个单方零泡局')
    if len(zero)<2:missing.append(f'P1单方零泡仅{len(zero)}局')
    brick=json.loads((B/'iter2/a3sr/BRICK_OBSERVATION_MANIFEST.json').read_text())
    for g in brick['games'][:3]:add(next(r for r in rows if r['seed']==g['seed'] and r['seat']==g['seat']),'u800砖隔开清单前3个同seed')
    manifest=dict(status='FROZEN_BEFORE_RECORDING',model=source,games=selected,missing=missing,source_files=files,
        spawn_manifest=proto['tiers']['hard']['spawn_manifest'],selection_instruction='.hermes/TASK_publish_a3sr_u1280_20261030.md',
        model_label='A3SR u1280 · 网页默认发布候选',html='20261030-a3sr-u1280-review.html',teacher=False,rule_version='stock-window-v1',unix=time.time())
    atomic_json(R/'MANIFEST.json',manifest)
    atomic_json(O/'TASK_CONSUMED.json',dict(status='GATE_PASS_ASSETS_PREPARED',instruction='.hermes/TASK_publish_a3sr_u1280_20261030.md',P1=p1,u800_P1=old,training_untouched=True,unix=time.time()))
    print(json.dumps(dict(gate='PASS',P1=p1,u800_P1=old,assets=[name,gname],replay_entries=len(selected),unique_games=len(set(x['index'] for x in selected)),missing=missing)))
if __name__=='__main__':main()
