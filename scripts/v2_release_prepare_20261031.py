"""Prepare only the complete fixed-matrix winner. No remote or Git writes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import base64
import hashlib
import json
import pickle
import re
from pathlib import Path
import numpy as np
from scripts.population_run_state import atomic_json,sha
from scripts.export_bun_web_model import extract_transformer

B=Path(os.environ.get('V2_SELECTION_OUTPUT','runs/v2_fresh_selection_20261031'))
O=B/'publication'
W=Path('../cx_publish_r1_1805_20261030')
def read(f):return json.loads(Path(f).read_text())
def main():
 O.mkdir(exist_ok=True)
 done=read(B/'COMPLETE.json');assert done['status']=='COMPLETE' and done['games']==6144
 assert sha(B/'RESULTS.json')==done['results_sha256']
 results=read(B/'RESULTS.json');selection=results['selection']
 if not selection['eligible_for_conditional_publish']:
  atomic_json(O/'STATUS.json',dict(status='RETAIN_BASELINE_NO_ELIGIBLE_CANDIDATE',selection=selection));return
 protocol=read(B/'PROTOCOL.json');label=selection['winner'];m=results['models'][label];cp=Path(m['checkpoint']);cm=read(cp/'COMMITTED.json')
 assert sha(cp/'policy.pt')==m['policy_sha256']==cm['policy_sha256'] and cm['passed']
 assert sha(m['model_json'])==m['model_sha256'] and sha(m['onnx_path'])==m['onnx_sha256']
 doc=read(m['model_json']);assert doc['meta']['checkpoint_sha256']==m['policy_sha256']
 # Compare every deployed packed fp32 tensor to the actual checkpoint Actor.
 with (cp/'policy.pt').open('rb') as f:params=pickle.load(f)
 flat=np.frombuffer(base64.b64decode(doc['flat']),dtype='<f4');expected=extract_transformer(params)
 assert set(expected)==set(doc['tensors'])
 for name,value in expected.items():
  start,count=doc['tensors'][name];assert np.array_equal(flat[start:start+count],value.reshape(-1)),name
 ident=f'bun392-v2-{label.replace("_","-")}-u{m["update"]}'
 tag=f'web-{ident}-20261031';meta=doc['meta'];meta.pop('onnx_path',None)
 meta.update(name=ident,display_name=ident,source=str(cp/'policy.pt'),committed_sha256=sha(cp/'COMMITTED.json'),
             strategy='raw',attention_heads=4,update=m['update'],rule_version='bun-native-v2')
 if m['r3']:assert meta['obs_shape']==[25,13,15] and meta['state_dim']==37
 else:assert meta['obs_shape']==[24,13,15] and meta['state_dim']==24
 data=(json.dumps(doc,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
 h=hashlib.sha256(data).hexdigest();name=f'{ident}-{h}.json';graph=f'{ident}-{m["onnx_sha256"]}.onnx'
 (O/name).write_bytes(data);(O/graph).write_bytes(Path(m['onnx_path']).read_bytes())
 chosen=selection['comparison'][label];baseline=selection['comparison']['baseline'];release=f'https://github.com/allwellll/qqt-RL/releases/download/{tag}/'
 desc=(f'网页真人P0、模型P1。bun-native-v2自由取包，生产JS Hunter、argmax、老师关闭。固定三档每档256局（各席128），P1三档胜负差{chosen["P1"]}，合计{chosen["P1_sum"]:+d}（网页旧Actor在v2下{baseline["P1_sum"]:+d}）；P0合计{chosen["P0_sum"]:+d}（基线{baseline["P0_sum"]:+d}）；自伤{chosen["self_deaths"]}。逐档/P0/自伤护栏全部通过。协作Hunter hard单列压力测试，不混排名。{16 if read(B/'PROTOCOL.json').get('spawn_side_version') else 64}出生几何重复；不宣称统计显著或真人泛化。旧v1可选、旧录像按原规则回放，成绩分榜不混排。R3如有使用25平面/37维可见输入；真人P0 frameStep与评估Sim.step范围分开，不宣称逐位等价。仍可能有零泡、振荡与持包停滞。')
 hard=results['metrics'][label]['hunter_hard'];score=sum(hard[s]['wins'] for s in ('P0','P1'))/256
 row=dict(id=ident,display_name=f'Bun392 v2 {label}（P1三档{chosen["P1_sum"]:+d}）',file=name,sha256=h,bytes=len(data),url=release+name,
          cycle=m['update'],candidate=cp.parent.name+'_'+cp.name,strategy='raw',score=score,source_sha256=m['policy_sha256'],
          rule_version='bun-native-v2',description=desc,
          score_protocol='production JS Hunter three tiers; bun-native-v2; argmax; teacher off; paired256/tier, P0/P1 128 each; fresh'+('/'.join(str(protocol['conditions'][k]['seed']) for k in ['hunter_hard','hunter_normal','hunter_easy']))+'; '+('16 own-side native geometries reused' if protocol.get('spawn_side_version') else '64 geometries reused')+'; coop_hard separate',
          onnx=dict(file=graph,sha256=m['onnx_sha256'],bytes=(O/graph).stat().st_size,url=release+graph,meta=meta))
 atomic_json(O/'CATALOG_ROW.json',row)
 atomic_json(O/'manifest.json',dict(schema='qqt.bun-release/v1',tag=tag,rule_version='bun-native-v2',source_ref=m['checkpoint'],
  policy_sha256=m['policy_sha256'],committed_sha256=sha(cp/'COMMITTED.json'),selection_sha256=sha(B/'RESULTS.json'),raw_tensors_unchanged=True,
  assets=[{k:x[k] for k in ('file','sha256','bytes')} for x in (row,row['onnx'])]))
 atomic_json(O/'EXPORT_EXACT.json',dict(status='PASS',tensor_count=len(expected),packed_weights_exact=True,policy_sha256=m['policy_sha256'],model_sha256=h,onnx_sha256=m['onnx_sha256'],new_input_abi=meta['obs_shape']))
 (O/'SHA256SUMS').write_text(''.join(f'{x["sha256"]}  {x["file"]}\n' for x in (row,row['onnx'])))
 atomic_json(O/'STATUS.json',dict(status='ASSETS_PREPARED_PENDING_NUMERICAL_BROWSER_DATABASE_GATES',winner=label,tag=tag))
 print(json.dumps({'winner':label,'tag':tag,'json_sha256':h,'tensors':len(expected)}))
if __name__=='__main__':main()
