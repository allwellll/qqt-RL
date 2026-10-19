"""Read only the five declared completed checkpoints, never live training state."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
MODELS=[('r01-g8','control_route_v38_r01_attempt04',8,'综合发布基线'),('r01-g128','control_route_v38_r01_attempt04',128,'完整运包'),('r13-g128','control_route_v38_r13_causal_clean',128,'干净因果链·完整运包参考'),('r16-g128','control_route_v38_r16_causal_clean',128,'战斗专项·注意自伤'),('r17-g128','control_route_v38_r17_causal_clean',128,'最新完成·库存恢复参考')]
rows=[];protocol=None
for mid,run,g,label in MODELS:
 r=a.runs/run;ep=r/f'authority_je10/g{g:06d}';cp=r/f'formal/bun392_bridge_pfsp8/gen{g:06d}'
 ph=sha(r/'authority_je10/PROTOCOL.json');assert protocol is None or protocol==ph;protocol=ph
 c=json.loads((ep/'COMPLETE.json').read_text());s=json.loads((ep/'SUMMARY.json').read_text());m=json.loads((cp/'COMMITTED.json').read_text());seal=json.loads((r/'SOURCE_SEAL.json').read_text());plan=json.loads((r/'plan.json').read_text())
 assert c['passed'] and m['passed'] and m['generation']==g and s['generation']==g
 for name,h in c['shards'].items():assert sha(ep/name)==h
 assert sha(cp/'policy.pt')==c['checkpoint_sha256']==s['policy_sha256']==m['policy_sha256']
 assert sha(cp/'state.pkl')==m['state_sha256'];assert sha(r/'plan.json')==seal['plan_sha256']==m['plan_sha256']
 for name,h in seal['sources'].items():assert sha(r/'inputs/sealed_sources'/name)==h
 if g==128:
  done=json.loads((r/'TRAINING_COMPLETE.json').read_text());assert done['passed'] and done['generation']==g
 rows.append(dict(id=mid,source_run=run,generation=g,label=label,checkpoint_sha256=m['policy_sha256'],state_sha256=m['state_sha256'],plan_sha256=seal['plan_sha256'],source_seal_sha256=sha(r/'SOURCE_SEAL.json'),sealed_source_count=len(seal['sources']),complete_sha256=sha(ep/'COMPLETE.json'),summary_sha256=sha(ep/'SUMMARY.json'),evaluation_shards=c['shards'],gamma=plan['gamma'],route_coefficient=plan['route_pbrs']['lambda_value'],hunter=s['hunter_js'],tactical=s['tactical_python'],route={k:{q:v.get(q) for q in ['games','pickup_episodes','delivery_episodes','death','self_death']} for k,v in s['route'].items()}))
 print(mid,'COMPLETE, checkpoint/state, plan and',len(seal['sources']),'sealed sources verified',flush=True)
result=dict(schema='qqt.good-model-selection/v1',date='2026-10-19',protocol_sha256=protocol,default_model='r01-g8',selection='declared five capability references from the same frozen protocol; default retained from existing balanced release, no replay outcome selection; Python Tactical and JS Hunter separate; latest completed reference is not overall best',excluded=[dict(id='r18',reason='live or incomplete training/evaluation; not read'),dict(id='r11',reason='historically invalid causal initialization; no capability conclusion')],models=rows)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
