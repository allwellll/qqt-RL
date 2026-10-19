"""Bind every imported numerical dependency to a run seal or immutable Git baseline.

The generic Bun reset is never executed: Web is the sole transition source and
je07's sealed native reset supplies only the conversion template. A dormant
causal-clean _fresh differs from baseline; all safety-used definitions/constants
must still be AST-identical. This exception never applies to reward/critic code.
"""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--input',required=True,type=Path);p.add_argument('--metrics',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--baseline',default='e5377f779ac2229a353994782732415f3153ca5f');a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest();seal=json.loads((a.run/'SOURCE_SEAL.json').read_text());proof=json.loads(a.metrics.read_text());rows=[]
for name,h in proof['reward_hashes'].items():
 data=(a.input/'authority'/name).read_bytes();assert sha(data)==h
 if name in seal['sources']:
  assert seal['sources'][name]==h;kind='run_source_seal'
 else:
  baseline=subprocess.check_output(['git','show',a.baseline+':'+name])
  if baseline==data:kind='immutable_production_baseline'
  else:
   assert name=='jax_bomb/bun_env.py'
   x,y=ast.parse(baseline),ast.parse(data);classes=(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)
   defs=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,classes)}
   A,B=defs(x),defs(y);assert {k for k in A.keys()|B.keys() if A.get(k)!=B.get(k)}=={'_fresh'}
   tops=lambda t:[ast.dump(n,include_attributes=False) for n in t.body if not isinstance(n,classes)]
   assert tops(x)==tops(y)
   safety=(a.input/'authority/jax_bomb/bun_safety.py').read_text();assert 'env._fresh' not in safety and 'env.init_batch' not in safety
   kind='immutable_baseline_safety_definitions_exact_dormant_reset_only'
  rows.append(dict(file=name,sha256=h,baseline_sha256=sha(baseline),scope=kind));continue
 rows.append(dict(file=name,sha256=h,scope=kind))
web_sources={}
for name in ['web/sim.js','web/bun_hunter_bot.js','scripts/je10_route_funnel.js','scripts/eval_web_hunter.js']:
 data=Path(name).read_bytes();assert data==subprocess.check_output(['git','show',a.baseline+':'+name]);web_sources[name]=sha(data)
episode=json.loads(Path(str(a.metrics).replace('.metrics.json','.json')).read_text())
assert episode['meta']['simulator_sha256']==web_sources['web/sim.js']
assert episode['meta']['opponent_sha256']==web_sources['web/bun_hunter_bot.js']
result=dict(schema='qqt.good-model-dependencies/v1',baseline_commit=a.baseline,passed=True,imported_count=len(rows),run_source_seal_sha256=sha((a.run/'SOURCE_SEAL.json').read_bytes()),scope='all imported numerical definitions sealed or immutable-baseline exact; dormant generic reset not called; sole state evolution is production Web; Web/scene/opponent sources are immutable production baseline, not part of training run seal',web_baseline_files=web_sources,files=rows)
a.output.write_text(json.dumps(result,indent=2)+'\n');print('Numerical dependency identities GREEN',len(rows))
