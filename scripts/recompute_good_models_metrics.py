"""CPU same-checkpoint Critic plus sealed JE10 ledger on production Web branches.

Offline approximation of training indicators, never historical rollout telemetry.
The authority directory is an explicitly assembled, hash-checked private input.
"""
import argparse,base64,copy,hashlib,inspect,json,os,pickle,subprocess,sys,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
import diagnostic_reward as R
import numpy as np
import jax
import jax.numpy as jnp
from jax_bomb import jax_net
from scripts import je10_ops as ops
from diagnostic_advantage import advance_linger,calculate
assert jax.default_backend()=='cpu'
ops.install()
R.reward=ops
BASE=R.COMPONENTS
EXTRA=('control_entry_reward','control_residual_reward','control_clawback','route_shaped_reward')
COMPONENTS=BASE+EXTRA
ROUTE=ops.pbrs.METRICS
# Keep the production diagnostic safety adapter in one place. Its output terms
# are replaced with the complete additive ledger terms, not metric counters.
source=inspect.getsource(R.evaluate_chunk.__wrapped__)
source=source.replace("(fresh | refreshed) & danger.exposed & ~danger.doomed", "(fresh | refreshed) & danger.exposed & ~danger.doomed")
source=source.replace("info = dict(info, danger_safe_resolution=resolved,", "info = dict(info, danger_exposure=jnp.any(danger.exposed, axis=-1), danger_safe_resolution=resolved,")
source=source.replace("(total, terms, info['own_bomb_defeat'])", "(total, jnp.concatenate([terms, jnp.stack([metrics[k] for k in extra_names], -1)], -1), jnp.stack([metrics[k] for k in route_names], -1))")
ns=dict(R.__dict__,extra_names=EXTRA,route_names=ROUTE,COMPONENTS=BASE)
exec(compile(source,'<sealed-ledger-web-adapter>','exec'),ns)
evaluate=ns['evaluate_chunk']

@jax.jit
def branches(carry,before,afters,infos):
 def step(after,info):
  new,(total,parts,route)=evaluate(carry,jax.tree.map(lambda x:x[None],(before,after,info)))
  # Same deferred terminal state update as the training collect loop.
  aux,pending,avoidable,chain,moved=new
  aux['terminal']=aux['terminal']|info['h_route_terminal']
  return (aux,pending,avoidable,chain,moved),total[0],parts[0],route[0]
 return jax.vmap(step)(afters,infos)

def convert(frame,linger,scene,causes):
 s=R.convert(frame,linger,scene)
 cause=np.array([causes],bool)
 return s._replace(trapped_causal=cause,trapped_source=cause)

def run(file,critic_path,verify=False,limit=None):
 doc=json.loads(file.read_text());cm=json.loads(critic_path.with_name('meta.json').read_text());critic=pickle.loads(critic_path.read_bytes())
 assert doc['meta']['checkpoint_sha256']==cm['checkpoint_sha256']
 assert hashlib.sha256(Path(jax_net.__file__).read_bytes()).hexdigest()==cm['network_sha256']
 forward=jax.jit(lambda o,g:jax_net.net_forward(critic,'transformer',o,g)[2])
 proc=subprocess.Popen(['node','scripts/stream_review_metrics.js',str(file)],stdout=subprocess.PIPE,text=True)
 initial=json.loads(proc.stdout.readline())['initial'];before_raw=initial['state'];prior_legal=initial['legal']
 # Full native episode starts have no previous hidden events. Fixtures with
 # pre-existing control provenance are rejected rather than guessed.
 assert not any(before_raw['trapped']) or (doc['meta'].get('initial_control_provenance')=='reset_unowned' and doc['meta'].get('native_fixture',{}).get('task')=='recovery_threat'), 'unrecorded initial control provenance'
 # Declared reset trap is deliberately unowned: no invented past control event,
 # reward balance or drop attribution. Subsequent native events use real sources.
 aux=ops.reward_state(1);zeros=jnp.zeros((1,2),bool)
 carry=(aux,jnp.zeros((1,2),jnp.int16),zeros,zeros,zeros)
 linger=np.zeros((2,2,13,15),np.int8);causes=[[False,False],[False,False]]
 scene=7 if doc['meta'].get('task')=='recovery_threat' else {'native':8,'open':0,'carry_home':2}[doc['meta']['scene']]
 result=[];digest=hashlib.sha256();last=None;start=time.monotonic()
 for line in proc.stdout:
  packet=json.loads(line);cs=packet['candidates'];tick=packet['tick'];n=len(cs);before=convert(before_raw,linger,scene,causes)
  afters=[];infos=[];causal=[];lingers=[]
  for c in cs:
   e=c['entry'];e['prior_legal']=prior_legal;cause=copy.deepcopy(causes);info=R.observed(before_raw,e,cause)
   info['h_route_terminal']=np.array([e['state']['done']],bool)
   info['trigger_damage_source']=np.array([e['info']['causalDamageSource']],bool)
   # Native contact pop is derived from the actual bun-interaction snapshot.
   pop=[]
   for p in range(2):
    b=e['bun_before'];q=1-p
    contact=before_raw['trapped'][p]>0 and e['info']['died'][p] and b['alive'][q] and not b['trapped'][q] and all(abs(b['pos'][p*2+k]-b['pos'][q*2+k])<41/40 for k in range(2))
    pop.append(q if contact else -1)
   info['native_trap_popper']=np.array([pop],np.int32)
   li=advance_linger(linger,e)
   # Causal ownership persists during native trap lifetime, independent of
   # its countdown. R.observed clears it on a real death edge.
   afters.append(convert(e['state'],li,scene,cause));infos.append(info);causal.append(cause);lingers.append(li)
  idx=list(range(n))+[n-1]*(15-n)
  after=jax.tree.map(lambda *a:np.stack(a),*[afters[i] for i in idx]);info=jax.tree.map(lambda *a:np.stack(a),*[infos[i] for i in idx])
  new,total,parts,routes=branches(carry,before,after,info)
  total,parts,routes=map(np.asarray,(total,parts,routes))
  obs=np.stack([np.frombuffer(base64.b64decode(cs[i]['obs']),np.float32).reshape(24,13,15) for i in idx]);gv=np.stack([np.frombuffer(base64.b64decode(cs[i]['global']),np.float32) for i in idx])
  values=np.asarray(forward(obs,gv))
  assert np.isfinite(parts).all() and np.isfinite(values).all()
  np.testing.assert_allclose(parts.sum(-1),total,atol=2e-5,err_msg='sealed complete ledger component sum')
  k=next(i for i,c in enumerate(cs) if c['action']==packet['taken'])
  rows=[]
  for i,c in enumerate(cs):
   shaped=float(total[i,0,0]);raw=float(parts[i,0,0,:len(BASE)].sum())
   rows.append(dict(action=c['action'],probability=c['probability'],raw_reward=raw,shaped_reward=shaped,value=float(values[i]),terminal=c['terminal'],next_state_sha256=hashlib.sha256(json.dumps(c['entry']['state'],separators=(',',':')).encode()).hexdigest()))
  a={}
  for role in ('raw','shaped'):
   candidates=[dict(c,reward=c[role+'_reward']) for c in rows]
   a[role+'_q1_taken'],a[role+'_q1_mean'],a[role+'_a1']=calculate(candidates,packet['taken'],cm['gamma'])
  a['delta_a']=a['shaped_a1']-a['raw_a1']
  for key in ('raw_a1','shaped_a1','delta_a'):a[key+'_increment']=None if last is None else a[key]-last[key]
  last=a
  digest.update(json.dumps(dict(tick=tick,candidates=rows),sort_keys=True,separators=(',',':')).encode())
  result.append(dict(tick=tick,components=[dict(zip(COMPONENTS,map(float,parts[k,0,p]))) for p in range(2)],total=list(map(float,total[k,0])),route=[dict(zip(ROUTE,map(float,routes[k,0,p]))) for p in range(2)],a1=a,decision=packet['decision'],before_hash=packet['before_hash'],after_hash=packet['after_hash']))
  carry=jax.tree.map(lambda x:x[k],new);before_raw=cs[k]['entry']['state'];prior_legal=cs[k]['entry']['legal'];causes=causal[k];linger=lingers[k]
  if tick%100==0 or tick==1:print(file.stem,tick,'seconds',round(time.monotonic()-start,1),flush=True)
  if limit and tick>=limit:proc.terminate();proc.wait();break
 assert limit or (proc.wait()==0 and len(result)==len(doc['actions']))
 used={}
 for name,m in list(sys.modules.items()):
  if name.startswith(('scripts.','jax_bomb.')):
   p=getattr(m,'__file__',None)
   if p and Path(p).is_file():
    try:key=str(Path(p).resolve().relative_to(R.AUTHORITY.resolve()))
    except ValueError:raise AssertionError('unsealed external training dependency: '+name)
    used[key]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
 proof=dict(source_run=doc['meta']['comparison_model_id'],initial_control_provenance=doc['meta'].get('initial_control_provenance','natural_reset'),schema='qqt.sealed-ledger-a1/v1',status='offline_approximate_one_step',critic=cm,reward_hashes=used,adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),component_names=list(COMPONENTS),counterfactual_sha256=digest.hexdigest(),measured_count=len(result),ticks=result)
 out=file.with_suffix('.metrics.json')
 if verify:assert json.loads(out.read_text())==proof,'independent all-tick ledger/A1 mismatch'
 else:out.write_text(json.dumps(proof,separators=(',',':')))
 print('VERIFIED' if verify else 'MEASURED',file.stem,len(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('files',nargs='+',type=Path);p.add_argument('--critic',required=True,type=Path);p.add_argument('--verify',action='store_true');p.add_argument('--limit',type=int);a=p.parse_args()
 for f in a.files:run(f,a.critic,a.verify,a.limit)
