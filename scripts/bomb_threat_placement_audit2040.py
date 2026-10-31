"""Passive real saved on-policy placement coverage; no rewards or GPU changes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
import json,pickle,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
B=Path('runs/bomb_threat_v1');OUT=B/'a3s/extension1280/placement_audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True);guard=lock(OUT/'worker.lock')
    if (OUT/'COMPLETE.json').exists():return
    cfg=json.loads((OUT/'MANIFEST.json').read_text());cp=Path(cfg['checkpoint']);cm=json.loads((cp/'COMMITTED.json').read_text())
    assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
    import jax,jax.numpy as j,numpy as np
    from scripts import ppo_scale_longrun_train as P,bomb_threat_collector_diagnostics_v1 as M,bomb_threat_reward as R
    from scripts.bomb_threat_iter2_recipe import settlement
    from scripts.bomb_threat_runtime import install
    from scripts.terminal_checkpoint_restore import digest
    P.T.configure();P.C.install_provenance();install();R.settle=settlement(5.,.6)
    payload=pickle.load((cp/'state.pkl').open('rb'));n=cfg['environments'];ids=np.linspace(0,payload['states'].t.shape[1]-1,n,dtype=int)
    select=lambda tree:jax.tree.map(lambda x:j.asarray(x[0,ids]),tree)
    st,ax,keys=select(payload['states']),select(payload['aux']),select(payload['keys'])
    params=jax.tree.map(j.asarray,payload['params']);ring=jax.tree.map(j.asarray,payload['combat_ring'][(payload['update']-1)%4]);fixed=jax.tree.map(j.asarray,payload['combat_fixed_actor'])
    kinds=np.asarray(ax['combat_mix']['kind']);groups={name:np.flatnonzero(kinds==i) for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]};groups['bot']=np.flatnonzero(kinds<3)
    def body(c,_):
        s,a,k,ended=c;nk,ke,actions,_=M.choose(params,ring,fixed,s,a,k,groups)
        ns,na,r,done,info,parts,sm=M.transition(s,a,actions,ke,0.)
        live=~ended;placed=info['bomb_placed'][:,0]&live
        cover=info['tactical_bomb_newly_threatens_enemy'][:,0];block=info['tactical_bomb_enemy_safe_moves_after'][:,0]<info['tactical_bomb_enemy_safe_moves_before'][:,0]
        m=dict(placed=placed,reachable_or_block=placed&(cover|block),newly_covers=placed&cover,reduces_safe_moves=placed&block,
            no_immediate_threat_evidence=placed&~(cover|block),rewarded_first_safe_threat=sm['threat_events']*live,
            live=live,ended=done&live,deaths=info['death'][:,0]&live)
        return (M.E.select_tree(ended,s,ns),M.E.select_tree(ended,a,na),M.E.select_tree(ended,k,nk),ended|done),m
    began=time.time();atomic_json(OUT/'PROGRESS.json',dict(status='COMPILING_PASSIVE_SCAN',unix=time.time()))
    scan=jax.jit(lambda carry:jax.lax.scan(body,carry,None,length=cfg['ticks']))
    _,trace=scan((st,ax,keys,j.zeros(n,bool)));jax.block_until_ready(trace)
    rows={k:np.asarray(v) for k,v in trace.items()};np.savez_compressed(OUT/'TRACE.npz',**rows)
    counts={k:int(v.sum()) for k,v in rows.items()};den=counts['placed']
    atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',checkpoint=str(cp),policy_sha256=cm['policy_sha256'],state_sha256=cm['state_sha256'],
        manifest_sha256=sha(OUT/'MANIFEST.json'),trace_sha256=sha(OUT/'TRACE.npz'),counts=counts,
        immediate_threat_fraction=counts['reachable_or_block']/den if den else None,
        no_immediate_threat_fraction=counts['no_immediate_threat_evidence']/den if den else None,
        elapsed_seconds=time.time()-began,initial_selected_state_digest=digest((st,ax,keys)),
        source='32 deterministically stratified rows of real student saved full env/aux/RNG; mixed current/ring/fixed/Hunter; categorical; no reset or artificial carrier',
        limitations=['short128tick continuation subset; not complete-game or all-training coverage','no immediate threat evidence does not prove tactically useless; future blast/control not replayed to full outcome','newly_covers and reduces_safe_moves overlap; union counted once','rewarded first safe threat additionally requires safety, dedup and live-credit eligibility'],GPU_used=False,training_steps=0,unix=time.time()))

if __name__=='__main__':main()
