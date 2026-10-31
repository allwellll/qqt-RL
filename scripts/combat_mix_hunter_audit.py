"""Fixed-observation JAX-vs-production-JS Hunter decision audit, CPU only."""
from pathlib import Path
import json
import os
import subprocess
import time

os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu', JAXBOMB_RULE='bun',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import jax
import jax.numpy as j
import numpy as np
from jax_bomb import bun_env as E, bun_jax_bots as B
from scripts.population_run_state import atomic_json, sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/combat_mix_v1'


def hunter_state(s):
    flat=lambda x:np.asarray(x).reshape(-1).tolist()
    players=[]
    for p in range(2):
        pos=np.asarray(s.pos[p]);cell=np.floor(pos).astype(int)
        live=int(np.sum((np.asarray(s.core.owner)==p)&(np.asarray(s.fuse)>0)))
        players.append(dict(y=float(pos[0]),x=float(pos[1]),row=int(cell[0]),col=int(cell[1]),
            cell=int(cell[0]*15+cell[1]),alive=bool(s.alive[p]),bombsCap=float(s.core.bombs_cap[p]),
            liveBombs=live,bombsLeft=max(0,int(s.core.bombs_cap[p])-live),blast=float(s.core.blast_cap[p]),
            speed=float(s.core.spd_g[p]),carrying=int(s.bun_carried[p]),invuln=int(s.core.invuln[p]),
            respawn=int(s.bun_respawn[p]),team=p,trapped=0))
    bombs=[]
    for row,col in np.argwhere(np.asarray(s.fuse)>0):
        bombs.append(dict(cell=int(row*15+col),row=int(row),col=int(col),fuse=int(s.fuse[row,col]),
                          blast=int(s.core.bomb_blast[row,col]),owner=int(s.core.owner[row,col])))
    return dict(schema='qqt.bun.hunter_state/v1',height=13,width=15,tick=int(s.t),
        wall=flat(s.wall),brick=flat(s.brick),brickLinger=flat(s.core.brick_linger),
        blastLinger=flat(s.core.blast_linger),crate=flat(s.core.crate),fieldItem=flat(s.field_item),
        crateType=flat(s.tactical_crate),heldItem=flat(s.held_item),bombs=bombs,players=players,teams=[0,1],
        bunBases=np.asarray(s.bun_spawn_pos).tolist(),bunStored=np.asarray(s.bun_stored).tolist(),
        bunLoose=flat(s.bun_loose))


def main():
    E.prepare();E.configure_training('danger_arena=1',1,reward_profile='legacy')
    states=E.init_batch(jax.random.PRNGKey(2026103150),96)
    cells=np.asarray(states.pos).copy()
    # Diagnostic-only close-contact states; never fed to training or carry Bank.
    for i in range(32):
        cells[i]=[[5.5,2.5],[5.5,4.5]]
    states=states._replace(core=states.core._replace(pos=j.asarray(cells),brick=j.zeros_like(states.brick)))
    fixtures=[]
    for i in range(96):
        st=jax.tree.map(lambda x:x[i],states)
        fixtures.append(dict(id=i,seed=2026103150+i,hunter_state=hunter_state(st)))
    atomic_json(BASE/'HUNTER_FIXED_STATES.json',dict(states=fixtures,scope='same observable states; no synthetic carrying'))
    mm,bm=jax.vmap(E.legal_mask)(states)
    actions={}
    for tier in ('bomber_easy','hunter','hunter_hard'):
        act=jax.jit(B.rule_bot_actions)(states,j.ones(96,j.int32),mm[:,1],bm[:,1],jax.random.PRNGKey(2026103150),j.full(96,B.TIER_NAMES.index(tier),j.int32))
        actions[tier]=np.asarray(act).tolist()
    atomic_json(BASE/'HUNTER_JAX_DECISIONS.json',dict(actions=actions,states_sha256=sha(BASE/'HUNTER_FIXED_STATES.json')))
    subprocess.run(['node','scripts/combat_mix_hunter_decisions.js',str((BASE/'HUNTER_FIXED_STATES.json').relative_to(ROOT)),str((BASE/'HUNTER_JS_DECISIONS.json').relative_to(ROOT))],cwd=ROOT,check=True)
    js=json.loads((BASE/'HUNTER_JS_DECISIONS.json').read_text());rows={}
    for difficulty,tier in [('easy','bomber_easy'),('normal','hunter'),('hard','hunter_hard')]:
        matches=[r for r in js['rows'] if r['difficulty']==difficulty]
        lhs=np.asarray(actions[tier]);rhs=np.asarray([r['action'] for r in matches])
        rows[difficulty]=dict(jax_tier=tier,states=96,move_matches=int((lhs[:,0]==rhs[:,0]).sum()),
            bomb_matches=int((lhs[:,1]==rhs[:,1]).sum()),joint_matches=int((lhs==rhs).all(-1).sum()),
            jax_bomb_requests=int(lhs[:,1].sum()),js_bomb_requests=int(rhs[:,1].sum()),
            invalid_jax_actions=int(sum(not bool(mm[i,1,lhs[i,0]]) or not bool(bm[i,1,lhs[i,1]]) for i in range(96))))
    status='PASS_APPROXIMATE_TRAINING_OPPONENT' if all(x['invalid_jax_actions']==0 for x in rows.values()) and rows['hard']['jax_bomb_requests']>0 else 'BLOCKED'
    atomic_json(BASE/'HUNTER_PARITY.json',dict(status=status,decision_comparison=rows,
        source_sha256={n:sha(ROOT/n) for n in ['jax_bomb/bun_jax_bots.py','web/bun_hunter_bot.js','scripts/combat_mix_hunter_audit.py','scripts/combat_mix_hunter_decisions.js']},
        limitations=['JAX approximate Hunter is not production JS Hunter; capped BFS vs time-expanded escape, distinct PRNGs.','Easy delay/mistake/hesitation and hard robustness/pressure parameters differ.','96 fixed states include32 declared synthetic close-contact audit states; zero artificial carrying; not training data.','JAX parent 64-game engine/opponent outcomes and production JS parent 64-game outcomes are reported separately, not a one-factor causal engine comparison.'],
        jax_rollout='runs/combat_mix_v1/JAX_HUNTER_PARENT.json',js_parent='runs/combat_mix_v1/js_web_compare_v1/common_parent',unix=time.time()))


if __name__=='__main__':main()
