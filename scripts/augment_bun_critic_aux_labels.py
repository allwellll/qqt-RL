#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();d=np.load(a.input,allow_pickle=False);x={k:d[k] for k in d.files};n,ac=x['legal'].shape;rows=np.arange(n)[:,None];acts=np.arange(ac)[None,:];actors=np.broadcast_to(x['actor_id'][:,None],(n,ac));mut=x['first_mutual_death'];kill=(x['first_credited_kill'][rows,acts,actors]|x['first_causal_kill'][rows,acts,actors])&~mut;own=x['first_own_bomb_defeat'][rows,acts,actors];surv=x['survivable'];abilities=np.arange(ac)%3
 x.update({'safe_attack_available':np.any(kill[:,abilities==1]&surv[:,abilities==1],axis=1),'self_survive_4s':surv.copy(),'enemy_survive_4s':(~kill).copy(),'safe_escape_exists':surv.copy(),'enemy_escape_count':np.where(kill,0,1).astype(np.int16),'minimum_escape_time':np.where(surv,1.0,40.0).astype(np.float32),'forced_kill_state':kill.copy(),'bomb_creates_forced_kill':kill&(abilities[None,:]==1),'bomb_creates_trade':mut&(abilities[None,:]==1),'own_bomb_death_risk':own|x['avoidable'],'causal_kill_onset':np.where(x['first_causal_kill'][rows,acts,actors],0,-1).astype(np.int16),'forced_kill_onset':np.where(kill&(abilities[None,:]==1),1,-1).astype(np.int16)})
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**x);m={'schema':'bun_critic_aux_upgrade_v1','input':a.input,'input_sha256':digest(a.input),'output':a.output,'output_sha256':digest(out),'states':n,'labels_added':sorted(set(x)-set(d.files)),'source_labels_are_legacy_inferences':True};Path(a.manifest).write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m))
if __name__=='__main__':main()
