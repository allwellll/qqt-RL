"""Versioned passive fullgame ABI extension; frozen evaluator stays untouched."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')


def main():
    from scripts import expert_reward_alignment_fullmatch as F
    import jax.numpy as j
    import numpy as np
    extra=('own_alive_before','own_trapped_before','game_horizon_before',
           'own_speed_status_before','own_speed_status_after','own_held_item_kind',
           'own_health_before','own_health_after')
    F.TELEMETRY_NAMES=(*F.TELEMETRY_NAMES,*extra)
    original=F._passive_snapshot
    def snapshot(st,after,info,actions,policy,mm,bm,legacy,minimal,active,done):
        old=original(st,after,info,actions,policy,mm,bm,legacy,minimal,active,done)
        values=j.stack([st.alive[:,0],st.trapped[:,0],st.t>=2400,
            st.move_status[:,0],after.move_status[:,0],st.held_item[:,0],
            st.hp[:,0],after.hp[:,0]],-1).astype(old.dtype)
        return j.concatenate([old,values],-1)
    F._passive_snapshot=snapshot
    summarize=F.summarize_episodes
    def episodes(model,mode,indices,arrays,states,aux,names):
        result=summarize(model,mode,indices,arrays,states,aux,names)
        telemetry=np.asarray(arrays[-1]);valid=np.asarray(arrays[-3]).astype(bool)
        fields={name:i for i,name in enumerate(F.TELEMETRY_NAMES)}
        for col,row in enumerate(result):
            selected=telemetry[valid[:,col],col];at=lambda name:selected[:,fields[name]]
            terminal_ticks=np.flatnonzero(np.isin(at('winner'),[-1,0,1])|(at('tick')>=2399))
            first_terminal=int(terminal_ticks[0]) if len(terminal_ticks) else len(selected)-1
            before_terminal=np.arange(len(selected))<=first_terminal
            alive=at('own_alive_before')>0;actionable=alive&(at('own_trapped_before')==0)
            row['passive_abi_v2']=dict(own_alive_ticks=int(alive.sum()),own_actionable_ticks=int(actionable.sum()),
                preterminal_alive_ticks=int((alive&before_terminal).sum()),
                settlement_tail_alive_ticks=int((alive&~before_terminal).sum()),
                terminal_boundary='first observed winner or 2400tick horizon; includes event tick in preterminal',
                enemy_distance_manhattan_values=at('enemy_distance_manhattan').tolist(),
                complete_causal_item_net_benefit=None,
                item_missing_reason='Passive observed item chains require matched causal branches to estimate net benefit',
                source='actual full natural JAX trace; observer fields never used by policy/reward')
        return result
    F.summarize_episodes=episodes
    from scripts.campaign_checkpoint_eval import main as run
    run()


if __name__=='__main__':main()
