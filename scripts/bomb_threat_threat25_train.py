"""Authorized slot01 threat-only reduction; preserve the established worker."""

def threat_settlement(multiplier,cap):
    import jax.numpy as j
    def settle(e,s):
        rows=j.arange(len(s['gross']));cell=j.clip(e['enemy_cell'],0,194);old=j.clip(e['old_cell'],0,194)
        valid=e['active']&~e['bad']&~e['terminal']
        threat=e['placed']&e['safe']&e['threat']&~s['enemy_seen'][rows,cell]&valid
        chain=e['placed']&e['safe']&e['chain']&(e['old_id']>s['old_seen'][rows,old])&valid
        bricks=e['owned_removed']&~s['brick_seen'];topology=e['improved']&bricks.any((1,2))&valid
        amount=(.02*2.5)*threat+(.01*multiplier)*chain+(.04*multiplier)*topology
        pending=j.minimum(s['pending']+amount,j.maximum(0.,cap-s['gross']))
        pending=j.where(valid,pending,0.);pay=j.where(e['settled']&valid,pending,0.)
        n=dict(s,pending=j.where(e['settled'],0.,pending),gross=s['gross']+pay,
            enemy_seen=s['enemy_seen'].at[rows,cell].set(s['enemy_seen'][rows,cell]|threat),
            old_seen=s['old_seen'].at[rows,old].set(j.where(chain,e['old_id'],s['old_seen'][rows,old])),
            brick_seen=s['brick_seen']|bricks,brick_pending=j.where(valid,s['brick_pending']+topology.astype(j.int32),0))
        paid_bricks=j.where(e['settled']&valid,n['brick_pending'],0);n['brick_pending']=j.where(e['settled'],0,n['brick_pending'])
        return n,pay,dict(candidate=threat|chain|topology,settled_count=(pay>0).astype(j.int32),pay=pay,
            threat_events=threat.astype(j.int32),chain_events=chain.astype(j.int32),brick_events=topology.astype(j.int32),brick_paid_events=paid_bricks,threat_pending=pending)
    return settle


def main():
    import sys,json,os
    from pathlib import Path
    p=json.loads(Path(sys.argv[1]).read_text());assert p['arms'][sys.argv[2]]['threat_multiplier']==2.5
    os.environ.update(CUDA_VISIBLE_DEVICES=os.environ['JAXBOMB_DEVICE_OVERRIDE'],JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='cuda')
    from scripts import bomb_threat_iter2_recipe as recipe,bomb_threat_iter2_diagnostics_train as worker
    recipe.settlement=threat_settlement
    worker.main()

if __name__=='__main__':main()
