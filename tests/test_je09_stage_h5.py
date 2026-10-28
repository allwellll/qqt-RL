import os
os.environ.setdefault('JAXBOMB_RULE','bun')
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from scripts import je09_stage_h5_dagger as d


class DaggerTests(unittest.TestCase):
    def test_mode_edges_and_item_negative(self):
        # Native trap or death AND attributed window; glue/banana are absent.
        free=jnp.array([True,False,False,True,True])
        trap=jnp.array([0,60,0,0,0])
        owned=jnp.array([True,True,True,True,False])
        self.assertEqual(d.mode(free,trap,owned).tolist(),[False,True,True,False,False])

    def test_gate_phase_exclusions(self):
        active=d.eligible(jnp.array([8,8,8,8,8,0,8,8]),
            jnp.array([1,1,1,1,1,1,1,1],bool),
            jnp.array([0,0,0,0,0,0,1,0],bool),
            jnp.array([0,0,0,0,0,0,0,1]),
            jnp.array([-1,-1,1,1,-1,-1,-1,-1]),
            jnp.array([1,0,0,0,1,1,1,1],bool),
            jnp.array([0,0,1,1,0,0,0,0],bool),
            jnp.array([0,0,0,120,0,0,0,0]))
        self.assertEqual(active.tolist(),[True,False,True,False,True,False,False,False])

    def test_equal_actions_and_zero_excluded_gradient(self):
        from scripts.je09_return_expert import supervised_loss
        target=jnp.array([[.5,.5,0,0,0],[.5,.5,0,0,0]])
        gate=jnp.array([True,False]); legal=jnp.ones((2,5),bool)
        grad=jax.grad(lambda x:supervised_loss(x,legal,target,gate,.05,1.)[0])(jnp.zeros((2,5)))
        np.testing.assert_array_equal(grad[1],0)
        self.assertEqual(float(grad[0,0]),float(grad[0,1]))

    def test_aggregation_keeps_old_and_queries_updated_round(self):
        b=d.buffer(8,2,(1,2,2),3)
        def batch(x):
            return (jnp.ones((2,2,1,2,2))*x,jnp.ones((2,2,3)),None,None,None,None,
                    (jnp.ones((2,2,5),bool),None),
                    jnp.broadcast_to(jnp.array([.5,.5,0,0,0,1,2,1,1,0,0,0]),(2,2,12)))
        b,_,_=d.aggregate(b,batch(1),jax.random.PRNGKey(1),1,offset=0)
        b,replay,stats=d.aggregate(b,batch(2),jax.random.PRNGKey(2),2,offset=0)
        self.assertEqual(int((b['round']==1).sum()),2)
        self.assertEqual(int((b['round']==2).sum()),2)
        self.assertTrue(bool((replay['round']==1).any()))
        self.assertTrue(bool((replay['round']==2).any()))
        self.assertGreater(float(stats['history_fraction']),0)

    def test_actual_native_update_and_labels(self):
        from jax_bomb import je07_bun_env as env
        env.prepare()
        from scripts import je09_bun_scenes as scenes
        st=scenes.init_batch(jax.random.PRNGKey(5),2,jnp.eye(10)[8])
        n=2;rows=jnp.arange(n);seat=rows%2;enemy=1-seat
        st=st._replace(scene=jnp.full(n,8,jnp.int8),
            trapped=st.trapped.at[rows,enemy].set(60),
            trapped_causal=st.trapped_causal.at[rows,enemy,seat].set(True))
        info={k:jnp.zeros((n,2),bool) for k in ['surviving_kill','steal','pickup_loose','native_transfer']}
        info['pickup_origin']=jnp.full((n,2),-1)
        aux=dict(terminal=jnp.zeros(n,bool),drop=dict(credit=jnp.full((n,2),-1),nonce=jnp.zeros((n,2),jnp.int32)),
                 routing=dict(target=jnp.full((n,2,2),-1)))
        ledger=d.update(st,st,info,aux,d.state(n));aux['dagger']=ledger
        self.assertTrue(bool(ledger['window'][rows,seat].all()))
        targets=d.labels(st,aux,jnp.ones((n,5),bool),seat,jnp.zeros((n,13,15)))
        self.assertEqual(targets.shape,(n,12))
        free=st._replace(trapped=jnp.zeros_like(st.trapped))
        ended=d.update(st,free,info,aux,ledger)
        self.assertFalse(bool(ended['window'].any()))

    def test_relay_rejects_complete_only(self):
        import tempfile
        from pathlib import Path
        from scripts.je09_stage_h5_relay import accepted_successor
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'CAMPAIGN_COMPLETE.json').write_text('{"completed":true}')
            self.assertFalse(accepted_successor(base))


if __name__=='__main__':unittest.main()
