import unittest
from scripts import je03_bun_stage1_20261011 as subject
from scripts.bun_checkpoint_state import boundary


class BunStage1Test(unittest.TestCase):
    def test_real_bun_identity_and_attribution(self):
        p = subject.plan()
        self.assertEqual(p['environment']['rule'], 'bun')
        self.assertEqual(p['architecture']['embed'], 192)
        self.assertEqual(p['architecture']['obs'], [24,13,15])
        self.assertFalse(p['stage2_enabled'])
        self.assertFalse(p['opponents']['dynamic'])
        self.assertEqual(p['variable'], 'shared versus independent Critic only')
        self.assertGreaterEqual(p['generations'], 4096)
        self.assertEqual(p['ticks_per_arm'], p['generations']*4*p['envs_per_device']*p['steps'])

    def test_split_routes_actor_and_value_separately(self):
        def f(p,a,o,s):
            return p+'m',p+'b',p+'v',p+'l'
        self.assertEqual(subject.split_forward({'actor':'A','critic':'C'},'t',None,None,f), ('Am','Ab','Cv','Cl'))
        self.assertEqual(subject.split_forward('S','t',None,None,f), ('Sm','Sb','Sv','Sl'))

    def test_stop_cannot_interrupt_update(self):
        self.assertFalse(boundary(dict(phase='UPDATE_OR_COMPILE',generation=2177)))
        self.assertFalse(boundary(dict(phase='ROLLOUT_OR_COMPILE',generation=2176)))
        self.assertTrue(boundary(dict(phase='ROLLOUT_OR_COMPILE',generation=2177)))

    def test_checkpoint_hash_tampering_rejected(self):
        import tempfile
        from pathlib import Path
        from scripts.population_run_state import atomic_json,sha
        with tempfile.TemporaryDirectory() as d:
            cp=Path(d)
            (cp/'state.pkl').write_bytes(b'state')
            (cp/'policy.pt').write_bytes(b'policy')
            atomic_json(cp/'COMMITTED.json',dict(passed=True,plan_sha256='P',state_sha256=sha(cp/'state.pkl'),policy_sha256=sha(cp/'policy.pt')))
            subject.verify_checkpoint(cp,'P')
            (cp/'policy.pt').write_bytes(b'changed')
            with self.assertRaises(AssertionError):subject.verify_checkpoint(cp,'P')

    def test_safety_rejects_kill_then_death(self):
        from scripts.je03_bun_evaluator_20261011 import summarize_hunter
        row=dict(counts=dict(surviving_causal_kill=1,surviving_physical_kill=0,deaths=1,
                 own_bomb_defeat=1,mutual_death=0,killed_by_bot=0,bombs=3),
                 residual=dict(hazards_remaining=False,alive=[True,True]))
        metrics=summarize_hunter([row])
        self.assertEqual(metrics['safe_win'],0)
        self.assertEqual(metrics['kill_then_death'],1)
        self.assertEqual(metrics['self_episodes'],1)

    def test_value_gradient_does_not_touch_actor(self):
        import os
        os.environ['JAX_PLATFORMS']='cpu'
        import jax
        import jax.numpy as jnp
        params=dict(actor=jnp.array(2.),critic=jnp.array(3.))
        def forward(p,*_):return p,p,2*p,4*p
        gradient=jax.grad(lambda p:subject.split_forward(p,None,None,None,forward)[2])(params)
        self.assertEqual(float(gradient['actor']),0.)
        self.assertEqual(float(gradient['critic']),2.)


if __name__=='__main__':unittest.main()
