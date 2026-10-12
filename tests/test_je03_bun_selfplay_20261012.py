import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun')
import unittest
from unittest.mock import patch
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from scripts import je03_bun_selfplay_20261012 as stage
from scripts import je03_bun_selfplay_ops_20261012 as ops

def row(safe,death=1,tactical=20):
    return dict(safe=safe,death=death,self=0,trade=0,tactical=dict(safe_win=tactical),tasks=dict(carry=dict(capture_episodes=14,steal_episodes=0)))

class Tests(unittest.TestCase):
    def test_stable_interval_not_terminal_peak(self):
        rows={512:row(8),768:row(16),1024:row(18),4096:row(30,death=40)}
        chosen,d=stage.choose(rows)
        self.assertEqual(d['window'],[768,1024])
        self.assertEqual(chosen,768)
    def test_missing_late_result_does_not_block(self):
        self.assertEqual(stage.choose({512:row(17)})[0],512)
    def test_occupied_gpu_cannot_be_taken(self):
        with patch.object(stage,'gpu_owners',return_value={0:[],1:[],2:[42],3:[],4:[],5:[],6:[],7:[]}):
            self.assertFalse(stage.available('0,1,2,3'))
            self.assertTrue(stage.available('4,5,6,7'))
    def test_sustained_hard_normal_beats_easy_only(self):
        rows={512:row(30),768:row(29),1536:row(16),2048:row(17)}
        for g,r in rows.items():
            r['hunter']={t:dict(safe_win=2 if g>=1536 else 0) for t in ('hard','normal')}
        selected,d=stage.choose(rows)
        self.assertEqual(d['window'],[1536,2048])
        self.assertEqual(selected,1536)
    def test_paired_plan_only_changes_opponent_source(self):
        parent = dict(recipes={'bun392_retention_teacher': {}}, architecture={})
        with patch.object(stage, 'read', return_value=parent), patch.object(stage, 'sha', return_value='parent-seal'):
            p=stage.plan()
        self.assertEqual(p['recipes'][stage.ARMS[0]],p['recipes'][stage.ARMS[1]])
        self.assertLessEqual(p['recipes'][stage.ARMS[0]]['lr'],1e-4)
        self.assertEqual(p['clip'],.1)
        self.assertFalse(p.get('start_gate',False))
    def test_balanced_learner_and_opponent_rows(self):
        learner,opponent=map(np.asarray,ops.balanced_indices(8))
        self.assertEqual(sum(learner<8),4)
        self.assertEqual(sum(opponent<8),4)
        np.testing.assert_array_equal(learner%8,opponent%8)
        self.assertTrue(np.all(learner!=opponent))
    def test_regression_monitor_reads_recovered_authority_namespace(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); authority=root/'authority_stage2_recovery_v2'
            arm=stage.ARMS[0]
            for g in [0,8,16,32]:
                p=authority/f'{arm}_g{g:06d}'
                p.mkdir(parents=True)
                (p/'COMPLETE_WITH_BUN_TASKS.json').write_text('{}')
            def metric(path):
                return dict(safe=20 if path.name.endswith('000000') else 1,
                            death=0,self=0)
            with patch.object(stage,'RUN',root), patch.object(stage,'metrics',side_effect=metric):
                self.assertEqual(stage.regression_points(arm),[8,16,32])
    def test_league_refresh_and_pickle_restore(self):
        start=jnp.array([1.,2.]); actor=jnp.array([5.,9.])
        league=(start,start)
        self.assertIs(ops.advance_league(league,actor,127),league)
        current,history=ops.advance_league(league,actor,128)
        np.testing.assert_array_equal(current,actor)
        np.testing.assert_array_equal(history,start)
        restored=pickle.loads(pickle.dumps(jax.device_get((current,history))))
        np.testing.assert_array_equal(restored[0],actor)
        np.testing.assert_array_equal(restored[1],start)
    def test_dynamic_actions_respond_to_new_snapshot(self):
        # Execute the actual snapshot selector, not a named scripted patrol.
        kinds=jnp.array([2,3,4,5,6],jnp.int8)
        candidates=jnp.zeros((7,5,2),jnp.int32)
        before=ops.select_bun_opponent_actions(kinds,candidates)
        changed=candidates.at[5].set(jnp.array([1,1])).at[6].set(jnp.array([2,2]))
        after=ops.select_bun_opponent_actions(kinds,changed)
        np.testing.assert_array_equal(np.asarray(before)[:3],np.asarray(after)[:3])
        self.assertTrue(np.all(np.asarray(after)[3:]!=np.asarray(before)[3:]))
if __name__=='__main__':unittest.main()
