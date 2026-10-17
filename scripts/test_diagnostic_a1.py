"""Formula and extraction invariants independent of heavyweight training imports."""
import ast
import json
import unittest
from pathlib import Path

source=ast.parse(Path('scripts/diagnostic_advantage.py').read_text())
scope={}
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='calculate'],type_ignores=[]),'calculate','exec'),scope)
calculate=scope['calculate']


class Formula(unittest.TestCase):
    def test_weighted_legal_renormalization(self):
        c=[dict(action=[0,0],probability=.2,reward=1,value=4,terminal=False),
           dict(action=[1,0],probability=.3,reward=-1,value=2,terminal=False),
           dict(action=[2,0],probability=.5,reward=999,value=999,terminal=False,legal=False)]
        q,mean,a=calculate(c,[0,0],.5)
        self.assertEqual(q,3);self.assertAlmostEqual(mean,1.2);self.assertAlmostEqual(a,1.8)
        c[0]['terminal']=True
        self.assertEqual(calculate(c,[0,0],.5)[0],1)
        with self.assertRaises(StopIteration):calculate(c,[2,0],.5)
        for row in c:row['probability']=0
        with self.assertRaises(AssertionError):calculate(c,[0,0],.5)

    def test_real_provenance(self):
        p=Path('web/review-evidence/h8-g32-critic.json')
        if not p.exists():self.skipTest('export not finalized')
        meta=json.loads(p.read_text());self.assertEqual(meta['parity']['max_error'],0)
        self.assertEqual(meta['gamma'],.995);self.assertTrue(meta['replicas_exact'])
        catalog=json.loads(Path('web/review-catalog.json').read_text())
        self.assertEqual(len([s for s in catalog['scenes'] if s['id'] != 'route_drop_return']),3)
        for row in catalog['replays']:
            if row.get('diagnostic'):
                continue
            d=json.loads((Path('web/review-replays')/row['file']).read_text())
            self.assertEqual(d['meta']['advantage']['critic_sha256'],meta['critic_sha256'])
            self.assertEqual(d['meta']['advantage']['measured_count'],len(d['actions']))
            self.assertEqual(d['advantages']['delta'][0],None)


if __name__=='__main__':unittest.main()
