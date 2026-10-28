"""Declare H2 independently before any technical or formal training."""
import json
import subprocess
from pathlib import Path

BASE = Path('runs/je09_bun_20261016/stage_h2_item_coverage_pfsp8_v15')
assert not BASE.exists(), 'never overwrite a namespace'
BASE.mkdir(parents=True)
d = json.loads(Path('reports/je09_bun_20261015/stage_h_declaration_20261016.json').read_text())
d.update(experiment_id='JE09-STAGE-H2-ITEM-COVERAGE-PFSP8-20261016-v15', namespace=str(BASE),
         parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), seed=2801000000)
d['seeds'] = {k: v+100000000 for k, v in d['seeds'].items()}
d['budget'].update(formal_generations=32, formal_fresh_ticks=67108864,
    candidates=[0,4,8,12,16,24,32], atomic_checkpoint_generations=[0,4,8,12,16,24,32])
d['change'] = 'Only expand native item opportunity coverage in real-drop scene9 at initialization; no post-pickup injection, teleport, action override, reward or KL changes.'
d['reason'] = 'H hit signals began at g5 but short-score stopper cut the run; sparse real-custody item opportunities and retention risk need a fixed longer observation budget.'
d['item_state_coverage'] = dict(fraction=.75, scene=9, initial_learner_inventory=[2,1], counts=[1,1],
    field='one enemy-owned armed banana on a legal equal-shortest homeward neighbor of initial carrier cell; exclude learner start and goal',
    time='episode initialization only; actual native attributed death/drop and learner first pickup required for reward',
    unaffected='all non-scene9 states byte-exact; positions, carry, topology, combat birth .60 and pool unchanged')
d['stop'] = 'Fixed32 generations/67108864 fresh ticks; no two-point/no-improvement/capability-score stop. Stop only technical NaN/OOM/corruption or declared gross native safety failures.'
d['hard_safety'] = dict(per_generation_native_death_fraction_fresh=.02,
    own_bomb_defeat_fraction_fresh=.005, mutual_death_fraction_fresh=.005,
    ordinary_retention_degradation='observe only; does not truncate fixed budget')
d['evaluation_policy'] = 'Existing assist-off protocols, asynchronous g0/4/8/12/16/24/32; metrics never authorize or stop GPU. No new web protocol.'
d['parent_preservation'] = 'G g4 raw Actor+same Critic+11-slot pool; fresh Adam/RNG/env; unchanged .60 combat, G combat .05 and existing historical KL. Do not conflate historical KL reference with G parent.'
text = json.dumps(d, ensure_ascii=False, indent=2)+'\n'
(BASE/'ROUND_DECLARATION.json').write_text(text)
Path('reports/je09_bun_20261015/stage_h2_declaration_20261016.json').write_text(text)
print(json.dumps(dict(declared=True, namespace=str(BASE), fixed_generations=32)))
