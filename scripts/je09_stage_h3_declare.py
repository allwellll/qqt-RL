"""Declare H3 in a new namespace before any smoke or training."""
import json
import subprocess
from pathlib import Path

BASE = Path('runs/je09_bun_20261016/stage_h3_natural_combat_coverage_pfsp8_v17')
assert not BASE.exists(), 'H3 namespace must be new and immutable'
BASE.mkdir(parents=True)
source = json.loads(Path('runs/je09_bun_20261016/stage_h2_item_coverage_pfsp8_v15/ROUND_DECLARATION.json').read_text())
source.update(
    experiment_id='JE09-STAGE-H3-NATURAL-COMBAT-COVERAGE-PFSP8-20261016-v17',
    namespace=str(BASE),
    parent_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    parent='runs/je09_bun_20261016/stage_h2_item_coverage_pfsp8_v15/formal/bun392_bridge_pfsp8/gen000000',
    seed=3001000000,
)
source['seeds'] = {k: int(v) + 200000000 for k,v in source['seeds'].items()}
source['seeds']['train'] = 3001000000
source['budget'] = dict(formal_generations=32, formal_fresh_ticks=67108864,
    technical_gpu_generations=2, technical_gpu_ticks=4194304,
    candidates=[0,4,8,12,16,24,32], atomic_checkpoint_generations=[0,4,8,12,16,24,32], max_rounds=1)
source['change'] = 'Only increase full natural-scene lesson sampling to 0.10; retain H2 item opportunity coverage, reward, return/drop supervision, network, opponent pool, native timing, and all assist-off protocols.'
source['full_scene_coverage'] = dict(scene=8, fraction=0.10, combat_fraction=0.60,
    controlled_bridge_fraction=0.25, source='native full scene; no initialized carry/drop/pickup')
source['reason'] = 'H2 item opportunity coverage is real but full natural first-pickup/capture remains zero; target the earliest combat-to-attributed-drop funnel with one lesson-mixture variable.'
source['stop'] = 'Fixed32 generations/67108864 fresh ticks; no capability trend stopper. Stop only NaN/OOM/state corruption or declared gross native safety failure.'
source['selection'] = 'Observation only; choose by complete existing protocols after the fixed budget, never by a short score.'
source['evaluation_policy'] = 'Existing assist-off authority/fixed32/controlled/Hunter/Tactical/carry/full2400 protocols run asynchronously; they never control training.'
source['parent_preservation'] = 'H2 g0, equivalent to G g4 raw Actor + matching Critic + complete 11-slot pool; fresh Adam/RNG/env.'
(BASE/'ROUND_DECLARATION.json').write_text(json.dumps(source, ensure_ascii=False, indent=2)+'\n')
Path('reports/je09_bun_20261015/stage_h3_declaration_v17_20261016.json').write_text(json.dumps(source, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'declared': True, 'namespace': str(BASE), 'budget_generations': 32}))
