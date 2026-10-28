from tests.external_artifacts import requires_artifacts
"""Real student snapshot: known imminent native respawn invalidates slide plan."""
import pickle
from pathlib import Path
import jax
from scripts import combat_item_engine as T,dagger_rl_oracle as O


@requires_artifacts('runs/dagger_rl_20261024/r1/audit_student_opportunities.pkl')
def test_reject_slide_that_enters_known_imminent_respawn_threat():
 T.configure()
 p=Path('runs/dagger_rl_20261024/r1/audit_student_opportunities.pkl')
 xs=pickle.load(p.open('rb'))
 x=next(x for x in xs if x['source']=='O5_g95-train-004' and x['tick']==817)
 assert int(x['states'].bun_respawn[0,1])==6
 labels,_=jax.jit(O.teacher)(x['states'],x['aux'])
 assert not labels.any(),'public respawn timer+spawn position predicts threat before complete slide'
