import numpy as np
import pytest
from scripts.bun_spawn_side import manifest,native_pairs,validate,side_counts

def test_all_native_and_balanced_seeds():
    assert len(native_pairs())==16
    for seed in range(500):
        d=manifest(seed,256);a=np.asarray(d['spawn_cells'])
        assert side_counts(a)['both_own']==256
        assert len({tuple(x.flatten()) for x in a})==16
        assert min(np.abs(a[:,0]-a[:,1]).sum(-1))>=10

@pytest.mark.parametrize('pair',[[[4,12],[4,2]],[[4,7],[4,12]],[[4,2],[4,7]],[[4,-1],[4,12]],[[4,2.1],[4,12]]])
def test_reject_wrong_center_and_invalid(pair):
    with pytest.raises(ValueError):validate(pair)
