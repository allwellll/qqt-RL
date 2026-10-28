import numpy as np
from scripts.campaign_paired_train_canonical import canonical_collector

def test_initial_placement_retains_all_logical_state_and_reuses_collector(tmp_path):
    calls=[]
    def original(p,s,a,k,*bank):
        calls.append((s,a,k))
        return s,a,k
    receipt=tmp_path/'PLACEMENT.json'
    collect=canonical_collector(original,receipt)
    initial=(np.array([[1.,2.]],np.float32),{'paid':np.array([[3.]],np.float32)},np.array([[1,2]],np.uint32))
    out=collect(None,*initial)
    np.testing.assert_array_equal(out[0],initial[0])
    np.testing.assert_array_equal(out[1]['paid'],initial[1]['paid'])
    np.testing.assert_array_equal(out[2],initial[2])
    assert receipt.exists()
    collect(None,*out)
    assert len(calls)==2 and calls[1][0] is out[0]
