"""CPU-only acceptance of retained reward ABI using actual GPU witness data."""
import json
from pathlib import Path
import pickle
import numpy as np
from scripts.population_run_state import atomic_json,sha


def main():
    import jax.numpy as j
    from scripts import campaign_reward_v2 as reward
    from scripts import combat_item_engine as engine
    base=Path('runs/authorized_campaign_20261026')
    protocol=base/'protocols/reward_task_semantics_v2_technical02.json'
    spec=json.loads(protocol.read_text())
    for name,h in spec['source_sha256'].items():assert sha(name)==h,name
    witness=Path('runs/gpu_efficiency_20261026/performance_probe_v1/hoisted_updater_cold/FINAL_WITNESS.pkl')
    data=pickle.load(witness.open('rb'))
    names=(*engine.E.ops.COMPONENTS[:-1],*engine.events.METRICS,'combined_total')
    parts=np.asarray(data['diag'][0])
    assert parts.shape[-1]==len(names)
    # This is a per-transition wiring witness, not reconstructed episode return.
    # Independent ledger/death/reset semantics are exercised by focused tests.
    selected,_,out=reward.select(j.asarray(parts),names,j.zeros(parts.shape[:-1],bool),j.zeros(parts.shape[:-1]))
    actual=np.asarray(data['diag'][3]);selected=np.asarray(selected)
    assert np.allclose(parts[...,names.index('combined_total')],actual)
    changed=np.abs(selected-actual)>1e-7
    assert changed.any(),'Intervention must reach a real collector batch'
    assert np.isfinite(selected).all()
    for name in reward.KEEP:
        np.testing.assert_array_equal(np.asarray(out)[...,names.index(name)],parts[...,names.index(name)])
    result=dict(passed=True,protocol=str(protocol),protocol_sha256=sha(protocol),
        witness=str(witness),witness_sha256=sha(witness),actual_collector_transitions=int(actual.size),
        reward_changed_transitions=int(changed.sum()),finite=True,retained_channels_exact=list(reward.KEEP),
        limitation='Actual batch ABI/wiring only; independent row selection is not full-trajectory return. GPU smoke verifies optimizer and complete fresh restore.',
        RED='runs/authorized_campaign_20261026/REWARD_V2_LEDGER_RED.log',
        GREEN='runs/authorized_campaign_20261026/REWARD_V2_LEDGER_GREEN.log',controls_gpu=False)
    atomic_json(base/'reward_wiring/RESULT.json',result)
    atomic_json(base/'reward_wiring/COMPLETE.json',dict(passed=True,result_sha256=sha(base/'reward_wiring/RESULT.json')))
    print(json.dumps(result))


if __name__=='__main__':main()
