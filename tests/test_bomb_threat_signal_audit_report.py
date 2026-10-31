import json
import pytest
from scripts import bomb_threat_signal_audit_report as R
from scripts.population_run_state import sha


def fixture(root):
    out=root/'audit';out.mkdir();trace=out/'ticks.json'
    t={k:[[0.]] for k in ('shadow_balance','balance_residual','first_safe_threat','topology_events','chain_events',
        'bomb_request','actual_place','rejected','paid','cancelled')}
    t.update(live=[[True]],completed=[[True]])
    trace.write_text(json.dumps(t));manifest=out/'MANIFEST.json';manifest.write_text('{"games_per_mode":1}')
    e={k:0. for k in ('threat_events','topology_events','chain_events','requests','actual_bombs','rejected','paid','admitted',
        'cancelled','cancel_bad','cancel_terminal','cancel_inactive','closing_pending','deaths','deliveries','net_reward','preterminal_ticks','tail_ticks')}
    e.update(index=0,complete=True,reject_reasons={str(i):0 for i in range(6)},paid_by_type_proportional_shadow=[0.,0.,0.])
    r=dict(name='frozen',mode='greedy',episodes=[e],complete_games=1,trace_files={'audit/ticks.json':sha(trace)},
        policy_sha256='a'*64,first_transition_passive_exact=True)
    (out/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':sha(manifest),'results':[r]}))
    return out,trace


def test_retained_trace_recount_and_hash_acceptance(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'ROOT',tmp_path);out,trace=fixture(tmp_path)
    _,rows=R.fullgame_rows(out)
    assert rows[0]['games']==1 and rows[0]['completed']==1
    trace.write_text('{}')
    with pytest.raises(AssertionError):R.fullgame_rows(out)
