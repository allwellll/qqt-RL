"""Host evidence counts actual selected return queries without training changes."""
import numpy as np


def test_host_audit_counts_first_and_second_queries_with_native_terminal_reset():
    from scripts.return_query_diversity import observed_query_counts
    ctx=np.zeros((2,24,2,127),np.float32)
    ctx[...,54]=3
    for tick in [0,8,11,19]:ctx[:,tick,:,53]=1
    ctx[:,10,:,47]=1
    result=observed_query_counts(ctx)
    assert result['ordinary_return_queries']==16
    assert result['first_return_queries']==result['additional_return_queries']==8
    assert result['max_queries_per_episode_chunk']==2
    assert result['quota_violations']==0 and result['controls_training'] is False


def test_other_phases_and_preparation_are_excluded_from_return_audit():
    from scripts.return_query_diversity import observed_query_counts
    ctx=np.zeros((1,16,2,127),np.float32);ctx[...,53]=1
    ctx[...,54]=2
    ctx[:,:,1,54]=3;ctx[:,:,1,110]=1
    result=observed_query_counts(ctx)
    assert result['ordinary_return_queries']==0
    assert result['max_queries_per_episode_chunk']==0
    ctx[...,54]=3;ctx[...,110]=0;ctx[...,57]=1
    result=observed_query_counts(ctx)
    assert result['ordinary_return_queries']==0
    assert result['kind3_delivery_queries']==32


def test_audit_discloses_invalid_quota_without_an_ability_gate():
    from scripts.return_query_diversity import observed_query_counts
    ctx=np.zeros((1,16,1,127),np.float32);ctx[...,54]=3
    ctx[:,[0,8,15],:,53]=1
    result=observed_query_counts(ctx)
    assert result['max_queries_per_episode_chunk']==3
    assert result['quota_violations']==1 and result['controls_training'] is False


def test_worker_audit_is_installed_only_as_host_telemetry():
    import ast
    from pathlib import Path
    from scripts.return_query_recipe import patched_worker
    source=Path('scripts/je10_worker.py').read_text()
    source=source.replace(
        "\n            from scripts.return_query_diversity import observed_query_counts\n"
        "            row['dagger']['return_query_diversity']=observed_query_counts(ctx)", '')
    after=patched_worker(source)
    assert patched_worker(after)==after
    a,b=map(ast.parse,[source,after])
    before={n.name:ast.dump(n) for n in a.body if isinstance(n,ast.FunctionDef)}
    after_funcs={n.name:ast.dump(n) for n in b.body if isinstance(n,ast.FunctionDef)}
    assert {name for name in before if before[name]!=after_funcs[name]}=={'worker'}
    assert "row['dagger']['return_query_diversity']=observed_query_counts(ctx)" in after
