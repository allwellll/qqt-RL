from pathlib import Path

from scripts import campaign_remaining_cpu as c


def test_readiness_uses_inputs_never_task_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'sha', lambda p: 'test-hash')
    p = dict(namespace=str(tmp_path), required_evidence=['result.json'],
             source_checkpoint_protocol='source.json', model_selection=[])
    job = c.execution_job(dict(id='NATURAL-BANK-COVERAGE', protocol='protocol.json'), p)
    assert str(tmp_path / 'result.json') not in job['ready_files']
    assert job['args'][1] == 'scripts.campaign_remaining_cpu'


def test_source_split_is_shared_across_models_modes():
    assert c.source_split(1) == c.source_split(1)
    assert set(c.source_split(i) for i in range(96)) == {'train', 'dev', 'holdout'}


def test_window_delivery_requires_same_custody():
    g = dict(window_funnel=dict(windows=[dict(window_id='w', kind='death', stock_pickup_ticks=[5])]),
             carry_episodes=dict(episodes=[dict(start_tick=5, outcome='died', stock_window_link={'window_id':'w'}),
                                           dict(start_tick=7, outcome='delivered', stock_window_link=None)]))
    r = c.linked_funnel([g])
    assert r['death']['windows_with_pickup'] == 1
    assert r['death']['same_custody_deliveries'] == 0


def test_missing_data_is_not_scientific_completion(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        c.raw_source(tmp_path / 'missing.json')
