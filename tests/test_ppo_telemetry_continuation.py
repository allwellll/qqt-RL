"""Inherited replay telemetry must compose with normalized evidence once."""
import ast
from pathlib import Path


def normalized_source():
    source=Path('scripts/je10_worker.py').read_text()
    return source.replace('                    **selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index))),\n','')


def test_reinstalling_phase_diagnostics_preserves_normalized_evidence_exactly():
    from scripts.phase_balanced_replay_recipe import patched_diagnostics
    from scripts.ppo_advantage_diagnostic_recipe import patched_worker
    source=normalized_source()
    assert '**collector_evidence,' in source
    result=patched_worker(patched_diagnostics(source))
    assert result==source
    assert result.count('**selected_ppo_diagnostic(')==0


def test_normalized_installer_repairs_duplicate_legacy_unpack_without_numerical_change():
    from scripts.ppo_advantage_diagnostic_recipe import patched_worker
    source=normalized_source()
    duplicate=source.replace('                    **collector_evidence,',
        '                    **selected_ppo_diagnostic(batch,(int(dev),int(tick),int(env_index))),\n                    **collector_evidence,')
    result=patched_worker(duplicate)
    assert result==source and patched_worker(result)==result
    ast.parse(result)
