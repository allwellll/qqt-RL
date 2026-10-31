import inspect

from scripts.dagger_rl_native_teacher import single_with_masks
from scripts import dagger_rl_oracle as original


def test_diagnostic_adapter_changes_only_native_mask_provider():
    expected = inspect.getsource(original.single).replace('def single(st,ax):', 'def single_with_masks(st,ax,mm,bm):')
    expected = expected.replace('    mm,bm=E.env.legal_mask(st)\n', '')
    assert single_with_masks.adapted_source == expected


def test_original_oracle_source_is_not_modified():
    assert 'mm,bm=E.env.legal_mask(st)' in inspect.getsource(original.single)
