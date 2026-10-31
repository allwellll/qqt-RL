"""Confirmation must preserve the screened math and isolate every writer."""
import ast
import importlib.util
from pathlib import Path
import pytest


def test_confirmation_preserves_screened_training_mechanism_and_new_namespace():
    source=Path('scripts/dagger_rl_r3_train.py')
    assert source.exists(), 'independent confirmation implementation missing'
    old=ast.parse(Path('scripts/dagger_rl_r2_train.py').read_text())
    new=ast.parse(source.read_text().replace('scripts.dagger_rl_r3_train','scripts.dagger_rl_r2_train'))
    names=['install_provenance','initial','collector','supervised','objective','updater','aggregate','commit','smoke','restore']
    functions=lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.FunctionDef)}
    prior,current=functions(old),functions(new)
    for name in names:
        assert prior[name]==current[name], f'confirmation unexpectedly changed {name}'
    spec=importlib.util.find_spec('scripts.dagger_rl_r3_run')
    assert spec is not None
    owner=Path(spec.origin).read_text()
    assert "runs/dagger_rl_20261024/r3_confirmation" in owner
    assert "prior=R/'runs/dagger_rl_20261024/r2_parent_anchor'" in owner
    cpu_functions=lambda path:{node.name:ast.dump(node,include_attributes=False) for node in ast.parse(Path(path).read_text()).body if isinstance(node,ast.FunctionDef)}
    assert cpu_functions('scripts/dagger_rl_r3_cpu.py')['natural']==cpu_functions('scripts/dagger_rl_r2_cpu.py')['natural']


@pytest.mark.parametrize('declined', [False, True])
def test_conditional_confirmation_cannot_start_before_scientific_selection(tmp_path,monkeypatch,declined):
    from scripts import dagger_rl_r3_run as run
    monkeypatch.setattr(run,'B',tmp_path)
    if declined:
        (tmp_path/'SCIENTIFIC_SELECTION.json').write_text('{"execute_confirmation": false}')
    with pytest.raises(AssertionError,match='scientific selection'):
        run.authorize()
