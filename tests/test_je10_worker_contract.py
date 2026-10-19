"""Host-worker regression: diagnostic arrays must not replace run budget."""
import ast
from pathlib import Path


def test_generation_budget_has_one_binding():
    module=ast.parse(Path('scripts/je10_worker.py').read_text())
    worker=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    bindings=[n for n in ast.walk(worker) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='target']
    assert len(bindings)==1, 'diagnostic oracle must not overwrite the fixed generation budget'


def test_event_log_contains_final_diagnostics():
    source=Path('scripts/je10_worker.py').read_text()
    write=source.index("with (out/'events.jsonl').open('a')")
    assert write>source.index("row['actual_league_exposure']")
    assert write>source.index("row['signal_decomposition']")
