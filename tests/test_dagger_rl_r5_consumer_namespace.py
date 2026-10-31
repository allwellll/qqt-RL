"""R5 read-only consumers must never source R4 endpoint directories."""
import ast
from pathlib import Path


def test_r5_consumers_do_not_reference_old_endpoint_namespace():
    for p in Path('.hermes').glob('dagger_rl_r5_*.py'):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert 'r4_reservoir_candidate' not in node.value, p.name
