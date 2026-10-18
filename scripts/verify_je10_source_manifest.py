"""Verify each published JE10 source and its local import dependency closure."""
import argparse
import ast
import hashlib
import json
import posixpath
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'reports/je10_source_manifest_20261018.json'
OPTIONAL_IMPORTS = {
    'web/sim.js': {'web/time_astar_ai.js'},
    'web/review-sim/route-v35.js': {'web/review-sim/time_astar_ai.js'},
}


def dependencies(filename, root=ROOT):
    path = root / filename
    text = path.read_text()
    result = set()
    if path.suffix == '.py':
        package = filename.split('/')[:-1]
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                base = '.'.join(prefix + ([node.module] if node.module else []))
                modules = [base] + [base + '.' + a.name for a in node.names]
            else:
                continue
            for module in modules:
                target = module.replace('.', '/') + '.py'
                if (root / target).is_file():
                    result.add(target)
                init = module.replace('.', '/') + '/__init__.py'
                if (root / init).is_file() and init != filename:
                    result.add(init)
    elif path.suffix == '.js':
        for module in re.findall(r'''require\(['"]([^'"]+)['"]\)''', text):
            if module.startswith('.'):
                target = posixpath.normpath(str(Path(filename).parent / module))
                if not Path(target).suffix:
                    target += '.js'
                if not (root / target).is_file():
                    if target in OPTIONAL_IMPORTS.get(filename, set()):
                        continue  # Existing try/catch plugin, unused by Bun Hunter.
                    raise ValueError(f'missing JS module: {target}')
                result.add(target)
    # Runtime file reads / generated Module source cannot be found by imports.
    if filename == 'scripts/je05_web_hunter.js':
        result.update(['scripts/eval_web_hunter.js', 'scripts/eval_web_hunter_residual.js'])
    if filename in ('jax_bomb/je05_bun_env.py', 'jax_bomb/je07_bun_env.py', 'scripts/eval_web_hunter.js'):
        result.add('web/assets/maps/levels.json')
    return sorted(result)


def verify(manifest=MANIFEST):
    data = json.loads(Path(manifest).read_text())
    files = data['files']
    for filename, record in files.items():
        path = ROOT / filename
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('manifest path escapes repository')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['published_sha256']:
            raise ValueError(f'published source hash mismatch: {filename}')
        actual = dependencies(filename) if path.suffix in ('.py', '.js') else []
        if actual != record['dependencies']:
            raise ValueError(f'dependency list changed: {filename}')
        if not set(actual) <= files.keys():
            raise ValueError(f'dependency closure incomplete: {filename}')
    print(f"verified {len(files)} files and their local dependency edges")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    verify(parser.parse_args().manifest)
