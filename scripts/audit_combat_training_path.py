"""Audit local Python/Node dependencies, including runtime-compiled source literals.

Run from the repository root. External run/checkpoint inputs remain explicit and
are intentionally outside this source audit. No imports or workers are executed.
"""
import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path

PACKAGES = ('scripts', 'jax_bomb', 'qqt_rl')
OPAQUE = {'r.py', 'a.py', 'a.js', 'q'}
# Legacy sim catches this optional bot import; the native training/eval path
# never selects TimeAStarAI. Keep the exception visible in the manifest.
OPTIONAL_NODE = {name: {'web/time_astar_ai.js'} for name in
                 ('web/sim.js', 'web/sim-stock-window-v1.js', 'web/sim-bun-native-v2.js')}
OPTIONAL_NODE['web/review-sim/stock-legacy-20261028.js'] = {'web/review-sim/time_astar_ai.js'}
OPTIONAL_NODE['web/review-sim/bun-native-v2-20261104.js'] = {'web/review-sim/time_astar_ai.js'}


def module_path(name):
    base = name.replace('.', '/')
    return next((p for p in (base + '.py', base + '/__init__.py')
                 if Path(p).is_file()), None)


def dependencies(name):
    path = Path(name)
    source = path.read_text()
    found = set()
    if path.suffix == '.py':
        for node in ast.walk(ast.parse(source)):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                parent = node.module or ''
                if node.level:
                    prefix = '.'.join(path.parts[:-node.level])
                    parent = prefix + ('.' + parent if parent else '')
                names = [parent] + [parent + '.' + item.name for item in node.names]
            for module in names:
                if module.split('.')[0] in PACKAGES:
                    local = module_path(module)
                    if local:
                        found.add(local)
                    elif not module.endswith('.*') and '.' not in module and not Path(module).is_dir():
                        raise FileNotFoundError(module)
    # Worker transformations and generic launcher manifests contain imports as strings.
    for module in re.findall(r'\b(?:scripts|jax_bomb|qqt_rl)(?:\.[A-Za-z_][\w]*)+', source):
        local = module_path(module)
        if local:
            found.add(local)
    for local in re.findall(r'(?:scripts|jax_bomb|qqt_rl|web|tests)/[\w/.-]+\.(?:py|js|json)', source):
        if Path(local).is_file():
            found.add(local)
    if path.suffix == '.js':
        for match in re.finditer(r'''require\(['"](\.[^'"]+)['"]\)|from\s+['"](\.[^'"]+)['"]''', source):
            local = Path(os.path.normpath(path.parent / (match[1] or match[2])))
            if not local.suffix:
                local = local.with_suffix('.js')
            if not local.is_file():
                if local.as_posix() in OPTIONAL_NODE.get(name, set()):
                    continue
                raise FileNotFoundError(f'{name}: {local}')
            found.add(local.as_posix())
    return sorted(found)


def audit(roots):
    pending = list(roots)
    graph = {}
    while pending:
        name = pending.pop()
        if name in graph:
            continue
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.name in OPAQUE:
            raise ValueError(f'invalid source dependency: {name}')
        graph[name] = dependencies(name)
        pending.extend(graph[name])
    return dict(schema='combat-training-source-closure-v1', roots=sorted(roots),
                optional_dependencies={name: sorted(paths) for name, paths in OPTIONAL_NODE.items() if name in graph},
                files={name: dict(sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest(),
                                 dependencies=graph[name]) for name in sorted(graph)})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', default='docs/combat_league_v2_dependency_manifest_20261106.json')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    manifest = Path(args.manifest)
    old = json.loads(manifest.read_text())
    result = audit(old['roots'])
    if args.write:
        manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    elif old != result:
        raise RuntimeError('source closure changed; review and regenerate with --write')
    print(f"PASS: {len(result['roots'])} roots, {len(result['files'])} local dependencies")


if __name__ == '__main__':
    main()
