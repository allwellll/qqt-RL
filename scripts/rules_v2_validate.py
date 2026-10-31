"""Run regression suites with an explicit rule ABI and local tests namespace."""
import argparse
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-rules', action='store_true')
    parser.add_argument('--reviewed-exclusions', action='store_true')
    parser.add_argument('tests', nargs='*')
    args, extra = parser.parse_known_args()
    # Some installed packages export `tests`; cross-test imports must resolve
    # to this repository without modifying those regression files.
    package = types.ModuleType('tests')
    package.__path__ = [str(Path('tests').resolve())]
    sys.modules['tests'] = package
    if args.new_rules:
        from scripts.rules_v2_runtime import install
        install()
    import pytest
    paths = args.tests or ['tests']
    options = ['-q', '--tb=short']
    if args.reviewed_exclusions:
        # User-reviewed pre-existing collection debt; these files depend on a
        # helper absent from both rule versions. Do not manufacture the helper.
        excluded = ('tests/test_je08_boundaries.py', 'tests/test_je08_integration.py')
        paths = [p for p in paths if p not in excluded]
        options += ['--deselect=' + p for p in excluded]
        # deselect operates after collection; ignore is additionally necessary
        # to prevent the known missing import from aborting all other files.
        options += ['--ignore=' + p for p in excluded]
    if not args.new_rules:
        options += ['--ignore=tests/test_campaign_side_parity_round%d.py' % i
                    for i in range(2, 6)]
    raise SystemExit(pytest.main(options + paths + extra))


if __name__ == '__main__':
    main()
