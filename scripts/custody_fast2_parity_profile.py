"""New independent real-tick parity evidence, preserves first profile."""
import inspect
from types import ModuleType
from scripts import bun_native_v2_custody_fast_profile as original


def main():
    source = inspect.getsource(original)
    source = source[:source.index("if __name__=='__main__':")]
    source = source.replace('custody_performance_v1', 'custody_performance_v2')
    source = source.replace('bun_native_v2_custody_fast as F', 'bun_native_v2_custody_fast2 as F')
    source = source.replace("'scripts/bun_native_v2_custody_fast.py']", "'scripts/bun_native_v2_custody_fast.py','scripts/bun_native_v2_custody_fast2.py','scripts/custody_distance_candidate.py']")
    module = ModuleType('custody_fast2_parity_profile')
    module.__file__ = __file__
    exec(compile(source, __file__, 'exec'), module.__dict__)
    module.main()


if __name__ == '__main__':
    main()
