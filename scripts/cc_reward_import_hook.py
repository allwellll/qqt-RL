"""Apply a reward-variant patch right after the first real import of a module, whichever finder resolves it.

Why: cc_attack_reward_v2 / cc_fountain_reward_v1 install_on_import() put their finder at sys.meta_path[0], but the
worker then runs model_cuda_overlay.install(), which inserts the overlay finder at [0] in front of it. The overlay
finder resolves scripts.r3_items_reward itself, so the variant hook never fired (L2_ATK V1 since 12:00 and L2_FTN V1
trained on the base r3 coefficients; check: candidate PRESSURE / r3_pressure == .01 exactly).
This finder delegates to every other finder in the current meta_path at lookup time (the overlay included), wraps
the resulting loader, and patches the module after it executed. It imports neither jax nor the reward module.
Receipt: if CC_REWARD_RECEIPT is set, writes {module file, version, sha256 of patched sources} there after patching.
usage (before the worker starts): from scripts.cc_reward_import_hook import hook; hook(variant_module)
"""
import hashlib, importlib.abc, json, os, sys, time

TARGET = 'scripts.r3_items_reward'


def _receipt(variant, module):
    path = os.environ.get('CC_REWARD_RECEIPT')
    if not path:
        return
    data = dict(version=variant.VERSION, module=module.__name__, file=getattr(module, '__file__', None),
                cuda_overlay=bool(getattr(module, 'CUDA_OVERLAY', False)), caps=[round(float(x), 6) for x in module.CAPS],
                patched_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in variant.SOURCES.items()},
                pid=os.getpid(), unix=time.time())
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, variant, name):
        self.variant, self.name = variant, name

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        spec = None
        for finder in list(sys.meta_path):
            find = getattr(finder, 'find_spec', None)
            spec = find(fullname, path, target) if find else None
            if spec is not None:
                break
        assert spec is not None and spec.loader is not None, fullname
        loader, variant = spec.loader, self.variant
        original = loader.exec_module
        def exec_module(module):
            original(module); variant.install(module); _receipt(variant, module)
        loader.exec_module = exec_module
        return spec


def hook(variant, name=TARGET):
    """Insert the finder now and again right after model_cuda_overlay.install() (which inserts its finder at [0]).
    model_cuda_overlay imports only hashlib/importlib/sys/pathlib, so importing it here does not touch jax."""
    if name in sys.modules:
        module = variant.install(sys.modules[name]); _receipt(variant, module); return module
    finder = _Finder(variant, name)
    sys.meta_path.insert(0, finder)
    from scripts import model_cuda_overlay as M
    original = M.install
    def install():
        out = original()
        if finder in sys.meta_path:
            sys.meta_path.remove(finder); sys.meta_path.insert(0, finder)
        return out
    M.install = install
    return finder
