"""Explicit reusable XLA calls for K4's wide mixed-state argument ABI.

Compile once, then use the compiled stage's explicit Python call path, which
prepends recorded const_args on every invocation. This does not repair module
globals created inside a trace: those predictors must be initialized before
tracing. Actual full-operator recovery, rather than the adapter alone, is the
acceptance gate. No rollout/reward/optimizer expressions or samples change.
"""
import jax


class CompiledOperator:
    def __init__(self,function):
        self.function=function;self.compiled=None

    def __call__(self,*args):
        if self.compiled is None:self.compiled=self.function.lower(*args).compile()
        # The public compiled stage uses this same call implementation as its
        # fallback, but its version-specific params object is private. Keep
        # this opt-in adapter pinned and covered by actual complete recovery.
        result,_,_=type(self.compiled).call(self.compiled._params,*args)
        return result

    def abi(self):
        if self.compiled is None:raise RuntimeError('operator not compiled')
        p=self.compiled._params;e=self.compiled._executable
        return dict(jax_version=jax.__version__,call='Compiled.call explicit const_args; no repeated jit C++ fastpath',
            const_args=len(p.const_args),dynamic_leaves=self.compiled.in_tree.num_leaves,
            executable_kept_arguments=len(getattr(e,'_kept_var_idx',())),
            simplified_jaxpr_constants=bool(jax.config.jax_use_simplified_jaxpr_constants))
