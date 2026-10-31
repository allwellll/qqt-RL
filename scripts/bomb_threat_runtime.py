"""Process-local opt-in keeps all old ITER-PBT rule replays reproducible."""
from functools import partial

def install():
    from jax_bomb import je07_bun_env as env
    if getattr(env.step,'_stock_window_v1',False):return
    original=env.step
    def step(*args,**kwargs):
        kwargs['require_stock_window']=True
        return original(*args,**kwargs)
    step._stock_window_v1=True
    env.step=step
