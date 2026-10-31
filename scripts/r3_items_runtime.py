"""Process-local passive contact event ABI, preserving R2 simulator dynamics."""
import inspect


def install():
    from jax_bomb import je07_bun_env_r1_items as env
    original=env.step
    scope=dict(original.__globals__);code=inspect.getsource(original)
    def replace(a,b):
        nonlocal code
        assert code.count(a)==1,a
        code=code.replace(a,b)
    replace('    trap_hit = jnp.zeros((2,), jnp.bool_)',
        '    r3_contact_type=jnp.zeros((2,),jnp.int8)\n    r3_contact_cell=jnp.full((2,2),-1,jnp.int32)\n    trap_hit = jnp.zeros((2,), jnp.bool_)')
    replace('        trap_hit = trap_hit.at[player].set(trigger)',
        '        r3_contact_type=r3_contact_type.at[player].set(jnp.where(trigger,item,0))\n        r3_contact_cell=r3_contact_cell.at[player].set(jnp.where(trigger,jnp.array([row,column]),-1))\n        trap_hit = trap_hit.at[player].set(trigger)')
    replace('        "trap_hit": trap_hit,','        "r3_contact_type": r3_contact_type,\n        "r3_contact_cell": r3_contact_cell,\n        "trap_hit": trap_hit,')
    exec(compile(code,__file__,'exec'),scope)
    env.step=scope['step']
    from scripts.bun_native_v2_runtime import install as native
    return native()
