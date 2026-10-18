"""Continuous extension of a topology distance field, prepared for successors.

Each open cell is split into four triangles around its center. Adjacent open
cells share corner values; disconnected diagonal cells do not share values.
This is a continuous planning potential, not an exact continuous geodesic or Q.
"""
import jax
import jax.numpy as jnp
from scripts.je08_bun_potential import INF


def sampled_distance(pos,distance,blocked,channels):
    """Gather local vertices without materializing per-player full map copies.

    pos is [env,player,2], distance [env,channel,h,w], and blocked [env,h,w].
    Center vertices preserve grid values. Each corner takes the minimum local
    component value plus half an edge; goal-only corners are zero.
    Edge interpolation is therefore identical from both traversable sides.
    """
    n,players=pos.shape[:2];h,w=blocked.shape[-2:]
    cell=jnp.floor(pos).astype(jnp.int32)
    inside=jnp.all((cell>=0)&(cell<jnp.array([h,w])),-1)
    safe_cell=jnp.clip(cell,jnp.array([0,0]),jnp.array([h-1,w-1]))
    rows=jnp.arange(n)[:,None]
    center=distance[rows,channels,safe_cell[...,0],safe_cell[...,1]].astype(jnp.float32)
    usable=inside&~blocked[rows,safe_cell[...,0],safe_cell[...,1]]&(center<INF)
    signs=jnp.array([[-1,-1],[-1,1],[1,-1],[1,1]],jnp.int32)
    vertical=signs*jnp.array([1,0]);horizontal=signs*jnp.array([0,1])
    offsets=jnp.stack([jnp.zeros_like(signs),vertical,horizontal,signs],1)
    cells=cell[:,:,None,None,:]+offsets[None,None]
    valid=jnp.all((cells>=0)&(cells<jnp.array([h,w])),-1)
    cells=jnp.clip(cells,jnp.array([0,0]),jnp.array([h-1,w-1]))
    rr=jnp.arange(n)[:,None,None,None]
    open_cell=valid&~blocked[rr,cells[...,0],cells[...,1]]
    connected=jnp.stack([open_cell[...,0],open_cell[...,0]&open_cell[...,1],
        open_cell[...,0]&open_cell[...,2],
        open_cell[...,0]&open_cell[...,3]&(open_cell[...,1]|open_cell[...,2])],-1)
    values=distance[rr,channels[:,:,None,None],cells[...,0],cells[...,1]].astype(jnp.float32)
    connected &= values<INF
    corners=jnp.min(jnp.where(connected,values,float(INF)),axis=-1)+.5
    corners=jnp.where((~connected|(values==0)).all(-1),0.,corners)
    y,x=(pos-cell.astype(jnp.float32)-.5).transpose(2,0,1)
    ay,ax=jnp.abs(y),jnp.abs(x)
    left=jnp.where(y<0,corners[...,0],corners[...,2])
    right=jnp.where(y<0,corners[...,1],corners[...,3])
    upper=jnp.where(x<0,corners[...,0],corners[...,1])
    lower=jnp.where(x<0,corners[...,2],corners[...,3])
    vertical_value=(1.-2*ay)*center+(ay-x)*left+(ay+x)*right
    horizontal_value=(1.-2*ax)*center+(ax-y)*upper+(ax+y)*lower
    # Native steering often centers the orthogonal axis before it can turn.
    # A small continuous cell-centering cost makes those alignment steps
    # measurable without changing center values or grid-edge continuity.
    value=jnp.where(ay>=ax,vertical_value,horizontal_value)+.125*(ay+ax)
    return jax.lax.stop_gradient(jnp.where(usable,value,float(INF)))


def continuous_distance(pos,distance,blocked):
    return sampled_distance(pos[:,None],distance[:,None],blocked,
        jnp.zeros((pos.shape[0],1),jnp.int32))[:,0]
