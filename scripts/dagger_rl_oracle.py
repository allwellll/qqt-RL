"""Public-current-state banana plan labels; never changes learner actions or reward.

Static complete-slide forecast abstains on hazards, pushables, nearby enemies,
other fields, incomplete slides and nonpositive net walking-adjusted benefit.
It is a limited item oracle, not a proof of safety against future enemy actions.
"""
import jax
import jax.numpy as j
from scripts import carrier_mechanism_engine as E
from scripts.je10_route_oracle import continuous_distance
DIR=j.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]])
OPP=j.array([1,0,3,2])


def combine(route,item):
    return j.where(item.any((1,2))[:,None,None],item,route)


def set_loss(move,ability,mask):
    joint=jax.nn.log_softmax(move)[:,:,None]+jax.nn.log_softmax(ability)[:,None,:]
    valid=mask.any((1,2));loss=-jax.nn.logsumexp(j.where(mask,joint,-1e9),(1,2))
    return j.where(valid,loss,0).sum()/j.maximum(valid.sum(),1)


def one_plan(st,distance,direction,field,release):
    """Exact native AABB own movement; release leaves original cell then reverses.

    Contact direction points home. Full slide ends at native collision, not a
    chosen advantageous intermediate frame. 64 tick cap rejects long plans.
    """
    p=st.pos[0];home=direction;away=OPP[home]
    origin=p.astype(j.int32);blocked=st.wall|st.brick|st.core.pushable|(st.fuse>0)
    start=continuous_distance(p[None],distance[None],blocked[None])[0]
    # Existing field may already be armed, or need its owner to leave the cell.
    armed=j.where(release,False,st.field_armed[field[0],field[1]])
    def body(_,c):
        pos,armed,slide,finished,bad,t,contact,first=c
        on=j.all(pos.astype(j.int32)==field)
        move=j.where(~armed,away,home)
        # Do not teach a long walk to an unrelated field.
        move=j.where(slide,home,move)
        speed=j.where(slide,.67,.18)
        nxt=E.env.base._move_player(pos,move,True,blocked,speed/E.env.base.STEP)
        moved=j.linalg.norm(nxt-pos)
        arm=armed|~j.all(nxt.astype(j.int32)==field)
        hit=arm&j.all(nxt.astype(j.int32)==field)&~slide
        stop=slide&(moved<.67*.35)
        # Native expiry/respawn is public deterministic state, not future RNG.
        revived=(~st.alive[1])&((t+1)>=j.maximum(st.bun_respawn[1],1))
        enemy_pos=j.where(revived,st.bun_spawn_pos[1],st.pos[1])
        free=(st.alive[1]&(st.trapped[1]<=t+1))|revived
        enemy_close=(j.abs(nxt-enemy_pos).sum()<4.)&free
        cell=j.clip(nxt.astype(j.int32),j.array([0,0]),j.array([12,14]))
        other=(st.field_item[cell[0],cell[1]]>0)&~j.all(cell==field)
        invalid=(~slide&~hit&(moved<.01))|enemy_close|other
        active=~finished
        return (j.where(active,nxt,pos),arm,slide|hit,finished|stop,bad|(active&invalid),t+active.astype(j.int32),contact|hit,j.where(t==0,move,first))
    end=jax.lax.fori_loop(0,64,body,(p,armed,False,False,False,j.int32(0),False,j.int32(4)))
    pos,armed,slide,finished,bad,t,contact,first=end
    dist=continuous_distance(pos[None],distance[None],blocked[None])[0]
    gain=start-dist-.18*t
    valid=finished&contact&~bad&(gain>.5)&(start<32767.)
    return valid,gain,first,t,pos


def single(st,ax):
    # Only current public geometry/field ownership feeds the plan. Custody ledger
    # certifies real acquisition but contains no future trajectory or RNG.
    distance=ax['carry_items']['distance'][0]
    own=(st.field_item==1)&(st.field_owner==0)
    pos=st.pos[0];cell=j.clip(pos.astype(j.int32),j.array([0,0]),j.array([12,14]))
    near=own&(j.abs(j.indices((13,15))[0]+.5-pos[0])+j.abs(j.indices((13,15))[1]+.5-pos[1])<=2.)
    has=near.any();idx=j.argmax(near.reshape(-1));field=j.array([idx//15,idx%15])
    release=~has
    field=j.where(release,cell,field)
    potential=(st.held_item[0]==1)&(st.field_item[cell[0],cell[1]]==0)
    eligible=(st.bun_carried[0]==1)&ax['carry_items']['ledger']['custody'][0]&st.alive[0]&(st.trapped[0]==0)&(st.move_status[0]==0)&~ax['terminal']
    eligible&=has|potential
    eligible&=~((st.fuse>0).any()|(st.core.blast_linger>0).any()|st.core.pushable.any())
    # Low-cost public-state precondition; the forecast also checks the entire path.
    eligible&=(~st.alive[1])|(st.trapped[1]>0)|(j.abs(st.pos[1]-pos).sum()>=8.)
    def plan(_):return jax.vmap(lambda d:one_plan(st,distance,d,field,release))(j.arange(4))
    def empty(_):return (j.zeros(4,bool),j.zeros(4),j.full(4,4,j.int32),j.zeros(4,j.int32),j.zeros((4,2)))
    valid,gain,first,t,end=jax.lax.cond(eligible,plan,empty,None)
    best=j.max(j.where(valid,gain,-1e9));valid&=gain>=best-1e-5
    mm,bm=E.env.legal_mask(st)
    ability=j.where(release,2,0)
    mask=(j.arange(5)[:,None,None]==first[None,None,:])&(j.arange(3)[None,:,None]==ability)&valid[None,None,:]
    mask=mask.any(-1)&mm[0,:,None]&bm[0,None,:]
    return mask,j.array([eligible,has,release&mask.any(),(~release)&mask.any(),j.max(j.where(valid,gain,0.)),j.max(j.where(valid,t,0))])


def teacher(states,aux):
    return jax.vmap(single)(states,aux)
