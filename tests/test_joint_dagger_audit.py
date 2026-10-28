"""Exact-input oracle conflicts must respect masks and full joint pairs."""
import numpy as np


def test_no_common_joint_pair_detected_without_claiming_geometry_conflict():
    from scripts.audit_joint_dagger import conflict_groups
    obs=np.zeros((3,1,1,1),np.float32);gv=np.zeros((3,2),np.float32)
    mm=np.ones((3,5),bool);am=np.ones((3,3),bool);sets=np.zeros((3,5,3),bool)
    sets[0,[0,1],0]=True;sets[1,[1,2],0]=True;sets[2,[0,2],0]=True
    out=conflict_groups(obs,gv,mm,am,sets,np.ones(3,bool))
    assert out['duplicate_input_groups']==1 and out['no_common_pair_groups']==1
    assert out['examples'][0]['indices']==[0,1,2]
    # Distinct observed state or legal head masks cannot be called the same input.
    gv[2,0]=1;am[1,2]=False
    out=conflict_groups(obs,gv,mm,am,sets,np.ones(3,bool))
    assert out['duplicate_input_groups']==0 and out['no_common_pair_groups']==0


def test_invalid_or_illegal_targets_are_excluded_and_joint_correlations_retained():
    from scripts.audit_joint_dagger import conflict_groups
    obs=np.zeros((4,1,1,1),np.float32);gv=np.zeros((4,2),np.float32)
    mm=np.ones((4,5),bool);am=np.ones((4,3),bool);sets=np.zeros((4,5,3),bool)
    sets[0,0,0]=sets[0,1,1]=True;sets[1,0,1]=sets[1,1,0]=True
    sets[2,2,2]=True;am[2,2]=False
    sets[3]=sets[0]
    out=conflict_groups(obs,gv,mm,am,sets,np.array([1,1,1,0],bool))
    assert out['valid_rows']==2 and out['no_common_pair_groups']==1
