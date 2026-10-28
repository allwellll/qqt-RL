"""Independent Monte Carlo summaries; repeated rolls cluster by source episode."""
import numpy as np


def event_outcomes_from(flags,valid,start=0):
    """Inclusive branch/deviation transition onward; ignore earlier events and padding."""
    flags=np.asarray(flags,bool);valid=np.asarray(valid,bool)
    if flags.ndim!=2 or flags.shape[0]!=valid.size or not 0<=start<len(valid):
        raise ValueError('invalid event timeline or start')
    events=flags[start:,:3]&valid[start:,None]
    delivered,died,dropped=events.any(0)
    failure=events[:,1:].any(-1)
    delivery_ticks=np.flatnonzero(events[:,0]);failure_ticks=np.flatnonzero(failure)
    first_delivery=delivery_ticks[0] if delivery_ticks.size else np.inf
    first_failure=failure_ticks[0] if failure_ticks.size else np.inf
    return dict(died_later=bool(died),dropped_later=bool(dropped),eventual_delivery=bool(delivered),
        immediate_death=bool(events[0,1]),immediate_drop=bool(events[0,2]),
        direct_custody_delivery=bool(first_delivery<first_failure),
        recovery_delivery_after_failure=bool(first_failure<first_delivery<np.inf))


def discounted_returns(rewards,terminal,gamma=.995,tail=None):
    r=np.asarray(rewards,float);d=np.asarray(terminal,bool)
    if r.shape!=d.shape:raise ValueError('return/terminal shape mismatch')
    out=np.zeros(r.shape[:-1]) if tail is None else np.asarray(tail,float)
    for t in range(r.shape[-1]-1,-1,-1):out=r[...,t]+gamma*np.where(d[...,t],0.,out)
    return out


def calibration(prediction,returns):
    v=np.asarray(prediction,float);samples=np.asarray(returns,float)
    y=samples.mean(-1) if samples.ndim>v.ndim else samples
    error=v-y;variance=float(np.var(y))
    return dict(states=int(v.size),explained_variance=float(1-np.var(error)/variance) if variance>1e-12 else None,
        target_variance=variance,rmse=float(np.sqrt(np.mean(error**2))),bias=float(error.mean()),
        error_variance=float(np.var(error)),mean_v=float(v.mean()),mean_return=float(y.mean()),
        within_state_rollout_variance=float(samples.var(-1,ddof=1).mean()) if samples.ndim>v.ndim else None)


def cluster_interval(values,clusters,seed=20261023,repetitions=5000):
    values=np.asarray(values,float);clusters=np.asarray(clusters);ids=np.unique(clusters)
    if len(ids)<2:return [None,None]
    means=np.array([values[clusters==i].mean() for i in ids]);rng=np.random.default_rng(seed)
    boot=means[rng.integers(0,len(ids),(repetitions,len(ids)))].mean(-1)
    return np.quantile(boot,[.025,.975]).tolist()


def validate_bank_record(record):
    if not (record['before_carried']<0 and record['after_carried']>=0
        and record['learner_alive_before'] and record['learner_alive_after']
        and record['stock_steal'] and (not record['enemy_alive'] or record['enemy_trapped']>0)
        and record['provenance']=='actual_env_step'):
        raise ValueError('Bank needs real eligible acquisition event')
    for k in ['source_episode','layout','split']:
        if not record.get(k):raise ValueError('missing causal source identity')


def paired_q_contrasts(returns,names):
    x=np.asarray(returns,float);out=dict(estimand='subset Q contrasts',actions=names)
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            delta=x[i]-x[j];out[names[i]+'_minus_'+names[j]]=dict(mean=float(delta.mean()),
                paired_samples=int(delta.size),se=float(delta.std(ddof=1)/np.sqrt(delta.size)) if delta.size>1 else None)
    return out


def calibration_intervals(prediction,returns,seed=20261023,repetitions=5000):
    """Resample source episodes, keeping all repeated rollouts in each cluster."""
    v=np.asarray(prediction,float);r=np.asarray(returns,float)
    if v.size<2:return {k:[None,None] for k in ['rmse','bias','target_variance','explained_variance']}
    rng=np.random.default_rng(seed);ii=rng.integers(0,len(v),(repetitions,len(v)))
    y=r.mean(-1) if r.ndim>v.ndim else r
    err=v[ii]-y[ii];variance=np.var(y[ii],-1)
    metrics=dict(rmse=np.sqrt(np.mean(err**2,-1)),bias=err.mean(-1),target_variance=variance,
        explained_variance=np.where(variance>1e-12,1-np.var(err,-1)/np.maximum(variance,1e-12),np.nan))
    return {k:np.nanquantile(a,[.025,.975]).tolist() if np.isfinite(a).any() else [None,None] for k,a in metrics.items()}
