"""Scheduling and throughput guard without importing a JAX backend."""
import statistics

def batches(games,completed):
    missing=[i for i in range(games) if i not in completed]
    return [missing] if missing else []

def end_checkpoint(update,start):
    return update-start==80

def reduce_concurrency(current,baseline,samples):
    current=max(1,current)
    if len(samples)<3 or not baseline:return current
    if statistics.median(samples[-3:])>=baseline*.95:return current
    return max(1,current//2)

def recover_concurrency(current,ceiling,baseline,samples,now,last_change,cooldown=300):
    """A legacy zero cap is repaired; subsequent recovery needs warm evidence."""
    current=max(1,current);ceiling=max(1,ceiling)
    if current>=ceiling or now-last_change<cooldown:return min(current,ceiling)
    if len(samples)>=3 and baseline and statistics.median(samples[-3:])>=baseline*.95:
        return min(ceiling,current+1)
    return current

def dedicated_gpus(config):
    return {int(g) for g in config.get('dedicated_gpus',[])}

def gpu_lock_name(gpu,lane=None):
    return f'NATIVE_GPU{gpu}'+(f'_lane{lane}' if lane is not None else '')+'.lock'

def native_priority(request,first_names=()):
    # Match full request identities, including the holdout suffix.
    from scripts.iter_pbt_eval_scheduling import priority
    return (request['name'] not in first_names,)+priority(request)

def evaluation_slots(config,training_gpus=(),recovery_pending=True):
    dedicated=dedicated_gpus(config)
    if dedicated:
        return [(g,i) for g in sorted(dedicated) if g not in training_gpus
                and (recovery_pending or g not in config.get('temporary_borrowed_gpus',[]))
                for i in range(config.get('lanes_per_gpu',6))]
    return [(g,None) for g in range(8)]

def returnable_gpus(config,active_gpus,recovery_waiting):
    # Assigned work stays on its card; an idle borrowed card need not wait for
    # another card's compilation once every recovery request has an owner.
    if recovery_waiting:return set()
    return set(config.get('temporary_borrowed_gpus',[]))-set(active_gpus)
