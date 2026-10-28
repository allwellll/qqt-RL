"""Scheduling and throughput guard without importing a JAX backend."""
import statistics

def batches(games,completed):
    missing=[i for i in range(games) if i not in completed]
    return [missing] if missing else []

def end_checkpoint(update,start):
    return update-start==80

def reduce_concurrency(current,baseline,samples):
    if len(samples)<3 or not baseline:return current
    if statistics.median(samples[-3:])>=baseline*.95:return current
    return current//2
