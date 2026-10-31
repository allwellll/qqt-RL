"""Summarize all fixed Bank endpoints; never choose peaks or launch training."""
import json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
from scripts.bank_ratio_slot_train import B,CP

def main():
    train=json.loads((B/'TRAINING_COMPLETE.json').read_text());assert train['passed'] and len(train['runs'])==6
    protocol=CP/'protocols/bank_ratio_eval_v1.json';p=json.loads(protocol.read_text());rows=[]
    for name,spec in p['conditions'].items():
        target=Path(p['namespace'])/name;done=json.loads((target/'COMPLETE.json').read_text())
        assert done['passed'] and sha(target/'SUMMARY.json')==done['summary_sha256']
        value=json.loads((target/'SUMMARY.json').read_text())
        rows.append(dict(condition=name,split=spec['split'],checkpoint=spec['checkpoint'],source_sha256=sha(target/'SUMMARY.json'),summary=value))
    atomic_json(B/'RESULTS.json',dict(passed=True,protocol_sha256=sha(protocol),fixed_endpoint_only=True,results=rows))
    atomic_json(B/'CAMPAIGN_COMPLETE.json',dict(passed=True,training_complete_sha256=sha(B/'TRAINING_COMPLETE.json'),results_sha256=sha(B/'RESULTS.json'),evaluation_conditions=len(rows),scientific_superiority_not_asserted=True,unix=time.time()))
    from scripts.bank_ratio_slot_train import notify
    notify('BANK_RATIO_CAMPAIGN_COMPLETE',dict(runs=6,evaluation_conditions=len(rows),results=str(B/'RESULTS.json')))

if __name__=='__main__':main()
