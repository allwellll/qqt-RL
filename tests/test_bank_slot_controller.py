import pytest
from scripts import authorized_experiment_campaign as c
from scripts.campaign_eval_resources import cpu_group


def test_nonoverlapping_slot_and_evaluation_core_ranges():
    train=[set(range(16+3*i,19+3*i)) for i in range(6)]
    assert all(not a&b for i,a in enumerate(train) for b in train[i+1:])
    used=set().union(*train)
    assert not used&set(range(16))
    for lane,count in [('cpu',3),('evaluation',4),('analysis',2),('js',1)]:
        for i in range(count):assert not used&set(cpu_group(list(range(192)),lane,i,40))


def test_protocol_rebind_requires_blocked_status_and_user_amendment(monkeypatch,tmp_path):
    import json,hashlib
    p=tmp_path/'auth.json';p.write_text(json.dumps(dict(id='CAMPAIGN-BANK-RATIO-AB-20261026-V1-AMENDMENT01',authorized_by_user='explicit')))
    h=hashlib.sha256(p.read_bytes()).hexdigest();jobs=[dict(id='BANK-RATIO-AB',status='BLOCKED_PROTOCOL',complete='old/BLOCKED.json')]
    replacement=dict(id='BANK-RATIO-AB',status='REGISTERED',complete='new/COMPLETE.json')
    result=c.rebind_authorized_block(jobs,replacement,str(p),h)
    assert result['passed'] and jobs[0]['prior_protocol_block']['status']=='BLOCKED_PROTOCOL'
    assert jobs[0]['status']=='REGISTERED'
    assert not c.rebind_authorized_block(jobs,replacement,str(p),h)['passed']
