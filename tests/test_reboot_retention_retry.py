import sys
import pytest
from scripts import checkpoint_retention as R


def test_watcher_preserves_incomplete_audit_and_retries_without_deletion(tmp_path,monkeypatch):
    root=tmp_path
    (root/'runs/iter_pbt_v1').mkdir(parents=True)
    monkeypatch.setattr(R,'ROOT',root)
    monkeypatch.setattr(sys,'argv',['retention','--apply','--watch','--interval','5'])
    calls=[]
    def bad():
        calls.append('audit')
        raise RuntimeError('Open-file audit incomplete: inaccessible process table')
    monkeypatch.setattr(R,'build_plan',bad)
    monkeypatch.setattr(R,'apply_plan',lambda *a:calls.append('delete'))
    monkeypatch.setattr(R.time,'sleep',lambda *a:(_ for _ in ()).throw(SystemExit(0)))
    with pytest.raises(SystemExit):R.main()
    proof=R.read(root/'runs/iter_pbt_v1/RETENTION_STATUS.json')
    assert proof['status']=='BLOCKED' and proof['retry_enabled']
    assert proof['deletion_not_attempted_after_failed_audit'] and calls==['audit']
