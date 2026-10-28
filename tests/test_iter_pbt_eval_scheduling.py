from scripts.iter_pbt_eval_scheduling import pending_latest, priority, fresh_cohort, lag_report

def test_latest_uses_slot_chunk_not_inherited_update_and_preserves_holdout():
    rows=[dict(name='old',slot=0,chunk=4,update=900,split='dev'),dict(name='new',slot=0,chunk=5,update=100,split='dev'),dict(name='best_holdout',split='holdout'),dict(name='initial_x',split='dev')]
    assert pending_latest(rows)=={'new','best_holdout'}
    assert priority(dict(name='best_holdout',split='holdout',priority='best_candidate')) < priority(rows[1])

def test_only_two_chunk_fresh_scores_can_replace():
    assert fresh_cohort(7,{str(i):9 for i in range(8)})
    assert not fresh_cohort(6,{str(i):9 for i in range(8)})
    assert not fresh_cohort(10,{'0':9})

def test_lag_counts_saved_chunk_not_live_unsaved_attempt():
    rows=[dict(name='a',slot=0,chunk=9,update=100,split='dev'),dict(name='b',slot=0,chunk=10,update=20,split='dev')]
    assert lag_report(rows,{'a'})['0']==dict(latest_saved='b',latest_scored='a',saved_chunk=10,scored_chunk=9,lag_chunks=1)

def test_cohort_rejects_stale_scores_before_loading_checkpoint(tmp_path,monkeypatch):
    from scripts import iter_pbt_owner as owner
    monkeypatch.setattr(owner,'BASE',tmp_path)
    assert owner.cohort_decisions(2,{'0':5,'1':4})==[]

def test_superseded_initials_do_not_starve_latest_parent_is_kept():
    rows=[dict(name='parent_dev',split='dev'),dict(name='initial_a',split='dev'),dict(name='old',split='dev',slot=1,chunk=2,update=100),dict(name='new',split='dev',slot=1,chunk=2,update=140)]
    assert pending_latest(rows)=={'parent_dev','new'}

def test_global_cpu_stage_tokens_bound_parallel_work(tmp_path):
    import threading,time
    from concurrent.futures import ThreadPoolExecutor
    from scripts.iter_pbt_eval_resources import stage_token
    gate=threading.Lock();running=0;peak=0
    def work(_):
        nonlocal running,peak
        with stage_token(tmp_path,'test',2):
            with gate:running+=1;peak=max(peak,running)
            time.sleep(.02)
            with gate:running-=1
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(work,range(6)))
    assert peak==2 and running==0
