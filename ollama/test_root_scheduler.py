"""Actual journal preparation and root fences; synthetic trusted registry views."""
from datetime import datetime,timedelta,timezone
from copy import deepcopy

import pytest

from .inference_arbiter import Arbiter,Conflict,StintPolicy
from .inference_root_scheduler import RootPreparingArbiter


@pytest.fixture
def scheduler(tmp_path):
    a=RootPreparingArbiter(tmp_path/'jobs.sqlite',registry='https://registry.test/api/v1',registry_actor='provider',service_id='models')
    yield a
    a.close()


def call(a,owner='one',key='first',job='parent'):
    grant=a.submit(owner,key,'host/gpu','text','execution')
    execution=a.enqueue(grant['id'],owner,key,{'fixture':True},job_key=job)
    return grant,execution


def admitted(prep,root_id='root-1',lease_id='lease-1'):
    lease={'id':lease_id,'resource_id':'host/gpu','agent_id':'provider','service_id':'models',
           'exclusive':1,'memory_mb':1000,'status':'active',
           'expires_at':(datetime.now(timezone.utc)+timedelta(seconds=120)).isoformat()}
    body={'root_id':root_id,'initial_lease_id':lease_id,'current_lease_id':lease_id,
          'lifetime':'explicit_end','resource_id':'host/gpu','state':'open','lease':lease}
    view={'id':'host/gpu','allocation_mode':'shared','expiry_policy':'hold',
          'specs':{'unified_memory_mb':1000},'leases':[deepcopy(lease)],'unsettled':[]}
    return body,view


def terminal(body):
    ended=deepcopy(body);ended['state']='complete';ended['lease']['status']='released'
    return ended


def test_prepare_is_durable_and_keeps_first_call_queued(scheduler):
    a=scheduler;_,first=call(a);call(a,'two','other','other')
    prep=a.prepare_root('host/gpu')
    assert prep['job_id']==first['job_id'] and prep['execution_id']==first['id']
    assert a.owns_job('host/gpu') and a.execution(first['id'],'one')['state']=='queued'
    assert a.prepare_root('host/gpu')['id']==prep['id']
    assert a.dispatch_next('host/gpu',permitted_model=lambda *_:True) is None
    body,view=admitted(prep);a.confirm_root(prep['id'],body,view)
    started=a.dispatch_next('host/gpu',permitted_model=lambda *_:True)
    assert started['execution']['id']==first['id']


def test_no_demand_no_preparation_and_earlier_legacy_work_is_not_skipped(scheduler):
    a=scheduler;assert a.prepare_root('host/gpu') is None
    _,legacy=call(a,job=None);call(a,'two','later','later')
    with pytest.raises(Conflict):a.prepare_root('host/gpu')
    assert a.execution(legacy['id'],'one')['state']=='queued'
    assert a.db.execute('SELECT COUNT(*) FROM requester_root_preparations').fetchone()[0]==0


def test_physical_admission_does_not_replace_model_headroom(scheduler):
    a=scheduler;_,first=call(a);prep=a.prepare_root('host/gpu')
    body,view=admitted(prep);a.confirm_root(prep['id'],body,view)
    with pytest.raises(Conflict):a.dispatch_next('host/gpu')
    assert a.dispatch_next('host/gpu',permitted_model=lambda *_:None) is None
    assert a.execution(first['id'],'one')['state']=='queued'


@pytest.mark.parametrize('change',['queued','expired','actor','service','bytes','debt','other-active','wrong-root','free-policy'])
def test_bad_or_changed_registry_evidence_withdraws_admission(scheduler,change):
    a=scheduler;_,first=call(a);prep=a.prepare_root('host/gpu');body,view=admitted(prep)
    a.confirm_root(prep['id'],body,view)
    if change=='queued':body['lease']['status']='queued'
    elif change=='expired':body['lease']['expires_at']='2000-01-01T00:00:00+00:00'
    elif change=='actor':body['lease']['agent_id']='other'
    elif change=='service':body['lease']['service_id']='other'
    elif change=='bytes':body['lease']['memory_mb']=999
    elif change=='debt':view['unsettled']=[deepcopy(body['lease'])]
    elif change=='other-active':view['leases'].append({**body['lease'],'id':'another'})
    elif change=='wrong-root':body['root_id']='replacement'
    elif change=='free-policy':view['expiry_policy']='free'
    with pytest.raises(Conflict):a.confirm_root(prep['id'],body,view)
    assert a.preparation(prep['id'])['state']=='prepared'
    assert a.dispatch_next('host/gpu',permitted_model=lambda *_:True) is None
    assert a.execution(first['id'],'one')['state']=='queued'


def test_parent_finish_waits_for_exact_physical_cleanup_before_next_job(scheduler):
    a=scheduler;_,first=call(a);_,other=call(a,'two','other','other')
    prep=a.prepare_root('host/gpu');body,view=admitted(prep);a.confirm_root(prep['id'],body,view)
    a.dispatch_next('host/gpu',permitted_model=lambda *_:True);a.finish(first['id'],{})
    a.end_job('one','parent')
    assert a.preparation(prep['id'])['state']=='ending'
    assert a.job('one','parent','host/gpu')['state']=='owned'
    assert a.prepare_root('host/gpu')['id']==prep['id']
    assert a.dispatch_next('host/gpu',permitted_model=lambda *_:True) is None
    with pytest.raises(Conflict):a.settle_preparation(prep['id'],body)
    with pytest.raises(Conflict):a.settle_preparation(prep['id'],terminal({**body,'root_id':'wrong'}))
    a.settle_preparation(prep['id'],terminal(body))
    assert a.job('one','parent','host/gpu')['state']=='closed'
    assert a.prepare_root('host/gpu')['job_id']==other['job_id']
    assert a.execution(other['id'],'two')['state']=='queued'


def test_end_before_root_ack_preserves_preparation_for_no_claim_end(scheduler):
    a=scheduler;grant,first=call(a);prep=a.prepare_root('host/gpu')
    a.end_job('one','parent');a.request_release(grant['id'],'one')
    assert a.execution(first['id'],'one')['state']=='cancelled'
    assert a.preparation(prep['id'])['state']=='ending'
    no_claim={'state':'complete','resource_id':'host/gpu','service_id':'models','lifetime':'explicit_end',
              'never_admitted':True,'root_id':None,'initial_lease_id':None,'current_lease_id':None,'lease':None}
    a.settle_preparation(prep['id'],no_claim)
    assert a.job('one','parent','host/gpu')['state']=='closed'


def test_explicit_yield_keeps_job_but_uses_new_physical_preparation_and_fifo(scheduler):
    a=scheduler;_,first=call(a);prep=a.prepare_root('host/gpu');body,view=admitted(prep)
    a.confirm_root(prep['id'],body,view);a.dispatch_next('host/gpu',permitted_model=lambda *_:True)
    a.finish(first['id'],{});a.end_job('one','parent',finish=False)
    a.settle_preparation(prep['id'],terminal(body))
    _,other=call(a,'two','other','other');_,resumed=call(a,key='resumed')
    second=a.prepare_root('host/gpu')
    assert second['job_id']==other['job_id'] and second['id']!=prep['id']
    # An old physical completion cannot end the newer root or resumed parent.
    a.settle_preparation(prep['id'],terminal(body))
    assert a.preparation(second['id'])['state']=='prepared'
    assert a.job('one','parent','host/gpu')['state']=='queued'
    assert resumed['job_id']==first['job_id']


def test_restart_has_no_cached_admission_and_legacy_reopen_refuses(scheduler):
    a=scheduler;_,first=call(a);prep=a.prepare_root('host/gpu');body,view=admitted(prep)
    a.confirm_root(prep['id'],body,view)
    with pytest.raises(Conflict):Arbiter(a.path,stint_policy=StintPolicy())
    b=RootPreparingArbiter(a.path,registry='https://registry.test/api/v1',registry_actor='provider',service_id='models')
    try:
        assert b.prepare_root('host/gpu')['id']==prep['id']
        assert b.dispatch_next('host/gpu',permitted_model=lambda *_:True) is None
        b.confirm_root(prep['id'],body,view)
        assert b.dispatch_next('host/gpu',permitted_model=lambda *_:True)['execution']['id']==first['id']
    finally:b.close()


def test_wrong_provider_cannot_reopen_or_replace_preparation(scheduler):
    with pytest.raises(PermissionError):
        RootPreparingArbiter(scheduler.path,registry='https://registry.test/api/v1',registry_actor='different',service_id='models')


def test_unknown_child_keeps_root_and_end_debt(scheduler):
    a=scheduler;_,first=call(a);prep=a.prepare_root('host/gpu');body,view=admitted(prep)
    a.confirm_root(prep['id'],body,view);a.dispatch_next('host/gpu',permitted_model=lambda *_:True)
    a.quarantine_running();a.end_job('one','parent')
    assert a.execution(first['id'],'one')['state']=='uncertain'
    assert a.preparation(prep['id'])['state']=='admitted'
    with pytest.raises(Conflict):a.settle_preparation(prep['id'],terminal(body))
    assert a.owns_job('host/gpu')


def test_concurrent_preparation_uses_one_slot_and_preserves_all_calls(scheduler):
    from concurrent.futures import ThreadPoolExecutor
    a=scheduler;_,first=call(a);_,other=call(a,'two','other','other')
    def prepare(_):
        b=RootPreparingArbiter(a.path,registry='https://registry.test/api/v1',registry_actor='provider',service_id='models')
        try:return b.prepare_root('host/gpu')
        finally:b.close()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(prepare,range(8)))
    assert len({r['id'] for r in rows})==1 and rows[0]['execution_id']==first['id']
    assert a.execution(first['id'],'one')['state']=='queued'
    assert a.execution(other['id'],'two')['state']=='queued'


def test_logical_settlement_failure_keeps_exact_physical_receipt_for_retry(scheduler):
    import sqlite3
    a=scheduler;_,first=call(a);prep=a.prepare_root('host/gpu');body,view=admitted(prep)
    a.confirm_root(prep['id'],body,view);a.dispatch_next('host/gpu',permitted_model=lambda *_:True)
    a.finish(first['id'],{});a.end_job('one','parent')
    a.db.execute("""CREATE TRIGGER refuse_logical_close BEFORE UPDATE OF state ON requester_jobs
        WHEN NEW.state='closed' BEGIN SELECT RAISE(ABORT,'logical close failure'); END""")
    with pytest.raises(sqlite3.IntegrityError):a.settle_preparation(prep['id'],terminal(body))
    assert a.preparation(prep['id'])['state']=='ending' and a.owns_job('host/gpu')
    a.db.execute('DROP TRIGGER refuse_logical_close')
    assert a.settle_preparation(prep['id'],terminal(body))['state']=='complete'
    assert a.job('one','parent','host/gpu')['state']=='closed'


def test_confirmed_receipt_does_not_outlive_other_controller_withdrawal(scheduler):
    a=scheduler;call(a);prep=a.prepare_root('host/gpu');body,view=admitted(prep)
    a.confirm_root(prep['id'],body,view)
    b=RootPreparingArbiter(a.path,registry='https://registry.test/api/v1',registry_actor='provider',service_id='models')
    try:b.withdraw_root(prep['id'])
    finally:b.close()
    assert a.dispatch_next('host/gpu',permitted_model=lambda *_:True) is None


def test_yielded_root_job_cannot_reopen_through_unqualified_legacy_scheduler(scheduler):
    a=scheduler;_,first=call(a);prep=a.prepare_root('host/gpu');body,view=admitted(prep)
    a.confirm_root(prep['id'],body,view);a.dispatch_next('host/gpu',permitted_model=lambda *_:True)
    a.finish(first['id'],{});a.end_job('one','parent',finish=False)
    a.settle_preparation(prep['id'],terminal(body))
    with pytest.raises(Conflict):Arbiter(a.path,stint_policy=StintPolicy())
