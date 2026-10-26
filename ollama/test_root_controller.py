"""Actual controller/supervisor/manager; deterministic provider and registry views."""
import asyncio
from types import SimpleNamespace

import pytest

from .test_root_models import env  # noqa: F401
from .inference_root_controller import RootController
from .inference_arbiter import Conflict
from .inference_root_scheduler import RootPreparingArbiter
from .inference_supervisor import Supervisor


@pytest.fixture
def controlled(env):
    e=env;path=e.a.path;e.a.close();entered=[];gate={'event':None}
    async def execute(grant,body):
        entered.append(grant['model'])
        if gate['event'] is not None:await gate['event'].wait()
        return {'fixture':grant['model']}
    def factory(path,*,stint_policy=None):
        return RootPreparingArbiter(path,registry=e.store.registry,registry_actor=e.store.actor,
            service_id=e.store.service,stint_policy=stint_policy)
    s=Supervisor(path,execute,lambda _:False,journal_factory=factory)
    e.a=s.journal;e.binding.scheduler=e.a
    c=RootController(s,e.binding,work_permitted=True)
    yield SimpleNamespace(e=e,s=s,c=c,entered=entered,gate=gate)
    s.close()


async def drain(x):
    # Drain only actual tracked tasks, not elapsed work or an arbitrary retry.
    while True:
        tasks=[t for t in (*x.s.preparations,*x.s.workers.values()) if not t.done()]
        if not tasks:return
        await asyncio.gather(*tasks)


def enqueue(x,key,model='text'):
    a=x.e.a;g=a.submit('one',key,'host/gpu',model,'execution')
    call=a.enqueue(g['id'],'one',key,{'fixture':True},job_key='parent')
    x.s.pump('host/gpu');return g,call


def test_controller_text_vision_text_one_claim_then_exact_finish(controlled):
    async def scenario():
        x=controlled;x.s.pump('host/gpu');await drain(x)
        assert x.e.a.execution(x.e.call['id'],'one')['state']=='complete'
        enqueue(x,'vision','vision');await drain(x)
        enqueue(x,'text-again');await drain(x)
        assert x.entered==['text','vision','text']
        assert x.e.m.ollama.loads==['text','text'] and x.e.m.ollama.unloads==['text']
        assert {r['lease_id'] for r in x.e.binding._rows()}=={'root-lease'}
        assert not [p for method,p in x.e.control if p.endswith('/roots') and method=='POST']
        assert x.e.a.owns_job('host/gpu')
        x.s.end_job('one','parent');await drain(x)
        assert x.e.a.job('one','parent','host/gpu')['state']=='closed'
        assert x.e.body['lease']['status']=='released' and not x.e.m.models
        x.e.sc.request_lease.assert_not_called();x.e.sc.release_lease.assert_not_called()
    asyncio.run(scenario())


def test_unknown_startup_never_repeats_or_dispatches_on_later_hints(controlled):
    async def scenario():
        x=controlled;x.e.m.ollama.crash=True
        x.s.pump('host/gpu');await drain(x)
        assert x.e.m.ollama.loads==['text'] and not x.entered
        assert x.e.binding._rows()[0]['state']=='loading'
        for _ in range(20):x.c.request('host/gpu')
        await drain(x)
        assert x.e.m.ollama.loads==['text'] and not x.entered
        assert x.e.a.execution(x.e.call['id'],'one')['state']=='queued'
        assert x.e.body['lease']['status']=='active'
        assert not x.s.preparations or all(t.done() for t in x.s.preparations)
    asyncio.run(scenario())


def test_work_permission_false_does_not_load_and_actual_resume_event_wakes(controlled):
    async def scenario():
        x=controlled;x.c.set_work_permitted(False);await drain(x)
        assert x.e.m.ollama.loads==[] and not x.entered and x.c.guard('host/gpu') is False
        x.c.set_work_permitted(True);await drain(x)
        assert x.entered==['text']
    asyncio.run(scenario())


def test_inflight_child_is_not_evicted_or_dispatched_again_by_hints(controlled):
    async def scenario():
        x=controlled;x.gate['event']=asyncio.Event();x.s.pump('host/gpu');await x.c.drain()
        while not x.entered:await asyncio.sleep(0)
        _,vision=enqueue(x,'vision','vision')
        await x.c.drain()
        assert x.e.m.ollama.unloads==[] and not x.e.child
        x.s.end_job('one','parent')
        assert x.e.a.owns_job('host/gpu')
        x.gate['event'].set();await drain(x)
        assert x.entered==['text','vision']
        assert x.e.a.execution(vision['id'],'one')['state']=='complete'
        assert x.e.a.job('one','parent','host/gpu')['state']=='closed'
    asyncio.run(scenario())


def test_failed_absence_holds_parent_until_new_event_without_idle_retry(controlled):
    async def scenario():
        x=controlled;x.s.pump('host/gpu');await drain(x)
        x.e.cold['confirmed']=False;x.s.end_job('one','parent');await drain(x)
        calls=len(x.e.control)
        assert x.e.a.job('one','parent','host/gpu')['state']=='owned'
        assert x.e.a.preparation(x.e.prep['id'])['state']=='ending'
        await asyncio.sleep(0);await asyncio.sleep(0)
        assert len(x.e.control)==calls
        assert x.e.body['lease']['status']=='active'
        x.e.cold['confirmed']=True;x.c.request('host/gpu');await drain(x)
        assert x.e.a.job('one','parent','host/gpu')['state']=='closed'
    asyncio.run(scenario())


def test_failed_release_preserves_same_root_and_recovers_without_new_load(controlled):
    async def scenario():
        x=controlled;x.s.pump('host/gpu');await drain(x)
        x.e.release_failure['enabled']=True;x.s.end_job('one','parent');await drain(x)
        assert x.e.body['lease']['status']=='active'
        assert x.e.a.owns_job('host/gpu') and x.e.a.preparation(x.e.prep['id'])['state']=='ending'
        x.e.release_failure['enabled']=False;x.c.request('host/gpu');await drain(x)
        assert x.e.body['lease']['status']=='released' and not x.e.a.owns_job('host/gpu')
        assert x.e.m.ollama.loads==['text']
    asyncio.run(scenario())


def test_unrelated_event_does_not_create_work_or_withdraw_permission(controlled):
    async def scenario():
        x=controlled
        assert x.c.request('other/gpu') is None
        assert not x.s.preparations and not x.e.m.ollama.loads
        x.s.pump('host/gpu');await drain(x)
        assert x.entered==['text']
    asyncio.run(scenario())


def test_resident_dispatch_still_requires_positive_host_headroom(controlled,monkeypatch):
    async def scenario():
        from . import manager as module
        x=controlled;x.e.binding.publish(x.e.prep['id'],x.e.body,x.e.view())
        x.e.m.load_ollama_model('text')
        monkeypatch.setattr(module.capacity,'collect',lambda:{'memory_available_mb':0,'memory_device_inuse_mb':46000})
        x.s.pump('host/gpu');await drain(x)
        assert not x.entered and x.e.a.execution(x.e.call['id'],'one')['state']=='queued'
        assert x.e.body['lease']['status']=='active'
        monkeypatch.setattr(module.capacity,'collect',lambda:{'memory_available_mb':18000,'memory_device_inuse_mb':46000})
        x.c.request('host/gpu');await drain(x)
        assert x.entered==['text'] and x.e.m.ollama.loads==['text']
    asyncio.run(scenario())


def test_many_hints_during_started_preload_coalesce_without_second_load(controlled):
    import threading
    async def scenario():
        x=controlled;load=x.e.m.ollama.load_model;started=threading.Event();finish=threading.Event()
        def delayed(*args,**kwargs):
            load(*args,**kwargs);started.set()
            if not finish.wait(3):raise AssertionError('fixture preload not released')
        x.e.m.ollama.load_model=delayed
        x.s.pump('host/gpu')
        try:
            assert await asyncio.to_thread(started.wait,2)
            task=x.c._task
            for _ in range(20):assert x.c.request('host/gpu') is task
            assert len(x.s.preparations)==1 and not x.entered
            finish.set();await drain(x)
            assert x.e.m.ollama.loads==['text'] and x.entered==['text']
            assert x.e.binding._rows()[0]['state']=='resident'
        finally:finish.set();await drain(x)
    asyncio.run(scenario())


def test_manager_stats_loop_uses_same_source_and_preserves_existing_sample_clock(controlled,monkeypatch):
    from unittest.mock import Mock
    from . import manager as module
    async def scenario():
        x=controlled;x.e.sc.push_stats=Mock();x.e.m._collect_stats=lambda:{'memory_available_mb':64000,'memory_device_inuse_mb':0}
        assert x.e.m.inference_capacity_source is x.c
        sleep=asyncio.sleep;sleeps=[]
        async def stop_after_sample(delay):
            if delay==15:
                sleeps.append(delay);raise asyncio.CancelledError()
            await sleep(delay)
        monkeypatch.setattr(module.asyncio,'sleep',stop_after_sample)
        with pytest.raises(asyncio.CancelledError):await x.e.m.stats_loop()
        await drain(x)
        assert sleeps==[15] and x.e.sc.push_stats.call_count==1
        assert x.entered==['text'] and x.e.m.ollama.loads==['text']
    asyncio.run(scenario())


def test_unknown_sample_blocks_new_work_until_actual_positive_sample(controlled):
    async def scenario():
        x=controlled;await x.c.refresh({'memory_available_mb':64000,'memory_device_inuse_mb':None});await drain(x)
        assert not x.entered and x.e.m.ollama.loads==[]
        assert x.c.snapshot()['reason']=='host_observation_unknown'
        assert x.e.a.execution(x.e.call['id'],'one')['state']=='queued'
        await x.c.refresh({'memory_available_mb':64000,'memory_device_inuse_mb':0});await drain(x)
        assert x.entered==['text']
    asyncio.run(scenario())


def test_source_pause_and_resume_require_new_observation(controlled):
    async def scenario():
        x=controlled;x.c.pause()
        await x.c.refresh({'memory_available_mb':64000,'memory_device_inuse_mb':0});await drain(x)
        assert not x.entered and x.c.snapshot()['paused']
        x.c.resume();x.s.pump('host/gpu');await drain(x)
        assert not x.entered and x.c.snapshot()['host_known'] is False
        await x.c.refresh({'memory_available_mb':64000,'memory_device_inuse_mb':0});await drain(x)
        assert x.entered==['text'] and x.c.retains_model('text')
    asyncio.run(scenario())


def test_actual_manager_root_heartbeat_notifies_same_controller_without_per_model_renew(controlled):
    async def scenario():
        x=controlled;x.s.pump('host/gpu');await drain(x)
        a=x.e.a;g=a.submit('one','saved-next','host/gpu','text','execution')
        call=a.enqueue(g['id'],'one','saved-next',{'fixture':True},job_key='parent')
        # No fresh enqueue pump: an actual maintenance completion wakes saved
        # demand. This is one real event, not an independently polling retry.
        await x.e.m.renew_model_leases();await drain(x)
        assert a.execution(call['id'],'one')['state']=='complete'
        assert x.entered==['text','text'] and x.e.m.ollama.loads==['text']
        assert len([p for method,p in x.e.control if p.endswith('/renew')])==1
        x.e.sc.renew_lease.assert_not_called()
    asyncio.run(scenario())


def test_failed_heartbeat_keeps_claim_and_needs_fresh_event_without_retry_loop(controlled,monkeypatch):
    async def scenario():
        import httpx
        x=controlled;x.s.pump('host/gpu');await drain(x)
        observe=x.e.client.observe
        def fail(*_):raise httpx.ConnectError('fixture control observation failed')
        monkeypatch.setattr(x.e.client,'observe',fail)
        with pytest.raises(httpx.ConnectError):await x.e.m.renew_model_leases()
        await drain(x);count=len(x.e.control)
        assert x.e.body['lease']['status']=='active' and x.e.a.owns_job('host/gpu')
        assert x.c.guard('host/gpu') is False
        await asyncio.sleep(0);await asyncio.sleep(0)
        assert len(x.e.control)==count and x.e.m.ollama.loads==['text']
        monkeypatch.setattr(x.e.client,'observe',observe)
        enqueue(x,'next');await drain(x)
        assert x.entered==['text','text'] and x.e.m.ollama.loads==['text']
    asyncio.run(scenario())


def test_host_sample_returns_while_service_owned_preload_is_still_running(controlled):
    import threading
    async def scenario():
        x=controlled;load=x.e.m.ollama.load_model;started=threading.Event();finish=threading.Event()
        def delayed(*args,**kwargs):
            load(*args,**kwargs);started.set()
            if not finish.wait(3):raise AssertionError('fixture preload not released')
        x.e.m.ollama.load_model=delayed
        sample=asyncio.create_task(x.c.refresh({'memory_available_mb':64000,'memory_device_inuse_mb':0}))
        try:
            assert await asyncio.to_thread(started.wait,2)
            assert sample.done() and x.c.snapshot()['preparing']
            assert not x.entered
            await x.c.refresh({'memory_available_mb':18000,'memory_device_inuse_mb':46000})
            assert x.c.snapshot()['preparing'] and not finish.is_set()
            finish.set();await drain(x)
            assert x.entered==['text'] and x.e.m.ollama.loads==['text']
        finally:finish.set();await sample;await drain(x)
    asyncio.run(scenario())


@pytest.mark.parametrize('field',['inference_capacity_source','root_binding'])
def test_cached_dispatch_refuses_changed_manager_source_or_accounting_owner(controlled,field):
    async def scenario():
        x=controlled;x.gate['event']=asyncio.Event();x.s.pump('host/gpu');await x.c.drain()
        while not x.entered:await asyncio.sleep(0)
        previous=getattr(x.e.m,field)
        try:
            assert x.c.guard('host/gpu') is True
            setattr(x.e.m,field,object())
            assert x.c.guard('host/gpu') is False
            assert x.e.body['lease']['status']=='active'
        finally:
            setattr(x.e.m,field,previous);x.gate['event'].set();await drain(x)
        assert x.e.a.execution(x.e.call['id'],'one')['state']=='complete'
    asyncio.run(scenario())


def test_root_shutdown_requires_pause_then_fences_late_health_and_host_events(controlled):
    from .test_root_models import shutdown_fixture
    async def scenario():
        x=controlled;tasks=shutdown_fixture(x.e)
        x.s.pump('host/gpu');await drain(x);x.s.end_job('one','parent');await drain(x)
        with pytest.raises(Conflict):x.e.m.shutdown()
        assert all(not t.cancelled for t in tasks)
        x.c.pause();await x.c.drain();assert x.e.m.shutdown()==[]
        assert x.e.m.shutdown()==[]  # Acknowledged stop can be retried.
        assert all(t.cancelled for t in tasks) and x.c.snapshot()['stopped']
        assert x.e.binding.on_health_observed is None
        assert x.c.request('host/gpu') is None
        await x.c.refresh({'memory_available_mb':64000,'memory_device_inuse_mb':0})
        with pytest.raises(Conflict):x.c.resume()
        with pytest.raises(asyncio.CancelledError):await x.e.binding.wait_for_renewal()
        with pytest.raises(Conflict):x.e.m.load_ollama_model('text')
    asyncio.run(scenario())


def test_explicit_restart_suspension_keeps_parent_and_exact_charged_root(controlled):
    from .test_root_models import shutdown_fixture
    async def scenario():
        x=controlled;shutdown_fixture(x.e);x.s.pump('host/gpu');await drain(x)
        before=x.e.a._current_preparation('host/gpu')['id']
        x.c.pause();await x.c.drain()
        proof=await x.e.binding.suspend_for_restart(observation='fixture-fresh-owned-backend-absence')
        assert proof['ownership_retained'] and proof['state']=='suspended'
        assert x.e.m.models=={} and x.e.binding._rows()==[]
        assert x.e.body['lease']['status']=='active' and x.e.body['state']=='open'
        assert x.e.a.owns_job('host/gpu') and x.e.a._current_preparation('host/gpu')['id']==before
        assert x.e.m.shutdown()==[]
        assert not [p for method,p in x.e.control if p.endswith('/settle') or p.endswith('/roots/end')]
        x.e.sc.release_lease.assert_not_called()
    asyncio.run(scenario())


def test_restart_suspension_false_absence_keeps_control_alive_until_proven(controlled):
    from .test_root_models import shutdown_fixture
    async def scenario():
        x=controlled;tasks=shutdown_fixture(x.e);x.s.pump('host/gpu');await drain(x);x.c.pause()
        x.e.cold['confirmed']=False
        with pytest.raises(Conflict):await x.e.binding.suspend_for_restart(observation='fixture-unconfirmed')
        with pytest.raises(Conflict):x.e.m.shutdown()
        assert all(not t.cancelled for t in tasks) and x.e.body['lease']['status']=='active'
        x.e.cold['confirmed']=True
        await x.e.binding.suspend_for_restart(observation='fixture-current-absence')
        assert x.e.m.shutdown()==[]
    asyncio.run(scenario())


def test_restart_suspension_does_not_erase_unknown_execution(controlled):
    async def scenario():
        x=controlled
        async def unknown(*_):raise RuntimeError('fixture unconfirmed execution')
        x.s.execute=unknown;x.s.pump('host/gpu');await drain(x);x.c.pause()
        with pytest.raises(Conflict):await x.e.binding.suspend_for_restart(observation='fixture-do-not-guess')
        assert x.e.a.execution(x.e.call['id'],'one')['state']=='uncertain'
        assert x.e.m.ollama.unloads==[] and x.e.body['lease']['status']=='active'
    asyncio.run(scenario())


def test_suspension_witness_cannot_cover_a_later_backend_load(controlled):
    from .test_root_models import shutdown_fixture
    async def scenario():
        x=controlled;shutdown_fixture(x.e);x.s.pump('host/gpu');await drain(x);x.c.pause()
        await x.e.binding.suspend_for_restart(observation='fixture-first-absence')
        x.c.resume()
        await x.c.refresh({'memory_available_mb':64000,'memory_device_inuse_mb':0});await drain(x)
        enqueue(x,'later');await drain(x);x.c.pause()
        x.e.m.unload('text',force=True,require_confirmed=True)
        with pytest.raises(Conflict):x.e.m.shutdown()
        assert x.e.m.ollama.loads==['text','text'] and x.e.body['lease']['status']=='active'
        await x.e.binding.suspend_for_restart(observation='fixture-later-fresh-absence')
        assert x.e.m.shutdown()==[]
    asyncio.run(scenario())
