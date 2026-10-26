"""Actual controller/supervisor/manager; deterministic provider and registry views."""
import asyncio
from types import SimpleNamespace

import pytest

from .test_root_models import env  # noqa: F401
from .inference_root_controller import RootController
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
