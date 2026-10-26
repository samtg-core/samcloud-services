"""Real supervisor and root scheduler; deterministic trusted observations only."""
import asyncio

import pytest

from .inference_arbiter import Arbiter
from .inference_root_scheduler import RootPreparingArbiter
from .inference_supervisor import Supervisor
from .test_root_scheduler import admitted,terminal,call


def factory(path,*,stint_policy=None):
    return RootPreparingArbiter(path,registry='https://registry.test/api/v1',
        registry_actor='provider',service_id='models',stint_policy=stint_policy)


def test_default_supervisor_still_uses_legacy_journal(tmp_path):
    s=Supervisor(tmp_path/'jobs.sqlite',None,lambda _:False)
    try:
        assert type(s.journal) is Arbiter
        assert s.on_resource_demand is None
    finally:s.close()


def test_demand_precedes_capacity_gate_and_failed_observation_never_dispatches(tmp_path):
    s=Supervisor(tmp_path/'jobs.sqlite',None,lambda _:False,journal_factory=factory)
    try:
        _,first=call(s.journal);events=[]
        s.on_resource_demand=events.append
        assert s.pump('host/gpu') is None and events==['host/gpu']
        assert s.capacity_available('host/gpu') is None and events==['host/gpu','host/gpu']
        def failed(_):raise RuntimeError('private observation detail')
        s.on_resource_demand=failed
        assert s.pump('host/gpu') is None
        assert s.journal.execution(first['id'],'one')['state']=='queued'
        assert not s.workers
    finally:s.close()


def test_root_end_phase_wakes_while_parent_still_owns_and_http_observer_can_leave(tmp_path):
    async def scenario():
        started=asyncio.Event();run=asyncio.Event();cleanup=asyncio.Event();finish_cleanup=asyncio.Event()
        async def execute(*_):started.set();await run.wait();return {'answer':'saved'}
        s=Supervisor(tmp_path/'jobs.sqlite',execute,lambda _:True,journal_factory=factory)
        s.permitted_model=lambda *_:True
        tasks=[];events=[]
        try:
            _,first=call(s.journal);prep=s.journal.prepare_root('host/gpu')
            body,view=admitted(prep);s.journal.confirm_root(prep['id'],body,view)
            async def reconcile():
                cleanup.set();await finish_cleanup.wait()
                s.journal.settle_preparation(prep['id'],terminal(body))
            def demand(resource):
                events.append(s.journal.preparation(prep['id'])['state'])
                if events[-1]=='ending' and not tasks:
                    tasks.append(s.prepare(reconcile()))
            s.on_resource_demand=demand
            assert s.pump('host/gpu')==first['id'];await started.wait()
            # Owner finish seals intake, but accepted work is not cut short.
            rows=s.end_job('one','parent')
            assert rows[0]['state']=='owned' and s.journal.preparation(prep['id'])['state']=='admitted'
            run.set();await asyncio.gather(*tuple(s.workers.values()))
            await cleanup.wait()
            assert 'ending' in events and s.journal.owns_job('host/gpu')
            assert s.journal.execution(first['id'],'one')['state']=='complete'
            with pytest.raises(RuntimeError):s.close()
            finish_cleanup.set();await asyncio.gather(*tasks)
            assert not s.journal.owns_job('host/gpu')
        finally:
            run.set();finish_cleanup.set()
            await asyncio.gather(*tuple(s.workers.values()),*tuple(s.preparations),return_exceptions=True)
            s.close()
    asyncio.run(scenario())


def test_explicit_end_between_calls_notifies_owned_root_without_waiting_for_next_call(tmp_path):
    async def scenario():
        async def execute(*_):return {}
        s=Supervisor(tmp_path/'jobs.sqlite',execute,lambda _:True,journal_factory=factory)
        s.permitted_model=lambda *_:True
        try:
            _,first=call(s.journal);prep=s.journal.prepare_root('host/gpu')
            body,view=admitted(prep);s.journal.confirm_root(prep['id'],body,view)
            s.pump('host/gpu');await asyncio.gather(*tuple(s.workers.values()))
            seen=[];s.on_resource_demand=lambda resource:seen.append(s.journal.preparation(prep['id'])['state'])
            s.end_job('one','parent')
            assert seen==['ending']
            assert s.journal.job('one','parent','host/gpu')['state']=='owned'
            assert s.journal.execution(first['id'],'one')['state']=='complete'
        finally:s.close()
    asyncio.run(scenario())


def test_gateway_passes_only_explicit_trusted_factory(tmp_path):
    from types import SimpleNamespace
    from .inference_gateway import GatewayQueue
    service=SimpleNamespace(config=SimpleNamespace(WHISPER_MAX_UPLOAD_MB=1))
    q=GatewayQueue(tmp_path/'jobs.sqlite',service,{'text':'host/gpu'},lambda _:False,journal_factory=factory)
    try:
        assert isinstance(q.supervisor.journal,RootPreparingArbiter)
        assert q.supervisor.journal.registry_actor=='provider'
    finally:q.supervisor.close()
