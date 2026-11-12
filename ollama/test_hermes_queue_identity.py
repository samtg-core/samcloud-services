"""Caller scope contracts; no native runtime or inference side effects."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
import tempfile
import unittest

from .hermes_queue_identity import HeldSource, HostBindings


class HostIdentity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'calls.sqlite'
        self.host = HostBindings(self.path, 'own-profile')
        self.messages = [{'role':'user', '_row_id':1, 'content':'private text'}]
        self.body = {'model':'qwen', 'messages':[{'role':'user','content':'private text'}]}

    def tearDown(self):
        self.temp.cleanup()

    def bind_tool(self,session='session',tool='tool'):
        source=HeldSource.capture('own-profile',session,'main',self.messages,
                                 turn_id='turn',api_request_id='api')
        keys=source.bind_stint(self.path,self.body)
        self.host.complete_api(session,'turn','api',(tool,))
        return keys

    def test_tool_auxiliary_yields_only_own_parent_and_success_advances_intent(self):
        main=self.bind_tool()
        body=self.body|{'model':'vision'}
        with self.host.tool_scope('session','turn','api','tool'):
            with self.host.tool_api_scope():first=self.host.tool_keys(body)
            with self.host.tool_api_scope():second=self.host.tool_keys(body)
        self.assertNotEqual(first.call_key,second.call_key)
        self.assertEqual(first.yield_grant_keys,(main.grant_key,))
        self.assertNotEqual(first.grant_key,main.grant_key)
        with self.assertRaises(ValueError):self.host.tool_keys(body)

    def test_unknown_tool_auxiliary_reattaches_saved_intent_after_host_reopen(self):
        self.bind_tool()
        body=self.body|{'model':'vision'}
        with self.assertRaises(RuntimeError):
            with self.host.tool_scope('session','turn','api','tool'),self.host.tool_api_scope():
                first=self.host.tool_keys(body)
                raise RuntimeError('observer unavailable')
        self.host=HostBindings(self.path,'own-profile')
        with self.host.tool_scope('session','turn','api','tool'),self.host.tool_api_scope():
            self.assertEqual(first,self.host.tool_keys(body))
        with self.host.tool_scope('session','turn','api','tool'),self.host.tool_api_scope():
            self.assertNotEqual(first,self.host.tool_keys(body))

    def test_unknown_auxiliary_changed_input_and_unrelated_tool_refuse(self):
        self.bind_tool()
        with self.assertRaises(RuntimeError):
            with self.host.tool_scope('session','turn','api','tool'),self.host.tool_api_scope():
                self.host.tool_keys(self.body)
                raise RuntimeError('observer unavailable')
        with self.assertRaises(ValueError):
            with self.host.tool_scope('session','turn','api','tool'),self.host.tool_api_scope():
                self.host.tool_keys(self.body|{'messages':[{'role':'user','content':'changed'}]})
        for session,tool in (('other','tool'),('session','unrecorded')):
            with self.subTest(session=session,tool=tool),self.assertRaises(ValueError):
                with self.host.tool_scope(session,'turn','api',tool),self.host.tool_api_scope():pass

    def test_source_is_frozen_and_recreated_keys_survive_store_reopen(self):
        held = HeldSource.capture('own-profile','session','compression',self.messages)
        first = held.bind(self.path,self.body)
        self.messages.append({'role':'user','_row_id':2,'content':'new input'})
        self.assertEqual(held.rows,(1,))
        recovered = HeldSource.capture('own-profile','session','compression',self.messages[:1])
        self.assertEqual(first,recovered.bind(self.path,self.body))
        new = HeldSource.capture('own-profile','session','compression',self.messages)
        self.assertNotEqual(first,new.bind(self.path,self.body))
        other_profile = HeldSource.capture('other-profile','session','compression',self.messages[:1])
        self.assertNotEqual(first,other_profile.bind(self.path,self.body))
        self.assertNotIn(b'private text',self.path.read_bytes())

    def test_missing_row_refuses_and_clears_previous_main_boundary(self):
        self.host.pre_api_request(session_id='session',conversation_history=self.messages)
        self.host.main_keys(self.body)
        with self.assertRaises(ValueError):
            self.host.pre_api_request(session_id='session',conversation_history=[{'role':'user','content':'same'}])
        with self.assertRaises(ValueError):
            self.host.main_keys(self.body)

    def test_auxiliary_requires_own_scope_and_restores_nested_owner(self):
        self.host.pre_api_request(session_id='session',conversation_history=self.messages)
        main = self.host.main_keys(self.body)
        with self.assertRaises(ValueError):
            self.host.auxiliary_keys(self.body)
        with self.host.compression_scope('session',self.messages):
            first = self.host.auxiliary_keys(self.body)
            with self.host.compression_scope('other-session',self.messages):
                self.assertNotEqual(first,self.host.auxiliary_keys(self.body))
            self.assertEqual(first,self.host.auxiliary_keys(self.body))
        with self.assertRaises(ValueError):
            self.host.auxiliary_keys(self.body)
        with self.assertRaises(RuntimeError):
            with self.host.compression_scope('session',self.messages):
                raise RuntimeError('synthetic failed compression')
        with self.assertRaises(ValueError):
            self.host.auxiliary_keys(self.body)
        self.assertEqual(main,self.host.main_keys(self.body))
        self.assertNotEqual(main,first)
        self.host.post_api_request()
        with self.assertRaises(ValueError):
            self.host.main_keys(self.body)

    def test_copied_worker_context_keeps_conversations_separate(self):
        def run(session):
            self.host.pre_api_request(session_id=session,conversation_history=self.messages)
            with ThreadPoolExecutor(max_workers=1) as worker:
                return worker.submit(copy_context().run,self.host.main_keys,self.body).result(timeout=2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run,['one','two','one']))
        self.assertEqual(results[0],results[2])
        self.assertNotEqual(results[0],results[1])

    def test_invalid_ids_refuse_but_unpersisted_system_prompt_is_not_source(self):
        for row in (None,0,-1,True,'1'):
            with self.assertRaises(ValueError):
                HeldSource.capture('profile','session','main',[{'role':'user','_row_id':row}])
        held = HeldSource.capture('profile','session','main',
            [{'role':'system','content':'ephemeral system'},*self.messages])
        self.assertEqual(held.rows,(1,))

    def test_stint_reuses_only_foreground_conversation_model_not_execution_identity(self):
        one = HeldSource.capture('profile', '1777', 'main', self.messages)
        first = one.bind_stint(self.path, self.body)
        later = HeldSource.capture('profile', '1777', 'main', [*self.messages,
            {'role': 'assistant', '_row_id': 2, 'content': 'tool call'},
            {'role': 'tool', '_row_id': 3, 'content': 'tool result'}])
        second = later.bind_stint(self.path, self.body)
        self.assertEqual(first.grant_key, second.grant_key)
        self.assertNotEqual(first.call_key, second.call_key)
        self.assertEqual(first, one.bind_stint(self.path, self.body))
        self.assertNotEqual(first.grant_key, one.bind(self.path, self.body).grant_key)
        other = HeldSource.capture('profile', '1788', 'main', self.messages)
        self.assertNotEqual(first.grant_key, other.bind_stint(self.path, self.body).grant_key)
        self.assertNotEqual(first.grant_key,
            one.bind_stint(self.path, {**self.body, 'model': 'other'}).grant_key)

    def test_stint_binding_refuses_unbound_auxiliary_and_changed_call(self):
        from .inference_arbiter import Conflict
        with self.assertRaises(ValueError):
            self.host.main_stint_keys(self.body)
        self.host.pre_api_request(session_id='session', conversation_history=self.messages)
        self.host.main_stint_keys(self.body)
        with self.assertRaises(Conflict):
            self.host.main_stint_keys({**self.body, 'temperature': .5})
        other = HeldSource.capture('profile', 'session', 'compression', self.messages)
        with self.assertRaises(ValueError):
            other.bind_stint(self.path, self.body)
        with self.assertRaises(ValueError):
            other.bind_stint(self.path, {})

    def test_compaction_yields_only_owning_conversation_stints_not_siblings(self):
        self.host.pre_api_request(session_id='session', conversation_history=self.messages)
        main = self.host.main_stint_keys(self.body)
        self.host.pre_api_request(session_id='sibling', conversation_history=self.messages)
        sibling = self.host.main_stint_keys(self.body)
        with self.host.compression_scope('session', self.messages):
            aux = self.host.auxiliary_keys(self.body)
            self.assertEqual(aux.yield_grant_keys, (main.grant_key,))
            self.assertNotIn(sibling.grant_key, aux.yield_grant_keys)
            self.assertNotEqual(aux.grant_key, main.grant_key)
        with self.host.compression_scope('never-submitted', self.messages):
            self.assertEqual(self.host.auxiliary_keys(self.body).yield_grant_keys, ())

    def test_unpersisted_compaction_source_keeps_exact_parent_scope_after_reopen(self):
        self.host.pre_api_request(session_id='session', conversation_history=self.messages)
        main = self.host.main_stint_keys(self.body)
        with self.host.compression_scope('session', [{'role': 'user', 'content': 'fresh'}],
                                         persist_unstored=True) as source:
            first = self.host.auxiliary_keys(self.body)
            self.assertEqual(first.yield_grant_keys, (main.grant_key,))
        with self.host.resume_compression('session', source.operation) as messages:
            with self.host.compression_scope('session', messages):
                self.assertEqual(self.host.auxiliary_keys(self.body), first)

    def test_summary_retry_does_not_gain_a_later_created_parent(self):
        with self.host.compression_scope('session', self.messages):
            initial = self.host.auxiliary_keys(self.body)
        self.assertEqual(initial.yield_grant_keys, ())
        self.host.pre_api_request(session_id='session', conversation_history=self.messages)
        self.host.main_stint_keys(self.body)
        with self.host.compression_scope('session', self.messages):
            self.assertEqual(self.host.auxiliary_keys(self.body), initial)


class DetachedIdentity(unittest.TestCase):
    def test_foreground_persistence_failure_cannot_masquerade_as_a_fork(self):
        from .hermes_queue_identity import HostBindings
        with tempfile.TemporaryDirectory() as folder:
            host=HostBindings(Path(folder)/'identity.sqlite','profile')
            host.begin_foreground('session','main-turn')
            with self.assertRaisesRegex(ValueError,'foreground or unowned'):
                host.detached_source('session','main-turn',[{'role':'user','content':'unstored'}])

    def test_detached_fork_has_own_durable_execution_not_foreground_affinity(self):
        from .hermes_queue_identity import HostBindings
        with tempfile.TemporaryDirectory() as folder:
            host=HostBindings(Path(folder)/'identity.sqlite','profile')
            request={'model':'qwen','messages':[{'role':'user','content':'synthetic'}]}
            host.pre_api_request(session_id='session',turn_id='main-turn',api_request_id='api',
                                 conversation_history=[{'role':'user','content':'synthetic','_row_id':1}])
            foreground=host.main_stint_keys(request)
            source=host.detached_source('session','fork-turn',[{'role':'user','content':'unstored review'}])
            host.main.set(source)
            first=host.main_stint_keys(request)
            self.assertTrue(first.execution_only)
            self.assertNotEqual(first.grant_key,foreground.grant_key)
            self.assertEqual(host.main_stint_keys(request),first)
            self.assertEqual(source.purpose,'detached_fork')
            self.assertEqual((source.directory/(source.operation+'.json')).stat().st_mode&0o777,0o600)
            with self.assertRaisesRegex(ValueError,'unowned fork'):
                host.detached_source('other-session','fork-turn',[{'role':'user','content':'unowned'}])


if __name__ == '__main__':
    unittest.main()
