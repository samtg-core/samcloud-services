"""Opt-in native provider component; no live installation or credential copying.

The filesystem entry point registers execution, stint and auxiliary policies. Host hooks
and the context-engine component must bind this same module's profile-local
HostBindings before inference. Loading a provider alone never authorizes a call.
"""
from functools import lru_cache
from pathlib import Path
import threading

from .hermes_queue_identity import HeldSource, HostBindings, UnpersistedSource
from .inference_sdk import QueuedClient
from .inference_identity import LocalSourceRefused

MAIN_PROFILES = frozenset(('samcloud-queue', 'samcloud-queue-stint'))


@lru_cache(maxsize=None)
def _host_for_home(home):
    home = Path(home)
    # A resolved owning profile, not a fallback search across other homes.
    state = home / 'samcloud-queue'
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    return HostBindings(state / 'identity.sqlite', str(home))


def host_for_home(home):
    return _host_for_home(str(Path(home).resolve()))


def register_hooks(ctx):
    # Observer hooks run in copied worker contexts. Store only an explicitly
    # correlated immutable snapshot; execution middleware binds it on its own
    # caller context. Never set a ContextVar in the observer and expect it to
    # propagate back, or use a profile-wide "last request" slot.
    pending, lock = {}, threading.Lock()

    def current_host():
        from hermes_constants import get_hermes_home
        return host_for_home(get_hermes_home())

    def key(host, provider, session_id, api_request_id):
        if not session_id or not api_request_id:
            raise ValueError('native queue handoff requires session and request IDs')
        return host.scope, provider, session_id, api_request_id

    def before(*, provider='', session_id='', api_request_id='', turn_id='', conversation_history=None, **kwargs):
        if provider not in MAIN_PROFILES:
            return
        host = current_host()
        call = key(host, provider, session_id, api_request_id)
        with lock:
            pending.pop(call, None)
        try:
            source = HeldSource.capture(host.scope, session_id, 'main', conversation_history,
                                        turn_id=turn_id, api_request_id=api_request_id)
        except UnpersistedSource:
            source = host.detached_source(session_id,turn_id,conversation_history)
        with lock:
            pending[call] = source

    def after(*, provider='', session_id='', api_request_id='', **kwargs):
        if provider in MAIN_PROFILES and session_id and api_request_id:
            with lock:
                pending.pop(key(current_host(), provider, session_id, api_request_id), None)

    def execute(*, request, next_call, provider='', session_id='', api_request_id='', **kwargs):
        if provider not in MAIN_PROFILES:
            return next_call(request)
        host = current_host()
        # Even when the middleware host skips a raising callback, the queued
        # client's key provider remains unbound and refuses before any I/O.
        token = host.main.set(None)
        try:
            with lock:
                source = pending.pop(key(host, provider, session_id, api_request_id), None)
            host.main.set(source)
            return next_call(request)
        finally:
            host.main.reset(token)

    def began(*,session_id='',turn_id='',**kwargs):
        if session_id and turn_id:
            current_host().begin_foreground(session_id,turn_id)
    ctx.register_hook('pre_llm_call', began)
    ctx.register_hook('pre_api_request', before)
    def completed(*, provider='', session_id='', turn_id='', api_request_id='', response=None, **kwargs):
        if provider in MAIN_PROFILES and session_id and turn_id and api_request_id:
            message = response.get('assistant_message') if isinstance(response,dict) else None
            tools = message.get('tool_calls') if isinstance(message,dict) else None
            if isinstance(tools,list):
                ids = tuple(tool['id'] for tool in tools if isinstance(tool,dict)
                            and isinstance(tool.get('id'),str) and tool['id'].strip())
                current_host().complete_api(session_id,turn_id,api_request_id,ids)
        after(provider=provider,session_id=session_id,api_request_id=api_request_id)
    ctx.register_hook('post_api_request', completed)
    ctx.register_hook('api_request_error', after)
    def finished(*, session_id='', turn_id='', **kwargs):
        if session_id and turn_id:
            current_host().finish_turn(session_id, turn_id)
    ctx.register_hook('post_llm_call', finished)
    def waiting(*, tool_name='', args=None, session_id='', turn_id='', api_request_id='', tool_call_id='', **kwargs):
        # Explicit annotation, not a duration/command-name heuristic. The native
        # modifier strips only this prefix; normal approval/tool dispatch still
        # handles the unchanged command tail. No network in the callback.
        prefix = 'sc-queue-wait -- '
        command = args.get('command') if isinstance(args,dict) else None
        if tool_name != 'terminal' or not isinstance(command,str) or not command.startswith(prefix):
            return
        tail = command[len(prefix):]
        if not tail.strip():
            return {'action':'block','message':'Queue wait annotation requires a command; nothing started.'}
        try:
            count = current_host().dependency_wait(session_id,turn_id,api_request_id,tool_call_id)
        except Exception:
            count = 0
        if not count:
            return {'action':'block','message':'Queue wait has no exact native API/tool identity; nothing started.'}
        return {'action':'modify','args':{'command':tail}}
    ctx.register_hook('pre_tool_call', waiting)
    ctx.register_middleware('llm_execution', execute)
    def tool_execute(*,tool_name='',args=None,next_call,session_id='',turn_id='',
                     api_request_id='',tool_call_id='',**kwargs):
        with current_host().tool_scope(session_id,turn_id,api_request_id,tool_call_id):
            return next_call(args)
    ctx.register_middleware('tool_execution',tool_execute)


def _classify_local_source_error(error, **kwargs):
    # The OpenAI SDK wraps exceptions raised by its local transport in
    # APIConnectionError. Only our typed, pre-submission refusal is a
    # local terminal failure; never classify by prose or provider name.
    seen = set()
    cause = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, LocalSourceRefused):
            return {'reason': 'format_error', 'retryable': False,
                    'should_fallback': False, 'should_compress': False,
                    'should_rotate_credential': False,
                    'error_context': {'origin': 'local_queue_adapter',
                        'code': 'queue_source_unbound',
                        'detail': 'No exact owning durable source is bound; no new inference was submitted.'}}
        cause = cause.__cause__ or cause.__context__
    return None


def register_profiles():
    from providers import register_provider
    from providers.base import ProviderProfile

    class QueueProfile(ProviderProfile):
        def create_client(self, **kwargs):
            from hermes_constants import get_hermes_home
            host = host_for_home(get_hermes_home())
            auxiliary = self.name == 'samcloud-queue-aux'
            tool = self.name == 'samcloud-queue-tool'
            stint = self.name == 'samcloud-queue-stint'
            return QueuedClient(**kwargs,
                key_provider=(host.main_stint_keys if stint else
                              host.tool_keys if tool else host.auxiliary_keys if auxiliary else host.main_keys),
                state_frames=not (auxiliary or tool), complete_responses=auxiliary or tool,
                grant_lifetime='work_session' if stint else 'execution',
                require_stint_policy=stint, event_source=host if (stint or auxiliary) else None,
                call_scope=host.tool_api_scope if tool else None)

    for name in ('samcloud-queue', 'samcloud-queue-stint', 'samcloud-queue-aux','samcloud-queue-tool'):
        register_provider(QueueProfile(name=name, auth_type='api_key',
            classify_api_error=_classify_local_source_error,
            env_vars=('SAMCLOUD_QUEUE_API_KEY', 'SAMCLOUD_QUEUE_BASE_URL'),
            base_url='https://models-cs.samtg.xyz/v1',
            default_aux_model='qwen3.8:27b-mlx'))
