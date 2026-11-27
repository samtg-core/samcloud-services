"""Opt-in EXO execution through the existing service-owned supervisor.

No physical lease or inference replay. A terminal wire frame is provisional
until the private runtime proves every assigned rank drained this exact command.
"""

import json
import logging

from .staging import HostFileGatewayUnavailable
from ..inference_arbiter import CapacityDeferred
from ..inference_gateway import StreamCapture
from ..inference_supervisor import BackendEnded


def _json(raw):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('duplicate private response field')
            result[key] = item
        return result

    def nonfinite(_):
        raise ValueError('nonfinite private response field')

    result = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(result, dict):
        raise ValueError('private response is not an object')
    return result


# Exact diagnostic emitted by the pinned native served-context guard before prefill.
_NATIVE_CONTEXT_ERROR = 'request token budget exceeds the configured served context'


def _context_rejected(error):
    return isinstance(error, dict) and error.get('message') == _NATIVE_CONTEXT_ERROR


def _choice(value, command, model, *, streamed):
    if value.get('id') != command or value.get('model') != model:
        raise ValueError('private response changed command or model')
    choices = value.get('choices')
    if (not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or type(choices[0].get('index')) is not int
            or choices[0]['index'] != 0):
        raise ValueError('private response has unexpected choices')
    choice = choices[0]
    finish = choice.get('finish_reason')
    if finish not in (None, 'stop', 'length', 'tool_calls', 'function_call', 'content_filter', 'error'):
        raise ValueError('unknown private finish reason')
    if (not isinstance(choice.get('delta' if streamed else 'message'), dict)
            or (not streamed and finish is None)):
        raise ValueError('private response lacks its message or terminal choice')
    return finish


class EXOCapture(StreamCapture):
    def __init__(self, journal, execution_id, command, model):
        super().__init__(journal, execution_id, 'text/event-stream')
        self.command, self.model = command, model
        self.done = self.failed = self.context_rejected = False

    def feed(self, chunk):
        super().feed(chunk)
        if len(self.pending) > 1024 * 1024:
            raise ValueError('private event exceeds bound')

    def _frame(self, frame, separator):
        if len(frame) > 1024 * 1024:
            raise ValueError('private event exceeds bound')
        payload = '\n'.join(line[5:].lstrip(' ') for line in frame.splitlines()
                            if line.startswith('data:'))
        if not payload:
            return
        if self.done:
            raise ValueError('private output follows DONE')
        if payload == '[DONE]':
            if not self.saw_terminal:
                raise ValueError('private DONE lacks a terminal result')
            self.done = True
        else:
            value = _json(payload)
            if self.saw_terminal:
                raise ValueError('private output follows terminal result')
            if isinstance(value.get('error'), dict):
                self.context_rejected = _context_rejected(value['error'])
                # Never expose/cache arbitrary backend diagnostics as output.
                self.failed = self.saw_terminal = self.provisional = True
                return
            finish = _choice(value, self.command, self.model, streamed=True)
            if finish == 'error':
                self.failed = self.saw_terminal = self.provisional = True
                return
        super()._frame(frame, separator)

    def finish(self):
        super().finish()
        if not self.done:
            raise ValueError('private stream has no complete DONE event')


class EXOBackend:
    def __init__(self, actor, private, bindings, contract, launches, verify_runtime):
        if actor.intents is not bindings.intents or actor.plan is None or not callable(verify_runtime):
            raise ValueError('exact acquisition actor, journal and runtime verifier required')
        self.actor, self.private, self.bindings = actor, private, bindings
        self.contract, self.launches = contract, launches
        self.verify_runtime = verify_runtime

    async def execute(self, grant, call):
        execution_id = grant['active_execution']
        if call.get('operation') != 'chat' or not isinstance(call.get('body'), dict):
            raise BackendEnded({'kind': 'error', 'status': 400, 'code': 'exo_request_unqualified',
                                'detail': 'EXO qualification accepts text chat only'})
        body = call['body']
        messages = body.get('messages')
        if (not isinstance(messages, list) or not messages or any(
                not isinstance(message, dict) or message.get('content') is not None
                and not isinstance(message.get('content'), str) for message in messages)):
            raise BackendEnded({'kind': 'error', 'status': 400, 'code': 'exo_request_unqualified',
                                'detail': 'EXO qualification accepts text chat only'})
        self.actor.wake()
        await self.actor.wait()
        if not self.actor.may_serve_requests():
            raise CapacityDeferred('capacity_unavailable')
        try:
            runtime = await self.verify_runtime()
        except HostFileGatewayUnavailable:
            # No binding or private request exists yet. The supervisor can
            # retain this execution until a fresh capacity event arrives.
            raise CapacityDeferred('capacity_unavailable') from None
        state = await self.private.state()
        self.contract.check_idle(self.actor.plan, self.launches, state, runtime_sha256=runtime)
        if not self.actor.may_serve_requests():
            raise CapacityDeferred('capacity_unavailable')
        interrupted = False
        async with self.private.chat(self.bindings, execution_id, self.contract, body,
                                    state_event_idx=state.get('lastEventAppliedIdx')) as response:
            if response.status_code != 200:
                raise RuntimeError('private inference outcome requires reconciliation')
            command = response.headers.get('X-EXO-Command-ID')
            self.bindings.observe_command(execution_id, command)
            media = response.headers.get('content-type', '').split(';')[0]
            if body.get('stream') is True:
                if media != 'text/event-stream':
                    raise RuntimeError('private inference stream is unconfirmed')
                capture = EXOCapture(self.bindings.journal, execution_id, command, self.actor.plan.model_id)
                async for chunk in response.aiter_bytes():
                    if not self.actor.may_serve_requests():
                        report = self.actor.report
                        authority = self.actor.authority
                        observation = authority.current if authority is not None else None
                        logging.getLogger(__name__).warning(
                            'EXO stream authority withdrawn execution=%s command=%s '
                            'acquisition=%s reason=%s saved_events=%d member_decisions=%s',
                            execution_id, command, report.phase, report.reason,
                            capture.count,
                            [(member.member.rank, decision.state, decision.reason)
                             for member, decision in zip(observation.expected, observation.decisions)]
                            if observation is not None else [],
                        )
                        if (chunk != b': keep-alive\n\n' or capture.count or capture.pending
                                or capture.decoder.getstate()[0] or capture.saw_terminal):
                            raise RuntimeError('private inference authority changed after partial output')
                        interrupted = True
                        break
                    for offset in range(0, len(chunk), 65536):
                        capture.feed(chunk[offset:offset + 65536])
                if not interrupted:
                    capture.finish()
                failed = capture.failed
                context_rejected = capture.context_rejected
                result = {'kind': 'stream', 'event_count': capture.count, 'media_type': media}
            else:
                if media != 'application/json':
                    raise RuntimeError('private inference JSON is unconfirmed')
                raw = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    if len(raw) + len(chunk) > 1024 * 1024:
                        raise ValueError('private response exceeds bound')
                    raw.extend(chunk)
                value = _json(raw)
                context_rejected = _context_rejected(value.get('error'))
                failed = isinstance(value.get('error'), dict)
                if not failed:
                    failed = _choice(value, command, self.actor.plan.model_id, streamed=False) == 'error'
                result = {'kind': 'json', 'body': value}
        # The runtime endpoint waits on its existing indexed events. HTTP EOF,
        # TaskDeleted and a single rank's Complete event cannot replace it.
        proof = await self.private.command_end(command)
        runtime = await self.verify_runtime()
        final = await self.private.state()
        self.contract.check_idle(self.actor.plan, self.launches, final, runtime_sha256=runtime)
        if (type(final.get('lastEventAppliedIdx')) is not int
                or type(proof.get('end_event_idx')) is not int
                or final['lastEventAppliedIdx'] < proof['end_event_idx']):
            raise RuntimeError('private state precedes its end evidence')
        self.bindings.record_end(execution_id, proof)
        if interrupted:
            raise BackendEnded({'kind': 'error', 'status': 502, 'code': 'exo_authority_interrupted',
                                'detail': 'Serving authority interrupted the original command before output'})
        # The exact command has ended on every rank, its terminal wire result
        # is valid, and the bound instance is idle. Admission for a *new* call
        # is independent: a later authority invalidation must not relabel this
        # completed result as a backend failure. Provider permission still
        # blocks new dispatch; physical ownership still needs exact cleanup.
        if context_rejected:
            raise BackendEnded({'kind': 'error', 'status': 400, 'code': 'context_length_exceeded',
                                'detail': 'Requested prompt and output exceed the served context length'})
        if failed:
            raise BackendEnded({'kind': 'error', 'status': 502, 'code': 'exo_backend_failed',
                                'detail': 'EXO generation ended without a usable result'})
        return result
