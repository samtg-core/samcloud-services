"""Connect service-owned jobs to the existing chat/completion implementations.

Explicit installation only; production bootstrap is not wired yet. Backend
mapping, lease continuity and unsupported adapters need qualification before
shared-resource deployment. Admitted Ollama/Whisper calls use service-owned reads;
consumer observation deadlines remain a separate layer, not generation limits.
"""
import asyncio
import codecs
import json
import logging
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from .inference_api import owner, build_router
from .inference_arbiter import Conflict
from .inference_supervisor import BackendEnded, Supervisor
from .ollama_client import OllamaRefused
from .whisper_client import WhisperFailed
from .inference_inputs import InferenceInputs, EmptyInput, InputTooLarge


class StreamCapture:
    """Parse the service's SSE/NDJSON output, including split UTF-8/frames.

    Save deltas as they arrive; keep finish frames provisional until the complete
    backend stream validates. Error text is not relayed or persisted here.
    """
    def __init__(self, journal, execution_id, media_type):
        self.journal = journal
        self.execution_id = execution_id
        self.sse = media_type == 'text/event-stream'
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.pending = ''
        self.saw_terminal = False
        self.provisional = False
        self.count = 0

    def feed(self, chunk):
        self.pending += self.decoder.decode(chunk if isinstance(chunk, bytes) else chunk.encode())
        # Routes emit LF. Normalize CRLF, including a CR/LF split across chunks.
        self.pending = self.pending.replace('\r\n', '\n')
        separator = '\n\n' if self.sse else '\n'
        while separator in self.pending:
            frame, self.pending = self.pending.split(separator, 1)
            self._frame(frame, separator)

    def _frame(self, frame, separator):
        if self.sse:
            payload = '\n'.join(line[5:].lstrip(' ') for line in frame.splitlines()
                                if line.startswith('data:'))
        else:
            payload = frame.strip()
        if not payload:
            return  # transport comments are not model output
        if payload == '[DONE]':
            self.provisional = True
        else:
            value = json.loads(payload)
            if value.get('error'):
                raise RuntimeError('provider stream ended without confirmed backend completion')
            terminal = value.get('done') is True or any(
                choice.get('finish_reason') is not None for choice in value.get('choices', []))
            self.saw_terminal = self.saw_terminal or terminal
            self.provisional = self.provisional or terminal
        self.journal.append_event(self.execution_id, frame + separator, terminal=self.provisional)
        self.count += 1

    def finish(self):
        self.pending += self.decoder.decode(b'', final=True)
        # NDJSON permits its final record without a newline. SSE does not
        # dispatch an unterminated event, so do not turn a cut frame into success.
        if self.pending.strip():
            if self.sse:
                raise RuntimeError('provider stream ended inside an event')
            self._frame(self.pending, '\n')
        if not self.saw_terminal:
            raise RuntimeError('provider stream has no terminal completion')


class GatewayQueue:
    def __init__(self, path, service, models, permitted, *, stint_policy=None, journal_factory=None):
        """models is trusted exact-name -> resource configuration, never a client field."""
        self.service = service
        self.models = dict(models)
        self.held_claims = {}
        self.input_handoff = asyncio.Lock()
        self.inputs = InferenceInputs(Path(path).parent/'inputs',
            service.config.WHISPER_MAX_UPLOAD_MB*1024*1024)
        self.supervisor = Supervisor(path, self.execute, permitted,
            on_finished=self.cleanup_input, stint_policy=stint_policy, journal_factory=journal_factory)

    def resolve(self, model):
        if model not in self.models:
            raise HTTPException(503, 'Model has no configured inference admission resource')
        return self.models[model], model

    def validate_call(self, model, body):
        req = self.service.ChatRequest(**body)
        if req.model != model:
            raise ValueError('request model differs from grant')
        return {'operation': 'chat', 'body': req.model_dump(), 'release_after': False}

    def router(self):
        return build_router(self.supervisor, self.resolve, self.validate_call)

    async def execute(self, grant, call):
        if call.get('release_after'):
            self.supervisor.journal.request_release(grant['id'], grant['owner'])
        if call['operation']=='transcription':
            return await self.execute_transcription(grant,call['body'])
        if call['operation'] in ('model_load', 'model_unload'):
            req_type, handler = (
                (self.service.LoadRequest, self.service._load_model_direct)
                if call['operation']=='model_load' else
                (self.service.UnloadRequest, self.service._unload_model_direct))
            # The ordinary HTTP waiter never owns this mutation. In particular,
            # force-unload waits its turn; it cannot bypass unknown live work.
            response = await handler(req_type(**call['body']), queue_owned=True)
            return {'kind':'json', 'body':response}
        req_type = self.service.ChatRequest if call['operation'] == 'chat' else self.service.CompletionRequest
        req = req_type(**call['body'])
        # Resolve/load only AFTER exclusive admission. A waiting request must
        # not evict/load a model under the current runner before it owns capacity.
        try:
            mm = await self.service._resolve_model(req.model, queue_owned=True)
        except HTTPException:
            # The legacy resolver can translate a timed-out load into an HTTP
            # error while the provider still owns the load. A transport/status
            # response is not evidence of backend termination. Keep admission
            # until reconciliation; do not expose arbitrary exception detail.
            raise RuntimeError('Model resolution requires backend reconciliation') from None
        if mm is None:
            raise RuntimeError('Model resolution requires backend reconciliation')
        if mm.backend not in (self.service.Backend.OLLAMA,self.service.Backend.VLM):
            raise BackendEnded({'kind': 'error', 'status': 503,
                                'detail': 'Backend is not qualified for this queue adapter'})
        claimed = self.service.mgr.claim(mm.name) is not None
        if not claimed:
            raise RuntimeError('Resolved model could not be claimed; reconciliation required')
        self.held_claims[grant['active_execution']] = mm.name
        terminal = False
        try:
            if call['operation'] == 'chat':
                response = await self.service._chat_completions_direct(req, resolved=mm, queue_owned=True)
            else:
                response = await self.service._completions_direct(req, resolved=mm, queue_owned=True)
            if isinstance(response, StreamingResponse):
                capture = StreamCapture(self.supervisor.journal, grant['active_execution'], response.media_type)
                async for frame in response.body_iterator:
                    capture.feed(frame)
                # Existing routes can turn a provider exception into an error
                # frame. A closed HTTP stream, even one ending in [DONE], is
                # not affirmative evidence that backend compute terminated.
                capture.finish()
                result = {'kind': 'stream', 'event_count': capture.count, 'media_type': response.media_type}
            else:
                if mm.backend==self.service.Backend.VLM:
                    choices=response.get('choices') if isinstance(response,dict) else None
                    if not isinstance(choices,list) or not choices or any(
                            not isinstance(choice,dict) or choice.get('finish_reason') is None
                            for choice in choices):
                        raise RuntimeError('Vision backend completion is unconfirmed')
                result = {'kind': 'json', 'body': response}
            terminal = True
            return result
        except OllamaRefused as error:
            # The client read a complete pre-stream bad-request/not-found
            # response. Nothing here treats arbitrary HTTP errors as proof.
            terminal = True
            raise BackendEnded(error.result) from None
        finally:
            if terminal and claimed:
                self.service.mgr.release(self.held_claims.pop(grant['active_execution']))
            # On ambiguity retain BOTH queue ownership and the residency claim.
            # A socket failure cannot make cooldown unload unknown live work.

    def enqueue_compat(self, operation, req, request):
        identity = owner(request)
        resource, model = self.resolve(req.model)
        key = self.request_key(request)
        try:
            grant = self.supervisor.journal.submit(identity, operation + ':' + key, resource, model)
            execution = self.supervisor.journal.enqueue(grant['id'], identity, 'call', {
                'operation': operation, 'body': req.model_dump(), 'release_after': True})
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError:
            raise HTTPException(422, 'Invalid idempotency key or request') from None
        self.supervisor.pump(resource)
        return execution, identity

    @staticmethod
    def request_key(request):
        key = request.headers.get('Idempotency-Key')
        if not isinstance(key,str) or not key.strip():
            # Once work outlives an HTTP waiter, assigning a fresh random key
            # on each transport retry silently duplicates compute. Native queue
            # consumers already allocate durable intent keys; do not dedupe text.
            raise HTTPException(428,{'code':'idempotency_key_required',
                                    'message':'Queued work requires a stable Idempotency-Key'})
        return key

    async def control(self, operation, req, request):
        if operation not in ('model_load', 'model_unload'):
            raise HTTPException(422, 'Unsupported model operation')
        execution, identity = self.enqueue_compat(operation, req, request)
        row = await self.supervisor.observe(execution['id'], identity)
        return self._result(row)['body']

    async def transcription(self, upload, model, options, request):
        identity = owner(request)
        key = self.request_key(request)
        match = self.service.match_whisper_model(model)
        if match is None:
            raise HTTPException(400,{'error':'unknown_transcription_model',
                                    'message':'Unknown transcription model'})
        resource, canonical = self.resolve(match[0])
        try:
            staged = await self.inputs.capture(upload)
        except InputTooLarge:
            raise HTTPException(413,{'error':'audio_too_large'}) from None
        except EmptyInput:
            raise HTTPException(400,{'error':'empty_file'}) from None
        handoff = self.supervisor.prepare(
            self._prepare_transcription(identity,key,resource,canonical,options,staged))
        try:
            grant,execution = await asyncio.shield(handoff)
        except Conflict as error:
            raise HTTPException(409,str(error)) from None
        except ValueError:
            raise HTTPException(422,'Invalid transcription request') from None
        row = await self.supervisor.observe(execution['id'],identity)
        # Reclaim accepted inputs only from saved terminal state, never from
        # a request's cancellation/finally. Replays do not reinstall uploads.
        try:
            await self.cleanup_input(grant,execution['id'])
        except Exception:
            logging.getLogger(__name__).warning('Saved execution input cleanup incomplete')
        result = self._result(row)
        if result['kind']=='http':
            return Response(result['body'],status_code=result['status'],media_type=result['media_type'])
        return result['body']

    async def _prepare_transcription(self, identity, key, resource, canonical, options, staged):
        install = None
        async with self.input_handoff:
            try:
                grant = self.supervisor.journal.submit(identity,'transcription:'+key,resource,canonical,
                                                       lifetime='execution')
                call = {'operation':'transcription','body':{**options,'input':staged.descriptor()},
                        'release_after':True}
                execution = self.supervisor.journal.existing_execution(grant['id'],identity,'call',call)
                if execution is None:
                    install = asyncio.create_task(asyncio.to_thread(self.inputs.install,grant['id'],staged))
                    await asyncio.shield(install)
                    execution = self.supervisor.journal.enqueue(grant['id'],identity,'call',call)
                # A saved retry never reinstalls its bytes, including saved failures.
                self.supervisor.pump(resource)
                return grant,execution
            finally:
                # The HTTP waiter cannot cancel this handoff. If shutdown itself
                # interrupts an install, its thread may still be using the file:
                # retain it rather than racing deletion underneath that worker.
                if install is None or install.done():
                    try:
                        await asyncio.to_thread(self.inputs.discard_staged,staged)
                    except Exception:
                        logging.getLogger(__name__).warning('Unaccepted input cleanup incomplete')

    async def execute_transcription(self, grant, body):
        path = await asyncio.to_thread(self.inputs.path,grant['id'],body['input'])
        async with self.service._whisper_gate:
            mm = await self.service._resolve_whisper(grant['model'])
            if self.service.mgr.claim(mm.name) is None:
                raise RuntimeError('Resolved transcriber requires claim reconciliation')
            self.held_claims[grant['active_execution']] = mm.name
        terminal = False
        try:
            result = await self.service.mgr.whisper.transcribe(path=str(path),
                language=body['language'],prompt=body['prompt'],temperature=body['temperature'],
                word_timestamps='word' in body['granularities'],service_owned=True)
            terminal = True
            try:
                response = self.service._format_transcription_result(mm,result,body['format'],
                                                                     set(body['granularities']))
                if isinstance(response,Response):
                    return {'kind':'http','body':response.body.decode(response.charset),
                            'status':response.status_code,'media_type':response.media_type}
                return {'kind':'json','body':response}
            except Exception:
                raise BackendEnded({'kind':'error','status':502,'code':'transcription_result_invalid',
                                    'detail':'Invalid transcription result'}) from None
        except WhisperFailed as error:
            if error.status in (400,404,422):
                terminal = True
                raise BackendEnded({'kind':'error','status':400,'code':'undecodable_audio',
                                    'detail':'Audio could not be decoded'}) from None
            raise
        finally:
            if terminal:
                self.service.mgr.release(self.held_claims.pop(grant['active_execution']))
            # Unknown retains both input and residency claim. HTTP observation
            # timeout/disconnect is not a backend-completion signal.

    async def cleanup_input(self, grant, execution_id):
        async with self.input_handoff:
            await self._cleanup_input(grant,execution_id)

    async def _cleanup_input(self, grant, execution_id):
        row = self.supervisor.journal.execution(execution_id,grant['owner'])
        call = json.loads(row['request'])
        if row['state'] not in ('complete','failed') or call.get('operation')!='transcription':
            return
        try:
            proof = await asyncio.to_thread(self.inputs.proof,grant['id'],call['body']['input'])
        except FileNotFoundError:
            return  # already retired by another saved-result observer
        self.inputs.retire(self.supervisor.journal,execution_id,grant['owner'],proof=proof)

    async def compat(self, operation, req, request):
        execution, identity = self.enqueue_compat(operation, req, request)
        if not req.stream:
            row = await self.supervisor.observe(execution['id'], identity)
            result = self._result(row)
            return result['body']

        # A compatibility stream may keep a socket alive while queued, without
        # inventing model tokens. Durable clients use the explicit job API.
        media_type = 'text/event-stream' if operation == 'chat' else 'application/x-ndjson'
        async def stream():
            cursor = 0
            heartbeat_at = 0
            while True:
                page = self.supervisor.journal.events(execution['id'], identity, after=cursor)
                for event in page['events']:
                    yield event['data']
                cursor = page['next_cursor']
                if page['has_more']:
                    continue
                if page['state'] not in ('queued', 'running', 'uncertain'):
                    break
                now = asyncio.get_running_loop().time()
                if now >= heartbeat_at:
                    yield ': model-service ' + page['state'] + '\n\n' if operation == 'chat' else '\n'
                    heartbeat_at = now + 10
                await asyncio.sleep(.1)
            try:
                row = self.supervisor.journal.execution(execution['id'], identity)
                self._result(row)
            except HTTPException:
                detail = {'error': {'type': 'inference_' + row['state'],
                                    'message': 'Read the saved execution state',
                                    'execution_id': execution['id']}}
                yield ('data: ' if operation == 'chat' else '') + json.dumps(detail) + '\n\n'
                if operation == 'chat':
                    yield 'data: [DONE]\n\n'
        return StreamingResponse(stream(), media_type=media_type,
                                 headers={'X-Inference-Execution': execution['id']})

    @staticmethod
    def _result(row):
        if row['state'] != 'complete':
            detail = {'state':row['state'],'execution_id':row['id']}
            if row['state']=='failed' and row['result'] is not None:
                result = json.loads(row['result'])
                if (isinstance(result,dict) and result.get('kind')=='error' and
                        type(result.get('status')) is int and 400<=result['status']<600):
                    detail.update(code=result.get('code'),message=result.get('detail'))
                    raise HTTPException(result['status'],detail)
            raise HTTPException(409,detail)
        return json.loads(row['result'])
