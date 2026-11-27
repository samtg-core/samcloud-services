"""Real audio route/admission; synthetic data and local socket timeout control.

No real model, registry or live service.
"""
import asyncio
import io
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, HTTPException, UploadFile
import httpx
from starlette.requests import Request

from . import server
from . import test_gateway_queue as fixture
from .manager import Backend
from .whisper_client import WhisperFailed, WhisperUnavailable
from .whisper_client import WhisperClient


class GatewayAudio(unittest.IsolatedAsyncioTestCase):
    request = fixture.GatewayLifetime.request
    call = fixture.GatewayLifetime.call
    wait_for = fixture.GatewayLifetime.wait_for

    async def asyncSetUp(self):
        await fixture.GatewayLifetime.asyncSetUp(self)
        self.queue.models[server.WHISPER_DEFAULT]='gpu'
        self.audio_gate = asyncio.Event()
        self.audio_calls, self.loads = [], []
        self.audio_error = None
        self.audio_model = SimpleNamespace(name=server.WHISPER_DEFAULT,backend=Backend.WHISPER,
            in_flight=0,last_used=0,request_count=0)
        def load(name):
            self.loads.append(name)
            self.manager.models[name]=self.audio_model
            return self.audio_model
        self.audio_patch = patch.object(self.manager,'load_whisper_model',load)
        self.audio_patch.start()
        async def transcribe(**kwargs):
            self.audio_calls.append(kwargs)
            self.assertEqual(Path(kwargs['path']).read_bytes(),b'synthetic audio')
            self.assertTrue(kwargs['service_owned'])
            self.assertEqual(self.audio_model.in_flight,1)
            await self.audio_gate.wait()
            if self.audio_error:
                raise self.audio_error
            return {'text':' hello ','language':'en','duration':1,'transcribe_seconds':.01,
                'segments':[{'id':0,'start':0.,'end':1.,'text':' hello',
                             'words':[{'word':'hello','start':0.,'end':1.}]}]}
        self.manager.whisper = SimpleNamespace(transcribe=transcribe)

    async def asyncTearDown(self):
        self.audio_gate.set()
        await fixture.GatewayLifetime.asyncTearDown(self)
        self.audio_patch.stop()

    async def audio(self, *, key='audio', data=b'synthetic audio', fmt='json',
                    model='whisper-1', granularities=None):
        upload = UploadFile(io.BytesIO(data),filename='../../ignored.wav')
        try:
            return await server.create_transcription(file=upload,model=model,language=None,prompt=None,
                response_format=fmt,temperature=None,timestamp_granularities=granularities,
                http_request=self.request('speaker',key))
        finally:
            await upload.close()

    async def test_audio_and_chat_share_fifo_and_canceled_observer_keeps_input_claim(self):
        jobs = []
        try:
            jobs.append(asyncio.create_task(self.call()))
            await self.wait_for(lambda: self.backend.entered==['creative'])
            jobs.append(asyncio.create_task(self.audio()))
            await self.wait_for(lambda: self.queue.supervisor.journal.db.execute(
                'SELECT COUNT(*) FROM grants').fetchone()[0]==2)
            self.assertEqual(self.loads,[])
            jobs.append(asyncio.create_task(self.call('assistant')))
            await self.wait_for(lambda: self.queue.supervisor.journal.db.execute(
                'SELECT COUNT(*) FROM grants').fetchone()[0]==3)
            self.backend.gate.set()
            await self.wait_for(lambda: len(self.audio_calls)==1)
            jobs[1].cancel()
            with self.assertRaises(asyncio.CancelledError):
                await jobs[1]
            path = Path(self.audio_calls[0]['path'])
            self.assertTrue(path.exists())
            self.assertEqual(self.audio_model.in_flight,1)
            self.assertEqual(self.backend.entered,['creative'])
            self.assertEqual((await server.health())['status'],'ok')
            self.audio_gate.set()
            await asyncio.wait_for(asyncio.gather(jobs[0],jobs[2]),2)
            self.assertEqual(self.backend.entered,['creative','assistant'])
            replay = await self.audio()
            self.assertEqual(replay,{'text':'hello'})
            self.assertEqual(len(self.audio_calls),1)
            self.assertEqual(self.loads,[server.WHISPER_DEFAULT])
            self.assertEqual(self.audio_model.in_flight,0)
            self.assertGreater(self.audio_model.last_used,0)
            await self.wait_for(lambda: not path.exists())
        finally:
            self.backend.gate.set()
            self.audio_gate.set()
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs,return_exceptions=True)

    async def test_all_five_audio_formats_reopen_without_retranscription(self):
        self.audio_gate.set()
        for fmt in ('json','text','srt','vtt','verbose_json'):
            with self.subTest(fmt=fmt):
                kwargs={'key':fmt,'fmt':fmt,'granularities':['word'] if fmt=='verbose_json' else None}
                first,second = await self.audio(**kwargs),await self.audio(**kwargs)
                if isinstance(first,dict):
                    self.assertEqual(first,second)
                    self.assertIn('text',first)
                else:
                    self.assertEqual(first.body,second.body)
                    self.assertEqual(first.media_type,second.media_type)
                if fmt=='verbose_json':
                    self.assertTrue(first['words'])
        self.assertEqual(len(self.audio_calls),5)
        self.assertFalse(list(self.queue.inputs.root.rglob('*.audio')))

    async def test_unknown_transport_keeps_input_claim_and_blocks_following_chat(self):
        self.audio_error = WhisperUnavailable('PRIVATE transport detail')
        self.audio_gate.set()
        with self.assertRaises(HTTPException) as caught:
            await self.audio()
        self.assertEqual(caught.exception.status_code,409)
        self.assertEqual(caught.exception.detail['state'],'uncertain')
        self.assertNotIn('PRIVATE',str(caught.exception.detail))
        self.assertTrue(Path(self.audio_calls[0]['path']).is_file())
        self.assertEqual(self.audio_model.in_flight,1)
        pending = asyncio.create_task(self.call())
        try:
            await asyncio.sleep(.03)
            self.assertFalse(pending.done())
            self.assertEqual(self.backend.entered,[])
        finally:
            pending.cancel()
            await asyncio.gather(pending,return_exceptions=True)

    async def test_confirmed_decode_refusal_releases_input_and_reopens_typed_failure(self):
        self.audio_error = WhisperFailed('PRIVATE child text',422)
        self.audio_gate.set()
        for replay in (False,True):
            with self.assertRaises(HTTPException) as caught:
                await self.audio()
            self.assertEqual(caught.exception.status_code,400)
            self.assertEqual(caught.exception.detail['code'],'undecodable_audio')
            self.assertNotIn('PRIVATE',str(caught.exception.detail))
        self.assertEqual(len(self.audio_calls),1)
        self.assertEqual(self.audio_model.in_flight,0)
        self.assertFalse(list(self.queue.inputs.root.rglob('*.audio')))
        self.backend.gate.set()
        await self.call()

    async def test_changed_upload_or_options_conflict_without_second_transcription(self):
        self.audio_gate.set()
        await self.audio()
        for changes in ({'data':b'different bytes'},{'fmt':'text'}):
            with self.assertRaises(HTTPException) as caught:
                await self.audio(**changes)
            self.assertEqual(caught.exception.status_code,409)
        self.assertEqual(len(self.audio_calls),1)
        self.assertFalse(list(self.queue.inputs.root.rglob('*.audio')))
        self.assertFalse(list(self.queue.inputs.root.glob('.pending-*')))

    async def test_saved_retry_does_not_reinstall_its_upload(self):
        self.audio_gate.set()
        await self.audio()
        with patch.object(self.queue.inputs,'install',side_effect=AssertionError('must reopen saved execution')):
            self.assertEqual(await self.audio(),{'text':'hello'})
        self.assertEqual(len(self.audio_calls),1)
        self.assertFalse(list(self.queue.inputs.root.glob('.pending-*')))

    async def test_cancel_during_install_keeps_handoff_then_enqueues_exactly_once(self):
        entered,release=threading.Event(),threading.Event()
        original=self.queue.inputs.install
        def blocked(grant,staged):
            entered.set()
            if not release.wait(3):
                raise AssertionError('test must release install')
            return original(grant,staged)
        pending=None
        with patch.object(self.queue.inputs,'install',blocked):
            try:
                pending=asyncio.create_task(self.audio(key='install-cancel'))
                await self.wait_for(entered.is_set)
                self.assertEqual(len(self.queue.supervisor.preparations),1)
                with self.assertRaises(RuntimeError):
                    self.queue.supervisor.close()
                self.assertEqual((await server.health())['status'],'ok')
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertEqual(len(self.queue.supervisor.preparations),1)
                self.assertTrue(list(self.queue.inputs.root.glob('.pending-*')))
                self.assertEqual(self.audio_calls,[])
                release.set()
                await self.wait_for(lambda: len(self.audio_calls)==1)
                self.assertTrue(Path(self.audio_calls[0]['path']).exists())
                self.assertEqual(self.audio_model.in_flight,1)
                self.audio_gate.set()
                await self.wait_for(lambda: not self.queue.supervisor.preparations)
                self.assertEqual(await self.audio(key='install-cancel'),{'text':'hello'})
                self.assertEqual(len(self.audio_calls),1)
                self.assertFalse(list(self.queue.inputs.root.glob('.pending-*')))
            finally:
                release.set()
                self.audio_gate.set()
                if pending is not None:
                    await asyncio.gather(pending,return_exceptions=True)

    async def test_failed_install_discards_only_its_unaccepted_copy(self):
        with patch.object(self.queue.inputs,'install',side_effect=ValueError('invalid copy')):
            with self.assertRaises(HTTPException) as caught:
                await self.audio()
        self.assertEqual(caught.exception.status_code,422)
        self.assertEqual(self.audio_calls,[])
        self.assertFalse(list(self.queue.inputs.root.glob('.pending-*')))
        self.assertFalse(list(self.queue.inputs.root.rglob('*.audio')))

    async def test_validation_precedes_loading_and_empty_or_oversized_uploads(self):
        self.queue.inputs.max_bytes=20
        for options,status in (({'data':b''},400),({'data':b'x'*21},413),
                               ({'fmt':'not-a-format'},400),({'model':'not-a-model'},400)):
            with self.subTest(options=list(options)),self.assertRaises(HTTPException) as caught:
                await self.audio(**options)
            self.assertEqual(caught.exception.status_code,status)
        self.assertEqual(self.loads,[])
        self.assertEqual(self.audio_calls,[])

    async def test_actual_multipart_route_preserves_owner_and_saved_text_response(self):
        self.audio_gate.set()
        app = FastAPI()
        @app.middleware('http')
        async def verified_identity(request,next_handler):
            request.state.caller={'username':'speaker'}
            return await next_handler(request)
        app.add_api_route('/v1/audio/transcriptions',server.create_transcription,methods=['POST'])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            for replay in (False,True):
                response = await client.post('/v1/audio/transcriptions',
                    headers={'Idempotency-Key':'multipart'},
                    data={'model':'whisper-1','response_format':'text','owner':'not-speaker',
                          'resource':'not-the-gpu','service_owned':'false'},
                    files={'file':('sample.wav',b'synthetic audio','audio/wav')})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.text,'hello\n')
                self.assertEqual(response.headers['content-type'],'text/plain; charset=utf-8')
        rows=self.queue.supervisor.journal.db.execute('SELECT owner,resource,model FROM grants').fetchall()
        self.assertEqual([tuple(row) for row in rows],[('speaker','gpu',server.WHISPER_DEFAULT)])
        self.assertEqual(len(self.audio_calls),1)
        self.assertTrue(self.audio_calls[0]['service_owned'])

    async def test_scheduler_cancel_is_unknown_not_a_backend_end_or_input_cleanup(self):
        pending=asyncio.create_task(self.audio())
        try:
            await self.wait_for(lambda: len(self.audio_calls)==1)
            execution,worker=next(iter(self.queue.supervisor.workers.items()))
            worker.cancel()
            await asyncio.gather(worker,return_exceptions=True)
            with self.assertRaises(HTTPException) as caught:
                await pending
            self.assertEqual(caught.exception.detail['state'],'uncertain')
            self.assertEqual(self.audio_model.in_flight,1)
            self.assertTrue(Path(self.audio_calls[0]['path']).is_file())
            self.assertIsNone(self.queue.supervisor.pump('gpu'))
            self.assertIsNone(self.queue.supervisor.journal.execution(execution,'speaker')['result'])
        finally:
            self.audio_gate.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending,return_exceptions=True)

    async def test_invalid_completed_result_is_failed_not_unknown_compute(self):
        with patch.object(self.manager.whisper,'transcribe',AsyncMock(return_value=['not-a-result'])):
            with self.assertRaises(HTTPException) as caught:
                await self.audio()
        self.assertEqual(caught.exception.status_code,502)
        self.assertEqual(caught.exception.detail['code'],'transcription_result_invalid')
        self.assertEqual(self.audio_model.in_flight,0)
        self.assertFalse(list(self.queue.inputs.root.rglob('*.audio')))
        self.backend.gate.set()
        await self.call()

    async def test_unkeyed_audio_is_refused_before_copy_or_load(self):
        upload=SimpleNamespace(read=AsyncMock(side_effect=AssertionError('must not read upload')))
        request=Request({'type':'http','headers':[],
                         'state':{'caller':{'username':'speaker'}}})
        with self.assertRaises(HTTPException) as caught:
            await server.create_transcription(file=upload,model='whisper-1',language=None,prompt=None,
                response_format='json',temperature=None,timestamp_granularities=None,
                http_request=request)
        self.assertEqual(caught.exception.status_code,428)
        upload.read.assert_not_called()
        self.assertEqual(self.loads,[])


class WhisperOwnedPolicy(unittest.IsolatedAsyncioTestCase):
    async def test_real_socket_owned_read_waits_past_legacy_timeout(self):
        received, tasks = [], set()
        async def handler(reader,writer):
            task=asyncio.current_task()
            tasks.add(task)
            try:
                headers=await reader.readuntil(b'\r\n\r\n')
                length=next(int(line.split(b':',1)[1]) for line in headers.split(b'\r\n')
                            if line.lower().startswith(b'content-length:'))
                body=await reader.readexactly(length)
                received.append(body)
                await asyncio.sleep(.1)
                response=b'{"text":"hello"}'
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '+
                             str(len(response)).encode()+b'\r\nConnection: close\r\n\r\n'+response)
                await writer.drain()
            except (BrokenPipeError,ConnectionResetError):
                pass  # legacy timeout intentionally disconnects its first socket
            finally:
                writer.close()
                await writer.wait_closed()
                tasks.discard(task)
        service=await asyncio.start_server(handler,'127.0.0.1',0)
        client=WhisperClient(host='127.0.0.1',port=service.sockets[0].getsockname()[1],timeout=.02)
        try:
            async with service:
                with self.assertRaises(WhisperUnavailable):
                    await client.transcribe('/synthetic/no-file-opened')
                result=await asyncio.wait_for(client.transcribe('/synthetic/no-file-opened',
                                                               service_owned=True),2)
                self.assertEqual(result,{'text':'hello'})
                self.assertEqual(len(received),2)
                self.assertTrue(all(b'service_owned' not in body for body in received))
                if tasks:
                    await asyncio.wait_for(asyncio.gather(*list(tasks)),2)
        finally:
            service.close()
            await service.wait_closed()
