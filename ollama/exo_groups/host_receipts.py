"""Authenticated finite observation of the host's exact journaled rank attempt.

This channel cannot start/stop a rank, publish files or settle ownership. Receipt
identity comes from the configured host and launch journal, not a cleanup report.
"""

import base64
from dataclasses import asdict, dataclass
import hashlib
from importlib.resources import files
import secrets
import shlex
import sys

from .cleanup import CleanupExpectation, NoStartExpectation, validate_cleanup
from .host_channel import HostRoute, run_probe
from .http import decode_json
from .member import ExpectedMember
from .plan import BindingRefused, LaunchBinding
from .prepare import canonical
from .rank_process import ProcessIdentity, Listener, RunBinding
from .receipt_files import ReceiptUnavailable


PREFIX=b'EXO_RECEIPTS_V1:'
MAX_FRAME=512*1024


def _program(source, *, historical=False):
    return ("import sys,json,hashlib,os\n"
        "r=json.loads(sys.stdin.buffer.read(524289))\n"
        f"s={source!r}\n"
        "assert hashlib.sha256(s.encode()).hexdigest()==r['collector_sha256']\n"
        "n={};exec(compile(s,'<host-receipt-collector>','exec'),n)\n"
        "v={'request_id':r['request_id'],'collector_sha256':r['collector_sha256'],"
        f"'report':n[{('collect_historical_attempt' if historical else 'collect_attempt')!r}](r['spec'])}}\n"
        "print('EXO_RECEIPTS_V1:'+json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False),flush=True)")


async def observe_local_attempt(spec, *, historical=False):
    """Read the host's journal in a separate, bounded metadata observer.

    POSIX locks are per process: closing a raw descriptor for SQLite's file in
    a thread can release the gateway's writer locks. The existing isolated
    collector/probe protocol avoids that without weakening any file checks.
    No environment/credential inheritance, shell, native model or lease call.
    """
    source = files(__package__).joinpath('receipt_files.py').read_text()
    sha = hashlib.sha256(source.encode()).hexdigest()
    nonce = secrets.token_hex(32)
    request = canonical({'request_id': nonce, 'collector_sha256': sha, 'spec': spec})
    if len(request) > MAX_FRAME:
        raise ReceiptUnavailable('local host receipt request exceeds bound')
    try:
        raw = await run_probe((sys.executable, '-I', '-S', '-B', '-c', _program(source, historical=historical)), request,
            environment={'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONDONTWRITEBYTECODE': '1'})
        if not isinstance(raw, bytes) or len(raw) > MAX_FRAME:
            raise ValueError('frame bound')
        frames = [line[len(PREFIX):] for line in raw.splitlines() if line.startswith(PREFIX)]
        if len(frames) != 1:
            raise ValueError('frame count')
        value = decode_json(frames[0])
        report = value.get('report')
        if (set(value) != {'request_id', 'collector_sha256', 'report'}
                or value['request_id'] != nonce or value['collector_sha256'] != sha
                or not isinstance(report, dict) or canonical(report.get('binding')) != canonical(spec)
                or type(report.get('uid')) is not int or report['uid'] != spec['uid']
                or report.get('boot_id') != spec['boot_id']):
            raise ValueError('local receipt identity')
        return report
    except Exception:
        raise ReceiptUnavailable('isolated host attempt observation unavailable') from None


def prepared_from_report(plan,value):
    try:
        run=value['binding']
        launch=dict(run['expected']['binding'])
        launch['layers']=tuple(launch['layers'])
        expected=ExpectedMember.from_launch(plan,LaunchBinding(**launch))
        pinned=NoStartExpectation(RunBinding.from_expected(expected,run['runtime_sha256']),
            value['host_boot'],value['launch_sha256'],value['attempt_sha256'],
            tuple(Listener(**item) for item in value['listeners']),
            value['baseline_in_use_bytes'],value['tolerance_bytes'],value['prepared_at'])
        if canonical(asdict(pinned))!=canonical(value):
            raise ValueError('preparation shape')
        return pinned
    except (KeyError,TypeError,ValueError):
        raise BindingRefused('host preparation differs from the pinned plan') from None


@dataclass(frozen=True)
class HostReceiptObservation:
    prepared: NoStartExpectation
    started: CleanupExpectation | None
    phase: str
    cleanup_bytes: bytes | None
    cleanup_sha256: str | None

    async def pin(self,completion,*,generation):
        # Independent preparation/start observation is journaled before any
        # artifact upload/readback. This method provides no launch permission.
        if self.started is not None:
            if type(generation) is not int or generation!=self.started.binding.generation:
                raise BindingRefused('actual startup receipt requires its explicit launch generation')
            completion.pin_launch(self.started)
        else:
            await completion.pin_prepared(self.prepared,generation=generation)


class AuthenticatedReceiptReader:
    def __init__(self,route,*,collector_sha256,environment,runner=run_probe):
        if (not isinstance(route,HostRoute) or not isinstance(environment,dict)
                or any(not isinstance(k,str) or not isinstance(v,str) for k,v in environment.items())
                or any(not k or '=' in k or '\x00' in k or '\x00' in v for k,v in environment.items())
                or any(any(part in k.upper() for part in ('TOKEN','AUTH','BEARER','SECRET'))
                       for k in environment) or not callable(runner)):
            raise ValueError('exact authenticated host route and explicit minimal environment required')
        source=files(__package__).joinpath('receipt_files.py').read_text()
        if hashlib.sha256(source.encode()).hexdigest()!=collector_sha256:
            raise ValueError('host receipt collector differs from its source pin')
        self.route,self.source,self.collector_sha256=route,source,collector_sha256
        self.environment,self.runner=dict(environment),runner

    async def read(self,plan,*,binding,journal_path,state_path,boot_id,launch_sha256,runtime_sha256):
        expected=ExpectedMember.from_launch(plan,binding)
        if (expected.binding.rank,expected.binding.resource_id)!=(self.route.rank,self.route.resource_id):
            raise BindingRefused('host receipt route differs from the planned member')
        spec={'resource_id':binding.resource_id,'group_id':binding.group_id,
            'generation':binding.generation,'lease_id':binding.lease_id,
            'runtime_sha256':runtime_sha256,'launch_sha256':launch_sha256,'boot_id':boot_id,'uid':self.route.uid,
            'state_path':state_path,'journal_path':journal_path}
        RunBinding.from_expected(expected,runtime_sha256)
        nonce=secrets.token_hex(32)
        request=canonical({'request_id':nonce,'collector_sha256':self.collector_sha256,'spec':spec})
        if len(request)>MAX_FRAME:
            raise BindingRefused('host receipt request exceeds bound')
        remote=shlex.join([self.route.interpreter,'-I','-S','-B','-c',_program(self.source)])
        try:
            raw=await self.runner((*self.route.command,remote),request,environment=dict(self.environment))
            if not isinstance(raw,bytes) or len(raw)>MAX_FRAME:
                raise ValueError('host frame')
            frames=[line[len(PREFIX):] for line in raw.splitlines() if line.startswith(PREFIX)]
            if len(frames)!=1:
                raise ValueError('host frame count')
            frame=decode_json(frames[0])
            if (set(frame)!={'request_id','collector_sha256','report'} or frame['request_id']!=nonce
                    or frame['collector_sha256']!=self.collector_sha256):
                raise ValueError('host frame identity')
            return self._checked(plan,binding,spec,frame['report'])
        except Exception:
            raise BindingRefused('authenticated host attempt observation unavailable') from None

    def _checked(self,plan,binding,spec,report):
        if (not isinstance(report,dict) or canonical(report.get('binding'))!=canonical(spec)
                or type(report.get('uid')) is not int or report['uid']!=self.route.uid
                or report.get('boot_id')!=spec['boot_id']
                or report.get('phase') not in ('prepared','launch_committed','spawning','launch_fenced')):
            raise BindingRefused('host attempt identity differs')
        prepared=prepared_from_report(plan,report['prepared'])
        if (prepared.binding!=RunBinding.from_expected(ExpectedMember.from_launch(plan,binding),
                                                       spec['runtime_sha256'])
                or prepared.host_boot!=spec['boot_id'] or prepared.launch_sha256!=spec['launch_sha256']):
            raise BindingRefused('host preparation differs from deployment pins')
        started,launch,phase=report.get('started'),report.get('launch_intent'),report['phase']
        actual=None
        if started is not None:
            if (phase!='spawning' or not isinstance(started,dict) or started.get('type')!='rank_started'
                    or not isinstance(launch,dict)
                    or canonical({k:started.get(k) for k in ('group_id','generation','lease_id','resource_id')})
                        !=canonical({k:getattr(binding,k) for k in ('group_id','generation','lease_id','resource_id')})
                    or started.get('host_boot')!=spec['boot_id'] or launch.get('host_boot')!=spec['boot_id']
                    or canonical(launch.get('binding'))!=canonical(asdict(prepared.binding))):
                raise BindingRefused('actual host launch differs from its prepared attempt')
            child=started.get('child')
            fields=ProcessIdentity.__dataclass_fields__
            if (not isinstance(child,dict) or set(child)!=set(fields)
                    or any(type(child[k]) is not int or child[k]<0 for k in fields)
                    or any(child[k]==0 for k in ('pid','group_id','started_seconds'))
                    or child['started_microseconds']>=1000000 or child['uid']!=self.route.uid):
                raise BindingRefused('actual child birth identity unavailable')
            actual=CleanupExpectation(prepared.binding,ProcessIdentity(**child),prepared.listeners,
                                      launch.get('baseline_in_use_bytes'),prepared.tolerance_bytes)
        elif phase in ('prepared','launch_fenced') and launch is not None:
            raise BindingRefused('host no-launch state has a launch intent')
        cleanup=report.get('cleanup')
        raw=digest=None
        if cleanup is not None:
            if (not isinstance(cleanup,dict) or set(cleanup)!={'bytes','sha256','base64'}
                    or type(cleanup['bytes']) is not int or not 0<cleanup['bytes']<=128*1024
                    or not isinstance(cleanup['base64'],str)):
                raise BindingRefused('host cleanup envelope differs')
            raw=base64.b64decode(cleanup['base64'],validate=True)
            digest=hashlib.sha256(raw).hexdigest()
            if len(raw)!=cleanup['bytes'] or digest!=cleanup['sha256']:
                raise BindingRefused('host cleanup bytes differ')
            if actual is None and phase!='launch_fenced':
                raise BindingRefused('missing actual child is not no-launch evidence')
            validate_cleanup(raw,actual or prepared)
        return HostReceiptObservation(prepared,actual,phase,raw,digest)
