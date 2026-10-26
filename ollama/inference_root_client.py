"""Staged registry-root control client; not wired into native/model admission.

The journal saves identity/intent, never capacity or dispatch permission. A
receipt is historical after disconnect. No background polling, allocation retry,
job deadline, backend operation or implicit end/release runs here.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import stat
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx


class RootProtocolError(RuntimeError):
    pass


def _identifier(value):
    if not isinstance(value,str) or not value.strip() or len(value)>512:
        raise ValueError('Explicit bounded identity required')
    value.encode('utf-8')
    return value


def _resource(value):
    _identifier(value)
    if any(part in ('','.','..') for part in value.split('/')):
        raise ValueError('Canonical resource identity required')
    return value


def _base(value):
    url=urlsplit(value)
    if (url.scheme!='https' or not url.hostname or url.username is not None or
        url.password is not None or url.query or url.fragment or url.path.rstrip('/')!='/api/v1'):
        raise ValueError('An HTTPS registry API origin without credentials is required')
    return str(httpx.URL(value)).rstrip('/')+'/'


class RootIntentStore:
    """Private durable client intent, bound to one registry/principal/service.

    The job_id argument identifies one durable physical intent, not a username
    or mutable session alias. RootPreparingArbiter supplies its preparation ID,
    bound to the logical requester job. Yield/resume uses a fresh physical intent
    without changing that logical job. Connections are operation-local.
    """
    def __init__(self,path,*,registry,actor,service):
        self.registry,self.actor,self.service=_base(registry),_identifier(actor),_identifier(service)
        self.path=Path(path).absolute()
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd=os.open(self.path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_mode&0o077 or info.st_uid!=os.getuid():
                raise PermissionError('Root client journal must be an owned private regular file')
        finally:
            os.close(fd)
        with self._transaction() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS root_client_binding (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),registry TEXT NOT NULL,
                actor TEXT NOT NULL,service TEXT NOT NULL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS root_client_intents (
                job_id TEXT NOT NULL,resource TEXT NOT NULL,submit_key TEXT NOT NULL UNIQUE,
                root_id TEXT,initial_lease_id TEXT,end_requested INTEGER NOT NULL DEFAULT 0,
                receipt TEXT,PRIMARY KEY(job_id,resource))''')
            conn.execute('''CREATE TABLE IF NOT EXISTS root_client_operations (
                job_id TEXT NOT NULL,resource TEXT NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL,
                operation_key TEXT NOT NULL UNIQUE,PRIMARY KEY(job_id,resource,kind,payload))''')
            conn.execute('INSERT OR IGNORE INTO root_client_binding VALUES(1,?,?,?)',
                         (self.registry,self.actor,self.service))
            binding=conn.execute('SELECT registry,actor,service FROM root_client_binding').fetchone()
            if tuple(binding)!=(self.registry,self.actor,self.service):
                raise PermissionError('Root journal belongs to another registry/principal/service')

    @contextmanager
    def _transaction(self):
        conn=sqlite3.connect(self.path,timeout=10,isolation_level=None)
        conn.row_factory=sqlite3.Row
        try:
            conn.execute('PRAGMA synchronous=FULL')
            conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def prepare(self,job_id,resource):
        _identifier(job_id);_resource(resource)
        with self._transaction() as conn:
            conn.execute('INSERT OR IGNORE INTO root_client_intents(job_id,resource,submit_key) VALUES(?,?,?)',
                         (job_id,resource,uuid4().hex))
            return dict(conn.execute('SELECT * FROM root_client_intents WHERE job_id=? AND resource=?',
                                     (job_id,resource)).fetchone())

    def request_end(self,job_id,resource):
        self.prepare(job_id,resource)
        with self._transaction() as conn:
            conn.execute('UPDATE root_client_intents SET end_requested=1 WHERE job_id=? AND resource=?',
                         (job_id,resource))
        return self.prepare(job_id,resource)

    def record(self,job_id,resource,receipt):
        with self._transaction() as conn:
            old=conn.execute('SELECT * FROM root_client_intents WHERE job_id=? AND resource=?',
                             (job_id,resource)).fetchone()
            if old is None:
                raise RootProtocolError('Root intent was not persisted before submission')
            if old['root_id'] is not None and (old['root_id']!=receipt['root_id'] or
                old['initial_lease_id']!=receipt['initial_lease_id']):
                raise RootProtocolError('Registry changed the saved root identity')
            if old['receipt'] is not None:
                previous=json.loads(old['receipt'])
                order={'open':0,'ending':1,'complete':2}
                if order[receipt['state']]<order[previous['state']]:
                    raise RootProtocolError('Stale root receipt cannot reopen an ended job')
                if previous.get('never_admitted') and not receipt.get('never_admitted'):
                    raise RootProtocolError('A never-admitted end cannot acquire a root')
            conn.execute('UPDATE root_client_intents SET root_id=?,initial_lease_id=?,receipt=? WHERE job_id=? AND resource=?',
                         (receipt['root_id'],receipt['initial_lease_id'],json.dumps(receipt,sort_keys=True),job_id,resource))
            # end_requested deliberately remains sticky, even on a delayed open ACK.

    def operation(self,job_id,resource,kind,payload):
        encoded=json.dumps(payload,sort_keys=True,separators=(',',':'))
        with self._transaction() as conn:
            if not conn.execute('SELECT 1 FROM root_client_intents WHERE job_id=? AND resource=?',
                                (job_id,resource)).fetchone():
                raise RootProtocolError('No saved root intent')
            conn.execute('INSERT OR IGNORE INTO root_client_operations VALUES(?,?,?,?,?)',
                         (job_id,resource,kind,encoded,uuid4().hex))
            return conn.execute('SELECT operation_key FROM root_client_operations WHERE job_id=? AND resource=? AND kind=? AND payload=?',
                                (job_id,resource,kind,encoded)).fetchone()[0]


class RegistryRootClient:
    """Explicit synchronous control operations; transport close never ends work.

    A finite HTTP observation timeout leaves intent recoverable. It is not a job
    deadline. The caller decides when to retry/read the same accepted work.
    Only a qualified controller with fresh registry/host evidence may load.
    """
    FEATURES=('idempotent_submit','end_before_intake','explicit_end','finite_health',
              'exact_successors','absence_recovery')

    def __init__(self,store,*,token,transport=None,observation_timeout=30):
        self.store=store
        self._verified=False
        self._http=httpx.Client(base_url=store.registry,
            headers={'Authorization':'Bearer '+token},follow_redirects=False,trust_env=False,
            timeout=observation_timeout,transport=transport)

    def close(self):
        self._http.close()

    def negotiate(self):
        if self._verified:
            return
        response=self._http.get('capabilities/lease-roots')
        response.raise_for_status()
        cap=response.json()
        if (cap.get('protocol')!='samcloud.lease-roots' or type(cap.get('version')) is not int or
            cap['version']!=1 or cap.get('actor')!=self.store.actor or
            not isinstance(cap.get('features'),dict) or
            any(cap['features'].get(name) is not True for name in self.FEATURES)):
            raise RootProtocolError('Registry does not support this identity and root recovery contract')
        self._verified=True

    def _receipt(self,job_id,resource,response):
        response.raise_for_status()
        body=response.json()
        if (not isinstance(body,dict) or body.get('lifetime')!='explicit_end' or
            body.get('resource_id')!=resource or body.get('state') not in ('open','ending','complete')):
            raise RootProtocolError('Invalid root receipt')
        if body.get('never_admitted') is True:
            if (body['state']!='complete' or body.get('service_id')!=self.store.service or
                any(body.get(k) is not None for k in ('root_id','initial_lease_id','current_lease_id','lease'))):
                raise RootProtocolError('Invalid never-admitted end receipt')
        else:
            for field in ('root_id','initial_lease_id','current_lease_id'):
                _identifier(body.get(field))
            lease=body.get('lease')
            if lease is not None and (not isinstance(lease,dict) or
                lease.get('id')!=body['current_lease_id'] or lease.get('resource_id')!=resource):
                raise RootProtocolError('Invalid exact root member receipt')
            if lease is None and body['state']!='complete':
                raise RootProtocolError('A nonterminal root is missing its exact physical claim')
        self.store.record(job_id,resource,body)
        return body

    def begin(self,job_id,resource):
        intent=self.store.prepare(job_id,resource)
        if intent['end_requested']:
            return self.end(job_id,resource)
        self.negotiate()
        if intent['root_id']:
            # A missing exact root is reconciliation, never replacement intake.
            response=self._http.get('lease-roots/'+quote(intent['root_id'],safe=''))
        else:
            response=self._http.post('resources/'+quote(resource,safe='/')+'/roots',
                json={'key':intent['submit_key'],'service_id':self.store.service})
        result=self._receipt(job_id,resource,response)
        # A concurrent explicit end may have committed locally while submit was
        # in flight. Never return that older open ACK as continuing work.
        if self.store.prepare(job_id,resource)['end_requested']:
            return self.end(job_id,resource)
        return result

    def observe(self,job_id,resource):
        intent=self.store.prepare(job_id,resource)
        if not intent['root_id']:
            raise RootProtocolError('No acknowledged root; resume the saved begin/end intent')
        self.negotiate()
        return self._receipt(job_id,resource,self._http.get('lease-roots/'+quote(intent['root_id'],safe='')))

    def end(self,job_id,resource):
        intent=self.store.request_end(job_id,resource)
        self.negotiate()
        response=self._http.post('resources/'+quote(resource,safe='/')+'/roots/end',
            json={'key':intent['submit_key'],'service_id':self.store.service})
        return self._receipt(job_id,resource,response)

    def renew(self,job_id,resource,*,lease_id):
        return self._member_operation(job_id,resource,'renew',lease_id)

    def settle(self,job_id,resource,*,lease_id,observation):
        return self._member_operation(job_id,resource,'settle',lease_id,observation)

    def recover_absent(self,job_id,resource,*,lease_id,observation):
        return self._member_operation(job_id,resource,'recover-absent',lease_id,observation)

    def _member_operation(self,job_id,resource,kind,lease_id,observation=None):
        intent=self.store.prepare(job_id,resource)
        _identifier(lease_id)
        if not intent['root_id']:
            raise RootProtocolError('An exact acknowledged root is required')
        if kind=='recover-absent' and intent['end_requested']:
            raise RootProtocolError('Explicit local end forbids readmission; reconcile/end the same root')
        payload={'lease_id':lease_id}
        if kind!='renew':
            payload['observation']=_identifier(observation)
            payload['key']=self.store.operation(job_id,resource,kind,payload)
        self.negotiate()
        response=self._http.post('lease-roots/'+quote(intent['root_id'],safe='')+'/'+kind,json=payload)
        return self._receipt(job_id,resource,response)
