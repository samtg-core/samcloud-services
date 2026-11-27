"""Read-only host journal and immutable rank receipts; standard library only.

No lease/process/model effects. Missing or changing records are unknown, never
proof of cleanup. The trusted route supplies the exact paths and member binding.
"""

import base64
from contextlib import closing
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import sqlite3
import stat
import subprocess
from urllib.parse import quote
from uuid import UUID


class ReceiptUnavailable(ValueError):
    pass


# Explicit metadata budgets: the observed closed setup has 10,168 small
# authority events. Read its complete history, including every possible start;
# do not truncate or reset it. The independent 8 MiB content bound is unchanged.
MAX_EVENT_FILES=20000
MAX_EVENT_BYTES=8*1024*1024


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                      allow_nan=False).encode()


def decode(raw):
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:
                raise ReceiptUnavailable('ambiguous host receipt')
            value[key]=item
        return value
    def nonfinite(_):
        raise ReceiptUnavailable('nonfinite host receipt')
    value=json.loads(raw,object_pairs_hook=unique,parse_constant=nonfinite)
    if not isinstance(value,dict):
        raise ReceiptUnavailable('host receipt is not an object')
    return value


def boot_id():
    raw=subprocess.check_output(['/usr/sbin/sysctl','-n','kern.bootsessionuuid'],text=True).strip()
    return str(UUID(raw))


def _parts(path):
    if (not isinstance(path,str) or not path.startswith('/') or path=='/'
            or str(PurePosixPath(path))!=path or '..' in PurePosixPath(path).parts or '\x00' in path):
        raise ReceiptUnavailable('exact absolute host path required')
    return PurePosixPath(path).parts[1:]


def _directory(path):
    descriptor=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in _parts(path):
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
            os.close(descriptor)
            descriptor=child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _identity(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_mode,info.st_uid)


def _read(directory,name,uid,*,private=True):
    if '/' in name or name in ('.','..'):
        raise ReceiptUnavailable('exact rank receipt filename required')
    descriptor=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
    try:
        before=os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=uid
                or before.st_mode & (0o077 if private else 0o022) or before.st_size>128*1024):
            raise ReceiptUnavailable('host receipt identity/mode unavailable')
        raw=bytearray()
        while chunk:=os.read(descriptor, min(65536,128*1024+1-len(raw))):
            raw.extend(chunk)
            if len(raw)>128*1024:
                raise ReceiptUnavailable('host receipt exceeds bound')
        if _identity(before)!=_identity(os.fstat(descriptor)):
            raise ReceiptUnavailable('host receipt changed while reading')
        return bytes(raw)
    finally:
        os.close(descriptor)


def collect_attempt(spec,*,boot_reader=boot_id):
    return _collect_attempt(spec, boot_reader=boot_reader, historical=False)


def collect_historical_attempt(spec,*,boot_reader=boot_id):
    """Read old immutable metadata across reboot; never itself cleanup proof.

    Normal live receipts continue requiring their original boot. Historical
    callers must separately validate a normal prior cleanup or reboot evidence
    AND exact plane settlement before authorizing any new preparation.
    """
    return _collect_attempt(spec, boot_reader=boot_reader, historical=True)


def _collect_attempt(spec,*,boot_reader,historical):
    current_boot=boot_reader()
    try:
        if str(UUID(current_boot)) != current_boot or UUID(current_boot).int == 0:
            raise ValueError('invalid boot')
    except (ValueError,TypeError,AttributeError):
        raise ReceiptUnavailable('current kernel boot unavailable') from None
    keys={'resource_id','group_id','generation','lease_id','runtime_sha256','launch_sha256',
          'boot_id','uid','state_path','journal_path'}
    if (not isinstance(spec,dict) or set(spec)!=keys or type(spec['uid']) is not int
            or spec['uid']!=os.getuid() or (not historical and current_boot!=spec['boot_id'])
            or type(spec['generation']) is not int or spec['generation']<1
            or any(not isinstance(spec[k],str) or not re.fullmatch(r'[a-f0-9]{64}',spec[k])
                   for k in ('runtime_sha256','launch_sha256'))):
        raise ReceiptUnavailable('exact host attempt policy required')
    state=_directory(spec['state_path'])
    _parts(spec['journal_path'])
    journal=PurePosixPath(spec['journal_path'])
    parent=_directory(str(journal.parent))
    database=None
    try:
        before=os.fstat(state)
        parent_before=os.fstat(parent)
        if before.st_uid!=spec['uid'] or before.st_mode & 0o077:
            raise ReceiptUnavailable('private owned host attempt directory required')
        # Pin the actual DB inode while SQLite performs a read-only snapshot.
        database=os.open(journal.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
        db_before=os.fstat(database)
        if (not stat.S_ISREG(db_before.st_mode) or db_before.st_nlink!=1
                or db_before.st_uid!=spec['uid'] or db_before.st_mode & 0o022):
            raise ReceiptUnavailable('owned host journal required')
        with closing(sqlite3.connect('file:'+quote(str(journal),safe='/')+'?mode=ro',
                                     uri=True,timeout=0)) as connection, connection:
            connection.row_factory=sqlite3.Row
            connection.execute('BEGIN')
            row=connection.execute('SELECT * FROM exo_host_rank_attempts WHERE resource_id=? AND lease_id=?',
                                   (spec['resource_id'],spec['lease_id'])).fetchone()
            if row is None:
                raise ReceiptUnavailable('exact host attempt is not registered')
            row=dict(row)
            if (any(row.get(k)!=spec[k] or type(row.get(k)) is not type(spec[k]) for k in
                    ('resource_id','group_id','generation','lease_id','runtime_sha256','launch_sha256','state_path'))
                    or row.get('boot_id')!=spec['boot_id'] or not isinstance(row.get('prepared_json'),str)
                    or row.get('phase') not in ('prepared','launch_committed','spawning','launch_fenced')):
                raise ReceiptUnavailable('exact host attempt journal binding is unknown or changed')
            prepared=decode(_read(state,'host-prepared.json',spec['uid']))
            if canonical(prepared)!=canonical(decode(row['prepared_json'])):
                raise ReceiptUnavailable('host preparation differs from the journal')
            # Only typed immutable metadata is read. stdout/stderr, credentials
            # and arbitrary files are never returned to the coordinator.
            names=os.listdir(state)
            events=[name for name in names if re.fullmatch(r'event-[0-9]{6}\.json',name)]
            if len(events)>MAX_EVENT_FILES:
                raise ReceiptUnavailable('host event index requires explicit reconciliation')
            started=[]
            event_bytes=0
            for name in sorted(events):
                raw=_read(state,name,spec['uid'])
                event_bytes+=len(raw)
                if event_bytes>MAX_EVENT_BYTES:
                    raise ReceiptUnavailable('host event metadata exceeds observation bound')
                event=decode(raw)
                if event.get('type')=='rank_started':
                    started.append(event)
            if len(started)>1:
                raise ReceiptUnavailable('ambiguous host launch receipt')
            launch=decode(_read(state,'launch-intent.json',spec['uid'])) if 'launch-intent.json' in names else None
            cleanup=None
            if 'cleanup-evidence.json' in names:
                raw=_read(state,'cleanup-evidence.json',spec['uid'])
                cleanup={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                         'base64':base64.b64encode(raw).decode()}
            fresh=connection.execute('SELECT * FROM exo_host_rank_attempts WHERE resource_id=? AND lease_id=?',
                                     (spec['resource_id'],spec['lease_id'])).fetchone()
            if dict(fresh)!=row:
                raise ReceiptUnavailable('host journal attempt changed')
        # No pathname replacement or boot/attempt-directory race may hide
        # between the snapshot and immutable artifact reads.
        actual=os.stat(journal.name,dir_fd=parent,follow_symlinks=False)
        if ((actual.st_dev,actual.st_ino)!=(db_before.st_dev,db_before.st_ino)
                or _identity(db_before)!=_identity(os.fstat(database))
                or _identity(parent_before)!=_identity(os.fstat(parent))
                or _identity(before)!=_identity(os.fstat(state)) or boot_reader()!=current_boot):
            raise ReceiptUnavailable('host attempt changed during observation')
        return {**({'observed_host_boot':current_boot} if historical else {}),
                'uid':spec['uid'],'boot_id':spec['boot_id'],'binding':spec,'phase':row['phase'],
                'prepared':prepared,'started':started[0] if started else None,
                'launch_intent':launch,'cleanup':cleanup}
    finally:
        if database is not None:
            os.close(database)
        os.close(parent)
        os.close(state)


def verify_prepare_only(session, uid, state, prefix, binding, boot):
    """Check both known pre-registration exits after the exact lease was fenced."""
    parent, name = session.rsplit('/', 1)
    base = _directory(parent)
    try:
        if state == 'owned' and prefix == 'prepare':
            try:
                os.stat(name, dir_fd=base, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise ReceiptUnavailable('read-only preparation has local state')
        if state != 'unconfirmed' or prefix is not None:
            raise ReceiptUnavailable('unknown preparation boundary')
        directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=base)
        try:
            info = os.fstat(directory)
            if info.st_uid != uid or info.st_mode & 0o077 or set(os.listdir(directory)) != {'deployment.json', 'attempt'}:
                raise ReceiptUnavailable('unexpected files in incomplete preparation')
            deployment = decode(_read(directory, 'deployment.json', uid))
            attempt = os.open('attempt', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            try:
                info = os.fstat(attempt)
                if info.st_uid != uid or info.st_mode & 0o077 or set(os.listdir(attempt)) != {'prepared.json'}:
                    raise ReceiptUnavailable('attempt advanced beyond preparation')
                prepared = decode(_read(attempt, 'prepared.json', uid))
            finally:
                os.close(attempt)
            # This lease was fenced before registration. The host wrote these
            # two records at preparation time. A successor may run a new build.
            runtime = deployment.get('runtime_sha256')
            if runtime != binding['runtime_sha256']:
                argv = deployment.get('argv')
                if (not isinstance(runtime, str) or not re.fullmatch('[a-f0-9]{64}', runtime)
                        or not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv)
                        or hashlib.sha256(json.dumps(argv).encode()).hexdigest() != prepared.get('argv_sha256')):
                    raise ReceiptUnavailable('historical runtime or launch arguments differ')
                binding = dict(binding, runtime_sha256=runtime)
            if (deployment.get('kind') != 'prepared_host_configuration' or deployment.get('host_boot') != boot
                    or canonical(deployment.get('binding')) != canonical(binding['expected'])
                    or prepared.get('state') != 'prepared_not_launched'
                    or canonical(prepared.get('binding')) != canonical(binding)):
                raise ReceiptUnavailable('historical preparation binding differs')
        finally:
            os.close(directory)
    finally:
        os.close(base)
