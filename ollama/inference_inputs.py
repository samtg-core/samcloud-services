"""Private durable upload copies for queued work, not a public file service.

Explicit grant identity owns each copy. A digest checks integrity; identical bytes
under two work intents do not share an execution or a lifetime. Pending/uncertain
work retains its input. Route wiring and abandoned-upload collection are separate.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile


class InputTooLarge(ValueError):
    pass


class EmptyInput(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedInput:
    path: Path
    sha256: str
    stamp: tuple


@dataclass(frozen=True)
class StagedInput:
    path: Path
    sha256: str
    size: int

    def descriptor(self):
        return {'sha256':self.sha256,'bytes':self.size}


class InferenceInputs:
    def __init__(self, root, max_bytes):
        if type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('positive upload bound required')
        self.root = Path(root).absolute()
        self.max_bytes = max_bytes
        self._directory(self.root)

    @staticmethod
    def _directory(path, *, create=True):
        if create:
            path.mkdir(mode=0o700,parents=True,exist_ok=True)
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode&0o077 or info.st_uid!=os.geteuid():
            raise PermissionError('input directory must be private and service-owned')

    @staticmethod
    def _sync_directory(path):
        fd = os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    async def capture(self, upload):
        """Bound memory/bytes; persist before handing an upload to admission.

        Cancellation before capture returns removes only this unaccepted copy.
        No client filename or filesystem path is used as a destination.
        """
        fd, filename = tempfile.mkstemp(prefix='.pending-',dir=self.root)
        path = Path(filename)
        digest, size = hashlib.sha256(), 0
        try:
            with os.fdopen(fd,'wb') as output:
                while chunk:=await upload.read(1024*1024):
                    if not isinstance(chunk,bytes):
                        raise ValueError('upload must supply bytes')
                    size+=len(chunk)
                    if size>self.max_bytes:
                        raise InputTooLarge('upload exceeds configured byte bound')
                    output.write(chunk)
                    digest.update(chunk)
                if not size:
                    raise EmptyInput('upload is empty')
                output.flush()
                os.fsync(output.fileno())
            self._sync_directory(self.root)
            return StagedInput(path,digest.hexdigest(),size)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def _validate(self, descriptor):
        if (not isinstance(descriptor,dict) or set(descriptor)!={'sha256','bytes'} or
                not isinstance(descriptor['sha256'],str) or
                re.fullmatch('[0-9a-f]{64}',descriptor['sha256']) is None or
                type(descriptor['bytes']) is not int or
                not 0<descriptor['bytes']<=self.max_bytes):
            raise ValueError('invalid input descriptor')

    def discard_staged(self, staged):
        """Retire this unaccepted temporary copy, never an installed execution.

        Call off-loop and only after its installation worker has ended. Unknown
        or changed files remain for reconciliation, not deletion by path alone.
        """
        if (not isinstance(staged,StagedInput) or staged.path.parent!=self.root or
                not staged.path.name.startswith('.pending-')):
            raise ValueError('input was not staged in this store')
        try:
            self._verify(staged.path,staged.descriptor())
            staged.path.unlink()
        except FileNotFoundError:
            return False  # install already transferred the copy
        self._sync_directory(self.root)
        return True

    def _path(self, grant_id, descriptor):
        self._validate(descriptor)
        if not isinstance(grant_id,str) or re.fullmatch('[0-9a-f]{32}',grant_id) is None:
            raise ValueError('invalid service grant identity')
        return self.root/grant_id/(descriptor['sha256']+'.audio')

    def _verify(self, path, descriptor):
        self._validate(descriptor)
        fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_mode&0o077 or
                    info.st_uid!=os.geteuid() or info.st_size!=descriptor['bytes']):
                raise ValueError('input is not the recorded private file')
            digest = hashlib.sha256()
            while chunk:=os.read(fd,1024*1024):
                digest.update(chunk)
            if digest.hexdigest()!=descriptor['sha256']:
                raise ValueError('input integrity mismatch')
            after = os.fstat(fd)
            if self._stamp(info)!=self._stamp(after):
                raise ValueError('input changed during verification')
            return self._stamp(after)
        finally:
            os.close(fd)

    @staticmethod
    def _stamp(info):
        return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,
                info.st_ctime_ns,info.st_mode,info.st_uid)

    def install(self, grant_id, staged):
        """Immutable per-grant copy; call off-loop for large upload verification."""
        if (not isinstance(staged,StagedInput) or staged.path.parent!=self.root or
                not staged.path.name.startswith('.pending-')):
            raise ValueError('input was not staged in this store')
        descriptor = staged.descriptor()
        target = self._path(grant_id,descriptor)
        self._verify(staged.path,descriptor)
        self._directory(target.parent)
        try:
            os.link(staged.path,target,follow_symlinks=False)
        except FileExistsError:
            self._verify(target,descriptor)  # never overwrite an existing input
        self._sync_directory(target.parent)
        self._sync_directory(self.root)
        staged.path.unlink()
        return descriptor

    def path(self, grant_id, descriptor):
        target = self._path(grant_id,descriptor)
        self._directory(target.parent,create=False)
        self._verify(target,descriptor)
        return target

    def proof(self, grant_id, descriptor):
        """Verify large bytes off-loop; retirement rechecks metadata/references."""
        target = self._path(grant_id,descriptor)
        self._directory(target.parent,create=False)
        return VerifiedInput(target,descriptor['sha256'],self._verify(target,descriptor))

    def retire(self, journal, execution_id, owner, *, proof=None):
        """Delete only a saved-terminal execution's own copy; unknown retains it.

        Called after the supervisor commits its result, never from an HTTP
        disconnect/finally. Authoritative journal ownership gates the lookup.
        """
        row = journal.execution(execution_id,owner)
        if row['state'] not in ('complete','failed'):
            return False
        request = json.loads(row['request'])
        if request.get('operation')!='transcription':
            return False
        descriptor = request['body']['input']
        target = self._path(row['grant_id'],descriptor)
        # A work-session grant may have another sequential call holding the
        # same input. One finished call is not that other call's completion.
        for other in journal.db.execute('SELECT id,state,request FROM executions WHERE grant_id=?',
                                        (row['grant_id'],)):
            if other['id']==execution_id or other['state'] in ('complete','failed','cancelled'):
                continue
            body = json.loads(other['request'])
            if body.get('operation')=='transcription' and body.get('body',{}).get('input')==descriptor:
                return False
        try:
            self._directory(target.parent,create=False)
            if proof is None:
                self._verify(target,descriptor)
            elif (not isinstance(proof,VerifiedInput) or proof.path!=target or
                  proof.sha256!=descriptor['sha256'] or
                  self._stamp(target.lstat())!=proof.stamp):
                return False  # the off-loop proof no longer names this copy
            target.unlink()
        except FileNotFoundError:
            return False
        self._sync_directory(target.parent)
        try:
            target.parent.rmdir()
        except OSError:
            pass  # another immutable upload in this grant is not ours to remove
        self._sync_directory(self.root)
        return True
