"""Explicit offline recovery of a proven-dead, output-free backend attempt.

No consumer route, timer, network probe or automatic retry lives here. The
operator must fence the service and positively establish backend death. Holding
the scheduler lock proves only scheduler exclusion, not backend termination.
"""
from dataclasses import dataclass, asdict
import fcntl
import hashlib
import os
from pathlib import Path
import time
import uuid

from .inference_arbiter import Arbiter, Conflict, NotFound, encoded, required


@dataclass(frozen=True)
class DeadLocalBackend:
    """Trusted operator attestation, NOT supplied by an inference consumer.

    The caller verifies that this execution used a local backend in old_boot,
    and that the same host now runs new_boot. An empty /api/ps, changed PID,
    socket failure or scheduler lock alone cannot justify this attestation.
    evidence_sha256 identifies the archived evidence, not an authenticity claim.
    """
    execution_id: str
    resource: str
    host: str
    old_boot: str
    new_boot: str
    evidence_sha256: str

    def validate(self):
        required(self.execution_id); required(self.resource)
        required(self.host)
        old, new = uuid.UUID(self.old_boot), uuid.UUID(self.new_boot)
        if old == new:
            raise Conflict('unchanged boot is not proof of local backend death')
        if (len(self.evidence_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in self.evidence_sha256)):
            raise ValueError('archived evidence SHA256 required')


@dataclass(frozen=True)
class BackendProcess:
    """Archived process identity, including birth evidence to exclude PID reuse."""
    pid: int
    uid: int
    birth_token: str
    executable: str

    def validate(self):
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError('positive backend PID required')
        if type(self.uid) is not int or self.uid < 0:
            raise ValueError('backend UID required')
        required(self.birth_token)
        if not Path(self.executable).is_absolute():
            raise ValueError('absolute backend executable required')


@dataclass(frozen=True)
class StoppedLocalBackend:
    """Trusted operator attestation of a deliberately stopped backend generation.

    This is NOT a process probe or a consumer-supplied recovery request. While
    scheduler AND backend revival are fenced, the operator archives the daemon
    and complete owned runner inventory (UID, PID, birth token, executable),
    observes every identity gone and the listener closed, and archives that
    evidence. Killing only the gateway, observing a new PID, an empty model list,
    or subsequent unrelated requests is insufficient. Unknown inventory refuses
    recovery. This permits the same host boot without weakening attempt death.
    """
    execution_id: str
    resource: str
    host: str
    boot_id: str
    processes: tuple[BackendProcess, ...]
    observed_gone: tuple[BackendProcess, ...]
    complete_inventory: bool
    revival_fenced: bool
    listener_closed: bool
    evidence_sha256: str

    def validate(self):
        required(self.execution_id); required(self.resource); required(self.host)
        uuid.UUID(self.boot_id)
        if (self.complete_inventory is not True or self.revival_fenced is not True
                or self.listener_closed is not True):
            raise Conflict('complete fenced backend termination evidence required')
        if not self.processes:
            raise Conflict('daemon and owned runner inventory required')
        for process in (*self.processes, *self.observed_gone):
            if not isinstance(process, BackendProcess):
                raise TypeError('backend process identities required')
            process.validate()
        if (len(set(self.processes)) != len(self.processes)
                or len(set(self.observed_gone)) != len(self.observed_gone)
                or set(self.processes) != set(self.observed_gone)):
            raise Conflict('every exact backend process identity must be observed gone')
        if (len(self.evidence_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in self.evidence_sha256)):
            raise ValueError('archived evidence SHA256 required')


class OfflineRecovery:
    """Exclusive journal operation; never pumps or starts a provider.

    Stop/fence automatic service revival before opening this session. The same
    lock as Supervisor prevents races with a still-running scheduler, but does
    not itself prevent a service manager from starting it after this closes.
    """
    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError('recovery requires an existing journal')
        self._lock = os.open(str(path)+'.supervisor', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if os.fstat(self._lock).st_mode & 0o077:
                raise PermissionError('supervisor lock must be private')
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.journal = Arbiter(path)
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS backend_attempt_recoveries (
                id TEXT PRIMARY KEY, execution_id TEXT NOT NULL REFERENCES executions(id),
                previous_lifecycle_seq INTEGER NOT NULL, at REAL NOT NULL,
                request_sha256 TEXT NOT NULL, operator TEXT NOT NULL, evidence TEXT NOT NULL,
                UNIQUE(execution_id,previous_lifecycle_seq))''')
            self.journal.db.execute('''CREATE TABLE IF NOT EXISTS backend_attempt_terminations (
                id TEXT PRIMARY KEY, execution_id TEXT NOT NULL REFERENCES executions(id),
                previous_lifecycle_seq INTEGER NOT NULL, at REAL NOT NULL,
                request_sha256 TEXT NOT NULL, operator TEXT NOT NULL, evidence TEXT NOT NULL,
                UNIQUE(execution_id,previous_lifecycle_seq))''')
        except BaseException:
            os.close(self._lock)
            raise

    def inspect(self, execution_id):
        """Metadata-only compare-and-swap token. Never returns the saved input."""
        row = self.journal.db.execute('SELECT e.*,g.resource,g.owner,g.release_requested,'
            'g.active_execution,g.state AS grant_state FROM executions e '
            'JOIN grants g ON g.id=e.grant_id WHERE e.id=?', (execution_id,)).fetchone()
        if row is None:
            raise NotFound()
        seq = self.journal.db.execute('SELECT MAX(seq) FROM execution_lifecycle WHERE execution_id=?',
                                      (execution_id,)).fetchone()[0]
        return dict(execution_id=row['id'], grant_id=row['grant_id'], resource=row['resource'],
            state=row['state'], grant_state=row['grant_state'], updated=row['updated'],
            active_execution=row['active_execution'], release_requested=bool(row['release_requested']),
            request_sha256=hashlib.sha256(row['request'].encode()).hexdigest(),
            lifecycle_seq=seq, has_result=row['result'] is not None,
            output_events=self.journal.db.execute('SELECT COUNT(*) FROM execution_events '
                'WHERE execution_id=?',(execution_id,)).fetchone()[0])

    def retry_dead_attempt(self, expected, *, proof, operator):
        """Retain logical ID/input; audit old attempt death before making it queued.

        Neither age nor an empty output stream establishes death. The separate
        trusted proof is mandatory. Any persisted output/result or user release
        rejects retry, even when the previous host demonstrably rebooted.
        """
        if not isinstance(proof, (DeadLocalBackend, StoppedLocalBackend)):
            raise TypeError('trusted local-backend death attestation required')
        proof.validate(); required(operator)
        with self.journal.transaction():
            current = self.inspect(expected['execution_id'])
            if current != expected:
                raise Conflict('execution changed after recovery inspection')
            if proof.execution_id != current['execution_id'] or proof.resource != current['resource']:
                raise Conflict('backend death attestation names different work or resource')
            if (current['state'] != 'uncertain' or current['grant_state'] != 'uncertain' or
                    current['active_execution'] != current['execution_id'] or
                    current['has_result'] or current['output_events'] or current['release_requested']):
                raise Conflict('only owned, unreleased, output-free uncertain work may retry')
            now = time.time(); recovery_id = uuid.uuid4().hex
            self.journal.db.execute('INSERT INTO backend_attempt_recoveries VALUES(?,?,?,?,?,?,?)',
                (recovery_id,current['execution_id'],current['lifecycle_seq'],now,
                 current['request_sha256'],operator,
                 encoded({'attestation_type': type(proof).__name__, **asdict(proof)})))
            self.journal.db.execute("UPDATE executions SET state='queued',updated=?,wait_reason=NULL WHERE id=?",
                                    (now,current['execution_id']))
            self.journal.db.execute("UPDATE grants SET state='queued',active_execution=NULL,updated=?,"
                'queue_order=?,stint_started=NULL,idle_since=NULL WHERE id=?',
                (now,self.journal._next_order(),current['grant_id']))
        return recovery_id

    def fail_dead_attempt(self, expected, *, proof, operator, code):
        """Audit an operator-proven dead attempt as failed, without redispatch.

        The proof and CAS requirements match retry_dead_attempt. This is not a
        consumer cancellation or an inference-error shortcut. Physical claims
        and residency still require their separate exact settlement receipts.
        The fixed result is a provider failure, never a model-quality verdict.
        """
        if not isinstance(proof, (DeadLocalBackend, StoppedLocalBackend)):
            raise TypeError('trusted local-backend death attestation required')
        proof.validate(); required(operator)
        if code != 'backend_attempt_dead':
            raise ValueError('fixed operator failure code required')
        with self.journal.transaction():
            current = self.inspect(expected['execution_id'])
            if current != expected:
                raise Conflict('execution changed after recovery inspection')
            if proof.execution_id != current['execution_id'] or proof.resource != current['resource']:
                raise Conflict('backend death attestation names different work or resource')
            if (current['state'] != 'uncertain' or current['grant_state'] != 'uncertain' or
                    current['active_execution'] != current['execution_id'] or
                    current['has_result'] or current['output_events'] or current['release_requested']):
                raise Conflict('only owned, unreleased, output-free uncertain work may terminate')
            grant = self.journal.db.execute('SELECT * FROM grants WHERE id=?',
                                            (current['grant_id'],)).fetchone()
            now = time.time(); termination_id = uuid.uuid4().hex
            result = encoded({'kind': 'error', 'status': 502, 'code': code,
                'type': 'provider_error',
                'detail': 'Backend attempt ended without a result (operator recovery '
                          + termination_id + ')'})
            self.journal.db.execute('INSERT INTO backend_attempt_terminations VALUES(?,?,?,?,?,?,?)',
                (termination_id,current['execution_id'],current['lifecycle_seq'],now,
                 current['request_sha256'],operator,
                 encoded({'attestation_type': type(proof).__name__, **asdict(proof)})))
            self.journal.db.execute("UPDATE executions SET state='failed',result=?,updated=?,"
                                    'wait_reason=NULL WHERE id=?',
                                    (result,now,current['execution_id']))
            released = bool(grant['release_requested'] or grant['lifetime'] == 'execution')
            self.journal.db.execute("UPDATE grants SET state=?,active_execution=NULL,updated=?,"
                                    'idle_since=? WHERE id=?',
                                    ('released' if released else 'granted',now,
                                     None if released else now,current['grant_id']))
        return termination_id

    def close(self):
        self.journal.close()
        os.close(self._lock)
