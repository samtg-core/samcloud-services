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
        if not isinstance(proof, DeadLocalBackend):
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
                 current['request_sha256'],operator,encoded(asdict(proof))))
            self.journal.db.execute("UPDATE executions SET state='queued',updated=?,wait_reason=NULL WHERE id=?",
                                    (now,current['execution_id']))
            self.journal.db.execute("UPDATE grants SET state='queued',active_execution=NULL,updated=?,"
                'queue_order=?,stint_started=NULL,idle_since=NULL WHERE id=?',
                (now,self.journal._next_order(),current['grant_id']))
        return recovery_id

    def close(self):
        self.journal.close()
        os.close(self._lock)
