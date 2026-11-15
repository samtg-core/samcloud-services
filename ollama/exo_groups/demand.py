"""Wake acquisition from saved calls before capacity-gated logical dispatch."""

from ..inference_arbiter import Conflict, NotFound
from .acquire import GroupAcquisition


class AcquisitionDemand:
    """Explicit provider installation on the existing supervisor, not startup.

    All native/compatibility enqueue paths already call pump(). The hook runs
    before its capacity check, so waiting for EXO capacity cannot prevent the
    acquisition that establishes it. It links only new saved calls; capacity
    wakeups do not invalidate/restart an already-linked actor in a loop.
    """

    def __init__(self, supervisor, actors):
        self.supervisor = supervisor
        self.actors = dict(actors)
        if not self.actors or supervisor.prepare_demand is not None:
            raise ValueError('demand preparation already installed')
        for model, actor in self.actors.items():
            if (not isinstance(actor, GroupAcquisition) or actor.plan.model_id != model
                    or actor.intents.journal is not supervisor.journal
                    or actor._task is not None or actor._closed):
                raise ValueError('fresh exact model actors on this supervisor required')
        # Metadata operations are owned/drained by this same supervisor too.
        for actor in self.actors.values():
            actor.task_factory = supervisor.prepare
        supervisor.prepare_demand = self.prepare

    def prepare(self, resource):
        journal = self.supervisor.journal
        rows = journal.db.execute('''SELECT x.id,g.model FROM executions x
            JOIN grants g ON g.id=x.grant_id
            LEFT JOIN exo_group_demands d ON d.execution_id=x.id
            WHERE g.resource=? AND x.state='queued' AND d.execution_id IS NULL
            AND g.state IN ('queued','granted') AND g.release_requested=0''',
            (resource,)).fetchall()
        for row in rows:
            actor = self.actors.get(row['model'])
            if actor is None:
                continue
            try:
                actor.note_demand(row['id'])
            except (Conflict, NotFound):
                # Another local connection can cancel after the read. Ignore
                # only a now-ended call; a live configuration conflict blocks.
                current = journal.db.execute('SELECT state FROM executions WHERE id=?',
                    (row['id'],)).fetchone()
                if current is not None and current['state'] == 'queued':
                    raise
