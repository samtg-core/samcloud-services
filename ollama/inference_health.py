"""Read-only operator report: completion cohorts, not occupancy-as-success.

No HTTP endpoint, worker changes, prompts or response bodies. A request is one
durable execution, not an HTTP retry, tool, native turn or publication. Windows
are [start,end), observed as of a separately stated cutoff. Open work is censored,
never silently excluded from the completion denominator or called a failure.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3


def percentage(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def distribution(values):
    ordered = sorted(values)
    return {'count':len(ordered), 'total':sum(ordered),
            'median':ordered[(len(ordered)-1)//2] if ordered else None,
            'max':ordered[-1] if ordered else None}


def summarize(db, *, start, end, as_of):
    if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in (start, end, as_of)) or not start < end <= as_of):
        raise ValueError('finite start < end <= as_of required')
    # One SQLite read snapshot; do not mix a new state with an older event list.
    db.execute('BEGIN')
    try:
        rows = db.execute('SELECT e.id,e.created,g.id,g.resource FROM executions e '
                          'JOIN grants g ON g.id=e.grant_id WHERE e.created<?', (end,)).fetchall()
        events = db.execute('SELECT execution_id,state,at,seq FROM execution_lifecycle '
                            'WHERE at<=? ORDER BY seq', (as_of,)).fetchall()
    finally:
        db.rollback()
    history = defaultdict(list)
    for execution, state, at, seq in events:
        history[execution].append((state, at, seq))
    states, carry_in, waits, service_time = Counter(), Counter(), [], []
    cohort_ids, resources = set(), {}
    for execution, created, grant, resource in rows:
        transitions = history[execution]
        state = transitions[-1][0] if transitions else 'unobserved'
        if created < start:
            before = [kind for kind, at, _ in transitions if at < start]
            if not before or before[-1] not in ('complete', 'failed', 'cancelled'):
                carry_in[state] += 1
            continue
        cohort_ids.add(execution)
        resources[execution] = (resource, grant)
        states[state] += 1
        # Missing historical insertion is not a zero-wait measurement.
        if transitions and transitions[0][0] in ('queued', 'running') and transitions[0][1] == created:
            started = next((at for kind, at, _ in transitions if kind == 'running'), None)
            if started is not None and started >= created:
                # A confirmed pre-provider refusal can return this SAME job to
                # queue. Count subsequent completed waits, not new requests or
                # provider service time. An ongoing queued interval is censored.
                queued_at = created
                completed_wait = 0
                occupied = 0
                for index, (kind, at, _) in enumerate(transitions):
                    if kind == 'queued':
                        queued_at = at
                    elif kind == 'running' and queued_at is not None:
                        completed_wait += max(0, at-queued_at)
                        queued_at = None
                    if kind in ('running', 'uncertain') and index+1 < len(transitions):
                        occupied += max(0, transitions[index+1][1]-at)
                waits.append(completed_wait)
                if state in ('complete', 'failed') and transitions[-1][1] >= started:
                    service_time.append(occupied)
    total = sum(states.values())
    terminal = sum(states[s] for s in ('complete', 'failed', 'cancelled'))
    # Switching conversations is measurable; its cache/prefill penalty is NOT
    # inferred from a switch count. Provider usage requires a separate exact-ID join.
    dispatches = defaultdict(list)
    for execution, state, at, seq in events:
        if execution in cohort_ids and state == 'running':
            resource, grant = resources[execution]
            dispatches[resource].append((seq, grant))
    transitions = switches = 0
    for sequence in dispatches.values():
        for previous, current in zip(sequence, sequence[1:]):
            transitions += 1
            switches += previous[1] != current[1]
    return {
        'schema':1, 'window':{'start':start, 'end_exclusive':end, 'as_of':as_of},
        'unit':'durable service execution accepted in the window',
        'requests':{'accepted':total, 'states':dict(states),
            'completed_percent':percentage(states['complete'], total),
            'terminal_percent':percentage(terminal, total),
            'success_among_terminal_percent':percentage(states['complete'], terminal),
            'open_or_unobserved':total-terminal},
        'carry_in_states_as_of':dict(carry_in),
        'queue_wait_seconds_dispatched_only':distribution(waits),
        'service_elapsed_seconds_confirmed_ended_only':distribution(service_time),
        'interleaving':{'within_cohort_adjacent_dispatch_pairs':transitions,
                        'different_grant_pairs':switches,
                        'cache_prefill_cost':None},
        'native_turn_completion_percent':None,
        'watchdog_cancellations':None,
        'limits':[
            'Queued, running and uncertain work remains in the denominator; not failed by age.',
            'Carry-in work is reported separately, not mixed into newly accepted requests.',
            'No synthetic history for pre-instrumentation requests; unobserved means unknown.',
            'Service elapsed includes uncertainty and is not GPU compute time.',
            'Native turns, watchdog cancellations, publications and provider cache usage need their own exact-ID evidence.',
            'Native/tool progress and device occupancy are not service completion or useful-output acceptance.',
        ]}


def read_report(path, **bounds):
    # mode=ro must fail for a missing journal, never create one or install schema.
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        return summarize(db, **bounds)


def instant(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('explicit time zone required')
    return parsed.timestamp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', required=True, type=Path)
    for name in ('start', 'end', 'as-of'):
        parser.add_argument('--'+name, required=True, type=instant)
    args = parser.parse_args()
    print(json.dumps(read_report(args.journal, start=args.start, end=args.end,
                                as_of=args.as_of), indent=2, allow_nan=False))
