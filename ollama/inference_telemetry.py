"""Committed metadata transitions, never prompts/results or an inference control.

The SQLite rows are authoritative. A bounded notification file wakes filesystem
subscribers after commit; its contents are only a cursor, never event history.
Observers replay rows by sequence after reconnect, rather than sample state.
"""
import logging
import os
from pathlib import Path
import struct


def install(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS metadata_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL, at REAL NOT NULL,
            owner TEXT, resource TEXT, model TEXT,
            grant_id TEXT, execution_id TEXT,
            state TEXT, reason TEXT, active_execution TEXT);
        CREATE TRIGGER IF NOT EXISTS metadata_execution_insert AFTER INSERT ON executions
        BEGIN
            INSERT INTO metadata_events(kind,at,owner,resource,model,grant_id,execution_id,state,reason)
            SELECT 'execution',NEW.created,g.owner,g.resource,g.model,g.id,NEW.id,NEW.state,NEW.wait_reason
            FROM grants g WHERE g.id=NEW.grant_id;
        END;
        CREATE TRIGGER IF NOT EXISTS metadata_execution_update AFTER UPDATE ON executions
        WHEN NEW.state IS NOT OLD.state OR NEW.wait_reason IS NOT OLD.wait_reason
        BEGIN
            INSERT INTO metadata_events(kind,at,owner,resource,model,grant_id,execution_id,state,reason)
            SELECT 'execution',NEW.updated,g.owner,g.resource,g.model,g.id,NEW.id,NEW.state,NEW.wait_reason
            FROM grants g WHERE g.id=NEW.grant_id;
        END;
        CREATE TRIGGER IF NOT EXISTS metadata_grant_insert AFTER INSERT ON grants
        BEGIN
            INSERT INTO metadata_events(kind,at,owner,resource,model,grant_id,state,reason,active_execution)
            VALUES('grant',NEW.created,NEW.owner,NEW.resource,NEW.model,NEW.id,NEW.state,
                   NEW.handoff_reason,NEW.active_execution);
        END;
        CREATE TRIGGER IF NOT EXISTS metadata_grant_update AFTER UPDATE ON grants
        WHEN NEW.state IS NOT OLD.state OR NEW.active_execution IS NOT OLD.active_execution
          OR NEW.handoff_reason IS NOT OLD.handoff_reason OR NEW.yield_requested IS NOT OLD.yield_requested
          OR NEW.release_requested IS NOT OLD.release_requested
        BEGIN
            INSERT INTO metadata_events(kind,at,owner,resource,model,grant_id,state,reason,active_execution)
            VALUES('grant',NEW.updated,NEW.owner,NEW.resource,NEW.model,NEW.id,NEW.state,
                   CASE WHEN NEW.yield_requested=1 THEN 'yield_requested'
                        WHEN NEW.release_requested=1 THEN 'release_requested'
                        ELSE NEW.handoff_reason END,NEW.active_execution);
        END;
    ''')


def notify(path, sequence):
    """Best-effort wake only; failure cannot relabel already-committed work."""
    try:
        descriptor=os.open(str(Path(path).with_suffix('.events-notify')),
                           os.O_CREAT|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        try:
            if os.fstat(descriptor).st_mode & 0o077:
                raise PermissionError('event notification must be private')
            os.write(descriptor,struct.pack('>Q',sequence))
        finally:
            os.close(descriptor)
    except OSError:
        logging.getLogger(__name__).warning('Event notification unavailable; durable event rows retained')
