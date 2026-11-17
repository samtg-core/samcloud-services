"""Finite unprivileged queue writer barrier; pipe EOF resumes the exact gateway."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.parse


def identity(fence):
    value=subprocess.check_output(['/bin/ps','-p',str(fence['pid']),'-o','uid=,lstart=,comm='],text=True).strip()
    if value!=fence['identity'] or value.split()[0]!=str(os.getuid()):
        raise ValueError('owned gateway identity changed')
    listener=subprocess.run(['/usr/sbin/lsof','-nP','-t','-iTCP:8800','-sTCP:LISTEN'],capture_output=True,text=True)
    if listener.returncode!=0 or listener.stderr.strip() or set(listener.stdout.split())!={str(fence['pid'])}:
        raise ValueError('owned gateway listener changed')
    return value


def run(value):
    uid,fence=value['uid'],value['fence']
    if os.getuid()!=uid or os.geteuid()!=uid or uid<=0:
        raise ValueError('must run as the actual service user')
    journal=Path(fence['journal']);before=journal.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_uid!=uid or before.st_nlink!=1
            or [before.st_dev,before.st_ino]!=fence['journal_inode']):
        raise ValueError('owned queue journal changed')
    connection=sqlite3.connect('file:'+urllib.parse.quote(str(journal),safe='/')+'?mode=rw',uri=True,timeout=10)
    stopped=False
    try:
        if connection.execute('PRAGMA journal_mode').fetchone()[0]!='delete':raise ValueError('rollback journal required')
        connection.execute('BEGIN IMMEDIATE')
        if connection.execute("SELECT 1 FROM executions WHERE state NOT IN ('complete','failed','cancelled') LIMIT 1").fetchone():
            raise ValueError('ordinary work has not drained')
        held=connection.execute("SELECT state,active_execution,idle_since FROM grants WHERE state IN ('granted','uncertain')").fetchall()
        if any(state!='granted' or active is not None or idle is None or idle>time.time()-120 for state,active,idle in held):
            raise ValueError('seat turn is still active')
        identity(fence);os.kill(fence['pid'],signal.SIGSTOP);stopped=True
        identity(fence)
        status=subprocess.check_output(['/bin/ps','-p',str(fence['pid']),'-o','stat='],text=True).strip()
        if 'T' not in status:raise ValueError('gateway stop not observed')
        current=journal.lstat()
        if (current.st_dev,current.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('journal replaced')
        print(json.dumps({'state':'frozen','pid':fence['pid'],'identity':fence['identity'],'uid':uid,
                          'queue_writer_barrier':True,'journal_inode':[current.st_dev,current.st_ino]}),flush=True)
        # One controller owns this pipe. EOF is cleanup, never a compute retry.
        command=sys.stdin.buffer.readline(64)
        if command not in (b'',b'release\n'):raise ValueError('only release or EOF permitted')
    finally:
        try:
            if stopped:
                identity(fence)
                status=subprocess.check_output(['/bin/ps','-p',str(fence['pid']),'-o','stat='],text=True).strip()
                if 'T' in status:os.kill(fence['pid'],signal.SIGCONT)
        finally:
            connection.rollback();connection.close()
    print(json.dumps({'state':'resumed','pid':fence['pid'],'identity':fence['identity'],'writer_barrier_released':True}),flush=True)


if __name__=='__main__':
    line=sys.stdin.buffer.readline(16385)
    if len(line)>16384:raise SystemExit('bounded fence data required')
    run(json.loads(line))
