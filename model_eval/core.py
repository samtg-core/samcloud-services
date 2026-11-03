import hashlib
import json
import os
from pathlib import Path

SCHEMA='capability-playground/v1'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write_json(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    with temp.open('w') as f:
        os.chmod(temp,0o600);json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    temp.replace(path);fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def compare(paths):
    rows=[]
    for path in paths:
        root=Path(path);m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'state.json').read_text())
        for name,task in s['tasks'].items():
            rows.append({'run_id':s['run_id'],'task':name,'fixture_revision':m['fixture_revision'],
                'fixture_hash':m['fixture_hash'],'configuration':m['configuration'],'state':task['state'],
                'result':task.get('result'),'elapsed_s':task.get('elapsed_s'),'provenance':m.get('provenance','fresh')})
    return {'schema':SCHEMA,'rows':rows,'comparison_rule':'Match fixture hash, tool permissions and assistance policy. No aggregate ranking or speed/health proxy for capability.'}
