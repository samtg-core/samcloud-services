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
    rows=[];cohorts={};reference=None;reference_values=None
    for path in paths:
        root=Path(path);m=json.loads((root/'manifest.json').read_text());s=json.loads((root/'state.json').read_text())
        cohort_values={'fixture_hash':m.get('fixture_hash'),'source_hashes':m.get('source_hashes'),
                       'tool_permissions':m.get('configuration',{}).get('tool_permissions'),
                       'assistance_policy':m.get('assistance_policy')}
        cohort=digest(cohort_values)
        if reference is None:reference,reference_values=cohort,cohort_values
        complete=all(value is not None for value in cohort_values.values())
        cohorts.setdefault(cohort,{'run_ids':[],'criteria':cohort_values})['run_ids'].append(s['run_id'])
        issues=[k for k,v in cohort_values.items() if v!=reference_values[k] or v is None]
        for name,task in s['tasks'].items():
            rows.append({'run_id':s['run_id'],'task':name,'fixture_revision':m['fixture_revision'],
                'fixture_hash':m['fixture_hash'],'configuration':m['configuration'],'state':task['state'],
                'result':task.get('result'),'elapsed_s':task.get('elapsed_s'),'provenance':m.get('provenance','unknown'),
                'cohort':cohort,'comparable_to_reference':complete and cohort==reference,
                'comparison_issues':issues,'operator_interventions':task.get('operator_interventions') if task.get('assistance_events') else None,
                'assistance_attested':bool(task.get('assistance_events'))})
    return {'schema':SCHEMA,'rows':rows,'cohorts':cohorts,'reference_cohort':reference,
            'comparison_rule':'First run selects reference cohort. Missing provenance is unknown; unattended status needs operator attestation. No aggregate ranking.'}


def record_assistance(directory,task,note,*,attest_zero=False):
    import fcntl,time
    root=Path(directory)
    if not note.strip():raise ValueError('an operator note is required')
    with (root/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'state.json').read_text());row=state['tasks'][task]
        events=row.setdefault('assistance_events',[])
        if attest_zero and (row.get('operator_interventions') or any(e['kind']=='intervention' for e in events)):
            raise ValueError('cannot erase recorded interventions')
        events.append({'kind':'attest_zero' if attest_zero else 'intervention','note':note,'at':time.time()})
        row['operator_interventions']=0 if attest_zero else sum(e['kind']=='intervention' for e in events)
        if row.get('result') is not None:row['result']['unattended']=attest_zero and row['state']=='passed'
        write_json(root/'state.json',state)
        return {'task':task,'operator_interventions':row['operator_interventions']}
