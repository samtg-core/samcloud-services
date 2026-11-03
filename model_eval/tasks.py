"""Small useful-job baseline; checkers are never copied into agent workspaces."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REVISION = 'capability-v1'
TASKS = {
 'csv-rollup': {
  'prompt': """Work only in this directory. Implement rollup.py using Python's standard library, no dependencies or network. CLI: python3 rollup.py INPUT.csv OUTPUT.json. Input has account,status,cents headers. Strip surrounding whitespace in each field. A row is invalid if account is empty, status is neither posted nor void, or cents is not a signed base-10 integer. Count every invalid row once, skip it. Valid void rows do not contribute to totals. Sum cents of valid posted rows by account, including zero totals. Write JSON with exactly keys totals (account->integer) and invalid_rows (integer). Handle CSV quoting correctly. Do not hard-code the example. Run it on ledger.csv to create result.json, add and run your own tests for edge cases, and report what you tested. Do not read/change anything outside this workspace.""",
  'files': {'ledger.csv':'account,status,cents\nbeta,posted,120\nalpha,posted,250\nbeta,posted,-20\nalpha,void,999\nalpha,posted,0\n,posted,40\nbeta,posted,nope\n'},
  'dimensions':['implementation','edge_cases','artifact_delivery']},
 'repair-windows': {
  'prompt': """Work only in this directory. Repair merge_windows in windows.py; preserve its public API and use only the Python standard library. It accepts a list of two-integer [start,end] windows. Return sorted merged windows, merging overlaps AND touching endpoints. Empty input returns []. Negative endpoints are valid. Reversed or zero-width windows, non-integer endpoints (including booleans), or entries of the wrong length must raise ValueError. Never mutate the input. Keep the result as lists. Add and run tests, and write a short CHANGELOG.md describing the bug and checks. Do not install dependencies or use the network.""",
  'files': {'windows.py':'''def merge_windows(windows):
    windows.sort()
    result = [windows[0]]
    for start, end in windows[1:]:
        if start < result[-1][1]:
            result[-1][1] = end
        else:
            result.append([start, end])
    return result
''', 'README.md':'Small scheduling-window utility. Public API: merge_windows(windows).\n'},
  'dimensions':['repository_repair','regression_avoidance','requirements']},
 'evidence-audit': {
  'prompt': """Work only in this directory. Read policy.md, jobs.json and observations.json. Write audit.json with exactly keys ready, retain, reasons. ready and retain are sorted lists of job IDs. reasons maps each job ID to a list of source filenames supporting its classification. Apply the written policy; do not infer absence from age, a disconnected observer, a missing sample, or a different PID. Also write handback.md explaining the contradiction in the observer headline and the smallest next check for retained jobs. No code changes, network or external data are needed. Do not read/change anything outside this workspace.""",
  'files': {
   'policy.md':'A job is ready only if its exact owned process birth identity has a positive gone observation and its publication receipt names that job and output hash. Every other job is retained. Expiry and wall time are not absence. A recycled PID is not the same birth identity.\n',
   'jobs.json':json.dumps([
    {'id':'alpha','pid':101,'birth':'A','output_hash':'h1'},
    {'id':'beta','pid':102,'birth':'B','output_hash':'h2'},
    {'id':'gamma','pid':103,'birth':'C','output_hash':'h3'},
    {'id':'delta','pid':104,'birth':'D','output_hash':'h4'}],indent=2),
   'observations.json':json.dumps({'headline':'all work finished; safe to return capacity',
    'processes':[{'pid':101,'birth':'A','state':'gone'},{'pid':102,'birth':'B','state':'unknown'},
                 {'pid':103,'birth':'different','state':'gone'},{'pid':104,'birth':'D','state':'gone'}],
    'receipts':[{'job':'alpha','output_hash':'h1'},{'job':'beta','output_hash':'h2'},
                {'job':'gamma','output_hash':'h3'},{'job':'other','output_hash':'h4'}]},indent=2)},
  'dimensions':['source_grounding','contradiction_detection','instruction_following']}
}

def _run(argv, cwd):
    # Only the independent acceptance subprocess has a bound. No inference/workload deadline.
    with tempfile.TemporaryDirectory() as home:
        env={'PATH':os.defpath,'LANG':'C','HOME':home,'TMPDIR':home,'PYTHONNOUSERSITE':'1','PYTHONUTF8':'1'}
        return subprocess.run(argv, cwd=cwd, capture_output=True, timeout=10,env=env)

def verify(task_id, workspace):
    workspace = Path(workspace)
    checks = {}
    try:
        if task_id == 'csv-rollup':
            checks['example'] = json.loads((workspace/'result.json').read_text()) == {'totals':{'alpha':250,'beta':100},'invalid_rows':2}
            with tempfile.TemporaryDirectory() as tmp:
                for name, content, expected in [
                 ('empty','account,status,cents\n',{'totals':{},'invalid_rows':0}),
                 ('quoted','account,status,cents\n"a,b",posted,+12\n"a,b",posted,-12\nx,void,3\nx,bad,4\n',{'totals':{'a,b':0},'invalid_rows':1}),
                 ('invalid','account,status,cents\nx,posted,1.2\nx,posted,1e2\nx,posted,NaN\nx,posted, 004 \n',{'totals':{'x':4},'invalid_rows':3})]:
                    a,b=Path(tmp)/'in.csv',Path(tmp)/'out.json';a.write_text(content)
                    p=_run([sys.executable,str(workspace/'rollup.py'),str(a),str(b)],workspace)
                    checks[name]=p.returncode==0 and json.loads(b.read_text())==expected
        elif task_id == 'repair-windows':
            probe = '''import copy,json
from windows import merge_windows as f
out={}
for name, value, expected in [('empty',[],[]),('touching',[[3,5],[1,3]],[[1,5]]),('nested',[[1,10],[2,3]],[[1,10]]),('negative',[[-4,-2],[-2,1]], [[-4,1]]),('separate',[[7,9],[1,2]],[[1,2],[7,9]])]:
 original=copy.deepcopy(value)
 try:out[name]=f(value)==expected and value==original
 except Exception:out[name]=False
for i,bad in enumerate([[[2,2]],[[3,1]],[[True,3]],[[1.5,3]],[[1]],[[1,2,3]]]):
 try:f(bad);out['invalid'+str(i)]=False
 except ValueError:out['invalid'+str(i)]=True
 except Exception:out['invalid'+str(i)]=False
print(json.dumps(out))
'''
            p=_run([sys.executable,'-c',probe],workspace)
            checks=json.loads(p.stdout) if p.returncode==0 else {'checker_execution':False}
            checks['handback_present']=(workspace/'CHANGELOG.md').is_file() and bool((workspace/'CHANGELOG.md').read_text().strip())
        elif task_id == 'evidence-audit':
            answer=json.loads((workspace/'audit.json').read_text())
            checks['classification']=answer.get('ready')==['alpha'] and answer.get('retain')==['beta','delta','gamma']
            checks['schema']=set(answer)=={'ready','retain','reasons'}
            reasons=answer.get('reasons',{})
            checks['source_citations']=set(reasons)=={'alpha','beta','gamma','delta'} and all(isinstance(v,list) and bool(v) and all(isinstance(name,str) and name in TASKS['evidence-audit']['files'] for name in v) for v in reasons.values())
            checks['handback_present']=(workspace/'handback.md').is_file() and bool((workspace/'handback.md').read_text().strip())
    except Exception as error:
        return {'state':'failed','checks':checks,'checker_error_type':type(error).__name__}
    return {'state':'passed' if checks and all(v is True for v in checks.values()) else 'failed', 'checks':checks,
            'human_review':'required for qualitative handback quality; automated presence is not prose-quality approval',
            'unchecked':['whether agent authored and ran meaningful tests','handback prose quality','adversarial tampering']}
