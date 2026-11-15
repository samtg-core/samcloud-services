import argparse,json
from .core import compare,record_assistance
from .playground import load_config,run
p=argparse.ArgumentParser(description='Useful-agent capability playground, configured through model-services')
s=p.add_subparsers(dest='command',required=True)
r=s.add_parser('run');r.add_argument('--config',required=True);r.add_argument('--run-dir',required=True)
r=s.add_parser('compare');r.add_argument('run_dirs',nargs='+')
r=s.add_parser('assistance');r.add_argument('--run-dir',required=True);r.add_argument('--task',required=True);r.add_argument('--note',required=True);r.add_argument('--attest-zero',action='store_true')
a=p.parse_args()
try:
 result=(compare(a.run_dirs) if a.command=='compare' else record_assistance(a.run_dir,a.task,a.note,attest_zero=a.attest_zero) if a.command=='assistance' else run(load_config(a.config),a.run_dir))
 print(json.dumps(result if a.command!='run' else {'state':result['state'],'run_id':result['run_id']},indent=2))
 if a.command=='run' and result['state']!='complete':raise SystemExit(1)
except Exception as e:
 print(json.dumps({'state':'observer_stopped','error_type':type(e).__name__,'action':'Inspect the saved exact operation; do not relaunch it from this error.'}));raise SystemExit(2)
