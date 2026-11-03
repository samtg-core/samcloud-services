import argparse,json
from .core import compare
from .playground import load_config,run
p=argparse.ArgumentParser(description='Useful-agent capability playground, configured through model-services')
s=p.add_subparsers(dest='command',required=True)
r=s.add_parser('run');r.add_argument('--config',required=True);r.add_argument('--run-dir',required=True)
r=s.add_parser('compare');r.add_argument('run_dirs',nargs='+')
a=p.parse_args()
try:
 result=compare(a.run_dirs) if a.command=='compare' else run(load_config(a.config),a.run_dir)
 print(json.dumps(result if a.command=='compare' else {'state':result['state'],'run_id':result['run_id']},indent=2))
 if a.command=='run' and result['state']!='complete':raise SystemExit(1)
except Exception as e:
 print(json.dumps({'state':'observer_stopped','error_type':type(e).__name__,'action':'Inspect the saved exact operation; do not relaunch it from this error.'}));raise SystemExit(2)
