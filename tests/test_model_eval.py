import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from model_eval.core import compare,record_assistance
from model_eval.playground import prepare,run,load_config
from model_eval.tasks import TASKS,verify

class Playground(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.template=self.root/'template';self.template.mkdir()
  (self.template/'config.yaml').write_text('{"model":{"default":"fixture"}}')
  self.cfg={'execution_path':'native-service','model':{'name':'fixture','digest':'fixture'},'compute':{'hardware':'fixture'},
   'client':{'revision':'fixture'},'identity':{'owner_id':'user:test'},'service_id':'fixture/service','resource_id':'fixture/memory',
   'command':[sys.executable,'-c','pass'],'environment':{},'profile_template':str(self.template),'token_file':'not-used',
   'registry_url':'http://fixture','capability_checks_passed':True,'tasks':['repair-windows']}
 def tearDown(self):self.tmp.cleanup()
 def workspace(self,name):
  p=self.root/name;p.mkdir()
  for rel,text in TASKS[name]['files'].items():(p/rel).write_text(text)
  return p
 def test_operator_environment_is_not_inherited_by_agent_or_checker(self):
  script=self.root/'env-agent.py';script.write_text('import os,json;open("env.json","w").write(json.dumps(sorted(os.environ)))')
  cfg={**self.cfg,'command':[sys.executable,str(script)]}
  with patch.dict('os.environ',{'SAMCLOUD_AUTH_HDR':'private','ANTHROPIC_API_KEY':'private','CLAUDE_CODE_OAUTH_TOKEN':'private','SC_TOKEN':'private'}):
   run(cfg,self.root/'env-run',verified=('fixture-bearer',{}))
   seen=json.loads((self.root/'env-run/repair-windows/workspace/env.json').read_text())
   self.assertFalse(set(seen)&{'SAMCLOUD_AUTH_HDR','ANTHROPIC_API_KEY','CLAUDE_CODE_OAUTH_TOKEN','SC_TOKEN'})
   w=self.workspace('repair-windows');(w/'CHANGELOG.md').write_text('x')
   (w/'windows.py').write_text('import os,json;open("checker_env.json","w").write(json.dumps(sorted(os.environ)))\ndef merge_windows(w):return []')
   verify('repair-windows',w);seen=json.loads((w/'checker_env.json').read_text())
   self.assertNotIn('SC_TOKEN',seen);self.assertNotIn('ANTHROPIC_API_KEY',seen)
 def test_bearer_echo_fails_and_only_redacted_evidence_remains(self):
  script=self.root/'echo-agent.py';script.write_text('import os;v=os.environ["OPENAI_API_KEY"];open("notes.txt","w").write(v);print(v)')
  cfg={**self.cfg,'command':[sys.executable,str(script)],'token_environment':['OPENAI_API_KEY']}
  root=self.root/'secret-run';s=run(cfg,root,verified=('SECRET-FIXTURE-123',{}))
  self.assertEqual(s['tasks']['repair-windows']['result']['failure'],'secret_in_evidence')
  self.assertFalse(any(b'SECRET-FIXTURE-123' in p.read_bytes() for p in root.rglob('*') if p.is_file()))
  self.assertIn('[REDACTED]',(root/'repair-windows/agent.log').read_text())
 def test_changed_checker_refuses_resume(self):
  import model_eval.tasks as tasks
  d=self.root/'source-run';d.mkdir();prepare(d,self.cfg,{})
  def replacement(*args):return {'state':'passed'}
  with patch.object(tasks,'verify',replacement):
   with self.assertRaises(ValueError):prepare(d,self.cfg,{})
 def test_rail_label_is_refused_before_launch(self):
  cfg={**self.cfg,'execution_path':'rail-service','environment':{'X':'1'},'model_inventory_url':'http://fixture'}
  p=self.root/'rail.json';p.write_text(json.dumps(cfg))
  with self.assertRaises(ValueError):load_config(p)
  with self.assertRaises(ValueError):run(cfg,self.root/'rail-run',verified=('fixture',{}))
 def test_compare_flags_mismatches_and_unknown_provenance(self):
  a,b=self.root/'a',self.root/'b'
  run({**self.cfg,'tool_permissions':['terminal']},a,verified=('fixture-bearer',{}))
  run({**self.cfg,'tool_permissions':['terminal','network']},b,verified=('fixture-bearer',{}))
  m=json.loads((b/'manifest.json').read_text());m.pop('provenance');(b/'manifest.json').write_text(json.dumps(m))
  r=compare([a,b]);self.assertEqual(len(r['cohorts']),2)
  self.assertTrue(r['rows'][0]['comparable_to_reference']);self.assertFalse(r['rows'][1]['comparable_to_reference'])
  self.assertIn('tool_permissions',r['rows'][1]['comparison_issues']);self.assertEqual(r['rows'][1]['provenance'],'unknown')
 def test_interventions_are_unknown_until_attested_and_cannot_be_erased(self):
  d=self.root/'assist';s=run(self.cfg,d,verified=('fixture-bearer',{}));self.assertIsNone(s['tasks']['repair-windows']['operator_interventions'])
  record_assistance(d,'repair-windows','operator changed task code')
  s=json.loads((d/'state.json').read_text());self.assertEqual(s['tasks']['repair-windows']['operator_interventions'],1)
  with self.assertRaises(ValueError):record_assistance(d,'repair-windows','erase intervention',attest_zero=True)
 def test_audit_accepts_valid_subset_of_source_files(self):
  w=self.workspace('evidence-audit')
  (w/'audit.json').write_text(json.dumps({'ready':['alpha'],'retain':['beta','delta','gamma'],'reasons':{k:['observations.json'] for k in ['alpha','beta','gamma','delta']}}))
  (w/'handback.md').write_text('Headline contradicts detailed data.')
  self.assertEqual(verify('evidence-audit',w)['state'],'passed')
 def test_original_repair_is_rejected(self):
  w=self.workspace('repair-windows');(w/'CHANGELOG.md').write_text('claim fixed')
  self.assertEqual(verify('repair-windows',w)['state'],'failed')
 def test_repair_cannot_pass_by_printing_checker_verdicts_and_exiting(self):
  w=self.workspace('repair-windows');(w/'CHANGELOG.md').write_text('claim fixed')
  keys=['empty','touching','nested','negative','separate']+['invalid'+str(i) for i in range(6)]
  (w/'windows.py').write_text('import os,json\nprint('+repr(json.dumps(dict.fromkeys(keys,True)))+',flush=True)\nos._exit(0)\n')
  result=verify('repair-windows',w)
  self.assertEqual(result['state'],'failed')
  self.assertTrue(all(result['checks'][name] is False for name in keys))
 def test_repair_raw_json_does_not_hide_tuple_outputs(self):
  w=self.workspace('repair-windows');(w/'CHANGELOG.md').write_text('claim fixed')
  source='''def merge_windows(windows):
 if any(len(x)!=2 or any(type(n)!=int for n in x) or x[0]>=x[1] for x in windows):raise ValueError()
 out=[]
 for start,end in sorted(windows):
  if out and start<=out[-1][1]:out[-1][1]=max(out[-1][1],end)
  else:out.append([start,end])
 RETURN
'''
  for result_code in ('return tuple(out)','return [tuple(row) for row in out]'):
   with self.subTest(result_code=result_code):
    (w/'windows.py').write_text(source.replace('RETURN',result_code))
    self.assertEqual(verify('repair-windows',w)['state'],'failed')
 def test_correct_repair_may_print_debug_output(self):
  w=self.workspace('repair-windows');(w/'CHANGELOG.md').write_text('fixed and checked')
  source='''def merge_windows(windows):
 EMIT
 if any(len(x)!=2 or any(type(n)!=int for n in x) or x[0]>=x[1] for x in windows):raise ValueError()
 out=[]
 for start,end in sorted(windows):
  if out and start<=out[-1][1]:out[-1][1]=max(out[-1][1],end)
  else:out.append([start,end])
 return out
'''
  for emit in ("print('debug: merging windows')","__import__('sys').stdout.write('.')",
               "print('.',end='')","print('debug',file=__import__('sys').stderr)"):
   with self.subTest(emit=emit):
    (w/'windows.py').write_text(source.replace('EMIT',emit))
    self.assertEqual(verify('repair-windows',w)['state'],'passed')
 def test_repair_needs_input_immutability_and_type_validation(self):
  w=self.workspace('repair-windows');(w/'CHANGELOG.md').write_text('fixed')
  (w/'windows.py').write_text('''def merge_windows(windows):
 if any(len(x)!=2 or any(type(n)!=int for n in x) or x[0]>=x[1] for x in windows):raise ValueError()
 out=[]
 for start,end in sorted(windows):
  if out and start<=out[-1][1]:out[-1][1]=max(out[-1][1],end)
  else:out.append([start,end])
 return out
''')
  self.assertEqual(verify('repair-windows',w)['state'],'passed')
 def test_audit_correct_classification_and_sources(self):
  w=self.workspace('evidence-audit')
  (w/'audit.json').write_text(json.dumps({'ready':['alpha'],'retain':['beta','delta','gamma'],
   'reasons':{k:['policy.md','jobs.json','observations.json'] for k in ['alpha','beta','gamma','delta']}}))
  (w/'handback.md').write_text('Headline contradicts detailed evidence; inspect exact retained identities and receipt.')
  self.assertEqual(verify('evidence-audit',w)['state'],'passed')
 def test_audit_cannot_use_headline_as_proof(self):
  w=self.workspace('evidence-audit');(w/'audit.json').write_text('{"ready":["alpha","beta","delta","gamma"],"retain":[],"reasons":{}}');(w/'handback.md').write_text('all done')
  self.assertEqual(verify('evidence-audit',w)['state'],'failed')
 def test_example_only_csv_is_not_success(self):
  w=self.workspace('csv-rollup');(w/'result.json').write_text('{"totals":{"alpha":250,"beta":100},"invalid_rows":2}')
  (w/'rollup.py').write_text('import sys;open(sys.argv[2],"w").write(\'{"totals":{"alpha":250,"beta":100},"invalid_rows":2}\')')
  self.assertEqual(verify('csv-rollup',w)['state'],'failed')
 def test_successful_useful_task_and_clean_profile(self):
  (self.template/'auth.json').write_text('must not be copied')
  source="def merge_windows(windows):\n if any(len(x)!=2 or any(type(n)!=int for n in x) or x[0]>=x[1] for x in windows):raise ValueError()\n out=[]\n for a,b in sorted(windows):\n  if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])\n  else:out.append([a,b])\n return out\n"
  script=self.root/'fixture-agent.py';script.write_text('from pathlib import Path\nPath("windows.py").write_text('+repr(source)+')\nPath("CHANGELOG.md").write_text("fixed and checked")\n')
  cfg={**self.cfg,'command':[sys.executable,str(script)]}
  state=run(cfg,self.root/'success',verified=('inert',{}))
  self.assertEqual(state['state'],'complete')
  self.assertFalse((self.root/'success/repair-windows/profile/auth.json').exists())
 def test_missing_artifacts_fail_even_if_exit_zero(self):
  d=self.root/'run';s=run(self.cfg,d,verified=('inert',{}));self.assertEqual(s['state'],'failed')
  self.assertEqual(s['tasks']['repair-windows']['returncode'],0)
 def test_repeat_same_key_never_relaunches(self):
  d=self.root/'run';first=run(self.cfg,d,verified=('inert',{}));second=run(self.cfg,d,verified=('inert',{}))
  self.assertEqual(first['tasks']['repair-windows']['pid'],second['tasks']['repair-windows']['pid'])
 def test_launch_intent_is_unknown_not_permission_to_start(self):
  d=self.root/'run';d.mkdir();state,_=prepare(d,self.cfg,{})
  state['tasks']['repair-windows']={'state':'launch_intent'};(d/'state.json').write_text(json.dumps(state))
  s=run(self.cfg,d,verified=('inert',{}));self.assertEqual(s['tasks']['repair-windows'],{'state':'launch_intent'})
 def test_changed_profile_or_configuration_cannot_resume(self):
  d=self.root/'run';run(self.cfg,d,verified=('inert',{}));(self.template/'config.yaml').write_text('changed')
  with self.assertRaises(ValueError):run(self.cfg,d,verified=('inert',{}))
 def test_failure_stops_later_dispatch(self):
  cfg={**self.cfg,'tasks':['repair-windows','csv-rollup']};s=run(cfg,self.root/'run',verified=('inert',{}))
  self.assertEqual(list(s['tasks']),['repair-windows'])
 def test_compare_preserves_configuration_and_unknown_measurements(self):
  d=self.root/'run';run(self.cfg,d,verified=('inert',{}));row=compare([d])['rows'][0]
  self.assertEqual(row['configuration']['model']['digest'],'fixture');self.assertNotIn('token_file',row['configuration'])
  self.assertNotIn('memory_peak_bytes',row['result'])

AGENT="""import csv,json,re,sqlite3,subprocess,sys
from pathlib import Path
profile,grant,log=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
task='csv-rollup' if Path('ledger.csv').exists() else 'repair-windows'
open(log,'a').write('launch '+task+'\\n')
if task=='csv-rollup':
 Path('rollup.py').write_text(ROLLUP)
 subprocess.check_call([sys.executable,'rollup.py','ledger.csv','result.json'])
else:
 Path('windows.py').write_text(WINDOWS)
 Path('CHANGELOG.md').write_text('fixed and checked')
if grant!='none':
 store=profile/'samcloud-queue'/'identity.sqlite';store.parent.mkdir(parents=True,exist_ok=True)
 db=sqlite3.connect(store);db.execute('CREATE TABLE grant_binding(key TEXT PRIMARY KEY,grant_id TEXT NOT NULL,owner TEXT NOT NULL,resource TEXT NOT NULL,model TEXT NOT NULL,lifetime TEXT NOT NULL)')
 db.execute('INSERT INTO grant_binding VALUES(?,?,?,?,?,?)',('key-'+grant,grant,'hermes-obelisk','gpu','fixture','work_session'));db.commit();db.close()
"""
ROLLUP="""import csv,json,re,sys
def main(src,dst):
 totals={};bad=0
 with open(src,newline='') as f:
  for row in csv.DictReader(f):
   a,s,c=((row.get(k) or '').strip() for k in ('account','status','cents'))
   if not a or s not in ('posted','void') or not re.fullmatch('[+-]?[0-9]+',c,re.ASCII):bad+=1;continue
   if s=='posted':totals[a]=totals.get(a,0)+int(c)
 json.dump({'totals':totals,'invalid_rows':bad},open(dst,'w'))
main(sys.argv[1],sys.argv[2])
"""
WINDOWS="def merge_windows(windows):\n if any(len(x)!=2 or any(type(n)!=int for n in x) or x[0]>=x[1] for x in windows):raise ValueError()\n out=[]\n for a,b in sorted(windows):\n  if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])\n  else:out.append([list(x) for x in [[a,b]]][0])\n return out\n"

class FakeGateway:
 """Supported grant GET/release only. It has no submit: the harness cannot make a claim."""
 def __init__(self,log):
  self.log,self.calls,self.lose_release,self.reply_override={},[],set(),{}
  self.log=log
  self.grants={g:{'id':g,'owner':'hermes-obelisk','client_key':'key-'+g,'resource':'gpu','model':'fixture','state':st,
   'release_requested':0,'active_execution':None} for g,st in (('g-win','granted'),('g-csv','queued'),('g-other','granted'))}
 def __call__(self,method,url,bearer):
  from model_eval.playground import GrantReturnUnconfirmed
  assert bearer=='inert'
  path=url.split('/inference/grants/',1)[1];self.calls.append((method,path))
  gid=path.split('/')[0];grant=self.grants[gid]
  if method=='GET':return dict(grant)
  assert method=='POST' and path==gid+'/release'
  open(self.log,'a').write('release '+gid+'\n')
  grant.update(release_requested=1,state={'queued':'cancelled','granted':'released'}.get(grant['state'],grant['state']))
  if gid in self.lose_release:self.lose_release.discard(gid);raise GrantReturnUnconfirmed('POST grant reply unknown (ReadError)')
  return {**grant,**self.reply_override.get(gid,{})}

class CaseEndGrantReturn(unittest.TestCase):
 setUp_base=Playground.setUp;tearDown=Playground.tearDown
 def setUp(self):
  self.setUp_base();self.log=self.root/'order.log';self.gw=FakeGateway(self.log)
  self.script=self.root/'bind-agent.py';self.script.write_text('ROLLUP='+repr(ROLLUP)+'\nWINDOWS='+repr(WINDOWS)+'\n'+AGENT)
  self.cfg={**self.cfg,'identity':{'owner_id':'user:test','username':'hermes-obelisk'},'tasks':['repair-windows','csv-rollup'],
   'grant_return':{'identity_store':'samcloud-queue/identity.sqlite','gateway_url':'http://gateway.invalid/v1'}}
 def agent(self,grants):
  # Map each case to the grant id its fake client binds ('none' binds nothing).
  wrapper=self.root/'pick.py'
  wrapper.write_text('import os,sys,subprocess\ng=%r\ntask="csv-rollup" if os.path.exists("ledger.csv") else "repair-windows"\nsys.exit(subprocess.call([sys.executable,%r,sys.argv[1],g[task],%r]))\n'%(grants,str(self.script),str(self.log)))
  return {**self.cfg,'command':[sys.executable,str(wrapper),'{profile}']}
 def go(self,cfg,d):
  with patch('model_eval.playground.gateway',self.gw):return run(cfg,d,verified=('inert',{}))
 def order(self):return self.log.read_text().split('\n')[:-1] if self.log.exists() else []
 def test_passed_case_returns_its_exact_grant_before_next_dispatch(self):
  s=self.go(self.agent({'repair-windows':'g-win','csv-rollup':'g-csv'}),self.root/'run')
  self.assertEqual(s['state'],'complete')
  self.assertEqual(self.order(),['launch repair-windows','release g-win','launch csv-rollup','release g-csv'])
  self.assertEqual((self.gw.grants['g-win']['state'],self.gw.grants['g-csv']['state']),('released','cancelled'))
  self.assertEqual(self.gw.grants['g-other']['state'],'granted');self.assertNotIn('g-other',str(self.gw.calls))
  ret=s['tasks']['repair-windows']['grant_return'];self.assertEqual(ret['state'],'returned')
  self.assertEqual(ret['grants'][0]['ack']['observed'],'release_reply')
 def test_failed_case_keeps_its_grant_as_debt(self):
  cfg=self.agent({'repair-windows':'g-win','csv-rollup':'g-csv'});self.script.write_text(self.script.read_text().replace("Path('windows.py').write_text(WINDOWS)",'pass'))
  s=self.go(cfg,self.root/'run')
  self.assertEqual(s['state'],'failed');self.assertEqual(self.gw.calls,[]);self.assertNotIn('grant_return',s['tasks']['repair-windows'])
  self.assertEqual(self.gw.grants['g-win']['state'],'granted')
 def test_lost_release_reply_resumes_the_same_grant_without_a_new_claim(self):
  cfg=self.agent({'repair-windows':'g-win','csv-rollup':'g-csv'});d=self.root/'run';self.gw.lose_release.add('g-win')
  s=self.go(cfg,d)
  self.assertEqual(s['state'],'grant_return_unconfirmed');self.assertNotIn('csv-rollup',s['tasks'])
  self.assertEqual(json.loads((d/'state.json').read_text())['tasks']['repair-windows']['grant_return']['grants'][0]['binding']['grant_id'],'g-win')
  s=self.go(cfg,d)
  self.assertEqual(s['state'],'complete')
  self.assertEqual([c for c in self.gw.calls if c[0]=='POST'],[('POST','g-win/release'),('POST','g-csv/release')])
  self.assertEqual(s['tasks']['repair-windows']['grant_return']['grants'][0]['ack']['observed'],'terminal_without_new_request')
  self.assertEqual(self.order().count('launch repair-windows'),1)
 def test_crash_after_pass_before_intent_reads_the_same_binding_on_resume(self):
  cfg=self.agent({'repair-windows':'g-win','csv-rollup':'g-csv'});d=self.root/'run'
  def lost(method,url,bearer):
   from model_eval.playground import GrantReturnUnconfirmed
   raise GrantReturnUnconfirmed('GET grant reply unknown (URLError)')
  with patch('model_eval.playground.gateway',lost):run(cfg,d,verified=('inert',{}))
  state=json.loads((d/'state.json').read_text());state['tasks']['repair-windows'].pop('grant_return');(d/'state.json').write_text(json.dumps(state))
  s=self.go(cfg,d);self.assertEqual(s['state'],'complete');self.assertEqual(self.gw.grants['g-win']['state'],'released')
 def test_unreturnable_grants_stop_later_dispatch_without_release(self):
  for label,change in (('active',{'active_execution':'e1'}),('owner',{'owner':'someone-else'}),('model',{'model':'other'}),
                       ('key',{'client_key':'key-other'}),('state',{'state':'uncertain'})):
   with self.subTest(label):
    self.gw=FakeGateway(self.log);self.log.unlink(missing_ok=True);self.gw.grants['g-win'].update(change)
    s=self.go(self.agent({'repair-windows':'g-win','csv-rollup':'g-csv'}),self.root/('run-'+label))
    self.assertEqual(s['state'],'grant_return_unconfirmed');self.assertNotIn('csv-rollup',s['tasks'])
    self.assertFalse([c for c in self.gw.calls if c[0]=='POST'])
    self.assertEqual(s['tasks']['repair-windows']['grant_return']['state'],'unconfirmed')
 def test_release_reply_that_is_not_a_completed_return_stops(self):
  for label,override in (('still-granted',{'state':'granted'}),('active',{'active_execution':'e2'}),('not-requested',{'release_requested':0}),('other-id',{'id':'g-other'})):
   with self.subTest(label):
    self.gw=FakeGateway(self.log);self.gw.reply_override['g-win']=override
    s=self.go(self.agent({'repair-windows':'g-win','csv-rollup':'g-csv'}),self.root/('run-'+label))
    self.assertEqual(s['state'],'grant_return_unconfirmed');self.assertNotIn('csv-rollup',s['tasks'])
 def test_passed_case_without_a_binding_stops(self):
  s=self.go(self.agent({'repair-windows':'none','csv-rollup':'g-csv'}),self.root/'run')
  self.assertEqual(s['state'],'grant_return_unconfirmed');self.assertEqual(self.gw.calls,[])
  self.assertIn('identity store missing',s['tasks']['repair-windows']['grant_return']['reason'])
 def test_grant_return_configuration_is_explicit(self):
  for bad in ({'identity_store':'../x','gateway_url':'http://g/v1'},{'identity_store':'/abs','gateway_url':'http://g/v1'},
              {'identity_store':'s.sqlite','gateway_url':'file:///x'},{'identity_store':'s.sqlite'}):
   p=self.root/'c.json';p.write_text(json.dumps({**self.cfg,'grant_return':bad,'model_inventory_url':'http://fixture','environment':{'X':'1'}}))
   with self.subTest(bad),self.assertRaises(ValueError):load_config(p)
  p.write_text(json.dumps({**self.cfg,'model_inventory_url':'http://fixture','environment':{'X':'1'}}));self.assertIn('grant_return',load_config(p))

if __name__=='__main__':unittest.main()
