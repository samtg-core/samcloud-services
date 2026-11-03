import json
from pathlib import Path
import sys
import tempfile
import unittest
from model_eval.core import compare
from model_eval.playground import prepare,run
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
 def test_original_repair_is_rejected(self):
  w=self.workspace('repair-windows');(w/'CHANGELOG.md').write_text('claim fixed')
  self.assertEqual(verify('repair-windows',w)['state'],'failed')
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

if __name__=='__main__':unittest.main()
