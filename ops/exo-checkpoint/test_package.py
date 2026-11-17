"""Qualification and code/data boundaries before any packet directory exists."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
class Package(unittest.TestCase):
 def setUp(self):
  p=Path(__file__).with_name('package.py');s=importlib.util.spec_from_file_location('checkpoint_package',p);self.m=importlib.util.module_from_spec(s);s.loader.exec_module(self.m)
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name);self.remote=str(self.m.BASE/'stager-v1')
  self.plan={'model_id':self.m.MODEL,'revision':self.m.REVISION,'uid':502,'destination':str(self.m.BASE/'checkpoint-v1'),'small_gate_sha256':'0'*64,'small_gate':{'phase':'released','positive_cleanup_and_exact_settlement':True,'three_cases_16_raw_ids_each_rank':True,'cleanup_receipts':[{}, {}, {}]},'weight_files':[{'name':f'model-{i:05d}-of-00037.safetensors','bytes':1,'sha256':'0'*64} for i in range(1,38)],'metadata_files':[{'name':'config.json','bytes':1,'git_blob_sha1':'0'*40}],'weights_bytes':37}
 def test_completed_gate_builds_bound_read_only_packet(self):
  result=self.m.build(self.plan,self.remote,self.root/'packet');plan=(self.root/'packet/plan.json').read_bytes();script=(self.root/'packet/download.py').read_text()
  self.assertEqual(json.loads(plan),self.plan);self.assertIn(result['plan_sha256'],script);self.assertIn(self.remote,script);self.assertNotIn('REPLACE_',script);compile(script,'<packaged>','exec');self.assertEqual((self.root/'packet/download.py').stat().st_mode&0o777,0o400)
 def test_unreleased_or_incomplete_gate_refuses_before_directory_creation(self):
  for field,value in (('phase','serving'),('positive_cleanup_and_exact_settlement',False),('three_cases_16_raw_ids_each_rank','true'),('cleanup_receipts',[{},{}])):
   plan=json.loads(json.dumps(self.plan));plan['small_gate'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):self.m.build(plan,self.remote,self.root/'refused')
   self.assertFalse((self.root/'refused').exists())
 def test_path_injection_or_file_overlap_is_refused_before_any_output(self):
  for path in (str(self.m.BASE/"x');raise RuntimeError('injection')#"),str(self.m.BASE/'..'/'elsewhere')):
   with self.subTest(path=path),self.assertRaises(ValueError):self.m.build(self.plan,path,self.root/'refused')
   self.assertFalse((self.root/'refused').exists())
  self.plan['metadata_files'][0]['name']=self.plan['weight_files'][0]['name']
  with self.assertRaises(ValueError):self.m.build(self.plan,self.remote,self.root/'refused')
  self.assertFalse((self.root/'refused').exists())
if __name__=='__main__':unittest.main(verbosity=2)
