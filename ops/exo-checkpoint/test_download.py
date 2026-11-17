"""Pinned checkpoint transport controls with byte streams, no network/models."""
import hashlib,importlib.util,io,os,tempfile,unittest
from pathlib import Path
class Response(io.BytesIO):
 def __init__(self,raw,status=200,headers=None):super().__init__(raw);self.status=status;self.headers=headers or {}
class Download(unittest.TestCase):
 def setUp(self):
  p=Path(__file__).with_name('download.py');s=importlib.util.spec_from_file_location('download_pinned',p);self.m=importlib.util.module_from_spec(s);s.loader.exec_module(self.m)
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name);self.uid=os.getuid();self.body=b'complete immutable model bytes';self.row={'name':'model.safetensors','bytes':len(self.body),'sha256':hashlib.sha256(self.body).hexdigest()}
 def fetch(self,opener):return self.m.fetch(self.row,self.root,'https://example.invalid/model',self.uid,opener)
 def test_verified_bytes_publish_read_only_and_reuse_never_contacts_network(self):
  value=self.fetch(lambda *a,**k:Response(self.body));p=self.root/self.row['name'];self.assertEqual(p.read_bytes(),self.body);self.assertEqual(p.stat().st_mode&0o777,0o444);self.assertEqual(p.stat().st_nlink,1)
  self.assertEqual(self.fetch(lambda *a,**k:self.fail('verified file re-downloaded')),value)
 def test_checksum_mismatch_remains_partial_and_is_not_published(self):
  self.row['sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'SHA256'):self.fetch(lambda *a,**k:Response(self.body))
  self.assertFalse((self.root/self.row['name']).exists())
 def test_short_transfer_preserves_partial_for_explicit_range_continuation(self):
  with self.assertRaisesRegex(ValueError,'short'):self.fetch(lambda *a,**k:Response(self.body[:8]))
  def serve(request,**kwargs):
   self.assertEqual(request.get_header('Range'),'bytes=8-')
   return Response(self.body[8:],206,{'Content-Range':f'bytes 8-{len(self.body)-1}/{len(self.body)}'})
  self.fetch(serve);self.assertEqual((self.root/self.row['name']).read_bytes(),self.body)
 def test_ignored_or_wrong_range_does_not_append(self):
  part=self.root/(self.row['name']+'.part');part.write_bytes(self.body[:8])
  for response in (Response(self.body),Response(self.body[8:],206,{'Content-Range':'wrong'})):
   with self.assertRaises(ValueError):self.fetch(lambda *a,**k:response)
   self.assertEqual(part.read_bytes(),self.body[:8])
 def test_oversized_reply_is_refused(self):
  with self.assertRaisesRegex(ValueError,'oversized'):self.fetch(lambda *a,**k:Response(self.body+b'extra'))
  self.assertFalse((self.root/self.row['name']).exists())
 def test_existing_or_partial_symlink_is_never_followed(self):
  target=self.root/'target';target.write_bytes(self.body)
  for name in (self.row['name'],self.row['name']+'.part'):
   p=self.root/name;p.symlink_to(target)
   with self.assertRaises(ValueError):self.fetch(lambda *a,**k:self.fail('symlink led to download'))
   self.assertEqual(target.read_bytes(),self.body);p.unlink()
 def test_git_blob_checksum_covers_small_metadata(self):
  self.row.pop('sha256');self.row['git_blob_sha1']=hashlib.sha1(b'blob '+str(len(self.body)).encode()+b'\0'+self.body).hexdigest()
  self.fetch(lambda *a,**k:Response(self.body))
 def test_path_traversal_is_refused_before_network(self):
  self.row['name']='../outside'
  with self.assertRaisesRegex(ValueError,'unsafe'):self.fetch(lambda *a,**k:self.fail('unsafe URL used'))
if __name__=='__main__':unittest.main(verbosity=2)
