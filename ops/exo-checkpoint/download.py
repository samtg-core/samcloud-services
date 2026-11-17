"""Data-only checkpoint staging. No MLX import, model load, lease or inference.

The controller binds an immutable plan only AFTER the small full gate passes.
An explicit continuation may resume downloaded bytes; whole-file pinned hashes
must still match before atomic publication. No completed file is overwritten.
"""
import argparse,hashlib,json,os,re,stat,time,urllib.parse,urllib.request
from pathlib import Path
PACKET=Path('REPLACE_PACKET')
PLAN_SHA='REPLACE_PLAN_SHA'
CHUNK=8*1024*1024

def sha_file(path,git=False):
 h=hashlib.sha1() if git else hashlib.sha256()
 if git:h.update(('blob '+str(path.stat().st_size)+'\0').encode())
 with path.open('rb') as f:
  while part:=f.read(CHUNK):h.update(part)
 return h.hexdigest()

def owned(path,uid):
 info=path.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=uid or info.st_nlink!=1:raise ValueError('checkpoint file is redirected, linked or foreign')
 return info

def verify(path,row,uid):
 info=owned(path,uid)
 if info.st_size!=row['bytes']:raise ValueError('checkpoint size mismatch')
 actual=sha_file(path)
 if row.get('sha256'):
  if actual!=row['sha256']:raise ValueError('checkpoint SHA256 mismatch')
 elif sha_file(path,git=True)!=row['git_blob_sha1']:raise ValueError('checkpoint Git blob checksum mismatch')
 return {'bytes':info.st_size,'sha256':actual}

def save(path,value):
 raw=json.dumps(value,indent=2).encode()+b'\n';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def fetch(row,root,url,uid,opener=urllib.request.urlopen):
 name=row['name']
 if not re.fullmatch(r'[A-Za-z0-9._-]+',name) or name in ('.','..'):raise ValueError('unsafe checkpoint filename')
 final=root/name;part=root/(name+'.part')
 if final.exists():return verify(final,row,uid)
 if part.exists():
  info=owned(part,uid);offset=info.st_size
 else:
  fd=os.open(part,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd);offset=0
 if not 0<=offset<=row['bytes']:raise ValueError('partial file exceeds pinned size')
 if offset<row['bytes']:
  headers={'Accept-Encoding':'identity'}
  if offset:headers['Range']='bytes='+str(offset)+'-'
  request=urllib.request.Request(url,headers=headers)
  with opener(request,timeout=None) as response:
   if response.status!=(206 if offset else 200):raise ValueError('download response/range not honored')
   if offset and response.headers.get('Content-Range')!=f"bytes {offset}-{row['bytes']-1}/{row['bytes']}":raise ValueError('download range differs')
   if response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('encoded checkpoint response')
   if response.headers.get('Content-Length') not in (None,str(row['bytes']-offset)):raise ValueError('download response size differs')
   fd=os.open(part,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
   with os.fdopen(fd,'ab') as f:
    if os.fstat(f.fileno()).st_size!=offset:raise ValueError('partial file changed')
    position=offset;reported=offset
    while position<row['bytes']:
     chunk=response.read(min(CHUNK,row['bytes']-position))
     if not chunk:raise ValueError('short checkpoint download; preserved for explicit continuation')
     f.write(chunk);position+=len(chunk)
     if position-reported>=256*1024*1024:
      print(json.dumps({'file':name,'downloaded_bytes':position,'file_bytes':row['bytes']}),flush=True);reported=position
    if response.read(1):raise ValueError('oversized checkpoint response')
    f.flush();os.fsync(f.fileno())
 result=verify(part,row,uid)
 # Our exclusively owned destination has no competing publisher. Never replace
 # an independently appearing complete artifact.
 if final.exists() or final.is_symlink():raise ValueError('complete destination appeared')
 os.link(part,final,follow_symlinks=False);part.unlink();final.chmod(0o444);return result

def main():
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--start',action='store_true');g.add_argument('--continue-download',action='store_true');args=parser.parse_args()
 raw=(PACKET/'plan.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==PLAN_SHA;d=json.loads(raw)
 assert os.getuid()==d['uid']==502 and d['revision']=='6dd2ef235d39fb976237fd9243184741d8407a55'
 assert d['model_id']=='mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit'
 assert d['small_gate']['phase']=='released' and d['small_gate']['positive_cleanup_and_exact_settlement'] is True
 assert d['small_gate']['three_cases_16_raw_ids_each_rank'] is True
 root=Path(d['destination']);assert root.parent==Path('/Users/obelisk-services/var/obelisk-services/exo-qualification')
 os.umask(0o077)
 if args.start:
  root.mkdir(mode=0o700);save(root/'STAGING-INTENT.json',{'plan_sha256':PLAN_SHA,'at':time.time(),'data_only':True})
 else:
  assert root.lstat().st_uid==502 and stat.S_ISDIR(root.lstat().st_mode)
  assert json.loads((root/'STAGING-INTENT.json').read_bytes())['plan_sha256']==PLAN_SHA and not (root/'COMPLETE.json').exists()
 files=d['metadata_files']+d['weight_files'];assert len({v['name'] for v in files})==len(files)
 remaining=sum(v['bytes'] for v in files if not (root/v['name']).exists());fs=os.statvfs(root);assert fs.f_bavail*fs.f_frsize>=remaining+8*1024**3
 result=[]
 for row in files:
  url='https://huggingface.co/'+d['model_id']+'/resolve/'+d['revision']+'/'+urllib.parse.quote(row['name'],safe='')
  value=fetch(row,root,url,502);result.append({'name':row['name'],**value});print(json.dumps({'verified':row['name'],**value}),flush=True)
  assert os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>=8*1024**3
 save(root/'COMPLETE.json',{'model_id':d['model_id'],'revision':d['revision'],'plan_sha256':PLAN_SHA,'files':result,'weights_bytes':d['weights_bytes'],'load_or_fit_claim':False})
 for p in root.iterdir():assert p.is_file() and not p.is_symlink();p.chmod(0o444)
 root.chmod(0o555);print(json.dumps({'checkpoint_verified':True,'root':str(root),'files':len(result),'load_or_fit_claim':False}),flush=True)
if __name__=='__main__':main()
