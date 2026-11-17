"""One-time administrator install of a bounded EXO firewall capability.

Only root-owned helper files and one exact sudoers entry are installed. No PF
rule, service selection, environment, Python package or queue row is changed.
"""
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import sys

MANIFEST_SHA256 = 'REPLACE_WITH_INSTALL_MANIFEST_SHA256'
ROOT = Path('/Library/Preferences/xyz.samtg.exo-pf')
SUDOERS = Path('/private/etc/sudoers.d/91-samcloud-exo-control')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_payload(name, expected):
    path = Path(__file__).parent / name
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 262144:
            raise ValueError('bounded exact installer payload required')
        raw = b''
        while len(raw) <= 262144:
            chunk = os.read(fd, min(65536, 262145-len(raw)))
            if not chunk: break
            raw += chunk
        after = os.fstat(fd)
        fields = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        if digest(raw) != expected or any(getattr(before,k)!=getattr(after,k) for k in fields):
            raise ValueError('installer payload changed')
        return raw
    finally:
        os.close(fd)


def protected_directory(path):
    for p in (path, *path.parents):
        info = p.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError('root-owned protected installation parents required')


def write(path, raw, mode):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.chmod(path,mode)


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def sudoers_text(user, wrapper, sha):
    if not re.fullmatch('[a-z][a-z0-9_-]*',user) or not re.fullmatch('[a-f0-9]{64}',sha):
        raise ValueError('literal account and installed executable digest required')
    if str(wrapper) != str(ROOT/'control/run'):
        raise ValueError('fixed root-owned executable required')
    return ('# Bounded EXO anchor capability; no shell, Python or arbitrary PF commands.\n'+
        ''.join(user+' ALL=(root) NOPASSWD: NOSETENV: sha256:'+sha+' '+str(wrapper)+' '+flag+'\n'
                for flag in ('--apply','--status'))).encode()


def main():
    if os.geteuid()!=0 or sys.argv[1:]!=['--install']:
        raise SystemExit('one-time administrator --install required')
    raw=read_payload('install-manifest.json',MANIFEST_SHA256)
    manifest=json.loads(raw);cap=manifest['capability'];payloads={}
    if set(manifest['files'])!={'control.py','update.py','pf_transport.py','service-fence.py','run','capability.json'}:
        raise ValueError('exact helper payload set required')
    for name,pin in manifest['files'].items():payloads[name]=read_payload(name,pin['sha256'])
    if json.loads(payloads['capability.json'])!=cap:raise ValueError('capability differs')
    if (cap['rank'],cap['uid']) not in (('slice',502),('wafer',504),('obelisk',502)):
        raise ValueError('exact qualified service host required')
    account=pwd.getpwuid(cap['uid']).pw_name
    if account!=cap['username']:raise ValueError('local service identity differs')
    for p in (ROOT,SUDOERS.parent):protected_directory(p)
    main_sudoers=Path('/private/etc/sudoers')
    if not re.search(r'^(?:#|@)includedir\s+/private/etc/sudoers\.d\s*$',main_sudoers.read_text(),re.M):
        raise ValueError('existing sudoers include is not the qualified one')
    subprocess.run(['/usr/sbin/visudo','-c','-f',str(main_sudoers)],check=True,capture_output=True)
    receipt_path=ROOT/'receipt.json';before=receipt_path.read_bytes();receipt=json.loads(before)
    if receipt['phase']!='active' or receipt['policy']['rank']!=cap['rank'] or receipt['policy']['uid']!=cap['uid']:
        raise ValueError('existing active owned EXO installation required')
    if receipt['policy']['ports']!=cap['ports']:raise ValueError('port capability differs')
    read_sudoers=Path('/private/etc/sudoers.d/90-samcloud-exo-read')
    if digest(read_sudoers.read_bytes())!=receipt['sudoers_sha256']:
        raise ValueError('existing read-only delegation differs')
    wrapper=ROOT/'control/run';rule=sudoers_text(account,wrapper,digest(payloads['run']))
    stage=ROOT/('.control-stage-'+MANIFEST_SHA256)
    control=ROOT/'control'
    if SUDOERS.exists() and not control.exists():
        raise ValueError('delegation without its helper requires inspection')
    if control.exists():
        protected_directory(control)
        for name,value in payloads.items():
            p=control/name
            info=p.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_nlink!=1
                    or stat.S_IMODE(info.st_mode)!=(0o500 if name=='run' else 0o444) or p.read_bytes()!=value):
                raise ValueError('different existing helper; explicit upgrade review required')
    else:
        if stage.exists():raise ValueError('partial installer staging requires inspection')
        stage.mkdir(mode=0o755)
        for name,value in payloads.items():write(stage/name,value,0o500 if name=='run' else 0o444)
        (stage/'requests').mkdir(mode=0o700)
        write(stage/'install-manifest.json',raw,0o444)
        fsync_dir(stage)
    temporary=SUDOERS.with_name('.91-samcloud-exo-control-'+MANIFEST_SHA256)
    if SUDOERS.exists():
        if SUDOERS.is_symlink() or SUDOERS.lstat().st_uid!=0 or SUDOERS.read_bytes()!=rule:
            raise ValueError('different existing delegation')
    else:
        if temporary.exists():raise ValueError('partial sudoers candidate requires inspection')
        write(temporary,rule,0o440)
        subprocess.run(['/usr/sbin/visudo','-c','-f',str(temporary)],check=True,capture_output=True)
        if not control.exists():os.rename(stage,control);fsync_dir(ROOT)
        os.rename(temporary,SUDOERS);fsync_dir(SUDOERS.parent)
    subprocess.run(['/usr/sbin/visudo','-c','-f','/private/etc/sudoers'],check=True,capture_output=True)
    if receipt_path.read_bytes()!=before or digest(read_sudoers.read_bytes())!=receipt['sudoers_sha256']:
        raise ValueError('existing PF receipt or read-only delegation changed')
    print(json.dumps({'installed':True,'rank':cap['rank'],'service_account':account,
                     'command':str(wrapper),'scope':'only EXO anchor updates/status',
                     'pf_or_queue_or_service_changed':False}))


if __name__=='__main__':main()
