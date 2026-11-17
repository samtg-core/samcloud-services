"""Build a fresh hash-pinned administrator package; no install/network effects."""
import argparse
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
ACCOUNTS={'slice':('claude-services',502),'wafer':('wafer-services',504),'obelisk':('obelisk-services',502)}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def package(rank,journal,destination):
    user,uid=ACCOUNTS[rank]
    journal=Path(journal)
    if not journal.is_absolute() or '..' in journal.parts or not journal.is_relative_to(Path('/Users')/user):
        raise ValueError('literal journal under this service home required')
    files={'control.py':(BASE/'exo-pf-control.py').read_bytes(),
           'update.py':(BASE/'update-exo-pf.py').read_bytes(),
           'service-fence.py':(BASE/'exo-pf-service-fence.py').read_bytes(),
           'pf_transport.py':(BASE.parents[1]/'ollama/exo_groups/pf_transport.py').read_bytes(),
           'run':b'#!/bin/sh\nset -eu\nexec /usr/bin/python3 -I -S -B /Library/Preferences/xyz.samtg.exo-pf/control/control.py "$@"\n'}
    cap={'rank':rank,'uid':uid,'username':user,'ports':[52515,52514,52513,52518],'journal':str(journal),
         'private_networks':['10.0.0.0/8','172.16.0.0/12','192.168.0.0/16'],
         'control_sha256':sha(files['control.py']),'update_sha256':sha(files['update.py']),
         'transport_sha256':sha(files['pf_transport.py']),'fence_sha256':sha(files['service-fence.py']),
         'wrapper_sha256':sha(files['run'])}
    files['capability.json']=json.dumps(cap,sort_keys=True,separators=(',',':')).encode()
    manifest={'capability':cap,'files':{n:{'bytes':len(raw),'sha256':sha(raw)} for n,raw in files.items()}}
    raw=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    files['install-manifest.json']=raw
    source=(BASE/'install-exo-pf-control.py').read_text()
    if source.count('REPLACE_WITH_INSTALL_MANIFEST_SHA256')!=1:raise ValueError('installer template changed')
    files['install.py']=source.replace('REPLACE_WITH_INSTALL_MANIFEST_SHA256',sha(raw)).encode()
    destination=Path(destination);destination.mkdir(mode=0o700)
    for name,content in files.items():
        with (destination/name).open('xb') as out:out.write(content)
        (destination/name).chmod(0o400)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',choices=ACCOUNTS,required=True)
    parser.add_argument('--journal',required=True)
    parser.add_argument('--destination',required=True)
    a=parser.parse_args()
    value=package(a.rank,a.journal,a.destination)
    print(json.dumps({'rank':a.rank,'destination':a.destination,'files':len(value['files'])+2,
                      'installed':False,'PF_changed':False}))
