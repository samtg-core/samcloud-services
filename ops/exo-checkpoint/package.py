"""Seal the reviewed data-only stager and its explicit post-qualification plan."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

BASE=PurePosixPath('/Users/obelisk-services/var/obelisk-services/exo-qualification')
MODEL='mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit'
REVISION='6dd2ef235d39fb976237fd9243184741d8407a55'


def check_plan(plan):
    gate=plan.get('small_gate',{})
    if (plan.get('model_id')!=MODEL or plan.get('revision')!=REVISION
            or type(plan.get('uid')) is not int or plan['uid']!=502
            or gate.get('phase')!='released'
            or gate.get('positive_cleanup_and_exact_settlement') is not True
            or gate.get('three_cases_16_raw_ids_each_rank') is not True
            or len(gate.get('cleanup_receipts',[]))!=3
            or not re.fullmatch('[a-f0-9]{64}',plan.get('small_gate_sha256',''))):
        raise ValueError('exact model and completed three-host qualification required')
    path=PurePosixPath(plan.get('destination',''))
    if path.parent!=BASE or not re.fullmatch('[A-Za-z0-9._-]+',path.name) or path.name in ('.','..') or '..' in path.parts:
        raise ValueError('exact service-owned checkpoint destination required')
    weights,metadata=plan.get('weight_files'),plan.get('metadata_files')
    if not isinstance(weights,list) or len(weights)!=37 or not isinstance(metadata,list) or not metadata:
        raise ValueError('complete pinned checkpoint inventory required')
    rows=weights+metadata
    if len({v.get('name') for v in rows})!=len(rows):
        raise ValueError('checkpoint names overlap')
    for row in rows:
        if (not re.fullmatch('[A-Za-z0-9._-]+',row.get('name',''))
                or row['name'] in ('.','..','COMPLETE.json','STAGING-INTENT.json')
                or type(row.get('bytes')) is not int or row['bytes']<=0):
            raise ValueError('bounded regular checkpoint file required')
        if row.get('sha256'):
            if not re.fullmatch('[a-f0-9]{64}',row['sha256']):raise ValueError('pinned SHA256 required')
        elif row in weights or not re.fullmatch('[a-f0-9]{40}',row.get('git_blob_sha1','')):
            raise ValueError('weight LFS SHA256 or metadata Git blob checksum required')
    if plan.get('weights_bytes')!=sum(row['bytes'] for row in weights):
        raise ValueError('weight inventory size differs')


def build(plan,remote_packet,destination):
    check_plan(plan)
    remote=PurePosixPath(remote_packet)
    if remote.parent!=BASE or not re.fullmatch('[A-Za-z0-9._-]+',remote.name) or remote.name in ('.','..') or '..' in remote.parts:
        raise ValueError('exact service-owned stager packet path required')
    raw=json.dumps(plan,separators=(',',':')).encode();pin=hashlib.sha256(raw).hexdigest()
    template=Path(__file__).with_name('download.py').read_text()
    if template.count('REPLACE_PACKET')!=1 or template.count('REPLACE_PLAN_SHA')!=1:
        raise ValueError('stager template changed')
    script=template.replace('REPLACE_PACKET',str(remote)).replace('REPLACE_PLAN_SHA',pin).encode()
    compile(script,'<pinned-checkpoint-stager>','exec')
    destination=Path(destination);destination.mkdir(mode=0o700)
    for name,value in (('plan.json',raw),('download.py',script)):
        path=destination/name
        with path.open('xb') as f:f.write(value)
        path.chmod(0o400)
    return {'plan_sha256':pin,'script_sha256':hashlib.sha256(script).hexdigest(),
            'remote_packet':str(remote),'operation':'data staging only; no model, lease or inference'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--remote-packet',required=True)
    parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(build(json.loads(args.plan.read_bytes()),args.remote_packet,args.destination)))
if __name__=='__main__':main()
