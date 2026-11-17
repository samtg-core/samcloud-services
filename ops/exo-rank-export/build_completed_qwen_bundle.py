"""Data-only rank export from a sealed, fully verified checkpoint.

The packet builder replaces PACKET and PLAN_SHA. No native model imports,
network access, leases, gateway changes, or automatic retries occur here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import sys

PACKET = Path('REPLACE_PACKET')
PLAN_SHA = 'REPLACE_PLAN_SHA'


def regular(path, uid, size=None):
    item = path.lstat()
    if (not stat.S_ISREG(item.st_mode) or item.st_uid != uid or
            item.st_nlink != 1 or item.st_mode & 0o222 or
            (size is not None and item.st_size != size)):
        raise ValueError('input is not a sealed, owned, unlinked regular file')
    return item


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def validate_source(plan):
    """Inspect completion and small metadata; build_bundle hashes weight bytes."""
    uid = plan['uid']
    if os.getuid() != uid:
        raise ValueError('wrong staging principal')
    root = Path(plan['source'])
    entry = root.lstat()
    if (not stat.S_ISDIR(entry.st_mode) or entry.st_uid != uid or
            entry.st_mode & 0o222 or root.resolve() != root):
        raise ValueError('checkpoint directory is not sealed and owned')
    regular(root / 'COMPLETE.json', uid)
    receipt = json.loads((root / 'COMPLETE.json').read_bytes())
    for key in ('model_id', 'revision', 'plan_sha256', 'weights_bytes'):
        if receipt[key] != plan['checkpoint'][key]:
            raise ValueError('checkpoint completion identity differs')
    files = plan['checkpoint']['files']
    if receipt['files'] != files:
        raise ValueError('completion does not name every pinned file in order')
    expected_names = {row['name'] for row in files}
    if len(expected_names) != len(files) or set(p.name for p in root.iterdir()) != (
            expected_names | {'STAGING-INTENT.json', 'COMPLETE.json'}):
        raise ValueError('checkpoint contains missing, duplicate or extra entries')
    regular(root / 'STAGING-INTENT.json', uid)
    if json.loads((root / 'STAGING-INTENT.json').read_bytes())['plan_sha256'] != (
            plan['checkpoint']['plan_sha256']):
        raise ValueError('checkpoint staging intent differs')
    for row in files:
        name = row['name']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('unsafe checkpoint filename')
        path = root / name
        regular(path, uid, row['bytes'])
        if not name.endswith('.safetensors') and file_sha(path) != row['sha256']:
            raise ValueError('checkpoint metadata hash differs')
    return {'checkpoint_sha256': file_sha(root / 'COMPLETE.json'),
            'files': len(files), 'metadata_verified': True,
            'weight_hashes': 'verified again by build_bundle while exporting'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--review', action='store_true')
    group.add_argument('--build', action='store_true')
    args = parser.parse_args()
    raw = (PACKET / 'plan.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != PLAN_SHA:
        raise ValueError('rank-export plan changed')
    plan = json.loads(raw)
    if os.getuid() != plan['uid']:
        raise ValueError('wrong staging principal')
    for row in plan['packet_files']:
        name = row['name']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('unsafe packet filename')
        regular(PACKET / name, plan['uid'], row['bytes'])
        if file_sha(PACKET / name) != row['sha256']:
            raise ValueError('rank-export module or metadata changed')
    observation = validate_source(plan)
    destination = Path(plan['destination'])
    if destination.parent != Path(plan['source']).parent or destination.exists():
        raise ValueError('rank export destination is not fresh')
    print(json.dumps({'review': 'passed', **observation}), flush=True)
    if args.review:
        return
    sys.path.insert(0, str(PACKET))
    from rank_bundle import build_bundle, digest
    from checkpoint_metadata import Rank
    read = lambda n: json.loads((PACKET / n).read_bytes())
    parent, index, catalogue = read('parent.json'), read('index.json'), read('catalogue.json')
    if (catalogue['model_id'], catalogue['revision']) != (parent['model_id'], parent['revision']):
        raise ValueError('catalogue and parent identity differ')
    manifest = build_bundle(Path(plan['source']), destination, parent, index,
                            catalogue['files'], [Rank(**r) for r in plan['ranks']],
                            scratch_reserve_bytes=plan['scratch_reserve_bytes'])
    # The packer hashes every parent while selecting bytes; its verified marker
    # only attests byte staging. Serving and resident fit remain separate gates.
    for path in destination.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(destination.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    destination.chmod(0o555)
    print(json.dumps({'manifest_sha256': digest(manifest), 'byte_staging': 'verified',
                      'destination': str(destination), 'load_or_fit_claim': False}), flush=True)


if __name__ == '__main__':
    main()
