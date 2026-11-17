#!/usr/bin/env python3
"""Offline rank budgets, verified byte repacking and expected-manifest verification."""
import argparse
import json
from pathlib import Path
import re

from checkpoint_metadata import Rank, pipeline_plan, validate_catalogue
from rank_bundle import build_bundle, digest, verify_bundle


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'build'):
        p = sub.add_parser(command)
        p.add_argument('--parent', required=True, type=Path)
        p.add_argument('--index', required=True, type=Path)
        p.add_argument('--catalogue', required=True, type=Path)
        p.add_argument('--placement', required=True, type=Path)
        if command == 'plan':
            p.add_argument('--context', required=True, type=int)
            p.add_argument('--kv-bits', choices=(4, 8), type=int)
            p.add_argument('--kv-group-size', default=32, type=int)
        else:
            p.add_argument('--source', required=True, type=Path)
            p.add_argument('--destination', required=True, type=Path)
            p.add_argument('--scratch-reserve-bytes', required=True, type=int)
    verify = sub.add_parser('verify')
    verify.add_argument('directory', type=Path)
    verify.add_argument('--expected-manifest-sha256', required=True)
    verify.add_argument('--rank', help='verify only the assigned rank after a host-local copy')
    args = parser.parse_args()
    if args.command == 'verify':
        receipt = verify_bundle(args.directory, expected_manifest_sha256=args.expected_manifest_sha256,
                                rank_name=args.rank)
        manifest = receipt['manifest']
        print(json.dumps({'manifest_sha256': digest(manifest), 'byte_staging': 'verified',
                          'verified_ranks': receipt['verified_ranks'],
                          'loader_status': manifest['loader_status']}))
        return
    parent, index, catalogue = read(args.parent), read(args.index), read(args.catalogue)
    if not re.fullmatch(r'[a-f0-9]{40}', parent.get('revision', '')):
        parser.error('parent requires an immutable repository revision')
    if catalogue.get('model_id') != parent['model_id'] or catalogue.get('revision') != parent['revision']:
        parser.error('catalogue is not for the pinned parent model/revision')
    ranks = [Rank(**row) for row in read(args.placement)['ranks']]
    files = catalogue['files']
    if args.command == 'build':
        manifest = build_bundle(args.source, args.destination, parent, index, files, ranks,
                                scratch_reserve_bytes=args.scratch_reserve_bytes)
        print(json.dumps({'manifest_sha256': digest(manifest), 'byte_staging': 'verified',
                          'loader_status': manifest['loader_status']}))
    else:
        config = parent['config']
        result = pipeline_plan(validate_catalogue(index, files),
            {file['name']: file['bytes'] for file in parent['weight_files']}, ranks,
            layers=config['num_hidden_layers'], context=args.context,
            kv_heads=config['num_key_value_heads'], head_dim=config['head_dim'],
            kv_bits=args.kv_bits, kv_group_size=args.kv_group_size)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
