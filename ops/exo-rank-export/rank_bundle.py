"""Stream verified Qwen3 parent bytes into compact, explicitly partial rank files.

No MLX import, tensor conversion, model execution, network or source mutation.
The result is a staging artifact, not an EXO-installable model: its nonstandard
index and VERIFIED marker require a separately qualified shard-aware adapter.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from checkpoint_metadata import (LAYER, MAX_HEADER_BYTES, Rank, parse_header,
                                  validate_catalogue, validate_placement)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _header(tensors: dict) -> bytes:
    cursor, result = 0, {}
    for name, tensor in tensors.items():
        result[name] = {'dtype': tensor['dtype'], 'shape': tensor['shape'],
                        'data_offsets': [cursor, cursor + tensor['bytes']]}
        cursor += tensor['bytes']
    raw = json.dumps(result, separators=(',', ':'), ensure_ascii=False).encode()
    raw += b' ' * (-len(raw) % 8)
    if len(raw) > MAX_HEADER_BYTES:
        raise ValueError('derived header exceeds the bounded metadata size')
    return len(raw).to_bytes(8, 'little') + raw


def build_bundle(source: Path, destination: Path, parent: dict, index: dict,
                 catalogue: dict, ranks: list[Rank], *, scratch_reserve_bytes: int,
                 chunk_bytes: int = 4 * 1024 * 1024) -> dict:
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError('invalid bounded copy buffer')
    if type(scratch_reserve_bytes) is not int or scratch_reserve_bytes < 0:
        raise ValueError('explicit non-negative disk scratch reserve required')
    if not re.fullmatch(r'[a-f0-9]{40}', parent.get('revision', '')):
        raise ValueError('parent requires an immutable repository revision')
    config = parent.get('config', {})
    if config.get('model_type') not in ('qwen3', 'qwen3_moe'):
        raise ValueError('only the reviewed Qwen3 layer naming is supported')
    layers = config['num_hidden_layers']
    validate_placement(ranks, layers)
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', rank.name) for rank in ranks):
        raise ValueError('unsafe rank directory name')
    tensors = validate_catalogue(index, catalogue)
    present_layers = {int(match[1]) for name in tensors if (match := LAYER.match(name))}
    if present_layers != set(range(layers)):
        raise ValueError('tensor catalogue does not cover exactly the configured layers')
    files = {entry['name']: entry for entry in parent['weight_files']}
    if set(files) != set(catalogue):
        raise ValueError('parent and header file sets differ')
    for name, metadata in catalogue.items():
        entry = files[name]
        if not re.fullmatch(r'[a-f0-9]{64}', entry['sha256']):
            raise ValueError('parent must name full file SHA256 values')
        if entry['bytes'] != metadata['bytes'] or entry['sha256'] != metadata['sha256']:
            raise ValueError('header identity differs from pinned parent file')

    # Source order makes each parent a single sequential, simultaneously hashed
    # read. Selected bytes come from those exact hashed chunks, avoiding a
    # verify-then-reopen window or a second 191 GB parent read.
    ordered = dict(sorted(tensors.items(), key=lambda pair: (
        pair[1]['file'], pair[1]['data_offsets'][0], pair[0])))
    selected = []
    for rank in ranks:
        selected.append({name: tensor for name, tensor in ordered.items()
                         if not (match := LAYER.match(name)) or
                         rank.start <= int(match[1]) < rank.end})
    headers = [_header(kept) for kept in selected]
    output_bytes = sum(len(header) + sum(t['bytes'] for t in kept.values())
                       for header, kept in zip(headers, selected, strict=True))
    if shutil.disk_usage(destination.parent).free < output_bytes + scratch_reserve_bytes:
        raise ValueError('destination cannot fit rank payloads and the explicit scratch reserve')
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    streams, hashes, records = [], [], []
    try:
        for rank, kept, header in zip(ranks, selected, headers, strict=True):
            rank_dir = destination / rank.name
            rank_dir.mkdir(mode=0o700)
            relative = f'{rank.name}/rank.safetensors'
            stream = (destination / (relative + '.partial')).open('xb')
            os.chmod(stream.name, 0o600)
            streams.append(stream)
            stream.write(header)
            hashes.append(hashlib.sha256(header))
            records.append({'name': rank.name, 'layers': [rank.start, rank.end],
                            'file': relative, 'required_tensor_names': list(kept),
                            'expected_bytes': len(header) + sum(t['bytes'] for t in kept.values())})

        for name in sorted(files):
            metadata, expected = catalogue[name], files[name]
            source_hash = hashlib.sha256()
            with (source / name).open('rb') as incoming:
                if os.fstat(incoming.fileno()).st_size != expected['bytes']:
                    raise ValueError('parent file size differs from pinned checkpoint')
                prefix = incoming.read(8)
                length = int.from_bytes(prefix, 'little')
                if len(prefix) != 8 or length != metadata['header_bytes'] or length > MAX_HEADER_BYTES:
                    raise ValueError('parent header size differs from recorded metadata')
                raw = incoming.read(length)
                if hashlib.sha256(raw).hexdigest() != metadata['header_sha256'] or (
                    parse_header(raw, expected['bytes']) != metadata['tensors']):
                    raise ValueError('parent tensor header differs from recorded metadata')
                source_hash.update(prefix + raw)
                segments = [[tensor['data_offsets'] for tensor in kept.values() if tensor['file'] == name]
                            for kept in selected]
                position, pointers = 0, [0] * len(ranks)
                while chunk := incoming.read(chunk_bytes):
                    source_hash.update(chunk)
                    end = position + len(chunk)
                    for rank_number, spans in enumerate(segments):
                        pointer = pointers[rank_number]
                        while pointer < len(spans) and spans[pointer][0] < end:
                            start, stop = spans[pointer]
                            lower, upper = max(start, position), min(stop, end)
                            if lower < upper:
                                data = memoryview(chunk)[lower-position:upper-position]
                                streams[rank_number].write(data)
                                hashes[rank_number].update(data)
                            if stop > end:
                                break
                            pointer += 1
                        pointers[rank_number] = pointer
                    position = end
                if incoming.tell() != expected['bytes'] or source_hash.hexdigest() != expected['sha256']:
                    raise ValueError('parent file SHA256 does not match pinned checkpoint')

        for stream, hasher, record in zip(streams, hashes, records, strict=True):
            stream.flush()
            os.fsync(stream.fileno())
            if stream.tell() != record['expected_bytes']:
                raise ValueError('derived rank did not receive every required tensor byte')
            record['sha256'] = hasher.hexdigest()
            stream.close()
            os.replace(destination / (record['file'] + '.partial'), destination / record['file'])
        manifest = {'format': 'samcloud-qwen3-rank-staging-v1',
                    'model_id': parent['model_id'], 'revision': parent['revision'],
                    'parent_manifest_sha256': digest(parent), 'parent_index_sha256': digest(index),
                    'config_sha256': digest(config), 'rank_policy': 'assigned_layers_plus_all_nonlayers',
                    'ranks': records,
                    'loader_status': 'requires_qualified_shard_aware_adapter'}
        raw = canonical(manifest)
        with (destination / 'rank-bundle.json').open('xb') as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        # This marker attests the byte staging only. No standard EXO index is
        # written that could misidentify a partial checkpoint as a full model.
        with (destination / 'VERIFIED').open('xb') as out:
            out.write((hashlib.sha256(raw).hexdigest() + '\n').encode())
            out.flush()
            os.fsync(out.fileno())
        return manifest
    finally:
        for stream in streams:
            stream.close()


def verify_bundle(directory: Path, *, expected_manifest_sha256: str, rank_name: str | None = None) -> dict:
    if not re.fullmatch(r'[a-f0-9]{64}', expected_manifest_sha256):
        raise ValueError('a pinned expected manifest digest is required')
    raw = (directory / 'rank-bundle.json').read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_manifest_sha256:
        raise ValueError('bundle is not the requested immutable staging manifest')
    if (directory / 'VERIFIED').read_text().strip() != actual:
        raise ValueError('staging bundle has no matching verification marker')
    manifest = json.loads(raw)
    if manifest.get('format') != 'samcloud-qwen3-rank-staging-v1':
        raise ValueError('unknown staging bundle format')
    selected = [rank for rank in manifest['ranks'] if rank_name is None or rank['name'] == rank_name]
    if not selected:
        raise ValueError('requested rank is not in the pinned manifest')
    for rank in selected:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}/rank\.safetensors', rank['file']):
            raise ValueError('unsafe rank file path')
        hasher, size = hashlib.sha256(), 0
        with (directory / rank['file']).open('rb') as incoming:
            while chunk := incoming.read(4 * 1024 * 1024):
                hasher.update(chunk)
                size += len(chunk)
        if size != rank['expected_bytes'] or hasher.hexdigest() != rank['sha256']:
            raise ValueError('derived rank bytes do not match the verified manifest')
    return {'manifest': manifest, 'verified_ranks': [rank['name'] for rank in selected],
            'byte_staging': 'verified'}
