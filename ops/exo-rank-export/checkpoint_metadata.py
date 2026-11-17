"""Read bounded safetensor metadata and screen pipeline staging, without weights.

This is a planning tool, not a checkpoint loader or a proof of resident fit.
The pinned repository/LFS hashes must still be checked when bytes are staged.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from typing import Callable


MAX_HEADER_BYTES = 8 * 1024 * 1024
DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'U16': 2, 'I16': 2, 'F16': 2, 'BF16': 2,
    'U32': 4, 'I32': 4, 'F32': 4, 'U64': 8, 'I64': 8, 'F64': 8,
}
LAYER = re.compile(r'^model\.layers\.(\d+)\.')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key in checkpoint metadata')
        result[key] = value
    return result


def read_range(url: str, start: int, length: int, total: int,
               opener: Callable = urllib.request.urlopen) -> bytes:
    """Refuse a server ignoring Range; never fall back to fetching the model."""
    if not (type(start) is int and type(length) is int and type(total) is int and
            0 <= start < total and 0 < length <= MAX_HEADER_BYTES and start + length <= total):
        raise ValueError('invalid bounded metadata range')
    end = start + length - 1
    request = urllib.request.Request(url, headers={
        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with opener(request, timeout=60) as response:
        if response.status != 206:
            raise ValueError('server did not honor the metadata range; no payload read')
        if response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
            raise ValueError('metadata range is not the requested pinned file extent')
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('encoded range is not a safetensor file extent')
        if response.headers.get('Content-Length') not in (None, str(length)):
            raise ValueError('metadata response size disagrees with range')
        data = response.read(length + 1)
        if len(data) != length:
            raise ValueError('short or oversized metadata range')
        return data


def parse_header(raw: bytes, file_size: int) -> dict:
    if not 0 < len(raw) <= MAX_HEADER_BYTES or len(raw) + 8 > file_size:
        raise ValueError('invalid safetensor header size')
    header = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError('safetensor header must be an object')
    payload_size = file_size - 8 - len(raw)
    tensors = {}
    for name, entry in header.items():
        if name == '__metadata__':
            if not isinstance(entry, dict):
                raise ValueError('invalid safetensor metadata')
            continue
        if not isinstance(entry, dict) or entry.get('dtype') not in DTYPE_BYTES:
            raise ValueError('unknown or invalid tensor dtype')
        shape, offsets = entry.get('shape'), entry.get('data_offsets')
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('invalid tensor shape')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError('invalid tensor offsets')
        start, end = offsets
        expected = math.prod(shape) * DTYPE_BYTES[entry['dtype']]
        if not 0 <= start <= end <= payload_size or end - start != expected:
            raise ValueError('tensor byte count does not match shape or file bounds')
        tensors[name] = {'dtype': entry['dtype'], 'shape': shape, 'bytes': expected,
                         'data_offsets': offsets}
    cursor = 0
    for entry in sorted(tensors.values(), key=lambda entry: entry['data_offsets']):
        start, end = entry['data_offsets']
        if start != cursor:
            raise ValueError('tensor extents overlap or leave unaccounted payload')
        cursor = end
    if cursor != payload_size:
        raise ValueError('tensor extents do not account for the full payload')
    return tensors


def fetch_header(url: str, file_size: int, opener: Callable = urllib.request.urlopen) -> dict:
    prefix = read_range(url, 0, 8, file_size, opener)
    length = int.from_bytes(prefix, 'little')
    if not 0 < length <= MAX_HEADER_BYTES or length + 8 > file_size:
        raise ValueError('safetensor header length is out of bounds')
    raw = read_range(url, 8, length, file_size, opener)
    return {'header_bytes': length, 'header_sha256': hashlib.sha256(raw).hexdigest(),
            'tensors': parse_header(raw, file_size)}


def validate_catalogue(index: dict, catalogue: dict) -> dict:
    mapping = index.get('weight_map')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('missing checkpoint index map')
    tensors = {}
    for file, metadata in catalogue.items():
        if not re.fullmatch(r'[A-Za-z0-9._-]+\.safetensors', file):
            raise ValueError('unsafe checkpoint file name')
        for name, tensor in metadata['tensors'].items():
            if name in tensors or mapping.get(name) != file:
                raise ValueError('header tensor duplicates or disagrees with pinned index')
            tensors[name] = {**tensor, 'file': file}
    if set(tensors) != set(mapping):
        raise ValueError('headers do not cover every indexed tensor')
    return tensors


def packed_size(tensors: dict) -> int:
    """Exact bytes for a proposed one-file packing, not an existing artifact hash."""
    header, cursor = {}, 0
    for name, tensor in sorted(tensors.items(), key=lambda pair: (
            pair[1]['file'], pair[1]['data_offsets'][0], pair[0])):
        end = cursor + tensor['bytes']
        header[name] = {'dtype': tensor['dtype'], 'shape': tensor['shape'],
                        'data_offsets': [cursor, end]}
        cursor = end
    raw = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode()
    return 8 + ((len(raw) + 7) // 8 * 8) + cursor


@dataclass(frozen=True)
class Rank:
    name: str
    start: int
    end: int
    metal_recommended_bytes: int
    free_disk_bytes: int


def validate_placement(ranks: list[Rank], layers: int) -> None:
    if not ranks or ranks[0].start != 0 or ranks[-1].end != layers:
        raise ValueError('placement must cover every model layer')
    if len({rank.name for rank in ranks}) != len(ranks):
        raise ValueError('one pipeline rank per host required')
    previous_end = 0
    for rank in ranks:
        if rank.start != previous_end or rank.end <= rank.start or rank.end > layers:
            raise ValueError('placement has a gap, overlap or empty rank')
        if rank.metal_recommended_bytes <= 0 or rank.free_disk_bytes < 0:
            raise ValueError('invalid measured host envelope')
        previous_end = rank.end


def pipeline_plan(tensors: dict, file_sizes: dict, ranks: list[Rank], *, layers: int,
                  context: int, kv_heads: int, head_dim: int, kv_element_bytes: int = 2,
                  kv_bits: int | None = None, kv_group_size: int = 32) -> dict:
    validate_placement(ranks, layers)
    if any(type(n) is not int or n <= 0 for n in (layers, context, kv_heads, head_dim, kv_element_bytes)):
        raise ValueError('explicit positive cache geometry required')
    if kv_bits is not None and (type(kv_bits) is not int or kv_bits not in (4, 8) or
            type(kv_group_size) is not int or kv_group_size <= 0 or head_dim % kv_group_size):
        raise ValueError('unsupported quantized-cache geometry')
    layer_names = {n: [] for n in range(layers)}
    for name in tensors:
        match = LAYER.match(name)
        if match:
            number = int(match[1])
            if number not in layer_names:
                raise ValueError('tensor layer is outside model configuration')
            layer_names[number].append(name)
    if any(not names for names in layer_names.values()):
        raise ValueError('checkpoint misses a configured model layer')
    result = []
    for rank in ranks:
        # The reviewed generic EXO pipeline retains embedding, norm and output
        # modules on every rank. Do not assume first/last-only replication.
        selected = {name: tensor for name, tensor in tensors.items()
                    if not (match := LAYER.match(name)) or rank.start <= int(match[1]) < rank.end}
        files = sorted({tensor['file'] for tensor in selected.values()})
        payload = sum(tensor['bytes'] for tensor in selected.values())
        cache_width_bytes = head_dim * kv_element_bytes if kv_bits is None else (
            head_dim * kv_bits // 8 + (head_dim // kv_group_size) * 2 * kv_element_bytes)
        cache = 2 * (rank.end - rank.start) * kv_heads * context * cache_width_bytes
        whole = sum(file_sizes[file] for file in files)
        packed = packed_size(selected)
        result.append({'name': rank.name, 'layers': [rank.start, rank.end],
                       'tensor_count': len(selected), 'files': files,
                       'whole_file_bytes': whole, 'selected_payload_bytes': payload,
                       'proposed_packed_bytes': packed, 'cache_lower_bound_bytes': cache,
                       'resident_lower_bound_bytes': payload + cache,
                       'metal_margin_before_workspace_bytes': rank.metal_recommended_bytes - payload - cache,
                       'whole_file_disk_margin_bytes': rank.free_disk_bytes - whole,
                       'packed_disk_margin_bytes': rank.free_disk_bytes - packed})
    return {'ranks': result, 'cache_bits': kv_bits,
        'cache_group_size': kv_group_size if kv_bits is not None else None,
        'cache_scale_bias_dtype_bytes': kv_element_bytes if kv_bits is not None else None,
        'all_metal_lower_bounds_fit': all(
        rank['metal_margin_before_workspace_bytes'] >= 0 for rank in result),
        'note': 'A lower-bound screen, not a measured loading/prefill/serving fit. '
                'Packed files do not exist; a shard-aware loader still needs qualification.'}
