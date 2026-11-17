import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess
import sys
from unittest.mock import patch

from checkpoint_metadata import Rank, parse_header
from rank_bundle import build_bundle, verify_bundle, digest


def create_parent(directory):
    files, catalogue, mapping, originals = [], {}, {}, {}
    for file_number in range(2):
        name = f'model-{file_number:05d}.safetensors'
        names = [f'model.layers.{file_number*2+n}.weight' for n in range(2)]
        if file_number == 0:
            names += ['model.embed_tokens.weight']
        else:
            names += ['model.norm.weight', 'lm_head.weight']
        header, payload = {}, bytearray()
        for tensor in names:
            # Include arbitrary packed integer bytes, including zeroes. No
            # decoding/requantization may alter any of them during repacking.
            data = hashlib.sha256(tensor.encode()).digest()[:8]
            start = len(payload)
            payload += data
            header[tensor] = {'dtype': 'U32', 'shape': [2], 'data_offsets': [start, len(payload)]}
            mapping[tensor] = name
            originals[tensor] = data
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        contents = len(raw).to_bytes(8, 'little') + raw + payload
        (directory / name).write_bytes(contents)
        identity = {'name': name, 'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()}
        files.append(identity)
        catalogue[name] = {**identity, 'header_bytes': len(raw),
                           'header_sha256': hashlib.sha256(raw).hexdigest(),
                           'tensors': parse_header(raw, len(contents))}
    return ({'model_id': 'fixture/qwen3', 'revision': 'a' * 40, 'weight_files': files,
             'config': {'model_type': 'qwen3', 'num_hidden_layers': 4,
                        'num_key_value_heads': 2, 'head_dim': 64}},
            {'weight_map': mapping}, catalogue, originals)


def rank_tensors(path):
    raw = path.read_bytes()
    length = int.from_bytes(raw[:8], 'little')
    tensors = parse_header(raw[8:8+length], len(raw))
    payload = raw[8+length:]
    return {name: payload[t['data_offsets'][0]:t['data_offsets'][1]] for name, t in tensors.items()}


class BundleControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.parent, self.index, self.catalogue, self.originals = create_parent(self.source)
        self.ranks = [Rank('first', 0, 1, 100000, 100000), Rank('second', 1, 4, 100000, 100000)]
        self.destination = self.root / 'bundle'

    def build(self, **kwargs):
        return build_bundle(self.source, self.destination, self.parent, self.index,
                            self.catalogue, self.ranks, scratch_reserve_bytes=0, **kwargs)

    def test_every_selected_tensor_is_identical_and_global_names_are_preserved(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        manifest = self.build(chunk_bytes=3)  # force every tensor across chunks
        receipt = verify_bundle(self.destination, expected_manifest_sha256=digest(manifest))
        self.assertEqual(receipt['manifest'], manifest)
        self.assertEqual(receipt['verified_ranks'], ['first', 'second'])
        for rank, record in zip(self.ranks, manifest['ranks'], strict=True):
            expected = {name: data for name, data in self.originals.items()
                        if not name.startswith('model.layers.') or
                        rank.start <= int(name.split('.')[2]) < rank.end}
            self.assertEqual(rank_tensors(self.destination / record['file']), expected)
            self.assertEqual(set(record['required_tensor_names']), set(expected))
            self.assertFalse((self.destination / rank.name / 'model.safetensors.index.json').exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.assertEqual(manifest['loader_status'], 'requires_qualified_shard_aware_adapter')

    def test_corrupt_parent_payload_never_publishes_verified_bundle(self):
        path = self.source / self.parent['weight_files'][-1]['name']
        bad = bytearray(path.read_bytes())
        bad[-1] ^= 255
        path.write_bytes(bad)
        with self.assertRaisesRegex(ValueError, 'parent file SHA256'):
            self.build(chunk_bytes=7)
        self.assertFalse((self.destination / 'VERIFIED').exists())
        self.assertEqual(path.read_bytes(), bad)
        with self.assertRaises(FileNotFoundError):
            verify_bundle(self.destination, expected_manifest_sha256='0' * 64)

    def test_wrong_parent_header_metadata_cannot_silently_move_tensor_extents(self):
        self.catalogue[next(iter(self.catalogue))]['header_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'tensor header differs'):
            self.build()
        self.assertFalse((self.destination / 'VERIFIED').exists())

    def test_interrupted_publish_has_no_verification_marker(self):
        with patch('rank_bundle.os.replace', side_effect=OSError('simulated publication failure')):
            with self.assertRaisesRegex(OSError, 'publication failure'):
                self.build()
        self.assertFalse((self.destination / 'VERIFIED').exists())

    def test_existing_destination_is_never_overwritten(self):
        self.destination.mkdir()
        existing = self.destination / 'keep'
        existing.write_bytes(b'existing operator artifact')
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(existing.read_bytes(), b'existing operator artifact')

    def test_derived_corruption_fails_even_with_existing_verified_marker(self):
        manifest = self.build()
        path = self.destination / manifest['ranks'][0]['file']
        data = bytearray(path.read_bytes())
        data[-1] ^= 255
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'derived rank bytes'):
            verify_bundle(self.destination, expected_manifest_sha256=digest(manifest))

    def test_self_consistent_other_bundle_cannot_replace_the_requested_manifest(self):
        self.build()
        with self.assertRaisesRegex(ValueError, 'not the requested immutable'):
            verify_bundle(self.destination, expected_manifest_sha256='0' * 64)

    def test_host_copy_verifies_only_its_assigned_rank_with_the_full_pinned_manifest(self):
        manifest = self.build()
        shutil.rmtree(self.destination / 'first')
        receipt = verify_bundle(self.destination, expected_manifest_sha256=digest(manifest), rank_name='second')
        self.assertEqual(receipt['verified_ranks'], ['second'])
        with self.assertRaises(FileNotFoundError):
            verify_bundle(self.destination, expected_manifest_sha256=digest(manifest))
        with self.assertRaisesRegex(ValueError, 'not in the pinned manifest'):
            verify_bundle(self.destination, expected_manifest_sha256=digest(manifest), rank_name='foreign')

    def test_insufficient_disk_headroom_refuses_before_writing(self):
        with patch('rank_bundle.shutil.disk_usage', return_value=type('Disk', (), {'free': 0})()):
            with self.assertRaisesRegex(ValueError, 'explicit scratch reserve'):
                self.build()
        self.assertFalse(self.destination.exists())

    def test_cli_plan_build_and_scoped_verify_join_the_same_parent(self):
        values = {'parent': self.parent, 'index': self.index,
                  'catalogue': {'model_id': self.parent['model_id'], 'revision': self.parent['revision'],
                                'files': self.catalogue},
                  'placement': {'ranks': [rank.__dict__ for rank in self.ranks]}}
        arguments = []
        for name, value in values.items():
            path = self.root / f'{name}.json'
            path.write_text(json.dumps(value))
            arguments += ['--' + name, str(path)]
        cli = [sys.executable, str(Path(__file__).with_name('stage.py'))]
        planned = subprocess.run(cli + ['plan', *arguments, '--context', '100', '--kv-bits', '8'],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(planned.stdout)['cache_bits'], 8)
        built = subprocess.run(cli + ['build', *arguments, '--source', str(self.source),
                               '--destination', str(self.destination), '--scratch-reserve-bytes', '0'],
                               capture_output=True, text=True, check=True)
        expected = json.loads(built.stdout)['manifest_sha256']
        checked = subprocess.run(cli + ['verify', str(self.destination), '--rank', 'second',
                                 '--expected-manifest-sha256', expected],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(checked.stdout)['verified_ranks'], ['second'])

    def test_incomplete_or_overlapping_placement_is_rejected_before_writing(self):
        self.ranks[1] = Rank('second', 2, 4, 100000, 100000)
        with self.assertRaisesRegex(ValueError, 'gap, overlap'):
            self.build()
        self.assertFalse(self.destination.exists())

    def test_unreviewed_model_family_is_rejected_before_writing(self):
        self.parent['config']['model_type'] = 'unreviewed'
        with self.assertRaisesRegex(ValueError, 'only the reviewed Qwen3'):
            self.build()
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
