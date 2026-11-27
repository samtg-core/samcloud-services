"""Actual private HTTP-handler controls, MLX default CPU, no model or sockets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from run_exo_instance_types import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exo-source', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = args.exo_source.resolve()
    pins = json.loads((here / 'fixtures/exo-source-pins.json').read_bytes())
    for name, expected in pins['files'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError('EXO qualification source changed: ' + name)
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    import pytest
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    sys.path.insert(0, str(here.parent))
    sys.path.insert(0, str(source / 'src'))
    sys.addaudithook(audit)
    with tempfile.TemporaryDirectory(prefix='exo-private-handler-controls-') as private:
        os.chmod(private, 0o700)
        os.environ['EXO_HOME'] = private
        os.environ['EXO_DASHBOARD_DIR'] = str(source / 'dashboard')
        os.environ['EXO_RESOURCES_DIR'] = str(source / 'resources')
        result = pytest.main(['-p', 'pytest_asyncio.plugin', '-o', 'asyncio_mode=auto', '-q',
            str(here / 'test_exo_private_api_types.py')])
        if mx.default_device() != mx.cpu:
            raise AssertionError('CPU-only handler controls changed default device')
        raise SystemExit(result)


if __name__ == '__main__':
    main()
