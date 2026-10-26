"""Pure rollout planning for native queue consumers; never edits a live home.

Apply only under a separate delivery maintenance boundary, with exact before
hashes and preserved histories. Source staging alone is not activation.
"""
import ast
from copy import deepcopy
from pathlib import Path


COMPONENTS = ('samcloud-queue-provider','samcloud-queue-hooks','samcloud-queue-context')


def profile_plan(config):
    if not isinstance(config,dict):
        raise ValueError('Explicit native profile mapping required')
    planned = deepcopy(config)
    model = planned.get('model')
    if (not isinstance(model,dict) or model.get('provider') != 'custom' or
            not isinstance(model.get('default'),str) or not model['default'].strip() or
            not isinstance(model.get('base_url'),str) or not model['base_url'].startswith('https://')):
        raise ValueError('Qualified custom model/endpoint required')
    context = planned.setdefault('context',{})
    plugins = planned.setdefault('plugins',{})
    compression = planned.setdefault('compression',{})
    auxiliary = planned.setdefault('auxiliary',{})
    if any(not isinstance(value,dict) for value in (context,plugins,compression,auxiliary)):
        raise ValueError('Native component configuration must be mappings')
    if context.get('engine') not in (None,'compressor'):
        raise ValueError('Existing custom context engine requires separate qualification')
    enabled = plugins.get('enabled',[])
    if not isinstance(enabled,list) or any(not isinstance(value,str) for value in enabled):
        raise ValueError('Explicit native plugin list required')
    if enabled and plugins.get('isolation') != 'in_process':
        raise ValueError('Do not change another plugin isolation policy')
    summary = auxiliary.setdefault('compression',{})
    if not isinstance(summary,dict):
        raise ValueError('Compression route must be a mapping')
    if summary.get('model') not in (None,'',model['default']):
        raise ValueError('Different auxiliary model requires separate capacity qualification')
    model['provider'] = 'samcloud-queue-stint'
    context['engine'] = 'samcloud-queue-context'
    plugins.update(isolation='in_process',enabled=list(dict.fromkeys([*enabled,'samcloud-queue-hooks'])))
    # Native supported setting: no idle/total-clock abandonment of queued
    # summary work. This does not change its threshold, summary algorithm or
    # accepted-row/stop/commit fences.
    compression['context_timeout_seconds'] = 0
    # Native auxiliary config treats base_url+api_key as anonymous custom,
    # even with a named provider. Leave credentials to this registered
    # provider's existing SAMCLOUD_QUEUE_API_KEY binding in the wrapper.
    summary.pop('api_key',None)
    summary.update(provider='samcloud-queue-aux',model=model['default'],
                   base_url=model['base_url'])
    vision=auxiliary.get('vision')
    if (isinstance(vision,dict) and vision.get('provider')=='custom'
            and vision.get('base_url','').rstrip('/')==model['base_url'].rstrip('/')):
        vision.pop('api_key',None)
        vision.update(provider='samcloud-queue-tool')
    return planned


def wrapper_plan(source,source_root,*,activation_manifest=None):
    root = Path(source_root)
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('Explicit absolute adapter source required')
    tree = ast.parse(source)
    terminals = [node for node in tree.body if isinstance(node,ast.Expr) and
        isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and
        isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='os' and
        node.value.func.attr=='execv']
    if len(terminals) != 1 or terminals[0] is not tree.body[-1]:
        raise ValueError('Qualified final os.execv launch boundary required')
    if 'SAMCLOUD_QUEUE_API_KEY' in source or 'SAMCLOUD_QUEUE_BASE_URL' in source:
        raise ValueError('Wrapper already has queue bindings; reconcile rather than double-install')
    guard=''
    if activation_manifest is not None:
        manifest=Path(activation_manifest)
        if not manifest.is_absolute() or '..' in manifest.parts:
            raise ValueError('Explicit absolute activation manifest required')
        guard='''# An incomplete profile transaction is not permission to start a client.
import hashlib as _queue_hashlib, json as _queue_json
from pathlib import Path as _QueuePath
try:
    _queue_activation = _queue_json.loads(_QueuePath(ACTIVATION_MANIFEST).read_text())
    if (_queue_activation.get('state') != 'active' or
            _queue_activation.get('source_path') != ADAPTER_SOURCE or
            _queue_hashlib.sha256(_QueuePath(__file__).read_bytes()).hexdigest() !=
                _queue_activation.get('wrapper_sha256') or
            _queue_hashlib.sha256(_QueuePath(_queue_activation['config_path']).read_bytes()).hexdigest() !=
                _queue_activation.get('config_sha256')):
        raise ValueError('activation mismatch')
    _queue_files = _queue_activation.get('source_files')
    if not isinstance(_queue_files,dict) or not _queue_files:
        raise ValueError('source inventory missing')
    for _queue_name,_queue_digest in _queue_files.items():
        _queue_relative = _QueuePath(_queue_name)
        if (_queue_relative.is_absolute() or '..' in _queue_relative.parts or
                _queue_hashlib.sha256((_QueuePath(ADAPTER_SOURCE)/_queue_relative).read_bytes()).hexdigest() != _queue_digest):
            raise ValueError('adapter source changed')
except Exception:
    raise SystemExit('Queue activation incomplete or changed; no native client started') from None
'''.replace('ACTIVATION_MANIFEST',repr(str(manifest))).replace('ADAPTER_SOURCE',repr(str(root)))
    insert = '''# Queue adapter edge: existing seat credential, never another identity's token.
os.environ['SAMCLOUD_QUEUE_API_KEY'] = os.environ['OPENAI_API_KEY']
os.environ['SAMCLOUD_QUEUE_BASE_URL'] = os.environ['OPENAI_BASE_URL']
os.environ['PYTHONPATH'] = ADAPTER_SOURCE + os.pathsep + os.environ.get('PYTHONPATH','')
'''.replace('ADAPTER_SOURCE',repr(str(root)))
    lines = source.splitlines(keepends=True)
    lines.insert(terminals[0].lineno-1,guard+insert)
    planned = ''.join(lines)
    compile(planned,'planned-hermes-samcloud','exec')
    return planned
