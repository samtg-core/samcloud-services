"""Host-local, read-only file observations for an exact staging policy.

Standard library only: usable through an authenticated host channel before a
lease is requested. It hashes bytes, not tensors, and imports no MLX or EXO.
The coordinator validates host identity and decides admission separately.
"""

import ctypes
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import stat
import sys
import subprocess
from uuid import UUID


class StagingFilesUnavailable(ValueError):
    pass


class _Statfs64(ctypes.Structure):
    # Darwin SDK sys/mount.h __DARWIN_STRUCT_STATFS64, 64-bit inode ABI.
    _fields_ = [('block_size', ctypes.c_uint32), ('io_size', ctypes.c_int32)] + [
        (name, ctypes.c_uint64) for name in ('blocks', 'free_blocks', 'available_blocks',
                                           'files', 'free_files')] + [
        ('fsid', ctypes.c_int32 * 2), ('owner', ctypes.c_uint32),
        ('type', ctypes.c_uint32), ('flags', ctypes.c_uint32),
        ('subtype', ctypes.c_uint32), ('filesystem', ctypes.c_char * 16),
        ('mountpoint', ctypes.c_char * 1024), ('device', ctypes.c_char * 1024),
        ('extended_flags', ctypes.c_uint32), ('reserved', ctypes.c_uint32 * 7)]


def darwin_filesystem(directory):
    if sys.platform != 'darwin':
        raise StagingFilesUnavailable('qualified Darwin filesystem observation required')
    library = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
    function = library.fstatfs64
    function.argtypes, function.restype = [ctypes.c_int, ctypes.POINTER(_Statfs64)], ctypes.c_int
    result = _Statfs64()
    if function(directory, ctypes.byref(result)) != 0:
        raise StagingFilesUnavailable('filesystem observation failed')
    # MNT_IGNORE_OWNERSHIP (also MNT_UNKNOWNPERMISSIONS), Darwin SDK
    # sys/mount.h: st_uid cannot establish the required service ownership.
    if result.flags & 0x00200000:
        raise StagingFilesUnavailable('staging filesystem ignores ownership')
    return {'local': bool(result.flags & 0x1000),  # MNT_LOCAL, SDK sys/mount.h
            'filesystem': result.filesystem.decode('ascii'),
            'filesystem_id': list(result.fsid),
            'free_bytes': result.available_blocks * result.block_size}


def darwin_boot_id():
    if sys.platform != 'darwin':
        raise StagingFilesUnavailable('qualified host boot observation required')
    raw = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'kern.bootsessionuuid'], text=True).strip()
    return str(UUID(raw))


def collect_host(spec, previous=None, *, filesystem=darwin_filesystem, boot_reader=darwin_boot_id):
    """Run on the authenticated named host; this function mutates no resource."""
    boot, uid = boot_reader(), os.getuid()
    cached = previous if (isinstance(previous, dict) and previous.get('boot_id') == boot
        and previous.get('uid') == uid and previous.get('rank') == spec['rank']
        and previous.get('resource_id') == spec['resource_id']) else {}
    return {'rank': spec['rank'], 'resource_id': spec['resource_id'], 'uid': uid, 'boot_id': boot,
            'model': collect_files(spec['model'], filesystem=filesystem, previous=cached.get('model')),
            'runtime': collect_files(spec['runtime'], filesystem=filesystem, previous=cached.get('runtime'))}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def _parts(path, *, absolute=False):
    if not isinstance(path, str) or not path or '\x00' in path:
        raise StagingFilesUnavailable('exact staging path required')
    parsed = PurePosixPath(path)
    if (parsed.is_absolute() != absolute or str(parsed) != path
            or any(part in ('.', '..') for part in parsed.parts) or path in ('/', '.')):
        raise StagingFilesUnavailable('unredirected staging path required')
    return parsed.parts[1:] if absolute else parsed.parts


def _directory(parent, parts, seen=None):
    result = os.dup(parent)
    prefix = []
    try:
        for part in parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=result)
            os.close(result)
            result = child
            prefix.append(part)
            if seen is not None:
                name, identity = '/'.join(prefix), _identity(os.fstat(result))
                if name in seen and seen[name] != identity:
                    raise StagingFilesUnavailable('staging directory changed during observation')
                seen[name] = identity
        return result
    except BaseException:
        os.close(result)
        raise


def _identity(info):
    return {'device': info.st_dev, 'inode': info.st_ino, 'bytes': info.st_size,
            'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns,
            'mode': stat.S_IMODE(info.st_mode), 'uid': info.st_uid}


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {'root', 'files', 'views'}:
        raise StagingFilesUnavailable('exact staging file policy required')
    parts = _parts(policy['root'], absolute=True)
    files, views = policy['files'], policy['views']
    if not isinstance(files, list) or not files or not isinstance(views, dict):
        raise StagingFilesUnavailable('exact file and loading-view sets required')
    names = set()
    for pin in files:
        if (not isinstance(pin, dict) or set(pin) != {'path', 'bytes', 'sha256', 'link'}
                or type(pin['bytes']) is not int or pin['bytes'] < 0
                or not isinstance(pin['sha256'], str)
                or not re.fullmatch('[a-f0-9]{64}', pin['sha256'])):
            raise StagingFilesUnavailable('invalid staging file identity')
        _parts(pin['path'])
        if pin['path'] in names:
            raise StagingFilesUnavailable('duplicate staging file identity')
        names.add(pin['path'])
        if pin['link'] is not None and len(_parts(pin['link'])) != 1:
            raise StagingFilesUnavailable('only exact same-directory file aliases allowed')
    for view, entries in views.items():
        if view != '.':  # Explicit root loading view, never a file path alias.
            _parts(view)
        if (not isinstance(entries, list) or any(not isinstance(name, str)
                or len(_parts(name)) != 1 for name in entries) or len(set(entries)) != len(entries)):
            raise StagingFilesUnavailable('invalid exact loading view')
    return parts, files, views


def collect_files(policy, *, filesystem=darwin_filesystem, previous=None):
    """Fresh read-only facts; previous is the trusted owner's verified cache.

    Unchanged kernel file identities can reuse hashes after the first full read.
    This is not lease or execution permission. The native loader independently
    hashes all assigned bytes immediately before each model load.
    """
    parts, files, views = validate_policy(policy)
    slash = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    policy_sha256 = hashlib.sha256(_canonical(policy)).hexdigest()
    cached = {item['path']: item for item in previous.get('files', [])
              if isinstance(item, dict) and isinstance(item.get('path'), str)} if (
        isinstance(previous, dict) and previous.get('policy_sha256') == policy_sha256) else {}
    try:
        root = _directory(slash, parts)
    finally:
        os.close(slash)
    try:
        before = _identity(os.fstat(root))
        volume = filesystem(root)
        readonly = not bool(before['mode'] & 0o222)
        directories = {}
        records = []
        content_cache = {}
        hashed_bytes = 0
        for pin in files:
            parts = _parts(pin['path'])
            parent = _directory(root, parts[:-1], directories)
            try:
                readonly = readonly and not bool(os.fstat(parent).st_mode & 0o222)
                leaf = parts[-1]
                linked = stat.S_ISLNK(os.stat(leaf, dir_fd=parent, follow_symlinks=False).st_mode)
                if linked != (pin['link'] is not None):
                    raise StagingFilesUnavailable('unexpected staging file redirection')
                if linked and os.readlink(leaf, dir_fd=parent) != pin['link']:
                    raise StagingFilesUnavailable('staging alias target changed')
                opened = pin['link'] if linked else leaf
                descriptor = os.open(opened, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                     dir_fd=parent)
                try:
                    info = os.fstat(descriptor)
                    identity = _identity(info)
                    if (not stat.S_ISREG(info.st_mode) or info.st_size != pin['bytes']
                            or info.st_dev != before['device']):
                        raise StagingFilesUnavailable('staging file differs or crosses a mount')
                    prior = cached.get(pin['path'])
                    content_key = (tuple(sorted(identity.items())), pin['sha256'])
                    reused = (prior is not None and prior.get('identity') == identity
                              and prior.get('sha256') == pin['sha256']
                              and prior.get('link') == pin['link'])
                    reused = reused or content_key in content_cache
                    actual_hash = pin['sha256'] if reused else None
                    if not reused:
                        hasher = hashlib.sha256()
                        while data := os.read(descriptor, 4 * 1024 * 1024):
                            hasher.update(data)
                            hashed_bytes += len(data)
                        actual_hash = hasher.hexdigest()
                    if (_identity(os.fstat(descriptor)) != identity
                            or actual_hash != pin['sha256']):
                        raise StagingFilesUnavailable('staging file changed or hash differs')
                    content_cache[content_key] = actual_hash
                    readonly = readonly and not bool(identity['mode'] & 0o222)
                    records.append({'path': pin['path'], 'sha256': actual_hash,
                                    'link': pin['link'], 'identity': identity})
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent)
        for view, entries in views.items():
            directory = _directory(root, () if view == '.' else _parts(view), directories)
            try:
                if set(os.listdir(directory)) != set(entries):
                    raise StagingFilesUnavailable('loading view contains missing or extra files')
            finally:
                os.close(directory)
        if _identity(os.fstat(root)) != before:
            raise StagingFilesUnavailable('staging root changed during observation')
        for name, identity in directories.items():
            directory = _directory(root, _parts(name))
            try:
                if _identity(os.fstat(directory)) != identity:
                    raise StagingFilesUnavailable('staging directory changed during observation')
                readonly = readonly and not bool(identity['mode'] & 0o222)
            finally:
                os.close(directory)
        return {'policy_sha256': policy_sha256,
                'root': policy['root'], 'uid': os.getuid(), 'root_identity': before,
                'filesystem': volume, 'readonly': readonly, 'directories': directories,
                'files': records, 'hashed_bytes': hashed_bytes}
    finally:
        os.close(root)
