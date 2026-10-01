"""Family-mode namespace validator (Python build tooling only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile


class AssemblyError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AssemblyError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(data):
    return json.loads(data, object_pairs_hook=unique)


def fields(obj, required, optional=()):
    require(isinstance(obj, dict), 'expected object')
    require(set(required) <= obj.keys() <= set(required) | set(optional),
            f'invalid fields: expected {required}, received {list(obj)}')


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name, 'invalid relative path')
    parts = name.split('/')
    require(not PurePosixPath(name).is_absolute() and all(p not in ('', '.', '..') for p in parts),
            f'path traversal: {name}')
    return Path(*parts)


def source_file(root, name):
    path = root / relative(name)
    require(path.resolve().is_relative_to(root), f'source symlink escape: {name}')
    require(path.is_file(), f'missing file: {path}')
    return path


def module(name):
    require(isinstance(name, str) and re.fullmatch(r'moxie(?:\.[A-Za-z_][A-Za-z_0-9]*)*', name),
            f'invalid module: {name}')
    require('__init__' not in name.split('.'), 'initializer must use export manifest')
    return name


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA-256')


def git(root, *args):
    try:
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise AssemblyError(f'Git verification failed in {root}: {args}') from exc


def plan(consumer, lock_path):
    """Validate all inputs before writing anything; return immutable byte snapshots."""
    consumer = Path(consumer).resolve()
    lock_path = Path(lock_path).resolve()
    lock_bytes = lock_path.read_bytes()
    lock = read_json(lock_bytes)
    fields(lock, ('schema', 'projects', 'overrides'))
    require(type(lock['schema']) is int and lock['schema'] == 1, 'unsupported lock schema')
    require(isinstance(lock['projects'], list) and lock['projects'], 'projects must be nonempty')
    owners, candidates, assets, manifests = {}, {}, {}, []
    for project in lock['projects']:
        fields(project, ('id', 'path', 'revision', 'files', 'modules', 'assets', 'exports'))
        owner = project['id']
        require(isinstance(owner, str) and re.fullmatch('[A-Za-z_0-9.-]+', owner), 'invalid project id')
        require(owner not in owners, f'duplicate project: {owner}')
        require(isinstance(project['path'], str) and project['path'], 'invalid project path')
        root = (lock_path.parent / project['path']).resolve()
        revision = project['revision']
        require(isinstance(revision, str) and re.fullmatch('[0-9a-f]{40}', revision), 'invalid Git revision')
        head = git(root, 'rev-parse', 'HEAD').decode().strip()
        same_repository = git(root, 'rev-parse', '--show-toplevel') == git(consumer, 'rev-parse', '--show-toplevel')
        if same_repository:
            # Locks within one repository cannot pin the commit containing
            # themselves. Require provenance ancestry and exact file hashes.
            ancestry = subprocess.run(['git', '-C', str(root), 'merge-base',
                                      '--is-ancestor', revision, head],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            require(ancestry.returncode == 0, f'revision drift: {owner}')
        else:
            require(head == revision, f'revision drift: {owner}')
        owners[owner] = project
        require(isinstance(project['files'], dict), 'files must be an object')
        blobs = {}
        for name, expected in project['files'].items():
            sha(expected)
            data = source_file(root, name).read_bytes()
            require(digest(data) == expected, f'source drift: {owner}/{name}')
            blobs[name] = data
        def pinned(name):
            relative(name)
            require(name in blobs, f'unpinned file: {owner}/{name}')
            return blobs[name]
        require(isinstance(project['modules'], dict), 'modules must be an object')
        for name, spec in project['modules'].items():
            module(name)
            require(name != 'moxie', 'root must be an export package')
            fields(spec, ('file',), ('baseline',))
            data = pinned(spec['file'])
            baseline = spec.get('baseline')
            if baseline is not None:
                require(isinstance(baseline, str) and re.fullmatch('[0-9a-f]{40}', baseline), 'invalid baseline')
                # Verify genuine baseline provenance, not merely a caller-supplied label.
                top = Path(git(root, 'rev-parse', '--show-toplevel').decode().strip())
                repo_file = source_file(root, spec['file']).relative_to(top).as_posix()
                require(git(root, 'show', f'{baseline}:{repo_file}') == data, f'baseline drift: {name}')
            candidates.setdefault(name, []).append((owner, data, baseline))
        require(isinstance(project['assets'], dict), 'assets must be an object')
        for target, source in project['assets'].items():
            path = relative(target)
            require(path.parts[0] in ('numojo', 'mojo_linalg', 'mojo_optimize', 'numax', 'balsa', 'licenses'), f'unsupported asset namespace: {target}')
            data = pinned(source)
            require(target not in assets, f'asset collision: {target}')
            assets[target] = data
        if project['exports'] is not None:
            manifest = project['exports']
            fields(manifest, ('schema', 'packages'))
            require(type(manifest['schema']) is int and manifest['schema'] == 1, 'unsupported exports schema')
            require(isinstance(manifest['packages'], dict), 'packages must be an object')
            manifests.append((owner, manifest['packages']))
    overrides = {}
    require(isinstance(lock['overrides'], list), 'overrides must be a list')
    for override in lock['overrides']:
        fields(override, ('module', 'from', 'to', 'reason'))
        name = module(override['module'])
        require(name not in overrides, f'duplicate override: {name}')
        require(isinstance(override['from'], list) and override['from'] and
                all(isinstance(v, str) for v in override['from']), 'override from must name owners')
        require(len(set(override['from'])) == len(override['from']), 'duplicate override owner')
        require(isinstance(override['reason'], str) and override['reason'].strip(), 'override needs reason')
        require(isinstance(override['to'], str), 'invalid override winner')
        overrides[name] = override
    selected, displaced, ownership = {}, set(), {}
    for name, entries in candidates.items():
        ids = {entry[0] for entry in entries}
        if name in overrides:
            override = overrides.pop(name)
            require(set(override['from']) | {override['to']} == ids and override['to'] not in override['from'],
                    f'override owners mismatch: {name}')
            entries = [entry for entry in entries if entry[0] == override['to']]
            require(len(entries) == 1, f'override missing winner: {name}')
            displaced.update((owner, name) for owner in override['from'])
        elif len(entries) > 1:
            require(entries[0][2] is not None and
                    all(entry[1:] == entries[0][1:] for entry in entries), f'module collision: {name}')
        selected[name] = entries[0][1]
        ownership[name] = sorted(entry[0] for entry in entries)
    require(not overrides, f'unused overrides: {list(overrides)}')
    packages = {}
    for owner, manifest in manifests:
        for package, exports in manifest.items():
            module(package)
            require(isinstance(exports, dict), 'exports must be an object')
            merged = packages.setdefault(package, {})
            for symbol, target in exports.items():
                require(isinstance(symbol, str) and re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', symbol), 'invalid export symbol')
                module(target)
                require(target in owners[owner]['modules'], f'export lacks module ownership: {owner}/{target}')
                if (owner, target) in displaced:
                    continue
                require(target in selected, f'missing export module: {target}')
                if symbol in merged:
                    require(merged[symbol] == target and len(ownership[target]) > 1,
                            f'export collision: {package}.{symbol}')
                merged[symbol] = target
    require('moxie' in packages, 'missing moxie export manifest')
    for name in list(selected) + list(packages):
        parts = name.split('.')
        for i in range(1, len(parts)):
            require('.'.join(parts[:i]) in packages, f'missing package manifest: {name}')
    require(not (selected.keys() & packages.keys()), 'module/package collision')
    output = dict(assets)
    for name, data in selected.items():
        output[name.replace('.', '/') + '.mojo'] = data
    for package, exports in packages.items():
        content = '# Generated by foundation.assembly schema 1; edit source manifests.\n'
        content += ''.join(f'from {target} import {symbol}\n' for symbol, target in sorted(exports.items()))
        output[package.replace('.', '/') + '/__init__.mojo'] = content.encode()
    for name in output:
        for parent in PurePosixPath(name).parents:
            require(str(parent) not in output, f'file/directory collision: {name}')
    receipt = {'schema': 1, 'lock_sha256': digest(lock_bytes), 'modules': ownership,
               'files': {name: digest(data) for name, data in sorted(output.items())}}
    output['ASSEMBLY.json'] = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
    return output


def assemble(consumer, lock_path, output):
    consumer = Path(consumer).resolve()
    output = Path(output).absolute()
    allowed = consumer / 'builds' / 'deps'
    require(output == allowed, f'output must be {allowed}')
    require(output.resolve() == output and output.parent.resolve() == output.parent, 'output symlink escape')
    require(subprocess.run(['git', '-C', str(consumer), 'check-ignore', '-q', str(output / 'probe')]).returncode == 0,
            'builds/deps must be Git-ignored')
    snapshots = plan(consumer, lock_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Serialize cooperating writers; compilation must follow assembly, not race it.
    import fcntl
    descriptor = os.open(output.parent / '.assembly.lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        temporary = Path(tempfile.mkdtemp(prefix='.deps-', dir=output.parent))
        try:
            for name, data in snapshots.items():
                target = temporary / relative(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            if output.exists():
                shutil.rmtree(output)
            temporary.rename(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumer', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(assemble(args.consumer, args.lock, args.output))
    except (AssemblyError, OSError, ValueError, TypeError) as exc:
        parser.exit(1, f'assembly: {exc}\n')


if __name__ == '__main__':
    main()
