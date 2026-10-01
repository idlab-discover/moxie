"""Family development assembly: live owned sources, immutable external inputs.

Uses the versioned namespace validator for namespace, baseline and collision checks.
External snapshots contain exactly the locked bytes, recovered from the source
checkout or its Git history, never silently updated to the owner's HEAD.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

LEGACY_SHA256 = '37b87a23e9006c4c16925733709a242818ca8a27f150ba0ad4c322ba019ac276'


def legacy():
    path = Path(__file__).with_name('family_validator.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != LEGACY_SHA256:
        raise ValueError('family validator drift; review the family assembler adoption')
    spec = importlib.util.spec_from_file_location('moxie_assembly_v1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repository(path):
    return Path(subprocess.check_output(
        ['git', '-C', str(path), 'rev-parse', '--show-toplevel'],
        stderr=subprocess.PIPE, text=True).strip()).resolve()


def recover(api, source, name, expected, revision):
    """Recover an exact content pin, including pins made from uncommitted files."""
    path = source / api.relative(name)
    if path.is_file():
        data = api.source_file(source, name).read_bytes()
        if api.digest(data) == expected:
            return data
    top = repository(source)
    repo_name = (source.relative_to(top) / name).as_posix()
    revisions = [revision]
    revisions += api.git(top, 'log', '--all', '--format=%H', '--', repo_name).decode().splitlines()
    for commit in dict.fromkeys(revisions):
        try:
            data = api.git(top, 'show', f'{commit}:{repo_name}')
        except api.AssemblyError:
            continue
        if api.digest(data) == expected:
            return data
    raise api.AssemblyError(f'source drift: exact pinned bytes unavailable: {source}/{name} ({expected})')


def snapshot(api, project, source, cache):
    """Keep a private Git snapshot so baseline proofs need no live sibling checkout."""
    key = api.digest(json.dumps([str(source), project['revision'], project['files']], sort_keys=True).encode())
    target = cache / key
    metadata = target / 'snapshot.json'
    if metadata.exists():
        info = api.read_json(metadata.read_bytes())
        root = target / 'repository' / api.relative(info['subdirectory']) if info['subdirectory'] != '.' else target / 'repository'
    else:
        top = repository(source)
        subdirectory = source.relative_to(top).as_posix()
        target.mkdir(parents=True, exist_ok=True)
        destination = target / 'repository'
        if not destination.exists():
            subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout',
                            str(top), str(destination)], check=True, stdout=subprocess.DEVNULL)
        # Detach without checking out the whole owner: only pinned files are needed.
        api.git(destination, 'cat-file', '-e', project['revision'] + '^{commit}')
        api.git(destination, 'update-ref', '--no-deref', 'HEAD', project['revision'])
        info = {'source': str(source), 'revision': project['revision'], 'subdirectory': subdirectory}
        metadata.write_text(json.dumps(info, indent=2) + '\n')
        root = destination / subdirectory
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in project['files'].items():
        api.sha(expected)
        path = root / api.relative(name)
        api.require(path.resolve().is_relative_to(root.resolve()), 'snapshot symlink escape')
        if path.is_file() and api.digest(path.read_bytes()) == expected:
            continue
        data = recover(api, source, name, expected, project['revision'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def assemble(consumer, lock_path, output):
    api = legacy()
    consumer, lock_path = Path(consumer).resolve(), Path(lock_path).resolve()
    family = repository(consumer)
    original = lock_path.read_bytes()
    lock = api.read_json(original)
    api.fields(lock, ('schema', 'projects', 'overrides'))
    api.require(type(lock['schema']) is int and lock['schema'] == 1, 'unsupported lock schema')
    api.require(isinstance(lock['projects'], list) and lock['projects'], 'projects must be nonempty')
    cache = family / 'builds' / 'family-dependencies'
    api.require(cache.resolve() == cache, 'dependency cache symlink escape')
    cache.mkdir(parents=True, exist_ok=True)
    # Serialize snapshot creation across children of the same family.
    import fcntl
    with (cache / '.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        for project in lock['projects']:
            api.fields(project, ('id', 'path', 'revision', 'files', 'modules', 'assets', 'exports'))
            source = (lock_path.parent / project['path']).resolve()
            try:
                owned = repository(source) == family
            except subprocess.CalledProcessError:
                owned = False
            if owned:
                # Manifests still declare module ownership; hashes of live family
                # files are recorded in the generated receipt, not development gates.
                project['revision'] = api.git(source, 'rev-parse', 'HEAD').decode().strip()
                for name in project['files']:
                    project['files'][name] = api.digest(api.source_file(source, name).read_bytes())
                project['path'] = str(source)
            else:
                # The lock already flattens ownership and exports. Upstream
                # instruction files and nested locks are provenance, not compiler
                # inputs; their incidental edits must not gate this family.
                consumed = {spec['file'] for spec in project['modules'].values()}
                consumed.update(project['assets'].values())
                if project['id'] == 'assembly':
                    consumed.update(name for name in project['files'] if name.startswith('tools/'))
                for name in consumed:
                    api.require(name in project['files'], f'unpinned file: {project["id"]}/{name}')
                project['files'] = {name: value for name, value in project['files'].items() if name in consumed}
                project['path'] = str(snapshot(api, project, source, cache))
        output = Path(output).absolute()
        api.require(output == consumer / 'builds/deps', 'output must be consumer/builds/deps')
        output.parent.mkdir(parents=True, exist_ok=True)
        # The resolved manifest is a build artifact; source locks remain unchanged.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=output.parent) as resolved:
            json.dump(lock, resolved)
            resolved.flush()
            result = api.assemble(consumer, resolved.name, output)
        receipt_path = result / 'ASSEMBLY.json'
        receipt = api.read_json(receipt_path.read_bytes())
        receipt['development'] = {'family': str(family), 'source_lock_sha256': api.digest(original)}
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumer', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(assemble(args.consumer, args.lock, args.output))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'family assembly: {exc}\n')


if __name__ == '__main__':
    main()
