"""Resolve dependency HEADs and assemble current local source files."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MOXIE = ROOT.parents[1]


def moxie_head():
    return subprocess.check_output(
        ['git', '-C', str(MOXIE), 'rev-parse', 'HEAD'], text=True).strip()


def head_project(project, source, revision):
    files = set(project['files'])
    files.update(m['file'] for m in project['modules'].values())
    files.update(project['assets'].values())
    hashes = {}
    for filename in sorted(files):
        relative = Path(filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'invalid dependency path: {filename}')
        path = source / relative
        if not path.is_file():
            raise ValueError(f'missing dependency source: {path}')
        hashes[filename] = digest(path.read_bytes())
    project['files'] = hashes
    project['revision'] = revision
    project['path'] = str(source)
    for module in project['modules'].values():
        module.pop('baseline', None)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(root, lock, revision=None):
    revision = revision or moxie_head()
    data = json.loads(lock.read_text())
    for project in data['projects']:
        source = (lock.parent / project['path']).resolve()
        if source == ROOT or source in [ROOT / 'tests/consumers/a', ROOT / 'tests/consumers/b']:
            project['path'] = str(source)
            if source == ROOT and project['id'] == 'semi-supervised':
                project['modules'] = {'moxie.semi_supervised.' + p.stem: {'file': str(p.relative_to(ROOT))}
                                      for p in (ROOT / 'src/moxie/semi_supervised').glob('*.mojo') if p.stem != '__init__'}
                project['exports'] = json.loads((ROOT / 'EXPORTS.json').read_text())
            project['files'] = {m['file']: digest((source / m['file']).read_bytes()) for m in project['modules'].values()}
            continue
        if source.is_relative_to(MOXIE):
            head_project(project, source, revision)
            continue
        dependency_revision = subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        head_project(project, source, dependency_revision)
    target = root / 'builds/effective-lock.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + '\n')
    return target, next(Path(p['path']) for p in data['projects'] if p['id'] == 'assembly')
