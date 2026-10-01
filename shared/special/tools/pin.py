"""Explicit refresh of owned special function files; preserve every external dependency pin."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main():
    lock = ROOT / 'DEPENDENCIES.lock.json'
    data = json.loads(lock.read_text())
    external = [p for p in data['projects'] if p['id'] != 'special']
    revision = subprocess.check_output(['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
    owned = ['src/moxie/_special.mojo', 'src/moxie/__init__.mojo', 'EXPORTS.json', 'LICENSES/scikit-learn.txt', 'LICENSES/fdlibm.txt']
    for consumer in (ROOT, ROOT / 'tests/consumers/a', ROOT / 'tests/consumers/b'):
        projects = json.loads(json.dumps(external))
        for project in projects:
            project['path'] = os.path.relpath((ROOT / project['path']).resolve(), consumer)
        files = owned if consumer == ROOT else owned + ['DEPENDENCIES.lock.json', 'pixi.lock', 'docs/special.md', 'docs/scope.md']
        projects.append(dict(id='special', path=os.path.relpath(ROOT, consumer), revision=revision,
                             files={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                             modules={'moxie._special': {'file': 'src/moxie/_special.mojo'}},
                             assets={'licenses/special-sklearn.txt': 'LICENSES/scikit-learn.txt', 'licenses/fdlibm.txt': 'LICENSES/fdlibm.txt'},
                             exports=json.loads((ROOT/'EXPORTS.json').read_text())))
        write(consumer/'DEPENDENCIES.lock.json', dict(schema=1, projects=projects, overrides=[]))


if __name__ == '__main__':
    main()
