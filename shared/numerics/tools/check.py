"""Verify the pinned candidate before any native build; no parent assembly."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'DEPENDENCIES.lock.json').read_text())

def verify():
    (ROOT / 'docs/private/numojo').mkdir(parents=True, exist_ok=True)
    for metadata in Path(sys.prefix, 'conda-meta').glob('max-core-*.json'):
        if json.loads(metadata.read_text())['version'] != LOCK['toolchain']['max-core']:
            raise RuntimeError('MAX toolchain mismatch')
        break
    else:
        raise RuntimeError('Locked max-core package is missing')
    dep = LOCK['numojo']
    source = (ROOT / dep['path']).resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != dep['revision']:
        raise RuntimeError('NuMojo revision drift')
    target = ROOT / 'builds/deps'
    for name, digest in dep['files'].items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'NuMojo source drift: {name}')
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.read_bytes() != data:
            dest.write_bytes(data)
    version = subprocess.check_output(['mojo', '--version'], text=True)
    if '1.0.0' not in version:
        raise RuntimeError(f'Compiler mismatch: {version}')
    return target

def build():
    deps = verify()
    for source in sorted((ROOT / 'tests').glob('*.mojo')) + sorted((ROOT / 'examples').glob('*.mojo')):
        subprocess.run(['mojo', 'build', '-O3', '-I', str(deps), '-I', str(ROOT/'src'), '-I', str(ROOT/'tests'), str(source), '-o', str(ROOT/'builds'/source.stem)], check=True)

if __name__ == '__main__':
    action = sys.argv[1]
    build()
    if action == 'test':
        subprocess.run([str(ROOT/'builds/contracts')], check=True)
        subprocess.run([str(ROOT/'builds/edges')], check=True)
        subprocess.run([str(ROOT/'builds/exchange')], check=True)
        subprocess.run([sys.executable, str(ROOT/'tests/negative_contracts.py')], check=True)
    elif action == 'parity':
        subprocess.run([sys.executable, str(ROOT/'tests/parity.py')], check=True)
