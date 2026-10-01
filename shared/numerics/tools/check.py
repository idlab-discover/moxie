"""Build current shared numerical sources with the locked toolchain."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY = ROOT.parent / 'assembly'

def verify():
    (ROOT / 'docs/private/numojo').mkdir(parents=True, exist_ok=True)
    for metadata in Path(sys.prefix, 'conda-meta').glob('max-core-*.json'):
        if json.loads(metadata.read_text())['version'] != '26.5.0':
            raise RuntimeError('MAX toolchain mismatch')
        break
    else:
        raise RuntimeError('Locked max-core package is missing')
    target = ROOT / 'builds/deps'
    subprocess.run([sys.executable, str(ASSEMBLY / 'tools/assemble.py'),
                    '--consumer', str(ROOT), '--lock', str(ROOT / 'DEPENDENCIES.lock.json'),
                    '--output', str(target)], check=True)
    version = subprocess.check_output(['mojo', '--version'], text=True)
    if '1.0.0' not in version:
        raise RuntimeError(f'Compiler mismatch: {version}')
    return target

def build():
    deps = verify()
    for source in sorted((ROOT / 'tests').glob('*.mojo')) + sorted((ROOT / 'examples').glob('*.mojo')):
        subprocess.run(['mojo', 'build', '-O3', '-I', str(deps), '-I', str(ROOT/'tests'), str(source), '-o', str(ROOT/'builds'/source.stem)], check=True)

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
