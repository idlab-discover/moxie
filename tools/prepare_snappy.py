"""Materialize exact adopted source bytes; never import the live owner's HEAD."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    pin = json.loads((ROOT / 'SNAPPY.lock.json').read_text())
    source = (ROOT / pin['path']).resolve()
    output = ROOT / 'builds/snappy'
    for name, digest in pin['files'].items():
        target = output / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        data = subprocess.check_output(['git', '-C', str(source), 'show', f"{pin['revision']}:{name}"])
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'Snappy pin mismatch: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return output / 'src'


if __name__ == '__main__':
    print(prepare())
