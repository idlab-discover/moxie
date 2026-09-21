"""Reproducible small full-assembly benchmark using pyperf and GNU time."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import tempfile
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'benchmarks'
    out.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='run-', dir=out))
    timer = shutil.which('time')
    if timer is None:
        raise RuntimeError('GNU time is required; install the locked Pixi environment')
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    commands = []
    for letter in ('a', 'b'):
        consumer = ROOT / 'tests/fixtures' / f'consumer_{letter}'
        command = [sys.executable, str(ROOT / 'tools/consume.py'), '--consumer', str(consumer),
                   '--lock', str(consumer / 'DEPENDENCIES.lock.json'), '--tool', str(ROOT / 'tools/assemble.py')]
        measure = [sys.executable, '-m', 'pyperf', 'command', '--processes', '2', '--values', '3',
                   '--warmups', '1', '--loops', '1', '--affinity', str(min(os.sched_getaffinity(0))),
                   '--copy-env', '--name', f'assembly-{letter}', '-o', str(out / f'assembly-{letter}.json'), *command]
        commands.append(measure)
        subprocess.run(measure, check=True, cwd=ROOT, env=environment)
        with (out / f'memory-{letter}.txt').open('w') as stream:
            subprocess.run([timer, '-v', *command], check=True, env=environment, stderr=stream)
    metadata = {'commands': commands, 'platform': platform.platform(), 'python': sys.version,
                'cpu': Path('/proc/cpuinfo').read_text(), 'compiler_flags': '-O3 (native probes only)',
                'mojo': subprocess.check_output(['mojo', '--version'], text=True), 'seed': 'none',
                'threads': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                'lock_sha256': hashlib.sha256((ROOT / 'DEPENDENCIES.lock.json').read_bytes()).hexdigest(),
                'tool_sha256': hashlib.sha256((ROOT / 'tools/assemble.py').read_bytes()).hexdigest()}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
