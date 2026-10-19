"""Run after pixi run parity. Use pyperf's standard process/calibration runner."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import subprocess
import pyperf
from tree_checks import ROOT, FLAGS, compile_storage as compile
from check import run

def assemble():
    import sys
    run(sys.executable, ROOT / "shared/assembly/tools/assemble_family.py",
        "--consumer", ROOT, "--lock", ROOT / "DEPENDENCIES.json", "--output", ROOT / "builds/deps")


def elapsed(loops, rows):
    output = subprocess.check_output(
        [str(ROOT/'builds/tree/storage/benchmark'), str(rows), str(loops)], text=True)
    seconds, checksum = map(float, output.split())
    assert checksum > 0
    return seconds


if __name__ == '__main__':
    import sys
    if '--worker' not in sys.argv:
        assemble()
        compile(ROOT/'tests/tree/storage/benchmark.mojo', ROOT/'builds/tree/storage/benchmark')
        target = ROOT/'benchmarks/tree-storage'
        target.mkdir(parents=True, exist_ok=True)
        metadata = dict(platform=platform.platform(), cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                        flags=FLAGS, seed='none; deterministic alternating inputs',
                        compiler=subprocess.check_output(['mojo','--version'],text=True),
                        affinity=sorted(os.sched_getaffinity(0)), pyperf=pyperf.__version__,
                        source_sha256=hashlib.sha256((ROOT/'src/moxie/tree/_tree.mojo').read_bytes()).hexdigest(),
                        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},
                        timing='native region; includes validation, output allocation and checksum; excludes setup/startup')
        (target/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        with (target/'memory.txt').open('w') as stream:
            for rows in (1, 4096, 262144):
                stream.write(f'rows={rows}, 127 nodes, depth=6, Float32 input, Float64 output\n')
                stream.flush()
                subprocess.run([shutil.which('time'),'-v',str(ROOT/'builds/tree/storage/benchmark'),str(rows),'10'],
                               stdout=stream,stderr=stream,check=True)
    runner=pyperf.Runner()
    for rows in (1,4096,262144):
        runner.bench_time_func(f'predict_{rows}_rows_depth6',elapsed,rows)
