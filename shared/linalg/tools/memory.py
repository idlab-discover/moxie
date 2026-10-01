"""Whole-process RSS through GNU time; raw reports include runtime and setup."""
from pathlib import Path
import json
import os
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT/'benchmarks/memory'
    directory.mkdir(parents=True, exist_ok=True)
    time_binary = shutil.which("time")
    assert time_binary, "GNU time is required by the pinned environment"
    records = []
    for bits in (32,64):
        for layout in ('contiguous','strided'):
            checksums = []
            for mode in ('backend','adapter'):
                name = f'{mode}-{bits}-{layout}'
                command = [time_binary,'-v','-o',str(directory/f'{name}.txt'),
                           str(ROOT/'builds/benchmark'),str(bits),mode,'512','1',layout]
                out = subprocess.check_output(command,text=True)
                (directory/f'{name}.stdout').write_text(out)
                checksums.append(float(out.splitlines()[1]))
                records.append(dict(command=command,stdout=out))
            assert checksums[0] == checksums[1], (bits,layout,checksums)
    report = dict(platform=platform.platform(),python=sys.version,
                  cpu=Path('/proc/cpuinfo').read_text(),
                  compiler=subprocess.check_output(['mojo','--version'],text=True),
                  time_version=subprocess.check_output([time_binary,'--version'],text=True),
                  flags='-O3',seed='none; deterministic benchmark matrix',
                  thread_policy='native kernels serial; GNU time subprocess inherits CPU affinity',
                  affinity=sorted(os.sched_getaffinity(0)),measurements=records)
    (directory/'environment.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS backend/adapter workload checksums; GNU time records saved')


if __name__ == '__main__':
    main()
