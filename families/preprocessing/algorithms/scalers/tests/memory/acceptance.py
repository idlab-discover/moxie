"""Scaler memory acceptance: peak RSS of public operations against byte formulas.

Usage: python acceptance.py <assembled-deps-dir> [--quick]

Each operation runs in its own native process (tests/memory/workload.mojo) that
first allocates and touches the inputs. The `input` op is the control; an op's
extra peak RSS over the control must stay within its formula (docs/memory.md)
plus ALLOWANCE. Copying operations must also show at least 98% of their logical
result, proving the measurement sees allocations. `full_copy` is a deliberate
full-table temporary that the zero-extra budget must reject. Raw results go to
the ignored benchmarks/memory/ folder.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHILD = HERE.parents[1]
FLAGS = ['-O3', '--fp-mode=contract=off']

# The Mojo runtime allocator keeps up to ~24 MiB of freed blocks resident
# (measured: retention grows with freed size up to that cap, independent of n),
# so fixed allowance plus 1% of the formula absorbs it and page granularity.
ALLOWANCE = 32 * 2**20
RELATIVE = 0.01
TABLE = 400_000_000  # bytes of Float64 input per configuration; float32 is half

# op -> extra bytes as a function of rows n, cols p, element bytes d.
# "R" marks copying ops whose logical result must be visible.
R = lambda n, p, d: n * p * d
OPS = {
    'std_fit': (lambda n, p, d: 0, False),
    'std_fit_weighted': (lambda n, p, d: 0, False),
    'std_fit_scalar': (lambda n, p, d: 0, False),
    'std_partial_fit': (lambda n, p, d: 0, False),
    'std_transform': (R, True),
    'std_inverse': (R, True),
    'std_inplace': (lambda n, p, d: 0, False),
    'minmax_fit': (lambda n, p, d: 0, False),
    'minmax_transform': (R, True),
    'minmax_inplace': (lambda n, p, d: 0, False),
    'maxabs_fit': (lambda n, p, d: 0, False),
    'maxabs_transform': (R, True),
    'robust_fit': (lambda n, p, d: 8 * n, False),          # one Float64 column
    'robust_transform': (R, True),                          # column freed before result
    'scale': (lambda n, p, d: R(n, p, d) + 3 * p * d, True),
    'scale_axis1': (lambda n, p, d: R(n, p, d) + 3 * n * d, True),   # per-row mean/std/scratch
    'minmax_scale': (lambda n, p, d: R(n, p, d) + 2 * n * d, True),  # axis=1: per-row shift/scale
    'maxabs_scale': (R, True),
    'maxabs_scale_axis1': (lambda n, p, d: R(n, p, d) + 2 * n * d, True),
    'robust_scale': (R, True),
    'robust_scale_axis1': (lambda n, p, d: R(n, p, d) + 2 * 8 * n, True),
    'view_fit_transform': (lambda n, p, d: 0, False),
}
CONFIGS = [('float64', 1_000_000, 50), ('float64', 10_000_000, 5), ('float64', 2_000, 25_000),
           ('float32', 1_000_000, 50)]


def env():
    e = os.environ.copy()
    e.setdefault('TMPDIR', str(CHILD / 'builds/tmp'))
    Path(e['TMPDIR']).mkdir(parents=True, exist_ok=True)
    return e


def gnu_time():
    candidate = Path(sys.prefix) / 'bin/time'
    if not candidate.exists():
        raise SystemExit('GNU time (pixi package `time`) is required for peak RSS')
    return str(candidate)


def measure(binary, op, dtype, n, p, cap_bytes):
    command = [gnu_time(), '-f', '%M', str(binary), op, dtype, str(n), str(p)]
    if shutil.which('systemd-run'):
        # Hard cap so a regression cannot exhaust the host while being measured.
        command = ['systemd-run', '--user', '--scope', '-q', f'-pMemoryMax={cap_bytes}',
                   '-pMemorySwapMax=0', *command]
    result = subprocess.run(command, capture_output=True, text=True, env=env())
    lines = result.stderr.strip().splitlines()
    if result.returncode or not result.stdout.startswith('OK ' + op):
        return dict(op=op, error=(result.stdout + result.stderr)[-2000:], command=command)
    return dict(op=op, peak_kib=int(lines[-1]), command=command)


def main():
    deps = Path(sys.argv[1]).resolve()
    quick = '--quick' in sys.argv
    out_dir = CHILD / 'builds/memory'
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = out_dir / 'workload'
    subprocess.run(['mojo', 'build', *FLAGS, '-I', str(deps), str(HERE / 'workload.mojo'), '-o', str(binary)],
                   check=True, env=env())
    configs = CONFIGS[:1] if quick else CONFIGS
    failures, records = [], []
    for dtype, n, p in configs:
        d = 8 if dtype == 'float64' else 4
        inputs = n * p * d + n * d
        cap = lambda extra: inputs + 2 * extra + 4 * ALLOWANCE + 256 * 2**20
        control = measure(binary, 'input', dtype, n, p, cap(0))
        if 'error' in control:
            raise SystemExit(f'control failed: {control["error"]}')
        base = control['peak_kib'] * 1024
        jobs = list(OPS.items()) + [('full_copy', (lambda n, p, d: 0, False))]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda item: (item, measure(binary, item[0], dtype, n, p, cap(R(n, p, d) + 3 * 8 * n))), jobs))
        for (op, (formula, copying)), result in results:
            label = f'{dtype} {n}x{p} {op}'
            expected = formula(n, p, d)
            budget = expected + ALLOWANCE + RELATIVE * expected
            record = dict(result, dtype=dtype, rows=n, cols=p, expected_bytes=expected, budget_bytes=budget,
                          control_kib=control['peak_kib'])
            records.append(record)
            if 'error' in result:
                failures.append(f'{label}: {result["error"]}')
                continue
            extra = result['peak_kib'] * 1024 - base
            record['extra_bytes'] = extra
            if op == 'full_copy':
                # The control must be rejected by a zero-extra budget.
                if extra <= budget:
                    failures.append(f'{label}: full-table temporary ({extra} B) was not detected')
                continue
            if extra > budget:
                failures.append(f'{label}: extra {extra / 2**20:.1f} MiB > budget {budget / 2**20:.1f} MiB')
            if copying and extra < 0.98 * R(n, p, d):
                failures.append(f'{label}: extra {extra / 2**20:.1f} MiB below logical result {R(n, p, d) / 2**20:.1f} MiB')
            print(f'{label:45s} extra {extra / 2**20:9.1f} MiB  budget {budget / 2**20:9.1f} MiB')
    bench = CHILD / 'benchmarks/memory'
    bench.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(['mojo', '--version'], capture_output=True, text=True, env=env()).stdout.strip()
    cpu = next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                if l.startswith('model name')), platform.processor())
    report = dict(mojo=version, flags=FLAGS, cpu=cpu, mem_total=os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
                  allowance=ALLOWANCE, relative=RELATIVE, failures=failures, records=records,
                  time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    (bench / f'acceptance-{time.strftime("%Y%m%d-%H%M%S")}.json').write_text(json.dumps(report, indent=2) + '\n')
    if failures:
        print('\n'.join(failures))
        raise SystemExit(f'FAIL memory scalers: {len(failures)} violations')
    print(f'PASS memory scalers: {len(records)} operations within byte budgets; full-table control rejected')


if __name__ == '__main__':
    main()
