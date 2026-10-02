"""Parity harness: pinned scikit-learn oracle in Python, compiled native driver in Mojo.

Usage (from a tests/parity/test_<group>.py script):

    from harness import Driver, oracle
    sk = oracle()                               # pinned sklearn (overlay)
    d = Driver('scalers')
    X = d.input(np.random.rand(5, 3))           # Mojo expression loading X
    d.case('std', f'var m = StandardScaler(); emit("t", m.fit_transform({X}))')
    out = d.run(imports='from moxie.preprocessing import StandardScaler')
    d.close(out['std']['t'], sk.preprocessing.StandardScaler().fit_transform(...))
    d.finish()                                  # raises if any comparison failed

Each case body runs in its own raising function; an escaped error is reported
as `out[case]['error']` instead of aborting the driver.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools'))


def oracle():
    from overlay import activate
    return activate()


def _env():
    env = os.environ.copy()
    for key, folder in {'XDG_CACHE_HOME': '.pixi/cache', 'PIXI_CACHE_DIR': '.pixi/pixi-cache',
                        'MODULAR_CACHE_DIR': '.pixi/modular-cache', 'TMPDIR': 'builds/tmp'}.items():
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


_MOJO_DTYPES = {np.dtype(np.float64): 'DType.float64', np.dtype(np.float32): 'DType.float32',
                np.dtype(np.int64): 'DType.int64', np.dtype(np.bool_): 'DType.bool'}
_NUMPY_DTYPES = {'float64': np.float64, 'float32': np.float32, 'int64': np.int64, 'bool': np.bool_}


def _parse_value(text, dtype):
    if dtype == 'bool':
        return text == 'True'
    if dtype == 'int64':
        return int(text)
    return float(text)


class Driver:
    def __init__(self, name):
        self.name = name
        self.dir = ROOT / 'builds/parity' / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.inputs = 0
        self.cases = []
        self.preludes = []
        self.failures = []
        self.checks = 0

    # ---- inputs -------------------------------------------------------------
    def input(self, array):
        """Persist an array; return a Mojo expression that loads it."""
        array = np.ascontiguousarray(array)
        if array.dtype not in _MOJO_DTYPES:
            raise TypeError(array.dtype)
        self.inputs += 1
        path = self.dir / f'input{self.inputs}.bin'
        path.write_bytes(array.tobytes())
        cols = array.shape[1] if array.ndim == 2 else -1
        assert array.ndim in (1, 2)
        return f'load[{_MOJO_DTYPES[array.dtype]}]("{path}", {array.shape[0]}, {cols})'

    def strings(self, values):
        """Persist strings (no newlines); return a Mojo List[String] expression."""
        self.inputs += 1
        path = self.dir / f'input{self.inputs}.txt'
        assert all('\n' not in v for v in values)
        path.write_text(''.join(v + '\n' for v in values))
        return f'load_strings("{path}")'

    def optional_strings(self, values):
        """Persist nullable strings (None = missing); return a List[Optional[String]] expression."""
        self.inputs += 1
        path = self.dir / f'input{self.inputs}.txt'
        assert all(v is None or '\n' not in v for v in values)
        path.write_text(''.join(('N' if v is None else 'S' + v) + '\n' for v in values))
        return f'load_optional_strings("{path}")'

    # ---- program ------------------------------------------------------------
    def prelude(self, code):
        self.preludes.append(code)

    def case(self, name, body):
        assert name not in {c[0] for c in self.cases}, name
        self.cases.append((name, body))

    def _source(self, imports):
        lines = ['from numojo import NDArray, Shape, Item, zeros', 'import numojo as nm',
                 'from parity_support import load, load_strings, load_optional_strings, emit, emit_scalar, emit_int, emit_names, emit_error',
                 imports, '', *self.preludes, '']
        for k, (name, body) in enumerate(self.cases):
            lines.append(f'def case_{k}() raises:')
            lines.append(f'    print("C {name}")')
            for line in body.strip('\n').split('\n'):
                lines.append('    ' + line)
            lines.append('')
        lines.append('def main() raises:')
        for k, (name, _) in enumerate(self.cases):
            lines += [f'    try:', f'        case_{k}()', f'    except error:',
                      f'        emit_error("__case__", error)']
        if not self.cases:
            lines.append('    pass')
        return '\n'.join(lines) + '\n'

    def run(self, imports=''):
        source = self.dir / 'driver.mojo'
        source.write_text(self._source(imports))
        binary = self.dir / 'driver'
        build = subprocess.run(['mojo', 'build', '--fp-mode=contract=off', '-I', str(ROOT / 'builds/deps'),
                                '-I', str(HERE), str(source), '-o', str(binary)],
                               cwd=ROOT, env=_env(), text=True, capture_output=True)
        if build.returncode:
            raise SystemExit(f'driver compile failed:\n{build.stdout}\n{build.stderr}')
        result = subprocess.run([str(binary)], cwd=ROOT, env=_env(), text=True, capture_output=True)
        if result.returncode:
            raise SystemExit(f'driver crashed ({result.returncode}):\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}')
        return self._parse(result.stdout)

    def _parse(self, text):
        out = {}
        current = None
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            if not line:
                continue
            tag, _, rest = line.partition(' ')
            if tag == 'C':
                current = out.setdefault(rest, {})
            elif tag == 'R':
                name, ndim, rows, cols, dtype, *values = rest.split(' ')
                dtype = dtype.removeprefix('DType.')
                ndim, rows, cols = int(ndim), int(rows), int(cols)
                array = np.array([_parse_value(v, dtype) for v in values], dtype=_NUMPY_DTYPES[dtype])
                current[name] = array.reshape((rows, cols) if ndim == 2 else (rows,))
            elif tag == 'F':
                name, value = rest.split(' ')
                current[name] = float(value)
            elif tag == 'I':
                name, value = rest.split(' ')
                current[name] = int(value)
            elif tag == 'S':
                name, count = rest.rsplit(' ', 1)
                current[name] = lines[i:i + int(count)]
                i += int(count)
            elif tag == 'E':
                name, _, message = rest.partition(' ')
                current['error' if name == '__case__' else name] = 'Error: ' + message
            else:
                raise ValueError(f'unexpected driver output: {line!r}')
        return out

    # ---- comparisons --------------------------------------------------------
    def _fail(self, label, message):
        self.failures.append(f'{label}: {message}')

    def close(self, label, got, expected, *, rtol=1e-12, atol=0.0, dtype=None):
        """Shape, dtype, NaN/inf class and values within rtol/atol."""
        self.checks += 1
        if isinstance(got, str):
            return self._fail(label, f'native raised {got}')
        expected = np.asarray(expected)
        got = np.asarray(got)
        if got.shape != expected.shape:
            return self._fail(label, f'shape {got.shape} != {expected.shape}')
        if dtype is not None and got.dtype != np.dtype(dtype):
            return self._fail(label, f'dtype {got.dtype} != {dtype}')
        if expected.dtype.kind in 'fc' or got.dtype.kind in 'fc':
            g = got.astype(np.float64)
            e = expected.astype(np.float64)
            if not np.array_equal(np.isnan(g), np.isnan(e)):
                return self._fail(label, f'NaN pattern differs\n got={g}\n exp={e}')
            finite = np.isfinite(e)
            if not np.array_equal(g[~finite & ~np.isnan(e)], e[~finite & ~np.isnan(e)]):
                return self._fail(label, 'infinities differ')
            # atol may be an array broadcastable to the expected shape (e.g. per column).
            atol = np.broadcast_to(np.asarray(atol, dtype=np.float64), e.shape)[finite]
            if not np.allclose(g[finite], e[finite], rtol=rtol, atol=atol):
                worst = np.max(np.abs(g[finite] - e[finite]) - rtol * np.abs(e[finite]) - atol)
                return self._fail(label, f'values differ (worst excess {worst:.3g})\n got={g}\n exp={e}')
        elif not np.array_equal(got, expected):
            return self._fail(label, f'values differ\n got={got}\n exp={expected}')

    def equal(self, label, got, expected):
        self.checks += 1
        if got != expected:
            self._fail(label, f'{got!r} != {expected!r}')

    def raises(self, label, got, needle=''):
        """Native must have raised (and sklearn too); optional message fragment."""
        self.checks += 1
        if not isinstance(got, str) or not got.startswith('Error'):
            return self._fail(label, f'expected an error, got {got!r}')
        if needle and needle not in got:
            self._fail(label, f'error {got!r} lacks {needle!r}')

    def finish(self, extra=None):
        report = dict(group=self.name, checks=self.checks, failures=self.failures, **(extra or {}))
        evidence = ROOT / 'docs/private/parity'
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / f'{self.name}.json').write_text(json.dumps(report, indent=2) + '\n')
        if self.failures:
            print('\n'.join(self.failures[:40]))
            raise SystemExit(f'FAIL parity {self.name}: {len(self.failures)}/{self.checks} checks failed')
        print(f'PASS parity {self.name}: {self.checks} checks against pinned scikit-learn')
