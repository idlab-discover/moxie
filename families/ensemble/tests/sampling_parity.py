"""Exact parity of ensemble sampling and class weighting against the reference sklearn.

The oracle calls sklearn's own private helpers (`_get_n_samples_bootstrap`,
`_generate_sample_indices`, `_generate_unsampled_indices`,
`_generate_bagging_indices`, `compute_sample_weight` and the forest's
`_validate_y_class_weight`) from the source-built reference revision. Floats
cross the boundary as raw bits. Every index stream must be identical, which
checks that Moxie's RandomState consumes MT19937 draws exactly like NumPy's
legacy `randint`, `choice(p=...)` (with and without replacement) and sklearn's
`sample_without_replacement`; weights and sums must be bit-identical, which
checks NumPy's pairwise summation order. Errors must occur on the same inputs.
"""
from pathlib import Path
import subprocess, warnings
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._bootstrap import _get_n_samples_bootstrap
from sklearn.ensemble._forest import _generate_sample_indices, _generate_unsampled_indices
from sklearn.ensemble._bagging import _generate_bagging_indices
from sklearn.utils.class_weight import compute_sample_weight
ROOT = Path(__file__).resolve().parents[1]

def bits(values):
    return ' '.join(map(str, np.ascontiguousarray(values, dtype=np.float64).view(np.int64).ravel()))

def weight_field(w):
    return 'none' if w is None else 'w ' + bits(w)

def random_weights(rng, n, kind):
    if kind == 'none':
        return None
    if kind == 'int':
        return rng.randint(0, 5, size=n).astype(np.float64)
    if kind == 'heavy':  # wide exponent range makes summation order observable
        return np.abs(rng.standard_normal(n)) * np.exp(rng.uniform(-30, 30, n))
    if kind == 'sparse':
        w = rng.uniform(size=n); w[rng.rand(n) < .7] = 0; w[0] = .25
        return w
    if kind == 'negative':
        w = rng.uniform(size=n); w[rng.randint(n)] = -.5
        return w
    return rng.uniform(.01, 3, size=n)

def reference(call):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            return call(), len(caught)
    except ValueError:
        return 'error', 0

def main():
    assert 'builds/oracle-env' in sklearn.__file__
    rng = np.random.RandomState(20261005)
    commands, expected = [], []
    kinds = ['none', 'uniform', 'int', 'heavy', 'sparse']
    for t in range(60):  # NumPy pairwise sums across every leaf/split regime
        n = int(rng.choice([1, 7, 8, 9, 127, 128, 129, 255, 1000, 4099]))
        x = rng.standard_normal(n) * np.exp(rng.uniform(-25, 25, n))
        commands.append(f'SUM {n} {bits(x)}'); expected.append(f'SUM {bits([np.sum(x)])}')
    for t in range(80):  # bootstrap sizes, including fractional and weighted scaling
        n = int(rng.randint(1, 300)); kind = kinds[t % len(kinds)]
        w = random_weights(rng, n, kind)
        option = [('none', None), ('int', int(rng.randint(1, 3 * n + 2))), ('float', float(rng.uniform(.001, 2.5))), ('float', .5)][t % 4]
        value = option[1]
        field = '0' if value is None else (str(value) if option[0] == 'int' else bits([value]))
        commands.append(f'NBOOT {n} {option[0]} {field} {weight_field(w)}')
        count, warned = reference(lambda: _get_n_samples_bootstrap(n, value, w))
        expected.append('NBOOT error' if count == 'error' else f'NBOOT {count} {warned}')
    for t in range(240):  # forest bootstrap draws, OOB rows and bincounts
        n = int(rng.choice([1, 2, 5, 33, 128, 400, 1500])); kind = (kinds + ['negative'])[t % 6]
        w = random_weights(rng, n, kind)
        draws = int(rng.choice([n, max(1, n // 3), 2 * n + 1]))
        seed = int(rng.randint(0, 2**31 - 1))
        commands.append(f'BOOT {seed} {n} {draws} {weight_field(w)}')
        try:
            rows = _generate_sample_indices(seed, n, draws, w)
            oob = _generate_unsampled_indices(seed, n, draws, w)
            expected.append('BOOT ' + ' '.join(map(str, rows)) + ' | ' + ' '.join(map(str, oob)) + ' | ' + ' '.join(map(str, np.bincount(rows, minlength=n))))
        except ValueError:
            expected.append('BOOT error')
    for t in range(160):  # bagging: feature then row draws, weighted choice with and without replacement
        nf = int(rng.randint(1, 20)); n = int(rng.choice([3, 40, 300, 2000]))
        bf, bs = int(rng.rand() < .5), int(rng.rand() < .5)
        mf = int(rng.randint(1, nf + 1)) if not bf else int(rng.randint(1, 2 * nf + 1))
        ms = int(rng.choice([1, max(1, n // 150), max(1, n // 10), n // 2 + 1, n])) if not bs else int(rng.randint(1, 2 * n))
        w = random_weights(rng, n, ['none', 'uniform', 'int', 'heavy'][t % 4])
        seed = int(rng.randint(0, 2**31 - 1))
        commands.append(f'BAG {seed} {bf} {bs} {nf} {n} {mf} {ms} {weight_field(w)}')
        try:
            features, rows = _generate_bagging_indices(seed, bf, bs, nf, n, mf, ms, w)
            expected.append('BAG ' + ' '.join(map(str, features)) + ' | ' + ' '.join(map(str, rows)))
        except ValueError:
            expected.append('BAG error')
    for t in range(60):  # balanced_subsample weights from a bootstrap draw, multi-output products
        n = int(rng.randint(2, 200)); outputs = int(rng.randint(1, 4))
        Y = np.column_stack([rng.randint(0, int(rng.randint(1, 6)), size=n) for _ in range(outputs)])
        Y = np.column_stack([np.unique(Y[:, k], return_inverse=True)[1] for k in range(outputs)]).astype(np.float64)
        indices = rng.randint(0, n, size=int(rng.choice([n, max(1, n // 4)])))
        commands.append(f'SUB {n} {outputs} {len(indices)} ' + ' '.join(map(str, indices)) + ' ' + bits(Y))
        expected.append('SUB ' + bits(compute_sample_weight('balanced', Y, indices=indices)))
    for t in range(60):  # forest class_weight="balanced", weighted class counts, through the forest's own code path
        n = int(rng.randint(2, 300)); outputs = int(rng.randint(1, 4))
        labels = np.column_stack([3 * rng.randint(0, int(rng.randint(1, 6)), size=n) - 2 for _ in range(outputs)]).astype(np.float64)
        codes = np.column_stack([np.unique(labels[:, k], return_inverse=True)[1] for k in range(outputs)]).astype(np.float64)
        w = random_weights(rng, n, ['none', 'uniform', 'int', 'heavy'][t % 4])
        if w is not None and t % 4 == 2:
            w = w + 1  # integer weights without empty classes
        forest = RandomForestClassifier(class_weight='balanced'); forest.n_outputs_ = outputs
        _, expanded = forest._validate_y_class_weight(labels, w)
        commands.append(f'BAL {n} {outputs} {bits(codes)} {weight_field(w)}')
        expected.append('BAL ' + bits(expanded))
    dest = ROOT/'docs/private/forest_engine'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'sampling_input.txt').write_text('\n'.join(commands) + '\n')
    output = subprocess.check_output([ROOT/'builds/sampling_driver', dest/'sampling_input.txt'], text=True).splitlines()
    assert len(output) == len(expected), (len(output), len(expected))
    draws = 0
    for command, got, want in zip(commands, output, expected):
        assert got.split() == want.split(), (command[:80], got[:200], want[:200])
        draws += len(want.split())
    errors = sum(line.endswith('error') for line in expected)
    print(f'PASS sampling: {len(commands)} cases ({errors} matched errors), {draws} identical values from sklearn helpers')

if __name__ == '__main__':
    main()
