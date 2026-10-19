"""Dense categorical differential checks against the pinned local sklearn build."""
from pathlib import Path
import json
import subprocess
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

ROOT = Path(__file__).resolve().parents[2]

def main():
    assert 'builds/tree/oracle-env' in sklearn.__file__
    revision = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
    source = ROOT.parent / 'scikit-learn'
    for name in ('_classes.py', '_partitioner.pyx', '_splitter.pyx', '_tree.pyx'):
        expected = subprocess.check_output(['git', '-C', str(source), 'show', f'{revision}:sklearn/tree/{name}'])
        assert (ROOT / 'builds/tree/reference/sklearn/tree' / name).read_bytes() == expected
    output = subprocess.check_output([ROOT / 'builds/tree/categorical_driver'], text=True)
    actual = {}
    for line in output.splitlines():
        tag, *fields = line.split()
        size = 2 if tag == 'CASE' else 3
        actual[tag, *map(int, fields[:size])] = list(map(float, fields[size:]))
    failures = []
    checks = 0
    def check(key, expected, exact=True):
        nonlocal checks
        checks += 1
        try:
            np.testing.assert_allclose(actual[key], expected, rtol=0 if exact else 2e-12, atol=0 if exact else 2e-12, equal_nan=True)
        except (AssertionError, KeyError):
            failures.append(dict(key=key, got=actual.get(key), want=np.asarray(expected).tolist()))
    for tag, cls in enumerate((DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor, ExtraTreeClassifier)):
        classifier = tag % 2 == 1
        extra = tag >= 2
        for case in range(40):
            n = 16 + case
            outputs = 2 if extra and case % 4 == 0 else 1
            X = np.array([[(i * (3 + j) + case + j) % (2 + case % 19) * 2 - 13 for j in range(3)] for i in range(n)], dtype=np.float32)
            X[:, 1] = np.arange(n) % 7 / 4
            if case % 3 == 0:
                X[::5, :] = np.nan
            if case == 1:
                X[:, 0] = np.nan
            y = np.array([[(i + i // 3 + k) % (3 if extra else 2) if classifier else (i * 7 + case + k) % 13 / 4 for k in range(outputs)] for i in range(n)])
            weights = np.array([0 if i % 11 == 0 else 1 + i % 3 for i in range(n)], dtype=float)
            if case >= 30:
                weights[::11] = -.125
            kw = dict(categorical_features=[0, 2], random_state=case, max_depth=4, max_features=1 + case % 3)
            if case % 4 == 0:
                kw['max_leaf_nodes'] = 4
            if case % 5 == 0:
                kw['ccp_alpha'] = .02
            if case % 2 == 0:
                kw['criterion'] = 'entropy' if classifier else 'poisson'
            model = cls(**kw).fit(X, y, sample_weight=weights)
            tree = model.tree_
            check(('CASE', tag, case), [tree.node_count])
            for i in range(tree.node_count):
                check(('NODE', tag, case, i), [tree.children_left[i], tree.children_right[i], tree.feature[i], tree.impurity[i], tree.n_node_samples[i], tree.weighted_n_node_samples[i], tree.missing_go_to_left[i], (-1 if tree.children_left[i] == -1 else tree.split_kind[i])], False)
                # Leaf split kind is normalized to the native -1 contract.
                check(('THRESHOLD', tag, case, i), [tree.threshold[i]], False)
                for b in range(8):
                    check(('BITS', tag, case, i * 8 + b), [tree._left_cat_bitset[i, b]])
            for i, value in enumerate(tree.value.flat):
                check(('VALUE', tag, case, i), [value], False)
            query = X.copy()
            query[0, 0] = 9999
            query[0, 2] = np.nan
            for i, value in enumerate(model.predict(query).flat):
                check(('PRED', tag, case, i), [value], classifier)
            if classifier:
                probabilities = model.predict_proba(query)
                if outputs == 1:
                    probabilities = [probabilities]
                for i, value in enumerate(np.concatenate([column.ravel() for column in probabilities])):
                    check(('PROBA', tag, case, i), [value], False)
            for i, value in enumerate(model.apply(query)):
                check(('LEAF', tag, case, i), [value])
    dest = ROOT / 'docs/private/tree/categorical'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'native.txt').write_text(output)
    (dest / 'results.json').write_text(json.dumps(dict(checks=checks, failures=failures), indent=2))
    print(f'{checks} categorical comparisons; {len(failures)} mismatches across 160 fits')
    if failures:
        print(json.dumps(failures[:8], indent=2))
        raise SystemExit(1)

if __name__ == '__main__':
    main()
