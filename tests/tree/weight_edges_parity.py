"""Defined nonpositive-weight behavior; MAE's unsafe reference median is excluded."""
from pathlib import Path
import json
import subprocess
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
ROOT = Path(__file__).resolve().parents[2]

def main():
    assert 'builds/tree/oracle-env' in sklearn.__file__
    output = subprocess.check_output([ROOT/'builds/tree/weight_edges_driver'], text=True)
    dest = ROOT/'docs/private/tree/weight_edges'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'native.txt').write_text(output)
    actual = {}
    for line in output.splitlines():
        tag, *fields = line.split()
        count = 2 if tag in ('ERROR', 'CASE') else 3
        actual[tag, *map(int, fields[:count])] = list(map(float, fields[count:]))
    checks = 0
    def check(key, expected):
        nonlocal checks
        np.testing.assert_allclose(actual[key], np.atleast_1d(expected), rtol=2e-12, atol=2e-12, equal_nan=True, err_msg=str(key))
        checks += 1
    for tag in range(10):
        kind = tag % 5
        cls = (ExtraTreeClassifier if kind < 2 else ExtraTreeRegressor) if tag >= 5 else (DecisionTreeClassifier if kind < 2 else DecisionTreeRegressor)
        for case in range(12):
            X = np.arange(4, dtype=np.float32).reshape(-1, 1)
            y = np.ones(4) if case >= 8 else np.array([1, 2, 1, 2])
            w = np.array([[0]*4, [-1]*4, [1,-1,1,-1], [1]*4][case % 4], dtype=float)
            kw = dict(random_state=7, criterion=['gini','entropy','squared_error','poisson','absolute_error'][kind])
            if case >= 4: kw.update(max_leaf_nodes=3, ccp_alpha=.02, min_weight_fraction_leaf=.5 if case >= 8 else .25)
            if case >= 8: kw['monotonic_cst'] = [1]
            if kind == 4 and w.sum() <= 0:
                assert ('ERROR', tag, case) in actual
                continue  # Do not execute the reference's out-of-bounds median search.
            try:
                model = cls(**kw).fit(X, y, sample_weight=w)
            except ValueError:
                assert ('ERROR', tag, case) in actual
                checks += 1
                continue
            check(('CASE',tag,case), model.tree_.node_count)
            t = model.tree_
            for i in range(t.node_count): check(('NODE',tag,case,i), [t.impurity[i],t.n_node_samples[i],t.weighted_n_node_samples[i]])
            for i, v in enumerate(t.value.flat): check(('VALUE',tag,case,i), v)
            for i, v in enumerate(model.predict(X).flat): check(('PRED',tag,case,i), v)
    report = dict(checks=checks, cases=120, tolerance=2e-12)
    (dest/'parity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report)
if __name__ == '__main__': main()
