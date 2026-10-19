"""Seeded combined typed labels, selectors, signed weights, pruning and refits.

Native selectors cycle indices, masks, names and dtype metadata; the ndarray
oracle uses equivalent index/mask selectors. Each model is reused for 32 fits.
"""
from pathlib import Path
import json, subprocess
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
ROOT = Path(__file__).resolve().parents[2]

def main():
    assert 'builds/tree/oracle-env' in sklearn.__file__
    output = subprocess.check_output([ROOT/'builds/tree/randomized_driver'], text=True)
    dest = ROOT/'docs/private/tree/randomized'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'native.txt').write_text(output)
    dimensions = {'CASE': 2, 'CLASS': 4, 'PRED': 4, 'PROBA': 4, 'SCORE': 2, 'NODE': 3, 'LEAF': 3, 'BITS': 3}
    actual = {}
    for line in output.splitlines():
        tag, *fields = line.split(); count = dimensions[tag]
        actual[tag, *map(int, fields[:count])] = fields[count:]
    checks = 0
    for tag in range(6):
        domain = np.array(['zebra', 'amber', 'é']) if tag < 2 else np.array([9007199254740993, -9223372036854775808, 9007199254740992], dtype=np.int64) if tag < 4 else np.array([8, -3, 2], dtype=np.float32)
        def check(key, expected, labels=False):
            nonlocal checks
            checks += 1
            # Never float-cast Int64 labels: adjacent values above 2**53 must differ.
            if labels:
                got = actual[key][0]
                if tag < 2: assert got == str(expected), (key, got, expected)
                elif tag < 4: assert int(got) == int(expected), (key, got, expected)
                else: assert np.float32(got) == expected, (key, got, expected)
            else:
                np.testing.assert_allclose(np.array(actual[key], dtype=float), np.atleast_1d(expected), rtol=0 if key[0] in ('CASE','LEAF','BITS') else 2e-12, atol=0 if key[0] in ('CASE','LEAF','BITS') else 2e-12, err_msg=str(key))
        for case in range(32):
            n = 17 + case; outputs = 1
            rng = np.random.RandomState(7300 + case)
            X = np.zeros((n,2), dtype=np.float32)
            codes = np.zeros(n, dtype=np.int64)
            weights = np.zeros(n)
            for i in range(n):
                codes[i] = rng.randint(0,2) * (2 if case % 2 else 1)
                weights[i] = rng.randint(1,5)
                if i % 7 == 0: weights[i] = -.125
                for j in range(2): X[i,j] = rng.randint(0,7 if j == 0 else 127)/4
            y = domain[codes]
            kw = dict(random_state=case, max_depth=4, criterion='entropy' if case%2 else 'gini',
                      categorical_features=[0] if case%4 == 0 else [True,False],
                      ccp_alpha=.02 if case%3 else .1, max_leaf_nodes=5 if case%2 else None)
            if case == 0: model = (ExtraTreeClassifier if tag%2 else DecisionTreeClassifier)()
            model.set_params(**kw)
            model.fit(X, y, sample_weight=weights)
            check(('CASE',tag,case), [outputs,model.tree_.node_count])
            classes = [model.classes_] if outputs == 1 else model.classes_
            for k, column in enumerate(classes):
                assert column.dtype == y.dtype
                for c, label in enumerate(column): check(('CLASS',tag,case,k,c), label, True)
            prediction = model.predict(X)
            assert prediction.dtype == y.dtype
            for i, row in enumerate(prediction.reshape(n,outputs)):
                for k, label in enumerate(row): check(('PRED',tag,case,i,k), label, True)
            proba = model.predict_proba(X)
            for k, column in enumerate([proba] if outputs == 1 else proba):
                for i, value in enumerate(column.flat): check(('PROBA',tag,case,k,i), value)
            if outputs == 1: check(('SCORE',tag,case), model.score(X,y,sample_weight=weights))
            t = model.tree_
            for i in range(t.node_count):
                check(('NODE',tag,case,i), [t.children_left[i],t.children_right[i],t.feature[i],t.threshold[i],t.impurity[i],t.weighted_n_node_samples[i]])
                assert list(map(int, actual['NODE',tag,case,i][:3])) == [t.children_left[i],t.children_right[i],t.feature[i]]
            for i, leaf in enumerate(model.apply(X)): check(('LEAF',tag,case,i), leaf)
            for i, word in enumerate(t._left_cat_bitset.flat): check(('BITS',tag,case,i), word)
    report = dict(checks=checks, fits=192, label_identity='exact; Int64 never converted to float', tolerance='2e-12 for Float64 statistics')
    (dest/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__ == '__main__': main()
