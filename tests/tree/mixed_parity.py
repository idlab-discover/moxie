"""Dense categorical differential checks against the pinned local sklearn build."""
from pathlib import Path
import ast
import json
import subprocess
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

ROOT = Path(__file__).resolve().parents[2]

def categorical_report(model, classifier):
    """Category-aware text oracle from sklearn's fitted dictionaries and nodes.

    sklearn export_text does not render categorical rules. Decode each native
    bitset/hash partition independently and use its documented numeric layout.
    """
    tree = model.tree_
    dictionaries = dict(zip(np.flatnonzero(model.is_categorical_), model._categorical_encoder.categories_))
    lines = []
    def visit(node, depth):
        indent = '|   ' * depth + '|---'
        if tree.children_left[node] == -1:
            value = tree.value[node, 0] if tree.n_outputs == 1 else tree.value[node].T[0]
            if classifier:
                chosen = np.argmax(value)
                label = model.classes_[chosen] if tree.n_outputs == 1 and tree.n_classes[0] != 1 else chosen
                lines.append(f'{indent} class: {label}\n')
            else:
                lines.append(indent + ' value: [' + ', '.join(f'{v:.2f}' for v in value) + ']\n')
            return
        feature = tree.feature[node]
        name = f'feature_{feature}'
        if tree.split_kind[node] == 0:
            left = f'{name} <= {tree.threshold[node]:.2f}'
            right = f'{name} >  {tree.threshold[node]:.2f}'
        else:
            groups = [[], []]
            categories = [v for v in dictionaries[feature] if isinstance(v, str) or not np.isnan(v)]
            for code, value in enumerate(categories):
                if tree.split_kind[node] == 1:
                    go_left = bool(int(tree._left_cat_bitset[node, code // 32]) & (1 << (code % 32)))
                else:
                    h = int(tree._left_cat_bitset[node, 0]) ^ code
                    h ^= h >> 16
                    h = h * 0x7feb352d & 0xffffffff
                    h ^= h >> 15
                    h = h * 0x846ca68b & 0xffffffff
                    h ^= h >> 16
                    go_left = bool(h & 1)
                groups[0 if go_left else 1].append(repr(value) if isinstance(value, str) else str(np.float32(value)))
            left, right = [name + ' in {' + ', '.join(group) + '}' for group in groups]
            if tree.missing_go_to_left[node]:
                left += ' or missing/unknown'
            else:
                right += ' or missing/unknown'
        lines.append(indent + ' ' + left + '\n')
        visit(tree.children_left[node], depth + 1)
        lines.append(indent + ' ' + right + '\n')
        visit(tree.children_right[node], depth + 1)
    visit(0, 0)
    return ''.join(lines)

def main():
    assert 'builds/tree/oracle-env' in sklearn.__file__
    revision = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
    source = ROOT.parent / 'scikit-learn'
    for name in ('_classes.py', '_partitioner.pyx', '_splitter.pyx', '_tree.pyx'):
        expected = subprocess.check_output(['git', '-C', str(source), 'show', f'{revision}:sklearn/tree/{name}'])
        assert (ROOT / 'builds/tree/reference/sklearn/tree' / name).read_bytes() == expected
    output = subprocess.check_output([ROOT / 'builds/tree/mixed_driver'], text=True)
    actual = {}
    for line in output.splitlines():
        if line.startswith('NAMES '):
            _, tag, case, *names = line.split()
            actual['NAMES', int(tag), int(case)] = names
            continue
        if line.startswith('TEXT '):
            _, tag, case, report = line.split(' ', 3)
            actual['TEXT', int(tag), int(case)] = ast.literal_eval(report)
            continue
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
            y = np.array([[(i + i // 3 + k) % (3 if extra else 2) if classifier else (i * 7 + case + k) % 13 / 4 for k in range(outputs)] for i in range(n)], dtype=np.float64)
            weights = np.array([0 if i % 11 == 0 else 1 + i % 3 for i in range(n)], dtype=float)
            if case >= 30:
                weights[::11] = -.125
            kw = dict(categorical_features=([0, 2], [True, False, True], ["color", "group"], "from_dtype")[case % 4], random_state=case, max_depth=4, max_features=1 + case % 3)
            if case % 4 == 0:
                kw['max_leaf_nodes'] = 4
            if case % 5 == 0:
                kw['ccp_alpha'] = .02
            if case % 2 == 0:
                kw['criterion'] = 'entropy' if classifier else 'poisson'
            def mixed(values):
                result = values.astype(object)
                result[:, 0] = [f"v{int((v + 13) / 2):02d}" if np.isfinite(v) else np.nan for v in values[:, 0]]
                return pd.DataFrame({"color": pd.Categorical(result[:, 0]),
                                     "height": values[:, 1],
                                     "group": pd.Categorical(values[:, 2])})
            model = cls(**kw).fit(mixed(X), y, sample_weight=weights)
            checks += 1
            if actual['NAMES', tag, case] != model.feature_names_in_.tolist():
                failures.append(dict(key=['NAMES', tag, case], got=actual['NAMES',tag,case], want=model.feature_names_in_.tolist()))
            tree = model.tree_
            report = categorical_report(model, classifier)
            checks += 1
            if actual['TEXT', tag, case] != report:
                failures.append(dict(key=['TEXT', tag, case], got=actual['TEXT', tag, case], want=report))
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
            for i, value in enumerate(model.predict(mixed(query)).flat):
                check(('PRED', tag, case, i), [value], classifier)
            if classifier:
                probabilities = model.predict_proba(mixed(query))
                if outputs == 1:
                    probabilities = [probabilities]
                for i, value in enumerate(np.concatenate([column.ravel() for column in probabilities])):
                    check(('PROBA', tag, case, i), [value], False)
            for i, value in enumerate(model.apply(mixed(query))):
                check(('LEAF', tag, case, i), [value])
    dest = ROOT / 'docs/private/tree/mixed_tables'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'native.txt').write_text(output)
    (dest / 'results.json').write_text(json.dumps(dict(checks=checks, failures=failures), indent=2))
    print(f'{checks} mixed-table comparisons; {len(failures)} mismatches across 160 fits')
    if failures:
        print(json.dumps(failures[:8], indent=2))
        raise SystemExit(1)

if __name__ == '__main__':
    main()
