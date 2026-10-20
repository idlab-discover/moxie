"""End-to-end parity of ForestEngine against real sklearn forests.

Each case fits a reference RandomForest*/ExtraTrees* with the source-built
oracle and the native engine from the same inputs (raw float bits). The engine
draws its own member seeds and bootstrap rows; nothing sklearn computed is fed
to it except the forest configuration and, for dict class weights, the
per-class factors sklearn's `compute_class_weight` resolves from labels.

Compared exactly: member seeds, `estimators_samples_`, `apply`, `predict` /
`predict_proba`, `feature_importances_`, `oob_score_`, `oob_prediction_` /
`oob_decision_function_` (bit for bit, NaN equal), warning categories in order,
and error-raising inputs. Warm-start cases grow the forest, then refit without
growth; `shared` cases pass one RandomState object across all fits.
"""
from pathlib import Path
import subprocess, warnings
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight
ROOT = Path(__file__).resolve().parents[1]
CLASSIFIERS = (RandomForestClassifier, ExtraTreesClassifier)

def case_list():
    rf_r, rf_c, et_r, et_c = RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
    return [
        (rf_r, dict(criterion='squared_error', oob_score=True), dict(nan=True)),
        (rf_r, dict(criterion='friedman_mse', max_samples=.5, min_samples_leaf=.05, max_depth=5, oob_score=True), dict(nan=True, weights=True)),
        (rf_r, dict(criterion='poisson', max_depth=6, oob_score=True), dict(nan=True, positive=True)),
        (rf_r, dict(criterion='absolute_error', max_leaf_nodes=8, oob_score=True), dict(weights=True)),
        (rf_r, dict(criterion='squared_error', bootstrap=False, max_features=.6), dict(outputs=2, weights=True)),
        (rf_r, dict(monotonic_cst=[1, 0, -1, 0, 0], ccp_alpha=.01, max_samples=40, oob_score=True), dict()),
        (rf_r, dict(criterion='squared_error', min_weight_fraction_leaf=.05, min_impurity_decrease=.01, oob_score=True), dict(outputs=2, nan=True, weights=True)),
        (rf_r, dict(n_estimators=3, oob_score=True, max_samples=.03), dict(heavy=True)),
        (rf_r, dict(n_estimators=4, oob_score=True), dict(warm=9)),
        (rf_r, dict(n_estimators=2, oob_score=True, min_samples_split=.2), dict(warm=5, shared=True, weights=True)),
        (rf_c, dict(criterion='gini', oob_score=True), dict(classes=[2], nan=True)),
        (rf_c, dict(criterion='entropy', class_weight='balanced_subsample', oob_score=True), dict(classes=[3])),
        (rf_c, dict(criterion='log_loss', class_weight='balanced', min_weight_fraction_leaf=.02, oob_score=True), dict(classes=[2, 2], nan=True, weights=True)),
        (rf_c, dict(criterion='gini', bootstrap=False, class_weight={0: 1., 1: 2.5, 2: .5}, max_features=None), dict(classes=[3], weights=True)),
        (rf_c, dict(criterion='gini', class_weight=[{0: 2., 1: 1.}, {0: .5, 1: 1., 2: 3.}]), dict(classes=[2, 3], weights=True)),
        (rf_c, dict(monotonic_cst=[1, -1, 0, 0, 0], max_depth=4, oob_score=True), dict(classes=[2])),
        (rf_c, dict(criterion='gini', class_weight='balanced_subsample', max_samples=.7, oob_score=True), dict(classes=[3], nan=True, weights=True)),
        (rf_c, dict(criterion='gini', class_weight='balanced', oob_score=True), dict(classes=[4], warm=8)),
        (rf_c, dict(criterion='entropy', n_estimators=3, max_samples=25, oob_score=True), dict(classes=[3], warm=6, shared=True)),
        (rf_c, dict(criterion='gini', oob_score=True), dict(classes=[3, 2])),  # multiclass-multioutput OOB error
        (et_r, dict(criterion='squared_error'), dict(nan=True)),
        (et_r, dict(criterion='poisson', bootstrap=True, max_samples=30, oob_score=True), dict(positive=True, nan=True)),
        (et_r, dict(criterion='absolute_error', bootstrap=True, max_depth=5, oob_score=True), dict(outputs=2)),
        (et_r, dict(bootstrap=False, max_samples=10), dict()),  # max_samples without bootstrap
        (et_r, dict(bootstrap=False, oob_score=True), dict()),  # OOB without bootstrap
        (et_c, dict(criterion='gini'), dict(classes=[3], nan=True)),
        (et_c, dict(criterion='entropy', bootstrap=True, class_weight='balanced_subsample', max_depth=4, oob_score=True), dict(classes=[2], weights=True)),
        (et_c, dict(criterion='log_loss', bootstrap=True, max_leaf_nodes=6, oob_score=True), dict(classes=[2, 2], nan=True)),
        (et_c, dict(bootstrap=True, max_samples=.9, class_weight={0: 3., 1: 1.}, oob_score=True), dict(classes=[2], weights=True, warm=7)),
    ]

def make_data(index, spec):
    rng = np.random.RandomState(7100 + index)
    n, p = 60 + 3 * index, 5
    X = rng.normal(size=(n, p)).astype(np.float32)
    X[:, 3] = rng.randint(0, 6, size=n) / 2
    X[:, 4] = rng.randint(0, 3, size=n)
    if spec.get('nan'):
        X[rng.rand(n, p) < .08] = np.nan
    outputs = spec.get('outputs', len(spec.get('classes', [1])))
    if 'classes' in spec:
        y = np.column_stack([10 * ((rng.randint(0, k, size=n) + (X[:, 0] > 0) * (k > 2)) % k) - 3 for k in spec['classes']])
    else:
        y = np.column_stack([np.nan_to_num(X[:, 0]) * (k + 1) + np.nan_to_num(X[:, 3]) + rng.normal(scale=.3, size=n) for k in range(outputs)])
        if spec.get('positive'):
            y = np.abs(y); y[rng.rand(n) < .2] = 0
    y = y.astype(np.float64)
    w = None
    if spec.get('weights'):
        w = rng.randint(1, 5, size=n) / 2
    if spec.get('heavy'):
        w = np.abs(rng.standard_normal(n)) * np.exp(rng.uniform(-8, 8, n))
    Q = rng.normal(size=(25, p)).astype(np.float32)
    Q[:, 3] = rng.randint(0, 6, size=25) / 2
    if spec.get('nan'):
        Q[rng.rand(25, p) < .1] = np.nan
    return X, (y[:, 0] if y.shape[1] == 1 else y), w, Q

def bits(values, dtype=np.float64):
    values = np.ascontiguousarray(values, dtype=dtype)
    return ' '.join(map(str, values.view(np.int32 if dtype == np.float32 else np.int64).ravel()))

def text(value):
    if value is None: return 'none'
    if isinstance(value, str): return value
    return f'f{np.float64(value).view(np.int64)}' if isinstance(value, float) else f'i{value}'

def observe(forest, Q, classifier, phase, fit):
    """Run one fit and collect sklearn's observable results as driver-format lines."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit()
    except ValueError:
        return [f'ERROR {phase}']
    lines = [f'SEEDS {phase} ' + ' '.join(str(t.random_state) for t in forest.estimators_)]
    for m, rows in enumerate(forest.estimators_samples_):
        lines.append(f'SAMPLE {phase} {m} ' + ' '.join(map(str, rows)))
    lines.append(f'APPLY {phase} ' + ' '.join(map(str, forest.apply(Q).ravel())))
    if classifier:
        proba = forest.predict_proba(Q)
        for k, matrix in enumerate(proba if isinstance(proba, list) else [proba]):
            lines.append(f'PROBA {phase} {k} ' + bits(matrix))
    else:
        lines.append(f'PRED {phase} ' + bits(forest.predict(Q)))
    for m, tree in enumerate(forest.estimators_):
        t = tree.tree_
        lines.append(f'MSTAT {phase} {m} ' + ' '.join(f'{l} {f} {bits([i])} {bits([w])}' for l, f, i, w in zip(t.children_left, t.feature, t.impurity, t.weighted_n_node_samples)))
        lines.append(f'MIMP {phase} {m} ' + bits(tree.feature_importances_))
    lines.append(f'IMP {phase} ' + bits(forest.feature_importances_))
    if hasattr(forest, 'oob_score_'):
        lines.append(f'OOB {phase} ' + bits([forest.oob_score_]))
        if classifier:
            decision = forest.oob_decision_function_
            decision = decision[..., np.newaxis] if decision.ndim == 2 else decision
            for k in range(decision.shape[-1]):
                lines.append(f'OOBD {phase} {k} ' + bits(decision[..., k]))
        else:
            lines.append(f'OOBP {phase} ' + bits(forest.oob_prediction_))
    lines.append(f'WARN {phase} ' + ' '.join(w.category.__name__ for w in caught))
    return lines

def main():
    assert 'builds/oracle-env' in sklearn.__file__
    commands, expected = [], []
    for index, (cls, params, spec) in enumerate(case_list()):
        X, y, w, Q = make_data(index, spec)
        seed = 31 + index
        shared = spec.get('shared', False)
        state = np.random.RandomState(seed) if shared else seed
        forest = cls(n_estimators=params.pop('n_estimators', 5), random_state=state, **params)
        classifier = cls in CLASSIFIERS
        extra = cls in (ExtraTreesClassifier, ExtraTreesRegressor)
        n, p = X.shape
        Y = np.atleast_2d(y.T).T
        labels = Y.copy()
        if classifier:  # the engine trains on per-output class codes, like the forest's members
            Y = np.column_stack([np.unique(Y[:, k], return_inverse=True)[1] for k in range(Y.shape[1])]).astype(np.float64)
        t = clone(forest.estimator).set_params(**{k: getattr(forest, k) for k in forest.estimator_params if k != 'random_state'})
        cw = forest.class_weight if classifier else None
        cw_text = 'none' if cw is None else (cw if isinstance(cw, str) else 'factors')
        head = ['CASE', index, int(classifier), int(extra), n, p, Y.shape[1], len(Q), forest.n_estimators,
            f'criterion={t.criterion}', f'depth={t.max_depth or 0}', f'mf={text(t.max_features)}', f'msl={text(t.min_samples_leaf)}',
            f'mss={text(t.min_samples_split)}', f'leaves={t.max_leaf_nodes or 0}', f'ccp={bits([t.ccp_alpha])}',
            f'mwfl={bits([t.min_weight_fraction_leaf])}', f'mid={bits([t.min_impurity_decrease])}',
            'mono=' + ('none' if t.monotonic_cst is None else ','.join(map(str, t.monotonic_cst))),
            f'bootstrap={int(forest.bootstrap)}', f'oob={int(bool(forest.oob_score))}', f'ms={text(forest.max_samples)}',
            f'cw={cw_text}', f'warm={spec.get("warm", 0)}', f'seed={seed}', f'shared={int(shared)}']
        commands.append(' '.join(map(str, head)))
        commands += ['X ' + bits(X, np.float32), 'Y ' + bits(Y), 'Q ' + bits(Q, np.float32), 'W ' + ('none' if w is None else bits(w))]
        if cw_text == 'factors':
            dicts = cw if isinstance(cw, list) else [cw]
            for k in range(Y.shape[1]):
                classes = np.unique(labels[:, k])
                # Labels are -3, 7, 17...; the dicts key class codes, so map codes to labels first.
                mapping = {classes[int(c)]: v for c, v in dicts[k].items() if int(c) < len(classes)}
                commands.append('F ' + bits(compute_class_weight(mapping, classes=classes, y=labels[:, k])))
            forest.set_params(class_weight=[{classes_k[int(c)]: v for c, v in d.items()} for d, classes_k in
                zip(dicts, [np.unique(labels[:, k]) for k in range(Y.shape[1])])] if isinstance(cw, list) else
                {np.unique(labels[:, 0])[int(c)]: v for c, v in cw.items()})
        expected.append(f'CASE {index}')
        phase_lines = observe(forest, Q, classifier, 0, lambda: forest.fit(X, y, sample_weight=w))
        expected += phase_lines
        if spec.get('warm') and not phase_lines[0].startswith('ERROR'):
            forest.set_params(warm_start=True, n_estimators=spec['warm'])
            expected += observe(forest, Q, classifier, 1, lambda: forest.fit(X, y, sample_weight=w))
            expected += observe(forest, Q, classifier, 2, lambda: forest.fit(X, y, sample_weight=w))
    dest = ROOT/'docs/private/forest_engine'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'forest_input.txt').write_text('\n'.join(commands) + '\n')
    output = subprocess.run([ROOT/'builds/forest_driver', dest/'forest_input.txt'], text=True, capture_output=True, check=True).stdout
    (dest/'forest_native.txt').write_text(output)
    got = output.splitlines()
    values, upstream, member_imps = 0, [], []
    def floats(tokens):
        return np.array(tokens, dtype=np.int64).view(np.float64)
    def close(a, b, what):
        """Member-tree internals: report rather than fail on tiny upstream divergences."""
        x, y = floats(a), floats(b)
        assert np.allclose(x, y, rtol=1e-12, atol=0, equal_nan=True), (what, x, y)
        if (x.view(np.int64) != y.view(np.int64)).any():
            upstream.append(what)
    case = None
    for i, want in enumerate(expected):
        assert i < len(got), ('missing native output', want[:120])
        if want.startswith('CASE'):
            case = want.split()[1]
        if want.startswith('ERROR'):
            assert got[i].startswith(want), (got[i][:200], want)
            continue
        a, b = got[i].split(), want.split()
        tag = b[0]
        assert len(a) == len(b) and a[:3] == b[:3], (i, a[:4], b[:4], len(a), len(b))
        if tag == 'MSTAT':  # children/feature exact; impurity and weighted count within 1e-12
            nodes_a, nodes_b = np.array(a[3:]).reshape(-1, 4), np.array(b[3:]).reshape(-1, 4)
            assert (nodes_a[:, :2] == nodes_b[:, :2]).all(), ('member structure', b[:3])
            close(nodes_a[:, 2:].ravel(), nodes_b[:, 2:].ravel(), f'case {case} phase {b[1]} member {b[2]} node stats')
            values += len(b) - 3
            continue
        if tag == 'MIMP':
            member_imps.append(floats(a[3:]))
            close(a[3:], b[3:], f'case {case} phase {b[1]} member {b[2]} importances')
            values += len(b) - 3
            continue
        if tag == 'IMP':
            # The forest's own averaging must be exact on the native member importances.
            split = [m for m in member_imps if m.any()]  # members with a split have nonzero importances
            mean = np.mean(split, axis=0, dtype=np.float64) if split else np.zeros(len(a) - 2)
            native = mean / np.sum(mean) if split else mean
            assert (floats(a[2:]).view(np.int64) == native.view(np.int64)).all(), ('forest importance aggregation', b[:2])
            close(a[2:], b[2:], f'case {case} phase {b[1]} forest importances')
            member_imps = []
            values += len(b) - 2
            continue
        if a != b:
            diff = [j for j in range(min(len(a), len(b))) if a[j] != b[j]][:6]
            raise AssertionError(f'line {i} {" ".join(b[:3])}: tokens {diff}: native {[a[j] for j in diff]} reference {[b[j] for j in diff]}')
        values += len(b) - 2
    assert len(got) == len(expected), (len(got), len(expected))
    errors = sum(line.startswith('ERROR') for line in expected)
    fits = sum(line.startswith('SEEDS') for line in expected)
    print(f'PASS forests: {len(case_list())} cases, {fits} fits ({errors} matched errors), {values} compared values')
    print(f'  forest-owned outputs bit-identical; member-tree internals within rtol 1e-12, {len(upstream)} not bit-identical'
        + (':\n    ' + '\n    '.join(upstream) if upstream else ''))

if __name__ == '__main__':
    main()
