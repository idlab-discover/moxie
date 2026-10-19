"""Reproduce the member trees of real sklearn forests through the prepared-input engine.

For each case a reference forest is fitted with the source-built oracle. The
script recovers each member's integer seed and the exact sample_weight that
`_parallel_build_trees` passed to `tree._fit(check_input=False)`: bootstrap
counts (drawn with the forest's combined weights), optionally times
balanced_subsample factors, or the combined weights without bootstrap. The
native driver prepares the training input once per case and builds every
member from it. Floats cross the boundary as raw bits, so inputs are exact.

Every compared quantity must be identical to the reference: tree structure and
leaf IDs as integers; thresholds, impurities, weighted node counts, node values,
importances and member predictions bit for bit (NaN equal). Equal sample order
and evaluation order make this exact; summation order decides near-tied splits,
so bounded tolerances would hide structural divergence on realistic floats.
The mean of the native member predictions must equal forest.predict(_proba).
"""
from pathlib import Path
import subprocess
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble._forest import _generate_sample_indices
from sklearn.utils.class_weight import compute_sample_weight
ROOT = Path(__file__).resolve().parents[2]

def case_list():
    rf_r, rf_c, et_r, et_c = RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
    return [
        (rf_r, dict(criterion='squared_error'), dict(nan=True)),
        (rf_r, dict(criterion='friedman_mse', max_samples=.5, min_samples_leaf=.05, max_depth=5), dict(nan=True)),
        (rf_r, dict(criterion='poisson', max_depth=6), dict(nan=True, positive=True)),
        (rf_r, dict(criterion='absolute_error', max_leaf_nodes=8), dict(weights=True)),
        (rf_r, dict(criterion='squared_error', bootstrap=False, max_features=.6), dict(outputs=2, weights=True)),
        (rf_r, dict(monotonic_cst=[1, 0, -1, 0, 0], ccp_alpha=.01, max_samples=40), dict()),
        (rf_r, dict(criterion='squared_error', min_weight_fraction_leaf=.05, min_impurity_decrease=.01), dict(outputs=2, nan=True, weights=True)),
        (rf_c, dict(criterion='gini'), dict(classes=[2], nan=True)),
        (rf_c, dict(criterion='entropy', class_weight='balanced_subsample'), dict(classes=[3])),
        (rf_c, dict(criterion='log_loss', class_weight='balanced', min_weight_fraction_leaf=.02), dict(classes=[2, 3], nan=True)),
        (rf_c, dict(criterion='gini', bootstrap=False, class_weight={0: 1., 1: 2.5, 2: .5}, max_features=None), dict(classes=[3], weights=True)),
        (rf_c, dict(monotonic_cst=[1, -1, 0, 0, 0], max_depth=4), dict(classes=[2])),
        (rf_c, dict(criterion='gini', class_weight='balanced_subsample', max_samples=.7), dict(classes=[3], nan=True, weights=True)),
        (et_r, dict(criterion='squared_error'), dict(nan=True)),
        (et_r, dict(criterion='poisson', bootstrap=True, max_samples=30), dict(positive=True, nan=True)),
        (et_r, dict(criterion='absolute_error', bootstrap=True, max_depth=5), dict(outputs=2)),
        (et_c, dict(criterion='gini'), dict(classes=[3], nan=True)),
        (et_c, dict(criterion='entropy', bootstrap=True, class_weight='balanced_subsample', max_depth=4), dict(classes=[2], weights=True)),
        (et_c, dict(criterion='log_loss', bootstrap=True, max_leaf_nodes=6), dict(classes=[2, 2], nan=True)),
    ]

def make_data(index, spec):
    rng = np.random.RandomState(4100 + index)
    n, p = 60 + 3 * index, 5
    X = rng.normal(size=(n, p)).astype(np.float32)
    X[:, 3] = rng.randint(0, 6, size=n) / 2  # ties exercise thresholds
    X[:, 4] = rng.randint(0, 3, size=n)
    if spec.get('nan'):
        X[rng.rand(n, p) < .08] = np.nan
    outputs = spec.get('outputs', len(spec.get('classes', [1])))
    if 'classes' in spec:
        y = np.column_stack([(rng.randint(0, k, size=n) + (X[:, 0] > 0) * (k > 2)) % k for k in spec['classes']])
    else:
        y = np.column_stack([np.nan_to_num(X[:, 0]) * (k + 1) + np.nan_to_num(X[:, 3]) + rng.normal(scale=.3, size=n) for k in range(outputs)])
        if spec.get('positive'):
            y = np.abs(y)
            y[rng.rand(n) < .2] = 0
    y = y.astype(np.float64)
    w = rng.randint(1, 5, size=n) / 2 if spec.get('weights') else None
    Q = rng.normal(size=(25, p)).astype(np.float32)
    Q[:, 3] = rng.randint(0, 6, size=25) / 2
    if spec.get('nan'):
        Q[rng.rand(25, p) < .1] = np.nan
    return X, (y[:, 0] if y.shape[1] == 1 else y), w, Q

def member_weights(forest, tree, y, n):
    """The sample_weight `_parallel_build_trees` handed to this member's `_fit`."""
    if not forest.bootstrap:
        return forest._sample_weight
    indices = _generate_sample_indices(tree.random_state, n, forest._n_samples_bootstrap, forest._sample_weight)
    counts = np.bincount(indices, minlength=n).astype(np.float64)
    if forest.class_weight == 'balanced_subsample':
        counts = counts * compute_sample_weight('balanced', y, indices=indices)
    return counts

def bits(values, dtype):
    values = np.ascontiguousarray(values, dtype=dtype)
    return ' '.join(map(str, values.view(np.int32 if dtype == np.float32 else np.int64).ravel()))

def main():
    assert 'builds/tree/oracle-env' in sklearn.__file__
    lines, forests = [], []
    for index, (cls, params, spec) in enumerate(case_list()):
        X, y, w, Q = make_data(index, spec)
        forest = cls(n_estimators=5, random_state=17 + index, **params).fit(X, y, sample_weight=w)
        n, p = X.shape
        classifier = cls in (RandomForestClassifier, ExtraTreesClassifier)
        extra = cls in (ExtraTreesClassifier, ExtraTreesRegressor)
        template = forest.estimators_[0]
        Y = np.atleast_2d(y.T).T
        if classifier:  # forests train members on class indices
            Y = np.column_stack([np.unique(Y[:, k], return_inverse=True)[1] for k in range(Y.shape[1])]).astype(np.float64)
        mf = template.max_features
        mf_text = 'none' if mf is None else (mf if isinstance(mf, str) else (f'f{np.float64(mf).view(np.int64)}' if isinstance(mf, float) else f'i{mf}'))
        msl = template.min_samples_leaf
        mono = template.monotonic_cst
        lines.append(' '.join(map(str, ['CASE', index, int(classifier), int(extra), template.criterion, n, p, Y.shape[1], len(Q),
            template.max_depth or 0, mf_text, f'f{np.float64(msl).view(np.int64)}' if isinstance(msl, float) else f'i{msl}', template.max_leaf_nodes or 0,
            bits([template.ccp_alpha, template.min_weight_fraction_leaf, template.min_impurity_decrease], np.float64),
            'none' if mono is None else ','.join(map(str, mono)), len(forest.estimators_)])))
        lines += ['X ' + bits(X, np.float32), 'Y ' + bits(Y, np.float64), 'Q ' + bits(Q, np.float32)]
        lines.append('W ' + ('none' if w is None else bits(w, np.float64)))
        for tree in forest.estimators_:
            weights = member_weights(forest, tree, y, n)
            lines.append(f'M {tree.random_state} ' + ('none' if weights is None else bits(weights, np.float64)))
        forests.append((forest, Q, classifier))
    dest = ROOT/'docs/private/tree/forest_members'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'input.txt').write_text('\n'.join(lines) + '\n')
    output = subprocess.check_output([ROOT/'builds/tree/forest_member_driver', dest/'input.txt'], text=True)
    (dest/'native.txt').write_text(output)
    actual = {}
    for line in output.splitlines():
        tag, case, member, *fields = line.split()
        actual[tag, int(case), int(member)] = np.array(fields, dtype=np.int64).view(np.float64) if tag in ('VALUE', 'PRED', 'IMP', 'STAT') else np.array(fields, dtype=np.int64)
    checks = 0
    def close(key, expected):
        nonlocal checks
        got = actual[key]; expected = np.ravel(np.asarray(expected, dtype=np.float64))
        assert got.shape == expected.shape, (key, got.shape, expected.shape)
        same = (got.view(np.int64) == expected.view(np.int64)) | (np.isnan(got) & np.isnan(expected))
        assert same.all(), (key, np.flatnonzero(~same)[:5], got[~same][:5], expected[~same][:5])
        checks += got.size
    def same(key, expected):
        nonlocal checks
        np.testing.assert_array_equal(actual[key], np.ravel(np.asarray(expected, dtype=np.int64)), err_msg=str(key))
        checks += actual[key].size
    for case, (forest, Q, classifier) in enumerate(forests):
        member_predictions = []
        for m, tree in enumerate(forest.estimators_):
            t = tree.tree_
            same(('STRUCT', case, m), np.column_stack([t.children_left, t.children_right, t.feature, t.n_node_samples, t.missing_go_to_left]))
            close(('STAT', case, m), np.column_stack([np.where(t.children_left >= 0, t.threshold, 0), t.impurity, t.weighted_n_node_samples]))
            close(('VALUE', case, m), t.value)
            close(('IMP', case, m), tree.feature_importances_)
            same(('LEAF', case, m), tree.apply(Q))
            prediction = np.column_stack(tree.predict_proba(Q)) if classifier and isinstance(tree.predict_proba(Q), list) else (tree.predict_proba(Q) if classifier else tree.predict(Q))
            close(('PRED', case, m), prediction)
            member_predictions.append(actual['PRED', case, m].reshape(np.atleast_2d(prediction.T).T.shape))
        mean = sum(member_predictions[1:], member_predictions[0].copy()) / len(member_predictions)
        reference = forest.predict_proba(Q) if classifier else forest.predict(Q)
        reference = np.column_stack(reference) if isinstance(reference, list) else reference
        np.testing.assert_array_equal(mean, np.atleast_2d(reference.T).T if reference.ndim == 1 else reference, err_msg=f'forest mean {case}')
        checks += mean.size
    members = sum(len(f.estimators_) for f, _, _ in forests)
    print(f'PASS forest members: {checks} exact comparisons across {members} member trees in {len(forests)} reference forests')

if __name__ == '__main__':
    main()
