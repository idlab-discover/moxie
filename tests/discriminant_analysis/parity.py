"""Differential parity against the pinned scikit-learn source.

Python only generates fixtures from the pinned oracle (tests/oracle.py); the
compiled native runner (tests/parity_runner.mojo, built from the verified
assembly) fits every case in Mojo and compares. Environment:
MOXIE_PARITY_SEEDS (default "0,1,2") selects the random dataset seeds.
"""
from collections import defaultdict
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn import config_context
from sklearn.datasets import make_blobs
from sklearn.exceptions import UnsetMetadataPassedError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import load, ROOT, REVISION, _module
from named_input_parity import check_named_input

ORACLE = load()
DA = ORACLE['da']
SEEDS = [int(s) for s in os.environ.get('MOXIE_PARITY_SEEDS', '0,1,2').split(',')]
# Norm-wise tolerances relative to each compared array's largest magnitude.
# Float64: both sides compute in double precision with backward-stable
# factorizations (Jacobi vs LAPACK); 1e-10 covers the conditioning of these
# datasets. Float32: sklearn computes several paths in single precision while
# the native port accumulates in Float64 and rounds at the end, so the
# difference is dominated by sklearn's float32 rounding (~1e-6 relative),
# amplified by conditioning; 5e-4 is used.
TOLERANCE = {np.float64: 1e-10, np.float32: 5e-4}


def enc(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return f'{a.shape[0]}:-1:' + ','.join(repr(float(x)) for x in a)
    return f'{a.shape[0]}:{a.shape[1]}:' + ','.join(repr(float(x)) for x in a.ravel())


def enc_labels(labels):
    return ','.join(str(x) for x in labels)


class StandardizedLedoitWolf:
    """sklearn test helper (test_lda_ledoitwolf): Ledoit-Wolf on standardized data."""

    def fit(self, X):
        sc = ORACLE['data'].StandardScaler()
        X_sc = sc.fit_transform(X)
        s = ORACLE['shrunk'].ledoit_wolf(X_sc)[0]
        self.covariance_ = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
        return self


class DiagonalCovariance:
    """User-defined estimator: diagonal of the biased variance."""

    def fit(self, X):
        self.covariance_ = np.diag(np.var(X, axis=0))
        return self


def covariance_estimator(kind):
    if kind == 'none':
        return None
    if kind.startswith('shrunk:'):
        return ORACLE['shrunk'].ShrunkCovariance(shrinkage=float(kind.split(':')[1]), store_precision=False)
    if kind == 'lw':
        return ORACLE['shrunk'].LedoitWolf(store_precision=False)
    if kind == 'oas':
        return ORACLE['shrunk'].OAS(store_precision=False)
    if kind == 'stdlw':
        return StandardizedLedoitWolf()
    if kind == 'diag':
        return DiagonalCovariance()
    if kind == 'empirical':
        return ORACLE['empirical'].EmpiricalCovariance(store_precision=False)
    raise ValueError(kind)


def sign_aligned_columns(values):
    """Number of leading columns whose values are well separated (bases comparable)."""
    values = np.asarray(values, dtype=np.float64)
    scale = max(np.max(np.abs(values)) if values.size else 0.0, 1e-300)
    count = 0
    for i, v in enumerate(values):
        gaps = [abs(v - w) for j, w in enumerate(values) if j != i]
        if abs(v) <= 1e-9 * scale or (gaps and min(gaps) <= 1e-6 * scale):
            break
        count += 1
    return count


CASES = []


def add_case(estimator, name, X, y, T, dtype, labels='int', expect_extra='', **params):
    X = np.asarray(X, dtype=dtype)
    T = np.asarray(T, dtype=dtype)
    y = np.asarray(y)
    if labels == 'str':
        y = np.array([f'c{v}' for v in y])
    cov_kind = params.pop('covariance_estimator', 'none')
    kwargs = dict(params)
    if cov_kind != 'none':
        kwargs['covariance_estimator'] = covariance_estimator(cov_kind)
    cls = DA.LinearDiscriminantAnalysis if estimator == 'lda' else DA.QuadraticDiscriminantAnalysis
    lines = [f'case {len(CASES)}:{name}', f'est {estimator}', f'dtype {np.dtype(dtype).itemsize * 8}',
             f'labels {labels}', f'tol_check {TOLERANCE[dtype]!r}', f'cov {cov_kind}']
    for key in ['solver', 'shrinkage', 'priors', 'n_components', 'store_covariance', 'tol', 'reg_param']:
        if key in params:
            value = params[key]
            if value is None:
                text = 'none'
            elif isinstance(value, (list, tuple, np.ndarray)):
                text = ','.join(repr(float(v)) for v in value)
            elif isinstance(value, bool):
                text = str(int(value))
            else:
                text = str(value)
            lines.append(f'param {key} {text}')
    lines += [f'X {enc(X)}', f'y {enc_labels(y)}', f'T {enc(T)}']
    model = cls(**kwargs)
    try:
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter('always')
            model.fit(X, y)
    except Exception as exc:  # the reference rejected this configuration
        kind = type(exc).__name__
        kind = {'InvalidParameterError': 'ValueError'}.get(kind, kind)
        lines.append(f'error {kind}{expect_extra}')
        CASES.append('\n'.join(lines) + '\nend')
        return
    if estimator == 'lda':
        lines.append(f'fit_warning_count {len(fit_warnings)}')
        for i, warning in enumerate(fit_warnings):
            lines.append(f'fit_warning_{i} {warning.category.__name__}: {warning.message}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lines.append(f'classes {enc_labels(model.classes_)}')
        lines.append(f'means_ {enc(model.means_)}')
        lines.append(f'priors_ {enc(model.priors_)}')
        lines.append(f'decision {enc(model.decision_function(T))}')
        lines.append(f'proba {enc(model.predict_proba(T))}')
        lines.append(f'log_proba {enc(model.predict_log_proba(T))}')
        lines.append(f'predict {enc_labels(model.predict(T))}')
        lines.append(f'score {model.score(X, y)!r}')
        if estimator == 'lda':
            lines.append(f'coef_ {enc(model.coef_)}')
            lines.append(f'intercept_ {enc(model.intercept_)}')
            for attr in ['covariance_', 'explained_variance_ratio_', 'xbar_']:
                lines.append(f'{attr} {enc(getattr(model, attr)) if hasattr(model, attr) else "absent"}')
            lines.append(f'feature_names {len(model.get_feature_names_out())}')
            if hasattr(model, 'scalings_'):
                if params.get('solver', 'svd') == 'eigen':
                    evals = np.linalg.eigvals(np.linalg.solve(model.covariance_, _between(model, X, y, kwargs))).real
                    comparable = sign_aligned_columns(np.sort(evals)[::-1])
                else:
                    comparable = model.scalings_.shape[1]
                lines.append(f'scalings_ {enc(model.scalings_)}')
                lines.append(f'comparable {comparable}')
                lines.append(f'transform {enc(model.transform(T))}')
            else:
                lines += ['scalings_ absent', 'transform absent']
        else:
            lines.append(f'n_scalings {len(model.scalings_)}')
            for k, (s, r) in enumerate(zip(model.scalings_, model.rotations_)):
                lines.append(f'q_scaling_{k} {enc(s)}')
                lines.append(f'q_rotation_{k} {enc(r)}')
                lines.append(f'q_comparable_{k} {sign_aligned_columns(s)}')
            if hasattr(model, 'covariance_'):
                for k, c in enumerate(model.covariance_):
                    lines.append(f'q_covariance_{k} {enc(c)}')
            else:
                lines.append('q_covariance absent')
    CASES.append('\n'.join(lines) + '\nend')


def _between(model, X, y, kwargs):
    """Between-class scatter St - Sw as used by the eigen solver (for eigenvalue gaps)."""
    St = DA._cov(np.asarray(X, dtype=np.float64), kwargs.get('shrinkage'), kwargs.get('covariance_estimator'))
    return St - model.covariance_


def toy():
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    y3 = np.array([1, 1, 2, 2, 3, 3])
    X1 = np.array([[-2], [-1], [-1], [1], [1], [2]])
    X6 = np.array([[0, 0], [-2, -2], [-2, -1], [-1, -1], [-1, -2], [1, 3], [1, 2], [2, 1], [2, 2]])
    y6 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    y7 = np.array([1, 2, 3, 2, 3, 1, 2, 3, 1])
    X7 = np.array([[-3], [-2], [-1], [-1], [0], [1], [1], [2], [3]])
    X2 = np.array([[-3, 0], [-2, 0], [-1, 0], [-1, 0], [0, 0], [1, 0], [1, 0], [2, 0], [3, 0]])
    y4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2])
    return X, y, y3, X1, X6, y6, y7, X7, X2, y4


def composition_features(n):
    """Mirror the deterministic feature generator in composition.mojo."""
    i, j = np.ogrid[:n, :4]
    return ((i % 3) * (j % 2 + 1) * 1.5 + j * 10.0
            + (np.sin((i * 31 + j * 17) * 12.9898) + 0.5 * np.cos((i * 7 + j) * 3.1)) * (j + 1))


def check_pipeline_projection(output):
    """Compare both projection calls with one consistent sign per LDA axis."""
    projections = {}
    for line in output.splitlines():
        if line.startswith('projection '):
            _, dtype, split, *values = line.split()
            projections.setdefault((dtype, split), []).append([float(v) for v in values])
    assert set(projections) == {(dtype, split) for dtype in ('float32', 'float64')
                               for split in ('train', 'query')}

    X = composition_features(52)
    y = np.arange(45) % 3
    errors = {}
    # sklearn's Float32 scaler and SVD round earlier than the native solver.
    for dtype, tolerance in [(np.float64, 1e-10), (np.float32, 2e-5)]:
        features = X.astype(dtype)
        pipe = ORACLE['pipeline'].make_pipeline(ORACLE['data'].StandardScaler(),
                                                DA.LinearDiscriminantAnalysis(n_components=2))
        train = pipe.fit_transform(features[:45], y)
        query = pipe.transform(features[45:])
        actual_train = np.asarray(projections[dtype.__name__, 'train'])
        signs = np.where(np.sum(actual_train * train, axis=0) < 0, -1, 1)
        for split, expected in [('train', train), ('query', query)]:
            actual = np.asarray(projections[dtype.__name__, split]) * signs
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance,
                                       err_msg=f'{dtype.__name__} {split} projection')
            errors[f'{dtype.__name__}/{split}'] = float(np.max(np.abs(actual - expected)))
    print(f'PASS: pinned sklearn scaler → LDA projection, four comparisons (max errors {errors})')


def check_pipeline_routed_scores(output):
    records = {}
    for line in output.splitlines():
        if line.startswith('routed_score '):
            _, dtype, kind, score = line.split()
            assert (dtype, kind) not in records
            records[dtype, kind] = float(score)
    models = {'lda': DA.LinearDiscriminantAnalysis, 'qda': DA.QuadraticDiscriminantAnalysis}
    dtypes = (np.float32, np.float64)
    assert set(records) == {(dtype.__name__, kind) for dtype in dtypes for kind in models}

    routing = _module('sklearn.utils._da_pipeline_routing', 'sklearn/utils/_metadata_requests.py')
    training_y = np.arange(45) % 3
    score_y = training_y.copy()
    score_y[0] = 1
    for dtype in dtypes:
        X = composition_features(45).astype(dtype)
        weights = (1 + np.arange(45) % 7).astype(dtype)
        for kind, model in models.items():
            pipe = ORACLE['pipeline'].make_pipeline(ORACLE['data'].StandardScaler(), model())
            pipe.fit(X, training_y)
            final_name, final = pipe.steps[-1]
            with config_context(enable_metadata_routing=True):
                with np.testing.assert_raises(UnsetMetadataPassedError):
                    pipe.score(X, score_y, sample_weight=weights)
                final.set_score_request(sample_weight=False)
                with np.testing.assert_raises_regex(TypeError, 'unexpected argument'):
                    pipe.score(X, score_y, sample_weight=weights)
                final.set_score_request(sample_weight=True)
                expected = pipe.score(X, score_y, sample_weight=weights)
                final.set_score_request(sample_weight='validation_weight')
                # The pinned Pipeline.score inserts sample_weight=None alongside
                # the alias. Exercise the pinned router with the supplied key only.
                routed = routing.process_routing(pipe, 'score', validation_weight=weights)
                alias_score = final.score(pipe.steps[0][1].transform(X), score_y,
                                          **routed[final_name].score)
                assert alias_score == expected
            # The sklearn metric may round its weighted result in Float32.
            np.testing.assert_allclose(records[dtype.__name__, kind], expected, rtol=0,
                                       atol=1e-7 if dtype == np.float32 else 1e-15)
    print('PASS: pinned sklearn Pipeline score routing for LDA and QDA in Float32 and Float64')


def check_lda_search(output):
    records = defaultdict(list)
    for line in output.splitlines():
        if line.startswith('lda_'):
            field, dtype, *values = line.split()
            records[dtype, field].append([float(value) for value in values])

    labels = np.arange(61) % 3
    order = np.argsort(labels, kind='stable')
    y = labels[order]
    for dtype in (np.float32, np.float64):
        X = composition_features(61)[order].astype(dtype)
        model = DA.LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.25)
        search = ORACLE['search'].GridSearchCV(
            model, {'shrinkage': [1.0, 0.25, 0.0, 0.7]}, error_score='raise',
        ).fit(X, y)
        folds = ORACLE['split'].StratifiedKFold().split(X, y)
        results = search.cv_results_
        dtype_name = dtype.__name__
        np.testing.assert_array_equal(
            records[dtype_name, 'lda_index'],
            [(fold, row) for fold, (_, test) in enumerate(folds) for row in test],
        )
        np.testing.assert_array_equal(
            records[dtype_name, 'lda_cv'],
            [(fold, results[f'split{fold}_test_score'][1]) for fold in range(search.n_splits_)],
        )
        np.testing.assert_array_equal(
            records[dtype_name, 'lda_fold'],
            [(candidate, fold, results[f'split{fold}_test_score'][candidate])
             for candidate in range(len(results['params'])) for fold in range(search.n_splits_)],
        )
        np.testing.assert_allclose(
            records[dtype_name, 'lda_mean'], list(enumerate(results['mean_test_score'])),
            atol=1e-15, rtol=0,  # Only the mean reduction order can affect the result.
        )
        np.testing.assert_array_equal(
            records[dtype_name, 'lda_rank'], list(enumerate(results['rank_test_score'])),
        )
        np.testing.assert_array_equal(records[dtype_name, 'lda_best'], [[search.best_index_]])
        np.testing.assert_array_equal(records[dtype_name, 'lda_predict'], search.predict(X)[:, None])
    print('PASS: pinned sklearn LDA stratified CV and shrinkage grid search (Float32/Float64)')


def main():
    X, y, y3, X1, X6, y6, y7, X7, X2, y4 = toy()
    grid = [('svd', None), ('lsqr', None), ('eigen', None), ('lsqr', 'auto'), ('lsqr', 0),
            ('lsqr', 0.43), ('eigen', 'auto'), ('eigen', 0), ('eigen', 0.43)]
    # Classes have different horizontal variances and a shared vertical variance.
    # Far along the vertical axis, every unshifted score can exponentiate to zero
    # while the normalized class probabilities still have several nonzero entries.
    offsets = np.array([[-1, -2], [-1, 2], [1, -2], [1, 2]])
    qda_train = np.vstack([offsets * [width, 1] + [center, 0]
                           for center, width in [(-3, 1), (0, 1.5), (3, 2.5)]])
    qda_labels = np.repeat([1, 2, 3], len(offsets))
    axis = [-128, -32, -8, -4, -2, -1, 0, 1, 2, 4, 8, 32, 128]
    qda_queries = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    for dtype in [np.float64, np.float32]:
        for solver in ['svd', 'eigen']:
            add_case('qda', f'posterior-grid-{solver}', qda_train, qda_labels, qda_queries, dtype, solver=solver)
        # The plot_lda gallery example has one informative feature plus noise.
        # With 20 training rows and two classes, the covariance rank is at most 18.
        rng = np.random.RandomState(0)
        for width in [33, 49, 73]:
            X_train, y_train = make_blobs(20, centers=[[-2], [2]], random_state=rng)
            X_train = np.column_stack((X_train, rng.normal(size=(20, width - 1))))
            X_test, _ = make_blobs(200, centers=[[-2], [2]], random_state=rng)
            X_test = np.column_stack((X_test, rng.normal(size=(200, width - 1))))
            for shrinkage in [None, 0]:
                add_case('lda', f'rank-deficient-{width}-lsqr-{shrinkage}',
                         X_train, y_train, X_test, dtype, solver='lsqr', shrinkage=shrinkage)
        for labels in ['int', 'str']:
            for solver, shrinkage in grid:
                for name, Xa, ya in [('toy', X, y), ('toy1d', X1, y), ('toy3', X, y3), ('toy6', X6, y6), ('toy7', X6, y7)]:
                    add_case('lda', f'{name}-{solver}-{shrinkage}', Xa, ya, Xa, dtype, labels,
                             solver=solver, shrinkage=shrinkage, store_covariance=True)
            for solver in ['svd', 'eigen']:
                for name, Xa, ya in [('toy6', X6, y6), ('toy7', X7, y6), ('toy6y7', X6, y7), ('toy', X, y)]:
                    add_case('qda', f'{name}-{solver}', Xa, ya, Xa, dtype, labels, solver=solver, store_covariance=True)
        # Priors, n_components, tol, store_covariance defaults and error branches.
        for solver in ['svd', 'lsqr', 'eigen']:
            add_case('lda', 'priors', X6, y6, X6, dtype, solver=solver, priors=[0.2, 0.8])
            add_case('lda', 'priors-renorm', X6, y6, X6, dtype, solver=solver, priors=[0.5, 0.6])
            add_case('lda', 'priors-within-tolerance', X6, y6, X6, dtype, solver=solver, priors=[0.5, 0.500008])
            add_case('lda', 'priors-negative', X6, y6, X6, dtype, solver=solver, priors=[0.5, -0.5])
            add_case('lda', 'default-cov', X6, y6, X6, dtype, solver=solver)
            add_case('lda', 'n-samples-eq-classes', [[0.5, 0.6], [0.6, 0.5]], [0, 1], [[0.5, 0.6]], dtype, solver=solver)
            add_case('lda', 'n-components-too-large', X6, y7, X6, dtype, solver=solver, n_components=3)
            add_case('lda', 'n-components-1', X6, y7, X6, dtype, solver=solver, n_components=1)
            add_case('lda', 'bad-solver', X6, y7, X6, dtype, solver='foo')
            add_case('lda', 'bad-shrinkage', X6, y7, X6, dtype, solver='lsqr', shrinkage=1.5)
        add_case('lda', 'svd-shrinkage', X, y, X, dtype, solver='svd', shrinkage='auto')
        add_case('lda', 'svd-estimator', X, y, X, dtype, solver='svd', covariance_estimator='lw')
        add_case('lda', 'estimator-and-shrinkage', X, y, X, dtype, solver='lsqr', shrinkage=0.1, covariance_estimator='shrunk:0.5')
        add_case('lda', 'estimator-and-zero-shrinkage', X, y, X, dtype, solver='lsqr', shrinkage=0, covariance_estimator='shrunk:0.5')
        add_case('lda', 'svd-tol-large', X6, y7, X6, dtype, solver='svd', tol=0.5)
        add_case('lda', 'eigen-singular', X2, y6, X2, dtype, solver='eigen')
        add_case('lda', 'svd-constant-feature', X2, y6, X2, dtype, solver='svd')
        add_case('qda', 'priors-tiny', X, y, X, dtype, priors=[1e-30, 1 - 1e-30])
        add_case('qda', 'eigen-priors', X6, y6, X6, dtype, solver='eigen', priors=[0.5, 0.5])
        add_case('qda', 'one-sample-class', X6, y4, X6, dtype)
        add_case('qda', 'single-class', X6, np.ones(9, int), X6, dtype)
        add_case('qda', 'svd-not-full-rank', X2, y6, X2, dtype, solver='svd')
        add_case('qda', 'eigen-not-full-rank', X2, y6, X2, dtype, solver='eigen')
        add_case('qda', 'svd-reg', X2, y6, X2, dtype, solver='svd', reg_param=0.01, store_covariance=True)
        add_case('qda', 'eigen-shrinkage', X2, y6, X2, dtype, solver='eigen', shrinkage=0.01, store_covariance=True)
        add_case('qda', 'svd-shrinkage', X, y, X, dtype, solver='svd', shrinkage='auto')
        add_case('qda', 'svd-estimator', X, y, X, dtype, solver='svd', covariance_estimator='lw')
        add_case('qda', 'estimator-and-shrinkage', X, y, X, dtype, solver='eigen', shrinkage=0.1, covariance_estimator='shrunk:0.5')
        add_case('qda', 'bad-reg', X6, y6, X6, dtype, reg_param=2.0)
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            # Tall multiclass Gaussian blobs.
            means = rng.normal(scale=3, size=(3, 4))
            Xg = np.vstack([rng.normal(size=(20, 4)) @ rng.normal(size=(4, 4)) * 0.5 + m for m in means])
            yg = np.repeat([0, 1, 2], 20)
            Tg = rng.normal(scale=3, size=(12, 4))
            for solver, shrinkage in grid:
                add_case('lda', f'blobs-{seed}-{solver}-{shrinkage}', Xg, yg, Tg, dtype, solver=solver,
                         shrinkage=shrinkage, store_covariance=True)
            for solver in ['svd', 'eigen']:
                add_case('qda', f'blobs-{seed}-{solver}', Xg, yg, Tg, dtype, solver=solver, store_covariance=True)
            add_case('qda', f'blobs-{seed}-svd-reg', Xg, yg, Tg, dtype, solver='svd', reg_param=0.3)
            for shrinkage in ['auto', 0.2]:
                add_case('qda', f'blobs-{seed}-eigen-{shrinkage}', Xg, yg, Tg, dtype, solver='eigen', shrinkage=shrinkage)
            for kind in ['shrunk:0.3', 'stdlw', 'lw', 'diag', 'empirical', 'oas']:
                for solver in ['lsqr', 'eigen']:
                    add_case('lda', f'blobs-{seed}-{solver}-{kind}', Xg, yg, Tg, dtype, solver=solver,
                             covariance_estimator=kind)
                add_case('qda', f'blobs-{seed}-eigen-{kind}', Xg, yg, Tg, dtype,
                         solver='eigen', covariance_estimator=kind, store_covariance=True)
            # Four classes, strongly different feature scales (test_lda_scaling-like).
            Xs = np.vstack([rng.uniform(-1, 1, (15, 3)) + [c * 5, 0, 0] for c in range(4)]) * [1, 100, 10000]
            ys = np.repeat([3, 5, 7, 9], 15)
            Ts = rng.uniform(-1, 16, (10, 3)) * [1, 100, 10000]
            for solver in ['svd', 'lsqr', 'eigen']:
                add_case('lda', f'scaled-{seed}-{solver}', Xs, ys, Ts, dtype, solver=solver, store_covariance=True)
                add_case('lda', f'scaled-{seed}-{solver}-k2', Xs, ys, Ts, dtype, solver=solver, n_components=2)
            for solver in ['svd', 'eigen']:
                add_case('qda', f'scaled-{seed}-{solver}', Xs, ys, Ts, dtype, solver=solver)
            # Wide data (more features than samples): svd rank handling, shrinkage solvers.
            Xw = rng.normal(size=(12, 20))
            yw = np.tile([0, 1, 2], 4)
            Tw = rng.normal(size=(5, 20))
            add_case('lda', f'wide-{seed}-svd', Xw, yw, Tw, dtype, solver='svd')
            for solver in ['lsqr', 'eigen']:
                for shrinkage in ['auto', 0.3]:
                    add_case('lda', f'wide-{seed}-{solver}-{shrinkage}', Xw, yw, Tw, dtype, solver=solver, shrinkage=shrinkage)
            add_case('qda', f'wide-{seed}-eigen-auto', Xw, yw, Tw, dtype, solver='eigen', shrinkage='auto')
            add_case('qda', f'wide-{seed}-svd', Xw, yw, Tw, dtype, solver='svd')
            # Collinear duplicated feature: rank-deficient within-class data for svd.
            Xc = rng.normal(size=(30, 3))
            Xc = np.hstack([Xc, Xc[:, :1] * 2.0])
            yc = rng.randint(0, 2, size=30)
            yc[:2] = [0, 1]
            add_case('lda', f'collinear-{seed}-svd', Xc, yc, rng.normal(size=(6, 4)), dtype, solver='svd')
            add_case('lda', f'collinear-{seed}-lsqr-auto', Xc, yc, rng.normal(size=(6, 4)), dtype, solver='lsqr', shrinkage='auto')
            # Binary with extreme separation (probabilities saturate to 0/1).
            Xb = np.vstack([rng.normal(size=(10, 2)) - 30, rng.normal(size=(10, 2)) + 30])
            yb = np.repeat([0, 1], 10)
            for solver in ['svd', 'lsqr', 'eigen']:
                add_case('lda', f'saturated-{seed}-{solver}', Xb, yb, Xb, dtype, solver=solver)
            add_case('qda', f'saturated-{seed}', Xb, yb, Xb, dtype)
    out = ROOT / 'builds/discriminant_analysis/parity'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'fixtures.txt').write_text('\n'.join(CASES) + '\n')
    (out / 'metadata.json').write_text(json.dumps({
        'revision': REVISION, 'seeds': SEEDS, 'cases': len(CASES), 'numpy': np.__version__,
        'support_sklearn': __import__('sklearn').__version__, 'source_hashes': ORACLE['hashes'],
        'tolerances': {'float64': TOLERANCE[np.float64], 'float32': TOLERANCE[np.float32]}}, indent=2))
    runner = ROOT / 'builds/tests/discriminant_analysis/parity_runner'
    result = subprocess.run([str(runner), str(out / 'fixtures.txt')], capture_output=True, text=True)
    (out / 'result.txt').write_text(result.stdout + result.stderr)
    print(result.stdout[-4000:], end='')
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f'parity failed ({out / "result.txt"})')
    composition = ROOT / 'builds/tests/discriminant_analysis/composition'
    output = subprocess.check_output([str(composition), '--composition-parity'], text=True)
    check_pipeline_projection(output)
    check_pipeline_routed_scores(output)
    check_lda_search(output)
    check_named_input(DA, ROOT / 'builds/tests/discriminant_analysis/serialization')


if __name__ == '__main__':
    main()
