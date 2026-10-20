import json
import subprocess
import warnings
import numpy as np
from oracle import load, ROOT

oracle = load()
CASES = 0

def compare(mode, strategy, y, weights=None, constant=None, quantile=None, seed=42, count=11, dtype=np.float64, exact=False):
    global CASES
    y = np.asarray(y, dtype=dtype if mode == 'reg' else None)
    if weights is not None: weights = np.asarray(weights, dtype=dtype)
    if constant is not None: constant = np.atleast_1d(np.asarray(constant, dtype=dtype if mode == 'reg' else None))
    native_mode = mode + ('32' if dtype == np.float32 else '')
    tolerance = 3e-6 if dtype == np.float32 else 2e-12
    n = len(y)
    outputs = y.shape[1] if y.ndim == 2 else 1
    kind = 2 if y.dtype.kind in 'US' else 1 if y.dtype.kind == 'f' else 0
    args = [ROOT/'builds/tests/dummy/probe', native_mode, strategy, str(seed), str(-1 if quantile is None else quantile), str(n), str(outputs), str(count), str(kind), str(int(weights is not None)), str(int(constant is not None)), str(y.ndim)]
    args += [str(x) for x in y.ravel()]
    args += [str(x) for x in (np.ones(n) if weights is None else weights)]
    if constant is not None: args += [str(x) for x in np.atleast_1d(constant)]
    actual = subprocess.run(args, capture_output=True, text=True)
    X = np.zeros((n, 2), dtype=dtype); test = np.zeros((count, 5), dtype=dtype)
    if mode == 'global': np.random.seed(seed)
    rng = np.random.RandomState(seed) if mode == 'mutable' else (None if mode == 'global' else seed)
    model = (oracle.DummyRegressor(strategy=strategy, constant=constant, quantile=quantile) if mode == 'reg'
             else oracle.DummyClassifier(strategy=strategy, constant=constant, random_state=rng))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X, y, sample_weight=weights)
            prediction = model.predict(test)
    except (ValueError, TypeError, ZeroDivisionError):
        assert actual.returncode != 0 and (not actual.stdout or actual.stdout.startswith('stack trace was not collected.')), ('native accepted invalid reference fit/predict', args, actual.stdout)
        CASES += 1
        return
    assert actual.returncode == 0, (args, actual.stderr)
    lines = iter(actual.stdout.splitlines())
    def array(expected, tolerance=tolerance):
        bits = int(next(lines))
        assert np.asarray(expected).dtype == np.dtype(f'float{bits}'), (native_mode, strategy, bits, np.asarray(expected).dtype)
        ndim = int(next(lines)); shape = tuple(int(next(lines)) for _ in range(ndim))
        assert shape == np.shape(expected), (args, shape, np.shape(expected))
        # Native output prints shortest round-trip decimals; parse them back in the emitted dtype.
        got = np.array([float(next(lines)) for _ in range(np.size(expected))]).astype(f'float{bits}').reshape(shape)
        np.testing.assert_allclose(got, expected, atol=tolerance, rtol=tolerance, equal_nan=True, err_msg=str(args[:12]))
    def labels(expected):
        got = [next(lines) for _ in range(np.size(expected))]
        if kind == 2: np.testing.assert_array_equal(got, np.asarray(expected).ravel())
        else: np.testing.assert_array_equal(np.array(got, dtype=float if kind == 1 else np.int64), np.asarray(expected).ravel())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if mode == 'reg':
            # np.median is a plain middle-pair average, so median results are compared exactly.
            array(model.constant_, 0 if exact else tolerance)
            array(prediction, 0 if exact else tolerance)
            array(model.predict(test, return_std=True)[1])
            np.testing.assert_allclose(float(next(lines)), model.score(X, y, sample_weight=weights), atol=tolerance, rtol=tolerance, equal_nan=True)
        else:
            classes = [model.classes_] if outputs == 1 else model.classes_
            priors = [model.class_prior_] if outputs == 1 else model.class_prior_
            for c, p in zip(classes, priors):
                assert int(next(lines)) == len(c)
                labels(c); array(p)
            for iteration in range(2):
                labels(prediction if iteration == 0 else model.predict(test))
                p = model.predict_proba(test)
                if outputs == 1: p = [p]
                if mode not in ('mutable', 'global'):
                    lp = model.predict_log_proba(test)
                    if outputs == 1: lp = [lp]
                for j in range(outputs):
                    array(p[j])
                    if mode not in ('mutable', 'global'): array(lp[j])
            try:
                expected_score = model.score(X, y, sample_weight=weights)
            except ValueError:
                expected_score = None
            line = next(lines)
            if expected_score is None:
                assert line == 'SCORE_ERROR', (args[:12], line)
            else:
                np.testing.assert_allclose(float(line), expected_score, atol=tolerance, rtol=tolerance, equal_nan=True)
    assert list(lines) == []
    CASES += 1

from functools import partial
_compare = compare
for dtype in [np.float64, np.float32]:
    compare = partial(_compare, dtype=dtype)
    for y in [[0., 1., 2., 8.], [[0.], [1.], [2.], [8.]], [[1., 2.], [9., -2.], [3., 5.], [1., 8.]], [3.], [7., 7., 7.]]:
        for strategy in ['mean', 'median', 'quantile', 'constant']:
            for weights in [None, np.ones(len(y)), np.arange(len(y)) + 1., np.array([0.]*(len(y)-1)+[1.])]:
                for q in ([0., .1, .5, .9, 1.] if strategy == 'quantile' else [None]):
                    outputs = np.asarray(y).shape[1] if np.asarray(y).ndim == 2 else 1
                    compare('reg', strategy, y, weights, [4.] * outputs if strategy == 'constant' else None, q)
    for strategy in ['mean','median','quantile']:
        for weights in [[0.,0.,0.], [2.,-1.,3.]]:
            compare('reg', strategy, [1.,3.,9.], weights, quantile=.5 if strategy == 'quantile' else None)
    for strategy in ['mean','median','quantile','constant']:
        compare('reg',strategy,[[1.,2.],[3.,4.]],weights=[0.,0.],constant=[0.,0.] if strategy=='constant' else None,quantile=.5 if strategy=='quantile' else None)
    # Non-integral values and weights exercise dtype rounding near CDF boundaries.
    for strategy in ['mean', 'median', 'quantile']:
        for weights in [None, [.1, .2, .3, .4]]:
            for q in ([0., .1, .3, .5, .9, 1.] if strategy == 'quantile' else [None]):
                compare('reg', strategy, [.125, -.3, 1.1, 7.25], weights, quantile=q)
    # Even-length unweighted medians average the middle pair exactly like np.median.
    values = np.random.RandomState(7).normal(size=(6, 3)) * np.array([1e-4, 1., 1e3])
    for n in [2, 4, 6]:
        compare('reg', 'median', values[:n, 0], exact=True)
        compare('reg', 'median', values[:n], exact=True)
    for strategy in ['bad', 'quantile', 'constant']:
        compare('reg',strategy,[1.,2.])
    for q in [-1., 2.]: compare('reg','quantile',[1.,2.],quantile=q)
    # Multilabel indicators, integral floats and string multioutput exercise accuracy target types.
    for y in [[3,1,1,2,3], [[3],[1],[1],[2],[3]], [[2,0],[2,1],[1,0],[1,1],[3,1]], ['東京','Paris','Paris','Berlin','α'], [2**60,2**60+1,2**60,2**60+1,2**60], [1.5,2.5,1.5,3.25,2.5],
              [[0,1],[1,1],[0,0],[1,0],[1,1]], [1.,2.,2.,3.,1.], [['a','x'],['b','x'],['a','y'],['a','x'],['b','y']]]:
        for strategy in ['prior','most_frequent','constant','uniform','stratified']:
            for weights in [None, [0.,1.,2.,0.,3.]]:
                for seed in [0,42,4294967295]:
                    for mode in (['clf','mutable','global','flat'] if np.asarray(y).ndim == 1 or np.asarray(y).shape[1] == 1 else ['clf','mutable','global']):
                        constant = np.asarray(y)[0] if strategy == 'constant' else None
                        compare(mode,strategy,y,weights,constant,seed=seed)
    for strategy in ['uniform','stratified']:
        compare('clf',strategy,[8,8,8],count=0)
        compare('clf',strategy,[8,8,8],count=1)
    compare('clf','constant',[1,2,3],constant=[8])
    compare('clf','bad',[1,2,3])
    for strategy in ['prior','most_frequent','uniform','stratified','constant']:
        compare('clf',strategy,[1,2,3],weights=[0.,0.,0.],constant=[1] if strategy=='constant' else None)
(ROOT/'docs/private/dummy.estimators/parity.json').write_text(json.dumps({'cases':CASES,'tolerances':{'float32':3e-6,'float64':2e-12},'dtypes_checked':True})+'\n')
print(f'PASS {CASES} pinned-source differential cases')
