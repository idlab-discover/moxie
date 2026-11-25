"""Compare native calibration to the recorded local sklearn revision."""
import importlib.util
import json
from pathlib import Path
import platform
import os
import subprocess
import sys
import warnings
from unittest.mock import patch

import numpy as np
import scipy
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.utils.metaestimators import available_if

ROOT = Path(__file__).resolve().parents[2]
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
SEED = 53117


def reference():
    source = ROOT.parent / 'scikit-learn'
    rev = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != REVISION:
        raise RuntimeError(f'Expected {REVISION}, found {rev}')
    # Read committed bytes, never uncommitted oracle edits.
    filename = ROOT / 'builds/calibration-parity/reference_calibration.py'
    filename.write_bytes(subprocess.check_output(['git', '-C', str(source), 'show', f'{REVISION}:sklearn/calibration.py']))
    spec = importlib.util.spec_from_file_location('sklearn._moxie_calibration_reference', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    iso_file = filename.with_name('reference_isotonic.py')
    iso_file.write_bytes(subprocess.check_output(['git', '-C', str(source), 'show', f'{REVISION}:sklearn/isotonic.py']))
    iso_spec = importlib.util.spec_from_file_location('sklearn._moxie_isotonic_reference', iso_file)
    iso = importlib.util.module_from_spec(iso_spec)
    iso_spec.loader.exec_module(iso)
    module.IsotonicRegression = iso.IsotonicRegression
    return module


def production_reference():
    from sklearn.linear_model import _cd_fast
    from sklearn.linear_model._glm import _newton_solver
    from sklearn.utils import _array_api, fixes

    source = ROOT.parent / 'scikit-learn'

    def load(path, package='sklearn'):
        name = path.rsplit('/', 1)[-1]
        filename = ROOT / 'builds/calibration-parity' / f'reference_{name}.py'
        filename.write_bytes(subprocess.check_output(
            ['git', '-C', str(source), 'show', f'{REVISION}:sklearn/{path}.py']
        ))
        spec = importlib.util.spec_from_file_location(f'{package}._moxie_{name}_reference', filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def supply_missing(target, pinned, *names):
        for name in names:
            if not hasattr(target, name):
                setattr(target, name, getattr(pinned, name))

    # The pinned source imports newer private helpers than the support wheel
    # provides. Supply the pinned implementations where those names are absent.
    supply_missing(_array_api, load('utils/_array_api', 'sklearn.utils'),
                   '_cov', '_swapaxes', '_unravel_index', 'array_device')
    supply_missing(fixes, load('utils/fixes', 'sklearn.utils'), '_is_gil_enabled')
    if not hasattr(_cd_fast, 'enet_coordinate_descent_sparse'):
        _cd_fast.enet_coordinate_descent_sparse = _cd_fast.sparse_enet_coordinate_descent
    supply_missing(_newton_solver, load('linear_model/_glm/_newton_solver', 'sklearn.linear_model._glm'),
                   'NewtonCDGramSolver', 'NewtonCDSolver')
    da = load('discriminant_analysis')
    logistic = load('linear_model/_logistic', 'sklearn.linear_model')
    logistic.LinearModelLoss = load(
        'linear_model/_linear_loss', 'sklearn.linear_model'
    ).LinearModelLoss
    svm_base = load('svm/_base', 'sklearn.svm')
    with patch.dict(sys.modules, {'sklearn.svm._base': svm_base}):
        svm = load('svm/_classes', 'sklearn.svm')
    return (
        load('naive_bayes').GaussianNB,
        da.LinearDiscriminantAnalysis,
        da.QuadraticDiscriminantAnalysis,
        logistic.LogisticRegression,
        svm.LinearSVC,
        svm.SVC,
    )


class Centroid(ClassifierMixin, BaseEstimator):
    def __init__(self, probability_response=False):
        self.probability_response = probability_response

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.centers_ = np.array([np.average(X[y == c], axis=0,
            weights=None if sample_weight is None else sample_weight[y == c]) for c in self.classes_])
        return self

    def _responses(self, X):
        return -((X[:, None, :] - self.centers_) ** 2).sum(axis=2)

    @available_if(lambda self: not self.probability_response)
    def decision_function(self, X):
        values = self._responses(X)
        return values[:, 1] - values[:, 0] if len(self.classes_) == 2 else values

    def predict_proba(self, X):
        values = self._responses(X)
        p = np.exp(values - values.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._responses(X).argmax(axis=1)]


def encode(value):
    if value is None:
        return '-'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int)):
        return str(value)
    a = np.asarray(value)
    return f'{a.shape[0]}:{a.shape[1] if a.ndim == 2 else 0}:' + ','.join(repr(float(v)) for v in a.flat)


def main():
    out = ROOT / 'builds/calibration-parity'
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref = reference()
    warnings.filterwarnings("ignore", message="Since FrozenEstimator does not appear to accept sample_weight")
    rng = np.random.default_rng(SEED)
    records = []

    def add(*fields):
        records.append('|'.join(encode(v) for v in fields))

    for size in [3, 13, 64, 101]:
        for scale in [1., 1e-3, 1e8]:
            x = rng.normal(size=size) * scale
            y = rng.integers(0, 2, size=size).astype(float)
            q = np.linspace(-3, 3, 21) * scale
            for w in [None, rng.uniform(.1, 2, size)]:
                model = ref._SigmoidCalibration().fit(x, y, w)
                add('sigmoid', x, y, w, q, model.predict(q))
    for label in [0., 1.]:
        for constant in [0., 3., 1e10]:
            x = np.full(17, constant)
            y = np.full(17, label)
            model = ref._SigmoidCalibration().fit(x, y)
            add('sigmoid', x, y, None, x, model.predict(x))
    for size in [3, 13, 64, 101]:
        x = rng.integers(-4, 5, size=size).astype(float)
        y = rng.integers(0, 2, size=size).astype(float)
        q = np.linspace(-5, 5, 31)
        for w in [None, rng.integers(0, 5, size=size).astype(float)]:
            if w is not None:
                w[0] = 1
            model = ref.IsotonicRegression(out_of_bounds='clip').fit(x, y, w)
            add('isotonic', x, y, w, q, model.predict(q))
    x = np.array([0., 1e-16, 2., 2., 3.])
    y = np.array([1., 0., 0., 1., 1.])
    model = ref.IsotonicRegression(out_of_bounds='clip').fit(x, y)
    add('isotonic', x, y, None, x, model.predict(x))
    for classes in [2, 3, 5]:
        for probabilities in [False, True]:
            x = rng.normal(size=(41, classes)) * 3
            q = rng.normal(size=(17, classes)) * 3
            if probabilities:
                x = scipy.special.softmax(x, axis=1)
                q = scipy.special.softmax(q, axis=1)
            y = rng.integers(classes, size=41).astype(float)
            for w in [None, rng.uniform(.1, 2, 41)]:
                model = ref._TemperatureScaling().fit(x, y, w)
                add('temperature', x, y, w, q, model.predict(q))
    x = rng.normal(size=31)
    y = (x > 0).astype(float)
    model = ref._TemperatureScaling().fit(x, y)
    add('temperature', x, y, None, x, model.predict(x))
    for strategy in ['uniform', 'quantile']:
        for bins in [1, 2, 5, 100]:
            p = rng.choice([0., .1, .2, .5, 1.], size=37)
            y = rng.choice([-3., 2.], size=37)
            a, b = ref.calibration_curve(y, p, n_bins=bins, strategy=strategy, pos_label=2)
            add('curve', p, y, None, b, a, bins, strategy, 2)
    # Bin-edge arithmetic: numpy's linspace edges are j * (1 / n) and its
    # percentile lerp switches formula at t >= 0.5. Probabilities one ulp
    # around those edges and tied quantiles must land in the same bins.
    for bins in [3, 7, 10, 29, 100]:
        edges = np.linspace(0, 1, bins + 1)[1:-1]
        p = np.concatenate([edges, np.nextafter(edges, 0), np.nextafter(edges, 1),
            [j / bins for j in range(1, bins)]])
        p = np.clip(p, 0, 1)
        y = rng.integers(0, 2, size=p.size).astype(float)
        a, b = ref.calibration_curve(y, p, n_bins=bins, strategy='uniform', pos_label=1)
        add('curve', p, y, None, b, a, bins, 'uniform', 1)
    for size in [2, 7, 40, 97]:
        for bins in [2, 3, 6, 13]:
            p = np.round(rng.uniform(size=size), 2)
            p = np.concatenate([p, np.nextafter(p, 1)])
            y = rng.integers(0, 2, size=p.size).astype(float)
            a, b = ref.calibration_curve(y, p, n_bins=bins, strategy='quantile', pos_label=1)
            add('curve', p, y, None, b, a, bins, 'quantile', 1)
    for classes in [2, 3]:
        X = rng.normal(size=(60, 4))
        y = np.tile(np.arange(classes), 60 // classes).astype(float)
        rng.shuffle(y)
        X[:, 0] += y * .6
        y = y * 3 - 2
        q = rng.normal(size=(19, 4))
        for method in ['sigmoid', 'isotonic', 'temperature']:
            for ensemble in [False, True]:
                for probability in [False, True]:
                    for frozen in [False, True]:
                        w = rng.uniform(.2, 2, 60)
                        # The pinned temperature CV path only encodes string labels.
                        # Its multinomial loss requires consecutive class indices.
                        oracle_y = np.searchsorted(np.unique(y), y) if method == 'temperature' else y
                        estimator = Centroid(probability)
                        if frozen:
                            estimator = FrozenEstimator(estimator.fit(X, oracle_y, w))
                        model = ref.CalibratedClassifierCV(estimator, method=method, cv=3, ensemble=ensemble).fit(X, oracle_y, sample_weight=w)
                        predicted = model.predict(q)
                        if method == 'temperature':
                            predicted = np.unique(y)[predicted.astype(int)]
                        add('cv', X, y, w, q, model.predict_proba(q), method, ensemble, probability, frozen, predicted)
    # Materialized iterables, missing train/test classes, repeated holdouts,
    # and the default splitter's minority-class warning path.
    for classes in [2, 3, 4]:
        X = rng.normal(size=(24, 3))
        y = np.repeat(np.arange(classes), 24 // classes).astype(float)
        q = rng.normal(size=(13, 3))
        splits = list(KFold(3).split(X, y))
        serialized = ';'.join(','.join(map(str, a)) + ':' + ','.join(map(str, b)) for a, b in splits)
        for method in ['sigmoid', 'isotonic']:
            for ensemble in [False, True]:
                for probability in [False, True]:
                    if not ensemble and not probability and classes == 3:
                        continue  # sklearn rejects binary decision folds in a multiclass partition.
                    w = rng.uniform(.2, 2, len(y))
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        model = ref.CalibratedClassifierCV(Centroid(probability), method=method, cv=splits, ensemble=ensemble).fit(X, y, sample_weight=w)
                    add('cv_splits', X, y, w, q, model.predict_proba(q), method, ensemble, probability, False, model.predict(q), serialized)
    X = rng.normal(size=(18, 3))
    y = np.array([0.] * 15 + [1.] * 3)
    q = rng.normal(size=(9, 3))
    for method in ['sigmoid', 'isotonic', 'temperature']:
        for ensemble in [False, True]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                model = ref.CalibratedClassifierCV(Centroid(True), method=method, ensemble=ensemble).fit(X, y)
            add('cv_default', X, y, None, q, model.predict_proba(q), method, ensemble, True, False, model.predict(q))
    splits = list(ShuffleSplit(4, test_size=.4, random_state=19).split(X))
    serialized = ';'.join(','.join(map(str, a)) + ':' + ','.join(map(str, b)) for a, b in splits)
    for method in ['sigmoid', 'isotonic', 'temperature']:
        model = ref.CalibratedClassifierCV(Centroid(True), method=method, cv=splits).fit(X, y)
        add('cv_splits', X, y, None, q, model.predict_proba(q), method, True, True, False, model.predict(q), serialized)
    for strategy in ['uniform', 'quantile']:
        for bins in [2, 5]:
            p = rng.uniform(size=37).astype(np.float32)
            y = rng.integers(0, 2, size=37).astype(np.float32)
            a, b = ref.calibration_curve(y, p, n_bins=bins, strategy=strategy)
            add('curve_float32', p, y, None, b, a, bins, strategy)
            y = np.where(y == 0, -3., 2.)
            labels = np.where(y == -3, 'cat', 'dog')
            for positive in ['cat', 'dog']:
                a, b = ref.calibration_curve(labels, p, n_bins=bins, strategy=strategy, pos_label=positive)
                add('curve_string', p, y, None, b, a, bins, strategy, positive)
            for positive in [-3., 2.]:
                disp = ref.CalibrationDisplay.from_predictions(y, p, n_bins=bins, strategy=strategy, pos_label=positive, name='Example', alpha=.8)
                add('display', p, y, None, disp.prob_pred, disp.prob_true, bins, strategy, str(positive), 'Example', p, disp.ax_.get_xlabel(), disp.ax_.get_ylabel())
                plt.close(disp.figure_)
    p = rng.uniform(size=20)
    y = rng.integers(0, 2, size=20).astype(float)
    disp = ref.CalibrationDisplay.from_predictions(y, p)
    add('display', p, y, None, disp.prob_pred, disp.prob_true, 5, 'uniform', '-', 'Classifier', p, disp.ax_.get_xlabel(), disp.ax_.get_ylabel())
    plt.close(disp.figure_)
    X = rng.normal(size=(30, 3))
    y = np.tile([-2., 3.], 15)
    estimator = Centroid().fit(X, y)
    for positive in [-2., 3.]:
        disp = ref.CalibrationDisplay.from_estimator(estimator, X, y, pos_label=positive)
        add('display_estimator', X, y, None, disp.prob_pred, disp.prob_true, 5, 'uniform', str(positive), 'Centroid', disp.y_prob, disp.ax_.get_xlabel(), disp.ax_.get_ylabel())
        plt.close(disp.figure_)
    # Keep these fixtures after the existing cases so their seeded draws stay stable.
    X = rng.normal(size=(61, 3))
    y = rng.permutation(np.repeat(np.arange(3), [23, 19, 19]))
    X[:, 0] += y * .6
    q = rng.normal(size=(13, 3))
    w = rng.uniform(.2, 2, len(y))
    for kind, cv in [
        ('cv_kfold', KFold(3)),
        ('cv_kfold_shuffled', KFold(3, shuffle=True, random_state=17)),
        ('cv_stratified_shuffled', StratifiedKFold(3, shuffle=True, random_state=17)),
    ]:
        for ensemble in [False, True]:
            for method in ['sigmoid', 'isotonic', 'temperature']:
                model = ref.CalibratedClassifierCV(
                    Centroid(), method=method, cv=cv, ensemble=ensemble
                ).fit(X, y, sample_weight=w)
                add(kind, X, y, w, q, model.predict_proba(q), method, ensemble,
                    False, False, model.predict(q))
    # Cover both response protocols, weighted fitting and class-column ordering
    # with production classifiers rather than a shared test double.
    gaussian, lda, qda, logistic, ref.LinearSVC, svc = production_reference()
    production = [
        ('gaussian', gaussian(), True),
        ('lda', lda(), False),
        ('qda', qda(), False),
        ('logistic', logistic(tol=1e-10, max_iter=500), True),
    ]
    for n_classes in (2, 3):
        codes = np.arange(90) % n_classes
        X = rng.normal(size=(len(codes), 3)) + codes[:, None] * np.array([0.8, -0.3, 0.5])
        q = rng.normal(size=(17, 3))
        w = rng.uniform(0.25, 2, size=len(codes))
        for method in ('sigmoid', 'isotonic', 'temperature'):
            y = codes if method == 'temperature' else 5 * codes - 7
            for ensemble in (False, True):
                model = ref.CalibratedClassifierCV(
                    svc(tol=1e-8), method=method, cv=3, ensemble=ensemble,
                ).fit(X, y)
                add('serialized_svc', X, y, None, q, model.predict_proba(q),
                    method, ensemble, model.predict(q), model.classes_)
        for name, estimator, weighted in production:
            for method in ('sigmoid', 'isotonic', 'temperature'):
                y = codes if method == 'temperature' else 5 * codes - 7
                for ensemble in (False, True):
                    sample_weight = w if weighted else None
                    model = ref.CalibratedClassifierCV(
                        estimator, method=method, cv=3, ensemble=ensemble
                    ).fit(X, y, sample_weight=sample_weight)
                    add(f'production_{name}', X, y, sample_weight, q, model.predict_proba(q),
                        method, ensemble, model.predict(q), model.classes_)
                if name != 'logistic':
                    prefit = estimator.fit(X[:45], y[:45])
                    frozen = ref.CalibratedClassifierCV(
                        FrozenEstimator(prefit), method=method, cv=3,
                    ).fit(X[45:], y[45:], sample_weight=w[45:])
                    add(f'production_frozen_{name}', X[45:], y[45:], w[45:], q,
                        frozen.predict_proba(q), method, 'auto', frozen.predict(q),
                        frozen.classes_, X[:45], y[:45])
        y = 5 * codes - 7
        prefit = logistic(tol=1e-10, max_iter=500).fit(X[:45], y[:45])
        model = ref.CalibratedClassifierCV(FrozenEstimator(prefit), cv=3).fit(
            X[45:], y[45:], sample_weight=w[45:]
        )
        add('production_frozen_logistic', X[45:], y[45:], w[45:], q, model.predict_proba(q),
            'sigmoid', 'auto', model.predict(q), model.classes_, X[:45], y[:45])
        for method in ('sigmoid', 'isotonic', 'temperature'):
            y = codes if method == 'temperature' else 5 * codes - 7
            for ensemble in ('auto', False, True):
                for sample_weight in (None, w):
                    model = ref.CalibratedClassifierCV(
                        method=method, ensemble=ensemble
                    ).fit(X, y, sample_weight=sample_weight)
                    assert isinstance(model.calibrated_classifiers_[0].estimator, ref.LinearSVC)
                    assert model.calibrated_classifiers_[0].estimator.random_state == 0
                    add('default_linear_svc', X, y, sample_weight, q,
                        model.predict_proba(q), method, ensemble, model.predict(q), model.classes_)

        X32 = X.astype(np.float32)
        q32 = q.astype(np.float32)
        for name, estimator in [('default', None), ('gaussian', gaussian())]:
            # Match the documented widening for GaussianNB, whose sklearn
            # implementation selects its statistics dtype from X.
            X_ref = X32 if estimator is None else X32.astype(np.float64)
            q_ref = q32 if estimator is None else q32.astype(np.float64)
            for method in ('sigmoid', 'isotonic', 'temperature'):
                y = codes if method == 'temperature' else 5 * codes - 7
                for ensemble in (False, True):
                    model = ref.CalibratedClassifierCV(
                        estimator, method=method, cv=3, ensemble=ensemble
                    ).fit(X_ref, y, sample_weight=w)
                    add(f'float32_{name}', X32, y, w, q32, model.predict_proba(q_ref),
                        method, ensemble, model.predict(q_ref), model.classes_,
                        [model.score(X_ref, y, sample_weight=w)])
    for n_classes in (2, 3):
        codes = np.arange(96) % n_classes
        X = rng.normal(size=(len(codes), 3)) + codes[:, None] * np.array([0.8, -0.3, 0.5])
        q = rng.normal(size=(19, 3))
        w = rng.uniform(0.25, 2, size=len(codes))
        for kind, offset, spacing in [
            ('list_int64', 2**40, 3),
            ('list_float32', 2**24, 2),
        ]:
            for method in ('sigmoid', 'isotonic', 'temperature'):
                backing = np.full(2 * len(codes), np.nan)
                y = backing[1::2]
                y[:] = codes if method == 'temperature' else offset + spacing * codes
                for ensemble in (False, True):
                    model = ref.CalibratedClassifierCV(
                        ref.LinearSVC(dual=False, random_state=0, tol=1e-10),
                        method=method, cv=3, ensemble=ensemble,
                    ).fit(X, y, sample_weight=w)
                    add(kind, X, y, w, q, model.predict_proba(q),
                        method, ensemble, model.predict(q), model.classes_)
    fixtures = out / 'cases.txt'
    fixtures.write_text('\n'.join(records) + '\n')
    command = ['mojo', 'build', '-O3', '--fp-mode=contract=off', '-I', 'builds/deps', '-I', 'examples/calibration_support', 'tests/calibration/parity_runner.mojo', '-o', str(out / 'runner')]
    subprocess.run(command, cwd=ROOT, check=True)
    result = subprocess.run([str(out / 'runner'), str(fixtures)], cwd=ROOT, text=True, capture_output=True)
    (out / 'results.txt').write_text(result.stdout + result.stderr)
    (out / 'environment.json').write_text(json.dumps(dict(reference=REVISION, seed=SEED, python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__,
        mojo=subprocess.check_output(['mojo', '--version'],cwd=ROOT,text=True).strip(),
        cases=len(records), command=command), indent=2) + '\n')
    print(result.stdout, end='')
    if result.returncode:
        raise RuntimeError(result.stderr)


if __name__ == '__main__':
    main()
