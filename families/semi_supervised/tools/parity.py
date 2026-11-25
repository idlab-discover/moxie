"""Compare native estimators with exact local sklearn algorithm sources."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform
import runpy
import subprocess
import warnings
import numpy as np
import sklearn
from check import ROOT, assemble, compile_file, run

REFERENCE = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
OUT = ROOT / 'builds/parity'
LOGISTIC_SEED = 2801
LOGISTIC_TOLERANCES = {
    'float64': {'probability': 2e-5, 'decision_or_log_probability': 1e-4},
    'float32': {'probability': 9e-4, 'decision_or_log_probability': 3e-3},
}
CALIBRATED_SVC_SEED = 2802
# LIBSVM kernel arithmetic and the calibration optimizer differ at low bits.
CALIBRATED_SVC_TOLERANCE = 3e-5
DIRECT_SVC_SEED = 2803
DIRECT_SVC_TOLERANCE = 3e-5


def reference(name, relative):
    path = REFERENCE / relative
    expected = subprocess.check_output(['git', '-C', str(REFERENCE), 'show', REVISION + ':' + relative])
    if path.read_bytes() != expected:
        raise RuntimeError(f'reference source drift: {relative}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal(value):
    if not np.isfinite(value):
        return f'Float64("{float(value)}")'
    result = format(float(value), '.17g')
    return result if '.' in result or 'e' in result else result + '.0'


def array(name, values, dtype='float64'):
    values = np.asarray(values)
    entries = ', '.join(literal(v) for v in values.ravel())
    return [f'    var {name} = zeros[DType.{dtype}](Shape({values.shape[0]}, {values.shape[1]}))',
            f'    var {name}_values = List[Float64]([{entries}])',
            f'    for i in range(len({name}_values)):',
            f'        {name}.unsafe_set(i, {name}_values[i].cast[DType.{dtype}]())']


def check_array(name, expected, tolerance):
    expected = np.asarray(expected)
    entries = ', '.join(literal(v) for v in expected.ravel())
    return [f'    check_matrix({name}, List[Float64]([{entries}]), {expected.shape[0]}, {expected.shape[1]}, {tolerance})']


def check_labels(name, expected, probabilities, classes, tolerance):
    probabilities = np.asarray(probabilities)
    entries = ', '.join(literal(v) for v in probabilities.ravel())
    return [f'    check_labels({name}, {ints(expected)}, List[Float64]([{entries}]), {ints(classes)}, {tolerance})']


def ints(values):
    return 'List[Int]([' + ', '.join(str(int(v)) for v in values) + '])'


def logistic_cases(training, logistic):
    rng = np.random.default_rng(LOGISTIC_SEED)
    for dtype, tolerances in LOGISTIC_TOLERANCES.items():
        proba_tol = tolerances['probability']
        decision_tol = tolerances['decision_or_log_probability']
        for n_classes in (2, 3):
            centers = [[-3., -1.], [3., -1.], [0., 4.]][:n_classes]
            X = np.concatenate([rng.normal(center, .9, (6, 2)) for center in centers]).astype(dtype)
            truth = np.repeat([35, -7, 12][:n_classes], 6)
            y = truth.copy()
            y.reshape(n_classes, 6)[:, 3:] = -1
            query = (X[1::2] + .2).astype(dtype)
            base = logistic(C=1.7, tol=1e-9, max_iter=500)
            model = training.SelfTrainingClassifier(base, threshold=.8)
            name = f'logistic_{dtype}_{n_classes}'
            lines = [f'@no_inline\ndef {name}() raises:'] + array('X', X, dtype) + array('Q', query, dtype)
            lines += [
                f'    var base = LogisticRegressionAdapter(LogisticRegression[DType.{dtype}](C=1.7, tol=1e-9, max_iter=500))',
                f'    var model = SelfTrainingClassifier[type_of(base), DType.{dtype}](base.copy(), threshold=.8)',
                '    with assert_raises():',
                '        _ = base.classes()',
                '    with assert_raises():',
                '        _ = model.predict_proba(Q)',
            ]
            for threshold, offset in ((.8, 0), (.95, 100)):
                targets = np.where(y == -1, -1, y + offset)
                model.set_params(threshold=threshold).fit(X, targets)
                lines += [
                    f'    _ = model.set_params(threshold={threshold}).fit(X, {ints(targets)})',
                    f'    assert_equal(model.classes_, {ints(model.classes_)})',
                    f'    assert_equal(model.labeled_iter_, {ints(model.labeled_iter_)})',
                    f'    assert_equal(model.n_iter_, {model.n_iter_})',
                    f'    assert_equal(model.termination_condition_, "{model.termination_condition_}")',
                ]
                for i, label in enumerate(model.transduction_):
                    lines.append(f'    assert_equal(model.transduction_[{i}].value(), {label})'
                                 if label != -1 else f'    assert_true(not model.transduction_[{i}])')
                lines += check_array('model.predict_proba(Q)', model.predict_proba(query), proba_tol)
                lines += check_array('model.predict_log_proba(Q)', model.predict_log_proba(query), decision_tol)
                decision = model.decision_function(query)
                lines += [f'    assert_equal(model.decision_function(Q).ndim, {decision.ndim})']
                decision = decision.reshape(len(query), -1)
                lines += check_array(
                    f'model.decision_function(Q).reshape(Shape({len(query)}, {decision.shape[1]}))',
                    decision, decision_tol)
                lines += [f'    assert_equal(model.predict(Q), {ints(model.predict(query))})',
                          f'    assert_equal(model.score(X, {ints(truth + offset)}), {literal(model.score(X, truth + offset))})']
            invalid = np.where(y == -1, -1, y[0])
            try:
                model.fit(X, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("sklearn accepted one labeled class for logistic regression")
            lines += [
                '    assert_true(not base.estimator.__sklearn_is_fitted__())',
                '    var cloned = model.__sklearn_clone__()',
                '    assert_true(not cloned.__sklearn_is_fitted__())',
                '    assert_equal(cloned.estimator.value().estimator.C, 1.7)',
                '    assert_equal(cloned.threshold, .95)',
                '    with assert_raises():',
                f'        _ = model.fit(X, {ints(invalid)})',
                '    with assert_raises():',
                '        _ = model.predict(Q)',
            ]
            yield name, '\n'.join(lines)


def svc_reference():
    from sklearn.svm import SVC

    wheel = Path(sklearn.__file__).parent
    paths = [REFERENCE / 'sklearn/svm' / name
             for name in ('_base.py', '_classes.py', '_libsvm.pyx', '_libsvm.pxi')]
    for folder in ('libsvm', 'newrand'):
        paths.extend(sorted((REFERENCE / 'sklearn/svm/src' / folder).glob('*')))
    hashes = {}
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(REFERENCE)
        committed = subprocess.check_output(
            ['git', '-C', str(REFERENCE), 'show', f'{REVISION}:{relative}'])
        installed = wheel / relative.relative_to('sklearn')
        if committed != path.read_bytes() or committed != installed.read_bytes():
            raise RuntimeError(f'SVC oracle source mismatch: {relative}')
        hashes[str(relative)] = hashlib.sha256(committed).hexdigest()
    return SVC, hashes


def calibrated_svc_cases(training, calibration, svc):
    rng = np.random.default_rng(CALIBRATED_SVC_SEED)
    for n_classes in (2, 3):
        centers = [[-3., -1.], [3., -1.], [0., 4.]][:n_classes]
        X = np.concatenate([rng.normal(center, .7, (10, 2)) for center in centers])
        truth = np.repeat([35, -7, 12][:n_classes], 10)
        y = truth.copy()
        y.reshape(n_classes, 10)[:, 6:] = -1
        query = X[1::2] + .25
        name = f'calibrated_svc_{n_classes}'
        lines = [f'@no_inline\ndef {name}() raises:'] + array('X', X) + array('Q', query)
        lines += [
            '    var svc = SVC[Int](gamma=.5, random_state=42, tol=1e-7)',
            '    var calibrated = CalibratedClassifierCV(ClassifierCalibrationAdapter(svc^, svc_classes, "SVC"), cv=5)',
            '    var base = CalibratedClassifierAdapter(calibrated^)',
            '    var model = SelfTrainingClassifier[type_of(base)](base.copy(), threshold=.60)',
            '    with assert_raises():',
            '        _ = base.classes()',
            '    with assert_raises():',
            '        _ = model.predict_proba(Q)',
        ]
        oracle_svc = svc(gamma=.5, random_state=42, tol=1e-7)
        model = training.SelfTrainingClassifier(
            calibration.CalibratedClassifierCV(oracle_svc, cv=5), threshold=.60)
        for threshold, offset in ((.60, 0), (.88, 100)):
            targets = np.where(y == -1, -1, y + offset)
            model.set_params(threshold=threshold).fit(X, targets)
            if not offset:
                assert (model.labeled_iter_ > 0).any()
            lines += [
                f'    _ = model.set_params(threshold={threshold}).fit(X, {ints(targets)})',
                f'    assert_equal(model.classes_, {ints(model.classes_)})',
                f'    assert_equal(model.labeled_iter_, {ints(model.labeled_iter_)})',
                f'    assert_equal(model.n_iter_, {model.n_iter_})',
                f'    assert_equal(model.termination_condition_, "{model.termination_condition_}")',
            ]
            for i, label in enumerate(model.transduction_):
                lines.append(f'    assert_equal(model.transduction_[{i}].value(), {label})'
                             if label != -1 else f'    assert_true(not model.transduction_[{i}])')
            lines += check_array('model.predict_proba(Q)', model.predict_proba(query), CALIBRATED_SVC_TOLERANCE)
            lines += [
                f'    assert_equal(model.predict(Q), {ints(model.predict(query))})',
                f'    assert_equal(model.score(X, {ints(truth + offset)}), {literal(model.score(X, truth + offset))})',
            ]
        lines += [
            '    var cloned = model.__sklearn_clone__()',
            '    assert_true(not cloned.__sklearn_is_fitted__())',
            '    assert_true(not base.estimator.__sklearn_is_fitted__())',
            '    assert_equal(cloned.threshold, .88)',
            '    assert_equal(cloned.estimator.value().estimator.estimator.estimator().random_state, 42)',
            '    var overflowing_labels = List[Int](length=X.shape[0], fill=9007199254740993)',
            '    with assert_raises():',
            '        _ = model.fit(X, overflowing_labels)',
            '    with assert_raises():',
            '        _ = model.predict(Q)',
        ]
        yield name, '\n'.join(lines)


def direct_svc_cases(training, svc):
    rng = np.random.default_rng(DIRECT_SVC_SEED)
    for n_classes in (2, 3):
        centers = [[-3., -1.], [3., -1.], [0., 4.]][:n_classes]
        features = np.concatenate([rng.normal(center, .7, (12, 2)) for center in centers])
        truth = np.repeat([35, -7, 12][:n_classes], 12)
        y = truth.copy()
        y.reshape(n_classes, 12)[:, 8:] = -1
        for dtype in ('float64', 'float32'):
            X = features.astype(dtype)
            query = (features[1::3] + .25).astype(dtype)
            name = f'direct_svc_{n_classes}_{dtype}'
            lines = [f'@no_inline\ndef {name}() raises:']
            lines += array('X', X, dtype) + array('Q', query, dtype)
            lines += [
                '    var svc = SVC[Int](gamma=.5, probability=True, random_state=42, tol=1e-8)',
                f'    var base = SVCAdapter[Int, DType.{dtype}](svc^)',
                f'    var model = SelfTrainingClassifier[type_of(base), DType.{dtype}](base.copy())',
                '    with assert_raises():',
                '        _ = base.classes()',
                '    with assert_raises():',
                '        _ = model.predict_proba(Q)',
            ]
            model = training.SelfTrainingClassifier(
                svc(gamma=.5, probability=True, random_state=42, tol=1e-8))
            for threshold, offset in ((.65, 0), (.98, 100)):
                targets = np.where(y == -1, -1, y + offset)
                model.set_params(threshold=threshold).fit(X, targets)
                assert (model.labeled_iter_ > 0).any() == (offset == 0)
                lines += [
                    f'    _ = model.set_params(threshold={threshold}).fit(X, {ints(targets)})',
                    f'    assert_equal(model.classes_, {ints(model.classes_)})',
                    f'    assert_equal(model.labeled_iter_, {ints(model.labeled_iter_)})',
                    f'    assert_equal(model.n_iter_, {model.n_iter_})',
                    f'    assert_equal(model.termination_condition_, "{model.termination_condition_}")',
                    f'    assert_equal(model.predict(Q), {ints(model.predict(query))})',
                    f'    assert_equal(model.score(X, {ints(truth + offset)}), {literal(model.score(X, truth + offset))})',
                ]
                for i, label in enumerate(model.transduction_):
                    lines.append(f'    assert_equal(model.transduction_[{i}].value(), {label})'
                                 if label != -1 else f'    assert_true(not model.transduction_[{i}])')
                lines += check_array('model.predict_proba(Q)', model.predict_proba(query), DIRECT_SVC_TOLERANCE)
                lines += check_array('model.predict_log_proba(Q)', model.predict_log_proba(query), DIRECT_SVC_TOLERANCE)
                decision = model.decision_function(query)
                lines.append(f'    assert_equal(model.decision_function(Q).ndim, {decision.ndim})')
                decision = decision.reshape(len(query), -1)
                lines += check_array(
                    f'model.decision_function(Q).reshape(Shape({len(query)}, {decision.shape[1]}))',
                    decision, DIRECT_SVC_TOLERANCE)
            lines += [
                '    var cloned = model.__sklearn_clone__()',
                '    assert_true(not cloned.__sklearn_is_fitted__())',
                '    assert_true(not base.estimator.__sklearn_is_fitted__())',
                '    assert_true(not cloned.estimator.value().estimator.__sklearn_is_fitted__())',
                '    assert_true(cloned.estimator.value().estimator.probability.enabled())',
                '    assert_equal(cloned.estimator.value().estimator.random_state, 42)',
                '    assert_equal(cloned.threshold, .98)',
            ]
            yield name, '\n'.join(lines)


def main():
    assert subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    OUT.mkdir(parents=True, exist_ok=True)
    graph = reference('reference_graph', 'sklearn/semi_supervised/_label_propagation.py')
    training = reference('reference_training', 'sklearn/semi_supervised/_self_training.py')
    nb = reference('reference_nb', 'sklearn/naive_bayes.py')
    metrics = reference('reference_metrics', 'sklearn/metrics/_classification.py')
    # Reuse the logistic owner's pinned source loader and support-wheel checks.
    logistic_oracle = runpy.run_path(str(ROOT.parent / 'linear_model/algorithms/logistic/tests/parity.py'))
    logistic = logistic_oracle['oracle']()
    calibration = reference('reference_calibration', 'sklearn/calibration.py')
    svc, svc_hashes = svc_reference()
    rng = np.random.RandomState(731)
    source = ['''from std.testing import assert_equal, assert_true, assert_raises
from std.math import abs, isnan
from numojo import NDArray, Shape, zeros
from moxie.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier, GaussianNBAdapter, LogisticRegressionAdapter, CalibratedClassifierAdapter, SVCAdapter
from moxie.calibration import CalibratedClassifierCV, ClassifierCalibrationAdapter
from moxie.svm import SVC
from moxie.linear_model import LogisticRegression
from moxie.semi_supervised._contracts import DenseAffinity, ZeroMassPolicy
from moxie.semi_supervised._probability import RowNormalizer
from moxie._nb_gaussian import GaussianNB
from moxie._naive_bayes_dense import IntListDomain

def svc_classes(model: SVC[Int]) raises -> NDArray[DType.float64]:
    var classes = NDArray[DType.float64](Shape(len(model.classes_)))
    for i in range(len(model.classes_)):
        classes.unsafe_set(i, Float64(model.classes_[i]))
    return classes^

@no_inline
def check_matrix(actual: NDArray[DType.float64], expected: List[Float64], rows: Int, columns: Int, tolerance: Float64) raises:
    assert_equal(actual.shape[0], rows)
    assert_equal(actual.shape[1], columns)
    for i in range(len(expected)):
        if isnan(expected[i]):
            assert_true(isnan(actual.unsafe_get(i)))
        else:
            assert_true(abs(actual.unsafe_get(i) - expected[i]) <= tolerance)

@no_inline
def check_labels(actual: List[Int], expected: List[Int], probabilities: List[Float64], classes: List[Int], tolerance: Float64) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        if actual[i] == expected[i]:
            continue
        var best = Float64(-1)
        var selected = Float64(-1)
        for j in range(len(classes)):
            best = max(best, probabilities[i * len(classes) + j])
            if classes[j] == actual[i]:
                selected = probabilities[i * len(classes) + j]
        assert_true(selected >= 0 and best - selected <= tolerance)

@fieldwise_init
struct FixtureAffinity(DenseAffinity, Copyable, Movable):
    comptime Input = NDArray[DType.float64]
    var matrix: NDArray[DType.float64]
    def affinity(self, train: Self.Input, query: Self.Input) raises -> NDArray[DType.float64]:
        return self.matrix.copy()
''']
    cases = []
    fixtures = [
        (np.array([[0.], [1.], [.9], [.1]]), np.array([9, 2, -1, -1])),
        (np.array([[0., 0.]]), np.array([4])),
        (np.array([[0.], [1.], [2.]]), np.array([4, -1, -1])),
        (np.array([[0.], [0.], [100.], [100.]]), np.array([0, 1, -1, -1])),
    ]
    for n, d in [(7, 3), (13, 5), (21, 2)]:
        X = rng.normal(size=(n, d))
        y = np.full(n, -1); y[:3] = [-7, 5, 12]
        fixtures.append((X, y))
    for dtype in ('float64', 'float32'):
        for name in ('LabelPropagation', 'LabelSpreading'):
            for fixture, (X, y) in enumerate(fixtures):
                for gamma, limit, tol in [(0., 30, 1e-3), (0.7, 1000, 1e-8), (20., 1, 0.)]:
                    X = X.astype(dtype)
                    estimator = getattr(graph, name)(gamma=gamma, max_iter=limit, tol=tol).fit(X, y)
                    Q = (X[:min(3, len(X))] + .07).astype(dtype)
                    identifier = f'case_{len(cases)}'
                    lines = [f'@no_inline\ndef {identifier}() raises:'] + array('X', X, dtype) + array('Q', Q, dtype)
                    lines += [f'    var model = {name}[DType.{dtype}](gamma={gamma}, max_iter={limit}, tol={tol})', f'    model.fit(X, {ints(y)})', f'    assert_equal(model.classes_, {ints(estimator.classes_)})', f'    assert_equal(model.n_iter_, {estimator.n_iter_})']
                    tolerance = '3e-5' if dtype == 'float32' else '2e-9'
                    lines += check_array('model.label_distributions_', estimator.label_distributions_, tolerance)
                    lines += check_labels('model.transduction_', estimator.transduction_, estimator.label_distributions_, estimator.classes_, '3e-7' if dtype == 'float32' else '1e-14')
                    lines += ['    var probabilities = model.predict_proba(Q)'] + check_array('probabilities', estimator.predict_proba(Q), tolerance)
                    lines += check_labels('model.predict(Q)', estimator.predict(Q), estimator.predict_proba(Q), estimator.classes_, '3e-7' if dtype == 'float32' else '1e-14')
                    source.append('\n'.join(lines)); cases.append(identifier)
    for dtype in ('float64', 'float32'):
        for name in ('LabelPropagation', 'LabelSpreading'):
            X = np.array([[0.], [1.], [.9], [.1]], dtype=dtype)
            y = [7, 3, -1, -1]
            target = [7, 3, 3, 3]
            estimator = getattr(graph, name)(gamma=2).fit(X, y)
            prediction = estimator.predict(X)
            identifier = f'case_{len(cases)}'
            lines = [f'@no_inline\ndef {identifier}() raises:'] + array('X', X, dtype)
            lines += [f'    var model = {name}[DType.{dtype}](gamma=2)', f'    model.fit(X, {ints(y)})']
            for weights in ([1., 2., 3., 4.], [1., 2., 3., -1.], [0., 0., 0., 1.],
                            [1., -1., 0., 0.], [1.],
                            [1., 1., 1., np.nan], [1., 1., 1., np.inf], [1., 1., 1., -np.inf]):
                entries = ', '.join(literal(w) for w in weights)
                call = f'model.score(X, {ints(target)}, List[Float64]([{entries}]))'
                try:
                    expected = metrics.accuracy_score(target, prediction, sample_weight=weights)
                except (ValueError, ZeroDivisionError):
                    lines += ['    with assert_raises():', f'        _ = {call}']
                else:
                    lines += [f'    assert_equal({call}, {literal(expected)})']
            source.append('\n'.join(lines)); cases.append(identifier)
    for matrix in [np.array([[1., 1., 0.], [2., 1., 1.], [0., 1., 3.]]), np.diag([1., 2., 3.]), np.array([[1., 2., 0.], [0., 1., 0.], [0., 0., 1.]])]:
        X = np.arange(3.).reshape(-1, 1); y = [0, 1, -1]
        for name in ('LabelPropagation', 'LabelSpreading'):
            estimator = getattr(graph, name)(kernel=lambda a,b: matrix.copy(), max_iter=100, tol=1e-8).fit(X, y)
            identifier = f'case_{len(cases)}'
            lines = [f'@no_inline\ndef {identifier}() raises:'] + array('X', X) + array('matrix', matrix)
            lines += [f'    var model = {name}[DType.float64, FixtureAffinity]("callable", affinity=FixtureAffinity(matrix.copy()), max_iter=100, tol=1e-8)', f'    model.fit(X, {ints(y)})', f'    assert_equal(model.n_iter_, {estimator.n_iter_})', f'    assert_equal(model.classes_, {ints(estimator.classes_)})']
            lines += check_array('model.label_distributions_', estimator.label_distributions_, '2e-9')
            lines += check_labels('model.transduction_', estimator.transduction_, estimator.label_distributions_, estimator.classes_, '1e-14')
            lines += ['    var probabilities = model.predict_proba(X)'] + check_array('probabilities', estimator.predict_proba(X), '2e-9')
            source.append('\n'.join(lines)); cases.append(identifier)
    for dtype in ('float64', 'float32'):
        X = np.concatenate([rng.normal(-1, .8, (12, 2)), rng.normal(1, .8, (12, 2))]).astype(dtype)
        y = np.array([0]*4+[-1]*8+[1]*4+[-1]*8)
        for criterion, threshold, k_best, limit in [('threshold', .75, 10, 10), ('threshold', .99999, 10, 10), ('threshold', .75, 10, 0), ('threshold', .75, 10, 1), ('k_best', .75, 3, None), ('k_best', .75, 100, 2)]:
            smoothing = 0.05 if dtype == 'float32' and criterion == 'k_best' else 1e-9
            estimator = training.SelfTrainingClassifier(nb.GaussianNB(var_smoothing=smoothing), criterion=criterion, threshold=threshold, k_best=k_best, max_iter=limit).fit(X, y)
            identifier = f'case_{len(cases)}'
            lines = [f'@no_inline\ndef {identifier}() raises:'] + array('X', X, dtype)
            native_limit = 'None' if limit is None else str(limit)
            lines += [f'    var base = GaussianNBAdapter[DType.{dtype}](GaussianNB[IntListDomain, DType.{dtype}](var_smoothing={smoothing}))', f'    var model = SelfTrainingClassifier[type_of(base), DType.{dtype}](base.copy(), criterion="{criterion}", threshold={threshold}, k_best={k_best}, max_iter={native_limit})', f'    model.fit(X, {ints(y)})', f'    assert_equal(model.n_iter_, {estimator.n_iter_})', f'    assert_equal(model.termination_condition_, "{estimator.termination_condition_}")', f'    assert_equal(model.labeled_iter_, {ints(estimator.labeled_iter_)})', f'    assert_equal(model.classes_, {ints(estimator.classes_)})']
            for i,label in enumerate(estimator.transduction_):
                lines.append(f'    assert_equal(model.transduction_[{i}].value(), {label})' if label != -1 else f'    assert_true(not model.transduction_[{i}])')
            lines += ['    var probabilities = model.predict_proba(X)'] + check_array('probabilities', estimator.predict_proba(X), '5e-5' if dtype == 'float32' else '2e-9')
            lines += [f'    assert_equal(model.predict(X), {ints(estimator.predict(X))})']
            # score delegates to estimator_.score (GaussianNB accuracy); exact count ratio.
            truth = np.where(np.arange(len(y)) < 12, 0, 1)
            lines.append(f'    assert_equal(model.score(X, {ints(truth)}), {literal(estimator.score(X, truth))})')
            source.append('\n'.join(lines));cases.append(identifier)
    for name, code in [
        *logistic_cases(training, logistic),
        *calibrated_svc_cases(training, calibration, svc),
        *direct_svc_cases(training, svc),
    ]:
        source.append(code)
        cases.append(name)
    source.append('def main() raises:\n' + '\n'.join(f'    {case}()\n    print("PASS {case}")' for case in cases))
    path = OUT/'oracle_cases.mojo';path.write_text('\n\n'.join(source)+'\n')
    assemble(ROOT)
    compile_file(ROOT, path, 'parity_estimators')
    result = run(ROOT/'builds/parity_estimators', check=False)
    (OUT/'native.log').write_text(result.stdout + result.stderr)
    manifest = {'reference_revision': REVISION, 'support_sklearn_version': sklearn.__version__, 'python': platform.python_version(), 'numpy': np.__version__, 'seed':731, 'cases':len(cases), 'tolerances':{'float64':2e-9,'float32_graph':3e-5,'float32_gaussian':5e-5,'near_tie_float64':1e-14,'near_tie_float32':3e-7}, 'threads':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}, 'k_best':'input-order tie breaking; Float32 k-best oracle fixtures use var_smoothing=.05 to avoid saturated boundary ties', 'flags':['-O3','--fp-mode=contract=off'], 'source_hashes':{str(p.relative_to(REFERENCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [REFERENCE/'sklearn/semi_supervised/_label_propagation.py',REFERENCE/'sklearn/semi_supervised/_self_training.py',REFERENCE/'sklearn/naive_bayes.py',REFERENCE/'sklearn/metrics/_classification.py']}}
    logistic_paths = [logistic_oracle[name] for name in ('LOSS', 'LOGISTIC', 'NEWTON')]
    manifest['source_hashes'].update(logistic_oracle['verify_reference'](logistic_paths))
    calibration_path = 'sklearn/calibration.py'
    manifest['source_hashes'][calibration_path] = hashlib.sha256(
        (REFERENCE / calibration_path).read_bytes()).hexdigest()
    manifest['source_hashes'].update(svc_hashes)
    manifest['calibrated_svc_seed'] = CALIBRATED_SVC_SEED
    manifest['tolerances']['calibrated_svc'] = CALIBRATED_SVC_TOLERANCE
    manifest['direct_svc_seed'] = DIRECT_SVC_SEED
    manifest['tolerances']['direct_svc'] = DIRECT_SVC_TOLERANCE
    manifest['logistic_seed'] = LOGISTIC_SEED
    manifest['tolerances']['logistic'] = LOGISTIC_TOLERANCES
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if result.returncode:
        print(result.stdout[-3000:],result.stderr[-4000:]);raise RuntimeError('native parity failed; see builds/parity/native.log')
    print(f'PASS {len(cases)} sklearn oracle cases; evidence in builds/parity')

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            main()
        except subprocess.CalledProcessError as error:
            (OUT / "compiler.log").write_text((error.stdout or "") + (error.stderr or ""))
            print((error.stderr or "")[:5000])
            raise
