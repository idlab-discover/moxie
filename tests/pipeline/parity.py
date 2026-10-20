"""Compare the native composition with the pinned local pipeline.py and dummy.py.

The locked sklearn wheel supplies supporting utilities, decorators and base
classes; the Pipeline and DummyRegressor algorithm bodies execute local source.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/pipeline.py', 'sklearn/dummy.py', 'sklearn/utils/metaestimators.py',
         'sklearn/base.py', 'sklearn/tests/test_pipeline.py', 'doc/modules/compose.rst']


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


assert subprocess.check_output(['git', '-C', SOURCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
assert not subprocess.check_output(['git', '-C', SOURCE, 'status', '--porcelain', '--', *FILES], text=True)
pipeline_module = load('sklearn._pipeline_parity', 'sklearn/pipeline.py')
Pipeline = pipeline_module.Pipeline
FeatureUnion = pipeline_module.FeatureUnion
dummy_module = load('sklearn._dummy_parity', 'sklearn/dummy.py')
DummyRegressor = dummy_module.DummyRegressor
DummyClassifier = dummy_module.DummyClassifier
for method, relative in [(Pipeline.fit, 'sklearn/pipeline.py'), (Pipeline.predict, 'sklearn/pipeline.py'),
                         (Pipeline.get_feature_names_out, 'sklearn/pipeline.py'),
                         (Pipeline.score, 'sklearn/pipeline.py'),
                         (Pipeline.predict_proba, 'sklearn/pipeline.py'),
                         (Pipeline.predict_log_proba, 'sklearn/pipeline.py'),
                         (Pipeline.decision_function, 'sklearn/pipeline.py'),
                         (Pipeline.score_samples, 'sklearn/pipeline.py'),
                         (DummyRegressor.fit, 'sklearn/dummy.py')]:
    while hasattr(method, '__wrapped__'):
        method = method.__wrapped__
    assert method.__code__.co_filename == str(SOURCE / relative)


class Shift(BaseEstimator):
    def __init__(self, offset=2):
        self.offset = offset
        self.fitted = False

    def fit(self, X, y):
        self.fitted = False
        if X < 0:
            raise ValueError('negative training input')
        self.seen, self.seen_y, self.fitted = X, y, True
        return self

    def transform(self, X):
        if not self.fitted:
            raise NotFittedError('Shift')
        return X + self.offset

    def __sklearn_is_fitted__(self):
        return self.fitted


class Recorder(BaseEstimator):
    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        self.fitted = False
        if y < 0:
            raise ValueError('negative target')
        self.seen, self.seen_y, self.fitted = X, y, True
        return self

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError('Recorder')
        return X * 3 + self.seen

    def __sklearn_is_fitted__(self):
        return self.fitted


class Fused(BaseEstimator):
    def __init__(self):
        self.calls = 0

    def fit(self, X, y):
        raise AssertionError("fit must not run when fit_transform exists")

    def fit_transform(self, X, y=None):
        self.calls += 1
        return X + y + 100

    def transform(self, X):
        return X + 10


class Unsupervised(BaseEstimator):
    def fit(self, X, y=None):
        self.seen_y = y
        return self

    def transform(self, X):
        return X * 2


def pair(a, b):
    return Pipeline([('transformer', a), ('estimator', b)])


expected = {}
p = pair(Shift(), Recorder()).fit(5, 7)
expected['fallback'] = [p[-1].seen, p[-1].seen_y, p.predict(1)]
other = clone(p).set_params(transformer__offset=4).fit(5, 7)
expected['clone'] = [other.predict(1), p.predict(1)]
p.fit(8, 3)
try:
    p.fit(-1, 3)
except ValueError:
    pass
else:
    raise AssertionError('expected failed prefix fit')
expected['prefix_failure'] = [int(p.__sklearn_is_fitted__())]
try:
    p.fit(5, -1)
except ValueError:
    pass
else:
    raise AssertionError('expected failed final fit')
expected['final_failure'] = [int(p.__sklearn_is_fitted__()), int(p[0].fitted)]
p = pair(Fused(), Recorder()).fit(5, 7)
expected['fused'] = [p[0].calls, p[-1].seen, p.predict(1)]
p = pair(Unsupervised(), Recorder()).fit(5, 7)
expected['unsupervised'] = [p[0].seen_y, p[-1].seen]
p = pair(Shift(2), pair(Shift(4), Recorder())).fit(5, 7)
expected['nested'] = [p[-1][-1].seen, p.predict(1)]

native = subprocess.check_output([ROOT / 'builds/tests/pipeline/test_sequential'], text=True)
actual = {fields[0]: list(map(int, fields[1:])) for line in native.splitlines()
          if (fields := line.split())[0] in expected}
assert actual == expected, (actual, expected)

native = subprocess.check_output([ROOT / 'builds/tests/pipeline/test_real_pipeline'], text=True)
actual = [list(map(float, line.split()[1:])) for line in native.splitlines() if line.startswith('real ')]
expected_real = []
for flag, dtype in [(0, np.float32), (1, np.float64)]:
    X = np.arange(8, dtype=dtype).reshape(4, 2)
    y = np.arange(4, dtype=dtype) * 2
    p = pair(FunctionTransformer(), DummyRegressor()).fit(X, y)
    first = p.predict(X)[0]
    other = clone(p).set_params(estimator__strategy='median')
    y[0] = 100
    other.fit(X, y)
    median = other.predict(X)[0]
    assert p.predict(X)[0] == first
    p.fit(X, y)
    nested = pair(FunctionTransformer(), p).fit(X, y)
    expected_real.append([flag, first, median, nested.predict(X)[0]])
np.testing.assert_array_equal(actual, expected_real)  # exactly representable fixtures
# Names use explicit probes: both implementations must delegate, not infer names.
class NameStep(BaseEstimator):
    def __init__(self, prefix):
        self.prefix = prefix
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        return X

    def __sklearn_is_fitted__(self):
        return self.fitted

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if not self.fitted:
                raise NotFittedError('default names require fit')
            input_features = ['x0', 'x1']
        if 'invalid' in input_features:
            raise ValueError('invalid feature name')
        return [self.prefix + name for name in input_features]


p = pair(NameStep('a_'), NameStep('b_'))
try:
    p.get_feature_names_out()
except NotFittedError:
    pass
else:
    raise AssertionError('expected unfitted default names')
expected_names = {'names_explicit': p.get_feature_names_out(['height', 'width'])}
p.fit(1, 1)
expected_names['names_default'] = p.get_feature_names_out()
expected_names['names_empty'] = p.get_feature_names_out([])
expected_names['names_duplicates'] = p.get_feature_names_out(['same', 'same'])
fresh = clone(p).set_params(transformer__prefix='new_').fit(2, 2)
expected_names['names_clone'] = fresh.get_feature_names_out()
p.fit(2, 2)
expected_names['names_refit'] = p.get_feature_names_out()
expected_names['names_nested'] = pair(NameStep('outer_'), p).get_feature_names_out(['height', 'width'])
for bad, features, exception, message in [
    (Pipeline([('missing_first', Shift()), ('last', NameStep('b_'))]), None,
     AttributeError, 'Estimator missing_first does not provide get_feature_names_out'),
    (pair(NameStep('a_'), Recorder()), None, NotFittedError, 'default names require fit'),
    (Pipeline([('first', NameStep('a_')), ('missing_last', Recorder())]), ['height'],
     AttributeError, 'Estimator missing_last does not provide get_feature_names_out'),
    (pair(NameStep('a_'), NameStep('b_')), ['invalid'], ValueError, 'invalid feature name'),
    (pair(NameStep('in'), NameStep('b_')), ['valid'], ValueError, 'invalid feature name'),
]:
    try:
        bad.get_feature_names_out(features)
    except exception as error:
        assert message in str(error), error
    else:
        raise AssertionError('expected feature-name failure')
native_names = subprocess.check_output([ROOT / 'builds/tests/pipeline/test_pipeline_names'], text=True)
actual_names = {fields[0]: fields[1:] for line in native_names.splitlines()
                if (fields := line.split())[0].startswith('names_')}
assert actual_names == expected_names, (actual_names, expected_names)

# A fitted prefix owns n_features_in_, independently of the final estimator.
expected_features = []
for flag, dtype in [(0, np.float32), (1, np.float64)]:
    X = np.zeros((4, 3), dtype=dtype)
    wider = np.zeros((4, 5), dtype=dtype)
    y = np.zeros(4, dtype=dtype)
    prefix = FunctionTransformer(lambda X: X[:, :1], validate=True)
    p = pair(prefix, DummyRegressor())
    assert not hasattr(p, 'n_features_in_')
    p.fit(X, y)
    expected_features.append(['features', flag, p.n_features_in_, prefix.transform(X).shape[1]])
    fresh = clone(p)
    assert not hasattr(fresh, 'n_features_in_')
    fresh.fit(wider, y)
    assert p.n_features_in_ == 3
    p.fit(wider, y)
    p.steps[-1] = ('estimator', DummyRegressor())
    assert not p.__sklearn_is_fitted__()
    expected_features.append(['features_refit', flag, fresh.n_features_in_, p.n_features_in_])
    p.steps[0] = ('transformer', clone(prefix))
    assert not hasattr(p, 'n_features_in_')
    missing = pair(NameStep(''), DummyRegressor()).fit(X, y)
    assert not hasattr(missing, 'n_features_in_')
    nested = pair(FeatureUnion([('a', 'passthrough'), ('b', 'passthrough')]),
                  pair(FunctionTransformer(lambda X: X[:, :1], validate=True), DummyRegressor()))
    nested.fit(X, y)
    expected_features.append(['features_union', flag, nested.n_features_in_, nested[-1].n_features_in_])
native_features = subprocess.check_output([ROOT / 'builds/tests/pipeline/test_pipeline_features'], text=True)
actual_features = [[fields[0], *map(int, fields[1:])] for line in native_features.splitlines()
                   if (fields := line.split())[0].startswith('features')]
assert actual_features == expected_features, (actual_features, expected_features)

class ScoreRecorder(Recorder):
    def predict(self, X):
        raise AssertionError('score must not call predict')

    def score(self, X, y, sample_weight=None):
        if not self.fitted:
            raise NotFittedError('score recorder')
        if y < 0 or (sample_weight is not None and sample_weight < 0):
            raise ValueError('rejected score arguments')
        return float(X * 100 + y * 10 + self.seen +
                     (sample_weight if sample_weight is not None else -1))


p = pair(Shift(), ScoreRecorder()).fit(5, 7)
expected_scores = {'score_scalar': [p.score(1, 7), p.score(1, 7, sample_weight=3)]}
assert p.score(1, 7, sample_weight=None) == p.score(1, 7)
fresh = clone(p).set_params(transformer__offset=4).fit(5, 7)
p.fit(8, 7)
expected_scores['score_refit'] = [fresh.score(1, 7), p.score(1, 7)]
fused = pair(Fused(), ScoreRecorder()).fit(5, 7)
expected_scores['score_fused'] = [fused.score(1, 7)]
nested = pair(Shift(2), pair(Shift(4), ScoreRecorder())).fit(5, 7)
expected_scores['score_nested'] = [nested.score(1, 7), nested.score(1, 7, sample_weight=5)]
assert not hasattr(pair(Shift(), Recorder()), 'score')
for estimator, target, weight, error in [
    (clone(p), 7, None, NotFittedError),
    (p, -1, None, ValueError),
    (p, 7, -1, ValueError),
]:
    try:
        estimator.score(1, target, sample_weight=weight)
    except error:
        pass
    else:
        raise AssertionError('expected score error')
expected_dense_scores = []
for flag, dtype in [(0, np.float32), (1, np.float64)]:
    X = np.zeros((4, 2), dtype=dtype)
    y = np.arange(4, dtype=dtype) * 2
    target = np.array([0, 0, 2, 2], dtype=dtype)
    weights = np.array([1, 1, 3, 3], dtype=dtype)
    p = pair(FunctionTransformer(), DummyRegressor()).fit(X, y)
    expected_dense_scores.append([flag, p.score(X, y), p.score(X, target),
                                  p.score(X, target, sample_weight=weights)])
native_scores = subprocess.check_output([ROOT / 'builds/tests/pipeline/test_pipeline_score'], text=True)
actual_scores = {fields[0]: list(map(float, fields[1:])) for line in native_scores.splitlines()
                 if (fields := line.split())[0] in expected_scores}
actual_dense_scores = [list(map(float, line.split()[1:])) for line in native_scores.splitlines()
                       if line.startswith('score_dense ')]
assert actual_scores == expected_scores, (actual_scores, expected_scores)
np.testing.assert_array_equal(actual_dense_scores, expected_dense_scores)

# Scalar tokens are dispatch probes, not probability estimates.
class ProbabilityProbe(Recorder):
    def predict(self, X):
        raise AssertionError('must not call predict')

    def predict_proba(self, X):
        if X < 0:
            raise ValueError('probability probe input')
        return X * 100 + self.seen

    def predict_log_proba(self, X):
        if X < 0:
            raise ValueError('log probability probe input')
        return -X * 10 - self.seen


def probability_tokens(p):
    return [p.predict_proba(1), p.predict_log_proba(1)]


p = pair(Shift(), ProbabilityProbe()).fit(5, 7)
expected_probability = {'prob_scalar': probability_tokens(p)}
fresh = clone(p).set_params(transformer__offset=4).fit(5, 7)
p.fit(8, 7)
expected_probability['prob_refit'] = probability_tokens(fresh) + probability_tokens(p)
expected_probability['prob_fused'] = probability_tokens(pair(Fused(), ProbabilityProbe()).fit(5, 7))
expected_probability['prob_nested'] = probability_tokens(pair(Shift(2), pair(Shift(4), ProbabilityProbe())).fit(5, 7))
for method in ['predict_proba', 'predict_log_proba']:
    assert not hasattr(pair(Shift(), Recorder()), method)
    try:
        getattr(clone(p), method)(1)
    except NotFittedError:
        pass
    else:
        raise AssertionError('expected unfitted probability error')
    try:
        getattr(p, method)(-10)
    except ValueError:
        pass
    else:
        raise AssertionError('expected child probability error')
expected_dense_probability = []
expected_hard_probability = []
for flag, dtype in [(0, np.float32), (1, np.float64)]:
    X = np.zeros((4, 2), dtype=dtype)
    y = [9, 2, 9, 9]
    p = pair(FunctionTransformer(), DummyClassifier()).fit(X, y)
    probs, logs = p.predict_proba(X), p.predict_log_proba(X)
    np.testing.assert_array_equal(p[-1].classes_, [2, 9])
    assert probs.shape == logs.shape == (4, 2)
    expected_dense_probability.append([flag, *probs[0], *logs[0]])
    fresh = clone(p).set_params(estimator__strategy='most_frequent').fit(X, y)
    with np.errstate(divide='ignore'):
        expected_hard_probability.append([flag, *fresh.predict_proba(X)[0], *fresh.predict_log_proba(X)[0]])
    p.fit(X, [7] * 4)
    np.testing.assert_array_equal(p.predict_proba(X), np.ones((4, 1)))
    np.testing.assert_array_equal(p.predict_log_proba(X), np.zeros((4, 1)))
native_probability = subprocess.check_output([ROOT / 'builds/tests/pipeline/test_pipeline_probability'], text=True)
actual_probability = {fields[0]: list(map(int, fields[1:])) for line in native_probability.splitlines()
                      if (fields := line.split())[0] in expected_probability}
assert actual_probability == expected_probability, (actual_probability, expected_probability)
dense_probability = {}
for key, expected_rows in [('prob_dense', expected_dense_probability), ('prob_hard', expected_hard_probability)]:
    rows = [list(map(float, line.split()[1:])) for line in native_probability.splitlines()
            if line.startswith(key + ' ')]
    assert len(rows) == 2
    np.testing.assert_array_equal(np.asarray(rows)[:, :3], np.asarray(expected_rows)[:, :3])
    # Classifier outputs Float64 for both input dtypes; use its precision.
    np.testing.assert_allclose(np.asarray(rows)[:, 3:], np.asarray(expected_rows)[:, 3:], rtol=1e-12, atol=1e-12)
    dense_probability[key] = rows

class DecisionProbe(Recorder):
    def predict(self, X):
        raise AssertionError('decision_function must not call predict')

    def decision_function(self, X):
        if X < 0:
            raise ValueError('decision probe input')
        return X * 10 - self.seen


p = pair(Shift(), DecisionProbe()).fit(5, 7)
expected_decision = {'decision_scalar': [p.decision_function(1)]}
fresh = clone(p).set_params(transformer__offset=4).fit(5, 7)
p.fit(8, 7)
expected_decision['decision_refit'] = [fresh.decision_function(1), p.decision_function(1)]
expected_decision['decision_fused'] = [pair(Fused(), DecisionProbe()).fit(5, 7).decision_function(1)]
expected_decision['decision_nested'] = [pair(Shift(2), pair(Shift(4), DecisionProbe())).fit(5, 7).decision_function(1)]
assert not hasattr(pair(Shift(), Recorder()), 'decision_function')
for model, value, exception in [(clone(p), 1, NotFittedError), (p, -10, ValueError)]:
    try:
        model.decision_function(value)
    except exception:
        pass
    else:
        raise AssertionError('expected decision failure')


class DenseDecision(BaseEstimator):
    def __init__(self, multiclass=False):
        self.multiclass = multiclass
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def __sklearn_is_fitted__(self):
        return self.fitted

    def predict(self, X):
        raise AssertionError('decision_function must not call predict')

    def decision_function(self, X):
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError('decision fixture requires three columns')
        return X.copy() if self.multiclass else X[:, 0].copy()


expected_dense_decision = []
for flag, dtype in [(0, np.float32), (1, np.float64)]:
    for multiclass in [False, True]:
        X = np.arange(9, dtype=dtype).reshape(3, 3) - 3
        p = pair(FunctionTransformer(), DenseDecision(multiclass)).fit(X, np.zeros(3))
        output = p.decision_function(X)
        assert output.dtype == dtype
        expected_dense_decision.append([flag, int(multiclass), output.ndim, *output.ravel()])
        assert p.decision_function(np.zeros((0, 3), dtype=dtype)).shape[0] == 0
        try:
            p.decision_function(np.zeros((3, 2), dtype=dtype))
        except ValueError:
            pass
        else:
            raise AssertionError('expected decision shape error')
native_decision = subprocess.check_output([ROOT / 'builds/tests/pipeline/test_pipeline_decision'], text=True)
actual_decision = {fields[0]: list(map(int, fields[1:])) for line in native_decision.splitlines()
                   if (fields := line.split())[0] in expected_decision}
actual_dense_decision = [list(map(float, line.split()[1:])) for line in native_decision.splitlines()
                         if line.startswith('decision_dense ')]
assert actual_decision == expected_decision, (actual_decision, expected_decision)
assert actual_dense_decision == expected_dense_decision, (actual_dense_decision, expected_dense_decision)

class SampleProbe(Recorder):
    def predict(self, X):
        raise AssertionError('score_samples must not call predict')

    def score_samples(self, X):
        if X < 0:
            raise ValueError('sample probe input')
        return self.seen - X * 10


p = pair(Shift(), SampleProbe()).fit(5, 7)
expected_samples = {'samples_scalar': [p.score_samples(1)]}
fresh = clone(p).set_params(transformer__offset=4).fit(5, 7)
p.fit(8, 7)
expected_samples['samples_refit'] = [fresh.score_samples(1), p.score_samples(1)]
expected_samples['samples_fused'] = [pair(Fused(), SampleProbe()).fit(5, 7).score_samples(1)]
expected_samples['samples_nested'] = [pair(Shift(2), pair(Shift(4), SampleProbe())).fit(5, 7).score_samples(1)]
assert not hasattr(pair(Shift(), Recorder()), 'score_samples')
for model, value, exception in [(clone(p), 1, NotFittedError), (p, -10, ValueError)]:
    try:
        model.score_samples(value)
    except exception:
        pass
    else:
        raise AssertionError('expected sample-score failure')


class DenseSampleProbe(DenseDecision):
    def score_samples(self, X):
        return super().decision_function(X)


expected_dense_samples = []
for flag, dtype in [(0, np.float32), (1, np.float64)]:
    X = np.arange(9, dtype=dtype).reshape(3, 3) - 3
    p = pair(FunctionTransformer(), DenseSampleProbe()).fit(X, np.zeros(3))
    output = p.score_samples(X)
    assert output.shape == (3,) and output.dtype == dtype
    expected_dense_samples.append([flag, *output])
    assert p.score_samples(np.zeros((0, 3), dtype=dtype)).shape == (0,)
    try:
        p.score_samples(np.zeros((3, 2), dtype=dtype))
    except ValueError:
        pass
    else:
        raise AssertionError('expected sample-score shape error')
native_samples = subprocess.check_output([ROOT / 'builds/tests/pipeline/test_pipeline_samples'], text=True)
actual_samples = {fields[0]: list(map(int, fields[1:])) for line in native_samples.splitlines()
                  if (fields := line.split())[0] in expected_samples}
actual_dense_samples = [list(map(float, line.split()[1:])) for line in native_samples.splitlines()
                        if line.startswith('samples_dense ')]
assert actual_samples == expected_samples, (actual_samples, expected_samples)
assert actual_dense_samples == expected_dense_samples, (actual_dense_samples, expected_dense_samples)

# Unsupervised prefix whose target type differs from the final estimator's: sklearn
# passes y to StandardScaler, which ignores it; Moxie passes y=None (same result).
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def prefix_data():
    """Must match `data()` in test_pipeline_prefix_target.mojo."""
    i = np.arange(12)
    label = (i % 2).astype(np.float64)
    X = np.column_stack([i * 10.0 + label * 25.0, ((i * 7) % 5) * 0.01 - label * 0.02,
                         1000.0 + (i * 3) % 4])
    return X, label


X_prefix, y_prefix = prefix_data()
reference_prefix = Pipeline([('standardscaler', StandardScaler()), ('svc', SVC())]).fit(X_prefix, y_prefix)
native_prefix = subprocess.check_output([ROOT / 'builds/tests/pipeline/test_pipeline_prefix_target'], text=True)
fields = [line.split()[1:] for line in native_prefix.splitlines() if line.startswith('prefix_svc ')][0]
prefix_accuracy, prefix_values = float(fields[0]), np.array(fields[1:], dtype=float).reshape(12, 2)
assert prefix_accuracy == reference_prefix.score(X_prefix, y_prefix), prefix_accuracy
assert np.array_equal(prefix_values[:, 0], reference_prefix.predict(X_prefix))
# Same SMO/libsvm working-set algorithm; decision values agree far below tol=1e-3.
prefix_error = float(np.max(np.abs(prefix_values[:, 1] - reference_prefix.decision_function(X_prefix))))
assert prefix_error <= 1e-6, prefix_error

report = {'reference': REVISION, 'support_sklearn': sklearn.__version__,
          'source_sha256': {f: hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES},
          'sequential': expected, 'real': actual, 'feature_names': actual_names, 'input_features': actual_features,
          'score': actual_scores, 'dense_score': actual_dense_scores,
          'probability': actual_probability, 'dense_probability': dense_probability,
          'decision': actual_decision, 'dense_decision': actual_dense_decision,
          'samples': actual_samples, 'dense_samples': actual_dense_samples,
          'prefix_svc_max_decision_error': prefix_error,
          'probability_tolerance': 'probabilities exact; logs rtol/atol 1e-12 (Float64 outputs for both input dtypes)',
          'tolerance': 'exact: feature-name strings, small integer scalar cases and exactly representable dense means/medians'}
out = ROOT / 'builds/pipeline-parity.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print('PASS: 7 sequential cases, 7 feature-name cases, 6 input-feature cases, 6 score cases, 8 probability cases, 8 decision cases, 6 sample-score cases, StandardScaler+SVC target-type prefix and real DummyRegressor Float32/Float64 integration')
