"""Focused synthetic delegation checks against the pinned local sklearn sources."""
import argparse
from pathlib import Path
import subprocess

from compare_reference import load, REVISION

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sklearn-root', type=Path, required=True)
    parser.add_argument('--native', type=Path)
    args = parser.parse_args()
    source = args.sklearn_root.resolve()
    revision = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f'Expected {REVISION}, got {revision}')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--',
                    'sklearn/frozen/_frozen.py', 'sklearn/base.py'], check=True)
    base = load('sklearn._moxie_updates_base', source / 'sklearn/base.py')
    Frozen = load('sklearn._moxie_updates_frozen', source / 'sklearn/frozen/_frozen.py').FrozenEstimator

    class RegressorProbe(base.RegressorMixin, base.BaseEstimator):
        def __init__(self):
            self.state = 0

        def __sklearn_is_fitted__(self):
            return self.state != 0

        def fit(self, X, y):
            raise AssertionError('inner fit must not run')

        def partial_fit(self, X, y):
            if y < 0:
                raise ValueError('invalid target')
            self.state += X + y
            return self

        def predict(self, X):
            return self.state

    class ClassifierProbe(base.ClassifierMixin, base.BaseEstimator):
        def __init__(self):
            self.state = 0
            self.classes_ = 0

        def __sklearn_is_fitted__(self):
            return self.classes_ != 0

        def fit(self, X, y):
            raise AssertionError('inner fit must not run')

        def partial_fit(self, X, y, classes=None):
            if not self.classes_ and classes is None:
                raise ValueError('classes required on first call')
            if classes is not None:
                if self.classes_ and classes != self.classes_:
                    raise ValueError('classes changed')
                self.classes_ = classes
            self.state += X + y
            return self

        def predict(self, X):
            return self.state

    class ClusterProbe(base.ClusterMixin, base.BaseEstimator):
        def __init__(self):
            self.state = 0

        def __sklearn_is_fitted__(self):
            return self.state != 0

        def fit(self, X, y=None):
            raise AssertionError('inner fit must not run')

        def partial_fit(self, X, y=None):
            if X < 0:
                raise ValueError('invalid input')
            self.state += X + (y if y is not None else 0)
            return self

        def transform(self, X):
            return self.state + X

    def rejected(call, message):
        try:
            call()
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError('expected delegated error')

    expected = {}
    regression = Frozen(RegressorProbe())
    shared = base.clone(regression)
    params = regression.get_params()
    regression.partial_fit(2, 3)
    expected['regression_first'] = shared.predict(0)
    regression.fit(999, 999)
    nested = Frozen(shared)
    nested.partial_fit(3, 4)
    expected['regression_nested'] = regression.predict(0)
    rejected(lambda: shared.partial_fit(1, -1), 'invalid target')
    restored = Frozen(params['estimator'])
    regression.set_params(estimator=RegressorProbe())
    regression.partial_fit(1, 1)
    expected['regression_replaced'] = regression.predict(0)
    expected['regression_old'] = restored.predict(0)
    regression.set_params(**params)

    classifier = Frozen(ClassifierProbe())
    rejected(lambda: classifier.partial_fit(1, 2), 'classes required')
    classifier.partial_fit(1, 2, classes=7)
    classifier_clone = base.clone(classifier)
    classifier.partial_fit(3, 4)
    rejected(lambda: classifier.partial_fit(1, 2, classes=8), 'classes changed')
    expected['classifier_state'] = classifier_clone.predict(0)
    expected['classifier_classes'] = classifier_clone.estimator.classes_

    cluster = Frozen(ClusterProbe())
    assert base.is_clusterer(cluster)  # identity does not require fitted state
    nested_cluster = Frozen(cluster)
    cluster.partial_fit(3)
    nested_cluster.partial_fit(4, 5)
    cluster.fit(999, None)
    rejected(lambda: cluster.partial_fit(-1), 'invalid input')
    expected['cluster_state'] = cluster.transform(0)
    expected['cluster_identity'] = int(base.is_clusterer(cluster))
    expected['nested_cluster_identity'] = int(base.is_clusterer(nested_cluster))
    expected['regressor_identity'] = int(base.is_clusterer(regression))
    for method in ('fit_predict', 'fit_transform'):
        assert not hasattr(cluster, method)

    command = [str(args.native.resolve())] if args.native else [
        str(ROOT / 'builds/tests/frozen/test_frozen_updates')]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    actual = {}
    for line in output.splitlines():
        key, value = line.split()
        assert key not in actual, key
        actual[key] = int(value)
    assert actual == expected, (actual, expected)
    print(f'Passed {len(expected)} focused incremental/identity comparisons at sklearn {REVISION}')


if __name__ == '__main__':
    main()
