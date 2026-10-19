"""Check the RFE observation path against pinned sklearn RFE._fit(step_score).

Native fitting uses real Lasso/ElasticNet. The Python oracle uses their exact
orthogonal-design solutions, as in parity_rfe.py; no native fitting double.
"""
import json
from types import SimpleNamespace
import warnings

import numpy as np

from parity_rfe import ROOT, REVISION, oracle, OrthogonalModel, literal, check

DRIVER = r'''
import numojo as nm
from moxie.linear_model import Lasso, ElasticNet
from moxie.feature_selection import RFE, FeatureCount, LassoImportance, ElasticNetImportance
from rfe_path import HeldOutPath

@no_inline
def run___MODEL__[dtype: DType](mut backing: nm.NDArray[dtype], y: nm.NDArray[dtype], y_test: nm.NDArray[dtype], weights: nm.NDArray[dtype], score_weights: nm.NDArray[dtype], count: Optional[FeatureCount], step: FeatureCount, reverse: Bool, weighted: Bool) raises:
    var X = backing.view()
    if reverse:
        X = backing.view_with_layout(nm.Shape(12, 6), nm.NDArrayStrides(6, -1), 5)
    var selector = RFE(__MODEL__Importance(__MODEL__[dtype](alpha=0.1, fit_intercept=False, tol=1e-7)), n_features_to_select=count, step=step)
    var observer = HeldOutPath[__MODEL__Importance[dtype]](X, y_test, score_weights.copy())
    var fit_weights: Optional[nm.NDArray[dtype]] = None
    if weighted:
        fit_weights = weights.copy()
    selector._fit_with_observer(X, y, observer, sample_weight=fit_weights)
    print(len(observer.steps))
    for i in range(len(observer.steps)):
        print(observer.steps[i].n_features)
        print(observer.steps[i].score)
        for j in range(6):
            print(Int(observer.steps[i].support[j]))
            print(observer.steps[i].ranking[j])
'''


def r2(estimator, X, y, weights):
    prediction = (X @ estimator.coef_.T).astype(np.float64)
    target = y.astype(np.float64)
    if target.ndim == 1:
        target = target[:, None]
        prediction = prediction[:, None]
    weights = weights.astype(np.float64)[:, None]
    average = (weights * target).sum(axis=0) / weights.sum()
    numerator = (weights * (target - prediction) ** 2).sum(axis=0)
    denominator = (weights * (target - average) ** 2).sum(axis=0)
    return (1 - numerator / denominator).mean()


def main():
    fit = oracle()
    expected, exact, dtypes, bodies = [], [], [], []
    cases = steps = 0

    def emit(value, discrete=False):
        if discrete:
            exact.append(len(expected))
        expected.append(float(value))
        dtypes.append(dtype)

    def array(lines, name, data):
        shape = ', '.join(map(str, data.shape))
        lines.append(f'    var {name} = nm.zeros[DType.{dtype}](nm.Shape({shape}))')
        for i, value in enumerate(data.ravel()):
            lines.append(f'    {name}.unsafe_set({i}, SIMD[DType.{dtype}, 1]({literal(value)}))')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for dtype in ('float32', 'float64'):
            for model, ratio in [('Lasso', 1.), ('ElasticNet', .5)]:
                for outputs in (1, 2):
                    for weighted in (False, True):
                        for mode in ('distinct', 'ties', 'reverse'):
                            X = np.tile(np.eye(6) * 2, (2, 1)).astype(dtype)
                            beta = np.array([6, -3, 1.5, -.5, .2, 0] if mode != 'ties' else [3, -3, 0, 0, 1, -1], dtype=dtype)
                            if outputs == 2:
                                beta = np.stack((beta, -2 * beta))
                            y = X @ beta.T
                            y_test = y + np.asarray(.25, dtype=dtype)
                            weights = np.array([1, 3, 2, 4, 2, 1, 2, 1, 1, 2, 1, 3], dtype=dtype)
                            score_weights = weights[::-1].copy()
                            reference_X = X[:, ::-1] if mode == 'reverse' else X
                            lines = []
                            for name, data in [('X', X), ('y', y), ('y_test', y_test), ('weights', weights), ('score_weights', score_weights)]:
                                array(lines, name, data)
                            for count, step in [(None, 1), (1, 2), (.5, .5), (1, .5), (1, 20), (9, 1)]:
                                holder = SimpleNamespace(estimator=OrthogonalModel(ratio), n_features_to_select=count, step=step, verbose=0, importance_getter='auto')
                                kwargs = dict(sample_weight=weights) if weighted else {}
                                fit(holder, reference_X, y,
                                    step_score=lambda fitted, features: r2(fitted, reference_X[:, features], y_test, score_weights), **kwargs)
                                count_literal = 'None' if count is None else f'FeatureCount({count!r})'
                                lines.append(f'    run_{model}[DType.{dtype}](X, y, y_test, weights, score_weights, {count_literal}, FeatureCount({step!r}), {mode == "reverse"}, {weighted})')
                                emit(len(holder.step_scores_), True)
                                for n, score, support, ranking in zip(holder.step_n_features_, holder.step_scores_, holder.step_support_, holder.step_ranking_, strict=True):
                                    emit(n, True)
                                    emit(score)
                                    for selected, rank in zip(support, ranking, strict=True):
                                        emit(selected, True)
                                        emit(rank, True)
                                    steps += 1
                                cases += 1
                            bodies.append(lines)
    source = ROOT / 'builds/rfe-path-parity-source/main.mojo'
    source.parent.mkdir(parents=True, exist_ok=True)
    header, driver = DRIVER.split('@no_inline', 1)
    lines = [header, *('@no_inline' + driver.replace('__MODEL__', model) for model in ('Lasso', 'ElasticNet'))]
    for i, body in enumerate(bodies):
        lines += ['@no_inline', f'def case{i}() raises:'] + body
    lines += ['def main() raises:'] + [f'    case{i}()' for i in range(len(bodies))]
    source.write_text('\n'.join(lines) + '\n')
    check.compile_file(ROOT, source, 'rfe-path-parity')
    output = check.run(ROOT / 'builds/rfe-path-parity', capture=True).stdout
    actual = np.array([float(line) for line in output.splitlines()])
    expected = np.array(expected)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual[exact], expected[exact])
    tolerances = {'float32': (3e-5, 2e-7), 'float64': (3e-8, 2e-12)}
    for dtype, (rtol, atol) in tolerances.items():
        selected = np.array(dtypes) == dtype
        np.testing.assert_allclose(actual[selected], expected[selected], rtol=rtol, atol=atol)
    report = dict(reference=REVISION, cases=cases, steps=steps, comparisons=len(expected), exact_comparisons=len(exact), tolerances=tolerances,
                  scope='original sklearn RFE._fit step_score with analytic orthogonal Lasso/ElasticNet fits and weighted R2')
    (ROOT / 'builds/rfe-path-parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS RFE path', json.dumps(report))


if __name__ == '__main__':
    main()
