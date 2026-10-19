"""Compare real native RFE fits with the pinned sklearn elimination body.

The Python fitting oracle is the exact Lasso/ElasticNet solution on orthogonal
columns, including weights and multiple outputs. It is validation-only, not a
replacement runtime or a claim of full sklearn estimator-validation parity.
"""
import argparse
import ast
import copy
import json
from numbers import Integral
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import feature_selection_checks as check
from parity_from_model import oracle as importance_oracle, literal, REVISION, REFERENCE


class OrthogonalModel:
    def __init__(self, ratio):
        self.ratio = ratio

    def fit(self, X, y, sample_weight=None):
        weights = np.ones(X.shape[0], dtype=X.dtype) if sample_weight is None else sample_weight
        gram = (X.T * weights) @ X / weights.sum()
        np.testing.assert_array_equal(gram, np.diag(np.diag(gram)))
        cross = (X.T * weights) @ y / weights.sum()
        coef = np.sign(cross) * np.maximum(np.abs(cross) - .1 * self.ratio, 0)
        denominator = np.diag(gram) + .1 * (1 - self.ratio)
        self.coef_ = (coef / (denominator if y.ndim == 1 else denominator[:, None])).T
        return self


def oracle():
    ns = importance_oracle()
    path = REFERENCE / 'sklearn/feature_selection/_rfe.py'
    assert not subprocess.check_output(['git', '-C', REFERENCE, 'status', '--porcelain', '--', path], text=True).strip()
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RFE')
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_fit')
    ns.update(clone=copy.deepcopy, validate_data=lambda self, X, y, **kwargs: (X, y), warnings=warnings, Integral=Integral)
    exec(compile(ast.Module(body=[fit], type_ignores=[]), str(path), 'exec'), ns)
    return ns['_fit']



NATIVE_HEADER = r"""
import numojo as nm
from moxie.linear_model import Lasso, ElasticNet
from moxie.feature_selection import RFE, FeatureCount, LassoImportance, ElasticNetImportance

@no_inline
def dump[dtype: DType](data: nm.NDArray[dtype]) raises:
    var rows = 1 if data.ndim == 1 else data.shape[0]
    var columns = data.shape[data.ndim - 1]
    for i in range(rows):
        for j in range(columns):
            var offset = j * Int(data.strides[data.ndim - 1])
            if data.ndim == 2:
                offset += i * Int(data.strides[0])
            print(Float64(data.unsafe_get(offset)))

"""

NATIVE_DRIVER = r"""
@no_inline
def run___MODEL__[dtype: DType](mut backing: nm.NDArray[dtype], y: nm.NDArray[dtype], weights: nm.NDArray[dtype], count: Optional[FeatureCount], step: FeatureCount, reverse: Bool, weighted: Bool) raises:
    var selector = RFE(__MODEL__Importance(__MODEL__[dtype](alpha=0.1, fit_intercept=False, tol=1e-7)), n_features_to_select=count, step=step)
    var X = backing.view()
    if reverse:
        X = backing.view_with_layout(nm.Shape(12, 6), nm.NDArrayStrides(6, -1), 5)
    if weighted:
        selector.fit(X, y, sample_weight=weights)
    else:
        selector.fit(X, y)
    var support = selector.support_()
    var ranks = selector.ranking_()
    for j in range(6):
        print(Float64(Int(support[j])))
        print(Float64(ranks[j]))
    print(Float64(selector.n_features_()))
    dump(selector.estimator_().importances("auto"))
    var transformed = selector.transform(X)
    dump(transformed)
    dump(selector.inverse_transform(transformed))
    dump(selector.predict(X))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly-root', type=Path, default=ROOT)
    assembly_root = parser.parse_args().assembly_root.resolve()
    fit = oracle()
    expected, exact, dtypes, bodies = [], [], [], []
    def emit(value, discrete=False):
        if discrete:
            exact.append(len(expected))
        expected.append(float(value))
        dtypes.append(dtype)
    def array(lines, name, data):
        lines.append(f'    var {name} = nm.zeros[DType.{dtype}](nm.Shape({", ".join(map(str, data.shape))}))')
        for j, value in enumerate(data.ravel()):
            lines.append(f'    {name}.unsafe_set({j}, SIMD[DType.{dtype}, 1]({literal(value)}))')
    cases = 0
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
                            weights = np.array([1, 3, 2, 4, 2, 1, 2, 1, 1, 2, 1, 3], dtype=dtype)
                            reference_X = X[:, ::-1] if mode == 'reverse' else X
                            lines = []
                            array(lines, 'X', X)
                            array(lines, 'y', y)
                            array(lines, 'weights', weights)
                            for count, step in [(None, 1), (1, 2), (.5, .5), (2, .1), (1, .5), (1, 20), (1., 20), (9, 1)]:
                                holder = SimpleNamespace(estimator=OrthogonalModel(ratio), n_features_to_select=count,
                                                         step=step, verbose=0, importance_getter='auto')
                                kwargs = dict(sample_weight=weights) if weighted else {}
                                fit(holder, reference_X, y, **kwargs)
                                count_literal = 'None' if count is None else f'FeatureCount({repr(count)})'
                                lines.append(f'    run_{model}[DType.{dtype}](X, y, weights, {count_literal}, FeatureCount({repr(step)}), {mode == "reverse"}, {weighted})')
                                for j in range(6):
                                    emit(holder.support_[j], True)
                                    emit(holder.ranking_[j], True)
                                emit(holder.n_features_, True)
                                for value in holder.estimator_.coef_.ravel():
                                    emit(value)
                                reduced = reference_X[:, holder.support_]
                                restored = np.zeros_like(reference_X)
                                restored[:, holder.support_] = reduced
                                for data in (reduced, restored):
                                    for value in data.ravel():
                                        emit(value, True)
                                predicted = reduced @ holder.estimator_.coef_.T
                                for value in predicted.ravel():
                                    emit(value)
                                cases += 1
                            bodies.append(lines)
    source = ROOT / 'builds/rfe-parity-source/main.mojo'
    source.parent.mkdir(parents=True, exist_ok=True)
    lines = [NATIVE_HEADER, *(NATIVE_DRIVER.replace("__MODEL__", model) for model in ("Lasso", "ElasticNet"))]
    for i, body in enumerate(bodies):
        lines += ['@no_inline', f'def case{i}() raises:'] + body
    lines += ['def main() raises:'] + [f'    case{i}()' for i in range(len(bodies))]
    source.write_text('\n'.join(lines) + '\n')
    check.compile_file(assembly_root, source, 'rfe-parity')
    output = check.run(assembly_root / 'builds/rfe-parity', capture=True).stdout
    actual = np.array([float(line) for line in output.splitlines()])
    expected = np.array(expected)
    np.testing.assert_array_equal(actual[exact], expected[exact])
    tolerances = {'float32': (3e-5, 2e-7), 'float64': (3e-8, 2e-12)}
    for dtype, (rtol, atol) in tolerances.items():
        chosen = np.array(dtypes) == dtype
        np.testing.assert_allclose(actual[chosen], expected[chosen], rtol=rtol, atol=atol)
    report = dict(reference=REVISION, real_model_cases=cases, comparisons=len(expected),
                  exact_comparisons=len(exact), tolerances=tolerances,
                  scope='sklearn RFE._fit body plus analytic orthogonal Lasso/ElasticNet oracle')
    (ROOT / 'builds/rfe-parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS RFE', json.dumps(report))


if __name__ == '__main__':
    main()
