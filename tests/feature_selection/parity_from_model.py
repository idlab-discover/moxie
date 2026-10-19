"""Compare importance math/masks to pinned sklearn and route real native fits.

The wrapper fixtures use orthogonal designs with analytic Lasso/ElasticNet
solutions. This validates fit routing without substituting a test optimizer.
It is not sklearn's entire estimator validation/metadata-routing pipeline.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from operator import attrgetter
from numbers import Integral
import json
import subprocess
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import feature_selection_checks as check
REFERENCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'


def oracle():
    assert subprocess.check_output(['git', '-C', REFERENCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    nodes = []
    namespace = dict(np=np, attrgetter=attrgetter, issparse=lambda x: False,
                     safe_sqr=lambda x: x * x, Integral=Integral,
                     check_is_fitted=lambda x: None, NotFittedError=ValueError)
    for filename, names in [('_base.py', {'_get_feature_importances'}), ('_from_model.py', {'_calculate_threshold'})]:
        path = REFERENCE / 'sklearn/feature_selection' / filename
        assert not subprocess.check_output(['git', '-C', REFERENCE, 'status', '--porcelain', '--', path], text=True).strip()
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if filename == '_from_model.py':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SelectFromModel')
            nodes.append(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_support_mask'))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def literal(value):
    if np.isnan(value): return 'nan[DType.float64]()'
    if np.isposinf(value): return 'inf[DType.float64]()'
    if np.isneginf(value): return '-inf[DType.float64]()'
    return repr(float(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly-root', type=Path, default=ROOT)
    assembly_root = parser.parse_args().assembly_root.resolve()
    ns = oracle()
    expected, exact, bodies, kinds = [], [], [], []
    def emit(lines, expression, value, discrete=False):
        if discrete: exact.append(len(expected))
        lines.append(f'    print({len(expected)}, Float64({expression}))')
        expected.append(float(value))
        kinds.append(dtype)
    def array(lines, name, data, dtype):
        lines.append(f'    var {name} = nm.zeros[DType.{dtype}](nm.Shape({", ".join(map(str, data.shape))}))')
        for i, value in enumerate(data.ravel()):
            lines.append(f'    {name}.unsafe_set({i}, SIMD[DType.{dtype}, 1]({literal(value)}))')
    rng = np.random.default_rng(719)
    kernels = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for dtype in ('float32', 'float64'):
            fixtures = [np.array([3., -3, 0, 2, -2]), rng.normal(size=(3, 5)),
                        np.array([[3., 0, -4, 3, np.nan], [4., 0, 3, 4, np.inf]])]
            for data in fixtures:
                data = data.astype(dtype)
                for reverse in (False, True):
                    lines = []
                    array(lines, 'backing', data, dtype)
                    dims = ', '.join(map(str, data.shape))
                    strides = [s // data.itemsize for s in data.strides]
                    signed = ', '.join(str(-s if reverse else s) for s in strides)
                    lines.append(f'    var raw = backing.view_with_layout(nm.Shape({dims}), nm.NDArrayStrides({signed}), {data.size-1 if reverse else 0})')
                    raw = data[tuple(slice(None, None, -1) for _ in data.shape)] if reverse else data
                    model = SimpleNamespace(coef_=raw)
                    for i, order in enumerate((1, 2, 3, -1, -3, np.inf, -np.inf)):
                        scores = ns['_get_feature_importances'](model, 'auto', 'norm', order)
                        lines.append(f'    var scores{i} = aggregate_importances(raw, {literal(order)})')
                        for j, value in enumerate(scores): emit(lines, f'scores{i}.unsafe_get({j})', value)
                        for threshold_id, (threshold, native) in enumerate([(None, 'None'), ('mean', 'ModelThreshold(String("mean"))'), ('1.2 * median', 'ModelThreshold(String("1.2 * median"))'), (-np.inf, 'ModelThreshold(-inf[DType.float64]())')]):
                            value = ns['_calculate_threshold'](model, scores, threshold)
                            emit(lines, f'importance_threshold(scores{i}, {native}, False)', value)
                            for k in (0, 2, 10):
                                holder = SimpleNamespace(estimator=model, prefit=True, max_features=k, importance_getter='auto', norm_order=order, threshold=threshold)
                                mask = ns['_get_support_mask'](holder)
                                name = f'mask{i}_{threshold_id}_{k}'
                                lines.append(f'    var {name} = importance_mask(scores{i}, {literal(value)}, {k})')
                                for j, keep in enumerate(mask): emit(lines, f'Int({name}[{j}])', keep, True)
                        kernels += 1
                    squared = ns['_get_feature_importances'](model, 'auto', 'square')
                    lines.append('    var squared = aggregate_importances(raw, square=True)')
                    for j, value in enumerate(squared): emit(lines, f'squared.unsafe_get({j})', value)
                    bodies.append(lines)
        wrappers = 0
        for dtype in ('float32', 'float64'):
            for model_name, ratio in [('Lasso', 1.), ('ElasticNet', .5)]:
                for outputs in (1, 2):
                    for weighted in (False, True):
                        for prefit in (False, True):
                            lines = []
                            X = np.tile(np.eye(4) * 2, (2, 1)).astype(dtype)
                            beta = np.array([6., -3., .5, 0], dtype=dtype)
                            if outputs == 2: beta = np.stack((beta, -2 * beta))
                            y = X @ beta.T
                            weights = np.array([3., 1, 2, 4, 1, 2, 1, 1], dtype=dtype) if weighted else np.ones(8, dtype=dtype)
                            gram = np.sum(weights[:, None] * X * X, axis=0) / weights.sum()
                            coefs = np.sign(beta) * np.maximum(np.abs(beta) * gram - .1 * ratio, 0) / (gram + .1 * (1 - ratio))
                            ref_model = type(model_name, (), {})()
                            ref_model.coef_ = coefs
                            ref_model.l1_ratio = ratio
                            array(lines, 'X', X, dtype)
                            array(lines, 'y', y, dtype)
                            array(lines, 'weights', weights, dtype)
                            lines.append(f'    var model = {model_name}[DType.{dtype}](alpha=0.1, fit_intercept=False, tol=1e-7)')
                            weight_arg = ', sample_weight=weights' if weighted else ''
                            if prefit: lines.append(f'    model.fit(X, y{weight_arg})')
                            lines.append(f'    var selector = SelectFromModel({model_name}Importance(model.copy()), prefit={prefit}, max_features=MaxFeatures(2))')
                            lines.append(f'    selector.fit(X, y.copy(){weight_arg})')
                            lines.append('    var fitted = selector.estimator_()')
                            for j, value in enumerate(coefs.ravel()): emit(lines, f'fitted.model.coef_.unsafe_get({j})', value)
                            for i, (threshold, native) in enumerate([(None, 'None'), ('median', 'ModelThreshold(String("median"))'), ('1.1*mean', 'ModelThreshold(String("1.1*mean"))'), (-np.inf, 'ModelThreshold(-inf[DType.float64]())')]):
                                lines.append(f'    selector.set_params(threshold={native})')
                                holder = SimpleNamespace(estimator=ref_model, estimator_=ref_model, prefit=prefit, max_features=2, max_features_=2, importance_getter='auto', norm_order=1, threshold=threshold)
                                mask = ns['_get_support_mask'](holder)
                                score = ns['_get_feature_importances'](ref_model, 'auto', 'norm', 1)
                                emit(lines, 'selector.threshold_()', ns['_calculate_threshold'](ref_model, score, threshold))
                                lines.append(f'    var mask{i} = selector.get_support()[List[Bool]].copy()')
                                for j, keep in enumerate(mask): emit(lines, f'Int(mask{i}[{j}])', keep, True)
                                lines.append(f'    var selected{i} = selector.transform(X)')
                                emit(lines, f'selected{i}.shape[1]', mask.sum(), True)
                                for j, value in enumerate(X[:, mask].ravel()): emit(lines, f'selected{i}.unsafe_get({j})', value, True)
                            bodies.append(lines)
                            wrappers += 1
    source = ROOT / 'builds/from-model-parity-source/main.mojo'
    source.parent.mkdir(parents=True, exist_ok=True)
    lines = ['import numojo as nm', 'from std.math import nan, inf',
             'from moxie.linear_model import Lasso, ElasticNet',
             'from moxie.feature_selection import SelectFromModel, LassoImportance, ElasticNetImportance, MaxFeatures, ModelThreshold',
             'from moxie.feature_selection._model_importance import aggregate_importances',
             'from moxie.feature_selection._from_model import importance_mask, importance_threshold']
    for i, body in enumerate(bodies): lines += ['@no_inline', f'def case{i}() raises:'] + body
    lines += ['def main() raises:'] + [f'    case{i}()' for i in range(len(bodies))]
    source.write_text('\n'.join(lines) + '\n')
    check.compile_file(assembly_root, source, 'from-model-parity')
    output = check.run(assembly_root / 'builds/from-model-parity', capture=True).stdout
    actual = np.array([float(line.split()[1]) for line in output.splitlines()])
    expected = np.array(expected)
    bad = [i for i in exact if actual[i] != expected[i]]
    if bad:
        expressions = {int(line.split("print(")[1].split(",")[0]): line.strip() for line in lines if "print(" in line}
        print("mask mismatches", [(i, actual[i], expected[i], expressions[i]) for i in bad[:20]])
    np.testing.assert_array_equal(actual[exact], expected[exact])
    tolerances = {'float32': (3e-5, 2e-8), 'float64': (3e-8, 2e-12)}
    for dtype, (rtol, atol) in tolerances.items():
        chosen = np.array(kinds) == dtype
        np.testing.assert_allclose(actual[chosen], expected[chosen], rtol=rtol, atol=atol, equal_nan=True)
    report = dict(reference=REVISION, norm_cases=kernels, real_model_cases=wrappers,
                  comparisons=len(expected), exact_comparisons=len(exact), tolerances=tolerances)
    (ROOT / 'builds/from-model-parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS SelectFromModel', json.dumps(report))


if __name__ == '__main__':
    main()
