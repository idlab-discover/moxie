"""Native string/exact Int64 labels against pinned sklearn scorer/policy bodies.

Uses the same documented validation/LabelBinarizer/SciPy test plumbing as the
existing parity runners; this is bounded dense parity, not full sklearn parity.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
from parity_mutual_info import oracle as mi_oracle
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import feature_selection_checks as check
spec = importlib.util.spec_from_file_location('univariate_parity', ROOT / 'tests/feature_selection/univariate/parity.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def main():
    ns, mi = u.oracle(), mi_oracle()
    lines = ['import numojo as nm', 'from moxie._labels import StringTarget',
             'from moxie.feature_selection import f_classif, chi2, mutual_info_classif, DiscreteFeatures, BuiltinScorer, MutualInfoScorer, SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect, SelectionParameter', 'from std.testing import assert_equal, assert_raises',
             'from moxie._random import RandomState', 'from moxie.base import clone']
    expected, exact, cases = [], [], []
    def emit(expr, values, mask=False):
        for j, value in enumerate(values):
            if mask:
                exact.append(len(expected))
            lines.append(f'    print({len(expected)}, {expr.format(j=j)})')
            expected.append(float(value))
    rng = np.random.RandomState(391)
    variants = [('SelectKBest', 'k=SelectionParameter(2)', 'SelectKBest', {'k': 2}),
                ('SelectPercentile', 'percentile=50', 'SelectPercentile', {'percentile': 50}),
                ('GenericUnivariateSelect', 'mode="k_best", param=SelectionParameter(2)', 'SelectKBest', {'k': 2})]
    variants += [(c, 'alpha=0.05', c, {'alpha': .05}) for c in ['SelectFpr', 'SelectFdr', 'SelectFwe']]
    for case in range(12):
        dtype = 'float32' if case % 2 == 0 else 'float64'
        x = rng.randint(0, 9, (18, 4)).astype(dtype)
        # Separate chi-square cutoffs; one-to-one scales preserve discrete MI.
        x *= np.array([1, 3, 7, 13], dtype=dtype)
        groups = (np.arange(18) * 5 + 1) % (2 if case % 3 == 0 else 3)
        strings = case % 4 < 2
        y = np.array(['spam', 'ham', 'é'][0:3])[groups] if strings else np.array([-2**63, 2**53, 2**53+1], dtype=np.int64)[groups]
        target_type = 'StringTarget' if strings else 'nm.NDArray[DType.int64]'
        target_dt = 'float64' if strings else 'int64'
        fn = f'case{case}'
        case_start = len(lines)
        lines.append(f'    var x = nm.zeros[DType.{dtype}](nm.Shape(18, 4))')
        for j, v in enumerate(x.ravel()):
            lines.append(f'    x.unsafe_set({j}, {int(v)})')
        lines.append(f'    var refit_x = nm.zeros[DType.{dtype}](nm.Shape(18, 3))')
        for j, v in enumerate(x[:, :3].ravel()):
            lines.append(f'    refit_x.unsafe_set({j}, {int(v)})')
        if strings:
            vals = ', '.join(json.dumps(str(v), ensure_ascii=False) for v in y)
            shape = ', 18, 1' if case % 3 == 1 else ''
            lines.append(f'    var y = StringTarget([{vals}]{shape})')
        else:
            shape = '18, 1' if case % 3 == 1 else '18'
            # Reversed, offset view exercises exact labels and native stride access.
            lines += ['    var backing = nm.zeros[DType.int64](nm.Shape(40))',
                      f'    var y = backing.view_with_layout(nm.Shape({shape}), nm.NDArrayStrides({"-2, 1" if "," in shape else "-2"}), 36)']
            for j, v in enumerate(y):
                lines.append(f'    y.unsafe_set({-2*j}, {int(v)})')
        for method in ['f_classif', 'chi2', 'mutual_info_classif']:
            is_mi = method.startswith('mutual')
            truth = mi[method](x.copy(), y, discrete_features=True, random_state=7) if is_mi else ns[method](x, y)
            opts = ', discrete_features=DiscreteFeatures(True), random_state=7' if is_mi else ''
            name = "result_" + method
            lines.append(f'    var {name} = {method}(x, y{opts})')
            scores, ps = (truth, None) if is_mi else truth
            emit(name + ('.unsafe_get({j})' if is_mi else '.scores.unsafe_get({j})'), scores)
            if ps is not None:
                emit(name + '.pvalues.value().unsafe_get({j})', ps)
            scorer = f'MutualInfoScorer[DType.{dtype}, DType.{target_dt}, {target_type}](discrete_features=DiscreteFeatures(True), random_state=7)' if is_mi else f'BuiltinScorer[DType.{dtype}, DType.{target_dt}, {target_type}]("{method}")'
            for index, (cls, args, policy, config) in enumerate(variants):
                name = f's{method}{index}'
                lines.append(f'    var {name} = {cls}({scorer}, {args})')
                if is_mi and index >= 3:
                    lines += [f'    {name}.fit(x, y.copy())', '    with assert_raises():', f'        _ = {name}.get_support()']
                    continue
                lines.append(f'    {name}.fit(x, y.copy())')
                for phase in range(3):
                    phase_scores = scores[:3] if phase == 2 else scores
                    phase_ps = ps[:3] if phase == 2 and ps is not None else ps
                    snapshot = f'{name}_phase{phase}'
                    lines.append(f'    var {snapshot}_scores = {name}.scores_()')
                    emit(snapshot + '_scores.unsafe_get({j})', phase_scores)
                    if ps is not None:
                        lines.append(f'    var {snapshot}_p = {name}.pvalues_()')
                        emit(snapshot + '_p.value().unsafe_get({j})', phase_ps)
                    mask = ns[policy](SimpleNamespace(scores_=phase_scores, pvalues_=phase_ps, **config))
                    lines.append(f'    var {snapshot}_mask = {name}.get_support()[List[Bool]].copy()')
                    emit('Int(' + snapshot + '_mask[{j}])', mask, True)
                    if phase == 0:
                        lines += [f'    var clone{name} = clone({name})', '    with assert_raises():', f'        _ = clone{name}.scores_()', f'    clone{name}.fit(x, y.copy())', f'    {name} = clone{name}^']
                    if phase == 1:
                        lines.append(f'    {name}.fit(refit_x, y.copy())')
        # Seeded continuous MI also exercises compact IDs in neighbor grouping.
        truth = mi['mutual_info_classif'](x.copy(), y, random_state=7)
        lines.append('    var continuous = mutual_info_classif(x, y, random_state=7)')
        emit('continuous.unsafe_get({j})', truth)
        lines += ['    var state = RandomState(7)', '    var stream = mutual_info_classif(x, y, random_state=state)']
        emit('stream.unsafe_get({j})', truth)
        cases.append((fn, lines[case_start:]))
        del lines[case_start:]
    folder = ROOT / 'builds/label-parity-source'; folder.mkdir(exist_ok=True)
    outputs = []
    for name, body in cases:
        source = folder / (name + '.mojo')
        source.write_text('\n'.join(lines + ['def main() raises:'] + body) + '\n')
        binary = 'label-parity-' + name
        check.compile_file(ROOT, source, binary)
        outputs.append(check.run(ROOT / 'builds' / binary, capture=True).stdout)
    out = ''.join(outputs)
    actual = np.array([float(row.split()[1]) for row in out.splitlines()])
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-12, equal_nan=True)
    np.testing.assert_array_equal(actual[exact], np.array(expected)[exact])
    (ROOT / 'builds/label-parity.json').write_text(json.dumps({'values': len(expected), 'cases': len(cases), 'rtol': 5e-5, 'atol': 2e-12}))
    print('PASS native label parity', len(expected), 'values')

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
