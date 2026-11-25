"""Native string, exact Int64 and list labels against pinned sklearn scorers.

Uses the same documented validation/LabelBinarizer/SciPy test plumbing as the
existing parity runners; this is bounded dense parity, not full sklearn parity.
"""
import argparse
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


def main(list_targets_only=False):
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
        if not list_targets_only:
            lines.append(f'    var refit_x = nm.zeros[DType.{dtype}](nm.Shape(18, 3))')
            for j, v in enumerate(x[:, :3].ravel()):
                lines.append(f'    refit_x.unsafe_set({j}, {int(v)})')
        if strings:
            vals = ', '.join(json.dumps(str(v), ensure_ascii=False) for v in y)
            shape = ', 18, 1' if case % 3 == 1 else ''
            if not list_targets_only:
                lines.append(f'    var y = StringTarget([{vals}]{shape})')
        elif not list_targets_only:
            shape = '18, 1' if case % 3 == 1 else '18'
            # Reversed, offset view exercises exact labels and native stride access.
            lines += ['    var backing = nm.zeros[DType.int64](nm.Shape(40))',
                      f'    var y = backing.view_with_layout(nm.Shape({shape}), nm.NDArrayStrides({"-2, 1" if "," in shape else "-2"}), 36)']
            for j, v in enumerate(y):
                lines.append(f'    y.unsafe_set({-2*j}, {int(v)})')
        if strings:
            list_type, list_dtype = 'String', 'float64'
            list_y, list_values = y, vals
        elif case % 4 == 2:
            list_type = 'Int' if case < 8 else 'Int64'
            list_dtype = 'int64'
            list_y, list_values = y, ', '.join(str(int(v)) for v in y)
        else:
            list_type = 'Float32' if case < 8 else 'Float64'
            list_dtype = list_type.lower()
            list_y = groups.astype(list_dtype)
            list_values = ', '.join(str(float(v)) for v in list_y)
        lines.append(f'    var list_y: List[{list_type}] = [{list_values}]')
        for method in ['f_classif', 'chi2', 'mutual_info_classif']:
            is_mi = method.startswith('mutual')
            # Check native list targets at the selector entry point. Use the
            # continuous MI path here alongside the array path above.
            list_target = f'DType.{dtype}, DType.{list_dtype}, List[{list_type}]'
            if is_mi:
                list_truth = mi[method](x.copy(), list_y, random_state=7)
                list_scorer = f'MutualInfoScorer[{list_target}](random_state=7)'
            else:
                list_truth, list_ps = ns[method](x, list_y)
                list_scorer = f'BuiltinScorer[{list_target}]("{method}")'
            selection = f'list_{method}'
            lines += [f'    var {selection} = SelectKBest({list_scorer}, k=2)',
                      f'    _ = {selection}.fit(x, list_y.copy())',
                      f'    var {selection}_scores = {selection}.scores_()',
                      f'    var {selection}_mask = {selection}.get_support()[List[Bool]].copy()']
            emit(f'{selection}_scores.unsafe_get({{j}})', list_truth)
            if not is_mi:
                lines.append(f'    var {selection}_p = {selection}.pvalues_()')
                emit(f'{selection}_p.value().unsafe_get({{j}})', list_ps)
            emit(f'Int({selection}_mask[{{j}}])',
                 ns['SelectKBest'](SimpleNamespace(scores_=list_truth, k=2)), True)
            if list_targets_only:
                continue
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
        if not list_targets_only:
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
    if list_targets_only:
        source = folder / 'list_targets.mojo'
        definitions = [
            '\n'.join(['@no_inline', f'def {name}() raises:', *body])
            for name, body in cases
        ]
        calls = [f'    {name}()' for name, _ in cases]
        source.write_text('\n'.join(lines + definitions + ['def main() raises:', *calls]) + '\n')
        binary = 'label-parity-list-targets'
        check.compile_file(ROOT, source, binary)
        outputs.append(check.run(ROOT / 'builds' / binary, capture=True).stdout)
    else:
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
    report = 'label-parity-list-targets.json' if list_targets_only else 'label-parity.json'
    (ROOT / 'builds' / report).write_text(json.dumps({'values': len(expected), 'cases': len(cases), 'rtol': 5e-5, 'atol': 2e-12}))
    print('PASS native label parity', len(expected), 'values')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-targets-only', action='store_true')
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main(args.list_targets_only)
