"""Compare cumulative SelectFromModel with the pinned sklearn implementation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
OUTPUT = ROOT / 'builds/from-model-incremental-parity'


def oracle():
    import numpy as np
    import sklearn
    assert sklearn.__version__ == '1.9.1'
    assert subprocess.check_output(['git', '-C', REFERENCE, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    modules = []
    for package, filename in [('feature_selection', '_from_model'), ('decomposition', '_incremental_pca')]:
        source = REFERENCE / f'sklearn/{package}/{filename}.py'
        assert not subprocess.check_output(['git', '-C', REFERENCE, 'diff', 'HEAD', '--', source], text=True)
        spec = importlib.util.spec_from_file_location(f'sklearn.{package}._moxie_reference_incremental', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    SelectFromModel, IncrementalPCA = modules[0].SelectFromModel, modules[1].IncrementalPCA
    rng = np.random.RandomState(124)
    cases = []
    for dtype in ['float32', 'float64']:
        for components in [1, 2]:
            for mode in ['stream', 'prefit', 'fit_then_partial']:
                for reverse in [False, True]:
                    batches = [rng.normal(size=(12, 4)).astype(dtype) * np.asarray(scale, dtype=dtype)
                               for scale in [[4, 1, .3, .1], [.1, 7, .3, 2], [.2, 1, 8, .1]]]
                    if reverse:
                        batches = [x[::-1, ::-1] for x in batches]
                    estimator = IncrementalPCA(n_components=components)
                    if mode == 'prefit':
                        estimator.partial_fit(batches[0])
                    selector = SelectFromModel(estimator, prefit=mode == 'prefit', importance_getter='components_', max_features=2)
                    if mode == 'fit_then_partial':
                        selector.fit(batches[0])
                    snapshots = []
                    for x in batches:
                        selector.partial_fit(x, check_input=True)
                        snapshots.append(dict(support=selector.get_support().astype(int).tolist(),
                            threshold=float(selector.threshold_), count=selector.estimator_.n_samples_seen_,
                            components=np.abs(selector.estimator_.components_).tolist(),
                            mean=selector.estimator_.mean_.tolist(), var=selector.estimator_.var_.tolist(),
                            transformed=selector.transform(x).tolist()))
                    cases.append(dict(dtype=dtype, components=components, mode=mode, reverse=reverse,
                                      batches=[x.tolist() for x in batches], snapshots=snapshots))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'oracle.json').write_text(json.dumps(dict(revision=REVISION, seed=124, cases=cases), indent=2))


def main():
    import numpy as np
    sys.path.insert(0, str(ROOT/'tools'))
    import feature_selection_checks as check
    python = os.environ.get('MOXIE_REFERENCE_PYTHON', sys.executable)
    subprocess.run([python, str(Path(__file__).resolve()), '--oracle'], check=True)
    cases = json.loads((OUTPUT/'oracle.json').read_text())['cases']
    source = ['import numojo as nm', 'from std.math import abs',
              'from moxie.decomposition import IncrementalPCA',
              'from moxie.feature_selection import SelectFromModel, IncrementalPCAImportance, IncrementalPCAUpdateParams, MaxFeatures']
    expected, tolerances, labels = [], [], []
    for i, case in enumerate(cases):
        dt = case['dtype']
        source.append(f'@no_inline\ndef case_{i}() raises:')
        for b, values in enumerate(case['batches']):
            x = np.asarray(values)
            backing = x[::-1, ::-1] if case['reverse'] else x
            source.append(f'    var backing{b} = nm.zeros[DType.{dt}](nm.Shape(12, 4))')
            for j, value in enumerate(backing.flat):
                source.append(f'    backing{b}.unsafe_set({j}, SIMD[DType.{dt}, 1]({float(value)!r}))')
            source.append(f'    var x{b} = backing{b}.' + ('view_with_layout(nm.Shape(12, 4), nm.NDArrayStrides(-4, -1), 47)' if case['reverse'] else 'view()'))
        source.append(f'    var estimator = IncrementalPCA[DType.{dt}]({case["components"]})')
        if case['mode'] == 'prefit':
            source.append('    estimator.partial_fit(x0)')
        source.append(f'    var selector = SelectFromModel(IncrementalPCAImportance(estimator^), prefit={case["mode"] == "prefit"}, importance_getter="components_", max_features=MaxFeatures(2))')
        if case['mode'] == 'fit_then_partial':
            source.append('    selector.fit(x0)')
        for b, snapshot in enumerate(case['snapshots']):
            source += [f'    selector.partial_fit(x{b}, partial_fit_params=IncrementalPCAUpdateParams(check_input=True))',
                       f'    var support{b} = selector.get_support()[List[Bool]].copy()', f'    for value in support{b}:', '        print(Int(value))',
                       '    print(selector.threshold_())', '    print(selector.estimator_().model.n_samples_seen_)']
            def expect(key, exact=False):
                flat = np.asarray(snapshot[key]).ravel().tolist()
                expected.extend(flat)
                tolerances.extend([0 if exact else (2e-4 if dt == 'float32' else 3e-8)] * len(flat))
                labels.extend([f'case {i} batch {b} {dt} {key}'] * len(flat))
            expect('support', True)
            expect('threshold')
            expect('count', True)
            for key in ['components', 'mean', 'var']:
                source += [f'    var {key}{b} = selector.estimator_().model.{key}_.value().copy()',
                           f'    for j in range({key}{b}.size):', f'        print(abs({key}{b}.unsafe_get(j)))' if key == 'components' else f'        print({key}{b}.unsafe_get(j))']
                expect(key)
            source += [f'    var transformed{b} = selector.transform(x{b})', f'    for j in range(transformed{b}.size):', f'        print(Float64(transformed{b}.unsafe_get(j)))']
            expect('transformed', True)
    source.append('def main() raises:\n' + '\n'.join(f'    case_{i}()' for i in range(len(cases))))
    path = OUTPUT/'main.mojo'
    path.write_text('\n'.join(source) + '\n')
    check.run('mojo', 'build', '-O3', '-I', ROOT/'builds/deps', path, '-o', OUTPUT/'run')
    result = check.run(OUTPUT/'run', capture=True)
    (OUTPUT/'native.txt').write_text(result.stdout)
    actual = [float(x) for x in result.stdout.splitlines()]
    assert len(actual) == len(expected)
    for got, want, tolerance, label in zip(actual, expected, tolerances, labels):
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance / 10, err_msg=label)
    report = dict(cases=len(cases), batches=3*len(cases), comparisons=len(expected), revision=REVISION, seed=124,
                  tolerances={'float32': {'rtol': 2e-4, 'atol': 2e-5}, 'float64': {'rtol': 3e-8, 'atol': 3e-9}},
                  command='pixi run python tests/feature_selection/parity_from_model_incremental.py', compiler=check.run('mojo', '--version', capture=True).stdout.strip())
    (OUTPUT/'report.json').write_text(json.dumps(report, indent=2))
    print(f'PASS SelectFromModel incremental parity: {len(cases)} cases, {report["batches"]} batches, {len(expected)} comparisons')


if __name__ == '__main__':
    oracle() if '--oracle' in sys.argv else main()
