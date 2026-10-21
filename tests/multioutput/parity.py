"""Compare native wrapper predictions with the recorded sklearn source."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'

def oracle():
    import numpy as np
    import sklearn.multioutput
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.dummy import DummyRegressor
    path = REFERENCE / 'sklearn/multioutput.py'
    expected = subprocess.check_output(['git', '-C', str(REFERENCE), 'show', REVISION + ':sklearn/multioutput.py'])
    if path.read_bytes() != expected:
        raise RuntimeError('Reference multioutput.py differs from the recorded revision')
    spec = importlib.util.spec_from_file_location('sklearn.multioutput', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.column_stack([np.arange(4)*10, 3-np.arange(4)]).astype(float)
    labels = np.column_stack([np.arange(4)%2, (np.arange(4)//2)%2])
    query = np.array([[0.4], [1.7], [2.6]])
    expected = {}
    for name, base, target in [
        ('MultiOutputRegressor', KNeighborsRegressor(n_neighbors=1), y),
        ('RegressorChain', KNeighborsRegressor(n_neighbors=1), y),
        ('MultiOutputClassifier', KNeighborsClassifier(n_neighbors=1), labels),
        ('ClassifierChain', KNeighborsClassifier(n_neighbors=1), labels),
    ]:
        options = {'order': [1, 0]} if name.endswith('Chain') else {}
        model = getattr(module, name)(base, **options).fit(X, target)
        expected[name] = model.predict(query).ravel().tolist()
        if name == 'MultiOutputClassifier':
            for j, probability in enumerate(model.predict_proba(query)):
                expected[name + 'Proba' + str(j)] = probability.ravel().tolist()
            changed_labels = labels.copy()
            changed_labels[0] = [1, 1]
            expected[name + 'Score'] = [model.score(X, changed_labels)]
        if name == 'MultiOutputRegressor':
            changed = y.copy()
            changed[:, 0] += 1
            expected[name + 'Score'] = [model.score(X, changed)]
            expected[name + 'WeightedScore'] = [model.score(X, changed, sample_weight=np.arange(1, 5))]
        if name == 'ClassifierChain':
            expected[name + 'Proba'] = model.predict_proba(query).ravel().tolist()
            with np.errstate(divide='ignore'):
                expected[name + 'LogProba'] = model.predict_log_proba(query).ravel().tolist()
            changed_labels = labels.copy()
            changed_labels[0] = [1, 1]
            expected[name + 'WeightedScore'] = [model.score(X, changed_labels, sample_weight=np.arange(1, 5))]
        if name.endswith('Chain'):
            model = getattr(module, name)(base, order=[1, 0], cv=KFold(n_splits=2)).fit(X, target)
            expected[name + 'CV'] = model.predict(query).ravel().tolist()
    probability_chain = module.ClassifierChain(KNeighborsClassifier(n_neighbors=1), chain_method='predict_proba').fit(X, labels)
    expected['ClassifierChainProbabilityResponse'] = probability_chain.predict(query).ravel().tolist()
    expected['ClassifierChainProbabilityResponseProba'] = probability_chain.predict_proba(query).ravel().tolist()
    random_chain = module.ClassifierChain(KNeighborsClassifier(n_neighbors=1), order='random', random_state=7).fit(X, labels)
    expected['ClassifierChainRandomOrder'] = random_chain.order_.tolist()
    expected['ClassifierChainRandom'] = random_chain.predict(query).ravel().tolist()
    weighted = module.MultiOutputRegressor(DummyRegressor()).fit(X, y, sample_weight=np.arange(1, 5))
    expected['MultiOutputRegressorWeightedFit'] = weighted.predict(query).ravel().tolist()
    expected.update(components(module, np))
    print(json.dumps(expected))


def verbose_lines(text):
    """[Chain] progress lines with the elapsed time masked (layout stays exact)."""
    import re
    return [re.sub(r'total=.*$', 'total=<time>', line) for line in text.splitlines() if line.startswith('[Chain]')]


def components(module, np):
    """Production-component scenario mirrored by test_components.mojo."""
    from sklearn.linear_model import Ridge, RidgeClassifier
    i = np.arange(12)
    X = np.array([[((r * (j + 3) + 2 * j) % 7) - 3.0 + 0.1 * r for j in range(3)] for r in i])
    Y = np.column_stack([0.5 * (i % 3) + X @ np.array([k + 1 - j for j in range(3)], dtype=float) for k in range(3)])
    binary = np.column_stack([(X[:, 0] + X[:, 1] > 0), i % 2, X[:, 2] > X[:, 0]]).astype(np.int64)
    multiclass = np.column_stack([binary[:, 0], i % 2, i % 3]).astype(np.int64)
    w = (i % 4 + 1).astype(float)
    out = {}
    regressor = module.MultiOutputRegressor(Ridge(alpha=0.5)).fit(X, Y)
    out['ComponentsRidge'] = regressor.predict(X).ravel().tolist()
    out['ComponentsRidgeScore'] = [regressor.score(X, Y)]
    regressor.fit(X, Y, sample_weight=w)
    out['ComponentsRidgeWeighted'] = regressor.predict(X).ravel().tolist()
    out['ComponentsRidgeWeightedScore'] = [regressor.score(X, Y, sample_weight=w)]
    out['ComponentsRidgeAlpha10'] = module.MultiOutputRegressor(Ridge(alpha=10.0)).fit(X, Y).predict(X).ravel().tolist()
    X32, Y32 = X.astype(np.float32), Y.astype(np.float32)
    narrow = module.MultiOutputRegressor(Ridge(alpha=0.5)).fit(X32, Y32, sample_weight=w)
    out['ComponentsRidge32Weighted'] = narrow.predict(X32).ravel().tolist()
    chain32 = module.RegressorChain(Ridge(alpha=0.5), order=[2, 0, 1]).fit(X32, Y32, sample_weight=w)
    out['ComponentsRegressorChain32Weighted'] = chain32.predict(X32).ravel().tolist()
    import contextlib, io
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        chain = module.RegressorChain(Ridge(alpha=0.5), order=[2, 0, 1], verbose=True).fit(X, Y)
    out['ComponentsChainVerbose'] = verbose_lines(log.getvalue())
    out['ComponentsRegressorChain'] = chain.predict(X).ravel().tolist()
    out['ComponentsRegressorChainScore'] = [chain.score(X, Y)]
    decision = module.ClassifierChain(RidgeClassifier(), order=[1, 0, 2], chain_method='decision_function').fit(X, binary)
    out['ComponentsDecisionChainLabels'] = decision.predict(X).ravel().tolist()
    out['ComponentsDecisionChain'] = decision.decision_function(X).ravel().tolist()
    out['ComponentsDecisionChainScore'] = [decision.score(X, binary, sample_weight=w)]
    labels = module.ClassifierChain(RidgeClassifier(), order=[2, 0, 1]).fit(X, multiclass)
    out['ComponentsLabelChainLabels'] = labels.predict(X).ravel().tolist()
    independent = module.MultiOutputClassifier(RidgeClassifier()).fit(X, multiclass)
    out['ComponentsIndependentLabels'] = independent.predict(X).ravel().tolist()
    out['ComponentsIndependentScore'] = [independent.score(X, multiclass)]
    random = module.ClassifierChain(RidgeClassifier(), order='random', random_state=3).fit(X, multiclass)
    out['ComponentsRandomOrder'] = random.order_.tolist()
    out['ComponentsRandomLabels'] = random.predict(X).ravel().tolist()
    from sklearn.ensemble import RandomForestClassifier
    X32 = X.astype(np.float32)
    forest = module.MultiOutputClassifier(RandomForestClassifier(n_estimators=5, random_state=0))
    out['ComponentsForestProba2'] = forest.fit(X32, multiclass.astype(float)).predict_proba(X32)[2].ravel().tolist()
    forest_chain = module.ClassifierChain(RandomForestClassifier(n_estimators=5, random_state=0), chain_method='predict_proba')
    out['ComponentsForestChainProba'] = forest_chain.fit(X32, binary.astype(float)).predict_proba(X32).ravel().tolist()
    return out

if __name__ == '__main__':
    if '--oracle' in sys.argv:
        oracle()
    else:
        # The oracle runtime is validation-only and can be supplied explicitly.
        python = os.environ.get('MOXIE_REFERENCE_PYTHON', sys.executable)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        reference = json.loads(subprocess.check_output([python, __file__, '--oracle'], env=env, text=True))
        output = ''.join(subprocess.check_output([str(ROOT/'builds/tests/multioutput'/name)], env=env, text=True)
                         for name in ('test_estimators', 'test_components'))
        actual = {'ComponentsChainVerbose': verbose_lines(output)}
        for line in output.splitlines():
            if line.startswith('PARITY '):
                _, name, *values = line.split()
                actual[name] = [float(v) for v in values]
        assert actual.keys() == reference.keys(), (actual, reference)
        assert actual['ComponentsChainVerbose'] == reference['ComponentsChainVerbose'], (
            actual['ComponentsChainVerbose'], reference['ComponentsChainVerbose'])
        for name in reference:
            assert len(actual[name]) == len(reference[name]), name
            if name == 'ComponentsChainVerbose':
                continue
            # Scores perform floating-point reductions; discrete predictions,
            # one-neighbor probabilities and seeded permutations must be exact.
            if name.startswith('Components') and name.endswith('32Weighted'):
                # Float32 Ridge solves: both sides round to float32 (eps 1.2e-7);
                # values reach ~15, so allow 2e-5 absolute and relative error.
                matches = all(math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-5)
                              for a, b in zip(actual[name], reference[name]))
            elif name.startswith('Components') and not name.endswith(('Labels', 'Order')) and 'Forest' not in name:
                # Float64 Ridge solves differ from LAPACK in operation order
                # (observed <= 6e-15); labels, orders and seeded forest
                # probabilities stay exact.
                matches = all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                              for a, b in zip(actual[name], reference[name]))
            elif name.endswith('Score'):
                matches = all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                              for a, b in zip(actual[name], reference[name]))
            else:
                matches = actual[name] == reference[name]
            assert matches, (name, actual[name], reference[name])
        (ROOT/'builds/multioutput-parity.json').write_text(json.dumps({'reference_revision': REVISION, 'expected': reference, 'actual': actual}, indent=2)+'\n')
        print(f'PASS: {len(reference)} sklearn comparisons for predictions, probabilities, scores, weights and seeded order')
