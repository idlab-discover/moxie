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
    print(json.dumps(expected))

if __name__ == '__main__':
    if '--oracle' in sys.argv:
        oracle()
    else:
        # The oracle runtime is validation-only and can be supplied explicitly.
        python = os.environ.get('MOXIE_REFERENCE_PYTHON', sys.executable)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        reference = json.loads(subprocess.check_output([python, __file__, '--oracle'], env=env, text=True))
        output = subprocess.check_output([str(ROOT/'builds/tests/multioutput/test_estimators')], env=env, text=True)
        actual = {}
        for line in output.splitlines():
            if line.startswith('PARITY '):
                _, name, *values = line.split()
                actual[name] = [float(v) for v in values]
        assert actual.keys() == reference.keys(), (actual, reference)
        for name in reference:
            assert len(actual[name]) == len(reference[name]), name
            # Scores perform floating-point reductions; discrete predictions,
            # one-neighbor probabilities and seeded permutations must be exact.
            if name.endswith('Score'):
                matches = all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                              for a, b in zip(actual[name], reference[name]))
            else:
                matches = actual[name] == reference[name]
            assert matches, (name, actual[name], reference[name])
        (ROOT/'builds/multioutput-parity.json').write_text(json.dumps({'reference_revision': REVISION, 'expected': reference, 'actual': actual}, indent=2)+'\n')
        print(f'PASS: {len(reference)} sklearn comparisons for predictions, probabilities, scores, weights and seeded order')
