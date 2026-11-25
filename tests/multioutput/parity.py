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
    from sklearn.linear_model import _logistic
    from sklearn.model_selection import _split, _validation
    from sklearn.model_selection import KFold
    from sklearn.naive_bayes import __file__ as naive_bayes_source
    from sklearn.neighbors import _classification
    from sklearn.tree import _classes
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.dummy import DummyRegressor, __file__ as dummy_source
    from sklearn.discriminant_analysis import __file__ as lda_source
    from sklearn.svm import _classes as svm_classes
    path = REFERENCE / 'sklearn/multioutput.py'
    for relative, source in [
        ('sklearn/multioutput.py', path),
        ('sklearn/linear_model/_logistic.py', Path(_logistic.__file__)),
        ('sklearn/model_selection/_split.py', Path(_split.__file__)),
        ('sklearn/model_selection/_validation.py', Path(_validation.__file__)),
        ('sklearn/naive_bayes.py', Path(naive_bayes_source)),
        ('sklearn/discriminant_analysis.py', Path(lda_source)),
        ('sklearn/dummy.py', Path(dummy_source)),
        ('sklearn/neighbors/_classification.py', Path(_classification.__file__)),
        ('sklearn/tree/_classes.py', Path(_classes.__file__)),
        ('sklearn/svm/_classes.py', Path(svm_classes.__file__)),
    ]:
        expected = subprocess.check_output(['git', '-C', str(REFERENCE), 'show', f'{REVISION}:{relative}'])
        if source.read_bytes() != expected:
            raise RuntimeError(f'Reference {source} differs from the recorded revision')
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
    expected.update(classifier_components(module, np))
    print(json.dumps(expected))


def verbose_lines(text):
    """[Chain] progress lines with the elapsed time masked (layout stays exact)."""
    import re
    return [re.sub(r'total=.*$', 'total=<time>', line) for line in text.splitlines() if line.startswith('[Chain]')]


def components(module, np):
    """Production-component scenario mirrored by test_components.mojo."""
    import warnings
    from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
    from sklearn.model_selection import KFold, StratifiedKFold
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
    rare = binary.astype(float)
    rare[:, 0] = -5
    rare[0, 0] = 9
    uneven = binary.copy()
    uneven[:, 0] = (i < 4) | (i == 7)
    label_cv = module.ClassifierChain(RidgeClassifier(), order=[0, 2, 1], cv=3)
    out['ComponentsClassifierChainIntegerCVLabels'] = label_cv.fit(X, uneven).predict(X).ravel().tolist()
    weighted_cv = module.RegressorChain(Ridge(alpha=0.5), order=[2, 0, 1], cv=KFold(3))
    out['ComponentsRegressorChainCVWeighted'] = weighted_cv.fit(X, Y, sample_weight=w).predict(X).ravel().tolist()
    fixed_folds = list(KFold(3, shuffle=True, random_state=7).split(X))
    fixed_regression = module.RegressorChain(Ridge(alpha=0.5), order=[2, 0, 1], cv=fixed_folds[::-1])
    out['ComponentsRegressorChainFixedCV'] = fixed_regression.fit(X, Y).predict(X).ravel().tolist()
    fixed_classification = module.ClassifierChain(
        RidgeClassifier(), order=[0, 2, 1], chain_method='decision_function', cv=fixed_folds,
    )
    out['ComponentsDecisionChainFixedCV'] = fixed_classification.fit(X, uneven).decision_function(X).ravel().tolist()
    class PerLinkKFold:
        def __init__(self, threshold):
            self.threshold = threshold

        def get_n_splits(self, features, target, groups=None):
            return 2 if target[0] > self.threshold else 3

        def split(self, features, target, groups=None):
            folds = self.get_n_splits(features, target)
            return KFold(folds, shuffle=True, random_state=7 + features.shape[1]).split(features)

    custom_regression = module.RegressorChain(Ridge(alpha=0.5), order=[2, 0, 1], cv=PerLinkKFold(-6.0))
    out['ComponentsRegressorChainCustomCV'] = custom_regression.fit(X, Y).predict(X).ravel().tolist()
    custom_classification = module.ClassifierChain(
        LogisticRegression(tol=1e-10, max_iter=1000),
        chain_method='predict_proba', cv=PerLinkKFold(0.5),
    ).fit(X, uneven)
    out['ComponentsLogisticChainCustomCVProba'] = custom_classification.predict_proba(X).ravel().tolist()
    out['ComponentsLogisticChainCustomCVLabels'] = custom_classification.predict(X).ravel().tolist()
    shuffled_cv = module.ClassifierChain(
        RidgeClassifier(), order=[0, 2, 1], chain_method='decision_function',
        cv=StratifiedKFold(3, shuffle=True, random_state=7),
    )
    out['ComponentsDecisionChainShuffledCV'] = shuffled_cv.fit(X, uneven).decision_function(X).ravel().tolist()
    for suffix, cv in [('KFold', KFold(3)), ('IntegerCV', 3)]:
        regression = module.RegressorChain(Ridge(alpha=0.5), order=[2, 0, 1], cv=cv)
        out['ComponentsRegressorChain' + suffix] = regression.fit(X, Y).predict(X).ravel().tolist()
        classification = module.ClassifierChain(
            RidgeClassifier(), order=[0, 2, 1], chain_method='decision_function', cv=cv,
        )
        out['ComponentsDecisionChain' + suffix] = classification.fit(X, uneven).decision_function(X).ravel().tolist()
        forest_cv = module.ClassifierChain(
            RandomForestClassifier(n_estimators=5, random_state=0), chain_method='predict_proba', cv=cv,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The least populated class in y has only')
            warnings.filterwarnings('ignore', message='Number of classes in training fold')
            out['ComponentsForestChain' + suffix] = forest_cv.fit(X32, rare).predict_proba(X32).ravel().tolist()
    logistic = LogisticRegression(tol=1e-10, max_iter=1000)
    independent = module.MultiOutputClassifier(logistic).fit(X, multiclass)
    out['ComponentsLogisticIndependentLabels'] = independent.predict(X).ravel().tolist()
    out['ComponentsLogisticIndependentProba'] = independent.predict_proba(X)[2].ravel().tolist()
    for suffix, cv in [('', None), ('CV', 3)]:
        logistic_chain = module.ClassifierChain(
            logistic, order=[2, 0, 1], chain_method='predict_proba', cv=cv,
        ).fit(X, binary)
        out['ComponentsLogisticChain' + suffix + 'Labels'] = logistic_chain.predict(X).ravel().tolist()
        out['ComponentsLogisticChain' + suffix + 'Proba'] = logistic_chain.predict_proba(X).ravel().tolist()
    return out


def classifier_components(module, np):
    """Single-target probability, weight and List-label interoperability."""
    import warnings

    from sklearn import config_context
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.dummy import DummyClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    i = np.arange(12)
    X = (i * i + 0.1 * i).reshape(-1, 1)
    X32 = X.astype(np.float32)
    multiclass = np.column_stack([np.where(i < 6, -3., 4.), np.where(i < 4, 8., np.where(i < 8, 11., 19.))])
    binary = multiclass.copy()
    binary[binary[:, 1] == 19., 1] = 11.
    tree = DecisionTreeClassifier(max_depth=2, random_state=17)
    wrapped_tree = module.MultiOutputClassifier(tree).fit(X32, multiclass)
    tree_chain = module.ClassifierChain(tree, chain_method='predict_proba', cv=3).fit(X32, binary)
    rare = binary.copy()
    rare[:, 0] = np.where(i == 11, 4., -3.)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The least populated class in y has only')
        warnings.filterwarnings('ignore', message='Number of classes in training fold')
        rare_chain = module.ClassifierChain(tree, chain_method='predict_proba', cv=3).fit(X32, rare)
    constant = np.zeros_like(X32)
    log_chain = module.ClassifierChain(tree, chain_method='predict_log_proba', cv=3).fit(constant, binary)
    knn = module.MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3, algorithm='brute')).fit(X, multiclass)
    weights = (i % 3 + 1).astype(float)
    bayes = module.MultiOutputClassifier(GaussianNB()).fit(X, multiclass, sample_weight=weights)
    with config_context(enable_metadata_routing=True):
        bayes_chain = module.ClassifierChain(
            GaussianNB().set_fit_request(sample_weight=True), chain_method='predict_proba',
        ).fit(X, binary, sample_weight=weights)
    integer_targets = multiclass.astype(np.int64)
    list_bayes = module.MultiOutputClassifier(GaussianNB()).fit(X, integer_targets, sample_weight=weights)
    incremental = module.MultiOutputClassifier(GaussianNB())
    incremental.partial_fit(X[:6], integer_targets[:6], classes=list_bayes.classes_)
    incremental.partial_fit(X[6:], integer_targets[6:])
    list_chain = module.ClassifierChain(GaussianNB(), chain_method='predict_proba', cv=3).fit(
        X, binary.astype(np.int64)
    )
    lda = module.MultiOutputClassifier(LinearDiscriminantAnalysis()).fit(X, multiclass)
    dummy = module.MultiOutputClassifier(DummyClassifier()).fit(
        X, integer_targets, sample_weight=weights
    )
    dummy32 = module.MultiOutputClassifier(DummyClassifier()).fit(
        X32, multiclass.astype(np.float32), sample_weight=weights
    )
    dummy_chain = module.ClassifierChain(
        DummyClassifier(), order=[1, 0], chain_method='predict_proba', cv=3,
    ).fit(X, binary)
    svc = module.MultiOutputClassifier(SVC(kernel='linear')).fit(X, integer_targets)
    svc_chain = module.ClassifierChain(
        SVC(kernel='linear'), chain_method='decision_function', cv=3,
    ).fit(X, binary.astype(np.int64))
    return {
        'ComponentsDummyWeightedLabels': dummy.predict(X).ravel().tolist(),
        'ComponentsDummyWeightedProba': dummy.predict_proba(X)[1].ravel().tolist(),
        'ComponentsDummy32WeightedLabels': dummy32.predict(X32).ravel().tolist(),
        'ComponentsDummy32WeightedProba': dummy32.predict_proba(X32)[1].ravel().tolist(),
        'ComponentsDummyChainCVLabels': dummy_chain.predict(X).ravel().tolist(),
        'ComponentsDummyChainCVProba': dummy_chain.predict_proba(X).ravel().tolist(),
        'ComponentsSVCLabels': svc.predict(X).ravel().tolist(),
        'ComponentsSVCChainCVLabels': svc_chain.predict(X).ravel().tolist(),
        'ComponentsSVCChainCVDecision': svc_chain.decision_function(X).ravel().tolist(),
        'ComponentsGaussianNBListLabels': list_bayes.predict(X).ravel().tolist(),
        'ComponentsGaussianNBListProba': list_bayes.predict_proba(X)[1].ravel().tolist(),
        'ComponentsGaussianNBListIncrementalLabels': incremental.predict(X).ravel().tolist(),
        'ComponentsGaussianNBListIncrementalProba': incremental.predict_proba(X)[1].ravel().tolist(),
        'ComponentsGaussianNBListChainCVLabels': list_chain.predict(X).ravel().tolist(),
        'ComponentsGaussianNBListChainCVProba': list_chain.predict_proba(X).ravel().tolist(),
        'ComponentsLDAListLabels': lda.predict(X).ravel().tolist(),
        'ComponentsLDAListProba': lda.predict_proba(X)[1].ravel().tolist(),
        'ComponentsTreeLabels': wrapped_tree.predict(X32).ravel().tolist(),
        'ComponentsTreeProba': wrapped_tree.predict_proba(X32)[1].ravel().tolist(),
        'ComponentsTreeChainCVProba': tree_chain.predict_proba(X32).ravel().tolist(),
        'ComponentsTreeChainCVLabels': tree_chain.predict(X32).ravel().tolist(),
        'ComponentsTreeChainMissingClassProba': rare_chain.predict_proba(X32).ravel().tolist(),
        'ComponentsTreeChainLogResponseProba': log_chain.predict_proba(constant).ravel().tolist(),
        'ComponentsKNNLabels': knn.predict(X).ravel().tolist(),
        'ComponentsKNNProba': knn.predict_proba(X)[1].ravel().tolist(),
        'ComponentsGaussianNBLabels': bayes.predict(X).ravel().tolist(),
        'ComponentsGaussianNBProba': bayes.predict_proba(X)[1].ravel().tolist(),
        'ComponentsGaussianNBChainProba': bayes_chain.predict_proba(X).ravel().tolist(),
    }

if __name__ == '__main__':
    if '--oracle' in sys.argv:
        oracle()
    else:
        # The oracle runtime is validation-only and can be supplied explicitly.
        python = os.environ.get('MOXIE_REFERENCE_PYTHON', sys.executable)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        reference = json.loads(subprocess.check_output([python, __file__, '--oracle'], env=env, text=True))
        output = ''.join(subprocess.check_output([str(ROOT/'builds/tests/multioutput'/name)], env=env, text=True)
                         for name in ('test_estimators', 'test_components', 'test_classifier_components'))
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
            elif name.startswith('ComponentsLogistic') and name.endswith('Proba'):
                # Use the linear-model family's Float64 probability bound for
                # independently converged L-BFGS fits, including every chain link.
                matches = all(math.isclose(a, b, rel_tol=0.0, abs_tol=2e-6)
                              for a, b in zip(actual[name], reference[name]))
            elif name.startswith('ComponentsGaussianNB') and name.endswith('Proba'):
                # Weighted moments and log normalization use different reduction orders.
                matches = all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                              for a, b in zip(actual[name], reference[name]))
            elif name == 'ComponentsLDAListProba':
                # SVD and probability normalization can differ in rounding.
                matches = all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
                              for a, b in zip(actual[name], reference[name]))
            elif name == 'ComponentsSVCChainCVDecision':
                # Match the standalone SVM solver tolerance.
                matches = all(math.isclose(a, b, rel_tol=3e-5, abs_tol=3e-6)
                              for a, b in zip(actual[name], reference[name]))
            elif name.startswith(('ComponentsTree', 'ComponentsKNN', 'ComponentsDummy')):
                matches = actual[name] == reference[name]
            elif name.startswith('Components') and not name.endswith(('Labels', 'Order')) and 'Forest' not in name:
                # Float64 Ridge solves differ from LAPACK in operation order.
                # Weighted chains also propagate those differences between links.
                # Labels, orders and seeded forest probabilities stay exact.
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
