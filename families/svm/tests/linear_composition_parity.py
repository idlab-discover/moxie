"""Compare the dense LinearSVC composition contract with pinned scikit-learn."""
from collections import defaultdict
from pathlib import Path
import subprocess

import numpy as np
from sklearn import calibration, feature_selection, model_selection, pipeline, svm
from sklearn.base import clone


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'


def main():
    revision = subprocess.check_output(
        ['git', '-C', REFERENCE, 'rev-parse', 'HEAD'], text=True,
    ).strip()
    assert revision == REVISION, revision
    for module in (calibration, feature_selection._from_model,
                   feature_selection._univariate_selection, model_selection._split,
                   pipeline, svm._base, svm._classes):
        source = Path(*module.__name__.split('.')).with_suffix('.py')
        expected = subprocess.check_output(['git', '-C', REFERENCE, 'show', f'{REVISION}:{source}'])
        assert Path(module.__file__).read_bytes() == expected, module.__file__

    row = np.arange(18)
    side = 2 * (row % 2) - 1
    X = np.column_stack((1.5 * side + (row % 5 - 2) / 4,
                         0.75 * side + (row % 7 - 3) / 4,
                         (row % 3 - 1) / 2))
    y = 2 + 5 * side
    row = np.arange(7)
    query = np.column_stack(((row - 3) / 2 + 0.125, (row % 3 - 1) / 4, row % 2 / 2))
    base = svm.LinearSVC(C=0.7, dual=False, tol=1e-9, random_state=0)
    calibrated = calibration.CalibratedClassifierCV(base, cv=3, ensemble=True).fit(X, y)
    selected = feature_selection.SelectFromModel(clone(base)).fit(X, y)
    composed = pipeline.make_pipeline(
        feature_selection.SelectKBest(feature_selection.f_classif, k=2),
        clone(base),
    ).fit(X, y)

    actual = defaultdict(list)
    output = subprocess.check_output([ROOT / 'builds/test_linear_composition'], text=True)
    for line in output.splitlines():
        name, value = line.split()
        actual[name].append(float(value))

    exact = {
        'calibration_class': calibrated.classes_,
        'calibration_prediction': calibrated.predict(query),
        'model_feature': selected.get_support(indices=True),
        'model_transform': selected.transform(query).ravel(),
        'pipeline_feature': composed[0].get_support(indices=True),
        'pipeline_prediction': composed.predict(query),
    }
    approximate = {
        'calibration_probability': calibrated.predict_proba(query).ravel(),
        'pipeline_decision': composed.decision_function(query),
    }
    assert actual.keys() == exact.keys() | approximate.keys(), actual.keys()
    for name, expected in exact.items():
        np.testing.assert_array_equal(actual[name], expected, err_msg=name)
    for name, expected in approximate.items():
        # LIBLINEAR and sigmoid fitting can stop on different iterations.
        np.testing.assert_allclose(actual[name], expected, rtol=2e-6, atol=2e-7, err_msg=name)
    comparisons = sum(map(len, actual.values()))
    print(f'PASS LinearSVC calibration, model selection and SelectKBest pipeline ({comparisons} comparisons)')


if __name__ == '__main__':
    main()
