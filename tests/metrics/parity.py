"""Check native dense classification scores against the pinned local sklearn build."""
import argparse
from pathlib import Path
import subprocess
import warnings

import numpy as np
import sklearn
from sklearn import config_context
from sklearn.metrics import accuracy_score, hamming_loss, make_scorer, zero_one_loss
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[2]
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def verify_oracle():
    checkout = ROOT.parent / "scikit-learn"
    installed = Path(sklearn.__file__).resolve().parent
    paths = [
        "metrics/_classification.py",
        "utils/_array_api.py",
        "utils/multiclass.py",
        "utils/sparsefuncs.py",
        "utils/validation.py",
        "naive_bayes.py",
        "model_selection/_split.py",
        "model_selection/_validation.py",
    ]
    for path in paths:
        committed = subprocess.check_output(
            ["git", "-C", checkout, "show", f"{REVISION}:sklearn/{path}"]
        )
        if (installed / path).read_bytes() != committed:
            raise RuntimeError(f"Oracle {path} differs from local sklearn {REVISION}")


def native_output(executable):
    lines = subprocess.check_output([executable.resolve()], text=True).splitlines()
    output = {}
    for line in lines:
        name, *values = line.split()
        if name in output:
            raise AssertionError(f"Duplicate native result {name}")
        output[name] = [float(value) for value in values]
    return output


def classification_cases():
    truth = [0, 1, 2, 3]
    pred = [0, 2, 1, 3]
    weights = np.array([0.5, 1.0, 2.0, 4.25])
    extremes = np.array([2**63 - 2, 2**63 - 1], dtype=np.int64)
    indicators = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
    predicted_indicators = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 1]])
    strings = ["é", "", "a", "Ω"]
    predicted_strings = ["é", "x", "b", "Ω"]
    cases = {
        "multiclass": (truth, pred, None),
        "weighted_multiclass": (truth, pred, weights),
        "int64_identity": (extremes, extremes[::-1], None),
        "column": (np.array(truth, np.float32)[:, None], np.array([0, 1, 3, 3], np.float32), None),
        "strided": (np.array([4, 99, 2, 99, 1, 99, 0, 99], dtype=np.float64)[6::-2],
                    np.array([1, 1, 3, 4], dtype=np.float64), weights[::-1]),
        "multilabel": (indicators, predicted_indicators, None),
        "weighted_multilabel": (indicators, predicted_indicators, weights),
        "strings": (strings, predicted_strings, weights),
        "string_target": (np.array(strings)[:, None], predicted_strings, None),
        "list_int": (truth, pred, None),
    }
    expected = {}
    for name, (y_true, y_pred, weight) in cases.items():
        expected[name] = [
            accuracy_score(y_true, y_pred, sample_weight=weight),
            accuracy_score(y_true, y_pred, normalize=False, sample_weight=weight),
            zero_one_loss(y_true, y_pred, sample_weight=weight),
            zero_one_loss(y_true, y_pred, normalize=False, sample_weight=weight),
            hamming_loss(y_true, y_pred, sample_weight=weight),
        ]
    signed = np.array([1.0, -1.0, 1.0, -1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected["signed_hamming"] = [
            hamming_loss(indicators, predicted_indicators, sample_weight=signed),
            hamming_loss(indicators, indicators, sample_weight=signed),
        ]
    return expected


def cv_scores():
    X = np.array([0, 1, 7, 8, 2, 3, 6, 9, 0.5, 2.5, 7.5, 9.5])[:, None]
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
    with config_context(enable_metadata_routing=True):
        estimator = GaussianNB().set_fit_request(sample_weight=False)
        scorer = make_scorer(accuracy_score).set_score_request(sample_weight=True)
        return cross_val_score(
            estimator, X, y, cv=KFold(3), scoring=scorer, error_score="raise",
            params={"sample_weight": np.arange(1.0, 13.0)},
        )


def check_reference_errors():
    y = [0, 1]
    matrix = np.zeros((4, 3), dtype=np.int64)
    strings = np.array([["é", ""], ["a", "Ω"]])
    cases = [
        (accuracy_score, ([], []), {}),
        (accuracy_score, (y, [0, 1, 2, 3]), {}),
        (accuracy_score, ([0, 1, 2, 3], matrix), {}),
        (accuracy_score, (matrix, matrix[:, :2]), {}),
        (accuracy_score, ([0, 1, 2, 3], matrix[:, :1, None]), {}),
        (accuracy_score, ([0.5, 1.5], [0.5, 1.5]), {}),
        (accuracy_score, ([np.nan, 1.0], [np.nan, 1.0]), {}),
        (hamming_loss, (y, y), {"sample_weight": [np.nan, 1.0]}),
        (zero_one_loss, (y, y), {"normalize": False, "sample_weight": [0.0, 0.0]}),
        (accuracy_score, (y, y), {"sample_weight": [0.5, 1.0, 2.0, 4.25]}),
        (accuracy_score, (y, y), {"sample_weight": np.ones((2, 1))}),
        (hamming_loss, (strings, strings), {}),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for metric, args, kwargs in cases:
            with np.testing.assert_raises(ValueError):
                metric(*args, **kwargs)
    for metric in (accuracy_score, hamming_loss, zero_one_loss):
        with np.testing.assert_raises(ZeroDivisionError):
            metric(y, y, sample_weight=[4.0, -4.0])
    return len(cases) + 3


def compare(actual, expected):
    if actual.keys() != expected.keys():
        raise AssertionError(f"Native cases {actual.keys()} differ from {expected.keys()}")
    for name, values in expected.items():
        # Sequential Float64 accumulation can round differently from NumPy's reductions.
        np.testing.assert_allclose(
            actual[name], values, atol=1e-14, rtol=0, equal_nan=True, err_msg=name
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("classification", type=Path)
    parser.add_argument("scorer", type=Path)
    args = parser.parse_args()
    verify_oracle()
    cases = classification_cases()
    compare(native_output(args.classification), cases)
    errors = check_reference_errors()
    scores = cv_scores()
    compare(native_output(args.scorer), {"cv": scores})
    print(
        f"Matched {sum(map(len, cases.values()))} metric values, {errors} error cases "
        f"and {len(scores)} CV scores against sklearn {REVISION}"
    )


if __name__ == "__main__":
    main()
