"""Check native dense classification scores against the pinned local sklearn build."""
import argparse
from pathlib import Path
import subprocess
import warnings

import numpy as np
import sklearn
from sklearn import config_context
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    hamming_loss,
    make_scorer,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    zero_one_loss,
)
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
        "utils/extmath.py",
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


def native_output(executable, expected_warnings=None):
    result = subprocess.run([executable.resolve()], text=True, capture_output=True, check=True)
    if expected_warnings is not None:
        np.testing.assert_equal(result.stderr.splitlines(), expected_warnings)
    lines = result.stdout.splitlines()
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


def confusion_cases():
    truth = [2, 0, 2, 2, 0, 1]
    predicted = [0, 0, 2, 2, 0, 2]
    weights = np.array([0.5, 1.0, 0.0, 1.0, 1.5, 3.0])
    strings = ["é", "", "a", "Ω"]
    predicted_strings = ["é", "x", "a", "Ω"]
    column = np.array([-0.0, 3, 0.0], dtype=np.float32)[:, None]
    strided = np.array([3, 99, -0.0, 99, 0.0], dtype=np.float32)[::-2]
    extremes = np.array([2**63 - 2, 2**63 - 1], dtype=np.int64)
    signed = [1.0, -1.0]
    overflow = ([0, 0, 0, 1], [0, 0, 1, 0])
    huge_weights = [1.5e308, 1.5e308, 0.25, 0.5]
    matrices = {
        "confusion_counts": (truth, predicted, {}),
        "confusion_subset": (truth, predicted, {"labels": [2, 0, 3]}),
        "confusion_duplicate": (truth, predicted, {"labels": [2, 0, 2]}),
        "confusion_no_pairs": (truth, predicted, {"labels": [1]}),
        "confusion_weights": (truth, predicted, {"sample_weight": weights[::-1]}),
        "confusion_true": (truth, predicted, {
            "normalize": "true", "labels": [2, 0, 1, 3], "sample_weight": weights
        }),
        "confusion_pred": (truth, predicted, {"normalize": "pred"}),
        "confusion_all": (truth, predicted, {"normalize": "all"}),
        "confusion_optional_weights": (truth, predicted, {}),
        "confusion_string_order": (strings, predicted_strings, {}),
        "confusion_string_target": (np.array(strings)[:, None], predicted_strings, {
            "labels": ["Ω", "", "x", "é"], "sample_weight": [1.0, 2.0, 3.0, 4.0]
        }),
        "confusion_strided": (column, strided, {}),
        "confusion_signed_zero": (column, strided, {
            "labels": np.array([3, -0.0], dtype=np.float32), "normalize": "true"
        }),
        "confusion_int64": (extremes, extremes[::-1], {}),
        "confusion_cancelling": ([0, 0], [0, 1], {
            "normalize": "true", "sample_weight": signed
        }),
        "confusion_overflow": (*overflow, {"sample_weight": huge_weights}),
        "confusion_normalized_overflow": (*overflow, {
            "normalize": "true", "sample_weight": huge_weights
        }),
    }
    scores = {
        "balanced_overflow": [(*overflow, {"sample_weight": huge_weights})],
        "balanced_weighted": [
            (truth, predicted, {"sample_weight": weights, "adjusted": adjusted})
            for adjusted in (False, True)
        ],
        "balanced_strings": [
            (np.array(strings)[:, None], predicted_strings, {}),
            (strings, predicted_strings, {"adjusted": True}),
        ],
        "balanced_int64": [(extremes, extremes[::-1], {"adjusted": True})],
        "balanced_single": [
            ([0, 0], [0, 0], {"adjusted": True}),
            ([0, 0], [0, 1], {"adjusted": True}),
        ],
        "balanced_cancelling": [
            ([0, 0], [0, 1], {"sample_weight": signed, "adjusted": True}),
            ([0, 0], [1, 1], {"sample_weight": signed}),
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        result = {
            name: confusion_matrix(y, pred, **kwargs).ravel()
            for name, (y, pred, kwargs) in matrices.items()
        }
        result.update({
            name: [balanced_accuracy_score(y, pred, **kwargs) for y, pred, kwargs in cases]
            for name, cases in scores.items()
        })
    return result


def check_confusion_reference_errors():
    y = [0, 0]
    unseen = [0, 1]
    indicator = np.zeros((3, 2), dtype=np.int64)
    cases = [
        (confusion_matrix, ([], []), {}),
        (confusion_matrix, (y, [0, 1, 2]), {}),
        (confusion_matrix, (y, unseen), {"labels": []}),
        (confusion_matrix, (y, unseen), {"labels": [1]}),
        (confusion_matrix, (y, unseen), {"normalize": "rows"}),
        (confusion_matrix, (y, unseen), {"sample_weight": [1.0]}),
        (confusion_matrix, (y, unseen), {"sample_weight": [0.0, 0.0]}),
        (confusion_matrix, (y, unseen), {"sample_weight": np.ones((2, 1))}),
        (balanced_accuracy_score, (y, unseen), {"sample_weight": [1.0, np.inf]}),
        (confusion_matrix, (indicator, indicator), {}),
        (confusion_matrix, ([0.5, 1.5], [0.5, 1.5]), {}),
    ]
    for metric, args, kwargs in cases:
        with np.testing.assert_raises(ValueError):
            metric(*args, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.testing.assert_raises(ZeroDivisionError):
            balanced_accuracy_score(y, [1, 1], adjusted=True, sample_weight=[1.0, -1.0])
    return len(cases) + 1


def cv_scores(metric):
    X = np.array([0, 1, 7, 8, 2, 3, 6, 9, 0.5, 2.5, 7.5, 9.5])[:, None]
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
    with config_context(enable_metadata_routing=True):
        estimator = GaussianNB().set_fit_request(sample_weight=False)
        scorer = make_scorer(metric).set_score_request(sample_weight=True)
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


def precision_recall_cases():
    truth = [0, 1, 2, 2, 0, 3]
    predicted = [0, 2, 1, 0, 0, 2]
    weights = [0.5, 1.0, 2.0, 4.25, 0.0, -0.25]
    labels = [3, 0, 4, 2, 0]
    binary = [0, 1, 1, 0, 1, 1]
    binary_prediction = [0, 0, 1, 1, 1, 0]
    negative = [0] * 6
    strings = ["Ω", "cat", "Ω", "cat"]
    string_predictions = ["cat", "cat", "Ω", "Ω"]
    indicators = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
    indicator_predictions = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 0]])
    indicator_weights = [1, 0, 2, -1]
    extremes = np.array([2**63 - 2, 2**63 - 1, 2**63 - 2], dtype=np.int64)

    def scores(y, p, *, average=None, **kwargs):
        result = precision_recall_fscore_support(y, p, average=average, **kwargs)
        if average is None:
            return np.concatenate(result)
        return result[:3]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases = {
            "default": np.concatenate(precision_recall_fscore_support(truth, predicted)),
            "selected": scores(
                truth, predicted, labels=labels, beta=2, sample_weight=weights,
                zero_division=np.nan,
            ),
            "binary": [
                precision_score(binary, binary_prediction, sample_weight=weights),
                recall_score(binary, binary_prediction, sample_weight=weights),
                f1_score(binary, binary_prediction, sample_weight=weights),
                fbeta_score(binary, binary_prediction, beta=0, sample_weight=weights),
                fbeta_score(binary, binary_prediction, beta=np.inf, sample_weight=weights),
            ],
            "binary_zero": scores(
                binary, binary_prediction, average="binary", pos_label=0, labels=[99],
                sample_weight=weights,
            ),
            "absent_positive": [f1_score([0, 0], [0, 0], zero_division=1)],
            "empty_selection": scores(truth, predicted, labels=[]),
            "cancelled_support": [
                f1_score([0, 1], [0, 0], average="weighted", sample_weight=[1., -1.],
                         zero_division=1),
            ],
            "precision_warning": [precision_score(binary, negative)],
            "recall_warning": [recall_score(negative, binary_prediction)],
            "fscore_warning": [f1_score([0, 0], [0, 0])],
            "all_undefined": [recall_score([0, 0], [0, 0], zero_division=np.nan)],
            "string_column": scores(np.array(strings)[:, None], string_predictions),
            "string_list": scores(strings, string_predictions, labels=["Ω", "absent", "cat"],
                                  zero_division=1),
            "string_binary": [
                precision_score(strings, string_predictions, pos_label="Ω"),
                recall_score(strings, string_predictions, pos_label="Ω"),
                f1_score(strings, string_predictions, pos_label="Ω"),
                fbeta_score(strings, string_predictions, pos_label="Ω", beta=2),
            ],
            "string_averages": [
                precision_score(strings, string_predictions, average="macro"),
                recall_score(strings, string_predictions, average="macro"),
                f1_score(strings, string_predictions, average="micro"),
                fbeta_score(strings, string_predictions, average="weighted", beta=2),
            ],
            "strided_numeric": scores(
                [0., 2., -0., 1.], [1., 2., 0., 1.], labels=[-0., 2., 1.],
                sample_weight=[4., 2., 1., 0.5],
            ),
            "int64_identity": scores(extremes, extremes[[1, 0, 2]]),
            "float_column": [
                precision_score([[0.], [1.], [1.]], [0., 1., 0.]),
                recall_score([[0.], [1.], [1.]], [0., 1., 0.]),
                f1_score([[0.], [1.], [1.]], [0., 1., 0.]),
                fbeta_score([[0.], [1.], [1.]], [0., 1., 0.], beta=2),
            ],
            "multilabel": scores(indicators, indicator_predictions),
            "multilabel_signed": scores(
                indicators, indicator_predictions, labels=[2, 0],
                sample_weight=indicator_weights, zero_division=1,
            ),
            "empty_multilabel_samples": scores(
                indicators, indicator_predictions, labels=[], average="samples",
                zero_division=1,
            ),
            "signed_samples": scores(
                indicators, indicator_predictions, labels=[2, 0], average="samples",
                sample_weight=indicator_weights, zero_division=1,
            ),
        }
        for average in ("micro", "macro", "weighted"):
            cases["selected_" + average] = scores(
                truth, predicted, labels=labels, beta=0.5, average=average,
                sample_weight=weights, zero_division=np.nan,
            )
        for average in ("micro", "macro", "weighted", "samples"):
            cases["multilabel_" + average] = scores(
                indicators, indicator_predictions, average=average, labels=[2, 0, 2],
                sample_weight=indicator_weights, zero_division=np.nan,
            )
    return cases


def check_precision_recall_reference_errors():
    from sklearn.utils._param_validation import InvalidParameterError

    binary = [0, 1]
    indicators = np.zeros((4, 3), dtype=np.int64)
    cases = [
        (indicators, indicators, {}),
        (indicators, indicators, {"average": "macro", "labels": [-1]}),
        (indicators, indicators, {"average": "macro", "labels": [3]}),
        (indicators, indicators[:, :2], {"average": "macro"}),
        (indicators, np.zeros(4), {"average": "macro"}),
        (binary, [2, 3], {}),
        (binary, binary, {"pos_label": 3}),
        (binary, binary, {"average": "samples"}),
        (binary, binary, {"average": "unknown"}),
        (binary, binary, {"zero_division": 2.0}),
        (binary, binary, {"sample_weight": [0, 0]}),
        (binary, binary, {"sample_weight": [1]}),
        (binary, binary, {"sample_weight": [1, np.inf]}),
        ([], [], {}),
        (binary, [0], {}),
        (["cat", "dog"], ["cat", "dog"], {}),
        ([0.5, 0.75], [0.5, 0.75], {"average": "macro"}),
    ]
    for truth, predicted, kwargs in cases:
        with np.testing.assert_raises(ValueError):
            f1_score(truth, predicted, **kwargs)
    with np.testing.assert_raises(InvalidParameterError):
        fbeta_score(binary, binary, beta=-1.0)
    with np.testing.assert_raises(OverflowError):
        fbeta_score(binary, binary, beta=1e200)
    return len(cases) + 2


def precision_recall_warnings():
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        precision_recall_fscore_support([0, 1, 2, 2, 0, 3], [0, 2, 1, 0, 0, 2], average=None)
        precision_score([0, 1], [0, 0])
        recall_score([0, 0], [0, 1])
        f1_score([0, 0], [0, 0])
    return [
        f"{entry.category.__name__}: "
        + str(entry.message).split(" and being set to 0.0")[0]
        + " and being set to 0.0"
        for entry in log
    ]


def check_reports(executable):
    truth = [0, 1, 2, 2, 0, 1]
    predicted = [0, 0, 2, 1, 0, 1]
    weights = [0.5, 1.0, 2.0, 4.25, 0.0, -0.25]
    strings = ["Ω", "chat", "Ω", "chat"]
    predictions = ["chat", "chat", "Ω", "Ω"]
    extremes = np.array([2**63 - 2, 2**63 - 1, 2**63 - 2], dtype=np.int64)
    indicators = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
    predicted_indicators = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 1]])
    subset = dict(labels=[2, 0, 2, 3], target_names=["left", "right", "left", "unseen"],
                  sample_weight=weights, zero_division=np.nan)
    cases = [
        ("TEXT", "multiclass", truth, predicted, {}),
        ("DICT", "multiclass", truth, predicted, {}),
        ("TEXT", "subset", truth, predicted, dict(subset, digits=5)),
        ("DICT", "subset", truth, predicted, subset),
        ("TEXT", "short_names", truth, predicted,
         dict(labels=[3, 2, 1, 0], target_names=["unseen", "second"], zero_division=1)),
        ("TEXT", "long_names", truth, predicted,
         dict(labels=[3, 2, 1, 0], target_names=["a", "b", "c", "d", "a long unused heading"],
              zero_division=0)),
        ("DICT", "superset", truth, predicted, dict(labels=[3, 2, 1, 0], zero_division=0)),
        ("DICT", "empty_selection", truth, predicted, dict(labels=[], zero_division=0)),
        ("DICT", "warnings", [0, 1], [0, 2], {}),
        ("TEXT", "ties", [0, 0], [0, 1], dict(digits=0, zero_division=0)),
        ("DICT", "cancelled_weights", [0, 1], [0, 0],
         dict(sample_weight=[1.0, -1.0], zero_division=1)),
        ("TEXT", "string_column", np.array(strings)[:, None], predictions,
         dict(target_names=["🙂é very_long_label", "Ω"])),
        ("DICT", "strings", strings, predictions, {}),
        ("DICT", "string_target", np.array(strings)[:, None], predictions, {}),
        ("DICT", "reserved_name", strings, predictions,
         dict(labels=["Ω"], target_names=["accuracy"])),
        ("DICT", "strided_numeric", np.array([1, 99, -0.0, 99, 2, 99, 0.0])[6::-2],
         [1.0, 2.0, 0.0, 1.0], dict(labels=[-0.0, 2.0, 1.0],
                                   sample_weight=[4.0, 2.0, 1.0, 0.5], zero_division=0)),
        ("TEXT", "float32_column", np.array([[1], [2], [1]], dtype=np.float32),
         np.array([2, 2, 1], dtype=np.float32), {}),
        ("DICT", "int64_identity", extremes, extremes[[1, 0, 2]], {}),
        ("TEXT", "multilabel", indicators, predicted_indicators, dict(zero_division=0)),
        ("DICT", "multilabel", indicators, predicted_indicators, dict(zero_division=0)),
        ("DICT", "multilabel_selected", indicators, predicted_indicators,
         dict(labels=[2, 0, 2], sample_weight=[0.5, 1.0, 2.0, 4.25], zero_division=np.nan)),
        ("DICT", "multilabel_empty", indicators, predicted_indicators,
         dict(labels=[], zero_division=0)),
    ]

    result = subprocess.run([executable.resolve()], text=True, capture_output=True, check=True)
    blocks = result.stdout.split("END\n")
    assert blocks.pop() == ""
    assert len(blocks) == len(cases)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        for block, (kind, name, y_true, y_pred, kwargs) in zip(blocks, cases, strict=True):
            heading, actual = block.split("\n", 1)
            assert heading == f"{kind} {name}"
            expected = classification_report(y_true, y_pred, output_dict=kind == "DICT", **kwargs)
            if kind == "TEXT":
                np.testing.assert_equal(actual, expected, err_msg=name)
                continue
            rows = {}
            for line in actual.splitlines():
                label, *values = line.split("\t")
                assert label not in rows
                rows[label] = list(map(float, values))
            expected_rows = {
                label: [value] if label == "accuracy" else [
                    value[key] for key in ("precision", "recall", "f1-score", "support")
                ]
                for label, value in expected.items()
            }
            compare(rows, expected_rows)
    expected_warnings = []
    for entry in log:
        category = entry.category.__name__
        message = str(entry.message)
        if category == "UndefinedMetricWarning":
            prefix, _ = message.split(" and being set to 0.0", 1)
            message = prefix + " and being set to 0.0"
        expected_warnings.append(f"{category}: {message}")
    np.testing.assert_equal(result.stderr.splitlines(), expected_warnings)
    return len(cases), len(expected_warnings)


def check_report_reference_errors():
    binary = [0, 1]
    indicators = np.array([[0, 1], [1, 0]])
    continuous = [0.1, 1.5]
    multioutput = [[0, 1], [1, 2]]
    cases = [
        (binary, binary, {"target_names": ["only"]}),
        (binary, binary, {"labels": [], "zero_division": 0}),
        (binary, binary, {"digits": -1}),
        (binary, binary, {"zero_division": 2.0, "output_dict": True}),
        ([], [], {"output_dict": True}),
        (binary, [0], {"output_dict": True}),
        (binary, binary, {"sample_weight": [1], "output_dict": True}),
        (binary, binary, {"sample_weight": [0, 0], "output_dict": True}),
        (binary, binary, {"sample_weight": [1, np.inf], "output_dict": True}),
        (indicators, indicators, {"labels": [2], "output_dict": True}),
        (indicators, [[0, 1, 0], [1, 0, 0]], {"output_dict": True}),
        (indicators, binary, {"output_dict": True}),
        (multioutput, multioutput, {"output_dict": True}),
        (continuous, continuous, {"output_dict": True}),
    ]
    for truth, predicted, kwargs in cases:
        with np.testing.assert_raises(ValueError):
            classification_report(truth, predicted, **kwargs)
    return len(cases)


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
    parser.add_argument("precision_recall", type=Path)
    parser.add_argument("classification_report", type=Path, nargs="?")
    args = parser.parse_args()
    verify_oracle()
    cases = classification_cases() | confusion_cases()
    compare(native_output(args.classification), cases)
    prf_cases = precision_recall_cases()
    compare(native_output(args.precision_recall, precision_recall_warnings()), prf_cases)
    errors = (
        check_reference_errors() + check_confusion_reference_errors()
        + check_precision_recall_reference_errors()
    )
    scores = {
        "cv": cv_scores(accuracy_score),
        "cv_balanced": cv_scores(balanced_accuracy_score),
        "cv_f1": cv_scores(f1_score),
    }
    compare(native_output(args.scorer), scores)
    if args.classification_report is not None:
        reports, report_warnings = check_reports(args.classification_report)
        report_errors = check_report_reference_errors()
        print(f"Matched {reports} classification reports, {report_errors} report errors, "
              f"and {report_warnings} report warnings against sklearn {REVISION}")
    print(
        f"Matched {sum(map(len, (cases | prf_cases).values()))} metric values, {errors} error cases "
        f"and {sum(map(len, scores.values()))} CV scores against sklearn {REVISION}"
    )


if __name__ == "__main__":
    main()
