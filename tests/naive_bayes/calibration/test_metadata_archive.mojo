from std.testing import assert_equal, assert_true
from numojo import Item, Shape, zeros

from moxie.calibration import CalibratedClassifierCV, adapt_classifier
from moxie.calibration_serialization import CalibratedGaussianNBClassifierCVSchema
from moxie.naive_bayes import GaussianNB
from moxie.serialization import dumps, loads


def main() raises:
    var X = zeros[DType.float64](Shape(42, 2))
    var y = zeros[DType.float64](Shape(42))
    for row in range(42):
        var label = Float64(row % 3)
        X[Item(row, 0)] = label + Float64(row % 5) / 7.0
        X[Item(row, 1)] = Float64(row % 11) / 13.0
        y[Item(row)] = label
    var estimator = GaussianNB()
    _ = estimator.set_fit_request(sample_weight="training").set_partial_fit_request(
        sample_weight=True,
    ).set_score_request(sample_weight=False)
    var calibrated = CalibratedClassifierCV(adapt_classifier(estimator^), cv=2)
    _ = calibrated.fit(X, y)
    comptime Schema = CalibratedGaussianNBClassifierCVSchema
    var before = dumps(Schema.encode(calibrated))
    var restored = Schema.decode(loads(before))
    assert_equal(dumps(Schema.encode(restored)), before)
    assert_equal(restored.predict_proba(X).tolist(), calibrated.predict_proba(X).tolist())
    for child in restored.calibrated_classifiers_:
        ref estimator = child.estimator.estimator()
        assert_equal(estimator.get_metadata_routing().fit.requested_name().value(), "training")
        assert_true(estimator.get_partial_fit_request().value()[Bool])
    print("PASS: GaussianNB requests survive calibration cloning and nested archives")
