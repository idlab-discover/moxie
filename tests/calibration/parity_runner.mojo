"""Execute externally generated oracle cases in a native process."""
from std.sys import argv
from std.utils import Variant
from std.math import isfinite, nan
import numojo as nm
from moxie.base import clone
from moxie.model_selection import KFold, StratifiedKFold
from moxie.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
    CalibrationCV,
    CalibrationSplit,
    CalibrationDisplay,
    CalibrationEstimator,
    adapt_classifier,
)
from moxie._calibration_algorithms import (
    Array,
    _SigmoidCalibration,
    _IsotonicCalibration,
    _TemperatureScaling,
)
from moxie.calibration_serialization import (
    CalibratedBernoulliNBClassifierCVSchema,
    CalibratedCategoricalNBClassifierCVSchema,
    CalibratedClassifierCVSchema,
    CalibratedComplementNBClassifierCVSchema,
    CalibratedGaussianNBClassifierCVSchema,
    CalibratedLDAClassifierCVSchema,
    CalibratedMultinomialNBClassifierCVSchema,
    CalibratedQDAClassifierCVSchema,
    CalibratedSVCClassifierCVSchema,
)
from moxie.serialization import ModelSchema, dumps, loads
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from moxie.frozen import FrozenEstimator
from moxie.linear_model import LogisticRegression
from moxie.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from moxie.svm import LinearSVC, SVC
from calibration_centroid import Centroid


def array(text: String) raises -> Array:
    var fields = text.split(":")
    var n = Int(String(fields[0]))
    var c = Int(String(fields[1]))
    var values = fields[2].split(",")
    if c == 0:
        var out = nm.zeros[DType.float64](nm.Shape(n))
        for i in range(n):
            out[nm.Item(i)] = Float64(String(values[i]))
        return out^
    var out = nm.zeros[DType.float64](nm.Shape(n, c))
    for i in range(n):
        for j in range(c):
            out[nm.Item(i, j)] = Float64(String(values[i * c + j]))
    return out^


def compare(
    actual: Array, expected: Array, tolerance: Float64, context: String
) raises:
    if actual.ndim != expected.ndim or actual.shape[0] != expected.shape[0]:
        raise Error(context + ": shape mismatch")
    var columns = actual.shape[1] if actual.ndim == 2 else 1
    if actual.ndim == 2 and actual.shape[1] != expected.shape[1]:
        raise Error(context + ": column mismatch")
    for i in range(actual.shape[0]):
        for j in range(columns):
            var a = actual.item(i, j) if actual.ndim == 2 else actual.item(i)
            var b = expected.item(
                i, j
            ) if expected.ndim == 2 else expected.item(i)
            if (
                not isfinite(a)
                or not isfinite(b)
                or abs(a - b) > tolerance * (1 + abs(b))
            ):
                raise Error(
                    context + ": actual=" + String(a) + " expected=" + String(b)
                )


def compare_estimator_result[E: CalibrationEstimator, origin: ImmOrigin](
    model: CalibratedClassifierCV[E],
    q: Array,
    expected: Array,
    fields: List[StringSpan[origin]],
    context: String,
) raises:
    compare(model.classes_, array(String(fields[9])), 0, context)
    compare(model.predict_proba(q), expected, 5e-6, context)
    compare(model.predict(q), array(String(fields[8])), 0, context)


def compare_production[E: CalibrationEstimator, origin: ImmOrigin](
    var estimator: E,
    x: Array,
    y: Array,
    w: Optional[Array],
    q: Array,
    expected: Array,
    fields: List[StringSpan[origin]],
    context: String,
) raises:
    var use_auto = String(fields[7]) == "auto"
    var ensemble: Variant[Bool, String] = "auto"
    if use_auto:
        estimator = clone(estimator)
        if not estimator.is_frozen() or not estimator.__sklearn_is_fitted__():
            raise Error(context + ": clone must preserve the fitted frozen classifier")
    else:
        ensemble = String(fields[7]) == "true"
    var model = CalibratedClassifierCV(
        estimator^, cv=3, method=String(fields[6]), ensemble=ensemble^
    )
    _ = model.fit(x, y, w)
    compare_estimator_result(model, q, expected, fields, context)
    if use_auto and len(model.calibrated_classifiers_) != 1:
        raise Error(context + ": frozen calibration must fit a single calibrator")


def compare_serialized[
    E: CalibrationEstimator, origin: ImmOrigin, //, Schema: ModelSchema
](
    var estimator: E,
    x: Array,
    y: Array,
    w: Optional[Array],
    q: Array,
    expected: Array,
    fields: List[StringSpan[origin]],
    context: String,
) raises where Schema.Model == CalibratedClassifierCV[E]:
    var model = CalibratedClassifierCV(
        estimator^, cv=3, method=String(fields[6]), ensemble=fields[7] == "true",
    )
    _ = model.fit(x, y, w)
    var archive = Schema.encode(rebind[Schema.Model](model))
    var restored = rebind_var[CalibratedClassifierCV[E]](
        Schema.decode(loads(dumps(archive)))
    )
    compare_estimator_result(restored, q, expected, fields, context)
    compare(restored.predict_proba(q), model.predict_proba(q), 0, context)


def compare_float32[E: CalibrationEstimator, origin: ImmOrigin](
    mut model: CalibratedClassifierCV[E],
    x: Array,
    y: Array,
    w: Optional[Array],
    q: Array,
    expected: Array,
    fields: List[StringSpan[origin]],
    context: String,
) raises:
    var x32 = x.astype[DType.float32]()
    var q32 = q.astype[DType.float32]()
    var probabilities: Array = model.fit(x32, y, w).predict_proba(q32)
    var predictions: Array = model.predict(q32)
    compare(model.classes_, array(String(fields[9])), 0, context)
    compare(probabilities, expected, 5e-6, context)
    compare(predictions, array(String(fields[8])), 0, context)
    var expected_score = array(String(fields[10])).item(0)
    if abs(model.score(x32, y, w) - expected_score) > 2e-12:
        raise Error(context + ": incorrect weighted score")


def main() raises:
    var args = argv()
    var data: String
    with open(args[1], "r") as f:
        data = f.read()
    var count = 0
    for line in data.split("\n"):
        if line == "":
            continue
        var fields = line.split("|")
        var kind = String(fields[0])
        var x = array(String(fields[1]))
        var y = array(String(fields[2]))
        var w: Optional[Array] = None
        if String(fields[3]) != "-":
            w = array(String(fields[3]))
        var q = array(String(fields[4]))
        var expected = array(String(fields[5]))
        var context = kind + " case " + String(count)
        if kind == "list_int64" or kind == "list_float32":
            var backing = nm.full[DType.float64](nm.Shape(2 * y.size), nan[DType.float64]())
            var view = backing.view_with_layout(y.shape, nm.NDArrayStrides(2), 1)
            for i in range(y.size):
                view[nm.Item(i)] = y.item(i)
            y = view^
        if kind == "sigmoid":
            var model = _SigmoidCalibration()
            _ = model.fit(x, y, w)
            compare(model.predict(q), expected, 5e-6, context)
        elif kind == "isotonic":
            var model = _IsotonicCalibration()
            _ = model.fit(x, y, w)
            compare(model.predict(q), expected, 2e-12, context)
        elif kind == "temperature":
            var model = _TemperatureScaling()
            _ = model.fit(x, y, w)
            compare(model.predict(q), expected, 5e-6, context)
        elif kind == "curve":
            var result = calibration_curve(
                y,
                x,
                n_bins=Int(String(fields[6])),
                strategy=String(fields[7]),
                pos_label=Float64(String(fields[8])),
            )
            compare(result[0], expected, 2e-12, context)
            compare(result[1], q, 2e-12, context)
        elif kind == "curve_float32":
            var y32 = y.astype[DType.float32]()
            var x32 = x.astype[DType.float32]()
            var result = calibration_curve(
                y32,
                x32,
                n_bins=Int(String(fields[6])),
                strategy=String(fields[7]),
            )
            compare(result[0], expected, 2e-12, context)
            compare(result[1], q, 2e-12, context)
        elif kind == "curve_string":
            var labels = List[String]()
            for i in range(y.size):
                labels.append("cat" if y.item(i) == -3 else "dog")
            var result = calibration_curve(
                labels,
                x,
                n_bins=Int(String(fields[6])),
                strategy=String(fields[7]),
                pos_label=String(fields[8]),
            )
            compare(result[0], expected, 2e-12, context)
            compare(result[1], q, 2e-12, context)
        elif kind == "display" or kind == "display_estimator":
            var pos: Optional[Float64] = None
            if String(fields[8]) != "-":
                pos = Float64(String(fields[8]))
            var display: CalibrationDisplay
            if kind == "display":
                display = CalibrationDisplay.from_predictions(
                    y,
                    x,
                    n_bins=Int(String(fields[6])),
                    strategy=String(fields[7]),
                    pos_label=pos,
                    name=String(fields[9]),
                    alpha=0.8,
                )
            else:
                var estimator = Centroid()
                estimator.fit(x, y)
                display = CalibrationDisplay.from_estimator(
                    estimator,
                    x,
                    y,
                    n_bins=Int(String(fields[6])),
                    strategy=String(fields[7]),
                    pos_label=pos,
                    alpha=0.8,
                )
            compare(display.prob_true, expected, 2e-12, context)
            compare(display.prob_pred, q, 2e-12, context)
            compare(display.y_prob, array(String(fields[10])), 2e-12, context)
            if display.ax_.value().xlabel != String(
                fields[11]
            ) or display.ax_.value().ylabel != String(fields[12]):
                raise Error(context + ": axis labels differ")
            if (
                display.estimator_name.value() != String(fields[9])
                or display.line_.value().alpha != 0.8
            ):
                raise Error(context + ": display metadata differs")
        elif (
            kind == "cv" or kind == "cv_splits" or kind == "cv_default"
            or kind == "cv_kfold" or kind == "cv_kfold_shuffled"
            or kind == "cv_stratified_shuffled"
        ):
            var cv: CalibrationCV = 3
            if kind == "cv_default":
                cv = None
            elif kind == "cv_kfold":
                cv = KFold(3)
            elif kind == "cv_kfold_shuffled":
                cv = KFold(3, shuffle=True, random_state=17)
            elif kind == "cv_stratified_shuffled":
                cv = StratifiedKFold(3, shuffle=True, random_state=17)
            elif kind == "cv_splits":
                var splits = List[CalibrationSplit]()
                for split in fields[11].split(";"):
                    var parts = split.split(":")
                    var train = List[Int]()
                    var test = List[Int]()
                    for index in parts[0].split(","):
                        train.append(Int(String(index)))
                    for index in parts[1].split(","):
                        test.append(Int(String(index)))
                    splits.append(CalibrationSplit(train^, test^))
                cv = splits^
            var estimator = Centroid(String(fields[8]) == "true")
            if String(fields[9]) == "true":
                estimator.fit(x, y, w)
                estimator.frozen = True
            var model = CalibratedClassifierCV(
                estimator^,
                cv=cv^,
                method=String(fields[6]),
                ensemble=String(fields[7]) == "true",
            )
            _ = model.fit(x, y, w)
            compare(model.predict_proba(q), expected, 5e-6, context)
            compare(model.predict(q), array(String(fields[10])), 0, context)
        elif kind == "default_linear_svc":
            var ensemble: Variant[Bool, String] = "auto"
            if String(fields[7]) != "auto":
                ensemble = fields[7] == "true"
            var model = CalibratedClassifierCV(
                method=String(fields[6]), ensemble=ensemble^
            )
            _ = model.fit(x, y, w)
            compare_estimator_result(model, q, expected, fields, context)
            var restored = CalibratedClassifierCVSchema.decode(
                loads(dumps(CalibratedClassifierCVSchema.encode(model)))
            )
            compare_estimator_result(restored, q, expected, fields, context + " restored")
            compare(restored.predict_proba(q), model.predict_proba(q), 0, context)
            var expected_folds = 1 if fields[7] == "false" else 5
            if len(model.calibrated_classifiers_) != expected_folds:
                raise Error(context + ": incorrect number of fitted classifiers")
        elif kind == "float32_default":
            var model = CalibratedClassifierCV(
                cv=3, method=String(fields[6]), ensemble=fields[7] == "true"
            )
            compare_float32(model, x, y, w, q, expected, fields, context)
        elif kind == "serialized_svc":
            comptime Schema = CalibratedSVCClassifierCVSchema
            var model = CalibratedClassifierCV(
                adapt_classifier(SVC(tol=1e-8)), cv=3,
                method=String(fields[6]), ensemble=fields[7] == "true",
            )
            _ = model.fit(x, y)
            var restored = Schema.decode(loads(dumps(Schema.encode(model))))
            compare_estimator_result(restored, q, expected, fields, context)
            compare(restored.predict_proba(q), model.predict_proba(q), 0, context)
        elif kind == "float32_gaussian":
            var model = CalibratedClassifierCV(
                adapt_classifier(GaussianNB()),
                cv=3, method=String(fields[6]), ensemble=fields[7] == "true",
            )
            compare_float32(model, x, y, w, q, expected, fields, context)
        elif kind == "production_gaussian":
            compare_serialized[CalibratedGaussianNBClassifierCVSchema](
                adapt_classifier(GaussianNB()),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "serialized_multinomial":
            compare_serialized[CalibratedMultinomialNBClassifierCVSchema](
                adapt_classifier(MultinomialNB(alpha=0.25)),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "serialized_complement":
            compare_serialized[CalibratedComplementNBClassifierCVSchema](
                adapt_classifier(ComplementNB(alpha=0.25, norm=True)),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "serialized_bernoulli":
            compare_serialized[CalibratedBernoulliNBClassifierCVSchema](
                adapt_classifier(BernoulliNB(alpha=0.25, binarize=1.5)),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "serialized_categorical":
            compare_serialized[CalibratedCategoricalNBClassifierCVSchema](
                adapt_classifier(CategoricalNB(
                    alpha=0.25, min_categories_values=List[Int]([4, 3, 5]),
                )),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "production_lda":
            compare_serialized[CalibratedLDAClassifierCVSchema](
                adapt_classifier(LinearDiscriminantAnalysis()),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "production_qda":
            compare_serialized[CalibratedQDAClassifierCVSchema](
                adapt_classifier(QuadraticDiscriminantAnalysis()),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "production_logistic":
            compare_production(
                adapt_classifier(LogisticRegression(tol=1e-10, max_iter=500)),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "list_int64":
            compare_production(
                adapt_classifier(LinearSVC[Int64](dual=False, random_state=0, tol=1e-10)),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "list_float32":
            compare_production(
                adapt_classifier(LinearSVC[Float32](dual=False, random_state=0, tol=1e-10)),
                x, y, w, q, expected, fields, context,
            )
        elif kind == "production_frozen_logistic":
            var estimator = LogisticRegression(tol=1e-10, max_iter=500)
            var train_x = array(String(fields[10]))
            var train_y = array(String(fields[11])).astype[DType.int64]()
            _ = estimator.fit(train_x, train_y)
            compare_production(
                adapt_classifier(FrozenEstimator(estimator^)),
                x, y, w, q, expected, fields, context,
            )
        elif kind.startswith("production_frozen_"):
            var train_x = array(String(fields[10]))
            var train_y = array(String(fields[11]))
            var labels = List[Int](capacity=train_y.size)
            for i in range(train_y.size):
                labels.append(Int(train_y.item(i)))
            if kind == "production_frozen_gaussian":
                var estimator = GaussianNB()
                _ = estimator.fit(train_x, labels)
                compare_production(
                    adapt_classifier(FrozenEstimator(estimator^)),
                    x, y, w, q, expected, fields, context,
                )
            elif kind == "production_frozen_lda":
                var estimator = LinearDiscriminantAnalysis()
                _ = estimator.fit(train_x, labels)
                compare_production(
                    adapt_classifier(FrozenEstimator(estimator^)),
                    x, y, w, q, expected, fields, context,
                )
            elif kind == "production_frozen_qda":
                var estimator = QuadraticDiscriminantAnalysis()
                _ = estimator.fit(train_x, labels)
                compare_production(
                    adapt_classifier(FrozenEstimator(estimator^)),
                    x, y, w, q, expected, fields, context,
                )
            else:
                raise Error("unknown frozen production parity case")
        else:
            raise Error("unknown parity case")
        count += 1
    print("PASS:", count, "scikit-learn parity cases")
