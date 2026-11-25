"""Custom covariance configuration archives for dense LDA and QDA."""

from std.testing import assert_equal, assert_false, assert_raises
from numojo import NDArray
from moxie.compose import ColumnTable
from moxie.covariance import empirical_covariance, shrunk_covariance
from moxie.discriminant_analysis import CovarianceEstimator
from moxie.discriminant_analysis_serialization import (
    DiscriminantCovarianceCodec,
    LinearDiscriminantAnalysisSchema,
    QuadraticDiscriminantAnalysisSchema,
)
from moxie.serialization import BinaryReader, BinaryWriter, dumps, load_model, loads, save_model
from label_serialization import Int32LabelCodec
from serialization import training_data


struct FixedShrinkage(CovarianceEstimator):
    var amount: Float64
    var _fitted: Optional[NDArray[DType.float64]]

    def __init__(out self, amount: Float64):
        self.amount = amount
        self._fitted = None

    def clone_configuration(self) -> Self:
        return Self(self.amount)

    def fit(mut self, X: NDArray[DType.float64]) raises:
        self._fitted = shrunk_covariance(empirical_covariance(X), self.amount)

    def covariance(self) raises -> NDArray[DType.float64]:
        if self._fitted:
            return self._fitted.value().copy()
        raise Error("NotFittedError: FixedShrinkage")


struct FixedShrinkageCodec[
    id: String = "moxie.tests.fixed-shrinkage.v1", trailing_data: Bool = False,
](DiscriminantCovarianceCodec):
    comptime Estimator = FixedShrinkage
    comptime TypeId = Self.id

    @staticmethod
    def encode(mut writer: BinaryWriter, estimator: FixedShrinkage):
        writer.scalar[DType.float64](estimator.amount)
        comptime if Self.trailing_data:
            writer.boolean(False)

    @staticmethod
    def decode(mut reader: BinaryReader[_]) raises -> FixedShrinkage:
        var amount = reader.scalar[DType.float64]()
        if not (amount >= 0 and amount <= 1):
            raise Error("ModelSchemaError: invalid fixed shrinkage")
        return FixedShrinkage(amount)


def lda_round_trip() raises:
    comptime Schema = LinearDiscriminantAnalysisSchema[
        DType.float32, Int32, Int32LabelCodec[], CovarianceCodec=FixedShrinkageCodec[],
    ]
    var X = training_data[DType.float32]()
    var names: List[String] = ["mass", "âge"]
    var table = ColumnTable(X, names)
    var y: List[Int32] = [-3, -3, -3, -3, -3, 9, 9, 9, 9]
    var priors: List[Float64] = [1.0, 3.0]
    var model = Schema.Model(
        solver="eigen", priors=priors^, covariance_estimator=FixedShrinkage(0.25),
    )
    _ = model.fit(table, y)
    var before = model.predict_proba(table).tolist()
    _ = model.set_params(covariance_estimator=Optional[FixedShrinkage](FixedShrinkage(0.7)))
    var path = "builds/discriminant-lda-custom-covariance.moxie"
    save_model[Schema](model, path)
    var restored = load_model[Schema](path)
    assert_equal(Schema.encode(restored).schema_version, 5)
    assert_equal(restored.covariance_estimator.value().amount, 0.7)
    assert_equal(restored.feature_names_in_.value(), names)
    assert_equal(len(restored.fit_warnings), 1)
    assert_equal(restored.predict_proba(table).tolist(), before)
    assert_equal(restored.transform(table).tolist(), model.transform(table).tolist())
    assert_equal(dumps(Schema.encode(restored)), dumps(Schema.encode(model)))

    _ = model.set_score_request(sample_weight=String("validation_weight"))
    var routed = Schema.decode(loads(dumps(Schema.encode(model))))
    assert_equal(routed.covariance_estimator.value().amount, 0.7)
    assert_equal(routed.feature_names_in_.value(), names)
    assert_equal(routed.predict_proba(table).tolist(), before)
    assert_equal(routed.get_metadata_routing().score.requested_name().value(), "validation_weight")
    assert_equal(dumps(Schema.encode(routed)), dumps(Schema.encode(model)))

    _ = restored.fit(table, y)
    _ = model.fit(table, y)
    var after = restored.predict_proba(table).tolist()
    assert_equal(after, model.predict_proba(table).tolist())
    assert_false(after == before)


def qda_round_trip() raises:
    comptime Schema = QuadraticDiscriminantAnalysisSchema[CovarianceCodec=FixedShrinkageCodec[]]
    comptime OtherSchema = QuadraticDiscriminantAnalysisSchema[
        CovarianceCodec=FixedShrinkageCodec["moxie.tests.fixed-shrinkage.v2"],
    ]
    comptime MalformedSchema = QuadraticDiscriminantAnalysisSchema[
        CovarianceCodec=FixedShrinkageCodec[trailing_data=True],
    ]
    var X = training_data[DType.float64]()
    var y: List[Int] = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    var model = Schema.Model(
        solver="eigen", store_covariance=True, covariance_estimator=FixedShrinkage(0.4),
    )
    var archive = Schema.encode(model)
    var restored = Schema.decode(loads(dumps(archive)))
    assert_false(restored.__sklearn_is_fitted__())
    assert_equal(restored.covariance_estimator.value().amount, 0.4)
    with assert_raises(contains="NotFittedError"):
        _ = restored.predict(X)
    with assert_raises(contains="covariance codec mismatch"):
        _ = OtherSchema.decode(archive)
    with assert_raises(contains="covariance codec mismatch"):
        _ = QuadraticDiscriminantAnalysisSchema[].decode(archive)
    with assert_raises(contains="covariance codec mismatch"):
        _ = Schema.decode(QuadraticDiscriminantAnalysisSchema[].encode(
            QuadraticDiscriminantAnalysisSchema[].Model()
        ))
    with assert_raises(contains="trailing data"):
        _ = Schema.decode(MalformedSchema.encode(model))
    with assert_raises(contains="invalid fixed shrinkage"):
        _ = Schema.decode(Schema.encode(Schema.Model(covariance_estimator=FixedShrinkage(2.0))))

    _ = model.fit(X, y)
    _ = restored.fit(X, y)
    assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())
    model.covariance_estimator = None
    archive = Schema.encode(model)
    assert_equal(archive.schema_version, 4)
    restored = Schema.decode(loads(dumps(archive)))
    assert_false(Bool(restored.covariance_estimator))
    assert_equal(restored.predict_proba(X).tolist(), model.predict_proba(X).tolist())
    assert_equal(dumps(Schema.encode(restored)), dumps(archive))


def main() raises:
    lda_round_trip()
    qda_round_trip()
    print("PASS: discriminant custom covariance archive contracts")
