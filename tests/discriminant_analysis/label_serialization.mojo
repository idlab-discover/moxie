"""Custom-label archive contracts for LDA and QDA."""

from std.testing import assert_equal, assert_false, assert_raises
from moxie.compose import ColumnTable
from moxie.discriminant_analysis_serialization import (
    DiscriminantLabelCodec,
    LinearDiscriminantAnalysisSchema,
    QuadraticDiscriminantAnalysisSchema,
)
from moxie.serialization import BinaryReader, BinaryWriter, dumps, loads
from serialization import training_data


struct Int32LabelCodec[
    id: String = "moxie.tests.int32-label.v1", trailing_data: Bool = False,
](DiscriminantLabelCodec):
    comptime Label = Int32
    comptime TypeId = Self.id

    @staticmethod
    def encode(mut writer: BinaryWriter, label: Int32):
        writer.scalar[DType.int32](label)
        comptime if Self.trailing_data:
            writer.scalar[DType.int32](0)

    @staticmethod
    def decode(mut reader: BinaryReader[_]) raises -> Int32:
        return reader.scalar[DType.int32]()


def main() raises:
    comptime LDA = LinearDiscriminantAnalysisSchema[DType.float32, Int32, Int32LabelCodec[]]
    comptime QDA = QuadraticDiscriminantAnalysisSchema[DType.float64, Int32, Int32LabelCodec[]]
    comptime OtherQDA = QuadraticDiscriminantAnalysisSchema[
        DType.float64, Int32, Int32LabelCodec["moxie.tests.int32-label.v2"],
    ]
    comptime MalformedQDA = QuadraticDiscriminantAnalysisSchema[
        DType.float64, Int32, Int32LabelCodec[trailing_data=True],
    ]
    var labels: List[Int32] = [2147483647, 2147483647, 2147483647, 2147483647,
                              2147483647, -2147483648, -2147483648, -2147483648, -2147483648]
    var classes: List[Int32] = [-2147483648, 2147483647]
    var names: List[String] = ["mass", "âge"]
    var X32 = training_data[DType.float32]()
    var X64 = training_data[DType.float64]()
    var named = ColumnTable(X32, names)
    var priors: List[Float64] = [1.0, 3.0]

    var lda = LDA.Model(priors=priors^)
    lda = LDA.decode(loads(dumps(LDA.encode(lda))))
    assert_false(lda.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = lda.predict(X32)
    _ = lda.fit(named, labels)
    var lda_archive = LDA.encode(lda)
    assert_equal(lda_archive.schema_version, 4)
    var restored_lda = LDA.decode(loads(dumps(lda_archive)))
    assert_equal(restored_lda.classes_, classes)
    assert_equal(restored_lda.feature_names_in_.value(), names)
    assert_equal(len(restored_lda.fit_warnings), 1)
    assert_equal(restored_lda.predict(named), lda.predict(named))
    assert_equal(restored_lda.transform(named).tolist(), lda.transform(named).tolist())
    assert_equal(dumps(LDA.encode(restored_lda)), dumps(lda_archive))
    with assert_raises(contains="label codec mismatch"):
        _ = LinearDiscriminantAnalysisSchema[DType.float32].decode(lda_archive)

    var qda = QDA.decode(loads(dumps(QDA.encode(QDA.Model()))))
    assert_false(qda.__sklearn_is_fitted__())
    with assert_raises(contains="NotFittedError"):
        _ = qda.predict(X64)
    _ = qda.fit(X64, labels)
    var qda_archive = QDA.encode(qda)
    assert_equal(qda_archive.schema_version, 3)
    var restored_qda = QDA.decode(loads(dumps(qda_archive)))
    assert_equal(restored_qda.classes_, classes)
    assert_equal(restored_qda.predict(X64), qda.predict(X64))
    assert_equal(restored_qda.predict_proba(X64).tolist(), qda.predict_proba(X64).tolist())
    assert_equal(dumps(QDA.encode(restored_qda)), dumps(qda_archive))
    with assert_raises(contains="label codec mismatch"):
        _ = OtherQDA.decode(qda_archive)
    with assert_raises(contains="label codec mismatch"):
        _ = QDA.decode(QuadraticDiscriminantAnalysisSchema[].encode(
            QuadraticDiscriminantAnalysisSchema[].Model()
        ))

    with assert_raises(contains="trailing data"):
        _ = QDA.decode(MalformedQDA.encode(qda))
    print("PASS: discriminant custom label archive contracts")
