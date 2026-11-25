from std.testing import assert_equal, assert_false, assert_raises, assert_true
from moxie.base import clone
from moxie.compose._table import ColumnTable
from moxie.naive_bayes import GaussianNB
from moxie._naive_bayes_dense import dense_from_lists
from moxie.naive_bayes_serialization import GaussianNBSchema
from moxie.serialization import BinaryWriter, dumps, loads


def main() raises:
    comptime Schema = GaussianNBSchema[]
    var X = dense_from_lists([[1.0, 3.0], [3.0, 2.0], [2.0, 4.0], [4.0, 1.0]])
    var y: List[Int] = [0, 1, 0, 1]
    var weights: List[Float64] = [1.0, 2.0, 2.0, 1.0]
    var names: List[String] = ["height", "café"]
    var table = ColumnTable(X, names)
    var swapped = ColumnTable(X, ["café", "height"])
    var renamed = ColumnTable(X, ["height", "width"])
    var narrow = ColumnTable(dense_from_lists([[1.0], [3.0], [2.0], [4.0]]), ["height"])
    var mixed = table.copy()
    mixed.add_strings("label", ["a", "b", "c", "d"])

    var model = GaussianNB()
    with assert_raises(contains="NotFittedError"):
        _ = model.predict(table)
    with assert_raises(contains="classes"):
        _ = model.partial_fit(table, y)
    assert_false(Bool(model.feature_names_in_))

    _ = model.set_fit_request(sample_weight="train_weight")
    _ = model.fit(table, y, sample_weight=weights.copy())
    assert_equal(model.feature_names_in_.value(), names)
    var encoded = dumps(Schema.encode(model))
    var archive = loads(encoded)
    assert_equal(archive.schema_version, 3)
    var restored = Schema.decode(archive)
    assert_equal(dumps(Schema.encode(restored)), encoded)
    assert_equal(restored.feature_names_in_.value(), names)
    assert_equal(restored.get_metadata_routing().fit.value()[String], "train_weight")
    assert_false(Bool(clone(restored).feature_names_in_))

    with assert_raises(contains="feature names"):
        _ = restored.predict(swapped)
    with assert_raises(contains="feature names"):
        _ = restored.predict_proba(renamed)
    with assert_raises(contains="feature names"):
        _ = restored.predict_log_proba(narrow)
    with assert_raises(contains="feature names"):
        _ = restored.score(swapped, y)
    with assert_raises(contains="feature names"):
        _ = restored.partial_fit(swapped, y)
    with assert_raises(contains="mixed ColumnTable"):
        _ = restored.fit(mixed, y)
    var short_y: List[Int] = [0]
    with assert_raises(contains="ValueError"):
        _ = restored.fit(renamed, short_y)
    assert_equal(dumps(Schema.encode(restored)), encoded)

    # Replace the final name without touching the archive's moments or metadata.
    var last_name = BinaryWriter()
    last_name.string("height")
    var duplicate = last_name^.finish()
    var malformed = archive.copy()
    for _ in range(8 + len(String("café").as_bytes())):
        _ = malformed.manifest.pop()
    for byte in duplicate:
        malformed.manifest.append(byte)
    with assert_raises(contains="duplicate feature name"):
        _ = Schema.decode(malformed)
    malformed = archive.copy()
    _ = malformed.manifest.pop()
    with assert_raises():
        _ = Schema.decode(malformed)

    _ = restored.partial_fit(table, y, sample_weight=2.0)
    _ = restored.partial_fit(X, y, sample_weight=weights.copy())
    assert_equal(restored.feature_names_in_.value(), names)
    var predictions = restored.predict(table)
    assert_equal(predictions, restored.predict(X))
    var proba = restored.predict_proba(table)
    var log_proba = restored.predict_log_proba(table)
    var joint = restored.predict_joint_log_proba(table)
    assert_equal(proba.shape.tolist(), List[Int]([4, 2]))
    for name in restored.feature_names_in_.value():
        print("name", name, sep="|")
    for row in range(4):
        print(
            "prediction", predictions[row], proba.item(row, 0), proba.item(row, 1),
            log_proba.item(row, 0), log_proba.item(row, 1), joint.item(row, 0), joint.item(row, 1),
            sep="|",
        )
    var scoring_y: List[Int] = [0, 0, 0, 1]
    print("score", restored.score(table, scoring_y, weights.copy()), sep="|")

    _ = restored.fit(X, y)
    _ = restored.partial_fit(table, y)
    assert_false(Bool(restored.feature_names_in_))
    assert_equal(Schema.encode(restored).schema_version, 2)
    _ = restored.fit(renamed, y, sample_weight=1.0)
    assert_equal(restored.feature_names_in_.value(), List[String](["height", "width"]))

    # Named first updates record names even in a different input arithmetic dtype.
    var f32_table = ColumnTable(X.astype[DType.float32](), names)
    _ = model.set_fit_request(sample_weight=None)
    var first = clone(model)
    _ = first.partial_fit(f32_table, y, classes=List[Int]([0, 1]))
    var first_archive = Schema.encode(first)
    assert_equal(first_archive.schema_version, 3)
    var first_restored = Schema.decode(first_archive)
    assert_equal(first_restored.feature_names_in_.value(), names)
    assert_true(first_restored.get_metadata_routing().fit.value().isa[NoneType]())
    assert_equal(first_restored.predict(f32_table), y)
