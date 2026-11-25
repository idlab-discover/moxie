from std.testing import assert_equal, assert_false, assert_raises
from numojo import NDArray
from moxie.base import clone
from moxie.compose import ColumnTable
from moxie._naive_bayes_dense import dense_from_lists
from moxie.naive_bayes_serialization import (
    BernoulliNBSchema, CategoricalNBSchema, ComplementNBSchema, MultinomialNBSchema,
)
from moxie.serialization import BinaryWriter, ModelSchema, dumps, loads


def round_trip[S: ModelSchema](model: S.Model) raises -> S.Model where conforms_to(S.Model, Deinitable):
    var archive = S.encode(model)
    assert_equal(archive.schema_version, 3)
    var encoded = dumps(archive)
    var restored = S.decode(loads(encoded))
    assert_equal(dumps(S.encode(restored)), encoded)
    var malformed = archive.copy()
    _ = malformed.manifest.pop()
    with assert_raises():
        _ = S.decode(malformed)
    var duplicate = BinaryWriter()
    duplicate.string("height")
    malformed = archive.copy()
    for _ in range(8 + len(String("café").as_bytes())):
        _ = malformed.manifest.pop()
    malformed.manifest.extend(duplicate^.finish())
    with assert_raises(contains="duplicate feature name"):
        _ = S.decode(malformed)
    return restored^


def report(
    estimator: String, names: List[String], labels: List[Int],
    proba: NDArray[DType.float64], log_proba: NDArray[DType.float64],
    joint: NDArray[DType.float64], score: Float64,
) raises:
    assert_equal(proba.shape.tolist(), List[Int]([4, 2]))
    assert_equal(log_proba.shape, proba.shape)
    assert_equal(joint.shape, proba.shape)
    print("model", estimator, sep="|")
    for name in names:
        print("name", name, sep="|")
    for row in range(4):
        print(
            "prediction", labels[row], proba.item(row, 0), proba.item(row, 1),
            log_proba.item(row, 0), log_proba.item(row, 1), joint.item(row, 0), joint.item(row, 1),
            sep="|",
        )
    print("score", score, sep="|")


def main() raises:
    var X = dense_from_lists([[0.0, 3.0], [3.0, 0.0], [2.0, 4.0], [4.0, 1.0]])
    var y: List[Int] = [0, 1, 0, 1]
    var scoring_y: List[Int] = [0, 0, 0, 1]
    var weights: List[Float64] = [1.0, 2.0, 2.0, 1.0]
    var names: List[String] = ["height", "café"]
    var table = ColumnTable(X, names)
    var reordered = ColumnTable(X, ["café", "height"])
    var renamed = ColumnTable(X, ["height", "width"])
    var mixed = table.copy()
    mixed.add_strings("label", ["a", "b", "c", "d"])
    var first_table = ColumnTable(X.astype[DType.float32](), names)

    comptime Multinomial = MultinomialNBSchema[]
    var multinomial = Multinomial.Model()
    with assert_raises(contains="NotFittedError"):
        _ = multinomial.predict(table)
    with assert_raises(contains="classes"):
        _ = multinomial.partial_fit(table, y)
    assert_false(Bool(multinomial.feature_names_in_))
    _ = multinomial.set_fit_request(sample_weight="train_weight")
    _ = multinomial.fit(table, y, sample_weight=weights.copy())
    assert_equal(multinomial.feature_names_in_.value(), names)
    multinomial = round_trip[Multinomial](multinomial)
    assert_equal(multinomial.get_metadata_routing().fit.value()[String], "train_weight")
    assert_false(Bool(clone(multinomial).feature_names_in_))
    var before_multinomial = dumps(Multinomial.encode(multinomial))
    with assert_raises(contains="feature names"):
        _ = multinomial.partial_fit(reordered, y)
    with assert_raises(contains="feature names"):
        _ = multinomial.predict_proba(renamed)
    with assert_raises(contains="mixed ColumnTable"):
        _ = multinomial.fit(mixed, y)
    assert_equal(dumps(Multinomial.encode(multinomial)), before_multinomial)
    _ = multinomial.partial_fit(table, y, sample_weight=2.0)
    _ = multinomial.partial_fit(X, y, sample_weight=weights.copy())
    assert_equal(multinomial.predict(table), multinomial.predict(X))
    report(
        Multinomial.TypeId, multinomial.feature_names_in_.value(), multinomial.predict(table),
        multinomial.predict_proba(table), multinomial.predict_log_proba(table),
        multinomial.predict_joint_log_proba(table), multinomial.score(table, scoring_y, weights.copy()),
    )
    _ = multinomial.fit(X, y)
    _ = multinomial.partial_fit(table, y)
    assert_false(Bool(multinomial.feature_names_in_))
    assert_equal(Multinomial.encode(multinomial).schema_version, 2)
    var first_multinomial = Multinomial.Model()
    _ = first_multinomial.partial_fit(first_table, y, classes=y.copy())
    assert_equal(first_multinomial.feature_names_in_.value(), names)

    comptime Complement = ComplementNBSchema[]
    var complement = Complement.Model()
    with assert_raises(contains="NotFittedError"):
        _ = complement.predict(table)
    with assert_raises(contains="classes"):
        _ = complement.partial_fit(table, y)
    assert_false(Bool(complement.feature_names_in_))
    _ = complement.set_fit_request(sample_weight="train_weight")
    _ = complement.fit(table, y, sample_weight=weights.copy())
    assert_equal(complement.feature_names_in_.value(), names)
    complement = round_trip[Complement](complement)
    assert_equal(complement.get_metadata_routing().fit.value()[String], "train_weight")
    assert_false(Bool(clone(complement).feature_names_in_))
    var before_complement = dumps(Complement.encode(complement))
    with assert_raises(contains="feature names"):
        _ = complement.partial_fit(reordered, y)
    with assert_raises(contains="feature names"):
        _ = complement.predict_proba(renamed)
    with assert_raises(contains="mixed ColumnTable"):
        _ = complement.fit(mixed, y)
    assert_equal(dumps(Complement.encode(complement)), before_complement)
    _ = complement.partial_fit(table, y, sample_weight=2.0)
    _ = complement.partial_fit(X, y, sample_weight=weights.copy())
    assert_equal(complement.predict(table), complement.predict(X))
    report(
        Complement.TypeId, complement.feature_names_in_.value(), complement.predict(table),
        complement.predict_proba(table), complement.predict_log_proba(table),
        complement.predict_joint_log_proba(table), complement.score(table, scoring_y, weights.copy()),
    )
    _ = complement.fit(X, y)
    _ = complement.partial_fit(table, y)
    assert_false(Bool(complement.feature_names_in_))
    assert_equal(Complement.encode(complement).schema_version, 2)
    var first_complement = Complement.Model()
    _ = first_complement.partial_fit(first_table, y, classes=y.copy())
    assert_equal(first_complement.feature_names_in_.value(), names)

    comptime Bernoulli = BernoulliNBSchema[]
    var bernoulli = Bernoulli.Model()
    with assert_raises(contains="NotFittedError"):
        _ = bernoulli.predict(table)
    with assert_raises(contains="classes"):
        _ = bernoulli.partial_fit(table, y)
    assert_false(Bool(bernoulli.feature_names_in_))
    _ = bernoulli.set_fit_request(sample_weight="train_weight")
    _ = bernoulli.fit(table, y, sample_weight=weights.copy())
    assert_equal(bernoulli.feature_names_in_.value(), names)
    bernoulli = round_trip[Bernoulli](bernoulli)
    assert_equal(bernoulli.get_metadata_routing().fit.value()[String], "train_weight")
    assert_false(Bool(clone(bernoulli).feature_names_in_))
    var before_bernoulli = dumps(Bernoulli.encode(bernoulli))
    with assert_raises(contains="feature names"):
        _ = bernoulli.partial_fit(reordered, y)
    with assert_raises(contains="feature names"):
        _ = bernoulli.predict_proba(renamed)
    with assert_raises(contains="mixed ColumnTable"):
        _ = bernoulli.fit(mixed, y)
    assert_equal(dumps(Bernoulli.encode(bernoulli)), before_bernoulli)
    _ = bernoulli.partial_fit(table, y, sample_weight=2.0)
    _ = bernoulli.partial_fit(X, y, sample_weight=weights.copy())
    assert_equal(bernoulli.predict(table), bernoulli.predict(X))
    report(
        Bernoulli.TypeId, bernoulli.feature_names_in_.value(), bernoulli.predict(table),
        bernoulli.predict_proba(table), bernoulli.predict_log_proba(table),
        bernoulli.predict_joint_log_proba(table), bernoulli.score(table, scoring_y, weights.copy()),
    )
    _ = bernoulli.fit(X, y)
    _ = bernoulli.partial_fit(table, y)
    assert_false(Bool(bernoulli.feature_names_in_))
    assert_equal(Bernoulli.encode(bernoulli).schema_version, 2)
    var first_bernoulli = Bernoulli.Model()
    _ = first_bernoulli.partial_fit(first_table, y, classes=y.copy())
    assert_equal(first_bernoulli.feature_names_in_.value(), names)

    comptime Categorical = CategoricalNBSchema[]
    var categorical = Categorical.Model()
    with assert_raises(contains="NotFittedError"):
        _ = categorical.predict(table)
    with assert_raises(contains="classes"):
        _ = categorical.partial_fit(table, y)
    assert_false(Bool(categorical.feature_names_in_))
    _ = categorical.set_fit_request(sample_weight="train_weight")
    _ = categorical.fit(table, y, sample_weight=weights.copy())
    assert_equal(categorical.feature_names_in_.value(), names)
    categorical = round_trip[Categorical](categorical)
    assert_equal(categorical.get_metadata_routing().fit.value()[String], "train_weight")
    assert_false(Bool(clone(categorical).feature_names_in_))
    var before_categorical = dumps(Categorical.encode(categorical))
    with assert_raises(contains="feature names"):
        _ = categorical.partial_fit(reordered, y)
    with assert_raises(contains="feature names"):
        _ = categorical.predict_proba(renamed)
    with assert_raises(contains="mixed ColumnTable"):
        _ = categorical.fit(mixed, y)
    assert_equal(dumps(Categorical.encode(categorical)), before_categorical)
    _ = categorical.partial_fit(table, y, sample_weight=2.0)
    _ = categorical.partial_fit(X, y, sample_weight=weights.copy())
    assert_equal(categorical.predict(table), categorical.predict(X))
    report(
        Categorical.TypeId, categorical.feature_names_in_.value(), categorical.predict(table),
        categorical.predict_proba(table), categorical.predict_log_proba(table),
        categorical.predict_joint_log_proba(table), categorical.score(table, scoring_y, weights.copy()),
    )
    _ = categorical.fit(X, y)
    _ = categorical.partial_fit(table, y)
    assert_false(Bool(categorical.feature_names_in_))
    assert_equal(Categorical.encode(categorical).schema_version, 2)
    var first_categorical = Categorical.Model()
    _ = first_categorical.partial_fit(first_table, y, classes=y.copy())
    assert_equal(first_categorical.feature_names_in_.value(), names)
