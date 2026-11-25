from std.testing import assert_equal, assert_false, assert_raises, assert_true

from moxie._naive_bayes_dense import dense_from_lists
from moxie.base import clone
from moxie.metadata_routing import FitScoreMetadataRequestProvider
from moxie.naive_bayes import NaiveBayesMetadataRequestProvider
from moxie.naive_bayes_serialization import (
    BernoulliNBSchema, CategoricalNBSchema, ComplementNBSchema, GaussianNBSchema, MultinomialNBSchema,
)
from moxie.serialization import BinaryWriter, ModelSchema, dumps, loads


def check_default[E: FitScoreMetadataRequestProvider](model: E) raises:
    var metadata = Dict[String, List[Float64]]()
    metadata["sample_weight"] = [1.0]
    var requests = model.get_metadata_routing()
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = requests.fit.route(metadata)
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = requests.score.route(metadata)


def check_partial[E: NaiveBayesMetadataRequestProvider](
    model: E, weights: Dict[String, List[Float64]],
) raises -> List[Float64]:
    var request = model.get_partial_fit_request()
    assert_equal(request.requested_name().value(), "sample_weight")
    var resolved = request.route(weights).value().copy()
    request.set(False)
    assert_equal(model.get_partial_fit_request().requested_name().value(), "sample_weight")
    return resolved^


def invalid_requests() raises:
    comptime Schema = GaussianNBSchema[]
    var model = Schema.Model()
    var metadata = Dict[String, Float64]()
    metadata["sample_weight"] = 2.0
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = model.get_partial_fit_request().route(metadata)

    _ = model.set_fit_request(sample_weight=True)
    with assert_raises(contains="metadata alias"):
        _ = model.set_fit_request(sample_weight="1weight")
    assert_equal(model.get_metadata_routing().fit.requested_name().value(), "sample_weight")
    _ = model.set_fit_request(sample_weight=None)
    with assert_raises(contains="UnsetMetadataPassedError"):
        _ = model.get_metadata_routing().fit.route(metadata)
    var legacy = Schema.encode(model)
    assert_equal(legacy.schema_version, 1)
    check_default(Schema.decode(legacy))

    var broken = legacy.copy()
    broken.schema_version = 2
    var writer = BinaryWriter()
    writer.uint(3)
    writer.string("$UNCHANGED$")
    broken.manifest.extend(writer^.finish())
    with assert_raises(contains="invalid sample-weight alias"):
        _ = Schema.decode(broken)


def check_archives[
    E: NaiveBayesMetadataRequestProvider & Deinitable,
    S: ModelSchema,
](model: E, fitted: E) raises where S.Model == E and conforms_to(S.Model, Deinitable):
    var archive = S.encode(rebind[S.Model](model))
    var before = dumps(archive)
    assert_equal(archive.schema_version, 2)
    var restored = rebind_var[E](S.decode(loads(before)))
    assert_false(restored.__sklearn_is_fitted__())
    assert_equal(dumps(S.encode(rebind[S.Model](restored))), before)

    var fitted_bytes = dumps(S.encode(rebind[S.Model](fitted)))
    restored = rebind_var[E](S.decode(loads(fitted_bytes)))
    assert_true(restored.__sklearn_is_fitted__())
    assert_equal(dumps(S.encode(rebind[S.Model](restored))), fitted_bytes)
    var cloned = clone(restored)
    assert_false(cloned.__sklearn_is_fitted__())
    assert_equal(dumps(S.encode(rebind[S.Model](cloned))), before)

    _ = archive.manifest.pop()
    with assert_raises():
        _ = S.decode(archive)


def fit_weights[E: FitScoreMetadataRequestProvider](
    model: E, metadata: Dict[String, List[Float64]],
) raises -> List[Float64]:
    var routing = model.get_metadata_routing()
    assert_equal(routing.fit.requested_name().value(), "training")
    var weights = routing.fit.route(metadata).value().copy()
    routing.fit.set(False)
    assert_equal(model.get_metadata_routing().fit.requested_name().value(), "training")
    return weights^


def score_weights[E: FitScoreMetadataRequestProvider](
    model: E, metadata: Dict[String, List[Float64]],
) raises -> Optional[List[Float64]]:
    var routing = model.get_metadata_routing()
    ref request = routing.score
    assert_true(request.value().isa[Bool]())
    assert_false(request.value()[Bool])
    return request.route(metadata)


def main() raises:
    invalid_requests()
    var X = dense_from_lists([[1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    var y: List[Int] = [0, 0, 1, 1]
    var scoring_y: List[Int] = [0, 0, 1, 0]
    var metadata = Dict[String, List[Float64]]()
    metadata["training"] = [1.0, 2.0, 3.0, 4.0]
    metadata["sample_weight"] = [2.0, 4.0, 1.0, 3.0]

    comptime Gaussian = GaussianNBSchema[]
    var gaussian = Gaussian.Model()
    check_default(gaussian)
    _ = gaussian.set_fit_request(sample_weight="training").set_partial_fit_request(
        sample_weight="sample_weight",
    ).set_score_request(sample_weight=False).set_fit_request()
    var fitted_gaussian = gaussian.copy()
    _ = fitted_gaussian.fit(X, y, sample_weight=fit_weights(gaussian, metadata))
    check_archives[Gaussian.Model, Gaussian](gaussian, fitted_gaussian)
    var gaussian_score = fitted_gaussian.score(X, scoring_y, score_weights(fitted_gaussian, metadata))
    var gaussian_direct = fitted_gaussian.score(X, scoring_y, metadata["sample_weight"].copy())
    var gaussian_fit_count = fitted_gaussian.class_count_.item(0)
    _ = fitted_gaussian.partial_fit(X, y, sample_weight=check_partial(fitted_gaussian, metadata))
    print(String(
        "GaussianNB|", gaussian_fit_count, "|", fitted_gaussian.class_count_.item(0),
        "|", gaussian_score, "|", gaussian_direct,
    ))

    comptime Multinomial = MultinomialNBSchema[]
    var multinomial = Multinomial.Model()
    check_default(multinomial)
    _ = multinomial.set_fit_request(sample_weight="training").set_partial_fit_request(
        sample_weight="sample_weight",
    ).set_score_request(sample_weight=False).set_fit_request()
    var fitted_multinomial = multinomial.copy()
    _ = fitted_multinomial.fit(X, y, sample_weight=fit_weights(multinomial, metadata))
    check_archives[Multinomial.Model, Multinomial](multinomial, fitted_multinomial)
    var multinomial_score = fitted_multinomial.score(X, scoring_y, score_weights(fitted_multinomial, metadata))
    var multinomial_direct = fitted_multinomial.score(X, scoring_y, metadata["sample_weight"].copy())
    var multinomial_fit_count = fitted_multinomial.class_count_.item(0)
    _ = fitted_multinomial.partial_fit(X, y, sample_weight=check_partial(fitted_multinomial, metadata))
    print(String(
        "MultinomialNB|", multinomial_fit_count, "|", fitted_multinomial.class_count_.item(0),
        "|", multinomial_score, "|", multinomial_direct,
    ))

    comptime Complement = ComplementNBSchema[]
    var complement = Complement.Model()
    check_default(complement)
    _ = complement.set_fit_request(sample_weight="training").set_partial_fit_request(
        sample_weight="sample_weight",
    ).set_score_request(sample_weight=False).set_fit_request()
    var fitted_complement = complement.copy()
    _ = fitted_complement.fit(X, y, sample_weight=fit_weights(complement, metadata))
    check_archives[Complement.Model, Complement](complement, fitted_complement)
    var complement_score = fitted_complement.score(X, scoring_y, score_weights(fitted_complement, metadata))
    var complement_direct = fitted_complement.score(X, scoring_y, metadata["sample_weight"].copy())
    var complement_fit_count = fitted_complement.class_count_.item(0)
    _ = fitted_complement.partial_fit(X, y, sample_weight=check_partial(fitted_complement, metadata))
    print(String(
        "ComplementNB|", complement_fit_count, "|", fitted_complement.class_count_.item(0),
        "|", complement_score, "|", complement_direct,
    ))

    comptime Bernoulli = BernoulliNBSchema[]
    var bernoulli = Bernoulli.Model()
    check_default(bernoulli)
    _ = bernoulli.set_fit_request(sample_weight="training").set_partial_fit_request(
        sample_weight="sample_weight",
    ).set_score_request(sample_weight=False).set_fit_request()
    var fitted_bernoulli = bernoulli.copy()
    _ = fitted_bernoulli.fit(X, y, sample_weight=fit_weights(bernoulli, metadata))
    check_archives[Bernoulli.Model, Bernoulli](bernoulli, fitted_bernoulli)
    var bernoulli_score = fitted_bernoulli.score(X, scoring_y, score_weights(fitted_bernoulli, metadata))
    var bernoulli_direct = fitted_bernoulli.score(X, scoring_y, metadata["sample_weight"].copy())
    var bernoulli_fit_count = fitted_bernoulli.class_count_.item(0)
    _ = fitted_bernoulli.partial_fit(X, y, sample_weight=check_partial(fitted_bernoulli, metadata))
    print(String(
        "BernoulliNB|", bernoulli_fit_count, "|", fitted_bernoulli.class_count_.item(0),
        "|", bernoulli_score, "|", bernoulli_direct,
    ))

    comptime Categorical = CategoricalNBSchema[]
    var categorical = Categorical.Model()
    check_default(categorical)
    _ = categorical.set_fit_request(sample_weight="training").set_partial_fit_request(
        sample_weight="sample_weight",
    ).set_score_request(sample_weight=False).set_fit_request()
    var fitted_categorical = categorical.copy()
    _ = fitted_categorical.fit(X, y, sample_weight=fit_weights(categorical, metadata))
    check_archives[Categorical.Model, Categorical](categorical, fitted_categorical)
    var categorical_score = fitted_categorical.score(X, scoring_y, score_weights(fitted_categorical, metadata))
    var categorical_direct = fitted_categorical.score(X, scoring_y, metadata["sample_weight"].copy())
    var categorical_fit_count = fitted_categorical.class_count_.item(0)
    _ = fitted_categorical.partial_fit(X, y, sample_weight=check_partial(fitted_categorical, metadata))
    print(String(
        "CategoricalNB|", categorical_fit_count, "|", fitted_categorical.class_count_.item(0),
        "|", categorical_score, "|", categorical_direct,
    ))
