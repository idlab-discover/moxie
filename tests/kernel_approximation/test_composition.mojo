"""Kernel feature maps inside moxie.pipeline: Pipeline with a linear classifier,
FeatureUnion branches, clone, feature names and input feature counts.

Expected names are sklearn's at 4358542a: make_union prefixes each branch's
names with `<lowercased class>__`, so RBFSampler gives `rbfsampler__rbfsampler0`
and AdditiveChi2Sampler `additivechi2sampler__additivechi2sampler_x0_sqrt`.
"""
from numojo import NDArray, Shape, zeros
from moxie.kernel_approximation import RBFSampler, SkewedChi2Sampler, AdditiveChi2Sampler, PolynomialCountSketch, Nystroem
from moxie.linear_model import RidgeClassifier
from moxie.pipeline import Pipeline, make_pipeline, make_union
from moxie.base import clone
from std.testing import assert_equal, assert_true, assert_false, assert_raises


def data[dtype: DType]() raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(12, 3))
    for i in range(12):
        for j in range(3):
            # Nonnegative, distinct, class-dependent values (chi2 maps need x >= 0).
            x.unsafe_set(i * 3 + j, Scalar[dtype](Float64((i * 7 + j * 3) % 11) / 10 + Float64(i % 2)))
    return x^


def labels() raises -> NDArray[DType.int64]:
    var y = zeros[DType.int64](Shape(12))
    for i in range(12):
        y.unsafe_set(i, Int64(i % 2))
    return y^


def same[dtype: DType](a: NDArray[dtype], b: NDArray[dtype]) raises:
    assert_equal(a.ndim, b.ndim)
    assert_equal(a.size, b.size)
    for k in range(a.size):
        assert_equal(a.unsafe_get(k), b.unsafe_get(k))


def pipeline_matches_two_stages[dtype: DType]() raises:
    # Pipeline.fit forwards y only to the classifier (the prefix target type
    # differs), exactly what sklearn's unsupervised map does by ignoring y.
    var x = data[dtype]()
    var y = labels()
    var pipe = make_pipeline(RBFSampler[dtype](gamma=0.5, n_components=20, random_state=3), RidgeClassifier[dtype]())
    assert_equal(pipe.transformer_name, "rbfsampler")
    assert_equal(pipe.estimator_name, "ridgeclassifier")
    _ = pipe.fit(x, y)
    var sampler = RBFSampler[dtype](gamma=0.5, n_components=20, random_state=3)
    var features = sampler.fit_transform(x)
    var model = RidgeClassifier[dtype]()
    _ = model.fit(features, y)
    same(pipe.predict(x), model.predict(features))
    same(pipe.transformer.random_weights_, sampler.random_weights_)
    assert_equal(pipe.score(x, y), model.score(features, y))
    assert_equal(pipe.n_features_in_(), 3)
    # Clone: configuration copied, nothing fitted.
    var fresh = clone(pipe)
    assert_false(fresh.transformer.__sklearn_is_fitted__())
    with assert_raises():
        _ = fresh.n_features_in_()
    # Typed nested set_params then refit replaces the learned basis.
    var params = fresh.get_params()
    params.transformer.n_components = 7
    _ = fresh.set_params(params)
    _ = fresh.fit(x, y)
    assert_equal(fresh.transformer.random_weights_.shape[1], 7)
    assert_equal(pipe.transformer.random_weights_.shape[1], 20)


def nystroem_pipeline[dtype: DType]() raises:
    var x = data[dtype]()
    var y = labels()
    var pipe = Pipeline(Nystroem[dtype](gamma=0.2, n_components=5, random_state=1), RidgeClassifier[dtype](),
                        transformer_name="feature_map", estimator_name="svm")
    _ = pipe.fit(x, y)
    var nys = Nystroem[dtype](gamma=0.2, n_components=5, random_state=1)
    var model = RidgeClassifier[dtype]()
    _ = model.fit(nys.fit_transform(x), y)
    same(pipe.predict(x), model.predict(nys.transform(x)))
    same(pipe.transformer.component_indices_, nys.component_indices_)


def union[dtype: DType]() raises:
    var x = data[dtype]()
    var u = make_union[dtype=dtype](RBFSampler[dtype](n_components=4, random_state=0), AdditiveChi2Sampler[dtype](sample_steps=1))
    with assert_raises():
        _ = u.get_feature_names_out()
    var z = u.fit_transform(x)
    assert_equal(z.shape[0], 12)
    assert_equal(z.shape[1], 4 + 3)
    var rbf = RBFSampler[dtype](n_components=4, random_state=0)
    var a = rbf.fit_transform(x)
    var additive = AdditiveChi2Sampler[dtype](sample_steps=1)
    var b = additive.fit_transform(x)
    for i in range(12):
        for j in range(4):
            assert_equal(z.unsafe_get(i * 7 + j), a.unsafe_get(i * 4 + j))
        for j in range(3):
            assert_equal(z.unsafe_get(i * 7 + 4 + j), b.unsafe_get(i * 3 + j))
    var names = u.get_feature_names_out()
    assert_equal(len(names), 7)
    assert_equal(names[0], "rbfsampler__rbfsampler0")
    assert_equal(names[3], "rbfsampler__rbfsampler3")
    assert_equal(names[4], "additivechi2sampler__additivechi2sampler_x0_sqrt")
    assert_equal(names[6], "additivechi2sampler__additivechi2sampler_x2_sqrt")
    var given: List[String] = ["a", "b", "c"]
    var named = u.get_feature_names_out(given.copy())
    assert_equal(named[4], "additivechi2sampler__additivechi2sampler_a_sqrt")
    assert_equal(u.n_features_in_(), 3)


def counts[dtype: DType]() raises:
    # get_n_features_in mirrors sklearn's n_features_in_ attribute: absent before fit.
    var x = data[dtype]()
    var r = RBFSampler[dtype](random_state=0)
    var s = SkewedChi2Sampler[dtype](random_state=0)
    var a = AdditiveChi2Sampler[dtype]()
    var p = PolynomialCountSketch[dtype](random_state=0)
    var n = Nystroem[dtype](random_state=0, n_components=4)
    with assert_raises():
        _ = r.get_n_features_in()
    with assert_raises():
        _ = s.get_n_features_in()
    with assert_raises():
        _ = a.get_n_features_in()
    with assert_raises():
        _ = p.get_n_features_in()
    with assert_raises():
        _ = n.get_n_features_in()
    # Stateless transform does not record a width, as in sklearn.
    _ = a.transform(x)
    with assert_raises():
        _ = a.get_n_features_in()
    assert_equal(r.fit(x).get_n_features_in(), 3)
    assert_equal(s.fit(x).get_n_features_in(), 3)
    assert_equal(a.fit(x).get_n_features_in(), 3)
    assert_equal(p.fit(x).get_n_features_in(), 3)
    assert_equal(n.fit(x).get_n_features_in(), 3)


def main() raises:
    pipeline_matches_two_stages[DType.float32]()
    pipeline_matches_two_stages[DType.float64]()
    nystroem_pipeline[DType.float32]()
    nystroem_pipeline[DType.float64]()
    union[DType.float32]()
    union[DType.float64]()
    counts[DType.float32]()
    counts[DType.float64]()
    print("PASS kernel feature maps in Pipeline/FeatureUnion: predictions, clone, set_params, names, input counts")
