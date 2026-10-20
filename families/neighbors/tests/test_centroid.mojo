import numojo as nm
from std.math import abs, isnan
from std.testing import assert_equal, assert_true, assert_raises
from std.sys import argv
from moxie.base import clone
from moxie.neighbors import NearestCentroid
from moxie.neighbors._distance import at


def array[dtype: DType](shape: nm.Shape, values: List[Float64]) raises -> nm.NDArray[dtype]:
    var result = nm.zeros[dtype](shape)
    for i in range(len(values)): result.unsafe_set(i, values[i].cast[dtype]())
    return result^


def contracts[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(6, 2))
    var y = nm.zeros[dtype](nm.Shape(6))
    var values = array[DType.float64](nm.Shape(6, 2), [-2, -1, -1, -1, -1, -2, 1, 1, 1, 2, 2, 1])
    for i in range(12): x.unsafe_set(i, values.unsafe_get(i).cast[dtype]())
    for i in range(6): y.unsafe_set(i, Float64((-1 if i < 3 else 1)).cast[dtype]())
    var model = NearestCentroid[dtype]()
    var params = model.get_params()
    assert_equal(params.metric, "euclidean")
    assert_equal(params.priors, "uniform")
    assert_true(not params.shrink_threshold)
    with assert_raises(): _ = model.predict(x)
    with assert_raises(): _ = model.decision_function(x)
    with assert_raises(): _ = model.predict_proba(x)
    model.fit(x, y)
    assert_equal(model.score(x, y), 1)
    assert_equal(model.n_features_in_, 2)
    assert_equal(model.classes_.shape[0], 2)
    assert_equal(model.classes_.unsafe_get(0), -1)
    assert_equal(model.decision_function(x).ndim, 1)
    assert_equal(model.predict_proba(x).ndim, 2)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    with assert_raises(): _ = copied.predict(x)
    var bad_y = nm.zeros[dtype](nm.Shape(5))
    with assert_raises(): model.fit(x, bad_y)
    assert_equal(model.score(x, y), 1)
    var one_class = nm.zeros[dtype](nm.Shape(6))
    with assert_raises(): model.fit(x, one_class)
    var bad_rank = nm.zeros[dtype](nm.Shape(6, 2))
    with assert_raises(): model.fit(x, bad_rank)
    var invalid = nm.zeros[dtype](nm.Shape(6, 0))
    with assert_raises(): model.fit(invalid, y)
    var wrong_features = nm.zeros[dtype](nm.Shape(2, 3))
    with assert_raises(): _ = model.predict(wrong_features)
    var empty = nm.zeros[dtype](nm.Shape(0, 2))
    with assert_raises(): _ = model.predict(empty)
    var constant = nm.ones[dtype](nm.Shape(6, 2))
    with assert_raises(): model.fit(constant, y)
    var nan_x = x.copy()
    nan_x.unsafe_set(0, Float64("nan").cast[dtype]())
    with assert_raises(): model.fit(nan_x, y)
    with assert_raises(): _ = model.predict(nan_x)
    var fractional = y.copy()
    fractional.unsafe_set(0, 0.5)
    with assert_raises(): model.fit(x, fractional)
    params.metric = "cosine"
    model.set_params(params)
    with assert_raises(): model.fit(x, y)
    params.metric = "manhattan"
    model.set_params(params)
    assert_true(model.__sklearn_is_fitted__())
    model.fit(x, y)
    assert_equal(at(model.centroids_, 0, 0), -1)
    with assert_raises(): _ = model.decision_function(x)
    with assert_raises(): _ = model.predict_proba(x)
    with assert_raises(): _ = model.predict_log_proba(x)
    for threshold in [Float64(0), Float64(-1), Float64("nan"), Float64("inf")]:
        var bad = NearestCentroid[dtype](shrink_threshold=threshold)
        with assert_raises(): bad.fit(x, y)
    var bad_prior = NearestCentroid[dtype](priors="invalid")
    with assert_raises(): bad_prior.fit(x, y)
    var priors = nm.ones[DType.float64](nm.Shape(2))
    priors.unsafe_set(0, 2)
    var weighted = NearestCentroid[dtype](priors=priors)
    priors.unsafe_set(0, 100)
    weighted.fit(x, y)
    assert_true(abs(weighted.class_prior_.unsafe_get(0) - Float64(2)/3) < 1e-12)
    var wp = weighted.get_params()
    wp.prior_probabilities.value().unsafe_set(0, 5)
    assert_equal(weighted.get_params().prior_probabilities.value().unsafe_get(0), 2)
    for value in [Float64(-1), Float64("nan"), Float64("inf")]:
        priors.unsafe_set(0, value)
        var bad = NearestCentroid[dtype](priors=priors)
        with assert_raises(): bad.fit(x, y)
    var zeros = nm.zeros[DType.float64](nm.Shape(2))
    var zero_prior = NearestCentroid[dtype](priors=zeros)
    with assert_raises(): zero_prior.fit(x, y)
    var short_prior = nm.ones[DType.float64](nm.Shape(1))
    var wrong_prior = NearestCentroid[dtype](priors=short_prior)
    with assert_raises(): wrong_prior.fit(x, y)
    var weights = nm.ones[dtype](nm.Shape(6))
    assert_equal(weighted.score(x, y, weights.copy()), 1)
    weights.unsafe_set(0, -1)
    with assert_raises(): _ = weighted.score(x, y, weights.copy())
    # Fit and query views have independent positive/negative strides and offsets.
    var reversed_x = x.view_with_layout(nm.Shape(6, 2), nm.NDArrayStrides(-2, 1), 10)
    var reversed_y = y.view_with_layout(nm.Shape(6), nm.NDArrayStrides(-1), 5)
    var strided = NearestCentroid[dtype]()
    strided.fit(reversed_x, reversed_y)
    assert_equal(strided.score(reversed_x, reversed_y), 1)
    var original_center = at(strided.centroids_, 0, 0)
    x.unsafe_set(0, 999)
    y.unsafe_set(0, 50)
    assert_equal(at(strided.centroids_, 0, 0), original_center)
    var new_x = array[dtype](nm.Shape(4, 1), [0, 2, 8, 10])
    var new_y = array[dtype](nm.Shape(4, 1), [10, 10, 20, 20])
    strided.fit(new_x, new_y)
    assert_equal(strided.n_features_in_, 1)
    assert_equal(strided.classes_.unsafe_get(0), 10)
    assert_equal(strided.predict(new_x).ndim, 1)
    # Duplicate centroids: ascending class wins exact prediction ties.
    var ties_x = array[dtype](nm.Shape(4, 1), [-1, 1, -1, 1])
    var ties_y = array[dtype](nm.Shape(4), [3, 3, -2, -2])
    strided.fit(ties_x, ties_y)
    assert_equal(strided.predict(ties_x).unsafe_get(0), -2)
    # Zero pooled variance remains zero; undefined deviations are not fabricated.
    var zero_x = array[dtype](nm.Shape(4, 1), [0, 0, 2, 2])
    strided.fit(zero_x, ties_y)
    assert_equal(strided.within_class_std_dev_.unsafe_get(0), 0)
    assert_equal(strided.score(zero_x, ties_y), 1)
    var singleton_x = array[dtype](nm.Shape(2, 1), [0, 2])
    var singleton_y = array[dtype](nm.Shape(2), [0, 1])
    strided.fit(singleton_x, singleton_y)
    assert_true(isnan(strided.within_class_std_dev_.unsafe_get(0)))
    assert_equal(strided.score(singleton_x, singleton_y), 1)
    with assert_raises(): _ = strided.predict_proba(singleton_x)


def fixture[dtype: DType](metric: String, threshold: String, prior_mode: String, scenario: Int) raises:
    var n = 13
    var d = 3
    var classes = 3
    if scenario == 0:
        n = 6
        d = 2
        classes = 2
    elif scenario == 1:
        n = 5
        d = 2
        classes = 2
    elif scenario == 3:
        n = 2
        d = 1
        classes = 2
    elif scenario == 4:
        n = 4
        d = 1
        classes = 2
    var x = nm.zeros[dtype](nm.Shape(n, d))
    var y = nm.zeros[dtype](nm.Shape(n))
    for i in range(n):
        var c = (i*i+2*i+1) % classes
        if classes == 3: c = 0 if i < 3 else (1 if i < 7 else 2)
        for j in range(d):
            var value = Float64((i*7+j*3)%17)/5 + Float64(c*2-j)
            if scenario == 5 and j == 0: value = 4
            if scenario == 6 and j == 1: value *= 100
            if scenario == 7: value += Float64(j+1)*123
            x.unsafe_set(i*d+j, value.cast[dtype]())
        y.unsafe_set(i, Float64(c*7-3).cast[dtype]())
    if scenario == 0:
        var data = array[dtype](nm.Shape(6, 2), [-2, -1, -1, -1, -1, -2, 1, 1, 1, 2, 2, 1])
        x = data^
        for i in range(n): y.unsafe_set(i, Float64(-1 if i < 3 else 1).cast[dtype]())
    elif scenario == 1:
        var data = array[dtype](nm.Shape(5, 2), [0, 1, 1, 0, 1, 1, 2, 0, 6, 8])
        x = data^
        for i in range(n): y.unsafe_set(i, Float64(1 if i < 2 else 2).cast[dtype]())
    elif scenario == 3 or scenario == 4:
        for i in range(n):
            x.unsafe_set(i, Float64(0 if i < n//2 else 2).cast[dtype]())
            y.unsafe_set(i, Float64(0 if i < n//2 else 1).cast[dtype]())
    var query = nm.zeros[dtype](nm.Shape(7, d))
    for i in range(7):
        for j in range(d):
            var value = Float64(i*3-j*2)/4
            if scenario == 6 and j == 1: value *= 100
            if scenario == 7: value += Float64(j+1)*123
            query.unsafe_set(i*d+j, value.cast[dtype]())
    var model = NearestCentroid[dtype](metric)
    var params = model.get_params()
    if threshold != "none": params.shrink_threshold = Float64(threshold)
    params.priors = prior_mode
    if prior_mode == "explicit" or prior_mode == "zero":
        var priors = nm.zeros[DType.float64](nm.Shape(classes))
        for c in range(classes): priors.unsafe_set(c, Float64(c+1))
        if prior_mode == "zero": priors.unsafe_set(0, 0)
        params.priors = "explicit"
        params.prior_probabilities = priors^
    model.set_params(params)
    if scenario == 8:
        var xv = x.view_with_layout(nm.Shape(n, d), nm.NDArrayStrides(-d, 1), (n-1)*d)
        var yv = y.view_with_layout(nm.Shape(n), nm.NDArrayStrides(-1), n-1)
        model.fit(xv, yv)
    else:
        model.fit(x, y)
    for c in range(classes):
        print("C", model.classes_.unsafe_get(c))
        print("R", model.class_prior_.unsafe_get(c))
        for j in range(d):
            print("M", at(model.centroids_, c, j))
            print("V", at(model.deviations_, c, j))
    for j in range(d): print("S", model.within_class_std_dev_.unsafe_get(j))
    try:
        var prediction = model.predict(query)
        for i in range(7): print("P", prediction.unsafe_get(i))
    except:
        print("P_ERROR")
    if metric == "euclidean":
        try:
            var decision = model.decision_function(query)
            var proba = model.predict_proba(query)
            var log_proba = model.predict_log_proba(query)
            assert_equal(decision.ndim, 1 if classes == 2 else 2)
            assert_equal(proba.ndim, 2)
            assert_equal(proba.shape[1], classes)
            assert_equal(log_proba.ndim, 2)
            for i in range(decision.size): print("D", decision.unsafe_get(i))
            for i in range(proba.size): print("B", proba.unsafe_get(i))
            for i in range(log_proba.size): print("L", log_proba.unsafe_get(i))
        except:
            print("D_ERROR")


def main() raises:
    var args = argv()
    if len(args) == 6:
        if args[1] == "f32": fixture[DType.float32](args[2], args[3], args[4], Int(args[5]))
        else: fixture[DType.float64](args[2], args[3], args[4], Int(args[5]))
    else:
        contracts[DType.float32]()
        contracts[DType.float64]()
        print("PASS NearestCentroid native contracts")
