import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.base import clone
from moxie.neighbors import LocalOutlierFactor
from std.sys import argv

def exercise_fit_warnings[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(4, 1))
    x.unsafe_set(3, 1)
    var model = LocalOutlierFactor[dtype](5)
    assert_equal(len(model.fit_warnings), 0)
    _ = model.fit(x)
    assert_equal(len(model.fit_warnings), 1)
    print("W", "clipped", model.fit_warnings[0].category, model.fit_warnings[0].message)

    _ = model.set_params(n_neighbors=4).fit(x)
    assert_equal(model.n_neighbors_, 3)
    assert_equal(len(model.fit_warnings), 0)

    _ = model.set_params(n_neighbors=2).fit_predict(x)
    assert_equal(len(model.fit_warnings), 1)
    var duplicate = model.fit_warnings[0].copy()
    print("W", "duplicates", duplicate.category, duplicate.message)
    var fresh = clone(model)
    assert_equal(len(fresh.fit_warnings), 0)

    var singleton = nm.zeros[dtype](nm.Shape(1, 1))
    with assert_raises():
        _ = model.fit(singleton)
    assert_equal(model.n_samples_fit_, 4)
    assert_equal(len(model.fit_warnings), 1)
    assert_equal(model.fit_warnings[0].category, duplicate.category)
    assert_equal(model.fit_warnings[0].message, duplicate.message)

    _ = model.set_params(novelty=True).fit(x)
    assert_equal(len(model.fit_warnings), 0)

def exercise[dtype: DType]() raises:
    var x = nm.zeros[dtype](nm.Shape(6,1))
    for i in range(5): x.unsafe_set(i,Float64(i).cast[dtype]())
    x.unsafe_set(5,20)
    var q = nm.zeros[dtype](nm.Shape(2,1))
    q.unsafe_set(0,2.5)
    q.unsafe_set(1,30)
    var model = LocalOutlierFactor[dtype](2,novelty=True)
    with assert_raises(): _ = model.predict(q)
    model.fit(x)
    var scores = model.score_samples(q)
    var decision = model.decision_function(q)
    var labels = model.predict(q)
    for i in range(6): print("F",model.negative_outlier_factor_.unsafe_get(i))
    for i in range(2): print("S",scores.unsafe_get(i),decision.unsafe_get(i),labels.unsafe_get(i))
    assert_equal(model.n_neighbors_,2)
    with assert_raises(): _ = model.fit_predict(x)
    var fresh = clone(model)
    assert_true(not fresh.__sklearn_is_fitted__())
    var detect = LocalOutlierFactor[dtype](2,contamination=0.25)
    var fitted = detect.fit_predict(x)
    print("O",detect.offset_)
    for i in range(6): print("L",fitted.unsafe_get(i))
    with assert_raises(): _ = detect.score_samples(q)
    var singleton = nm.zeros[dtype](nm.Shape(1,1))
    with assert_raises(): detect.fit(singleton)
    assert_equal(detect.n_samples_fit_,6)
    var default_model = LocalOutlierFactor[dtype]()
    default_model.fit(x)
    assert_equal(default_model.n_neighbors_,5)
    var duplicates = nm.zeros[dtype](nm.Shape(4,1))
    default_model.fit(duplicates)
    for i in range(4): assert_equal(Float64(default_model.negative_outlier_factor_.unsafe_get(i)),-1)

def main() raises:
    var args = argv()
    if len(args)>1 and args[1]=="f32":
        exercise[DType.float32]()
        exercise_fit_warnings[DType.float32]()
    else:
        exercise[DType.float64]()
        exercise_fit_warnings[DType.float64]()
