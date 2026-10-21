import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor
from moxie.tree._utils import TreeRandom
from moxie._random import RandomState

def main() raises:
    var X = nm.zeros[DType.float32](nm.Shape(8, 2))
    var y = nm.zeros[DType.float64](nm.Shape(8, 2))
    for i in range(8):
        X.unsafe_set(i * 2, Float32(i))
        X.unsafe_set(i * 2 + 1, Float32(i % 2))
        y.unsafe_set(i * 2, Float64(i // 4))
        y.unsafe_set(i * 2 + 1, Float64(i % 3))
    var c = DecisionTreeClassifier(random_state=42)
    _ = c.fit(X, y)
    assert_equal(c.n_outputs_, 2)
    assert_equal(c.n_classes_[0], 2)
    assert_equal(c.n_classes_[1], 3)
    with assert_raises():
        _ = c.score(X, y)
    var probabilities = c.predict_proba(X)
    assert_equal(probabilities[0].shape[1], 2)
    assert_equal(probabilities[1].shape[1], 3)
    # Strided features and target rows, including negative feature strides.
    var Xview = X.view_with_layout(nm.Shape(4, 2), nm.NDArrayStrides(4, -1), 1)
    var yview = y.view_with_layout(nm.Shape(4, 2), nm.NDArrayStrides(4, 1), 0)
    _ = c.fit(Xview, yview)
    assert_equal(c.predict(Xview).shape[0], 4)
    # Refit replaces class counts and output dimensionality.
    var single = nm.full[DType.float64](nm.Shape(8), 5)
    _ = c.fit(X, single)
    assert_equal(c.n_outputs_, 1)
    assert_equal(c.n_classes_[0], 1)
    assert_equal(c.get_n_leaves(), 1)
    var zero = nm.zeros[DType.float64](nm.Shape(8))
    with assert_raises():
        _ = c.fit(X, single, zero.copy())
    assert_equal(c.predict(X).unsafe_get(0), 5.0)
    var params = c.get_params()
    params.min_samples_leaf = 0
    _ = c.set_params(params)
    with assert_raises():
        _ = c.fit(X, single)
    var r = DecisionTreeRegressor(random_state=0)
    # Construct signs explicitly because List constructors vary across Mojo versions.
    var rp = r.get_params()
    var signs: List[Int] = [1, 0]
    rp.monotonic_cst = signs^
    _ = r.set_params(rp)
    _ = r.fit(X, single)
    assert_equal(r.score(X, single), 1.0)
    # sklearn maps the deprecated friedman_mse alias in the constructor and rejects it
    # afterwards; fractional min_samples_leaf excludes 1.0 while min_samples_split includes it.
    var friedman = DecisionTreeRegressor(criterion="friedman_mse", random_state=0)
    assert_equal(friedman.get_params().criterion, "squared_error")
    var fp = friedman.get_params()
    fp.criterion = "friedman_mse"
    _ = friedman.set_params(fp)
    with assert_raises(contains="invalid regression criterion"):
        _ = friedman.fit(X, single)
    var leaf_all = DecisionTreeClassifier(min_samples_leaf=Float64(1.0))
    with assert_raises(contains="min_samples_leaf fraction must be in (0,1)"):
        _ = leaf_all.fit(X, single)
    var split_all = DecisionTreeClassifier(min_samples_split=Float64(1.0))
    _ = split_all.fit(X, y)
    assert_true(split_all.tree_.node_count >= 3)
    var tx = TreeRandom()
    tx.state = 0
    assert_equal(tx.draw(), 270369)
    assert_equal(tx.draw(), 67634689)
    print("PASS multioutput, strides, refit, invalid parameters, friedman alias, fraction bounds, failed-fit state, RNG vectors")
