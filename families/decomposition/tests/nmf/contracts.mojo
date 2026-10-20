"""NMF/MiniBatchNMF lifecycle, ownership, invalid settings and dtype contracts."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import abs
from numojo import NDArray, Shape, zeros, NDArrayStrides
from moxie.base import clone
from moxie.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from moxie.decomposition._nmf import NMFComponents, BetaLoss, AlphaH


def counts[dtype: DType](n: Int, p: Int) raises -> NDArray[dtype]:
    var x = zeros[dtype](Shape(n, p))
    for i in range(n):
        for j in range(p): x.unsafe_set(i * p + j, Scalar[dtype](Float64((i * 5 + j * 3) % 7) + Float64(i % 3) * 0.5))
    return x^


def rejects[dtype: DType](var model: NMF[dtype], x: NDArray[dtype]) raises:
    with assert_raises(): model.fit(x)


def check[dtype: DType]() raises:
    var x = counts[dtype](12, 5)
    var model = NMF[dtype](2, random_state=0)
    with assert_raises(): _ = model.transform(x)
    var w = model.fit_transform(x)
    assert_equal(w.shape[0], 12)
    assert_equal(w.shape[1], 2)
    assert_equal(model.n_components_, 2)
    assert_true(model.n_iter_ >= 1)
    assert_true(model.reconstruction_err_ >= 0)
    assert_equal(model.get_feature_names_out()[1], "nmf1")
    for i in range(w.size): assert_true(w.unsafe_get(i) >= 0)
    var first = model.components_.value().unsafe_get(0)
    model.fit(x)
    assert_equal(model.components_.value().unsafe_get(0), first)
    x.unsafe_set(0, 50)
    assert_equal(model.components_.value().unsafe_get(0), first)
    x = counts[dtype](12, 5)
    var projected = model.transform(x)
    assert_equal(model.inverse_transform(projected).shape[1], 5)
    var copy = clone(model)
    assert_false(copy.__sklearn_is_fitted__())
    copy.set_params(solver="mu", beta_loss=BetaLoss(String("kullback-leibler")))
    copy.fit(x.view_with_layout(Shape(12, 5), NDArrayStrides(-5, 1), 55))
    assert_true(copy.__sklearn_is_fitted__())
    var auto = NMF[dtype]()
    auto.fit(x)
    assert_equal(auto.n_components_, 5)  # auto -> n_features
    var negative = x.copy()
    negative.unsafe_set(3, -1)
    rejects(NMF[dtype](2), negative)
    rejects(NMF[dtype](0), x)
    rejects(NMF[dtype](2, solver="pg"), x)
    rejects(NMF[dtype](2, beta_loss=BetaLoss(String("kullback-leibler"))), x)  # cd needs frobenius
    rejects(NMF[dtype](2, solver="mu", beta_loss=BetaLoss(String("itakura-saito"))), x)  # zeros with beta <= 0
    rejects(NMF[dtype](2, init=Optional[String]("svd")), x)
    rejects(NMF[dtype](8, init=Optional[String]("nndsvd")), x)  # k > min(n, p)
    rejects(NMF[dtype](2, init=Optional[String]("custom")), x)  # custom needs W and H
    rejects(NMF[dtype](2, l1_ratio=2.0), x)
    rejects(NMF[dtype](2, alpha_W=-1.0), x)
    rejects(NMF[dtype](2, alpha_H=AlphaH(String("other"))), x)
    rejects(NMF[dtype](2, max_iter=0), x)
    with assert_raises(): _ = model.transform(zeros[dtype](Shape(2, 4)))
    var fixed = non_negative_factorization(x, None, model.components_.copy(), NMFComponents(String("auto")), update_H=False)
    assert_equal(fixed.H.shape[0], 2)
    var mb = MiniBatchNMF[dtype](2, batch_size=5, random_state=0)
    var wb = mb.fit_transform(x)
    assert_equal(wb.shape[1], 2)
    assert_true(mb.n_steps_ >= 1)
    var online = MiniBatchNMF[dtype](2, batch_size=5, random_state=0)
    online.partial_fit(x)
    online.partial_fit(x)
    assert_equal(online.n_steps_, 2)
    with assert_raises(): online.partial_fit(zeros[dtype](Shape(3, 4)))
    assert_equal(online.transform(x).shape[1], 2)
    var bad = MiniBatchNMF[dtype](2, batch_size=0)
    with assert_raises(): bad.fit(x)


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("PASS: NMF contracts in Float32 and Float64")
