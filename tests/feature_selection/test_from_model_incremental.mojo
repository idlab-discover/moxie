import numojo as nm
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from moxie.base import clone, IncrementalTransformer
from moxie.decomposition import IncrementalPCA
from moxie.decomposition._pca import PCA
from moxie.feature_selection import SelectFromModel, IncrementalPCAImportance, IncrementalPCAUpdateParams, FeatureLimitPolicy, IncrementalSelectionEstimator

struct BatchRows(FeatureLimitPolicy):
    def __init__(out self):
        pass
    def resolve[dtype: DType](self, X: nm.NDArray[dtype]) raises -> Int:
        return X.shape[0]

# A routing probe around the real PCA adapter, not a statistical implementation.
@fieldwise_init
struct UpdateMetadata(Copyable, Movable):
    var classes: List[Int]
    var weight: Float64
    var fail: Bool

struct RoutingProbe(IncrementalSelectionEstimator):
    comptime dtype = DType.float64
    comptime Target = nm.NDArray[Self.dtype]
    comptime Params = IncrementalPCAImportance[Self.dtype].Params
    comptime PartialFitParams = UpdateMetadata
    var adapter: IncrementalPCAImportance[Self.dtype]
    var total: Float64

    def __init__(out self):
        self.adapter = IncrementalPCAImportance(IncrementalPCA[Self.dtype](1))
        self.total = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.adapter.get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.adapter.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.adapter = self.adapter.__sklearn_clone__()
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.adapter.__sklearn_is_fitted__()

    def fit_selection(mut self, X: nm.NDArray[Self.dtype], y: Optional[Self.Target], sample_weight: Optional[Self.Target]) raises:
        self.adapter.fit_selection(X, y, sample_weight)

    def partial_fit_selection(mut self, X: nm.NDArray[Self.dtype], y: Optional[Self.Target], params: Optional[Self.PartialFitParams] = None) raises:
        if not params or not y:
            raise Error("routing probe requires target and update metadata")
        assert_equal(params.value().classes, [3, 7])
        assert_equal(y.value().unsafe_get(0), 7)
        self.total += params.value().weight
        self.adapter.partial_fit_selection(X, None)
        if params.value().fail:
            raise Error("routing probe fails after mutating candidate state")

    def n_features(self) raises -> Int:
        return self.adapter.n_features()

    def feature_importances(self) raises -> nm.NDArray[Self.dtype]:
        return self.adapter.feature_importances()

    def uses_l1_threshold(self) -> Bool:
        return False


def routing() raises:
    var X = nm.zeros[DType.float64](nm.Shape(4, 2))
    for i in range(4):
        X.unsafe_set(i * 2, Float64(i))
    var y = nm.full[DType.float64](nm.Shape(4), 7)
    var selector = SelectFromModel(RoutingProbe())
    var params = UpdateMetadata([3, 7], 2, False)
    _ = selector.partial_fit(X, y.copy(), partial_fit_params=params)
    params.weight = 5
    selector.partial_fit_named(X, y.copy(), List[String](), partial_fit_params=params.copy())
    assert_equal(selector.estimator_().total, 7)
    assert_equal(selector.estimator_().adapter.model.n_samples_seen_, 8)
    params.fail = True
    with assert_raises():
        _ = selector.partial_fit(X, y.copy(), partial_fit_params=params)
    assert_equal(selector.estimator_().total, 7)
    assert_equal(selector.estimator_().adapter.model.n_samples_seen_, 8)
    assert_equal(selector.estimator.total, 0)
    var first_failure = SelectFromModel(RoutingProbe())
    with assert_raises():
        _ = first_failure.partial_fit(X, y.copy(), partial_fit_params=params)
    assert_false(first_failure.__sklearn_is_fitted__())


def check[dtype: DType]() raises:
    comptime Adapter = IncrementalPCAImportance[dtype]
    comptime Selector = SelectFromModel[Adapter]
    comptime assert conforms_to(Selector, IncrementalTransformer)
    var X = nm.zeros[dtype](nm.Shape(6, 3))
    for i in range(6):
        X.unsafe_set(i * 3, SIMD[dtype, 1](i * 3))
        X.unsafe_set(i * 3 + 1, SIMD[dtype, 1](i % 2))
        X.unsafe_set(i * 3 + 2, SIMD[dtype, 1](i % 3))
    var pca = PCA[dtype](1)
    var params = pca.get_params()
    assert_equal(pca.set_params(params).fit(X).transform(X).shape[1], 1)
    assert_equal(pca.n_samples_, 6)
    var chained = IncrementalPCA[dtype](1)
    var incremental_params = chained.get_params()
    _ = chained.set_params(incremental_params).partial_fit(X).partial_fit(X, check_input=False)
    assert_equal(chained.n_samples_seen_, 12)
    _ = chained.set_params(whiten=False).fit(X)
    assert_equal(chained.n_samples_seen_, 6)
    var selector = Selector(Adapter(IncrementalPCA[dtype](1)), importance_getter="components_")
    with assert_raises():
        _ = selector.estimator_()
    var names: List[String] = ["a", "b", "c"]
    selector.partial_fit_named(X, None, names)
    assert_equal(selector.estimator_().model.n_samples_seen_, 6)
    assert_false(selector.estimator.__sklearn_is_fitted__())
    assert_equal(selector.get_support(True)[List[Int]], [0])
    assert_equal(selector.get_feature_names_out(), [String("a")])
    _ = selector.partial_fit(X, partial_fit_params=IncrementalPCAUpdateParams(check_input=False))
    assert_equal(selector.estimator_().model.n_samples_seen_, 12)
    assert_equal(selector.get_feature_names_out(), [String("a")])
    var direct = IncrementalPCA[dtype](1)
    _ = direct.partial_fit(X)
    _ = direct.partial_fit(X)
    var fitted = selector.estimator_()
    for i in range(3):
        assert_equal(fitted.model.components_.value().unsafe_get(i), direct.components_.value().unsafe_get(i))
        assert_equal(fitted.model.mean_.value().unsafe_get(i), direct.mean_.value().unsafe_get(i))
    fitted.model.components_.value().unsafe_set(0, 0)
    assert_equal(selector.get_support(True)[List[Int]], [0])
    var fresh = clone(selector)
    assert_false(fresh.__sklearn_is_fitted__())
    _ = fresh.partial_fit(X)
    assert_equal(fresh.estimator_().model.n_samples_seen_, 6)
    # Reject name changes after updating a private candidate; preserve old state.
    with assert_raises():
        selector.partial_fit_named(X, None, [String("b"), String("a"), String("c")])
    assert_equal(selector.estimator_().model.n_samples_seen_, 12)
    with assert_raises():
        _ = selector.partial_fit(nm.zeros[dtype](nm.Shape(4, 2)))
    assert_equal(selector.estimator_().model.n_samples_seen_, 12)
    _ = selector.fit(X)
    assert_equal(selector.estimator_().model.n_samples_seen_, 6)
    selector.partial_fit_named(X, None, names)
    assert_equal(selector.estimator_().model.n_samples_seen_, 12)
    # A fit without names does not acquire them from a later batch.
    assert_equal(selector.get_feature_names_out(), [String("x0")])
    var capped = SelectFromModel(Adapter(IncrementalPCA[dtype](1)), max_features=BatchRows(), importance_getter="components_")
    _ = capped.partial_fit(X)
    var small = nm.ones[dtype](nm.Shape(2, 3))
    _ = capped.partial_fit(small)
    assert_equal(capped.max_features_(), 6)
    _ = capped.fit(small)
    assert_equal(capped.max_features_(), 2)
    var trained = Adapter(IncrementalPCA[dtype](1))
    trained.partial_fit_selection(X, None)
    var prefit = Selector(trained.copy(), prefit=True, importance_getter="components_")
    prefit.partial_fit_named(X, None, names)
    assert_equal(prefit.estimator_().model.n_samples_seen_, 6)
    assert_equal(prefit.get_feature_names_out(), [String("x0")])
    prefit.estimator.model.components_.value().unsafe_set(0, 0)
    _ = prefit.partial_fit(small)
    assert_equal(prefit.estimator_().model.n_samples_seen_, 6)
    assert_equal(prefit.get_support(True)[List[Int]], [0])
    var untrained_prefit = Selector(Adapter(IncrementalPCA[dtype](1)), prefit=True, importance_getter="components_")
    with assert_raises():
        _ = untrained_prefit.partial_fit(X)
    assert_false(untrained_prefit.__sklearn_is_fitted__())
    # Nonprefit starts from constructor configuration, even if it was fitted.
    var reset = Selector(trained.copy(), importance_getter="components_")
    _ = reset.partial_fit(X)
    assert_equal(reset.estimator_().model.n_samples_seen_, 6)
    var auto = Selector(trained.copy(), prefit=True)
    _ = auto.partial_fit(X)
    with assert_raises():
        _ = auto.get_support()

def main() raises:
    routing()
    check[DType.float32]()
    check[DType.float64]()
    print("PASS SelectFromModel incremental contracts")
