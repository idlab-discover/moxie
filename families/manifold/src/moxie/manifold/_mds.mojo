# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense multidimensional scaling estimator."""
from numojo import NDArray, Shape, zeros
from std.math import isfinite, nan
from std.utils import Variant
from moxie._feature_metadata import InputFeatureCount
from moxie.manifold._base import EmbeddingEstimator
from moxie.manifold._classical_mds import ClassicalMDS
from moxie.manifold._plumbing import validate_matrix
from moxie.manifold._smacof import (
    NormalizedStress, _copy_initial, _initial_warning,
    _row_major_cast, _smacof, _smacof_options, _validate_dissimilarities,
)
from moxie.manifold._t_sne import TSNEMetricParams, _scipy_bray_curtis
from moxie.neighbors._distance import Metric, at
from moxie.warnings import FitWarning


comptime MDSMetric = Variant[String, Bool]
comptime MDSMetricParams = TSNEMetricParams


@fieldwise_init
struct MDSParams[dtype: DType](Copyable, Movable):
    var n_components: Int
    var metric_mds: Bool
    var n_init: Int
    var init: String
    var max_iter: Int
    var verbose: Int
    var eps: Float64
    var n_jobs: Optional[Int]
    var random_state: Optional[Int]
    var dissimilarity: String
    var metric: MDSMetric
    var metric_params: Optional[MDSMetricParams[Self.dtype]]
    var normalized_stress: NormalizedStress


def _pairwise[dtype: DType](
    X: NDArray[dtype], name: String, options: Optional[MDSMetricParams[dtype]],
) raises -> NDArray[dtype]:
    if name == "precomputed":
        _validate_dissimilarities(X)
        return X.copy()
    var params = options.copy().or_else(MDSMetricParams[dtype]())
    if ((params.p or params.w) and name != "minkowski"
        or params.V and name != "seuclidean"
        or params.VI and name != "mahalanobis"
        or params.missing_values and name != "nan_euclidean"):
        raise Error("ValueError: metric_params do not match the selected metric")
    var kernel = Metric[dtype](
        name, params.p.or_else(2.0), params.w, params.V, params.VI,
        params.missing_values.or_else(nan[DType.float64]()),
    )
    kernel.validate_features(X.shape[1])
    var infer_parameters = (name == "seuclidean" and not params.V) or (
        name == "mahalanobis" and not params.VI
    )
    if infer_parameters and not X.is_c_contiguous():
        # The neighbors inference kernels cast their operands with NuMojo astype.
        kernel = kernel.self_query_metric(_row_major_cast[dtype](X))
    else:
        kernel = kernel.self_query_metric(X)
    var n = X.shape[0]
    var dissimilarities = zeros[dtype](Shape(n, n))
    for i in range(n):
        for j in range(i):
            var distance = _scipy_bray_curtis(X, i, j) if name == "braycurtis" else kernel.distance(X, i, X, j)
            var value = distance.cast[dtype]()
            if not isfinite(value):
                raise Error("ValueError: MDS requires finite pairwise distances")
            dissimilarities.unsafe_set(i * n + j, value)
            dissimilarities.unsafe_set(j * n + i, value)
    return dissimilarities^


struct MDS[dtype: DType = DType.float64](EmbeddingEstimator, InputFeatureCount, Copyable):
    """Metric or non-metric multidimensional scaling by stress majorization."""

    comptime Params = MDSParams[Self.dtype]
    comptime Input = NDArray[Self.dtype]
    comptime Target = Self.Input
    comptime Transformed = NDArray[DType.float64]

    var n_components: Int
    var metric_mds: Bool
    var n_init: Int
    var init: String
    var max_iter: Int
    var verbose: Int
    var eps: Float64
    var n_jobs: Optional[Int]
    var random_state: Optional[Int]
    var dissimilarity: String
    var metric: MDSMetric
    var metric_params: Optional[MDSMetricParams[Self.dtype]]
    var normalized_stress: NormalizedStress

    var embedding_: Optional[Self.Transformed]
    var dissimilarity_matrix_: Optional[Self.Input]
    var stress_: Float64
    var n_iter_: Int
    var n_features_in_: Int
    var fit_warnings: List[FitWarning]

    def __init__(
        out self, n_components: Int = 2, *, metric_mds: Bool = True,
        n_init: Int = 1, var init: String = "warn", max_iter: Int = 300,
        verbose: Int = 0, eps: Float64 = 1e-6, n_jobs: Optional[Int] = None,
        random_state: Optional[Int] = None, var dissimilarity: String = "deprecated",
        var metric: MDSMetric = "euclidean",
        var metric_params: Optional[MDSMetricParams[Self.dtype]] = None,
        var normalized_stress: NormalizedStress = "auto",
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components
        self.metric_mds = metric_mds
        self.n_init = n_init
        self.init = init^
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.dissimilarity = dissimilarity^
        self.metric = metric^
        self.metric_params = metric_params^
        self.normalized_stress = normalized_stress^
        self.embedding_ = None
        self.dissimilarity_matrix_ = None
        self.stress_ = 0
        self.n_iter_ = 0
        self.n_features_in_ = 0
        self.fit_warnings = []

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.n_components, self.metric_mds, self.n_init, self.init,
            self.max_iter, self.verbose, self.eps, self.n_jobs, self.random_state,
            self.dissimilarity, self.metric.copy(), self.metric_params.copy(),
            self.normalized_stress.copy(),
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params,
    ) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.metric_mds = params.metric_mds
        self.n_init = params.n_init
        self.init = params.init
        self.max_iter = params.max_iter
        self.verbose = params.verbose
        self.eps = params.eps
        self.n_jobs = params.n_jobs
        self.random_state = params.random_state
        self.dissimilarity = params.dissimilarity
        self.metric = params.metric.copy()
        self.metric_params = params.metric_params.copy()
        self.normalized_stress = params.normalized_stress.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[Int] = None,
        metric_mds: Optional[Bool] = None, n_init: Optional[Int] = None,
        init: Optional[String] = None, max_iter: Optional[Int] = None,
        verbose: Optional[Int] = None, eps: Optional[Float64] = None,
        n_jobs: Optional[Int] = None, random_state: Optional[Int] = None,
        dissimilarity: Optional[String] = None, metric: Optional[String] = None,
        metric_params: Optional[MDSMetricParams[Self.dtype]] = None,
        normalized_stress: Optional[NormalizedStress] = None,
    ) raises -> ref[self] Self:
        if n_components:
            self.n_components = n_components.value()
        if metric_mds:
            self.metric_mds = metric_mds.value()
        if n_init:
            self.n_init = n_init.value()
        if init:
            self.init = init.value()
        if max_iter:
            self.max_iter = max_iter.value()
        if verbose:
            self.verbose = verbose.value()
        if eps:
            self.eps = eps.value()
        if n_jobs:
            self.n_jobs = n_jobs
        if random_state:
            self.random_state = random_state
        if dissimilarity:
            self.dissimilarity = dissimilarity.value()
        if metric:
            self.metric = metric.value()
        if metric_params:
            self.metric_params = metric_params.copy()
        if normalized_stress:
            self.normalized_stress = normalized_stress.value().copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var fresh = Self()
        var params = self.get_params()
        _ = fresh.set_params(params)
        return fresh^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.embedding_)

    def get_n_features_in(self) raises -> Int:
        if not self.embedding_:
            raise Error("AttributeError: MDS has no n_features_in_ before fit")
        return self.n_features_in_

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        _ = self._fit(X, None)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, init_dtype: DType](
        ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None,
        *, init: NDArray[init_dtype],
    ) raises -> ref[self] Self:
        _ = self._fit(X, _copy_initial(init))
        return self

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None,
    ) raises -> Self.Transformed:
        return self._fit(X, None)

    def fit_transform[init_dtype: DType](
        mut self, X: Self.Input, y: Optional[Self.Target] = None, *, init: NDArray[init_dtype],
    ) raises -> Self.Transformed:
        return self._fit(X, _copy_initial(init))

    def _fit(mut self, X: Self.Input, var initial: Optional[Self.Transformed]) raises -> Self.Transformed:
        var warnings = List[FitWarning]()
        if self.init == "warn":
            warnings.append(FitWarning(
                "FutureWarning",
                "The default value of `init` will change from 'random' to 'classical_mds' in 1.10. "
                "To suppress this warning, provide some value of `init`.",
            ))
        elif self.init != "random" and self.init != "classical_mds":
            raise Error("ValueError: init must be random, classical_mds, or warn")
        var metric_name = self.metric[String] if self.metric.isa[String]() else "euclidean"
        if self.dissimilarity != "deprecated":
            if self.dissimilarity != "euclidean" and self.dissimilarity != "precomputed":
                raise Error("ValueError: dissimilarity must be euclidean, precomputed, or deprecated")
            if metric_name != "euclidean":
                raise Error(
                    "ValueError: You provided both `dissimilarity` and `metric`. "
                    "Please use only `metric`."
                )
            warnings.append(FitWarning(
                "FutureWarning",
                "The `dissimilarity` parameter is deprecated and will be removed in 1.10. "
                "Use `metric` instead.",
            ))
            metric_name = self.dissimilarity
        var metric_mds = self.metric_mds
        if self.metric.isa[Bool]():
            metric_mds = self.metric[Bool]
            warnings.append(FitWarning("FutureWarning", String(
                "Use metric_mds=", "True" if metric_mds else "False", " instead of metric=",
                "True" if metric_mds else "False",
                ". The support for metric={True/False} will be dropped in 1.10.",
            )))
        var options = _smacof_options(
            metric_mds, self.n_components, self.n_init, self.max_iter, self.verbose,
            self.eps, self.normalized_stress,
        )
        validate_matrix(X)
        if X.shape[0] == X.shape[1] and metric_name != "precomputed":
            warnings.append(FitWarning(
                "UserWarning", "The provided input is a square matrix. Note that ``fit`` constructs "
                "a dissimilarity matrix from data and will treat rows as samples and columns as features. "
                "To use a pre-computed dissimilarity matrix, set ``metric='precomputed'``.",
            ))
        if metric_name in ["dice", "jaccard", "rogerstanimoto", "russellrao", "sokalsneath", "yule"]:
            warnings.append(FitWarning(
                "DataConversionWarning", "Data was converted to boolean for metric " + metric_name,
            ))
        var distances = _pairwise(X, metric_name, self.metric_params)
        if initial:
            if initial.value().shape[0] != X.shape[0]:
                raise Error("ValueError: init matrix should have one row per sample")
        elif self.init == "classical_mds":
            var classical = ClassicalMDS[Self.dtype](self.n_components, metric="precomputed")
            var coordinates = classical.fit_transform(distances)
            validate_matrix(coordinates)
            initial = coordinates^
        if initial and self.n_init != 1:
            warnings.append(_initial_warning(self.n_init))
        var fitted = _smacof(
            _row_major_cast[DType.float64](distances), initial^, options, self.n_components, self.n_init,
            self.n_jobs, self.random_state,
        )
        var stress = fitted.stress
        var iterations = fitted.n_iter
        var embedding = fitted^.take_embedding()
        var embedding_snapshot = embedding.copy()
        self.embedding_ = embedding_snapshot^
        self.stress_ = stress
        self.n_iter_ = iterations
        self.dissimilarity_matrix_ = distances^
        self.n_features_in_ = X.shape[1]
        self.fit_warnings = warnings^
        return embedding^
