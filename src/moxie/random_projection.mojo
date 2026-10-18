"""Gaussian and sparse random projections in pure Mojo.

Behavior follows sklearn.random_projection; see docs/random_projection/api.md for the native
storage contract and THIRD_PARTY_NOTICES for source attributions.
"""
from std.math import isfinite, sqrt
from std.utils import Variant
from moxie.base import (
    FitTransformer,
    InverseTransformer,
    TransformerMixin,
    UnsupervisedEstimator,
)
from moxie.utils.validation import check_is_fitted
from ._projection.matrix import ProjectionMatrix, project
from moxie._random import RandomState, check_random_state
from ._projection.linalg import pseudoinverse
from ._projection.math import log64


trait RandomProjector(
    FitTransformer, InverseTransformer, TransformerMixin, UnsupervisedEstimator
):
    pass


comptime ComponentCount = Variant[Int, String]
comptime ProjectionDensity = Variant[Float64, String]


def johnson_lindenstrauss_min_dim(
    n_samples: Float64, *, eps: Float64 = 0.1
) raises -> Int:
    """Return the reference's truncated JL bound, including zero for one sample.
    """
    if not isfinite(n_samples) or n_samples < 1:
        raise Error("ValueError: n_samples must be finite and at least one")
    if not isfinite(eps) or eps <= 0 or eps >= 1:
        raise Error("ValueError: the JL bound requires 0 < eps < 1")
    var denominator = eps * eps / 2.0 - eps * eps * eps / 3.0
    var bound = 4.0 * log64(n_samples) / denominator
    if not isfinite(bound) or bound >= 9223372036854775808.0:
        raise Error("ValueError: JL bound exceeds the native Int range")
    return Int(bound)


def johnson_lindenstrauss_min_dim(
    n_samples: List[Float64], *, eps: Float64 = 0.1
) raises -> List[Int]:
    var result = List[Int]()
    # Validate eps even for an empty input.
    _ = johnson_lindenstrauss_min_dim(1.0, eps=eps)
    for count in n_samples:
        result.append(johnson_lindenstrauss_min_dim(count, eps=eps))
    return result^


def johnson_lindenstrauss_min_dim(
    n_samples: Float64, *, eps: List[Float64]
) raises -> List[Int]:
    var result = List[Int]()
    _ = johnson_lindenstrauss_min_dim(n_samples)
    for distortion in eps:
        result.append(johnson_lindenstrauss_min_dim(n_samples, eps=distortion))
    return result^


def johnson_lindenstrauss_min_dim(
    n_samples: List[Float64], *, eps: List[Float64]
) raises -> List[Int]:
    var result = List[Int]()
    for sample_count in n_samples:
        _ = johnson_lindenstrauss_min_dim(sample_count)
    for distortion in eps:
        _ = johnson_lindenstrauss_min_dim(1.0, eps=distortion)
    if len(n_samples) != len(eps) and len(n_samples) != 1 and len(eps) != 1:
        raise Error(
            "ValueError: JL vectors must have equal or broadcastable lengths"
        )
    var count = len(eps) if len(n_samples) == 1 else len(n_samples)
    for i in range(count):
        result.append(
            johnson_lindenstrauss_min_dim(
                n_samples[0 if len(n_samples) == 1 else i],
                eps=eps[0 if len(eps) == 1 else i],
            )
        )
    return result^


def _dimensions(
    n_components: ComponentCount, eps: Float64, rows: Int, cols: Int
) raises -> Int:
    if not isfinite(eps) or eps <= 0:
        raise Error("ValueError: eps must be finite and positive")
    var count: Int
    if n_components.isa[String]():
        if n_components[String] != "auto":
            raise Error(
                "ValueError: n_components must be a positive integer or 'auto'"
            )
        count = johnson_lindenstrauss_min_dim(Float64(rows), eps=eps)
        if count <= 0 or count > cols:
            raise Error(
                "ValueError: automatic target dimension must be positive and no"
                " larger than n_features"
            )
    else:
        count = n_components[Int]
        if count <= 0:
            raise Error("ValueError: n_components must be strictly positive")
    return count


def _density(density: ProjectionDensity, cols: Int) raises -> Float64:
    if density.isa[String]():
        if density[String] != "auto":
            raise Error("ValueError: density must be in (0, 1] or 'auto'")
        return 1.0 / sqrt(Float64(cols))
    var value = density[Float64]
    if not isfinite(value) or value <= 0 or value > 1:
        raise Error("ValueError: density must be in (0, 1]")
    return value


def _gaussian_random_matrix[
    dtype: DType = DType.float64
](
    n_components: Int, n_features: Int, random_state: Optional[Int] = None
) raises -> ProjectionMatrix[dtype]:
    if n_components <= 0 or n_features <= 0:
        raise Error("ValueError: matrix dimensions must be strictly positive")
    var handle = check_random_state(random_state)
    return _gaussian_from_rng[dtype](n_components, n_features, handle[])


def _gaussian_from_rng[dtype: DType](
    n_components: Int, n_features: Int, mut rng: RandomState
) raises -> ProjectionMatrix[dtype]:
    var result = ProjectionMatrix[dtype](n_components, n_features)
    var scale = 1.0 / sqrt(Float64(n_components))
    for i in range(len(result.data)):
        result.data[i] = Scalar[dtype](rng.normal() * scale)
    return result^


def _sparse_random_matrix[
    dtype: DType = DType.float64
](
    n_components: Int,
    n_features: Int,
    var density: ProjectionDensity = String("auto"),
    random_state: Optional[Int] = None,
) raises -> ProjectionMatrix[dtype]:
    if n_components <= 0 or n_features <= 0 or n_features > 4294967295:
        raise Error(
            "ValueError: require positive dimensions and at most 2**32 - 1"
            " features"
        )
    var fraction = _density(density, n_features)
    var handle = check_random_state(random_state)
    return _sparse_from_rng[dtype](n_components, n_features, fraction, handle[])


def _sparse_from_rng[dtype: DType](
    n_components: Int, n_features: Int, fraction: Float64, mut rng: RandomState
) raises -> ProjectionMatrix[dtype]:
    if n_features > 4294967295:
        raise Error("ValueError: at most 2**32 - 1 features are supported")
    var scale = sqrt(1.0 / fraction) / sqrt(Float64(n_components))
    if not isfinite(scale):
        raise Error(
            "ValueError: density produces an unrepresentable projection scale"
        )
    if fraction == 1:
        var result = ProjectionMatrix[dtype](n_components, n_features)
        for i in range(len(result.data)):
            result.data[i] = Scalar[dtype](
                Float64(2 * rng.binomial(1, 0.5) - 1) * scale
            )
        return result^
    var indices = List[Int]()
    var indptr: List[Int] = [0]
    for _ in range(n_components):
        var count = rng.binomial(n_features, fraction)
        var row = rng.sample_without_replacement(n_features, count)
        for j in range(count):
            indices.append(Int(row.unsafe_get(j)))
        indptr.append(len(indices))
    var data = List[Scalar[dtype]]()
    for _ in range(len(indices)):
        data.append(
            Scalar[dtype](Float64(2 * rng.binomial(1, 0.5) - 1) * scale)
        )
    return ProjectionMatrix[dtype].from_csr(
        n_components, n_features, data^, indices^, indptr^
    )


def _feature_names(
    prefix: String,
    count: Int,
    features: Int,
    input_features: Optional[List[String]],
) raises -> List[String]:
    if input_features:
        if len(input_features.value()) != features:
            raise Error(
                "ValueError: input_features length differs from n_features_in_"
            )
    var names = List[String]()
    for i in range(count):
        names.append(prefix + String(i))
    return names^


@fieldwise_init
struct GaussianRandomProjectionParams(Copyable, Movable):
    var n_components: ComponentCount
    var eps: Float64
    var compute_inverse_components: Bool
    var random_state: Optional[Int]


struct GaussianRandomProjection[dtype: DType = DType.float64](
    Copyable, RandomProjector
):
    """GaussianRandomProjection for owned dense and CSR float matrices."""

    comptime Params = GaussianRandomProjectionParams
    comptime Input = ProjectionMatrix[Self.dtype]
    comptime Target = List[Scalar[Self.dtype]]
    comptime Transformed = ProjectionMatrix[Self.dtype]
    var n_components: ComponentCount
    var eps: Float64
    var compute_inverse_components: Bool
    var random_state: Optional[Int]
    var n_components_: Int
    var n_features_in_: Int
    var components_: Self.Input
    var inverse_components_: Optional[Self.Input]
    var _fitted: Bool

    def __init__(
        out self,
        var n_components: ComponentCount = String("auto"),
        *,
        eps: Float64 = 0.1,
        compute_inverse_components: Bool = False,
        random_state: Optional[Int] = None,
    ) raises:
        self.n_components = n_components^
        self.eps = eps
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state
        self.n_components_ = 0
        self.n_features_in_ = 0
        self.components_ = Self.Input()
        self.inverse_components_ = None
        self._fitted = False

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.n_components.copy(),
            self.eps,
            self.compute_inverse_components,
            self.random_state,
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        self.n_components = params.n_components.copy()
        self.eps = params.eps
        self.compute_inverse_components = params.compute_inverse_components
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        n_components: Optional[ComponentCount] = None,
        eps: Optional[Float64] = None,
        compute_inverse_components: Optional[Bool] = None,
        random_state: Optional[Int] = None,
    ) raises -> ref[self] Self:
        """Update supplied values; use Params to reset random_state to None."""
        if n_components:
            self.n_components = n_components.value().copy()
        if eps:
            self.eps = eps.value()
        if compute_inverse_components:
            self.compute_inverse_components = compute_inverse_components.value()
        if random_state:
            self.random_state = random_state
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            n_components=self.n_components.copy(),
            eps=self.eps,
            compute_inverse_components=self.compute_inverse_components,
            random_state=self.random_state,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        var handle = check_random_state(self.random_state)
        self._fit_with_rng(X, handle[])
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, *, mut rng: RandomState
    ) raises -> ref[self] Self:
        if self.random_state:
            raise Error("ValueError: explicit rng requires random_state=None")
        self._fit_with_rng(X, rng)
        return self

    def _fit_with_rng(mut self, X: Self.Input, mut rng: RandomState) raises:
        X.validate(nonempty=True)
        var count = _dimensions(self.n_components, self.eps, X.rows, X.cols)
        var components = _gaussian_from_rng[Self.dtype](
            count, X.cols, rng
        )
        var inverse: Optional[Self.Input] = None
        if self.compute_inverse_components:
            inverse = pseudoinverse(components)
        self.components_ = components^
        self.inverse_components_ = inverse^
        self.n_components_ = count
        self.n_features_in_ = X.cols
        self._fitted = True

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        check_is_fitted(self)
        if X.cols != self.n_features_in_:
            raise Error("ValueError: feature count differs from fit")
        return project(X, self.components_, dense_output=True)

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        check_is_fitted(self)
        if X.cols != self.n_components_:
            raise Error(
                "ValueError: inverse input must have n_components_ columns"
            )
        if self.compute_inverse_components:
            if not self.inverse_components_:
                raise Error(
                    "ValueError: inverse components were not computed; refit or"
                    " disable caching"
                )
            return project(X, self.inverse_components_.value())
        var inverse = pseudoinverse(self.components_)
        return project(X, inverse)

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None
    ) raises -> List[String]:
        check_is_fitted(self)
        return _feature_names(
            "gaussianrandomprojection",
            self.n_components_,
            self.n_features_in_,
            input_features,
        )


@fieldwise_init
struct SparseRandomProjectionParams(Copyable, Movable):
    var n_components: ComponentCount
    var density: ProjectionDensity
    var eps: Float64
    var dense_output: Bool
    var compute_inverse_components: Bool
    var random_state: Optional[Int]


struct SparseRandomProjection[dtype: DType = DType.float64](
    Copyable, RandomProjector
):
    """SparseRandomProjection for owned dense and CSR float matrices."""

    comptime Params = SparseRandomProjectionParams
    comptime Input = ProjectionMatrix[Self.dtype]
    comptime Target = List[Scalar[Self.dtype]]
    comptime Transformed = ProjectionMatrix[Self.dtype]
    var n_components: ComponentCount
    var density: ProjectionDensity
    var eps: Float64
    var dense_output: Bool
    var compute_inverse_components: Bool
    var random_state: Optional[Int]
    var n_components_: Int
    var n_features_in_: Int
    var components_: Self.Input
    var inverse_components_: Optional[Self.Input]
    var _fitted: Bool
    var density_: Float64

    def __init__(
        out self,
        var n_components: ComponentCount = String("auto"),
        *,
        var density: ProjectionDensity = String("auto"),
        eps: Float64 = 0.1,
        dense_output: Bool = False,
        compute_inverse_components: Bool = False,
        random_state: Optional[Int] = None,
    ) raises:
        self.n_components = n_components^
        self.density = density^
        self.eps = eps
        self.dense_output = dense_output
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state
        self.n_components_ = 0
        self.n_features_in_ = 0
        self.components_ = Self.Input()
        self.inverse_components_ = None
        self._fitted = False
        self.density_ = 0

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(
            self.n_components.copy(),
            self.density.copy(),
            self.eps,
            self.dense_output,
            self.compute_inverse_components,
            self.random_state,
        )

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        self.n_components = params.n_components.copy()
        self.density = params.density.copy()
        self.eps = params.eps
        self.dense_output = params.dense_output
        self.compute_inverse_components = params.compute_inverse_components
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        n_components: Optional[ComponentCount] = None,
        density: Optional[ProjectionDensity] = None,
        eps: Optional[Float64] = None,
        dense_output: Optional[Bool] = None,
        compute_inverse_components: Optional[Bool] = None,
        random_state: Optional[Int] = None,
    ) raises -> ref[self] Self:
        """Update supplied values; use Params to reset random_state to None."""
        if n_components:
            self.n_components = n_components.value().copy()
        if density:
            self.density = density.value().copy()
        if eps:
            self.eps = eps.value()
        if dense_output:
            self.dense_output = dense_output.value()
        if compute_inverse_components:
            self.compute_inverse_components = compute_inverse_components.value()
        if random_state:
            self.random_state = random_state
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(
            n_components=self.n_components.copy(),
            density=self.density.copy(),
            eps=self.eps,
            dense_output=self.dense_output,
            compute_inverse_components=self.compute_inverse_components,
            random_state=self.random_state,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self._fitted

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        var handle = check_random_state(self.random_state)
        self._fit_with_rng(X, handle[])
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input, *, mut rng: RandomState
    ) raises -> ref[self] Self:
        if self.random_state:
            raise Error("ValueError: explicit rng requires random_state=None")
        self._fit_with_rng(X, rng)
        return self

    def _fit_with_rng(mut self, X: Self.Input, mut rng: RandomState) raises:
        X.validate(nonempty=True)
        var count = _dimensions(self.n_components, self.eps, X.rows, X.cols)
        var density = _density(self.density, X.cols)
        var components = _sparse_from_rng[Self.dtype](
            count, X.cols, density, rng
        )
        var inverse: Optional[Self.Input] = None
        if self.compute_inverse_components:
            inverse = pseudoinverse(components)
        self.components_ = components^
        self.inverse_components_ = inverse^
        self.n_components_ = count
        self.n_features_in_ = X.cols
        self._fitted = True
        self.density_ = density

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        check_is_fitted(self)
        if X.cols != self.n_features_in_:
            raise Error("ValueError: feature count differs from fit")
        return project(X, self.components_, dense_output=self.dense_output)

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Self.Transformed:
        _ = self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        check_is_fitted(self)
        if X.cols != self.n_components_:
            raise Error(
                "ValueError: inverse input must have n_components_ columns"
            )
        if self.compute_inverse_components:
            if not self.inverse_components_:
                raise Error(
                    "ValueError: inverse components were not computed; refit or"
                    " disable caching"
                )
            return project(X, self.inverse_components_.value())
        var inverse = pseudoinverse(self.components_)
        return project(X, inverse)

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None
    ) raises -> List[String]:
        check_is_fitted(self)
        return _feature_names(
            "sparserandomprojection",
            self.n_components_,
            self.n_features_in_,
            input_features,
        )
