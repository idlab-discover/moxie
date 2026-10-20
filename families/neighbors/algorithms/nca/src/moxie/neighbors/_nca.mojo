"""Dense NeighborhoodComponentsAnalysis; see docs/nca.md for native contracts."""
import numojo as nm
from std.math import isnan
from moxie.base import SupervisedEstimator, TransformerMixin, FitTransformer
from moxie._random import RandomState, entropy_state
from moxie.neighbors._contracts import validate_matrix
from moxie.neighbors._classification_targets import ClassificationTargets, LabelOps
from moxie.neighbors._nca_math import (
    Matrix,
    initialize_spectral,
    loss_gradient,
)

from moxie.neighbors._nca_lbfgs import NCACallback, NCAObjective, minimize, Stop


@fieldwise_init
struct _Objective(NCAObjective):
    var data: Matrix
    var codes: List[Int]

    def __call__(self, x: Matrix) raises -> Tuple[Float64, Matrix]:
        return loss_gradient(x, self.data, self.codes)


struct NoNCACallback(NCACallback):
    def __init__(out self):
        pass

    def __call__(mut self, transformation: Matrix, n_iter: Int) raises:
        pass


struct NCAParams[Callback: NCACallback = NoNCACallback](Copyable, Movable):
    var n_components: Optional[Int]
    var init: String
    var init_matrix: Optional[Matrix]
    var warm_start: Bool
    var max_iter: Int
    var tol: Float64
    var callback: Self.Callback
    var verbose: Int
    var random_state: Optional[Int]

    def __init__(
        out self,
        n_components: Optional[Int],
        init: String,
        init_matrix: Optional[Matrix],
        warm_start: Bool,
        max_iter: Int,
        tol: Float64,
        callback: Self.Callback,
        verbose: Int,
        random_state: Optional[Int],
    ):
        self.n_components = n_components
        self.init = init
        self.init_matrix = init_matrix.copy()
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback.copy()
        self.verbose = verbose
        self.random_state = random_state


struct NeighborhoodComponentsAnalysis[
    dtype: DType = DType.float64, Callback: NCACallback = NoNCACallback, Y: Copyable & Deinitable = nm.NDArray[dtype]
](FitTransformer, SupervisedEstimator, TransformerMixin):
    comptime Input = nm.NDArray[Self.dtype]
    comptime Target = Self.Y
    comptime Transformed = Matrix
    comptime Params = NCAParams[Self.Callback]
    var _params: Self.Params
    var components_: Matrix
    var n_features_in_: Int
    var n_iter_: Int
    var random_state_: Optional[RandomState]

    def __init__(
        out self,
        n_components: Optional[Int] = None,
        *,
        init: String = "auto",
        warm_start: Bool = False,
        max_iter: Int = 50,
        tol: Float64 = 1e-5,
        callback: Self.Callback = Self.Callback(),
        verbose: Int = 0,
        random_state: Optional[Int] = None,
    ) raises:
        self._params = Self.Params(
            n_components,
            init,
            None,
            warm_start,
            max_iter,
            tol,
            callback.copy(),
            verbose,
            random_state,
        )
        self.components_ = nm.zeros[DType.float64](nm.Shape(0, 0))
        self.n_features_in_ = 0
        self.n_iter_ = 0
        self.random_state_ = None

    def __init__(
        out self,
        n_components: Optional[Int] = None,
        *,
        init: Matrix,
        warm_start: Bool = False,
        max_iter: Int = 50,
        tol: Float64 = 1e-5,
        callback: Self.Callback = Self.Callback(),
        verbose: Int = 0,
        random_state: Optional[Int] = None,
    ) raises:
        self = Self(
            n_components,
            init="array",
            warm_start=warm_start,
            max_iter=max_iter,
            tol=tol,
            callback=callback,
            verbose=verbose,
            random_state=random_state,
        )
        self._params.init_matrix = init.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self._params.copy()

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self._params = params.copy()
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self,
        *,
        n_components: Optional[Int] = None,
        init: Optional[String] = None,
        warm_start: Optional[Bool] = None,
        max_iter: Optional[Int] = None,
        tol: Optional[Float64] = None,
        callback: Optional[Self.Callback] = None,
        verbose: Optional[Int] = None,
        random_state: Optional[Int] = None,
    ) raises -> ref[self] Self:
        if n_components:
            self._params.n_components = n_components.value()
        if init:
            self._params.init = init.value()
            self._params.init_matrix = None
        if warm_start:
            self._params.warm_start = warm_start.value()
        if max_iter:
            self._params.max_iter = max_iter.value()
        if tol:
            self._params.tol = tol.value()
        if callback:
            self._params.callback = callback.value().copy()
        if verbose:
            self._params.verbose = verbose.value()
        if random_state:
            self._params.random_state = random_state.value()
        return self

    def set_params[origin: MutOrigin](ref[origin] self, *, init: Matrix) raises -> ref[self] Self:
        self._params.init = "array"
        self._params.init_matrix = init.copy()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self()
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return self.n_features_in_ > 0

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, ref[y_origin] y: Self.Target) raises -> ref[self] Self:
        var rng = entropy_state()
        if self._params.random_state:
            rng = RandomState(self._params.random_state.value())
        self._fit(X, y, rng)
        return self

    def fit[origin: MutOrigin, x_origin: ImmOrigin, y_origin: ImmOrigin](
        ref[origin] self,
        ref[x_origin] X: Self.Input,
        ref[y_origin] y: Self.Target,
        *,
        mut random_state: RandomState,
    ) raises -> ref[self] Self:
        """Explicit mutable-stream adaptation of sklearn's RandomState parameter.
        """
        self._fit(X, y, random_state)
        return self

    def _fit(
        mut self, X: Self.Input, y: Self.Target, mut rng: RandomState
    ) raises:
        comptime assert (
            Self.dtype == DType.float32 or Self.dtype == DType.float64
        )
        ref p = self._params
        if p.max_iter < 1 or isnan(p.tol) or p.tol < 0 or p.verbose < 0:
            raise Error(
                "ValueError: max_iter must be positive; tol and verbose must be"
                " nonnegative"
            )
        if (
            p.init != "auto"
            and p.init != "identity"
            and p.init != "random"
            and p.init != "pca"
            and p.init != "lda"
            and p.init != "array"
        ):
            raise Error(
                "ValueError: init must be auto, identity, random, pca, lda or a"
                " matrix"
            )
        validate_matrix(X, training=True)
        var n = X.shape[0]
        var d = X.shape[1]
        if n < 2:
            raise Error("ValueError: NCA requires at least two samples")
        var target = ClassificationTargets(y,n)
        if target.outputs_2d:
            raise Error(
                "ValueError: NCA requires one-dimensional classification"
                " targets"
            )
        var count = LabelOps[Self.Y].rows(target.classes[0])
        var codes = List[Int]()
        for i in range(n):
            codes.append(Int(target.codes.unsafe_get(i)))
        var k = d
        if p.n_components:
            k = p.n_components.value()
        if k < 1 or k > d:
            raise Error("ValueError: n_components must be in [1, n_features]")
        if (
            p.warm_start
            and self.__sklearn_is_fitted__()
            and self.n_features_in_ != d
        ):
            raise Error("ValueError: warm_start input dimensionality changed")
        if p.init == "array":
            if not p.init_matrix:
                raise Error("ValueError: missing init matrix")
            ref initial = p.init_matrix.value()
            validate_matrix(initial, expected_columns=d)
            if initial.shape[0] > d or (
                p.n_components and initial.shape[0] != k
            ):
                raise Error(
                    "ValueError: init output dimensionality disagrees with"
                    " n_components or exceeds n_features"
                )
        var compact = X.contiguous()
        var data = compact.astype[DType.float64]()
        var A = nm.zeros[DType.float64](nm.Shape(k, d))
        if p.warm_start and self.__sklearn_is_fitted__():
            A = self.components_.copy()
        elif p.init == "array":
            A = p.init_matrix.value().contiguous()
        else:
            var mode = p.init
            if mode == "auto":
                if k <= min(d, count - 1):
                    mode = "lda"
                elif k < min(d, n):
                    mode = "pca"
                else:
                    mode = "identity"
            if mode == "pca" or mode == "lda":
                A = initialize_spectral(data, codes, count, k, mode == "lda")
            elif mode == "random":
                for i in range(A.size):
                    A.unsafe_set(i, rng.standard_normal())
            else:
                for i in range(k):
                    A.unsafe_set(i * d + i, 1)
        if A.shape[0] == 0:
            raise Error(
                "ValueError: NCA initialization produced zero output features"
            )
        var callback = p.callback.copy()
        var objective = _Objective(data^, codes^)
        var result = minimize(objective, A^, callback, p.max_iter, p.tol, p.verbose)
        var converged = (
            result.status == Stop.gradient or result.status == Stop.function
        )
        if p.verbose and not converged:
            print(
                "ConvergenceWarning: [NeighborhoodComponentsAnalysis] NCA did"
                " not converge"
            )
        self.n_iter_ = result.iterations + 1
        self.components_ = result.x.copy()
        self.n_features_in_ = d
        self.random_state_ = rng.copy()
        self._params.callback = callback^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        if not self.__sklearn_is_fitted__():
            raise Error(
                "NotFittedError: NeighborhoodComponentsAnalysis has not been"
                " fitted"
            )
        validate_matrix(X, expected_columns=self.n_features_in_)
        var compact = X.contiguous()
        var data = compact.astype[DType.float64]()
        return data @ self.components_.T()

    def fit_transform(
        mut self, X: Self.Input, y: Self.Target
    ) raises -> Self.Transformed:
        self.fit(X, y)
        return self.transform(X)

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Self.Transformed:
        if not y:
            raise Error("ValueError: NeighborhoodComponentsAnalysis requires y")
        self.fit(X, y.value())
        return self.transform(X)

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None
    ) raises -> List[String]:
        if not self.__sklearn_is_fitted__():
            raise Error(
                "NotFittedError: NeighborhoodComponentsAnalysis has not been"
                " fitted"
            )
        if (
            input_features
            and len(input_features.value()) != self.n_features_in_
        ):
            raise Error(
                "ValueError: input_features length must equal n_features_in_"
            )
        var names = List[String]()
        for i in range(self.components_.shape[0]):
            names.append("neighborhoodcomponentsanalysis" + String(i))
        return names^
