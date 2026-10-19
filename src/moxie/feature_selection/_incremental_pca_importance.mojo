"""IncrementalPCA loading adapter; numerical fitting stays in decomposition."""
import numojo as nm
from moxie.decomposition import IncrementalPCA
from ._model_importance import IncrementalSelectionEstimator


struct IncrementalPCAUpdateParams(Copyable, Movable):
    var check_input: Bool

    def __init__(out self, *, check_input: Bool = True):
        self.check_input = check_input


struct IncrementalPCAImportance[dt: DType = DType.float64](IncrementalSelectionEstimator):
    """Use with importance_getter='components_' for explicit loading selection."""
    comptime dtype = Self.dt
    comptime Target = nm.NDArray[Self.dtype]
    comptime Params = IncrementalPCA[Self.dtype].Params
    comptime PartialFitParams = IncrementalPCAUpdateParams
    var model: IncrementalPCA[Self.dtype]

    def __init__(out self, var model: IncrementalPCA[Self.dtype]):
        self.model = model^

    def __init__(out self, *, copy: Self):
        # IncrementalPCA is Movable. Reconstruct its owned state without refitting.
        self.model = IncrementalPCA[Self.dtype](copy.model.n_components, whiten=copy.model.whiten, copy=copy.model.copy, batch_size=copy.model.batch_size)
        self.model.components_ = copy.model.components_.copy()
        self.model.singular_values_ = copy.model.singular_values_.copy()
        self.model.explained_variance_ = copy.model.explained_variance_.copy()
        self.model.explained_variance_ratio_ = copy.model.explained_variance_ratio_.copy()
        self.model.mean_ = copy.model.mean_.copy()
        self.model.var_ = copy.model.var_.copy()
        self.model.n_samples_seen_ = copy.model.n_samples_seen_
        self.model.n_features_in_ = copy.model.n_features_in_
        self.model.n_components_ = copy.model.n_components_
        self.model.batch_size_ = copy.model.batch_size_
        self.model.noise_variance_ = copy.model.noise_variance_
        self.model.feature_names_in_ = copy.model.feature_names_in_.copy()

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.model.get_params(deep)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        _ = self.model.set_params(params)
        return self

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.model.__sklearn_clone__())

    def __sklearn_is_fitted__(self) -> Bool:
        return self.model.__sklearn_is_fitted__()

    def fit_selection(mut self, X: nm.NDArray[Self.dtype], y: Optional[Self.Target], sample_weight: Optional[nm.NDArray[Self.dtype]]) raises:
        if sample_weight:
            raise Error("ValueError: IncrementalPCA does not support sample_weight")
        _ = self.model.fit(X, y)

    def partial_fit_selection(mut self, X: nm.NDArray[Self.dtype], y: Optional[Self.Target], params: Optional[Self.PartialFitParams] = None) raises:
        var check_input = True if not params else params.value().check_input
        _ = self.model.partial_fit(X, y, check_input=check_input)

    def n_features(self) raises -> Int:
        if not self.__sklearn_is_fitted__():
            raise Error("NotFittedError: underlying IncrementalPCA is not fitted")
        return self.model.n_features_in_

    def feature_importances(self) raises -> nm.NDArray[Self.dtype]:
        _ = self.n_features()
        return self.model.components_.value().copy()

    def importances(self, getter: String) raises -> nm.NDArray[Self.dtype]:
        if getter != "components_":
            raise Error("ValueError: IncrementalPCA requires importance_getter='components_'")
        return self.feature_importances()

    def uses_l1_threshold(self) -> Bool:
        return False
