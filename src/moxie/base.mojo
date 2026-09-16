"""Estimator capabilities following sklearn.base; see docs/interfaces.md."""


trait BaseEstimator(Movable):
    """Typed constructor configuration, unfitted cloning, and fitted-state query.

    Params is an estimator-specific record or mapping, never learned state.
    Concrete constructors retain sklearn keyword names and defaults.
    """

    comptime Params: Copyable

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        ...

    def set_params(mut self, params: Self.Params) raises:
        ...

    def __sklearn_clone__(self) raises -> Self:
        ...

    def __sklearn_is_fitted__(self) -> Bool:
        ...


def clone[E: BaseEstimator](estimator: E) raises -> E:
    """Return an unfitted estimator with independently owned configuration."""
    return estimator.__sklearn_clone__()


trait InputEstimator(BaseEstimator):
    comptime Input: Movable


trait SupervisedEstimator(InputEstimator):
    comptime Target: Movable

    def fit(mut self, X: Self.Input, y: Self.Target) raises:
        ...


trait UnsupervisedEstimator(InputEstimator):
    """Accept an optional target for composition, even when it is ignored."""

    comptime Target: Movable

    def fit(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises:
        ...


trait Predictor(InputEstimator):
    comptime Prediction: Movable

    def predict(self, X: Self.Input) raises -> Self.Prediction:
        ...


trait SupervisedScorer(SupervisedEstimator):
    comptime SampleWeight: Movable

    def score(
        self,
        X: Self.Input,
        y: Self.Target,
        sample_weight: Optional[Self.SampleWeight] = None,
    ) raises -> Float64:
        ...


trait ClassifierMixin(Predictor, SupervisedScorer):
    """Score is accuracy (subset accuracy for multilabel); classes_ orders columns.
    """

    pass


trait RegressorMixin(Predictor, SupervisedScorer):
    """Score is R2 with uniform averaging for multioutput, unless documented."""

    pass


trait TransformerMixin(InputEstimator):
    comptime Transformed: Movable

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        ...


trait FitTransformer(InputEstimator):
    """Separate from transform: some embeddings only support fit_transform."""

    comptime Target: Movable
    comptime Transformed: Movable

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Self.Transformed:
        ...


trait InverseTransformer(TransformerMixin):
    def inverse_transform(self, X: Self.Transformed) raises -> Self.Input:
        ...


trait ProbabilityPredictor(Predictor):
    comptime Probabilities: Movable

    def predict_proba(self, X: Self.Input) raises -> Self.Probabilities:
        ...


trait LogProbabilityPredictor(ProbabilityPredictor):
    def predict_log_proba(self, X: Self.Input) raises -> Self.Probabilities:
        ...


trait DecisionFunction(InputEstimator):
    comptime Decision: Movable

    def decision_function(self, X: Self.Input) raises -> Self.Decision:
        ...


trait SampleScorer(InputEstimator):
    comptime SampleScores: Movable

    def score_samples(self, X: Self.Input) raises -> Self.SampleScores:
        ...


trait DensityMixin(SampleScorer, UnsupervisedEstimator):
    """Score aggregation follows the estimator; score_samples is per sample."""

    def score(
        self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Float64:
        ...


trait ClusterMixin(UnsupervisedEstimator):
    """Fit_predict does not imply prediction of unseen data (e.g. DBSCAN)."""

    comptime Labels: Movable

    def fit_predict(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Self.Labels:
        ...


trait OutlierMixin(DecisionFunction, Predictor, UnsupervisedEstimator):
    """Predict returns +1 for inliers and -1 for outliers; zero is the boundary.
    """

    pass


trait WeightedSupervisedEstimator(SupervisedEstimator):
    """Opt-in fit overload; unweighted fit remains available."""

    comptime SampleWeight: Movable

    def fit(
        mut self,
        X: Self.Input,
        y: Self.Target,
        *,
        sample_weight: Self.SampleWeight,
    ) raises:
        ...


trait IncrementalRegressor(RegressorMixin):
    def partial_fit(mut self, X: Self.Input, y: Self.Target) raises:
        ...


trait IncrementalClassifier(ClassifierMixin):
    comptime Classes: Movable

    def partial_fit(
        mut self,
        X: Self.Input,
        y: Self.Target,
        classes: Optional[Self.Classes] = None,
    ) raises:
        ...


trait IncrementalTransformer(TransformerMixin, UnsupervisedEstimator):
    def partial_fit(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises:
        ...
