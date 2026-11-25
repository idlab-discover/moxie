"""Typed search adapter for dense SVC and StandardScaler/SVC examples."""
import numojo as nm
from moxie.model_selection import (
    EstimatorScore, IndexArray, IndexSplit, SearchAdapter, SplitLabels, StratifiedKFold,
    select_rows,
)
from moxie.pipeline import Pipeline
from moxie.preprocessing import StandardScaler
from moxie.svm import SVC

comptime Dense = nm.NDArray[DType.float64]
comptime Labels = List[Float64]
comptime ScaledSVC = Pipeline[StandardScaler[], SVC[]]


def stratified(y: Labels, n_splits: Int = 5) raises -> List[IndexSplit]:
    var labels = Dense(nm.Shape(len(y)))
    for i in range(len(y)):
        labels[nm.Item(i)] = y[i]
    var cursor = StratifiedKFold(n_splits).split(len(y), SplitLabels(labels^))
    var folds = List[IndexSplit]()
    while True:
        var fold = cursor.next()
        if not fold:
            break
        folds.append(fold.value().copy())
    return folds^


struct SVCSearch[scaled: Bool = False](SearchAdapter):
    """Search numeric C and gamma for an SVC on ordinary feature matrices."""

    comptime Estimator = ScaledSVC if Self.scaled else SVC[]
    comptime Scorer = EstimatorScore[Self.Estimator, Dense, Labels]
    comptime Value = Float64

    def __init__(out self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        if X.ndim != 2 or X.shape[0] != len(y):
            raise Error("ValueError: expected an aligned feature matrix and target")
        return len(y)

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return select_rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return select_rows(y, rows)

    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata:
            return select_rows(metadata.value(), rows)
        return None

    def fit(self, mut estimator: Self.Estimator, X: Dense, y: Labels, metadata: Optional[Dense]) raises:
        comptime if Self.scaled:
            if metadata:
                raise Error("ValueError: this pipeline adapter only accepts weights for scoring")
            _ = rebind[ScaledSVC](estimator).fit(X, y)
        else:
            _ = rebind[SVC[]](estimator).fit(X, y, metadata)

    def configure(self, mut estimator: Self.Estimator, params: Dict[String, Float64]) raises:
        comptime if Self.scaled:
            ref pipeline = rebind[ScaledSVC](estimator)
            var config = pipeline.get_params()
            Self._configure_svc(config.estimator, pipeline.estimator_name + "__", params)
            _ = pipeline.set_params(config)
        else:
            ref model = rebind[SVC[]](estimator)
            var config = model.get_params()
            Self._configure_svc(config, "", params)
            _ = model.set_params(config)

    @staticmethod
    def _configure_svc(mut config: SVC[].Params, prefix: String, params: Dict[String, Float64]) raises:
        for key in params.keys():
            if key == prefix + "C":
                config.C = params[key]
            elif key == prefix + "gamma":
                config.gamma = params[key]
            else:
                raise Error("ValueError: Invalid parameter " + key)

    def default_scorer(self) raises -> Self.Scorer:
        return Self.Scorer()

    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        _ = self.n_samples(X, y)
        return stratified(y)
