"""Emit nested StandardScaler/PCA/SVC search results for composition_parity.py."""
from numojo import NDArray, Shape, Item
from moxie.decomposition import PCA
from moxie.svm import SVC
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline
from moxie.model_selection import (
    EstimatorScore, GridSearchCV, IndexArray, IndexSplit, KFold, SearchAdapter,
    cross_val_score, select_rows,
)

comptime Dense = NDArray[DType.float64]
comptime Labels = List[Float64]
comptime ReducedSVC = Pipeline[StandardScaler[], Pipeline[PCA[], SVC[]]]
comptime ReducedAccuracy = EstimatorScore[ReducedSVC, Dense, Labels]


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(30, 3))
    var y = Labels()
    for i in range(30):
        X[Item(i, 0)] = Float64(i % 3) + Float64(i) / 30
        X[Item(i, 1)] = Float64((i * 7) % 5)
        X[Item(i, 2)] = 10.0 * Float64((i * 11) % 4) - Float64(i % 3)
        y.append(Float64(i % 3))
    return (X^, y^)


def folds(n_samples: Int) raises -> List[IndexSplit]:
    var out = List[IndexSplit]()
    var cursor = KFold(3).split(n_samples)
    while True:
        var pair = cursor.next()
        if not pair:
            break
        out.append(pair.value().copy())
    return out^


struct ReducedSearch(SearchAdapter):
    """Tune PCA and SVC inside a StandardScaler/PCA/SVC pipeline."""
    comptime Scorer = ReducedAccuracy
    comptime Value = Float64

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        if X.ndim != 2 or X.shape[0] != len(y):
            raise Error("ValueError: expected an aligned matrix and target")
        return len(y)

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return select_rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return select_rows(y, rows)

    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata:
            return select_rows(metadata.value(), rows)
        return None

    def fit(self, mut estimator: ReducedSVC, X: Dense, y: Labels, metadata: Optional[Dense]) raises:
        if metadata:
            raise Error("ValueError: this pipeline adapter accepts weights for scoring")
        _ = estimator.fit(X, y)

    def configure(self, mut estimator: ReducedSVC, params: Dict[String, Float64]) raises:
        var config = estimator.get_params()
        for key in params.keys():
            if key == "pca__n_components":
                config.estimator.transformer.n_components = Int(params[key])
            elif key == "svc__C":
                config.estimator.estimator.C = params[key]
            else:
                raise Error("ValueError: Invalid parameter " + key)
        _ = estimator.set_params(config)

    def default_scorer(self) raises -> ReducedAccuracy:
        return ReducedAccuracy()

    def default_cv(self, X: Dense, y: Labels) raises -> List[IndexSplit]:
        return folds(self.n_samples(X, y))


def main() raises:
    var data = dataset()
    ref X = data[0]
    ref y = data[1]
    var pipe = Pipeline(StandardScaler(), Pipeline(PCA(), SVC()))
    var cv = cross_val_score[ReducedSearch](pipe, X, y, ReducedSearch(), ReducedAccuracy(), folds(len(y)), None, None)
    for i in range(cv.size):
        print("nested_pca_cv", cv.item(i))
    var grid = Dict[String, List[Float64]]()
    grid["pca__n_components"] = [1.0, 2.0]
    grid["svc__C"] = [0.1, 10.0]
    var search = GridSearchCV[ReducedSearch](pipe, grid, ReducedSearch())
    _ = search.fit(X, y)
    var results = search.cv_results_()
    for c in range(4):
        print("nested_pca_candidate", results.params[c]["pca__n_components"], results.params[c]["svc__C"])
        print("nested_pca_mean", results.mean_test_score.item(c, 0))
        print("nested_pca_rank", results.rank_test_score.item(c, 0))
        for f in range(3):
            print("nested_pca_fold", results.scores.test_scores.item(c, f, 0))
    print("nested_pca_best", search.best_index_())
    var prediction = search.predict(X)
    for i in range(len(prediction)):
        print("nested_pca_prediction", prediction[i])
    var weights = Dense(Shape(len(y)))
    for i in range(len(y)):
        weights[Item(i)] = Float64(i + 1)
    print("nested_pca_weighted_score", ReducedAccuracy().score(search.best_estimator_(), X, y, weights^))
    print("PASS nested PCA pipeline")
