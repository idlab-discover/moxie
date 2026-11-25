"""Emit nested StandardScaler/PCA/SVC search results for composition_parity.py."""
from numojo import NDArray, Shape, Item
from std.testing import assert_equal, assert_false
from moxie.decomposition import PCA
from moxie.svm import SVC
from moxie.preprocessing import StandardScaler
from moxie.pipeline import Pipeline
from moxie.model_selection import (
    DenseSupervisedAdapter, GridSearchCV, cross_val_score, cross_validate,
)

comptime Dense = NDArray[DType.float64]
comptime Labels = List[Float64]
comptime ReducedSVC = Pipeline[StandardScaler[], Pipeline[PCA[], SVC[]]]
comptime ReducedSearch = DenseSupervisedAdapter[ReducedSVC, Labels]


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(30, 3))
    var y = Labels()
    for i in range(30):
        X[Item(i, 0)] = Float64(i % 3) + Float64(i) / 30
        X[Item(i, 1)] = Float64((i * 7) % 5)
        X[Item(i, 2)] = 10.0 * Float64((i * 11) % 4) - Float64(i % 3)
        y.append(Float64(i % 3))
    return (X^, y^)


def configure(mut estimator: ReducedSVC, params: Dict[String, Float64]) raises:
    var config = estimator.get_params()
    for key in params.keys():
        if key == "pca__n_components":
            config.estimator.transformer.n_components = Int(params[key])
        elif key == "svc__C":
            config.estimator.estimator.C = params[key]
        else:
            raise Error("ValueError: Invalid parameter " + key)
    _ = estimator.set_params(config)


def main() raises:
    var data = dataset()
    ref X = data[0]
    ref y = data[1]
    var pipe = Pipeline(StandardScaler(), Pipeline(PCA(), SVC()))
    var adapter = ReducedSearch(configure, classifier=True)
    var validation = cross_validate[ReducedSearch](
        pipe, X, y, adapter, return_train_score=True, return_estimator=True, return_indices=True,
    )
    ref folds = validation.indices.value()
    var svc_folds = DenseSupervisedAdapter[SVC[], Labels]().default_cv(X, y)
    assert_equal(len(folds), 5)
    assert_equal(len(svc_folds), len(folds))
    for fold in range(len(folds)):
        for row in range(folds[fold].test.size):
            assert_equal(folds[fold].test.item(row), svc_folds[fold].test.item(row))
            print("nested_pca_test_index", fold, folds[fold].test.item(row))
    var cv = cross_val_score[ReducedSearch](pipe, X, y, adapter, scoring=adapter.default_scorer())
    for i in range(cv.size):
        assert_equal(cv.item(i), validation.scores.test_scores.item(0, i, 0))
        print("nested_pca_train", validation.scores.train_scores.value().item(0, i, 0))
        print("nested_pca_fitted_score", validation.estimators.value()[i].score(X, y))
        print("nested_pca_cv", cv.item(i))
    var grid = Dict[String, List[Float64]]()
    grid["pca__n_components"] = [1.0, 2.0]
    grid["svc__C"] = [0.1, 10.0]
    var search = GridSearchCV[ReducedSearch](pipe, grid, adapter)
    _ = search.fit(X, y)
    assert_false(pipe.__sklearn_is_fitted__())
    var results = search.cv_results_()
    _ = search.fit(X, y)
    assert_equal(search.cv_results_().scores.test_scores.shape[1], len(folds))
    for c in range(4):
        print("nested_pca_candidate", results.params[c]["pca__n_components"], results.params[c]["svc__C"])
        print("nested_pca_mean", results.mean_test_score.item(c, 0))
        print("nested_pca_rank", results.rank_test_score.item(c, 0))
        for f in range(len(folds)):
            print("nested_pca_fold", results.scores.test_scores.item(c, f, 0))
    print("nested_pca_best", search.best_index_())
    var prediction = search.predict(X)
    for i in range(len(prediction)):
        print("nested_pca_prediction", prediction[i])
    var weights = Dense(Shape(len(y)))
    for i in range(len(y)):
        weights[Item(i)] = Float64(i + 1)
    print("nested_pca_weighted_score", adapter.default_scorer().score(search.best_estimator_(), X, y, weights^))
    print("PASS nested PCA pipeline")
