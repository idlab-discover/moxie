"""Native GraphicalLassoCV differential test driver."""
from std.sys import argv
from parity_io import Tokens, emit, emit_scalar, emit_error
from moxie.covariance import GraphicalLassoCV
from moxie.covariance._graph_lasso import AlphaGrid
from moxie.model_selection._split import KFold
from moxie.model_selection._indices import IndexSplit
from parity_driver import costs


def run[dtype: DType](mut tokens: Tokens) raises:
    var X = tokens.matrix[dtype]()
    var test = tokens.matrix[dtype]()
    var n = tokens.integer()
    var grid = AlphaGrid(n)
    if tokens.integer():
        var values = List[Float64]()
        for i in range(n):
            values.append(tokens.real())
        grid = AlphaGrid(values)
    var refinements = tokens.integer()
    var folds = tokens.integer()
    var iterations = tokens.integer()
    var custom = tokens.integer()
    var mode = tokens.next()
    for centered in range(2):
        var key = String("cv", centered)
        try:
            var model = GraphicalLassoCV[dtype](alphas=grid.copy(), n_refinements=refinements,
                cv=folds, mode=mode.copy(), max_iter=iterations, assume_centered=centered == 1)
            if custom:
                var cursor = KFold(folds, shuffle=True, random_state=42).split(X.shape[0])
                var splits = List[IndexSplit]()
                while True:
                    var pair = cursor.next()
                    if not pair:
                        break
                    splits.append(pair.value().copy())
                model.fit(X, splits=splits)
            else:
                model.fit(X)
            var results = model.cv_results_()
            emit_scalar(key + "_alpha", model.alpha_())
            emit(key + "_alphas", results.alphas)
            emit(key + "_splits", results.split_test_score)
            emit(key + "_mean", results.mean_test_score)
            emit(key + "_std", results.std_test_score)
            emit(key + "_covariance", model.covariance_())
            emit(key + "_precision", model.get_precision())
            emit(key + "_location", model.location_())
            emit(key + "_costs", costs(model.costs_()))
            emit_scalar(key + "_n_iter", Float64(model.n_iter_()))
            emit_scalar(key + "_score", model.score(test))
            emit(key + "_mahalanobis", model.mahalanobis(test))
        except err:
            emit_error(key, err)


def main() raises:
    var tokens = Tokens(String(argv()[1]))
    if tokens.integer() == 32:
        run[DType.float32](tokens)
    else:
        run[DType.float64](tokens)
