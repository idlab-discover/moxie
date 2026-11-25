"""Kernel SVMs inside multiclass wrappers, Pipeline, CV and search.

Each composed result is checked against the same steps fitted by hand, so the
composition adds no behavior of its own (sklearn's Pipeline/CV/search contract).
LIBSVM fitting is deterministic without probability, so the comparisons are exact.
"""
from numojo import Shape, Item
from std.testing import assert_equal, assert_raises, assert_true
from moxie.base import clone
from moxie.multiclass import BinaryAdapter, OneVsOneClassifier, OneVsRestClassifier
from moxie.svm import SVC, SVR
from moxie.preprocessing import StandardScaler
from moxie.pipeline import make_pipeline
from moxie.model_selection import (
    GridSearchCV, IndexArray, IndexSplit, cross_val_score, select_rows,
)
from model_selection_adapter import Dense, Labels, SVCSearch, stratified


def dataset() raises -> Tuple[Dense, Labels]:
    var X = Dense(Shape(60, 3))
    var y = Labels()
    for i in range(60):
        var label = i % 3
        X[Item(i, 0)] = Float64(label) * 1.2 + Float64((i * 7) % 5) / 3
        X[Item(i, 1)] = Float64((i * 5) % 11) / 4 - Float64(label)
        X[Item(i, 2)] = 10.0 + Float64((i * 3) % 7)
        y.append(Float64(label * 2 - 1))
    return (X^, y^)


def assert_same(a: Labels, b: Labels) raises:
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def assert_same(a: Dense, b: Dense) raises:
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_equal(a.item(i), b.item(i))


def check_multiclass(X: Dense, y: Labels) raises:
    var target = IndexArray(Shape(len(y)))
    for i in range(len(y)):
        target[Item(i)] = Int64(y[i])

    var ovr = OneVsRestClassifier(BinaryAdapter(SVC(kernel="linear")))
    with assert_raises(contains="NotFittedError"):
        _ = ovr.predict(X)
    _ = ovr.fit(X, target)
    assert_equal(len(ovr.estimators_), 3)
    var decisions = ovr.decision_function(X)
    assert_equal(decisions.shape, Shape(X.shape[0], 3))
    # OvR trains one ordinary binary SVC per sorted class.
    for column in range(3):
        var label = Int64(2 * column - 1)
        assert_equal(ovr.classes_[Item(column)], label)
        var codes = Labels(capacity=len(y))
        for actual in y:
            codes.append(1.0 if actual == Float64(label) else 0.0)
        var binary = SVC(kernel="linear")
        var margins = binary.fit(X, codes).decision_function(X)
        for i in range(len(y)):
            assert_equal(decisions[Item(i, column)], margins[Item(i)])

    var ovo = OneVsOneClassifier(BinaryAdapter(SVC(kernel="linear")))
    _ = ovo.fit(X, target)
    assert_equal(len(ovo.estimators_), 3)
    assert_true(not ovo.pairwise_indices_)

    # A precomputed component fits on each pair's square training submatrix.
    var precomputed = OneVsOneClassifier(BinaryAdapter(SVC(kernel="precomputed")))
    with assert_raises(contains="square kernel matrix"):
        _ = precomputed.fit(X, target)
    var train_kernel = X @ X.T()
    _ = precomputed.fit(train_kernel, target)
    assert_true(Bool(precomputed.pairwise_indices_))
    ref indices = precomputed.pairwise_indices_.value()
    assert_equal(len(indices), 3)
    for pair in range(3):
        assert_equal(indices[pair].size, 40)

    # Queries keep one column per original training sample and can be rectangular.
    var rows = IndexArray(Shape(7))
    for i in range(rows.size):
        rows[Item(i)] = Int64(i * 5)
    var queries = select_rows(X, rows)
    var query_kernel = select_rows(train_kernel, rows)
    var expected = ovo.predict(queries)
    var actual = precomputed.predict(query_kernel)
    assert_equal(actual.shape, expected.shape)
    for i in range(actual.size):
        assert_equal(actual[Item(i)], expected[Item(i)])


def main() raises:
    var data = dataset()
    var X = data[0].copy()
    var y = data[1].copy()

    # clone copies configuration, not the fitted members; refitting reproduces them.
    var model = SVC(C=3.0, kernel="poly", degree=2, gamma="auto", class_weight="balanced")
    _ = model.fit(X, y)
    var copied = clone(model)
    assert_true(not copied.__sklearn_is_fitted__())
    assert_equal(copied.get_params().C, 3.0)
    assert_equal(copied.get_params().kernel, "poly")
    _ = copied.fit(X, y)
    assert_same(model.predict(X), copied.predict(X))
    assert_same(model.dual_coef_, copied.dual_coef_)
    var fitted_copy = model.copy()
    assert_true(fitted_copy.__sklearn_is_fitted__())
    assert_same(model.predict(X), fitted_copy.predict(X))
    var coefficient = model.dual_coef_[Item(0, 0)]
    fitted_copy.dual_coef_[Item(0, 0)] = coefficient + 1.0
    assert_equal(model.dual_coef_[Item(0, 0)], coefficient)

    # Pipeline: scaling then SVC equals the manual composition.
    var pipe = make_pipeline(StandardScaler(), SVC(C=2.0))
    _ = pipe.fit(X, y)
    var scaler = StandardScaler()
    _ = scaler.fit(X)
    var Xs = scaler.transform(X)
    check_multiclass(Xs, y)
    var manual = SVC(C=2.0)
    _ = manual.fit(Xs, y)
    assert_same(pipe.predict(X), manual.predict(Xs))
    assert_same(pipe.decision_function(X), manual.decision_function(Xs))
    assert_equal(pipe.score(X, y), manual.score(Xs, y))

    # A regression SVR in a pipeline as well.
    var target = Dense(Shape(60))
    for i in range(60):
        target[Item(i)] = X.item(i, 0) * 2 - X.item(i, 1)
    var rpipe = make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05))
    _ = rpipe.fit(X, target)
    var rmanual = SVR(C=10.0, epsilon=0.05)
    _ = rmanual.fit(Xs, target)
    assert_same(rpipe.predict(X), rmanual.predict(Xs))

    # cross_val_score clones the SVC per stratified fold; scores equal manual fits.
    var splits = stratified(y, 3)
    var template = SVC(C=0.5, kernel="linear")
    var scores = cross_val_score_svc(template, X, y, splits)
    assert_equal(scores.size, 3)
    for k in range(3):
        var fold = clone(template)
        _ = fold.fit(select_rows(X, splits[k].train), select_rows(y, splits[k].train))
        assert_equal(scores.item(k), fold.score(select_rows(X, splits[k].test), select_rows(y, splits[k].test)))

    # GridSearchCV over C refits the best configuration on all rows.
    var cs: List[Float64] = [0.01, 1.0, 100.0]
    var grid = Dict[String, List[Float64]]()
    grid["C"] = cs.copy()
    var search = GridSearchCV[SVCSearch[]](SVC(), grid, SVCSearch[](), cv=stratified(y, 3))
    _ = search.fit(X, y, None, None)
    var means = search.cv_results_().mean_test_score.copy()
    var best = 0
    for i in range(3):
        var by_hand = cross_val_score_svc(SVC(C=cs[i]), X, y, splits)
        var mean = (by_hand.item(0) + by_hand.item(1) + by_hand.item(2)) / 3
        assert_equal(means.item(i), mean)
        if mean > means.item(best):
            best = i
    assert_equal(search.best_params_()["C"], cs[best])
    var refit = SVC(C=cs[best])
    _ = refit.fit(X, y)
    assert_same(search.best_estimator_().predict(X), refit.predict(X))
    check_pipeline_search(X, y)
    print("PASS: SVM multiclass, clone, Pipeline, cross_val_score and GridSearchCV composition")


def cross_val_score_svc(template: SVC[], X: Dense, y: Labels, splits: List[IndexSplit]) raises -> Dense:
    var adapter = SVCSearch[]()
    return cross_val_score[SVCSearch[]](template, X, y, adapter, adapter.default_scorer(), splits, None, None)


def check_pipeline_search(X: Dense, y: Labels) raises:
    comptime Search = SVCSearch[True]
    var adapter = Search()
    var template = make_pipeline(StandardScaler(), SVC(C=2.0, gamma=0.2))
    var invalid = Dict[String, Float64]()
    invalid["gamma"] = 1.0
    with assert_raises(contains="Invalid parameter gamma"):
        adapter.configure(template, invalid)

    var folds = stratified(y)
    var scores = cross_val_score[Search](
        template, X, y, adapter, adapter.default_scorer(), folds, None, None,
    )
    assert_equal(scores.size, 5)
    for i in range(len(folds)):
        ref train = folds[i].train
        ref test = folds[i].test
        var scaler = StandardScaler()
        var X_train = select_rows(X, train)
        var train_scaled = scaler.fit(X_train).transform(X_train)
        var model = SVC(C=2.0, gamma=0.2)
        _ = model.fit(train_scaled, select_rows(y, train))
        var test_scaled = scaler.transform(select_rows(X, test))
        assert_equal(scores.item(i), model.score(test_scaled, select_rows(y, test)))

    var grid = Dict[String, List[Float64]]()
    grid["svc__C"] = [0.1, 2.0]
    grid["svc__gamma"] = [0.2, 1.0]
    var search = GridSearchCV[Search](template, grid, adapter)
    _ = search.fit(X, y)
    assert_equal(search.n_splits_(), 5)
    assert_true(not template.__sklearn_is_fitted__())
    var results = search.cv_results_()
    assert_equal(len(results.params), 4)
    var best = 0
    for i in range(len(results.params)):
        var candidate = results.params[i].copy()
        var pipe = make_pipeline(StandardScaler(), SVC(C=candidate["svc__C"], gamma=candidate["svc__gamma"]))
        var expected = cross_val_score[Search](
            pipe, X, y, adapter, adapter.default_scorer(), folds, None, None,
        )
        for fold in range(len(folds)):
            assert_equal(results.scores.test_scores.item(i, fold, 0), expected.item(fold))
        if results.mean_test_score.item(i, 0) > results.mean_test_score.item(best, 0):
            best = i
    assert_equal(search.best_index_(), best)
    var best_params = results.params[best].copy()
    var refit = make_pipeline(StandardScaler(), SVC(C=best_params["svc__C"], gamma=best_params["svc__gamma"]))
    _ = refit.fit(X, y)
    assert_same(search.predict(X), refit.predict(X))
