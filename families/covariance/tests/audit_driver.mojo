"""Public method and refit observations; compared with pinned sklearn by audit.py."""
from std.sys import argv
from numojo import NDArray
from parity_io import Tokens, emit, emit_scalar, emit_error
from moxie.base import clone
from moxie.covariance import EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS, MinCovDet, EllipticEnvelope, GraphicalLasso, GraphicalLassoCV
from moxie.covariance import empirical_covariance, shrunk_covariance, log_likelihood, ledoit_wolf, ledoit_wolf_shrinkage, oas, fast_mcd


def helpers[dtype: DType](mut tokens: Tokens) raises:
    var X = tokens.matrix[dtype]()
    var precision = tokens.matrix[dtype]()
    try:
        emit("empirical", empirical_covariance(X))
    except e:
        emit_error("empirical", e)
    try:
        emit("empirical_centered", empirical_covariance(X, assume_centered=True))
    except e:
        emit_error("empirical_centered", e)
    try:
        emit("shrunk", shrunk_covariance(X))
    except e:
        emit_error("shrunk", e)
    try:
        emit_scalar("likelihood", log_likelihood(X, precision))
    except e:
        emit_error("likelihood", e)
    try:
        var result = ledoit_wolf(X)
        emit("lw_covariance", result[0])
        emit_scalar("lw_shrinkage", result[1])
    except e:
        emit_error("lw", e)
    try:
        emit_scalar("lw_coefficient", ledoit_wolf_shrinkage(X))
    except e:
        emit_error("lw_coefficient", e)
    try:
        var result = oas(X)
        emit("oas_covariance", result[0])
        emit_scalar("oas_shrinkage", result[1])
    except e:
        emit_error("oas", e)
    try:
        var result = fast_mcd(X, None, 7)
        emit("mcd_covariance", result.covariance)
        emit("mcd_location", result.location)
        emit("mcd_support", result.support)
        emit("mcd_dist", result.dist)
    except e:
        emit_error("mcd", e)


def run_empirical[dtype: DType](mut tokens: Tokens) raises:
    comptime Model = EmpiricalCovariance[dtype]
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var scenario = tokens.next()
    var model = Model()
    if scenario == "verbose":
        model.fit(X)
        return
    model.fit(X)
    for stage in range(3):
        var prefix = String(stage)
        emit(prefix + "_covariance", model.covariance_())
        emit(prefix + "_location", model.location_())
        emit(prefix + "_precision", model.get_precision())
        emit_scalar(prefix + "_features", Float64(model.n_features_in_()))
        emit_scalar(prefix + "_score", model.score(X))
        emit(prefix + "_mahalanobis", model.mahalanobis(X))
        emit_scalar(prefix + "_error", model.error_norm(model.covariance_()))
        if stage == 0:
            model.set_params(assume_centered=True)
            var copied = clone(model)
            emit_scalar("clone_fitted", Float64(copied.__sklearn_is_fitted__()))
        elif stage == 1:
            try:
                model.fit(Y)
                emit_scalar("refit_succeeded", 1)
            except e:
                emit_error("refit", e)

def run_shrunk[dtype: DType](mut tokens: Tokens) raises:
    comptime Model = ShrunkCovariance[dtype]
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var scenario = tokens.next()
    var model = Model()
    if scenario == "verbose":
        model.fit(X)
        return
    model.fit(X)
    for stage in range(3):
        var prefix = String(stage)
        emit(prefix + "_covariance", model.covariance_())
        emit(prefix + "_location", model.location_())
        emit(prefix + "_precision", model.get_precision())
        emit_scalar(prefix + "_features", Float64(model.n_features_in_()))
        emit_scalar(prefix + "_score", model.score(X))
        emit(prefix + "_mahalanobis", model.mahalanobis(X))
        emit_scalar(prefix + "_error", model.error_norm(model.covariance_()))
        if stage == 0:
            model.set_params(assume_centered=True)
            var copied = clone(model)
            emit_scalar("clone_fitted", Float64(copied.__sklearn_is_fitted__()))
        elif stage == 1:
            if scenario == "parameter":
                model.set_params(shrinkage=-1.0)
            try:
                model.fit(Y)
                emit_scalar("refit_succeeded", 1)
            except e:
                emit_error("refit", e)

def run_ledoit_wolf[dtype: DType](mut tokens: Tokens) raises:
    comptime Model = LedoitWolf[dtype]
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var scenario = tokens.next()
    var model = Model()
    if scenario == "verbose":
        model.fit(X)
        return
    model.fit(X)
    for stage in range(3):
        var prefix = String(stage)
        emit(prefix + "_covariance", model.covariance_())
        emit(prefix + "_location", model.location_())
        emit(prefix + "_precision", model.get_precision())
        emit_scalar(prefix + "_features", Float64(model.n_features_in_()))
        emit_scalar(prefix + "_shrinkage", model.shrinkage_())
        emit_scalar(prefix + "_score", model.score(X))
        emit(prefix + "_mahalanobis", model.mahalanobis(X))
        emit_scalar(prefix + "_error", model.error_norm(model.covariance_()))
        if stage == 0:
            model.set_params(assume_centered=True)
            var copied = clone(model)
            emit_scalar("clone_fitted", Float64(copied.__sklearn_is_fitted__()))
        elif stage == 1:
            if scenario == "parameter":
                model.set_params(block_size=0)
            try:
                model.fit(Y)
                emit_scalar("refit_succeeded", 1)
            except e:
                emit_error("refit", e)

def run_oas[dtype: DType](mut tokens: Tokens) raises:
    comptime Model = OAS[dtype]
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var scenario = tokens.next()
    var model = Model()
    if scenario == "verbose":
        model.fit(X)
        return
    model.fit(X)
    for stage in range(3):
        var prefix = String(stage)
        emit(prefix + "_covariance", model.covariance_())
        emit(prefix + "_location", model.location_())
        emit(prefix + "_precision", model.get_precision())
        emit_scalar(prefix + "_features", Float64(model.n_features_in_()))
        emit_scalar(prefix + "_shrinkage", model.shrinkage_())
        emit_scalar(prefix + "_score", model.score(X))
        emit(prefix + "_mahalanobis", model.mahalanobis(X))
        emit_scalar(prefix + "_error", model.error_norm(model.covariance_()))
        if stage == 0:
            model.set_params(assume_centered=True)
            var copied = clone(model)
            emit_scalar("clone_fitted", Float64(copied.__sklearn_is_fitted__()))
        elif stage == 1:
            try:
                model.fit(Y)
                emit_scalar("refit_succeeded", 1)
            except e:
                emit_error("refit", e)

def run_min_cov_det[dtype: DType](mut tokens: Tokens) raises:
    comptime Model = MinCovDet[dtype]
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var scenario = tokens.next()
    var model = Model()
    model.set_params(random_state=7)
    if scenario == "verbose":
        model.fit(X)
        return
    model.fit(X)
    for stage in range(3):
        var prefix = String(stage)
        emit(prefix + "_covariance", model.covariance_())
        emit(prefix + "_location", model.location_())
        emit(prefix + "_precision", model.get_precision())
        emit_scalar(prefix + "_features", Float64(model.n_features_in_()))
        emit(prefix + "_raw_covariance", model.raw_covariance_())
        emit(prefix + "_raw_location", model.raw_location_())
        emit(prefix + "_raw_support", model.raw_support_())
        emit(prefix + "_support", model.support_())
        emit(prefix + "_dist", model.dist_())
        emit_scalar(prefix + "_score", model.score(X))
        emit(prefix + "_mahalanobis", model.mahalanobis(X))
        emit_scalar(prefix + "_error", model.error_norm(model.covariance_()))
        if stage == 0:
            model.set_params(assume_centered=True)
            var copied = clone(model)
            emit_scalar("clone_fitted", Float64(copied.__sklearn_is_fitted__()))
        elif stage == 1:
            if scenario == "parameter":
                model.set_params(support_fraction=2.0)
            try:
                model.fit(Y)
                emit_scalar("refit_succeeded", 1)
            except e:
                emit_error("refit", e)

def run_elliptic[dtype: DType](mut tokens: Tokens) raises:
    comptime Model = EllipticEnvelope[dtype]
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var scenario = tokens.next()
    var model = Model()
    model.set_params(random_state=7)
    if scenario == "verbose":
        model.fit(X)
        return
    model.fit(X)
    for stage in range(3):
        var prefix = String(stage)
        emit(prefix + "_covariance", model.covariance_())
        emit(prefix + "_location", model.location_())
        emit(prefix + "_precision", model.get_precision())
        emit_scalar(prefix + "_features", Float64(model.n_features_in_()))
        emit(prefix + "_raw_covariance", model.raw_covariance_())
        emit(prefix + "_raw_location", model.raw_location_())
        emit(prefix + "_raw_support", model.raw_support_())
        emit(prefix + "_support", model.support_())
        emit(prefix + "_dist", model.dist_())
        emit_scalar(prefix + "_offset", model.offset_())
        emit(prefix + "_predict", model.predict(X))
        emit(prefix + "_decision", model.decision_function(X))
        emit(prefix + "_samples", model.score_samples(X))
        emit_scalar(prefix + "_score", model.score(X, model.predict(X)))
        emit(prefix + "_mahalanobis", model.mahalanobis(X))
        emit_scalar(prefix + "_error", model.error_norm(model.covariance_()))
        if stage == 0:
            model.set_params(assume_centered=True)
            var copied = clone(model)
            emit_scalar("clone_fitted", Float64(copied.__sklearn_is_fitted__()))
        elif stage == 1:
            if scenario == "parameter":
                model.set_params(contamination=0.7)
            try:
                model.fit(Y)
                emit_scalar("refit_succeeded", 1)
            except e:
                emit_error("refit", e)

def run_graphical[dtype: DType](mut tokens: Tokens) raises:
    comptime Model = GraphicalLasso[dtype]
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var scenario = tokens.next()
    var model = Model()
    if scenario == "verbose":
        model.set_params(verbose=True, max_iter=1)
        model.fit(X)
        return
    model.fit(X)
    for stage in range(3):
        var prefix = String(stage)
        emit(prefix + "_covariance", model.covariance_())
        emit(prefix + "_location", model.location_())
        emit(prefix + "_precision", model.get_precision())
        emit_scalar(prefix + "_features", Float64(model.n_features_in_()))
        emit_scalar(prefix + "_score", model.score(X))
        emit(prefix + "_mahalanobis", model.mahalanobis(X))
        emit_scalar(prefix + "_error", model.error_norm(model.covariance_()))
        emit_scalar(prefix + "_iterations", Float64(model.n_iter_()))
        var costs = model.costs_()
        emit_scalar(prefix + "_cost_count", Float64(len(costs)))
        for i in range(len(costs)):
            emit_scalar(prefix + "_objective" + String(i), costs[i][0])
            emit_scalar(prefix + "_gap" + String(i), costs[i][1])
        if stage == 0:
            model.set_params(assume_centered=True)
            var copied = clone(model)
            emit_scalar("clone_fitted", Float64(copied.__sklearn_is_fitted__()))
        elif stage == 1:
            if scenario == "parameter":
                model.set_params(tol=-1.0)
            try:
                model.fit(Y)
                emit_scalar("refit_succeeded", 1)
            except e:
                emit_error("refit", e)

def run_graphical_cv[dtype: DType](mut tokens: Tokens) raises:
    comptime Model = GraphicalLassoCV[dtype]
    var X = tokens.matrix[dtype]()
    var Y = tokens.matrix[dtype]()
    var scenario = tokens.next()
    var model = Model()
    var grid: List[Float64] = [0.3, 0.1]
    model.set_params(alphas=grid, cv=3)
    if scenario == "verbose":
        model.set_params(verbose=True, max_iter=1)
        model.fit(X)
        return
    model.fit(X)
    for stage in range(3):
        var prefix = String(stage)
        emit(prefix + "_covariance", model.covariance_())
        emit(prefix + "_location", model.location_())
        emit(prefix + "_precision", model.get_precision())
        emit_scalar(prefix + "_features", Float64(model.n_features_in_()))
        emit_scalar(prefix + "_score", model.score(X))
        emit(prefix + "_mahalanobis", model.mahalanobis(X))
        emit_scalar(prefix + "_error", model.error_norm(model.covariance_()))
        emit_scalar(prefix + "_iterations", Float64(model.n_iter_()))
        var costs = model.costs_()
        emit_scalar(prefix + "_cost_count", Float64(len(costs)))
        for i in range(len(costs)):
            emit_scalar(prefix + "_objective" + String(i), costs[i][0])
            emit_scalar(prefix + "_gap" + String(i), costs[i][1])
        emit_scalar(prefix + "_alpha", model.alpha_())
        var results = model.cv_results_()
        emit(prefix + "_alphas", results.alphas)
        emit(prefix + "_mean", results.mean_test_score)
        emit(prefix + "_std", results.std_test_score)
        emit(prefix + "_splits", results.split_test_score)
        if stage == 0:
            model.set_params(assume_centered=True)
            var copied = clone(model)
            emit_scalar("clone_fitted", Float64(copied.__sklearn_is_fitted__()))
        elif stage == 1:
            if scenario == "parameter":
                model.set_params(tol=-1.0)
            try:
                model.fit(Y)
                emit_scalar("refit_succeeded", 1)
            except e:
                emit_error("refit", e)


def dispatch[dtype: DType](mut tokens: Tokens, kind: Int) raises:
    if kind == 0:
        run_empirical[dtype](tokens)
    elif kind == 1:
        run_shrunk[dtype](tokens)
    elif kind == 2:
        run_ledoit_wolf[dtype](tokens)
    elif kind == 3:
        run_oas[dtype](tokens)
    elif kind == 4:
        run_min_cov_det[dtype](tokens)
    elif kind == 5:
        run_elliptic[dtype](tokens)
    elif kind == 6:
        run_graphical[dtype](tokens)
    elif kind == 7:
        run_graphical_cv[dtype](tokens)
    elif kind == 8:
        helpers[dtype](tokens)
    elif kind == 9:
        var empirical = tokens.matrix[dtype]()
        var precision = tokens.matrix[dtype]()
        try:
            emit_scalar("likelihood", log_likelihood(empirical, precision))
        except e:
            emit_error("likelihood", e)


def main() raises:
    var tokens = Tokens(String(argv()[1]))
    var bits = tokens.integer()
    var kind = tokens.integer()
    if bits == 32:
        dispatch[DType.float32](tokens, kind)
    else:
        dispatch[DType.float64](tokens, kind)
