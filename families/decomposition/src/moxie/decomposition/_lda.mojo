"""Dense LatentDirichletAllocation with batch and online variational Bayes.

Follows sklearn.decomposition._lda and _online_lda_fast.pyx (BSD-3-Clause; see
LICENSES/scikit-learn.txt), after Hoffman, Blei & Bach (2010) "Online learning
for latent Dirichlet allocation" and Hoffman's onlineldavb. The digamma here is
the reference's own speed-oriented approximation (Bernardo 1976, AS 103), so
the variational updates match sklearn; it is not a general-purpose digamma.
"""
from numojo import NDArray, Shape, zeros
from std.math import lgamma, abs, isfinite
from std.utils.numerics import nan
from moxie.base import UnsupervisedEstimator
from moxie._random import RandomState
from ._base import DecompositionTransformer
from ._contracts import validate_matrix
from ._matrix import libm_exp as exp, libm_log as log
from ._matrix import get2, set2, get1, set1, contiguous, seeded_state, feature_names, machine_epsilon


def lda_psi[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    """sklearn _online_lda_fast.psi in the input precision (positive arguments)."""
    comptime EULER = 0.577215664901532860606512090082402431
    var x = value
    if x <= 1e-6:
        return Scalar[dtype](-EULER) - Scalar[dtype](1.0) / x
    var result = Scalar[dtype](0)
    while x < 6:
        result -= Scalar[dtype](1.0) / x
        x += 1
    var r = Scalar[dtype](1.0) / x
    result += Scalar[dtype](log(Float64(x))) - Scalar[dtype](0.5) * r
    r = r * r
    result -= r * (Scalar[dtype](1.0 / 12.0) - r * (Scalar[dtype](1.0 / 120.0) - r * Scalar[dtype](1.0 / 252.0)))
    return result


def dirichlet_expectation_2d[dtype: DType](arr: NDArray[dtype]) raises -> NDArray[dtype]:
    """E[log theta] for rows theta ~ Dir(arr): psi(arr) - psi(row sums)."""
    var result = zeros[dtype](Shape(arr.shape[0], arr.shape[1]))
    for i in range(arr.shape[0]):
        var total = Scalar[dtype](0)
        for j in range(arr.shape[1]): total += Scalar[dtype](get2(arr, i, j))
        var psi_total = lda_psi(total)
        for j in range(arr.shape[1]):
            set2(result, i, j, Float64(lda_psi(Scalar[dtype](get2(arr, i, j))) - psi_total))
    return result^


def _exp_matrix[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    var result = zeros[dtype](Shape(x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]): set2(result, i, j, exp(get2(x, i, j)))
    return result^


def update_doc_distribution[dtype: DType](X: NDArray[dtype], exp_topic_word: NDArray[dtype], doc_topic_prior: Float64, max_doc_update_iter: Int, mean_change_tol: Float64, cal_sstats: Bool, mut rng: Optional[RandomState]) raises -> Tuple[NDArray[dtype], NDArray[dtype]]:
    """sklearn _update_doc_distribution for dense X; rng=None initializes with ones.

    Returns (doc_topic_distr (n, k), suff_stats (k, p) or an empty (0, 0) array).
    """
    var n = X.shape[0]
    var p = X.shape[1]
    var k = exp_topic_word.shape[0]
    var doc_topic = zeros[dtype](Shape(n, k))
    for i in range(n):
        for t in range(k):
            if rng:
                set2(doc_topic, i, t, Float64(Scalar[dtype](rng.value().gamma(100.0, 0.01))))
            else:
                set2(doc_topic, i, t, 1)
    var exp_doc_topic = _exp_matrix(dirichlet_expectation_2d(doc_topic))
    var stats = zeros[dtype](Shape(k if cal_sstats else 0, p if cal_sstats else 0))
    var eps = machine_epsilon[dtype]()
    var prior = Scalar[dtype](doc_topic_prior)
    for d in range(n):
        var ids = List[Int]()
        for j in range(p):
            if get2(X, d, j) != 0: ids.append(j)
        var m = len(ids)
        var theta = List[Scalar[dtype]]()
        var expected = List[Scalar[dtype]]()
        for t in range(k):
            theta.append(Scalar[dtype](get2(doc_topic, d, t)))
            expected.append(Scalar[dtype](get2(exp_doc_topic, d, t)))
        var norm_phi = List[Float64](length=m, fill=0)
        for _ in range(max_doc_update_iter):
            var last = theta.copy()
            for w in range(m):
                var total: Float64 = 0
                for t in range(k): total += Float64(expected[t]) * get2(exp_topic_word, t, ids[w])
                norm_phi[w] = Float64(Scalar[dtype](total + eps))
            var sum_theta = Scalar[dtype](0)
            for t in range(k):
                var dot: Float64 = 0
                for w in range(m):
                    dot += Float64(Scalar[dtype](get2(X, d, ids[w]) / norm_phi[w])) * get2(exp_topic_word, t, ids[w])
                # _dirichlet_expectation_1d: add the prior in place, then exp(E[log theta]).
                var dt = Scalar[dtype](Float64(expected[t]) * Float64(Scalar[dtype](dot))) + prior
                theta[t] = dt
                sum_theta += dt
            var psi_total = lda_psi(sum_theta)
            for t in range(k):
                expected[t] = Scalar[dtype](exp(Float64(lda_psi(theta[t]) - psi_total)))
            var change: Float64 = 0
            for t in range(k): change += abs(Float64(last[t]) - Float64(theta[t]))
            if change / Float64(k) < mean_change_tol:
                break
        for t in range(k): set2(doc_topic, d, t, Float64(theta[t]))
        if cal_sstats:
            for w in range(m):
                var total: Float64 = 0
                for t in range(k): total += Float64(expected[t]) * get2(exp_topic_word, t, ids[w])
                var ratio = get2(X, d, ids[w]) / Float64(Scalar[dtype](total + eps))
                for t in range(k):
                    set2(stats, t, ids[w], get2(stats, t, ids[w]) + Float64(expected[t]) * ratio)
    return (doc_topic^, stats^)


@fieldwise_init
struct LatentDirichletAllocationParams(Copyable, Movable):
    """Typed get_params/set_params record; field names follow sklearn."""
    var n_components: Int
    var doc_topic_prior: Optional[Float64]
    var topic_word_prior: Optional[Float64]
    var learning_method: String
    var learning_decay: Float64
    var learning_offset: Float64
    var max_iter: Int
    var batch_size: Int
    var evaluate_every: Int
    var total_samples: Float64
    var perp_tol: Float64
    var mean_change_tol: Float64
    var max_doc_update_iter: Int
    var n_jobs: Optional[Int]
    var verbose: Int
    var random_state: Optional[Int]


struct LatentDirichletAllocation[dtype: DType = DType.float64](DecompositionTransformer, UnsupervisedEstimator):
    """Topic model on dense non-negative document-term counts (n_samples, n_features).

    `components_` holds topic-word variational parameters (not normalized
    probabilities). learning_method "batch" (default) or "online". The E-step
    runs sequentially; n_jobs=None or 1 is supported. Parallel settings are
    rejected because reference worker partitioning changes random draws.
    """
    comptime Params = LatentDirichletAllocationParams
    comptime Input = NDArray[Self.dtype]
    comptime Target = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.dtype]

    var n_components: Int
    var doc_topic_prior: Optional[Float64]
    var topic_word_prior: Optional[Float64]
    var learning_method: String
    var learning_decay: Float64
    var learning_offset: Float64
    var max_iter: Int
    var batch_size: Int
    var evaluate_every: Int
    var total_samples: Float64
    var perp_tol: Float64
    var mean_change_tol: Float64
    var max_doc_update_iter: Int
    var n_jobs: Optional[Int]
    var verbose: Int
    var random_state: Optional[Int]
    var components_: Optional[NDArray[Self.dtype]]
    var exp_dirichlet_component_: Optional[NDArray[Self.dtype]]
    var doc_topic_prior_: Float64
    var topic_word_prior_: Float64
    var n_batch_iter_: Int
    var n_iter_: Int
    var bound_: Float64
    var n_features_in_: Int
    var random_state_: Optional[RandomState]

    def __init__(
        out self, n_components: Int = 10, *, doc_topic_prior: Optional[Float64] = None,
        topic_word_prior: Optional[Float64] = None, learning_method: String = "batch",
        learning_decay: Float64 = 0.7, learning_offset: Float64 = 10.0, max_iter: Int = 10,
        batch_size: Int = 128, evaluate_every: Int = -1, total_samples: Float64 = 1e6,
        perp_tol: Float64 = 1e-1, mean_change_tol: Float64 = 1e-3, max_doc_update_iter: Int = 100,
        n_jobs: Optional[Int] = None, verbose: Int = 0, random_state: Optional[Int] = None,
    ):
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.components_ = None
        self.exp_dirichlet_component_ = None
        self.doc_topic_prior_ = 0
        self.topic_word_prior_ = 0
        self.n_batch_iter_ = 0
        self.n_iter_ = 0
        self.bound_ = 0
        self.n_features_in_ = 0
        self.random_state_ = None

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return Self.Params(self.n_components, self.doc_topic_prior, self.topic_word_prior, self.learning_method,
            self.learning_decay, self.learning_offset, self.max_iter, self.batch_size, self.evaluate_every,
            self.total_samples, self.perp_tol, self.mean_change_tol, self.max_doc_update_iter, self.n_jobs,
            self.verbose, self.random_state)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](ref[origin] self, ref[params_origin] params: Self.Params) raises -> ref[self] Self:
        self.n_components = params.n_components
        self.doc_topic_prior = params.doc_topic_prior
        self.topic_word_prior = params.topic_word_prior
        self.learning_method = params.learning_method
        self.learning_decay = params.learning_decay
        self.learning_offset = params.learning_offset
        self.max_iter = params.max_iter
        self.batch_size = params.batch_size
        self.evaluate_every = params.evaluate_every
        self.total_samples = params.total_samples
        self.perp_tol = params.perp_tol
        self.mean_change_tol = params.mean_change_tol
        self.max_doc_update_iter = params.max_doc_update_iter
        self.n_jobs = params.n_jobs
        self.verbose = params.verbose
        self.random_state = params.random_state
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *, n_components: Optional[Int] = None, learning_method: Optional[String] = None,
        max_iter: Optional[Int] = None, batch_size: Optional[Int] = None, evaluate_every: Optional[Int] = None,
        random_state: Optional[Optional[Int]] = None,
    ) raises -> ref[self] Self:
        """Partial update of common settings; use the Params record for the rest."""
        if n_components: self.n_components = n_components.value()
        if learning_method: self.learning_method = learning_method.value()
        if max_iter: self.max_iter = max_iter.value()
        if batch_size: self.batch_size = batch_size.value()
        if evaluate_every: self.evaluate_every = evaluate_every.value()
        if random_state: self.random_state = random_state.value()
        return self

    def __sklearn_clone__(self) raises -> Self:
        var result = Self(self.n_components)
        result.set_params(self.get_params())
        return result^

    def __sklearn_is_fitted__(self) -> Bool:
        return Bool(self.components_)

    def _validate_params(self) raises:
        if self.n_components <= 0:
            raise Error("ValueError: LatentDirichletAllocation n_components must be > 0, got " + String(self.n_components))
        for prior in [self.doc_topic_prior, self.topic_word_prior]:
            if prior and not (prior.value() >= 0 and prior.value() <= 1):
                raise Error("ValueError: LatentDirichletAllocation priors must be in [0, 1] or None")
        if self.learning_method != "batch" and self.learning_method != "online":
            raise Error("ValueError: LatentDirichletAllocation learning_method must be 'batch' or 'online', got '" + self.learning_method + "'")
        if not (self.learning_decay >= 0 and self.learning_decay <= 1):
            raise Error("ValueError: LatentDirichletAllocation learning_decay must be in [0, 1]")
        if not (self.learning_offset >= 1 and isfinite(self.learning_offset)):
            raise Error("ValueError: LatentDirichletAllocation learning_offset must be >= 1")
        if self.max_iter < 0 or self.max_doc_update_iter < 0:
            raise Error("ValueError: LatentDirichletAllocation max_iter and max_doc_update_iter must be >= 0")
        if self.batch_size <= 0:
            raise Error("ValueError: LatentDirichletAllocation batch_size must be > 0, got " + String(self.batch_size))
        if not (self.total_samples > 0 and isfinite(self.total_samples)):
            raise Error("ValueError: LatentDirichletAllocation total_samples must be > 0")
        if not (self.perp_tol >= 0) or not (self.mean_change_tol >= 0):
            raise Error("ValueError: LatentDirichletAllocation perp_tol and mean_change_tol must be >= 0")
        if self.n_jobs and self.n_jobs.value() != 1:
            raise Error("NotImplementedError: LatentDirichletAllocation supports n_jobs=None or 1 only")
        if self.verbose < 0:
            raise Error("ValueError: LatentDirichletAllocation verbose must be >= 0")

    def _check_counts(self, X: Self.Input, whom: String) raises -> NDArray[Self.dtype]:
        validate_matrix(X, nonnegative=False)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if get2(X, i, j) < 0:
                    raise Error("ValueError: Negative values in data passed to " + whom)
        return contiguous(X)

    def _init_latent_vars(mut self, n_features: Int) raises:
        var rng = seeded_state(self.random_state, "LatentDirichletAllocation")
        self.n_batch_iter_ = 1
        self.n_iter_ = 0
        self.doc_topic_prior_ = self.doc_topic_prior.value() if self.doc_topic_prior else 1.0 / Float64(self.n_components)
        self.topic_word_prior_ = self.topic_word_prior.value() if self.topic_word_prior else 1.0 / Float64(self.n_components)
        var components = zeros[Self.dtype](Shape(self.n_components, n_features))
        for t in range(self.n_components):
            for j in range(n_features): set2(components, t, j, rng.gamma(100.0, 0.01))
        self.exp_dirichlet_component_ = _exp_matrix(dirichlet_expectation_2d(components))
        self.components_ = components^
        self.random_state_ = rng^
        self.n_features_in_ = n_features

    def _e_step(mut self, X: NDArray[Self.dtype], cal_sstats: Bool, random_init: Bool) raises -> Tuple[NDArray[Self.dtype], NDArray[Self.dtype]]:
        var rng: Optional[RandomState] = None
        if random_init: rng = self.random_state_.take()
        var result = update_doc_distribution(X, self.exp_dirichlet_component_.value(), self.doc_topic_prior_,
            self.max_doc_update_iter, self.mean_change_tol, cal_sstats, rng)
        if random_init: self.random_state_ = rng^
        var stats = result[1].copy()
        if cal_sstats:
            for t in range(stats.shape[0]):
                for j in range(stats.shape[1]):
                    set2(stats, t, j, get2(stats, t, j) * get2(self.exp_dirichlet_component_.value(), t, j))
        return (result[0].copy(), stats^)

    def _e_step_fixed(self, X: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        var none: Optional[RandomState] = None
        return update_doc_distribution(X, self.exp_dirichlet_component_.value(), self.doc_topic_prior_,
            self.max_doc_update_iter, self.mean_change_tol, False, none)[0].copy()

    def _em_step(mut self, X: NDArray[Self.dtype], total_samples: Float64, batch_update: Bool) raises:
        var stats = self._e_step(X, True, True)[1].copy()
        var components = self.components_.value().copy()
        if batch_update:
            for t in range(components.shape[0]):
                for j in range(components.shape[1]):
                    set2(components, t, j, self.topic_word_prior_ + get2(stats, t, j))
        else:
            var weight = (self.learning_offset + Float64(self.n_batch_iter_)) ** (-self.learning_decay)
            var doc_ratio = total_samples / Float64(X.shape[0])
            for t in range(components.shape[0]):
                for j in range(components.shape[1]):
                    var decayed = Float64(Scalar[Self.dtype](get2(components, t, j) * (1 - weight)))
                    set2(components, t, j, decayed + weight * (self.topic_word_prior_ + doc_ratio * get2(stats, t, j)))
        self.exp_dirichlet_component_ = _exp_matrix(dirichlet_expectation_2d(components))
        self.components_ = components^
        self.n_batch_iter_ += 1

    def fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        """Batch or online variational Bayes over max_iter passes."""
        self._validate_params()
        var x = self._check_counts(X, "LatentDirichletAllocation.fit")
        var n = x.shape[0]
        self._init_latent_vars(x.shape[1])
        var last_bound: Float64 = 0
        for i in range(self.max_iter):
            if self.learning_method == "online":
                var start = 0
                while start < n:
                    var stop = min(n, start + self.batch_size)
                    self._em_step(_rows(x, start, stop), Float64(n), False)
                    start = stop
            else:
                self._em_step(x, Float64(n), True)
            if self.evaluate_every > 0 and (i + 1) % self.evaluate_every == 0:
                var distribution = self._e_step_fixed(x)
                var bound = self._perplexity(x, distribution, False)
                if last_bound != 0 and abs(last_bound - bound) < self.perp_tol:
                    break
                last_bound = bound
            self.n_iter_ += 1
        self.bound_ = self._perplexity(x, self._e_step_fixed(x), False)
        return self

    def partial_fit[origin: MutOrigin, x_origin: ImmOrigin](ref[origin] self, ref[x_origin] X: Self.Input, y: Optional[Self.Target] = None) raises -> ref[self] Self:
        """Online update on X in batch_size chunks; the first call initializes the model."""
        self._validate_params()
        var x = self._check_counts(X, "LatentDirichletAllocation.partial_fit")
        if not self.components_:
            self._init_latent_vars(x.shape[1])
        if x.shape[1] != self.components_.value().shape[1]:
            raise Error("ValueError: The provided data has " + String(x.shape[1])
                + " dimensions while the model was trained with feature size " + String(self.components_.value().shape[1]) + ".")
        var start = 0
        while start < x.shape[0]:
            var stop = min(x.shape[0], start + self.batch_size)
            self._em_step(_rows(x, start, stop), self.total_samples, False)
            start = stop
        return self

    def _check_fitted_input(self, X: Self.Input, whom: String) raises -> NDArray[Self.dtype]:
        if not self.components_:
            raise Error("NotFittedError: This LatentDirichletAllocation instance is not fitted yet. Call 'fit' first.")
        var x = self._check_counts(X, whom)
        if x.shape[1] != self.n_features_in_:
            raise Error("ValueError: X has " + String(x.shape[1]) + " features, but LatentDirichletAllocation is expecting "
                + String(self.n_features_in_) + " features as input.")
        return x^

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        """Normalized document-topic distribution (rows sum to 1)."""
        return self.transform(X, normalize=True)

    def transform(self, X: Self.Input, *, normalize: Bool) raises -> Self.Transformed:
        var x = self._check_fitted_input(X, "LatentDirichletAllocation.transform")
        var distribution = self._e_step_fixed(x)
        if normalize:
            for i in range(distribution.shape[0]):
                var total: Float64 = 0
                for t in range(distribution.shape[1]): total += get2(distribution, i, t)
                for t in range(distribution.shape[1]): set2(distribution, i, t, get2(distribution, i, t) / total)
        return distribution^

    def fit_transform(mut self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Self.Transformed:
        self.fit(X)
        return self.transform(X)

    def fit_transform(mut self, X: Self.Input, *, normalize: Bool) raises -> Self.Transformed:
        self.fit(X)
        return self.transform(X, normalize=normalize)

    def _approx_bound(self, X: NDArray[Self.dtype], distribution: NDArray[Self.dtype], sub_sampling: Bool) raises -> Float64:
        """Variational ELBO as in sklearn _approx_bound (dense)."""
        var n = distribution.shape[0]
        var k = distribution.shape[1]
        var components = self.components_.value().copy()
        var p = components.shape[1]
        var dirichlet_doc = dirichlet_expectation_2d(distribution)
        var dirichlet_comp = dirichlet_expectation_2d(components)
        var score: Float64 = 0
        for d in range(n):
            for j in range(p):
                var count = get2(X, d, j)
                if count == 0: continue
                var largest = get2(dirichlet_doc, d, 0) + get2(dirichlet_comp, 0, j)
                for t in range(1, k): largest = max(largest, get2(dirichlet_doc, d, t) + get2(dirichlet_comp, t, j))
                var total: Float64 = 0
                for t in range(k): total += exp(get2(dirichlet_doc, d, t) + get2(dirichlet_comp, t, j) - largest)
                score += count * (largest + log(total))
        score += _loglikelihood(self.doc_topic_prior_, distribution, dirichlet_doc, k)
        if sub_sampling:
            score *= self.total_samples / Float64(n)
        score += _loglikelihood(self.topic_word_prior_, components, dirichlet_comp, p)
        return score

    def _perplexity(self, X: NDArray[Self.dtype], distribution: NDArray[Self.dtype], sub_sampling: Bool) raises -> Float64:
        var bound = self._approx_bound(X, distribution, sub_sampling)
        var word_count: Float64 = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]): word_count += get2(X, i, j)
        if sub_sampling:
            word_count *= self.total_samples / Float64(X.shape[0])
        # Explicit IEEE indeterminate case: std.math.exp does not preserve
        # NaN from zero/zero on this pinned toolchain.
        if word_count == 0 and bound == 0:
            return nan[DType.float64]()
        return exp(-bound / word_count)

    def score(self, X: Self.Input, y: Optional[Self.Target] = None) raises -> Float64:
        """Approximate log-likelihood (variational bound) of X."""
        var x = self._check_fitted_input(X, "LatentDirichletAllocation.score")
        return self._approx_bound(x, self._e_step_fixed(x), False)

    def perplexity(self, X: Self.Input, sub_sampling: Bool = False) raises -> Float64:
        """exp(-bound / word_count) of X."""
        var x = self._check_fitted_input(X, "LatentDirichletAllocation.perplexity")
        return self._perplexity(x, self._e_step_fixed(x), sub_sampling)

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.components_:
            raise Error("NotFittedError: This LatentDirichletAllocation instance is not fitted yet. Call 'fit' first.")
        return feature_names("latentdirichletallocation", self.n_components, input_features, self.n_features_in_)

    def set_output(mut self, *, transform: Optional[String] = None) raises:
        if transform and transform.value() != "default":
            raise Error("NotImplementedError: LatentDirichletAllocation supports native array output only")

    def get_metadata_routing(self) raises:
        raise Error("NotImplementedError: Python metadata routing is not available")


def _loglikelihood[dtype: DType](prior: Float64, distr: NDArray[dtype], dirichlet: NDArray[dtype], size: Int) raises -> Float64:
    """sum((prior - distr) E[log]) + sum(lgamma(distr) - lgamma(prior)) + sum(lgamma(prior*size) - lgamma(row sums))."""
    var score: Float64 = 0
    for i in range(distr.shape[0]):
        var row: Float64 = 0
        for j in range(distr.shape[1]):
            var value = get2(distr, i, j)
            score += (prior - value) * get2(dirichlet, i, j)
            score += Float64(Scalar[dtype](lgamma(value))) - lgamma(prior)
            row += value
        score += lgamma(prior * Float64(size)) - Float64(Scalar[dtype](lgamma(Float64(Scalar[dtype](row)))))
    return score


def _rows[dtype: DType](x: NDArray[dtype], start: Int, stop: Int) raises -> NDArray[dtype]:
    var result = zeros[dtype](Shape(stop - start, x.shape[1]))
    for i in range(start, stop):
        for j in range(x.shape[1]): set2(result, i - start, j, get2(x, i, j))
    return result^
