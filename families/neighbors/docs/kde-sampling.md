# KernelDensity sampling

`KernelDensity(kernel="tophat", bandwidth=0.75)` now supports
`sample(n_samples=1, random_state=42)` and `sample_with_rng(n_samples, rng)`,
as does the Gaussian kernel. Fit first; both methods return an independent
Float64 array of shape `(n_samples, n_features)`, including one-dimensional
data and one sample. Float32 and Float64 training data are supported.

An integer seed starts the same sequence on each call. Passing a mutable shared
`moxie._random.RandomState` to `sample_with_rng` advances that stream, including
its cached Gaussian value. Omitting random_state uses the process-default stream
through shared check_random_state; repeated calls advance it.
seed_global_random_state(seed) resets that stream for reproducible tests. Weighted fits select centers according to cumulative
sample weights; zero-weight centers are skipped except at an exactly zero draw,
matching the reference's left-sided search. Fit snapshots strided input arrays.

Tophat sampling follows scikit-learn's Gaussian-radius transformation through
the regularized incomplete gamma function. It reuses `moxie._special.gammainc`
and `moxie._dense.row_squared_norms`; it does not replace shared numerical code.
The family manifests declare the live special-function owner
(`../../shared/special`) and Numax (`../../../numax`) sources and licenses with the
compatible Mojo 1.0.0 / MAX 26.5.0 toolchain, and consume the shared RNG
normalization interface from current Moxie HEAD.

The native contracts and `tests/parity_kde_sampling.py` compare 216 configurations
against the pinned local sklearn `KernelDensity.sample` method using NumPy and
SciPy as validation tools. Coverage includes dimensions 1–1000, both input
dtypes, three seeds, three bandwidths, weighted/unweighted fits, repeated seeds,
continued streams, strided snapshots, refits, parameter changes, cloning and
invalid calls. Absolute tolerance is `2e-9 * bandwidth + 2e-14`, accounting for
native normal/power rounding and translated-center addition. RNG continuation
is checked exactly. The two downstream consumers also execute sampling contracts.

Limits: `n_samples` must be positive. Epanechnikov, exponential,
linear and cosine sampling raise, as in sklearn. The gamma owner's bounded
iteration/domain limitations still apply. This change adds no tree backend or
non-Euclidean KDE support.
