# BernoulliRBM dense scope and implementation status

This project exports BernoulliRBM. It is **incomplete within dense-only scope**.
Implemented: dense Float32/Float64 fit, transform, fit_transform, Gibbs sampling,
pseudolikelihood, typed parameters, cloning and generated feature names;
Float64 partial_fit. Inputs use NuMojo arrays and shared native RandomState.

Tests cover singleton and nonmatching positive/fantasy batch sizes, seed 42,
Float32/64, stable activations at +/-1000, repeat fit, clone independence,
mutable contexts, negative feature stride, invalid feature counts, and unfitted
transform. Kernel differential comparisons cover six workloads and full RNG
state against pinned scikit-learn source. These are bounded checks, not complete
estimator parity. Float32 partial_fit raises NotImplementedError: its required
Float64 input promotion and typed learned-state transition remain unfinished.

CSR/CSC/LIL and all sparse branches are deferred by the session dense-only
policy. Reference sklearn/neural_network/_rbm.py accepts CSR in fit, partial_fit,
transform and score_samples. Native methods accept dense NDArray only; no sparse
conversion or implicit densification is supplied. Shared storage and family
plumbing likewise provide dense operations only. No constructor default requires
sparse input. Python reflection, pandas data, metadata routing and serialization
are separate binding work.

Native random_state=None requires an application-owned context passed explicitly;
use an integer random_state for the context-free methods. Dtype is selected at
compile time; integer input conversion is caller-owned. See [API](api.md).

Executed checks: `PIXI_CACHE_DIR="$PWD/.pixi/cache" pixi run build`, `pixi run test`
and `pixi run parity`. Build and scoped native tests pass. Parity's private
kernel comparison passes, then the task deliberately fails because full public
estimator parity is unfinished. The task must remain failing until the dense
implementation and acceptance coverage are complete. Sparse rejection is by
unavailable typed API; a negative compilation probe checks the missing sparse import.

Core rollup checks add bounded public fit/refit, transform, score_samples, Gibbs
and Float64 partial_fit comparisons against the pinned local reference. Run
`pixi run neural-network-check` from the Moxie root. This preserves the explicit
Float32 incremental exclusion and does not complete the historical full-parity gate.
