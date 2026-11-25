# BernoulliRBM dense scope and implementation status

Implemented: dense Float32/Float64 fit, transform, fit_transform, Gibbs sampling,
pseudolikelihood, typed parameters, cloning and generated feature names;
incremental fitting from Float32 or Float64 batches. Inputs use NuMojo arrays
and shared native RandomState. See the [API](api.md#bernoullirbm) for choosing
the model dtype for incremental fitting.

Tests cover singleton and nonmatching positive/fantasy batch sizes, seed 42,
Float32/64, stable activations at +/-1000, repeat fit, clone independence,
mutable contexts, negative feature stride, invalid feature counts, and unfitted
transform. Kernel differential comparisons cover six workloads and full RNG
state against pinned scikit-learn source. Lifecycle comparisons cover incremental
updates after fit in both dtypes, fresh Float32 batches on a Float64 model,
strided and offset views, exact particles and Gibbs samples, and explicit RNG
continuation.

The sigmoid and free-energy softplus use Numax's CPU Float64 exponential.
The kernel comparison checks both functions on logits from −500 to 500 with
relative tolerance `2e-14` and zero absolute tolerance to catch loss of
precision in small positive probabilities and small-magnitude free energies.

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

Run `python families/neural_network/tools/integration.py rbm` from the Moxie
root with the locked toolchain for the bounded native and oracle comparisons.
The standalone build offers a negative compilation probe for sparse imports.
