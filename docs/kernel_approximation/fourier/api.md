# Dense Fourier feature maps

Import `RBFSampler` and `SkewedChi2Sampler` from `moxie.kernel_approximation`.
Both implement the family's `KernelFeatureMap` contract on NuMojo matrices.
See [the executable example](../../../tests/kernel_approximation/fourier/example.mojo).

| Estimator | Constructor keywords and defaults |
| --- | --- |
| `RBFSampler[dtype=DType.float64]` | `gamma: Float64=1.0`, `n_components: Int=100`, `random_state: Optional[Int]=None` |
| `SkewedChi2Sampler[dtype=DType.float64]` | `skewedness: Float64=1.0`, `n_components: Int=100`, `random_state: Optional[Int]=None` |

RBF also accepts `gamma=String("scale")`, computing inverse feature count times
population variance, with gamma 1 for constant inputs. Skewedness can be any
finite real; transform requires every value to exceed `-skewedness`. Fit only
checks the input's shape and finiteness. Inputs and outputs have the same
compile-time float32 or float64 dtype. Both accept valid strided views.

`fit(X, y=None)` mutates in place. `transform(X)` returns a new `(n,m)` matrix;
`fit_transform` combines them. Targets are ignored; wrap a target copy or moved
value in the optional argument, as required by Mojo ownership. Learned arrays
are `random_weights_` `(p,m)` and `random_offset_` `(m,)`; `n_features_in_`
records p and RBF also exposes `_gamma` as Float64 (retaining float32
variance/scaling rounding when fitted with float32). Before fitting, these arrays are empty
and the count is zero; transform and names raise `NotFittedError`-prefixed errors.
No prediction, scoring, inverse, partial-fit or sample-weight interface is exposed.

Integer seeds rebuild the shared legacy `RandomState` on each fit. The native
adaptation for mutable RandomState and default None is explicit borrowing:
`fit_with_rng(X, context)` / `fit_transform_with_rng(X, context)`, where context
is a `mut RandomState`. Initialize a single application context with
`moxie._random.entropy_state()` for None semantics and share it across consumers.
Calling ordinary fit with None raises a descriptive error. The explicit-context
methods use the supplied stream instead of the integer configuration. They
consume the same sequence as sklearn; no context is retained or copied by the
estimator. For an independent mutable-stream clone, clone the estimator and
explicitly call `context.copy()`; neither copied object advances the original.

`moxie.base.clone(model)` copies constructor configuration into an unfitted
estimator. `get_params()` returns an independent typed record (`Model.Params`).
RBF represents its numeric gamma and string choice as `gamma` and `gamma_mode`
(`numeric` or `scale`) in this record. `set_params(record)` copies it; named
partial updates accept numeric gamma/skewedness, width and integer seed. RBF also
accepts `set_params(gamma=String("scale"))`, optionally with width and seed updates;
a later numeric gamma update switches back to numeric mode. Invalid gamma strings
are rejected on fit, just as for the constructor. To reset the seed to None,
edit a copied record and pass it back.
Parameter changes leave the learned basis until refit, matching sklearn;
transform uses current width for normalization and current skewedness for log.
Nonpositive widths are rejected before transform arithmetic, including invalid
post-fit updates; skewed sklearn may instead emit nonfinite results in this
invalid-configuration case. Sizes exceeding native byte-count limits raise a
descriptive error before allocation.

`get_feature_names_out()` returns a native `List[String]`, using `rbfsampler0`
or `skewedchi2sampler0` prefixes and learned output width. Optional input names
are checked for input-feature count; unnamed numeric matrices do not acquire
`feature_names_in_`. Passing a native list into Optional requires `.copy()` or
ownership transfer. Dynamic dictionaries, pandas, Python tags/metadata routing,
serialization, and Python exception classes remain binding work. Errors use
Mojo `Error` with descriptive sklearn-style prefixes.

The implementation uses shared storage and RNG, NuMojo's accepted rank-2 matrix
multiplication for the projection, one fused stdlib SIMD pass for offset,
cosine and scaling, and native scalar math for the spectral inverse CDF and log.
Fit stores O(pm+m) values; transform allocates O(nm), plus O(np) log scratch for
skewed input and NuMojo's matmul workspace. It does not retain the training data.
Scale gamma composes mean, centered residual, square and mean, using O(np)
temporaries. Sparse support is deferred; see [scope](scope.md).

## Validation

Run `pixi run build`, `pixi run test`, `pixi run parity`.
Parity executes unedited fit/transform method bodies extracted from the pinned
local sklearn source, with NumPy dense validation/matrix plumbing. It verifies
source revision, cleanliness and code filename. This is differential numerical
coverage of those bodies; it is not a full Python estimator-check suite.
Native tests separately cover validation, lifecycle, cloning, names, parameters,
contexts, aliasing and strides.

Fitted draws use rtol/atol 4e-5 (float32) and 3e-8 (float64), accounting for
native elementary functions (observed Float64 maximum 2.3e-10). Logarithms use
the LLVM intrinsic (platform libm) because Mojo 1.0's `std.math.log` for
Float64 measured up to 1.7e7 ulp off; each sampler's output scale is written
exactly as its sklearn source (`(2/m)**0.5` for RBF, `sqrt(2)/sqrt(m)` for skewed). Output adds a dot-product rounding bound:
`4*p*eps*(abs(X)@abs(W)+abs(b))*sqrt(2/m)`, using logged input for skewed.
Cosine is 1-Lipschitz, so this propagates phase error without assuming small
input values. A float32 stress case near coordinates 1e8 differs by about .396
in individual features, within this phase bound; float64 should be used when
small coordinate differences sit on such large offsets. These tolerances cover the tested finite inputs, not arbitrary
ill-conditioned phase calculations. RNG next-draw values must match exactly.
Kernel Gram tests use 3,000 components and an absolute Monte Carlo threshold .08.
