# BernoulliRBM

Import `BernoulliRBM` from `moxie.neural_network`.

The default storage type is Float64; select `BernoulliRBM[DType.float32]` for
Float32 fit/transform. Inputs are finite, nonempty rank-2 NuMojo NDArrays.
Arbitrary strides are accepted, and fit materializes each minibatch.
Weights currently use C layout, with normal draws generated in logical C order.
The reference materializes F layout; this physical layout difference does not
change matrix shapes or the initialization sequence.

Constructor parameters retain the reference names and defaults: n_components=256,
learning_rate=0.1, batch_size=10, n_iter=10, verbose=0, random_state=None.
Fit and typed set_params return mutable self; partial_fit mutates in place and returns nothing. `get_params()` returns a typed
RBMParams record; modify that record and pass it to set_params. Named partial
set_params overloads remain unfinished. `clone(model)` from moxie.base returns
unfitted configuration. Learned attributes are components_, intercept_hidden_,
intercept_visible_, h_samples_ and n_features_in_. Empty arrays before fitting
are a native representation, not fitted results.

Implemented methods: fit, transform, fit_transform, gibbs, score_samples,
get_feature_names_out, get_n_features_in, get_params, set_params and partial_fit.
Transform returns `(samples, components)`, Gibbs returns a bool array
`(samples, features)`, and score_samples returns a vector `(samples,)`.
Feature names are a native List[String] beginning with `bernoullirbm0`.
Optional input feature names are validated for length; named-data containers
and Python's conditional feature_names_in_ are binding work.

To start with `partial_fit`, use `BernoulliRBM[DType.float64]` (the default).
It accepts Float32 and Float64 batches and initializes Float64 learned arrays,
as scikit-learn does. Float32 batches are converted to Float64 by value, in logical
row order. After a Float32 `fit`, call `partial_fit` on the same
`BernoulliRBM[DType.float32]` model with Float32 batches. The positive phase and
parameter updates use Float64 arithmetic. Stored parameters and fantasy particles
retain Float32. The model dtype is fixed at compile time, so a Float32 model
requires an initial `fit` before `partial_fit`. Other methods take inputs of the
model dtype.

Integer random_state resets at ordinary fit and score calls; Gibbs and
partial_fit retain an owned advancing stream. For random_state=None, retain
one application-owned RandomState (or entropy_state()) and pass it as the
second argument to fit, fit_transform, partial_fit, gibbs and score_samples.
The context is borrowed mutably and advances in place. Calls without that
required context raise an error instead of silently inventing a new stream.
Use the same context consistently across methods for mutable-stream semantics.

There is no supervised score, predict, predict_proba or sample-weight fit.
The typed input interface excludes sparse input without densification.
Errors use Mojo Error strings with ValueError/NotFittedError prefixes.
Verbose output matches the reference line
`[BernoulliRBM] Iteration k, pseudo-likelihood = %.2f, time = %.2fs`; the mean
is accumulated in Float64 and the time is wall-clock.

BernoulliRBM declares the shared `FeatureNamesOut` and `InputFeatureCount`
traits: `get_n_features_in()` mirrors sklearn's `n_features_in_` (raises before
fit), so `Pipeline.n_features_in_()` and `make_union(BernoulliRBM(), ...)`
feature names (`bernoullirbm__bernoullirbm0`) work.

Run `pixi run start` for a Float64 example. See [scope](scope.md) for validation
status and deferred sparse branches. Private initialize/update/mean_hiddens,
free_energy and underscored helpers are implementation details, not additional
public estimators.

From the Moxie root, run `python families/neural_network/tools/integration.py rbm`
with the locked toolchain. The command compiles the RBM contracts, examples and
Pipeline and FeatureUnion tests, then compares RBM kernels, fit and incremental
lifecycle sequences against pinned scikit-learn.
