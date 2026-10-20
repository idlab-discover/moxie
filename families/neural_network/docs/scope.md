# Supported neural-network scope

The family is owned by Moxie. Dense Float32/Float64 BernoulliRBM supports fit,
transform, fit_transform, Gibbs sampling, pseudolikelihood, typed parameters,
cloning and feature names. Float64 partial_fit is implemented. Float32 partial_fit
explicitly raises because sklearn promotes its input to Float64 and the native
learned-state transition remains unfinished. See the [RBM scope](../algorithms/rbm/docs/scope.md).

Private MLP helpers allocate, initialize, pack and unpack parameters. Both public
MLP estimators are raising declarations. Forward/backpropagation, losses,
SGD/Adam/L-BFGS training, weighted objectives, early stopping, warm starts and
trainable MLP wrappers remain unfinished. See the [engine scope](../shared/mlp_engine/docs/scope.md).

Dense finite rank-2 inputs, both dtypes and arbitrary valid strides use NuMojo
storage and Moxie RNG. Sparse inputs, implicit densification, Python reflection,
pandas, metadata routing and serialization are unsupported. RBM named partial
set_params and reference verbose timing/formatting remain unfinished. Its typed
set_params and fit return mutable self; the family-specific RBMSampling
partial_fit contract retains its existing None return.

`pixi run neural-network-check` validates native lifecycle/error/ownership
contracts, six private RBM kernel cases with full RNG state, six public RBM
fit/refit/transform/score/Gibbs cases, three Float64 incremental cases, and 32
MLP initialization/packing cases. It also checks core ownership, licenses and an
isolated public consumer. Tolerances are Float32 3e-5 and Float64 2e-12 for RBM;
MLP parameters use relative 1e-14 and absolute 1e-15. These account for scalar
arithmetic and text output rounding on the bounded fixtures. They do not prove
complete estimator parity or performance. Historical child full-parity tasks
continue to report their unfinished acceptance gates.

The oracle remains local scikit-learn at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Validation executes its original method
bodies; learning and prediction remain pure Mojo.
