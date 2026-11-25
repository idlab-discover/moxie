# Dense MLP engine scope

The [private engine](api.md) implements Float32 and Float64 parameter
initialization and packing, forward propagation, and batch loss and gradients
for the four dense output/loss pairs. It also provides full-batch L-BFGS and
minibatch SGD and Adam. It has no package-level exports. The public estimator
wrappers and validation-set early stopping remain open. Sparse inputs and
targets follow the family's separate sparsity scope.

From the Moxie root, run the focused check in the locked root environment
with NumPy and SciPy installed.

```sh
pixi run python families/neural_network/shared/mlp_engine/tools/check.py parity
```

The runner assembles live sources, precompiles the child's package, builds
the test binaries and runs native contracts. The parity scripts execute
the initialization, forward, loss, backpropagation and training code from local
scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

Parameter parity covers 32 fixtures and subsequent RNG draws with relative
tolerance 1e-14 and absolute tolerance 1e-15. Forward and gradient parity
covers 40 fixtures in both dtypes, each evaluated with and without sample
weights. It uses absolute and relative tolerances of 3e-6 in Float32 and
2e-12 in Float64 to account for reduction order and rounding. Native
central differences cover the four loss pairs in both dtypes on weighted
batches, using steps of 2e-3 or 1e-5 and absolute gradient tolerances of
2e-4 or 1e-8 respectively to allow rounding and truncation error. The
multioutput squared-error fixture compares directly with the reference's
loss and gradient conventions.

L-BFGS parity compares 76 fixtures using SciPy 1.18.1 from the root lock.
Absolute and relative tolerances for loss and predictions are 3e-6 in
Float32 and 2e-12 in Float64. The accepted parameters use 1e-5 and 5e-12,
respectively, because rounding in the batch objective and gradient
accumulates across optimizer steps. The largest parameter differences were
7.3e-6 and 2.3e-12, both in the nonlinear binary classification fixture.
Iteration counts, evaluation counts and warning text agree exactly. Cases
exercise strided inputs and weights, gradient convergence, immediate
convergence, and both solver budgets. Native contracts also exercise invalid
inputs and verify that fitting leaves the supplied parameters intact.

Stochastic parity covers 200 snapshots for SGD and Adam in both dtypes,
including weighted and unweighted losses, shuffled and original row order,
a short final batch, automatic and clipped batch sizes, patience and
adaptive-rate reduction, an optimizer restart, and repeated one-epoch updates.
It compares the epoch and update counts, samples processed, warning conditions,
and next RNG draw exactly. Learning-rate tolerances are relative 1e-13 and
absolute 1e-16. Loss histories, fitted parameters and raw predictions use
absolute and relative tolerances of 3e-6 for Float32 and 2e-12 for Float64
to allow rounding from batch loss and gradient reductions. Native contracts
also exercise invalid options, zero-weight batches and independent ownership
of strided starting parameters and trainer copies.
