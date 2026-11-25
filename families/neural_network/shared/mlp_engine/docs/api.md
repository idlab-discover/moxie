# Private dense MLP engine

Import helpers from `moxie.neural_network._mlp_engine`. They use the family's
`LayerLayout` and `LayerTensors` with NuMojo Float32 or Float64 arrays.

- `allocate[dtype](layout)` creates independent zeroed layer tensors.
- `initialize(params, rng, activation)` applies scikit-learn's Glorot
  initialization and advances the borrowed RNG. It visits logical C order
  even with negative strides.
- `parameter_count(params)` returns the total coefficient and bias count.
- `pack(params)` returns an owned Float64 vector with coefficients in layer
  and C order followed by intercepts in layer order.
- `unpack(vector, params)` validates size, finiteness and dtype overflow before
  copying into existing tensors. The tensors retain their layout and own
  their storage independently of the solver vector.
- `forward(X, params, hidden_activation, output_activation)` takes a nonempty
  finite samples-by-features matrix and returns an owned samples-by-outputs
  matrix in its dtype. Hidden activations are `identity`, `logistic`, `tanh`
  and `relu`. Output activations are `identity`, `exp`, `logistic` and
  `softmax`.
- `loss_gradient(X, y, params, hidden_activation, output_activation, loss, alpha=0)`
  returns the batch loss as Float64 and owned gradients in the same layer
  order and dtype as the parameters. The weighted overload takes a borrowed,
  rank-one `sample_weight` after `alpha`. Weights are finite with a nonzero
  finite sum and can have arbitrary valid strides. L2 applies to coefficients
  only and is divided by the sample count or weight sum.

Targets are finite rank-two arrays matching the output shape. The accepted
output/loss pairs are identity with `squared_error`, exp with `poisson`,
logistic with `binary_log_loss`, and softmax with `log_loss`. Use nonnegative
Poisson targets, encoded binary targets, or categorical indicators as
appropriate. Wrappers are responsible for encoding targets. For multioutput
squared error the pinned implementation averages the reported loss across
outputs while using the unscaled residual in backpropagation. The engine
preserves both behaviors. Cross-entropy clips probabilities in the loss and
uses the original activations for the residual.

Forward and backward matrix products use NuMojo rank-two `@`, including its
vectorized CPU path. Numax's runtime-shape `numax.linalg.matmul` consumes
`TensorLike` and returns `Dynamic` storage. The engine uses NuMojo directly
to avoid converting these borrowed arrays to a second storage type at each layer.
NuMojo can materialize transposes and strided operands. Bias reduction uses
`moxie._dense.sum_axis2d`. Scalar loss and regularization sums accumulate in
Float64, while activations, residuals and gradient tensors retain the input
dtype. The CPU exponential and logarithm use LLVM intrinsics for Float64
accuracy on the locked toolchain.

Helpers raise Mojo `Error` with `ValueError` prefixes for invalid batch
shapes and settings. Concurrent mutation and manually invalid array metadata
are outside the shared storage contract.

## Full-batch L-BFGS

`moxie.neural_network._mlp_lbfgs.fit_lbfgs(X, y, initial, hidden_activation,
output_activation, loss, alpha=0.0001, options=LBFGSOptions())` takes the same
encoded targets and loss pairs as the objective. The weighted overload takes
`sample_weight` after `alpha`. `initial` supplies initialized coefficients
and biases in the input dtype. The solver reads the inputs and works on an
independent parameter copy.

`LBFGSOptions(max_iter=200, max_fun=15000, tol=1e-4)` configures the
iteration count, soft function-evaluation budget, and projected-gradient
tolerance. The adapter uses the current `mojo_optimize.LBFGSB` dependency
with infinite bounds and SciPy defaults for history, function tolerance and
line search. It evaluates the objective in the input dtype and returns
independent Float64 parameters from the accepted iterate, matching the
pinned scikit-learn and SciPy dtype behavior.

The result includes the loss, iteration and evaluation counts, and an empty
warning string on convergence. On solver termination without convergence, the
result carries the scikit-learn convergence warning text for the caller.
The loss follows SciPy's last evaluated objective, including after an abnormal
line search that restores an earlier parameter vector. Invalid settings,
callback errors and nonfinite output parameters raise Mojo errors.
