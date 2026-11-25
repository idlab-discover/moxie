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
