# Private MLP parameter helpers

Import helpers directly from `moxie.neural_network._mlp_engine`. This module
currently supplies parameter preparation only; it does not implement MLPKernel
or a trainable estimator. Package-level exports remain empty.

- `allocate[dtype](layout)` creates independent zeroed LayerTensors.
- `initialize(params, rng, activation)` mutates tensors with scikit-learn's
  Glorot initialization, advancing the borrowed RandomState. Hidden activation
  accepts identity, logistic, tanh or relu; logistic uses the narrower bound.
- `parameter_count(params)` returns the total coefficient/bias element count.
- `pack(params)` returns an independent Float64 rank-one vector, coefficients
  in layer/C order followed by intercepts in layer order.
- `unpack(vector, params)` copies into existing tensors, preserving their layout.
  The whole input is checked for size, finiteness and dtype overflow before writes.

These functions use the family's LayerLayout/LayerTensors and NuMojo Float32 or
Float64 arrays. Initialization visits logical C order even with negative strides.
Unpack copies rather than adopting sklearn's vector views; solver vectors and
model arrays therefore have independent ownership. Float32 casts are explicit.
Overlapping concurrent mutation and manually invalid array metadata are excluded
by the shared storage contract. Helpers raise Mojo Error with ValueError prefixes.

`pixi run build` compiles the supplied module and dependencies; `pixi run test`
runs parameter contracts. `pixi run parity` compares initialization and packing
with extracted function bodies from the pinned local sklearn checkout, then
fails explicitly because the remaining engine acceptance tests are unfinished.
The comparison includes subsequent RNG draws; tolerances of 1e-14 relative and
1e-15 absolute account for Float64 text output and scalar square-root rounding.
It does not validate learning, gradients, loss kernels or optimizers.
