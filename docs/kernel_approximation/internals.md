# Kernel feature-map internals

`src/moxie/kernel_approximation.mojo` explicitly exports the public estimators
and `KernelFeatureMap`. Implementations live in `_kernel_fourier.mojo`,
`_kernel_additive.mojo`, `_kernel_polynomial.mojo` and `_kernel_nystroem.mojo`.
`_kernel_approximation_contracts.mojo` defines the unchanged trait;
`_kernel_features.mojo` owns the helpers below. All are declared under Moxie's
owner in `DEPENDENCIES.json`. There are no per-algorithm projects or generated
public kernel-approximation packages.

The algorithms reuse `moxie.base`, `moxie.utils.validation`, shared `_dense`
and `_random`, and the integrated `_kernel_ridge` engine. Follow the
[shared numerical policy](../../shared/numerics/docs/numerics.md) when extending
these implementations. Compile with `-O3 --fp-mode=contract=off` to preserve
shared RNG arithmetic. Python dependencies remain validation tooling only.

## Dense helper contracts


All matrix arguments use `numojo.NDArray[dtype]`, dtype a compile-time Float32
or Float64. Use `numojo.Shape`, `Item` and upstream element-stride metadata;
no algorithm owns a competing dense buffer. Input shape is `(samples, features)`;
output shape is `(samples, mapped_features)`, preserving row order.
Borrow inputs. Outputs and retained fitted data own independent storage;
`owned_snapshot[dtype](x) raises -> NDArray[dtype]` validates and calls `.copy()`.
Copies of strided views preserve layout and copy the backing allocation, so
cost is O(backing storage), not necessarily O(logical view size). `.view()` is
explicit aliasing with retained lifetime. Positive/negative element strides and
nonzero offsets are accepted; arbitrary corrupt metadata and concurrent mutation
are outside the storage contract. New numerical outputs should be contiguous
row-major allocations. Integer sampling indices use NuMojo Int64 arrays.

`moxie._kernel_features` exports these private helpers:

- `validate_matrix[dtype](x, nonnegative=False) raises`: rank two, nonempty,
  finite; optionally nonnegative. Borrows, visits logical strides, O(rows*cols)
  time and O(1) workspace. It does not validate skewedness or kernel parameters.
- `component_width(n_components: Int) raises -> Int`: positive width.
- `additive_width(n_features: Int, sample_steps: Int) raises -> Int`:
  `n_features*(2*sample_steps-1)` with pre-arithmetic overflow validation.
- `basis_width(n_samples: Int, n_components: Int) raises -> Int`: positive
  `min(n_samples,n_components)`; the estimator owns the reference warning.
- `FeatureMapState`: contains shared `input_state: FittedState` and
  `n_features_out_: Int`; `record_fit(x,width)` validates before changing state,
  `check_transform(x)` rejects unfitted/dimension mismatch, `reset()` clears
  both fields. Copy is an independent metadata snapshot, **not estimator clone**.
  Estimators expose their own `n_features_in_` and learned arrays. Commit this
  state only after successful numerical fitting. Stateless additive maps must
  not require this helper's fitted check before transform.

Constructors retain typed Params separately from state. `get_params` returns a
configuration copy; `set_params` changes configuration; a subsequent ordinary
fit replaces learned arrays. `clone` creates an unfitted estimator with copied
configuration, never copied learned arrays. Mutable RNG configuration requires
an explicit state snapshot for clone, while fitting borrows the configured
stream; integer configuration recreates the stream per fit. None maps to one
application-owned entropy context passed by mutable reference, as specified by
shared random. Fit mutates in place, returns a mutable borrow of self, and accepts an ignored
optional target. Fit-transform is fit followed by transform. These native
adaptations are inherited, and exercised by trait/clone contract tests.
No probability, score, sample weight, incremental fit or inverse capability is
imposed. Python metadata routing, pandas output, dynamic parameter dictionaries
and device/Array API dispatch remain binding work.


See the [API pages](README.md) for the implemented RNG borrowing methods and
other native adaptations. Tests under `tests/kernel_approximation/` cover the
helper contracts and each estimator's lifecycle in both input dtypes.
