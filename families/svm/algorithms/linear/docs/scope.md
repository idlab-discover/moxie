# Dense linear SVM scope

`LinearSVC`, `LinearSVR`, and `l1_min_c` are implemented in pure Mojo and exported
through `moxie.svm`. They reuse the family LIBLINEAR engine, shared RNG, NuMojo
storage and the verified foundation assembler. See [api.md](api.md) for native
interface adaptations and runnable examples.

LinearSVC supports L2 squared-hinge primal/dual, L2 hinge dual, L1 squared-hinge
primal, automatic dual selection, binary/OVR classification, and joint
Crammer–Singer. Crammer–Singer ignores the valid loss/penalty/dual selections,
including combinations unavailable to OVR. Binary joint coefficients are reduced
to class-1 minus class-0. LinearSVR supports epsilon-insensitive dual and squared
epsilon-insensitive primal/dual, including automatic selection. Invalid loss,
penalty and dual combinations raise descriptive errors.

Both support intercept regularization/scaling, finite sample weights, repeated
fit, fresh clones, typed and keyword parameter updates, and seeded or mutable
RNG fitting. Float32 matrices convert to Float64; Float64 positive/negative
strides and offsets are supported. Classification labels are sorted, numeric or
native strings. Class weights support explicit label/value mappings or balanced
weights using weighted label frequencies. Accuracy and R2 accept score weights.
`n_iter_` reports actual maximum iterations; `fit_status_` supplements it with
0/1 convergence status. A limited fit emits a ConvergenceWarning message.

`l1_min_c` supports squared-hinge and log losses, binary/multiclass/single-class
labels, optional scaled intercepts, and ill-posed zero-denominator errors.
Its matrix product and axis reduction reuse existing numerical owners.

## Explicit limitations

- Sparse inputs/outputs, sparse coefficients and sparsify/densify are deferred
  by the workspace sparse policy. No implicit densification occurs.
- All training classes must retain a positive-weight sample. Removing an entire
  class raises rather than exposing the reference engine's compressed class
  layout through the original label list. Zero or negative balanced class totals
  are rejected. Signed sample weights otherwise follow the engine's removal of
  nonpositive-weight rows; they still enter balanced frequency construction.
- The engine requires finite positive C, tolerance and intercept scaling, finite
  nonnegative epsilon, and finite nonnegative effective class costs. Negative
  class costs and numerically undefined zero-cost solutions raise.
- Numeric NDArray classification targets adapt to Float64 lists. Use native
  `List[Int]` with `LinearSVC[Int]` to retain wide integer labels. Numeric inputs
  that cannot be represented exactly as Float64 should not use that adapter.
- None RNG is an estimator-owned native entropy context, not Python's global
  singleton. Integer seeds reset on every fit. Explicit mutable streams use
  `fit_with_random_state`. Python warnings machinery, reflection, metadata
  routing, serialization and bindings remain outside the native interface.
- Kernel SVMs and their probability fitting are not implemented by this child.

## Validation

From this directory run `pixi run build`, `pixi run test`, `pixi run parity`.
The native tests cover lifecycle, invalid modes, labels, clones, typed/keyword
parameters, Float32 conversion, negative strides, weights, seed draw consumption,
convergence and l1 thresholds. Public-wrapper parity executes the fitting and
solver-dispatch bodies from local sklearn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, with its compiled LIBLINEAR source.
The installed sklearn package supplies supporting validation and inheritance,
not the fitting or optimization oracle. The reference never runs in library code.

The suite includes 174 mode/weight/dtype comparisons, 32 numerical/iteration
edge cases, and 36 l1_min_c comparisons. Coefficients use rtol 5e-5/atol 3e-6;
decisions use rtol 7e-5/atol 2e-5; objectives use rtol 1e-7/atol 1e-8.
Reduction order and finite solver termination can move predictions at a tied
boundary: prediction equality is mandatory outside a reference score gap of
2e-5, and every ambiguous disagreement is recorded. Native exact ties choose
the first sorted class. Bound comparisons use rtol 1e-12/atol 1e-14.
These are bounded tested tolerances, not universal accuracy guarantees.
