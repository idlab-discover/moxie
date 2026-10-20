# Consuming the linear engine

Import `LibLinear` from `moxie.svm._liblinear` and the existing `LinearOptions`
from `moxie.svm._contracts`. See [scope.md](scope.md) for the solver table and
[the native example](../examples/linear.mojo) for a complete call. The
[logistic example](../examples/logistic.mojo) uses the same engine with solver 0.

`solve(X, y, sample_weight, class_weight, options)` borrows its arguments and
returns an owned `LinearModel`. Input X is `(n,p)`, y and sample_weight are
`(n,)`. Class weights are `(k,)` for encoded classification or `(0,)` for
regression. Absent weights must be resolved to explicit ones by the wrapper.
`LinearOptions` fields, in order, are solver, c, epsilon, tol, fit_intercept,
intercept_scaling, max_iter, seed. There are no additional engine defaults.
Public penalty/loss/dual-auto dispatch and class-weight balancing belong to
estimator wrappers, which resolve a solver ID before calling this engine.

The seed is the native LIBLINEAR seed. A sklearn-compatible wrapper draws it
with `RandomState.randint(2147483647)` at the `_fit_liblinear` call site. The
engine creates an isolated shared MT19937 state and applies sklearn's bounded
integer mapping and forward coordinate shuffle. Integer wrapper configurations
reset per fit; mutable RNG ownership remains the wrapper's responsibility.

The result fields are:

| Field | Shape / meaning |
| --- | --- |
| coef | `(r,p)`, one binary/regression row; k OVR or joint rows |
| intercept | `(r,)`, zeros when bias is disabled |
| iterations | `(r,)` for independent solves; `(1,)` for joint Crammer–Singer |
| fit_status | 1 when any count reaches max_iter; otherwise 0 |

Nonpositive sample weights are removed before sorted class grouping, exactly
as in sklearn's native source. If this removes entire classes, the result has
rows for the retained classes (binary still has one row), ordered by their
original codes. Wrappers can derive that order from y and sample_weight;
this fixed family model has no label field. A single remaining class uses the
reference one-row positive-class optimization. Zero remaining samples raise.
The engine never retains X, y or caller-owned weight storage after return.

The synthetic bias feature has value intercept_scaling. Its learned coefficient
is penalized alongside the others; the returned intercept multiplies it by
intercept_scaling. `decision_values(X, model)` returns `(m,r)` using the approved
NuMojo matrix product and the intercept. Wrappers flatten single-row outputs,
perform class-label prediction and implement logistic probability presentation.
Crammer–Singer binary output retains both rows; the LinearSVC wrapper subtracts
them according to its reference presentation. There is no unfitted engine state:
a wrapper must enforce its own fitted-state check before passing a model.

## Numerical behavior and reproducibility

The pure Mojo translation follows the pinned C++ optimizer algorithms, including
shrinking, reactivation and stopping. It replaces feature-node storage with one
owned, class-grouped dense matrix and optional synthetic feature. Working vectors
use family-owned NuMojo storage. Memory is O(np+n+p) for independent solves and
O(np+nk+pk) for joint multiclass; the implementation never forms an n-by-n Gram
matrix or a p-by-p Hessian.

Train-time shared row norms and scalar math can change rounding and stopping
trajectories. Full solves compare coefficients/intercepts at rtol 2e-5,
atol 2e-6 (edge cases 5e-5 / 3e-6), and objectives at 1e-8. Smooth and L1 KKT
residuals must be below 2e-5 on the test grid. These tolerances exceed observed
errors to accommodate reduction and platform math changes; they do not promise
accuracy for arbitrary ill-conditioned inputs. Iterations report actual work;
they are not forced to match the reference near a stopping threshold. Capped
0/1/3-iteration comparisons additionally require equal counts.

The native boundary raises descriptive Mojo Errors for invalid arrays/options
and detected nonfinite learned state, including cases where the C++ reference
returns NaNs for zero-cost logistic dual fits. This is a deliberate safety
adaptation, tested against that reference behavior. A numerical error returns
no partially installed model. Positive finite effective costs may also overflow
on extreme data: rescale inputs or adjust parameters after an explicit error.

## Build, tests and performance

Use `pixi run build`, `pixi run test`, and `pixi run parity`. The differential
suite directly compiles the clean pinned sklearn native sources with scalar
BLAS callbacks. Python and C++ are validation-only dependencies. Native tests
cover shapes, strides, ownership, repeated calls, invalid options, limited
iterations and unavailable sparse APIs. Final checks and evidence are summarized below.

For reproducible local baselines:

```sh
pixi run python tools/benchmark.py --processes 2 --values 3 --warmups 1 \
  --min-time 0.02 --affinity 0 -o benchmarks/linear.json
pixi run python tools/operation_cost.py --processes 2 --values 3 --warmups 1 \
  --min-time 0.02 --affinity 0 -o benchmarks/operation.json
pixi run python tools/memory.py
pixi run python tools/memcheck.py
```

Timing uses native elapsed regions, includes per-solve allocations, and excludes
process startup and input creation. GNU time measures whole-process RSS.
Raw reports, compiler/source hashes and environment metadata stay in ignored
benchmarks/ and profiling/. These small baselines do not establish competitive
performance against optimized BLAS or a cross-platform speed guarantee.

Final validation on 2026-09-18: build, test and parity passed for all eleven
solver IDs. The differential suite passed 356 cases plus 15003 RNG outputs.
The full native contract workload passed Valgrind 3.25.1 with zero reported
errors and no definitely/possibly lost blocks, without suppressions. This is
bounded test evidence, not a universal memory-safety certification.

On an AMD Ryzen 9 5950X with CPU affinity 0, the final short pyperf baseline
measured 7.1–21.9 microseconds for 8×3 solves and
78.3–632.6 microseconds for 128×16 solves across solver IDs.
Peak process RSS was 21268 / 21080 / 23536 KiB for 8×3 / 128×16 / 2048×64
TRON workloads. Short-run timing variability is retained in the raw reports;
these numbers are a reproducible baseline, not a speed claim.
