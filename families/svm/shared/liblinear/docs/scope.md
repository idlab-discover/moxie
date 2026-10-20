# Dense linear engine scope

The private `moxie.svm._liblinear.LibLinear` implementation consumes the shared
family `LinearOptions`, `LinearModel` and `LinearEngine` contracts. It implements
dense SVC/SVR and logistic optimization; estimator wrappers belong to their own
projects. No public estimator is exported here.

| Solver | Objective / method |
| --- | --- |
| 0 | L2 logistic, trust-region Newton |
| 1 | L2 squared hinge, dual coordinate descent |
| 2 | L2 squared hinge, trust-region Newton |
| 3 | L2 hinge, dual coordinate descent |
| 4 | Joint Crammer–Singer, dual coordinate descent |
| 5 | L1 squared hinge, coordinate descent and line search |
| 6 | L1 logistic, Newton and coordinate-descent inner QP |
| 7 | L2 logistic, dual coordinate descent |
| 11 | L2 squared epsilon-insensitive regression, trust-region Newton |
| 12 | L2 squared epsilon-insensitive regression, dual coordinate descent |
| 13 | L2 epsilon-insensitive regression, dual coordinate descent |

Supported engine inputs are finite Float64 dense arrays, including logical
strides and offsets. The family owns the input conversion contract: Float32
conversion and numeric/string label encoding belong to wrappers. Classification
uses explicit contiguous numeric class codes and a class-weight vector;
regression uses an empty class-weight vector. Sample weights are explicit.
Nonpositive sample weights are removed. Class costs are nonnegative; a zero
class cost may make a reference solver numerically undefined, in which case
this engine raises rather than returning nonfinite coefficients. Negative class
costs are rejected as invalid convex optimization problems. Finite positive C,
tolerance and intercept scaling, finite nonnegative epsilon and uint32 seeds
are required. `max_iter` is a nonnegative integer.

Every feature, including the synthetic bias feature, is regularized. Bias
scaling and no-intercept modes are supported. Sorted class grouping, binary
signs, OVR positive-vs-negative cost rules, per-row iterations, and the joint
Crammer–Singer model layout follow the pinned source. The Crammer–Singer
reference ignores the supplied max_iter during optimization (100000 internally);
fit_status still reports whether the returned count reaches the supplied limit.
A normal solve starts fresh and returns owned arrays; this engine has no fitted
estimator lifecycle, warm-start or partial-fit API.

## Deferred capabilities

CSR/CSC/COO inputs, sparse conversion, sparse outputs, and sklearn's sparse
`_liblinear.train_wrap` path are deferred under the [family dense scope](../../../docs/scope.md).
There is no sparse typed entry point and no implicit densification. The dense
translation directly traverses dense storage instead of recreating sklearn's
internal feature-node representation. Dense L1 coefficient sparsity is fully
in scope. No sparse-required default exists at this private dense boundary.

Kernel solvers, support-vector storage/cache, probability calibration and SVM
wrapper presentation belong to the other family owners. Logistic wrappers use
this same engine, then implement their own probability presentation. Python
bindings, reflection and metadata routing are separate work.

## Validation

Run `pixi run build`, `pixi run test`, `pixi run parity`. Build compiles every
owned Mojo module, the examples and test drivers using the verified assembled
namespace. Native tests exercise all solver IDs, invalid parameters/shapes,
strided matrices and vectors, repeat calls, independent result ownership,
decision shapes and iteration limits. A failed compile confirms the absence
of a sparse entry point.

Parity compiles unmodified `linear.cpp`, `tron.cpp` and `newrand.h` from the
clean local sklearn revision 4358542a1bd39af2e7e1cbace8b9fed067473c5b. The C++
adapter is validation only and uses scalar BLAS callbacks; it does not substitute
a wheel. Comparisons cover coefficients, bias, objective values, RNG words,
weighted binary/OVR/joint models, regression, capped iterations, zero designs,
collinearity and small C. Results and commands remain in ignored
builds/. Final execution results are recorded in docs/api.md.

Numerical comparison tolerances account for shared row-norm reduction order
and native scalar math. Counts near convergence can differ by a few iterations;
they report actual work, not copied oracle counts. Prediction uses NuMojo's
approved rank-2 product. These tests establish the listed dense coverage, not
blanket scikit-learn API parity or a performance guarantee.
