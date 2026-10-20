# Dense LIBSVM engine

`LibSVM` implements the family `KernelEngine` for `c_svc`, `nu_svc`, `one_class`,
`epsilon_svr`, and `nu_svr`. It uses Float64 optimizer state, Float32 cached kernel
columns, standard and nu working-set selection, shrinking, gradient
reconstruction, formulation-specific bounds and initialization, and convergence
or iteration-limit reporting. Classification uses sorted encoded labels, OVO
support packing, seeded five-fold sigmoid calibration, and probability coupling.
Library execution is pure Mojo; C++ and Python are validation-only.

## Kernel and storage behavior

`kernel_matrix` provides training arithmetic for linear, polynomial, RBF,
sigmoid, and precomputed matrices. `resolve_gamma` computes auto and scale
through the accepted dense operations. Prediction RBF uses direct differences,
matching LIBSVM's prediction arithmetic and avoiding norm-subtraction
cancellation at large coordinate offsets. RBF uses the LLVM Float64 exponential,
following live Moxie random's CPU math convention; the standard-library
approximation can flip near-zero classification decisions. Native callable evaluation is the
wrapper's responsibility; unresolved callable configurations raise.

The cache stores full Float32 columns under original row indices. A separate
solver permutation maps active variables to these rows, so shrinking does not
require cache-entry swaps. LRU eviction observes the column-payload byte budget,
with a minimum allowance of two columns, capped at the training row count.
Returned read-only leases own their storage lifetime even after eviction.
Metadata, the input snapshot, optimizer arrays, and outstanding leases are
additional memory beyond the column-payload budget. Training does not allocate a
full Gram matrix unless the caller supplies one.

Engine support indices refer to original input rows. Public wrappers adapt the
pinned sklearn filtered-row index convention when nonpositive weights mask
samples. Diagonals remain Float64. Class-specific C values affect C-SVC, while
nu-SVC follows the reference's weighted constraints and nu rescaling.

## Boundaries

The engine validates dimensions, finite inputs, formulation options, encoded
class labels, cache and stopping limits, and nu feasibility. It masks
nonpositive sample weights and rejects an empty retained training set. A class
with no positive-weight samples raises instead of reproducing upstream's
partially removed-class layout quirks. Nonfinite fitted coefficients or rho
raise. Infinite C follows the pinned source where it yields a finite model.
Sparse inputs and regression probability estimation are not exposed by this
family's dense public API. The engine makes no throughput claim.

## Reproduce verification

Run `pixi run build`, `pixi run test`, and `pixi run parity` here. Parity executes
the unmodified LIBSVM C++ source from sklearn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, compiled only into a validation process.

The suite covers 24 kernel fixtures and 144 solver fixtures across all five
formulations and kernels, weighted/masked samples, shrinking, small caches,
iteration limits, and seeded probabilities for binary and multiclass problems.
It compares support packing, convergence status, and actual iteration counts
exactly on these fixtures. Coefficients, thresholds, decisions, probabilities,
and dual objectives use rtol=atol=2e-6 at solver tolerance 1e-7. Independent
Float64 prediction reconstruction uses 2e-10. KKT checks use the Float32 training
Gram values and 2e-5 to allow cached-column quantization and finite stopping.
Native contracts also check eviction-safe leases, nu constraints, zero-iteration
fits, masked indices, and RBF prediction at large offsets.

Raw arrays, source hashes, compiler commands, and iteration counts are retained
under ignored `docs/private/libsvm/solver-parity.json`. Passing these bounded
fixtures does not establish universal numerical accuracy or a performance bound.
