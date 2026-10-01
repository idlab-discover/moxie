# Shared random state

Import the private engine from `moxie._random`. All families consume this owner
through current shared assembly. Library execution is pure Mojo; Python,
NumPy and Cython are validation/build tools only. Use the locked Mojo 1.0.0,
MAX 26.5.0 and NuMojo environment.

```mojo
from moxie._random import RandomState

var rng = RandomState(42)
var uniform = rng.random_sample()
var normal = rng.standard_normal()
var topic_weight = rng.gamma(100.0, 0.01)
var selected = rng.sample_without_replacement(100, 5)
var independent_snapshot = rng.copy()
```

## Seed and ownership semantics

`RandomState(seed: Int)` accepts integers from 0 through 2**32-1 and owns an
isolated MT19937 state. It implements NumPy's **legacy RandomState**, not the
new NumPy Generator API. Draw methods mutate their receiver. `.copy()` copies
all 624 words, position and the cached normal; the copy advances independently.
The state inspection methods expose a read-only diagnostic snapshot one word
at a time. Python object pickling and NumPy state tuple import are binding work.

An estimator configured with an integer constructs a fresh `RandomState(seed)`
at each point where sklearn calls `check_random_state(integer)`. An estimator
using a mutable stream borrows the same `mut RandomState` at each call; it must
not reset it or copy it implicitly. This matters during prediction as well as
fit: Dummy's integer-seeded prediction repeats, while its mutable-state
prediction advances. Explicit state copies model snapshots, not shared aliases.

`check_random_state()` and `check_random_state(None)` lease a process-wide
MT19937 stream, shared across Moxie consumers. It is initialized once using the
existing `entropy_state()` policy. The factory uses monotonic-clock seed bits;
it does not promise NumPy's OS-entropy initialization, cross-run sequences or
unique seeds across simultaneous processes. Use `seed_global_random_state(seed)`
to reset this default stream reproducibly, including its cached normal.

The Moxie singleton is separate from both NumPy and std.random's Philox stream.
`entropy_state()` remains available for application-owned independent contexts.
Native global ownership and synchronization are described below.

## Operations and shapes

| Method | Contract |
| --- | --- |
| `next_uint32()` | Raw MT19937 word |
| `random_sample()` | Float64 uniform in [0,1), NumPy's 27+26-bit construction |
| `uniform(low=0, high=1)` | Scalar Float64; zero width still consumes a draw |
| `randint(high)` / `randint(low, high)` | Native signed Int (64 bits on supported platform), high excluded; width 1 consumes no words |
| `standard_normal()` / `normal(loc=0, scale=1)` | Float64 polar normal with cached second result; scale 0 still consumes the normal stream |
| `standard_gamma(shape)` / `gamma(shape, scale=1)` | Scalar Float64 legacy gamma; shape 0 consumes no draws, scale 0 still consumes the gamma stream |
| `multivariate_normal(mean, cov, size=1, tol=1e-8)` | Float64 `(size, features)` result from Float32/Float64 inputs; native SVD covariance factor |
| `poisson(lam=1)` | Scalar native Int; zero mean consumes no draws; NumPy int64 safety limit |
| `binomial(n, p)` | Legacy inversion and BTPE, including endpoint draw consumption |
| `categorical(p)` | Integer class index, NumPy `choice(p=p)` cumulative-distribution protocol |
| `multinomial(n, p, size=1)` | NuMojo Int64 array `(size, classes)`, including size=1; final class receives remaining mass |
| `choice(a, size=1, replace=True)` / `choice(a, p, size=1, replace=True)` | Int population or numeric vector; always a `(size,)` vector, including size 1; optional Float64 probabilities |
| `shuffle(x)` | In-place vector or matrix-row shuffle, reverse Fisher–Yates order |
| `permutation(n)` / `permutation(x)` | Int64 index vector or independent shuffled copy of a vector/matrix |
| `sample_without_replacement(n_population, n_samples, method="auto")` | NuMojo Int64 vector; sklearn auto, tracking_selection, pool and reservoir_sampling |

Probability inputs are one-dimensional NuMojo `NDArray[DType.float64]`; results
use the accepted NuMojo storage rather than a new generic container. Scalar
uniform/normal/gamma outputs can be explicitly cast to Float32 by a consumer; there
is no promise of a distinct NumPy float32 generation protocol.

Gamma parameters must be finite and nonnegative; negative zero is rejected,
as in NumPy. NaN and infinity are explicitly unsupported and raise before
consuming state. This is a narrower contract than NumPy's nonfinite handling.

Arbitrary Python `size` tuples, array-valued distribution parameters and dynamic dtype dispatch
are not exposed. Native callers loop scalar methods in NumPy traversal order.
`multinomial` always preserves its leading sample dimension, a documented
language adaptation from NumPy's `size=None` vector result.

Categorical probabilities must sum to one within NumPy's Float64 tolerance.
Multinomial validates each probability and the sum excluding the last; the last
entry is residual, not a categorical normalization. Empty multinomial
probabilities produce `(size, 0)` without draws. Invalid sizes, seeds, bounds,
probabilities and method names raise Mojo `Error` with a `ValueError` prefix.
Sparse matrices and `_random_choice_csc` are deferred; see [scope.md](scope.md).

`tracking_selection` follows sklearn's actual dispatcher: it uses reservoir
sampling when the selected fraction is at least 0.2. `auto` uses permutation
strictly between 0.01 and 0.99. Sample order and draw consumption are part of the
comparison, not just membership or empirical frequencies.

## Standard-library reuse

`NativeRandom(seed: UInt64).step()` delegates directly to the pinned stdlib's
isolated `std.random.Random[10]`, returning four UInt32 words. Copies are
independent. Use it only for explicitly native streams whose contract permits
Philox. It is not a RandomState substitute. The pinned stdlib global functions
also use Philox; reseeding their global state would interfere with consumers.

Source inspection and executable comparison establish why compatibility
machinery is needed: [Modular's pinned Philox implementation](https://github.com/modular/modular/tree/b4497b7ce9ba96331c72c637ad41b44bab374f33/mojo/stdlib/std/random)
and [NumPy's pinned legacy implementation](https://github.com/numpy/numpy/tree/dd88c0c19b54ad9ed3533224221285bf0873249a/numpy/random)
use different generators and distribution protocols. Translated NumPy and
sklearn algorithms retain notices under `LICENSES/`. No C RNG is linked at
library runtime.

## Dependency consumption and validation

Import private types from `moxie._random`. Follow
[live-source policy](../../../docs/family-development.md); module/export
manifests define ownership, Pixi locks define compatible toolchains. Invoke
current `../assembly/tools/consume.py`. Consumer probes under `tests/consumers/`
compile separate namespaces from the same current implementation.

Run `pixi run build`, `pixi run test`, and `pixi run parity`. Parity compiles the
**unedited** local sklearn `_random.pyx` and its `.pxd` dependencies in an ignored
test package. It executes AST-selected original Dummy `predict`/`predict_proba`
and `check_random_state` bodies with dense test attributes and NumPy plumbing;
it does not substitute a wheel's implementations or claim full Dummy estimator
parity. Reference revision and cleanliness are checked before execution.

The suite compares sequences plus all 624 MT words, position, cache presence
and cached Gaussian after each workload. Integers and uniform values must match
exactly; normal values and the normal cache allow 2e-14 absolute/relative error
for native libm rounding. It exercises MT twists, rejection bounds across
2**32, both binomial branches, sampling ratio boundaries, interleaved draws and
Dummy call patterns. Native lifecycle tests cover copy isolation and errors.

Compile consumers with `mojo build -O3 --fp-mode=contract=off`. Mojo defaults to
fusing multiply-add across statements; disabling contraction is required for
NumPy-compatible arithmetic and rejection decisions. The two locked probes
use this flag. Keeping only the same seed without the arithmetic mode does not
establish parity. NuMojo storage remains the accepted dependency; RNG word state
and integer sampling bookkeeping are algorithm-specific structures.

For a small reproducible performance baseline after parity:

```sh
pixi run python tools/benchmark.py --processes 2 --values 3 --warmups 1 \
  --min-time 0.02 --affinity 0 -o benchmarks/random.json
```

The benchmark uses pyperf to measure native elapsed regions, excluding process
startup. It covers scalar uniforms/normals, native Philox four-word steps and
permutation allocation for small and larger batches. These are different
workloads, not a claim that one RNG is a faster compatible replacement. Raw
results and GNU time RSS measurements belong under ignored `benchmarks/`.

Use `tools/memory.py` for process RSS. Retain measurements with versions,
flags, hardware, seeds and consumed-source receipts; old timings are not current
guarantees. Result storage is O(n); MT state holds 624 UInt32 words plus metadata.

Compatibility scalar log/exp use `std.sys.llvm_intrinsic` (`llvm.log` and
`llvm.exp`), standard compiler math primitives lowered to the CPU platform math
runtime. This operation-specific exception avoids the pinned stdlib polynomial
approximation's observed normal errors around 1e-9. It adds no numerical package
or RNG implementation outside this owner. Other scalar operations and the
explicit native Philox stream reuse the stdlib directly.

Gamma follows NumPy's frozen `legacy_standard_gamma`: exponential at shape 1,
a uniform/exponential rejection method below 1, and Marsaglia–Tsang above 1.
It shares the normal cache with `standard_normal`. Numax inverse-CDF sampling
would change the seeded sequence and draw consumption. Differential tests
compare gamma values at Float64 rtol 2e-14 and atol 5e-324 (one smallest
subnormal, for platform math rounding), with
exact MT state words, position and cache occupancy, including mixed draws and
LDA initialization parameters `(100, 0.01)`. Use the existing
`--fp-mode=contract=off` build setting for this compatibility contract.

Choice supports uniform and weighted draws with and without replacement. As
with multinomial, omitted size means a one-element vector rather than NumPy's
scalar. Probabilities are Float64 vectors and must match the population length.
Unweighted choice without replacement consumes a complete permutation, even
for size zero. Weighted choice follows NumPy's batched duplicate rejection;
`sample_without_replacement` has a different sklearn-specific draw protocol.
Numeric vector inputs preserve dtype. Object/string populations and arbitrary
size tuples are outside this typed interface; callers may gather objects from
sampled indices.

Shuffle and array permutation support rank one and two on axis zero. Standard
contiguous, transposed, sliced and reversed disjoint layouts are supported;
layouts not demonstrably disjoint by row or column stride are rejected.
Shuffle consumes no draws for empty arrays. Matrix permutation still
permutes row indices when rows have zero columns, matching NumPy. Shuffle
mutates borrowed storage; permutation copies it. Higher ranks are explicitly unsupported.

Poisson uses NumPy's multiplication algorithm below mean 10 and Hoermann's
transformed rejection (PTRS) at or above 10. The mean must be finite,
nonnegative and at most `float64(INT64_MAX) - 10*sqrt(float64(INT64_MAX))`.
Negative zero is accepted. The legacy log-gamma polynomial is retained for
rejection decisions. Tests compare integer samples and subsequent RNG state
exactly, including means around 10 and near the int64 safety limit.

Multivariate normal uses `mojo-linalg.svd` and NuMojo matrix product:
`Z @ (sqrt(s)[:, None] * Vt) + mean`. The mean is a vector, covariance a square
matrix of matching order, and both must be finite. Float32 and Float64 inputs,
including strided views and mixed mean/covariance dtypes, produce Float64 output.
Singular positive-semidefinite covariance is supported. Symmetry and positive
semidefiniteness are checked using the SVD reconstruction with absolute and
relative tolerance `tol` (default 1e-8).

The native API always returns `(size, features)`, including size 1; arbitrary
size tuples are not accepted. Size zero and zero features produce empty results.
Invalid covariance raises before consuming RNG state. This deliberately differs
from NumPy's default `check_valid="warn"` and its failure-time state advancement;
warning/ignore policies are not exposed.

A seeded call is reproducible with the same backend and toolchain, and consumes exactly the
same normal stream as NumPy for supported valid inputs. **Individual samples are
not promised equal to NumPy**: NumPy uses LAPACK SVD, while this dependency uses
Jacobi SVD with potentially different vector signs and bases in repeated singular
subspaces. Both produce the requested covariance. Tests independently check
factor reconstruction, singular values, the seeded normal-to-sample transform,
empirical moments, copies, strided inputs and complete RNG state. This method's
numerical comparison tolerance is 2e-12; discrete sampling parity remains exact.

Numax SVD was evaluated first: its current Array/Tensor entry points require
compile-time dimensions, and its MAX 26.6 lock differs from this owner's 26.5
lock. The fallback `mojo-linalg` uses NuMojo storage and the compatible Mojo/MAX
toolchain. Shared assembly consumes its current local source and records output
hashes as receipts. Sampling uses no Python or foreign numerical runtime.

## sklearn XorShift primitive

`from moxie._random import our_rand_r` exposes
`our_rand_r(mut state: UInt32) -> UInt32` for sklearn's internal solver/tree
stream. The caller owns the state word. Zero is repaired to one before the
13-left, 17-right, 5-left XOR shifts, with UInt32 wrapping. The full 32-bit
updated state is retained; the returned value is reduced modulo 2**31.
Do not assign the returned value back to the state. This function allocates
no buffers and does not touch `RandomState` or the global/native stream.

```mojo
var state = UInt32(1)
var draw = our_rand_r(state)  # draw and state are both 270369 on this call
```

This is a separate compatibility primitive, not a replacement for MT19937.
Consumers keep sklearn's algorithm-specific seed initialization and draw order.
Parity compiles a test bridge to the unmodified pinned `_random.pxd` and checks
both output and full state exactly across boundary seeds and long sequences.
The BSD-3-Clause notice is retained in `LICENSES/scikit-learn.txt`.


## Normalizing random_state

`check_random_state` returns a move-only `RandomStateHandle`. It provides the
same generator methods through `handle[]`, or a scoped pointer in a `with` block:

```mojo
from moxie._random import RandomState, check_random_state, seed_global_random_state

def main() raises:
    seed_global_random_state(42)
    with check_random_state() as rng:
        print(rng[].standard_normal())
    with check_random_state(None) as rng:
        print(rng[].standard_normal())  # continues the same stream and cache
    with check_random_state(42) as rng:
        print(rng[].standard_normal())  # fresh stream on each normalization
    var shared = RandomState(42)
    with check_random_state(shared) as rng:
        print(rng[].standard_normal())  # advances shared, without copying
```

| Argument | Ownership and repeat calls |
| --- | --- |
| omitted or `None` | Exclusive handle to the process default stream; advances across calls |
| `Int` | Fresh owned state; repeats the seed's sequence on each normalization |
| `mut RandomState` | Borrows the same state; advances it and preserves its normal cache |
| `Optional[Int]` | Present seed means fresh owned state; empty means the default stream |

Supported integer range is unchanged: 0 through 2**32-1. Invalid integer seeds
raise before changing any existing stream. `seed_global_random_state` resets
only the default stream; independently owned and borrowed states are unaffected.

This is a native ownership adaptation of sklearn's `check_random_state`, not a
Python object identity API. Float/string seeds, NumPy module objects and NumPy
Generator objects have no overload. The handle cannot be copied. Borrowed state must remain alive, and conflicting
mutable arguments are rejected by the compiler. Sequential calls through the
original state or the handle access the same stream. A reference obtained
through the handle cannot outlive it. Call `.copy()` explicitly on the
underlying RandomState only when an independent snapshot is wanted.

The default handle holds a nonblocking exclusive lease for its lifetime.
Nested or concurrent acquisition and reseeding while a lease is outstanding
raise `RuntimeError`-prefixed errors instead of racing or deadlocking. Nested
algorithms should pass the active `rng[]` as a borrowed RandomState. Lease
release occurs on destruction, including exception unwinding and after moving
the handle to another function. Prefer `with check_random_state(...) as rng`
for operation-sized lifetimes. If a handle is stored in a variable, it retains
its lease for its lifetime, including across repeated `with` uses.

The standard-library `_Global` owns process storage and teardown; `Atomic`
protects initialization and access. No parallel draw ordering, shared Python
singleton, fork reseeding, or serialized global-state format is promised.

Validation compares repeated normalized calls and complete state against the
original pinned sklearn normalizer. Compile-failure tests enforce borrowing,
noncopyability and reference lifetimes. Independent compiled consumers verify
that their module calls and main code share one stream. Existing RNG tests
continue to establish draw-sequence behavior, including FP contraction settings.
