# Shared random state

Import the private engine from `moxie._random`. All families consume this owner
through the pinned assembly tool. Library execution is pure Mojo; Python,
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

The native equivalent of `None` is an application-owned context initialized
with `entropy_state()` once and subsequently passed by mutable reference to all
consumers sharing that context. This is an explicit adaptation of NumPy's
implicit global singleton. The factory uses monotonic-clock seed bits; it does
not promise NumPy's OS-entropy initialization, cross-run sequences or unique
seeds for simultaneous initialization. Retain one context rather than calling
the factory per draw. An integer seed is required for reproducibility. Sharing
one mutable state concurrently is outside this serial borrowing contract.

## Operations and shapes

| Method | Contract |
| --- | --- |
| `next_uint32()` | Raw MT19937 word |
| `random_sample()` | Float64 uniform in [0,1), NumPy's 27+26-bit construction |
| `uniform(low=0, high=1)` | Scalar Float64; zero width still consumes a draw |
| `randint(high)` / `randint(low, high)` | Native signed Int (64 bits on supported platform), high excluded; width 1 consumes no words |
| `standard_normal()` / `normal(loc=0, scale=1)` | Float64 polar normal with cached second result; scale 0 still consumes the normal stream |
| `standard_gamma(shape)` / `gamma(shape, scale=1)` | Scalar Float64 legacy gamma; shape 0 consumes no draws, scale 0 still consumes the gamma stream |
| `binomial(n, p)` | Legacy inversion and BTPE, including endpoint draw consumption |
| `categorical(p)` | Integer class index, NumPy `choice(p=p)` cumulative-distribution protocol |
| `multinomial(n, p, size=1)` | NuMojo Int64 array `(size, classes)`, including size=1; final class receives remaining mass |
| `permutation(n)` | NuMojo Int64 vector, reverse Fisher–Yates order |
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

`EXPORTS.json` declares an empty root initializer: private types are imported
from `moxie._random`, not exported as public estimators. The lock maps module
ownership explicitly. Consumers pin this project's module, export manifest,
lock and transitive inputs, then invoke `../assembly/tools/consume.py` with their
own lock and output directory. The two isolated examples under
`tests/consumers/a` and `b` call the same hashed implementation through separate
assembled namespaces. Builds stop on pinned source drift.

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

The initial serial baseline on an AMD Ryzen 9 5950X measured 10,000 uniforms
in about 196 µs, 10,000 normals in 414 µs, and a 10,000-element permutation in
1.56 ms. Two pyperf processes, three values and one warmup give a small baseline;
short samples emitted stability warnings. The scalar logarithm primitive used
for compatibility took 40.6 µs per 10,000 evaluations versus 52.9 µs for the
pinned stdlib approximation on the same input grid. This local result does not
establish performance on other hardware or numerical domains. The primitive
was chosen to meet the numerical contract, not to promise a speedup.

Whole-process GNU time peak RSS was approximately 21,232 KiB for a 16-element
permutation, 23,356 KiB for 10,000 elements, and 31,300 KiB for 1,000,000 elements.
This includes runtime overhead. Result storage is O(n); the generator holds 624
UInt32 words plus position/cache metadata. Run `pixi run python tools/memory.py`
to repeat those measurements. `benchmarks/random-final.json` records pyperf
results, compiler flags, tool versions, seeds and source/lock hashes.

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
