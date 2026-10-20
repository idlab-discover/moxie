# Reproducing the dense checks

Use the pinned Pixi environment and run `pixi run build`, `pixi run test`,
`pixi run parity`. Library code executes natively; Python only orchestrates
checks.

Parity checks the revision and cleanliness of the local sklearn checkout,
then executes the unchanged AST bodies of `_randomized_svd`,
`_randomized_range_finder` and `svd_flip` from that checkout, with NumPy/SciPy
namespace and RNG plumbing. It verifies their compiled filename. An installed
sklearn wheel is not used as a substitute. Deterministic factors are compared
with SciPy SVD/eigh and independent reconstruction/orthogonality equations.

The suite covers both dtypes, zero/rank-deficient and repeated-value cases,
small and rectangular matrices, all four range normalizers, zero/automatic and
multiple power iterations, explicit and automatic transposition, deterministic
seeds, and the Gaussian draw immediately following each randomized call.
Native contracts exercise negative-stride views, unchanged inputs, result-copy
independence, sign ties, invalid values/options, centering and explicit failures.
Both isolated consumers import the same pinned engine from separately generated
namespaces. The test task also rejects an unavailable sparse import.

Both direct kernels (`dense_svd`, `golub_kahan_svd`) and every randomized
case run under both `svd_lapack_driver` values against SciPy's matching driver,
plus ill-conditioned (condition 1e5), duplicated-column and Vandermonde inputs
in both orientations with tolerances scaled by the largest entry.

Exact small-factor comparisons use relative/absolute tolerance 3e-11 for Float64
and 3e-5 for Float32. Randomized factor/subspace checks allow multiples of those
bounds for accumulated products and differing factorization rounding; the
largest Float32 reconstruction allowance is 3e-3. These are bounded fixture
checks, not an arbitrary ill-conditioned-data guarantee. Repeated or zero
singular directions are compared through reconstruction/subspaces. RNG normal
draw comparison allows 2e-14 for scalar math rounding. The extreme diagonal
regressions use their analytic spectrum with relative tolerance 1e-12 and zero
absolute tolerance; SciPy itself loses a tiny diagonal singular value when the
input spans 600 orders of magnitude.

To record a small native performance baseline after correctness passes:

```sh
pixi run python tools/benchmark.py --processes 2 --values 2 --warmups 1 \
  --min-time 0.01 --affinity 0 -o benchmarks/svd.json
pixi run python tools/memory.py
```

Choose an available CPU if CPU 0 is restricted. The native timed region excludes
process startup and input construction, but includes workspace allocation,
changing inputs, factorization and result certificates. pyperf records compiler,
flags, seed, machine metadata and source/lock hashes. GNU time records whole-process
RSS; this is not an allocation count or memory-safety result. Short runs may be
noisy, especially while independent projects share the host. No throughput
advantage over BLAS/LAPACK is claimed.
