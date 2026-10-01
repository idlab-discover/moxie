# Regularized incomplete gamma

Import `gammainc` from `moxie._special` through the verified shared assembler.
The scalar overloads accept two Float64 values or two Float32 values and return
that same dtype. For example:

```mojo
from moxie._special import gammainc

def main() raises:
    print(gammainc(Float64(0.5), Float64(1)))  # approximately 0.842700792949715
```

Float32 evaluation widens to Float64 and rounds once on return.

`gammainc(a, x)` computes the regularized lower incomplete gamma P(a,x).
Require finite `a > 0` and `x >= 0`; positive infinity for x returns 1 and zero
returns 0. NaN, nonpositive a, infinite a and negative x raise a Mojo `Error`
with a `ValueError` prefix. This explicit validation differs from SciPy's ufunc
NaN/limit conventions outside this domain. No broadcasting or array overload is
provided: dense consumers can apply this scalar operation over NuMojo storage.

The CPU implementation uses Numax `core.libm.exp/log`, standard-library
`lgamma/expm1`, an adaptive positive series
and an upper-tail continued fraction. A Stirling correction stabilizes the
prefactor near x=a. Underflow rounds to zero. Iteration is bounded at 100000
terms; nonconvergence and invalid numerical results raise `ArithmeticError`
prefixed errors. Very large a near x=a may reach this limit; there is no uniform
asymptotic expansion or promise to evaluate every finite Float64 pair.

This operation allocates no arrays and changes no input or RNG state. KDE's
normal draws, selection order, bandwidth scaling and sampling API belong to the
neighbors family. Consume current sources under the
[live-source policy](../../../docs/family-development.md).

Run `pixi install --locked`, `pixi run build`, `pixi run test`, and
`pixi run parity`. See tests for the reproducible SciPy and sklearn comparisons.

## Accuracy and references

The parity suite covers shapes through 1e8, subnormal inputs, branch transitions,
small probabilities and saturated tails in both dtypes. Float64 uses relative
error 1e-11, Float32 3e-7, plus 16 smallest-subnormal units for rounding near
underflow. The looser Float64 bound accounts for reference-prefactor cancellation
in deep tails; selected independent 60-digit checks use 3e-12. These are tested
tolerances, not a uniform error theorem.

SciPy is the primary oracle (as in sklearn). Extreme regimes use independent
60-digit mpmath references: subnormal shapes/probabilities where SciPy's
prefactor underflows, and large-shape lower tails where its finite series
truncates. Those differences
are numerical accuracy corrections rather than bit-for-bit SciPy compatibility.
The suite also executes the pinned sklearn tophat sample body with only its gamma
call substituted, covering weighted/unweighted sampling in dimensions 1–1000.
This does not claim implementation of the neighbors estimator.

Mathematics: [DLMF series](https://dlmf.nist.gov/8.7),
[continued fractions](https://dlmf.nist.gov/8.9), and
[Stirling expansion](https://dlmf.nist.gov/5.11).
SciPy's gamma source and Numax's bounded series inform the choice of regimes;
the adaptive glue is local Mojo code, and Numax primitives are consumed unchanged.

To measure scalar cost after validation:

```sh
pixi run python tools/benchmark.py --processes 1 --values 2 --warmups 1 \
  --min-time 0.01 --affinity 0 -o benchmarks/gammainc.json
```

This is a serial, allocation-free scalar API. Iterations depend on the inputs;
large shapes near the transition are more expensive. SIMD/batch parallelism and
large-shape asymptotic acceleration are separate work, not performance claims.
