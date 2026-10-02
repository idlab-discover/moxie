# Preprocessing statistics

Private pure-Mojo dense moments, extrema and exact robust quantile engine.
Start with [the versioned API](docs/statistics.md), [dense scope](docs/scope.md),
and [memory contract](docs/memory.md). Run the example with
`pixi run build` followed by `builds/weighted_update`.

The project consumes live foundation, NuMojo and family sources through
the shared assembler. Install with `pixi install --locked`; validate using
`pixi run build`, `pixi run test`, `pixi run parity`, `pixi run memory`.
The locked toolchain is Mojo 1.0.0 / MAX 26.5.0.

Functional `scale` consumers use the distinct input-precision `nanmean_at` and
`nanmean_std_at` API; see [statistics.md](docs/statistics.md). StandardScaler
continues to use the Float64 logical-batch moments API.

The engine has a known unweighted overflow mismatch; see [scope](docs/scope.md#known-unweighted-overflow-limitation). It is not full StandardScaler parity.
