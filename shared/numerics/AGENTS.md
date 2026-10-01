# Shared numerics

Read [numerical contract](docs/numerics.md). Own `moxie._dense` adapters and
storage/operation validation. Reuse NuMojo `NDArray`, `Shape`, `Item`;
`sum_axis2d` and `row_squared_norms` provide tested matrix kernels.

Follow root live-source policy. Use shared assembler; preserve Pixi toolchain
locks and sklearn oracle revision. New operations require correctness and cost
evidence; current exclusions stay until resolved. No competing dense container.
Research: `docs/private/`; measurements: `benchmarks/`, `profiling/`.
Run build, test, parity after native changes. Failure probes document exclusions.
