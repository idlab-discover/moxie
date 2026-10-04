# Dense scope and status

Implemented (dense Float32/Float64): LedoitWolf, OAS, ledoit_wolf,
ledoit_wolf_shrinkage (all block sizes) and oas, both centering modes and
precision cache modes, single-feature and single-sample branches.

`pixi run covariance-test`: block-size invariance, parameter validation, lifecycle,
clone, lazy precision, feature mismatch, degenerate branches and engine trait
dispatch, for both dtypes. `pixi run covariance-parity`: 3300 checks over 54 seeded
datasets (tall, square, wide, 30x60 blocked, one feature, two samples, one
sample, constant column, offset; Float64/Float32) against the pinned sklearn
sources, with the tolerance derivation in tests/parity.py.

Sparse branches: none in these APIs; sparse input is deferred by SPARSITY-SCOPE.

Run these commands from the Moxie repository root.
