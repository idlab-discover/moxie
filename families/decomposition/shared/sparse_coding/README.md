# Sparse coding engine

The Moxie dependency manifest assembles threshold, Lasso CD and OMP coding over
shared NuMojo storage. Lasso CD and OMP use the linear-model family’s Gram
solvers. See [the public guide](../../docs/sparse_coding.md).

From the family root: `MOXIE_ESTIMATORS=sparse_coding pixi run test` and
`MOXIE_ESTIMATORS=sparse_coding pixi run parity`.
