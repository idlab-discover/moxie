# Sparse coding engine

The Moxie dependency manifest assembles threshold and Lasso CD coding over
shared NuMojo storage. Lasso CD uses the linear-model family’s cyclic Gram
solver. See [the public guide](../../docs/sparse_coding.md).

From the family root: `MOXIE_ESTIMATORS=sparse_coding pixi run test` and
`MOXIE_ESTIMATORS=sparse_coding pixi run parity`.
