# Sparse coding engine

Family-owned threshold sparse coding and SparseCoder over shared NuMojo storage.
The root family environment builds this module through its explicit dependency
manifest; there is no independent child environment or copied foundation.
See [the public guide](../../docs/sparse_coding.md). Regression-based coding
requires the external linear-model owners and remains unavailable.

From the family root: `MOXIE_ESTIMATORS=sparse_coding pixi run test` and
`MOXIE_ESTIMATORS=sparse_coding pixi run parity`.
