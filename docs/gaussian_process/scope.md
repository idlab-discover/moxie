# Dense Gaussian-process scope

The root package assembles implemented regression and kernels from Moxie-owned
sources. It also provides shared contracts, metadata, validation and dense training
retention helpers. See [API coverage](api-surface.md),
[regression scope](regression/scope.md), and
[kernel scope](kernels/scope.md).

Supported arrays use NuMojo Float32/Float64 storage. Regression accepts dense
rank-two features and vector or matrix targets. Classification, native structured
kernels and Python integration remain incomplete. Sparse input is not implicitly
densified; CSR/CSC/COO and sparse extensions remain outside the dense scope.

`pixi run test` builds the public namespace, runs native regression and kernel
contracts, and checks the foundation interfaces and two isolated plumbing consumers.
It also checks missing-source and namespace-collision rejection. `pixi run parity`
compares the numerical implementations against the pinned scikit-learn checkout.
