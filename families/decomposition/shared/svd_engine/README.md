# Dense decomposition engine

Native Mojo SVD, symmetric eigen, QR, and seeded randomized range kernels over
NuMojo arrays. Public export: `moxie.decomposition.randomized_svd`.

Both `svd_lapack_driver` selections are native: `"gesdd"` (default) uses
one-sided Jacobi SVD and `"gesvd"` uses Golub-Kahan-Reinsch bidiagonal QR.
See [scope](docs/scope.md) and [API](docs/api.md) for supported modes and
language adaptations.

```sh
pixi install --locked
pixi run build
pixi run test
pixi run parity
```

[The example](examples/randomized.mojo) shows an owned rank-two factorization.
Dependencies come from the current Moxie namespace assembler.
The standalone source-pin tool is retained under family provenance.
