# Historical performance baseline

The retired standalone `tools/benchmark.py` measured one `fit` plus
`predict_proba` on the same deterministic 3-class dataset for the native port
(`builds/benchmark`, in-process timer, `-O3 --fp-mode=contract=off`, serial) and the
pinned sklearn oracle (single-threaded BLAS). Medians of 7 repeats on an AMD Ryzen 9 5950X,
Mojo 1.0.0, NumPy 2.5.3 (raw data: ignored `benchmarks/baseline.json`):

| Mode | n=300, p=5 native / sklearn | n=3000, p=20 | n=20000, p=50 |
| --- | --- | --- | --- |
| lda-svd | 0.13 / 1.21 ms | 7.1 / 4.2 ms | 126 / 40 ms |
| lda-lsqr | 0.11 / 1.12 ms | 6.0 / 2.7 ms | 95 / 19 ms |
| lda-eigen | 0.09 / 1.35 ms | 4.7 / 3.2 ms | 89 / 22 ms |
| qda-svd | 0.25 / 0.72 ms | 20.7 / 2.8 ms | 439 / 37 ms |
| qda-eigen | 0.18 / 0.94 ms | 14.7 / 2.6 ms | 348 / 24 ms |

In that historical run, small problems were faster natively (no Python dispatch); larger ones are 1.5-12x slower.
Profiles (perf) show the cost in scalar loops and in mojo-linalg's Jacobi SVD/eigh, whose
checked element accessors dominate. The first implementation spent 0.43 s in a Jacobi SVD
of the tall 3000x20 matrix; reducing tall inputs to their Householder R factor cut the
`lda-svd` case to 7 ms. No other tuning (SIMD, parallelism, blocked GEMM) was attempted.

These measurements predate integration and live dependency assembly. They are
not performance claims for current Moxie. The original raw artifacts remain in
the retired checkout. The native benchmark fixture is retained under
`tests/discriminant_analysis/benchmark.mojo`.

To repeat the retained timing harness after a root build, run
`pixi run python tools/discriminant_analysis_benchmark.py`. It writes raw
results under `benchmarks/discriminant_analysis/`. Use pyperf for new
performance claims, as required by the repository policy.
