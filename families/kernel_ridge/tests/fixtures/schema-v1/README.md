# KernelRidge schema-v1 fixtures

These files are fixed compatibility inputs, read by `tests/test_serialization.mojo`
during `pixi run test`. Tests never regenerate them. Both the shared container
version and the family state schema are 1. `affine` uses the example adapter ID
`moxie.examples.AffineKernel`, state version 1.

Each Float32/Float64 case includes fitted and unfitted `.moxie` files:

| Case | Configuration | Training data |
| --- | --- | --- |
| `linear` | Default KernelRidge (alpha 1) | `X = arange(6).reshape(3, 2)`, `y = arange(3)` |
| `affine` | Alpha 0.25, native AffineKernel offset 1.5 | Same data |

The fitted linear files are byte-identical to files saved before callable
serialization was added. All arrays use the case's dtype. Other constructor
settings retain defaults. Unfitted files have empty learned arrays.

Each `.predictions` file contains two raw little-endian scalars in the case's
dtype, captured from the original fitted model on
`arange(4).reshape(2, 2)`. Tests compare their bits exactly to predictions from
the saved model without refitting. They also check configuration, training data,
fitted status, unfitted errors, and deterministic migration to state schema 3.
Schema-1 input files are preserved; migrated files are not byte-identical.

Fixtures were produced with the repository's locked Mojo 1.0.0 / MAX 26.5.0
runtime, on Linux x86-64, with `-O3 --fp-mode=contract=off`, `OMP_NUM_THREADS=1`
and `OPENBLAS_NUM_THREADS=1`; no randomness is used. Runtime source hashes are
recorded in the original repository's `RUNTIME.lock.json` at the revision
in [ORIGIN.json](../../../ORIGIN.json). Current builds use Moxie's live manifest.

The fixture maintenance tool now writes current-schema candidates under ignored
`builds/schema-v2-*` paths. It never overwrites these schema-1 compatibility
inputs. Compile it against `builds/deps` and `examples` after building, and
review candidates before adopting a new fixture baseline.
