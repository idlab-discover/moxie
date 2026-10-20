# Dense DictVectorizer scope and remaining work

Implemented public export: `moxie.feature_extraction.DictVectorizer`, with
explicit `sparse=False`. It directly constructs dense NuMojo `NDArray` output,
without CSR construction or a sparse conversion.

Implemented and tested real-valued settings:

- Float32, Float64, bool, and signed/unsigned 8-, 16-, 32-, 64-bit output.
- fit, transform, fit_transform, inverse_transform, get_feature_names_out,
  restrict, typed get_params/set_params, and unfitted clone.
- Sorting and encounter order, integer/string key identity, Unicode categories,
  arbitrary separators, duplicate categories and raw-name collisions, None,
  zero values, empty features, unknown keys and conversion failures.
- Repeat fitting, failed-fit state preservation, copied configuration and
  metadata, negative/duplicate restriction indices, list/NDArray masks and indices,
  fluent restriction, and strided inverse with independent input dtype.

Dense dtype work is **not complete**: reference complex64/complex128,
longdouble/clongdouble input/output remain unimplemented. These are in-scope
requirements, not sparse deferrals. The family's NDArray-only storage contract
does not cover NuMojo's separate ComplexNDArray. The pinned DType API also lacks
an extended-precision floating type. An upstream numerical-storage contract is
needed before this child can implement those branches without inventing storage
or silently narrowing values. Public [API documentation](api.md) describes the
implemented typed interface and its language adaptations.

Authorized sparse exclusions: `sparse=True`, all sparse inputs/outputs, and
`FeatureHasher` (always CSR in the reference). No dense FeatureHasher substitute
is exported. Reference sources are `_dict_vectorizer.py`, `_hash.py`, and
`_hashing_fast.pyx` at sklearn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The text/TF-IDF sessions and image APIs
have separate owners. The Python reflection, metadata-routing, serialization and
pandas restrictions of the foundation apply.

Defaults stay dtype=float64, separator="=", sparse=True, sort=True. The unsupported
sparse default raises on fit/transform: callers must select `sparse=False`.
Float16 output is rejected as in the reference conversion. No complex dtype is
silently mapped to a real dtype.

Checks executed: `pixi run build`, `pixi run test`, `pixi run parity`.
Native tests include two isolated consumers exchanging the same live storage
and vocabulary implementations, missing-source rejection, missing CSR imports and the family-level FeatureHasher rejection, compile rejection of nested/non-string iterable values, sparse-default
errors and unfitted errors. Parity executes the pinned local encoder class AST;
308 encoder cases and 43 array API cases cover real-valued inputs. External
decorator/base/check_array plumbing is replaced only in validation tooling. The inverse finite matrix algorithm is compared; external Python
check_array dispatch is not a full binding-parity claim. Values use exact dtype
comparison, with NaN masks checked separately.

Repeated restriction indices can leave the reference with out-of-range CSR
column metadata. This implementation preserves the learned metadata but rejects
such a transform instead of indexing outside the dense array. This explicit
safety difference is native-tested; valid duplicate-index selections are compared
against the reference. The local reproducer confirms scipy.check_format rejects
the malformed column indices.

Passing these real-valued checks does not establish all-dtype dense completion.
A short native pyperf timing and whole-process GNU time memory baseline is
reproducible with `pixi run python tools/performance.py`. Generated data stays
ignored; no performance superiority is claimed.
