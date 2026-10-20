# Supported scope and unfinished work

The implemented dense dictionary and image contracts are owned by Moxie.
See [DictVectorizer scope](../algorithms/dict_hash/docs/scope.md) and
[image scope](../algorithms/image/docs/scope.md) for dtype coverage and limitations.
The family also owns vocabulary snapshots, column naming, shape validation,
image geometry, analyzer and hasher traits, and the English stop-word tuple.

Sparse inputs, sparse graph output, CSR/CSC/COO construction and sparse
DictVectorizer output remain deferred. The default sparse settings raise;
callers must explicitly request dense behavior. FeatureHasher, CountVectorizer,
HashingVectorizer, TfidfTransformer and TfidfVectorizer remain raising declarations.
No tokenizer, feature hashing or TF-IDF engine is implemented.

Complex64/complex128 and longdouble/clongdouble dictionary input/output remain
unfinished dense requirements. The available NDArray and DType contracts do not
cover those cases. This rollup preserves that limitation; it does not reclassify
it as sparse work. Python reflection, metadata routing, pandas adapters and
serialization remain outside the native contract.

The reference revision is `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
The historical external `session-plan/SPARSITY-SCOPE.md` is absent in this
workspace; the explicit accepted exclusions above are retained from the source
family and child scope documents. Validation commands are in the [README](../README.md).
