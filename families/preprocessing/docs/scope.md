# Dense-only preprocessing scaffold scope

Supported and tested here: unchanged family/base capability contracts;
Float32/Float64 rank-2 finite/NaN validation; lazy NaN masks; noncontiguous,
negative-stride/offset input; category-code validation; owned feature names and
moments; vocabulary/statistics/output-construction trait probes; two isolated
consumers composing multiple namespace contributions. Implemented estimators are
listed on the [API pages](api.md); see [plumbing.md](plumbing.md) for owners.

Deferred by the authorized dense-only policy: CSR/CSC/COO containers, sparse
input/output/validation/conversion and sparse arithmetic. No sparse input is
silently densified. Foundation `_sparse` is unavailable and not a prerequisite.
Reference branches: sklearn `preprocessing/_data.py` sparse scaling/normalization,
`_encoders.py` OneHotEncoder sparse output, `_label.py` sparse indicators,
`_polynomial.py` sparse polynomial input and spline output, `_discretization.py`
KBins onehot sparse output; FunctionTransformer `accept_sparse` paths too.

OneHotEncoder's reference default `sparse_output=True` is unsupported: require
explicit `sparse_output=False`. KBinsDiscretizer's default `encode='onehot'`
requires sparse output: select `onehot-dense` or `ordinal`. LabelBinarizer,
MultiLabelBinarizer, label_binarize and SplineTransformer must keep
`sparse_output=False`; PolynomialFeatures consumes dense input. Every
estimator enforces this: a parameter selecting sparse output raises a clear
`ValueError` ([labels](api/labels.md), [encoders](api/encoders.md),
[discretization](api/discretization.md), [polynomial](api/polynomial.md)).
PolynomialFeatures and SplineTransformer implement their dense contracts (see
[polynomial](api/polynomial.md)); SplineTransformer(sparse_output=True) raises.
QuantileTransformer's `ignore_implicit_zeros` only affects sparse input; on
dense input it has no effect, as in sklearn ([quantiles](api/quantiles.md)).

Executed checks: `pixi run build`, `pixi run test` (locked Mojo 1.0.0/MAX 26.5.0).
Negative checks reject `require_dense_output(True)` and non-NDArray sparse-typed
input, invalid dimensions/rank, forbidden NaN/infinity, unknown category codes,
unfitted names, name mismatches, malformed moment shapes, missing dependencies and
namespace/export collisions. Scaffold parity is not required; no estimator
parity or performance claim follows from these tests.

Language adaptations: typed parameters and in-place fit retain foundation
semantics; typed target-only label trait, List[String] feature metadata and lazy
numeric mask replace Python-specific containers. Tests exercise their ownership,
shape and error contracts. Reflection, pandas, metadata routing, serialization
and Python bindings remain separate work. New integer/category storage operations
require the numerical owner's compatibility evidence rather than a local buffer.
